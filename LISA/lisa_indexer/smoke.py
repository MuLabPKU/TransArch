"""Run a small DeepSeek-V3.2 model with dummy weights through vLLM or SGLang."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import io
import json
import os
from pathlib import Path
import subprocess
import sys

LAYERS = 5
TOKENS = 66560
CHUNK = 1024
VARIANTS = ('linear', 'threshold', 'shared4')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_prompt(variants):
    require(0 < CHUNK <= 8192 and 0 < TOKENS <= 1048576,
            'Smoke requires 1..8192 tokens per chunk and 1..1048576 prompt tokens')
    if 'linear' in variants:
        # LinearRuntime.select uses dense FP8x2 through N=65536.
        last_full_chunk_start = (TOKENS // CHUNK - 1) * CHUNK
        require(last_full_chunk_start >= 65536,
                f'linear smoke needs a full {CHUNK}-token prefill chunk after 65536 tokens; '
                f'got {TOKENS} prompt tokens')


def model_config(heads):
    return dict(
        architectures=['DeepseekV32ForCausalLM'], model_type='deepseek_v3',
        hidden_size=256, intermediate_size=512, num_hidden_layers=LAYERS,
        num_attention_heads=128, num_key_value_heads=128, vocab_size=256,
        q_lora_rank=128, kv_lora_rank=512, qk_nope_head_dim=128,
        qk_rope_head_dim=64, v_head_dim=128, index_n_heads=heads,
        index_head_dim=128, index_topk=2048, first_k_dense_replace=LAYERS,
        n_routed_experts=4, n_shared_experts=1, moe_layer_freq=1,
        num_experts_per_tok=2, n_group=1, topk_group=1,
        num_nextn_predict_layers=0, max_position_embeddings=TOKENS + CHUNK,
        rope_theta=10000.0, rope_parameters=dict(rope_type='default', rope_theta=10000.0),
        torch_dtype='bfloat16', dtype='bfloat16', hidden_act='silu',
        rms_norm_eps=1e-6, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
    )


def engine_config(framework, model):
    common = dict(dtype='bfloat16', load_format='dummy', skip_tokenizer_init=True)
    if framework == 'vllm':
        return dict(common, model=str(model), tensor_parallel_size=1,
                    max_model_len=TOKENS + CHUNK, max_num_seqs=1,
                    max_num_batched_tokens=CHUNK, enable_chunked_prefill=True,
                    enable_prefix_caching=False, enforce_eager=True,
                    async_scheduling=False, kv_cache_dtype='fp8',
                    gpu_memory_utilization=0.25,
                    num_gpu_blocks_override=(TOKENS + CHUNK) // 64,
                    block_size=64, seed=0,
                    attention_config=dict(backend='FLASHINFER_MLA_SPARSE'),
                    kernel_config=dict(enable_cutedsl_warmup=False),
                    worker_extension_cls='lisa_indexer.smoke.VLLMWorker')
    return dict(common, model_path=str(model), tp_size=1,
                context_length=TOKENS + CHUNK, max_total_tokens=TOKENS + CHUNK,
                max_running_requests=1, chunked_prefill_size=CHUNK,
                max_prefill_tokens=CHUNK, disable_radix_cache=True,
                disable_overlap_schedule=True, mem_fraction_static=0.25,
                kv_cache_dtype='fp8_e4m3', attention_backend='dsa',
                dsa_prefill_backend='trtllm', dsa_decode_backend='trtllm',
                dsa_topk_backend='sgl-kernel',
                cuda_graph_backend_prefill='disabled', cuda_graph_backend_decode='disabled',
                random_seed=0, watchdog_timeout=1800, log_level='info')


def preflight(framework):
    print('Checking the framework patch, CUDA device and build dependencies...', flush=True)
    from .integrations import patch

    manifest = json.loads((Path(__file__).parent / 'integrations/sources.json').read_text())
    expected = manifest[framework]['version']
    try:
        installed = version(framework)
    except PackageNotFoundError:
        raise RuntimeError(f'Install official {framework}=={expected} in this environment') from None
    require(installed == expected,
            f'Expected official {framework}=={expected}; found {installed}')
    output = io.StringIO()
    with redirect_stdout(output):
        patch.main(['--framework', framework, '--check'])
    require(json.loads(output.getvalue())['ready'],
            f'Apply the hooks first: python -m lisa_indexer.integrations.patch --framework {framework} --apply')
    import torch
    from torch.utils.cpp_extension import CUDA_HOME
    from .build import include_dirs

    require(torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0),
            'This smoke test requires a B200/GB200 (SM100) GPU')
    require(CUDA_HOME is not None, 'Set CUDA_HOME to a CUDA 13 toolkit with nvcc')
    nvcc = subprocess.check_output([str(Path(CUDA_HOME) / 'bin/nvcc'), '--version'], text=True)
    require('release 13.' in nvcc, 'CUDA 13 nvcc is required')
    include_dirs()
    packages = {}
    for name in (framework, 'torch', 'triton', 'transformers', 'flashinfer-python',
                 'nvidia-cutlass-dsl', 'sglang-kernel', 'ninja'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return dict(python=sys.version.split()[0], packages=packages,
                gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, nvcc=nvcc.strip())


def prepare_request(variant, layers):
    import torch
    from .integrations import forward_scope, runtime_hooks

    require(layers == LAYERS, 'Unexpected model layer count')
    runtime_hooks.reset()
    runtime_hooks._ADAPTERS.clear()
    forward_scope.COUNTERS.clear()
    os.environ.update(LISA_INDEXER=variant, LISA_SINGLE_REQUEST='1', LISA_STRICT='1',
                      LISA_NUM_LAYERS=str(layers), LISA_REQUEST_ID='smoke-' + variant)
    os.environ.pop('LISA_REPORT_DIR', None)
    torch.cuda.synchronize()


def worker_reports():
    import torch
    from .integrations import runtime_hooks

    torch.cuda.synchronize()
    return runtime_hooks.reports()


class VLLMWorker:
    def lisa_smoke_prepare(self, variant):
        prepare_request(variant, self.vllm_config.model_config.hf_config.num_hidden_layers)

    def lisa_smoke_report(self):
        return worker_reports()


def sglang_prepare(self, variant):
    prepare_request(variant, self.model_config.hf_config.num_hidden_layers)


def sglang_report(self, destination):
    Path(destination).write_text(json.dumps(worker_reports(), indent=2) + '\n')


def run_scheduler(*args, **kwargs):
    from sglang.srt.managers import scheduler

    scheduler.Scheduler.lisa_smoke_prepare = sglang_prepare
    scheduler.Scheduler.lisa_smoke_report = sglang_report
    scheduler.run_scheduler_process(*args, **kwargs)


def check_report(reports, framework, variant):
    require(len(reports) == 1, 'Expected one LISA adapter on the single worker')
    report = reports[0]
    require(report['framework'] == framework and report['variant'] == variant,
            'Worker used a different framework or variant')
    counts = report['calls']
    require(counts.get('executed', 0) > 0, 'No LISA calls were observed')
    require(not any(v for k, v in counts.items() if k.startswith('fallback:')),
            'A prefill fell back to the framework indexer')
    if variant == 'linear':
        require(counts.get('route:hot', 0) > 0, 'The prompt did not exercise the linear HOT path')
    if variant == 'shared4':
        coarse, member = counts.get('coarse_calls', 0), counts.get('member_calls', 0)
        require(coarse > 0 and member > 0 and coarse * 3 == member * 2
                and coarse + member == counts['executed'], 'Incomplete five-layer sharing schedule')
        require(counts.get('forwards_completed', 0) > 0 and not counts.get('forwards_aborted', 0),
                'A shared4 forward did not complete')
        state = report['shared_state']
        require(state['live_pools'] == 0 and state['forward_active'] is False,
                'shared4 left candidate pools or a forward active')
    return report


def run_vllm(config, variants):
    from vllm import LLM, SamplingParams

    engine = LLM(**config)
    try:
        for variant in variants:
            print(f'START vllm {variant}: {TOKENS} prompt tokens, {CHUNK}-token chunks; '
                  'first use may compile kernels.', flush=True)
            engine.collective_rpc('lisa_smoke_prepare', args=(variant,))
            outputs = engine.generate([dict(prompt_token_ids=[7] * TOKENS)],
                                     SamplingParams(max_tokens=2, ignore_eos=True, temperature=0),
                                     use_tqdm=False)
            require(len(outputs) == 1 and len(outputs[0].outputs) == 1, 'Unexpected response count')
            require(outputs[0].prompt_token_ids == [7] * TOKENS
                    and outputs[0].num_cached_tokens == 0, 'Prompt or prefix-cache mismatch')
            tokens = list(outputs[0].outputs[0].token_ids)
            workers = engine.collective_rpc('lisa_smoke_report')
            require(len(workers) == 1, 'Expected one worker')
            yield variant, tokens, workers[0]
    finally:
        engine.llm_engine.engine_core.shutdown()


def run_sglang(config, variants, output):
    from sglang.srt.entrypoints.engine import Engine

    class SmokeEngine(Engine):
        run_scheduler_process_func = staticmethod(run_scheduler)

    engine = SmokeEngine(**config)
    try:
        for variant in variants:
            print(f'START sglang {variant}: {TOKENS} prompt tokens, {CHUNK}-token chunks; '
                  'first use may compile kernels.', flush=True)
            engine.flush_cache()
            engine.collective_rpc('lisa_smoke_prepare', variant=variant)
            response = engine.generate(input_ids=[7] * TOKENS,
                                       sampling_params=dict(max_new_tokens=2, ignore_eos=True, temperature=0))
            metadata = response['meta_info']
            require(metadata['prompt_tokens'] == TOKENS and metadata['cached_tokens'] == 0,
                    'Prompt or prefix-cache mismatch')
            tokens = response['output_ids']
            destination = output / (variant + '-worker.json')
            engine.collective_rpc('lisa_smoke_report', destination=str(destination))
            yield variant, tokens, json.loads(destination.read_text())
    finally:
        engine.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', choices=('vllm', 'sglang'), required=True)
    parser.add_argument('--heads', type=int, choices=(32, 64), default=64)
    parser.add_argument('--variant', choices=(*VARIANTS, 'all'), default='all')
    parser.add_argument('--output', type=Path, help='new result directory; defaults to results/smoke-<timestamp>')
    parser.add_argument('--dry-run', action='store_true', help='print model and launch settings without importing CUDA or frameworks')
    args = parser.parse_args(argv)
    if args.heads == 32 and args.variant == 'shared4':
        parser.error('shared4 requires --heads 64')
    variants = list(VARIANTS[:2] if args.heads == 32 else VARIANTS) if args.variant == 'all' else [args.variant]
    check_prompt(variants)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output or Path('results') / ('smoke-' + stamp)).resolve()
    config = engine_config(args.framework, output / 'model')
    model = model_config(args.heads)
    if args.dry_run:
        print(json.dumps(dict(framework=args.framework, variants=variants,
                              model=model, engine=config), indent=2))
        return
    environment = preflight(args.framework)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'model').mkdir()
    (output / 'model/config.json').write_text(json.dumps(model, indent=2) + '\n')
    summary = dict(status='running', framework=args.framework, heads=args.heads,
                   dummy_weights=True, prompt_tokens=TOKENS, environment=environment,
                   engine=config, variants=variants, results=[])
    report_path = output / 'summary.json'
    report_path.write_text(json.dumps(summary, indent=2) + '\n')
    # Enable LISA in each worker after the framework's own startup/profile passes.
    os.environ.pop('LISA_INDEXER', None)
    os.environ.pop('LISA_REPORT_DIR', None)
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    os.environ['SGLANG_JIT_DEEPGEMM_PRECOMPILE'] = '0'
    print(f'Starting {args.framework} with dummy weights. First import/JIT compilation can take several minutes.', flush=True)
    print(f'Results: {output}', flush=True)
    runner = run_vllm(config, variants) if args.framework == 'vllm' else run_sglang(config, variants, output)
    try:
        for variant, tokens, reports in runner:
            result = dict(variant=variant, output_ids=tokens, worker_reports=reports, status='failed')
            summary['results'].append(result)
            require(len(tokens) == 2 and all(isinstance(t, int) and 0 <= t < 256 for t in tokens),
                    'Expected two valid output token IDs')
            report = check_report(reports, args.framework, variant)
            result['status'] = 'passed'
            report_path.write_text(json.dumps(summary, indent=2) + '\n')
            print(f"PASS {args.framework} H{args.heads} {variant}: {report['calls']['executed']} LISA calls, zero fallbacks", flush=True)
        require(len(summary['results']) == len(variants), 'Not all selected variants completed')
        summary['status'] = 'passed'
    except BaseException as error:
        summary.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            runner.close()
        finally:
            report_path.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
