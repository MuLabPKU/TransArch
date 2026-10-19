"""Shared vLLM configuration and resume/provenance checks for both experiments."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

from paths import ROOT, MODEL_V4


def add_arguments(parser):
    parser.add_argument('--model', type=Path, default=MODEL_V4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Run only the first N inputs; partial results stay partial')
    parser.add_argument('--tp', type=int, default=8)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.92)
    parser.add_argument('--max-num-batched-tokens', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=33377335)


def engine_options(args, max_model_len):
    if args.tp < 1 or args.max_num_batched_tokens < 1:
        raise ValueError('TP and token batch size must be positive')
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError('GPU memory utilization must be between 0 and 1')
    if args.limit is not None and args.limit < 1:
        raise ValueError('Limit must be positive')
    return dict(model=str(args.model.resolve()), tensor_parallel_size=args.tp,
                enable_expert_parallel=True, trust_remote_code=True,
                max_model_len=max_model_len, max_num_seqs=1,
                max_num_batched_tokens=min(args.max_num_batched_tokens,max_model_len),
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=True, seed=args.seed, enable_prefix_caching=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(args, options, **inputs):
    config=args.model/'config.json'
    index=args.model/'model.safetensors.index.json'
    if not index.is_file():
        raise ValueError('Use the original Hugging Face checkpoint, including model.safetensors.index.json')
    return dict(backend='vllm', model=str(args.model.resolve()), engine_options=options,
                versions={name:version(name) for name in ('torch','vllm','tilelang','safetensors','transformers')},
                model_config_sha256=digest(config), model_index_sha256=digest(index), **inputs)


def pending_rows(rows, output, run_identity, limit=None):
    ids=[r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate input IDs')
    done=[]
    if output.exists() and output.stat().st_size:
        old=json.loads(output.with_suffix('.provenance.json').read_text())
        if any(old.get(k)!=v for k,v in run_identity.items()):
            raise ValueError('Resume model, input, or engine settings changed; choose another output')
        done=[json.loads(line)['id'] for line in output.read_text().splitlines()]
        if len(set(done))!=len(done) or not set(done).issubset(ids):
            raise ValueError('Duplicate or unknown IDs in existing results')
    selected=rows if limit is None else rows[:limit]
    completed=set(done)
    return [r for r in selected if r['id'] not in completed]


def load_engine(args, options, run_identity, script, **details):
    os.environ.setdefault('VLLM_DEEP_GEMM_WARMUP','skip')
    import torch
    from vllm import LLM
    start=time.monotonic()
    llm=LLM(**options)
    report={**run_identity, **details,
            'engine_initialization_seconds':time.monotonic()-start,
            'deep_gemm_warmup':os.environ['VLLM_DEEP_GEMM_WARMUP'],
            'gpu':torch.cuda.get_device_name(), 'cuda':torch.version.cuda,
            'source_sha256':{p.name:digest(p) for p in (Path(script),ROOT/'inference.py',ROOT/'paths.py')}}
    config=llm.llm_engine.vllm_config
    engram=getattr(config,'engram_config',None)
    report['engram_config']=str(engram) if engram is not None else None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    return llm


def append_result(output, result):
    with output.open('a') as f:
        f.write(json.dumps(result,ensure_ascii=False)+'\n')
