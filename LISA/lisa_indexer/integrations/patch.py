#!/usr/bin/env python3
"""Generate or apply a narrow, hash-checked patch to a supported framework source."""
import argparse
import ast
import difflib
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VLLM_INSERT = '''                # LISA_SOURCE_HOOK_V1: after FP8 gather, before logits/TopK.
                if os.environ.get("LISA_INDEXER"):
                    from lisa_indexer.integrations.runtime_hooks import prefill as lisa_prefill
                    if lisa_prefill(
                        "vllm", q_slice_cast, k_quant_cast, k_scale_cast,
                        weights[chunk.token_start:chunk.token_end],
                        cu_seqlen_ks, cu_seqlen_ke, topk_indices,
                        layer=k_cache_prefix,
                        num_requests=(attn_metadata_narrowed.num_prefills
                                      + attn_metadata_narrowed.num_decodes),
                        query_start=(chunk.total_seq_lens
                                     - attn_metadata_narrowed.num_prefill_tokens
                                     + chunk.token_start
                                     if attn_metadata_narrowed.num_prefills == 1
                                     and not has_decode and dcp_world_size == 1
                                     and not use_pcp else 0),
                        cp_size=max(dcp_world_size, 2 if use_pcp else 1),
                        capturing=torch.cuda.is_current_stream_capturing(),
                        topk=topk_tokens,
                    ):
                        continue
'''

SGLANG_INSERT = '''        # LISA_SOURCE_HOOK_V1: after FP8 gather, before logits/TopK.
        if os.environ.get("LISA_INDEXER"):
            from lisa_indexer.integrations.runtime_hooks import _adapter, prefill as lisa_prefill
            from lisa_indexer.integrations.sglang import transform_indices
            lisa_rows = len(ks)
            if (batch_size != 1 or self.dsa_enable_prefill_cp or get_is_capture_mode()
                    or self.num_init_tokens or self.num_local_tokens
                    or self.index_topk != 2048 or not 0 < lisa_rows <= 8192
                    or not 0 < seq_len_sum <= 1048576):
                _adapter("sglang", device).decline("scheduler_shape")
            elif lisa_prefill(
                "sglang", q_fp8[:lisa_rows], k_fp8, k_scale, weights[:lisa_rows],
                ks, ke, topk_result[:lisa_rows], layer=layer_id,
                num_requests=batch_size, topk=self.index_topk,
            ):
                transform_indices(metadata, topk_result[:lisa_rows])
                return topk_result
'''


def patched(framework, original):
    if 'LISA_SOURCE_HOOK_V1' in original:
        raise ValueError('source is already patched')
    if framework == 'vllm':
        anchor = '                if current_platform.is_xpu():\n'
        insert = VLLM_INSERT + anchor
    else:
        anchor = '        kv_fp8 = (k_fp8, k_scale)\n'
        insert = anchor + '\n' + SGLANG_INSERT
    if original.count(anchor) != 1:
        raise ValueError('expected call-site anchor is missing or ambiguous')
    result = original.replace(anchor, insert, 1)
    if framework == 'vllm':
        if original.count('\nimport torch\n') != 1:
            raise ValueError('missing or ambiguous indexer import anchor')
        result = result.replace('\nimport torch\n', '\nimport os\nimport torch\n', 1)
    else:
        if original.count('import contextlib\n') != 1:
            raise ValueError('missing or ambiguous indexer import anchor')
        result = result.replace('import contextlib\n', 'import contextlib\nimport os\n', 1)
    ast.parse(result)
    return result


def patched_model(framework, original):
    if 'LISA_MODEL_FORWARD_V1' in original:
        raise ValueError('model forward is already patched')
    tree = ast.parse(original)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DeepseekV2Model']
    if len(classes) != 1:
        raise ValueError('expected exactly one DeepseekV2Model class')
    forwards = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == 'forward']
    if len(forwards) != 1:
        raise ValueError('expected exactly one model forward method')
    node = forwards[0]
    line = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    lines = original.splitlines(keepends=True)
    lines.insert(line, f'    @lisa_model_forward("{framework}")  # LISA_MODEL_FORWARD_V1\n')
    result = ''.join(lines)
    if result.count('\nimport torch\n') != 1:
        raise ValueError('missing or ambiguous model import anchor')
    result = result.replace('\nimport torch\n', '\nimport torch\nfrom lisa_indexer.integrations.forward_scope import model_forward as lisa_model_forward\n', 1)
    ast.parse(result)
    return result


def patched_metadata(framework, original):
    if framework != 'vllm' or 'LISA_PREFILL_BUDGET_V1' in original:
        raise ValueError('metadata patch requires an unpatched vLLM source')
    anchor = '            max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024\n'
    if original.count(anchor) != 1:
        raise ValueError('missing or ambiguous vLLM logits-budget anchor')
    insert = '''            # LISA_PREFILL_BUDGET_V1: keep supported LISA queries unsplit.
            if os.environ.get("LISA_INDEXER"):
                from lisa_indexer.integrations.vllm import prefill_logits_budget
                max_logits_bytes = prefill_logits_budget(
                    self, max_logits_bytes, num_prefills, num_decodes,
                    prefill_query_lens_cpu, compressed_seq_lens_cpu[num_decodes:],
                )
'''
    result = original.replace(anchor, anchor + insert, 1)
    if original.count('\nimport torch\n') != 1:
        raise ValueError('missing or ambiguous metadata import anchor')
    result = result.replace('\nimport torch\n', '\nimport os\nimport torch\n', 1)
    ast.parse(result)
    return result


def resolve_inputs(args, manifest):
    """Find framework sources in an installed package or a source checkout."""
    if args.framework_root:
        root = args.framework_root.resolve()
    else:
        spec = find_spec(args.framework)
        if spec is None or spec.origin is None:
            raise ValueError(f'install {args.framework} or provide --framework-root')
        root = Path(spec.origin).resolve().parent.parent
    if args.framework == 'sglang' and (root / 'python/sglang').is_dir():
        root = root / 'python'
    inputs = [(root / manifest['path'], manifest, patched),
              (root / manifest['model']['path'], manifest['model'], patched_model)]
    if args.framework == 'vllm':
        inputs.append((root / manifest['metadata']['path'], manifest['metadata'], patched_metadata))
    for record in manifest.get('model_dependencies', {}).get(args.model_family, []):
        inputs.append((root / record['path'], dict(record, patched_sha256=record['sha256']), None))
    return inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', choices=['vllm', 'sglang'], required=True)
    parser.add_argument('--model-family', choices=['dsv32', 'glm52'], default='dsv32')
    parser.add_argument('--framework-root', type=Path, help='checkout root or installed package parent; defaults to the installed framework')
    parser.add_argument('--output', type=Path, help='write unified diff (default: stdout)')
    parser.add_argument('--apply', action='store_true', help='also write source, retaining an exact .lisa-original backup')
    parser.add_argument('--check', action='store_true', help='report source compatibility as JSON without writing files')
    args = parser.parse_args(argv)
    if args.check and (args.apply or args.output):
        parser.error('--check cannot be combined with --apply or --output')
    manifest = json.loads((ROOT / 'integrations/sources.json').read_text())[args.framework]
    try:
        inputs = resolve_inputs(args, manifest)
    except ValueError as error:
        parser.error(str(error))
    changes, patches, statuses = [], [], []
    # Check all hashes and backups before writing; allow a partially installed
    # patch to resume without rewriting files that already match.
    for path, expected, transform in inputs:
        if not path.is_file():
            raise SystemExit(f'missing framework source: {path}; provide the checkout root, not its Python package directory')
        original = path.read_text()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        backup = path.with_suffix(path.suffix + '.lisa-original')
        if backup.exists() and hashlib.sha256(backup.read_bytes()).hexdigest() != expected['sha256']:
            raise SystemExit(f'backup differs from the inspected original: {backup}')
        state = ('verified' if transform is None else 'patched') if digest == expected['patched_sha256'] else 'original'
        statuses.append(dict(path=expected['path'], sha256=digest, state=state))
        if digest == expected['patched_sha256']:
            continue
        if digest != expected['sha256']:
            raise SystemExit(f'source hash mismatch for {path}: {digest}; see README.md')
        result = transform(args.framework, original)
        if hashlib.sha256(result.encode()).hexdigest() != expected['patched_sha256']:
            raise SystemExit('generated patch differs from the packaged source manifest')
        changes.append((path, backup, original, result))
        patches.append(''.join(difflib.unified_diff(original.splitlines(True), result.splitlines(True),
                                        fromfile='a/' + expected['path'], tofile='b/' + expected['path'])))
    if args.check:
        print(json.dumps(dict(framework=args.framework, compatible=True,
                              ready=all(s['state'] in ('patched', 'verified') for s in statuses), files=statuses), indent=2))
        return
    patch = ''.join(patches)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(patch)
    else:
        print(patch, end='')
    if args.apply:
        for path, backup, original, result in changes:
            if not backup.exists():
                backup.write_text(original)
            path.write_text(result)


if __name__ == '__main__':
    main()
