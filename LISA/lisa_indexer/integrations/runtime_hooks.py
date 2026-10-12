"""Worker-local indexer state for one offline request at a time."""
import atexit
import json
import os
from pathlib import Path
import re

from ..api import PrefillContext

_ADAPTERS = {}


def _adapter(framework, device):
    variant = os.environ.get('LISA_INDEXER', '')
    if variant not in ('linear', 'threshold', 'shared4'):
        raise ValueError('LISA_INDEXER must be linear, threshold, or shared4')
    if os.environ.get('LISA_SINGLE_REQUEST') != '1':
        raise RuntimeError('hooks support one offline request per process; set LISA_SINGLE_REQUEST=1')
    scope = None
    if variant == 'shared4':
        from .forward_scope import current_scope
        scope = current_scope(framework)
        num_layers = scope.num_layers
        if 'LISA_NUM_LAYERS' in os.environ and int(os.environ['LISA_NUM_LAYERS']) != num_layers:
            raise ValueError('LISA_NUM_LAYERS disagrees with model.config.num_hidden_layers')
    else:
        num_layers = int(os.environ.get('LISA_NUM_LAYERS', '128'))
    key = (framework, str(device), variant, id(scope.model) if scope is not None else None)
    if key not in _ADAPTERS:
        if framework == 'vllm':
            from .vllm import VLLMAdapter as Adapter
        else:
            from .sglang import SGLangAdapter as Adapter
        _ADAPTERS[key] = Adapter(variant=variant, strict=os.environ.get('LISA_STRICT', '0') == '1',
                                 num_layers=num_layers)
    return _ADAPTERS[key]


def prefill(framework, q, k, scales, weights, starts, ends, out, *, layer,
            num_requests=1, cp_size=1, capturing=False, sliding_window=False, topk=2048,
            query_start=None):
    adapter = _adapter(framework, q.device)
    transaction = None
    if adapter.indexer.variant == 'shared4':
        from .forward_scope import current_scope
        transaction = current_scope(framework)
        if transaction.unsupported:
            return adapter.decline(transaction.unsupported)
    if isinstance(layer, str):
        match = re.search(r'(?:layers|h)\.(\d+)(?:\.|$)', layer)
        if match is None:
            return adapter.decline('layer_name')
        layer = int(match.group(1))
    context = PrefillContext(request_id=os.environ.get('LISA_REQUEST_ID', 'offline-single-request'),
        layer_id=layer, query_start=max(0, len(k) - len(q)) if query_start is None else query_start,
        sequence_length=len(k),
        num_requests=num_requests, context_parallel_size=cp_size,
        cuda_graph=capturing, sliding_window=sliding_window)
    executed = adapter.select_prefill(q, k, scales, weights, starts, ends, out,
        context=context, topk=topk, transaction=transaction)
    if executed:
        adapter.calls['route:' + adapter.indexer.last_stats['route']] += 1
    return executed


def reset():
    """Call in each worker before another offline request in the same process."""
    for adapter in _ADAPTERS.values():
        adapter.reset()


def reports():
    from .forward_scope import COUNTERS
    return [dict(framework=key[0], device=key[1],
                 model_forward_hooks={k: v for k, v in COUNTERS.items() if k.startswith(key[0] + ':')},
                 **adapter.report()) for key, adapter in _ADAPTERS.items()]


def write_reports():
    directory = os.environ.get('LISA_REPORT_DIR')
    if directory:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / f'lisa-worker-{os.getpid()}.json').write_text(json.dumps(reports(), indent=2) + '\n')


atexit.register(write_reports)
