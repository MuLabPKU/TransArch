"""Adapters decline unsupported scheduler shapes BEFORE touching the output."""
from collections import Counter

from ..api import Indexer
from ..api import PreparedFP8


def decline_reason(q, k, scales, weights, starts, ends, out, context, *, topk=2048, variant='linear'):
    import torch
    if not context.is_prefill:
        return 'decode'
    if context.num_requests != 1:
        return 'multiple_requests'
    if context.context_parallel_size != 1:
        return 'context_parallel'
    if context.cuda_graph:
        return 'cuda_graph'
    if context.sliding_window:
        return 'sliding_window'
    if topk != 2048:
        return 'topk'
    if q.ndim != 3 or q.shape[1] not in (32, 64) or q.shape[2] != 128:
        return 'query_shape'
    if variant == 'shared4' and q.shape[1] != 64:
        return 'shared4_heads'
    if not 0 < len(q) <= 8192 or k.ndim != 2 or k.shape[1] != 128:
        return 'shape'
    if not 0 < len(k) <= 1048576 or context.query_start + len(q) > len(k) or context.sequence_length != len(k):
        return 'sequence_layout'
    if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn:
        return 'quantization'
    if weights.shape != q.shape[:2] or weights.dtype != torch.float32 or scales.dtype != torch.float32 or scales.shape != (len(k),):
        return 'scales_or_weights'
    if any(t.device != q.device or not t.is_contiguous() for t in (q, k, scales, weights, starts, ends, out)):
        return 'tensor_layout'
    if (any(t.shape != (len(q),) or t.dtype != torch.int32 for t in (starts, ends))
            or out.shape != (len(q), 2048) or out.dtype != torch.int32):
        return 'output_or_ranges'
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) != (10, 0):
        return 'device'
    return None


class PrefillAdapter:
    """One adapter per worker/device, shared across that model's indexer layers."""
    def __init__(self, *, variant='linear', strict=False, max_n=1048576, num_layers=61):
        self.indexer = Indexer(variant, max_n=max_n, num_layers=num_layers)
        self.strict = strict
        self.calls = Counter()

    def decline(self, reason):
        self.calls['fallback:' + reason] += 1
        if self.strict:
            raise RuntimeError(f'LISA strict mode declined prefill: {reason}')
        return False

    def select_into(self, q, k, scales, weights, starts, ends, out, *, context, topk=2048, transaction=None):
        reason = decline_reason(q, k, scales, weights, starts, ends, out, context,
                                topk=topk, variant=self.indexer.variant)
        if reason is not None:
            return self.decline(reason)
        if transaction is not None:
            transaction.before(self, context, len(q))
        self.indexer.select(PreparedFP8(q, k, scales, weights), starts, ends, context=context, out=out)
        if transaction is not None:
            transaction.after(self, context, len(q))
        # Never catch kernel errors and silently report a successful fallback.
        self.calls['executed'] += 1
        if self.indexer.variant == 'shared4':
            role = 'coarse_calls' if context.layer_id == 0 or (context.layer_id - 1) % 4 == 0 else 'member_calls'
            self.calls[role] += 1
        return True

    def begin_forward(self, request_id):
        self.indexer.begin_forward(request_id)

    def end_forward(self):
        self.indexer.end_forward()
        self.calls['forwards_completed'] += 1

    def abort_forward(self):
        self.calls['forwards_aborted'] += 1
        self.indexer.reset()

    def reset(self):
        self.indexer.reset()

    def report(self):
        result = dict(variant=self.indexer.variant, calls=dict(self.calls))
        if self.indexer.variant == 'shared4':
            result['shared_state'] = self.indexer.shared_state()
        return result
