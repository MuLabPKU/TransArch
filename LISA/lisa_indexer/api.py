"""LISA tensor contracts, request state and selection API."""
from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class PreparedFP8:
    """Native Q/K E4M3 tensors; query scales are folded into weights once.

    q: [Q,H,128], keys: [N,128], key_scales: FP32 [N],
    effective_weights: FP32 [Q,H]. All are contiguous on one SM100 device.
    The core does not quantize inputs or apply another query scale.
    """

    q: "Tensor"
    keys: "Tensor"
    key_scales: "Tensor"
    effective_weights: "Tensor"


@dataclass(frozen=True)
class PrefillContext:
    """CPU scheduler metadata. query_start is relative to this request."""

    request_id: str
    layer_id: int
    query_start: int
    sequence_length: int
    num_requests: int = 1
    is_prefill: bool = True
    context_parallel_size: int = 1
    cuda_graph: bool = False
    sliding_window: bool = False

    def __post_init__(self):
        for key in ("layer_id", "query_start", "sequence_length", "num_requests", "context_parallel_size"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if not self.request_id:
            raise ValueError("a stable request_id is required for carry/pool lifetime")
        if self.query_start > self.sequence_length:
            raise ValueError("query_start exceeds sequence_length")


def validate_prepared(prepared, *, require_cuda=True):
    import torch

    if not isinstance(prepared, PreparedFP8):
        raise TypeError("PreparedFP8 required; no implicit input quantization")
    q, k, scales, weights = (prepared.q, prepared.keys, prepared.key_scales,
                             prepared.effective_weights)
    if q.ndim != 3 or q.shape[1] not in (32, 64) or q.shape[2] != 128:
        raise ValueError("expected Q[rows,32|64,128]")
    if k.ndim != 2 or k.shape[1] != 128 or weights.shape != q.shape[:2]:
        raise ValueError("expected K[N,128] and effective weights[rows,H]")
    if not 0 < len(q) <= 8192 or not 0 < len(k) <= 1048576:
        raise ValueError("supported sizes: 1<=Q<=8192 and 1<=N<=1048576")
    if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn:
        raise TypeError("native E4M3 Q and K required")
    if scales.shape != (len(k),) or scales.dtype != torch.float32 or weights.dtype != torch.float32:
        raise TypeError("FP32 key scales[N] and effective weights[Q,H] required")
    if any(t.device != q.device or not t.is_contiguous() for t in (q, k, scales, weights)):
        raise ValueError("inputs must be contiguous and on one device")
    if require_cuda and (not q.is_cuda or torch.cuda.get_device_capability(q.device) != (10, 0)):
        raise RuntimeError("LISA kernels require an SM100 GPU (GB200/B200); no kernel was launched")


def validate_ranges(prepared, starts, ends):
    import torch

    if any(t.shape != (len(prepared.q),) or t.dtype != torch.int32
           or t.device != prepared.q.device or not t.is_contiguous() for t in (starts, ends)):
        raise ValueError("ranges must be contiguous int32 [Q] on the input device")


def validate_output(prepared, out):
    import torch

    if (out.shape != (len(prepared.q), 2048) or out.dtype != torch.int32
            or out.device != prepared.q.device or not out.is_contiguous()):
        raise ValueError("output must be contiguous int32 [Q,2048] on the input device")
    if any(out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
           for t in (prepared.q, prepared.keys, prepared.key_scales, prepared.effective_weights)):
        raise ValueError("output must not alias an input")


class ForwardLedger:
    def __init__(self, num_layers):
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError('num_layers must be a positive integer')
        self.num_layers = num_layers
        self.generation = 0
        self.active = False
        self.chunks = {}

    def begin(self):
        if self.active:
            # A new forward can begin only after every chunk completes all layers.
            self.finish()
        self.chunks = {}
        self.generation += 1
        self.active = True

    def check(self, layer, chunk, layout):
        if not self.active:
            raise RuntimeError('shared4 requires an active model forward')
        if not 0 <= layer < self.num_layers:
            raise ValueError('indexer layer outside the configured model')
        expected, saved = self.chunks.get(chunk, (0, layout))
        if saved != layout:
            raise ValueError('shared4 chunk layout changed between layers')
        if layer != expected:
            raise RuntimeError(f'shared4 chunk {chunk}: expected layer {expected}, got {layer}')

    def commit(self, layer, chunk, layout):
        self.check(layer, chunk, layout)
        self.chunks[chunk] = (layer + 1, layout)

    def finish(self):
        if not self.active:
            raise RuntimeError('no active shared4 forward to finish')
        missing = {str(chunk): step for chunk, (step, _) in self.chunks.items() if step != self.num_layers}
        if missing:
            raise RuntimeError(f'incomplete shared4 forward; next expected layers: {missing}')
        self.active = False

    def abort(self):
        self.active = False
        self.chunks = {}


class Indexer:
    """Single-request, single-stream FP8 prefill indexer.

    variants: linear (LISA), threshold (LISA dagger), shared4 (double dagger).
    start/end indices and outputs are request-relative; starts must be zero.
    shared4 requires begin_forward() before each chunk and ordered layer calls.
    """

    def __init__(self, variant='linear', *, max_n=1048576, num_layers=61):
        if variant not in ('linear', 'threshold', 'shared4'):
            raise ValueError('variant must be linear, threshold, or shared4')
        if isinstance(max_n, bool) or not isinstance(max_n, int) or not 0 < max_n <= 1048576:
            raise ValueError('max_n must be an integer in [1,1048576]')
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError('num_layers must be positive')
        self.variant, self.max_n, self.num_layers = variant, max_n, num_layers
        self._lock = threading.Lock()
        self._stream = self._device = self._done = self._runtime = None
        self._keys = self._scales = self._workspace = None
        self._request = None
        self._forward_pending = False
        self._broken = False
        self.last_stats = {}

    def _reset_locked(self):
        if self._done is not None:
            self._done.synchronize()
        self._runtime = self._keys = self._scales = self._workspace = None
        self._stream = self._device = self._done = self._request = None
        self._forward_pending = self._broken = False
        self.last_stats = {}

    def reset(self):
        """Complete pending work and release request-specific state."""
        with self._lock:
            self._reset_locked()

    close = reset

    def begin_forward(self, request_id):
        """Explicit model-forward hook for shared4 (before its first indexer)."""
        with self._lock:
            if self.variant != 'shared4':
                raise ValueError('begin_forward is only required for shared4')
            if not request_id:
                raise ValueError('a stable request_id is required')
            if self._forward_pending:
                raise RuntimeError('shared4 forward already pending')
            if request_id != self._request:
                self._reset_locked()
                self._request = request_id
            if self._runtime is not None:
                self._runtime.begin_forward()
            else:
                self._forward_pending = True

    def end_forward(self):
        """Check every chunk reached every layer, with no outstanding leases."""
        with self._lock:
            if self.variant != 'shared4':
                raise ValueError('end_forward is only for shared4')
            if self._runtime is not None:
                self._runtime.end_forward()
            elif not self._forward_pending:
                raise RuntimeError('no shared4 forward is pending')
            self._forward_pending = False

    def shared_state(self):
        if self.variant != 'shared4' or self._runtime is None:
            return dict(coarse_calls=0, member_calls=0, live_pools=0, forward_active=self._forward_pending)
        return dict(coarse_calls=self._runtime.coarse_calls, member_calls=self._runtime.member_calls,
                    live_pools=len(self._runtime.pools), forward_active=self._runtime.ledger.active)

    def _init_threshold(self, prepared):
        import torch
        if self._keys is None:
            # Recovery graphs retain these pointers across layer calls.
            self._keys = torch.empty((self.max_n, 128), dtype=prepared.keys.dtype, device=prepared.q.device)
            self._scales = torch.empty(self.max_n, dtype=torch.float32, device=prepared.q.device)
        self._keys[:len(prepared.keys)].copy_(prepared.keys)
        self._scales[:len(prepared.keys)].copy_(prepared.key_scales)
        return PreparedFP8(prepared.q, self._keys[:len(prepared.keys)], self._scales[:len(prepared.keys)],
                            prepared.effective_weights)

    def select(self, prepared, starts, ends, *, context, out=None):
        import torch

        validate_prepared(prepared)
        validate_ranges(prepared, starts, ends)
        rows, n = len(prepared.q), len(prepared.keys)
        if (n > self.max_n or n != context.sequence_length or context.query_start + rows > n
                or context.num_requests != 1 or not context.is_prefill or context.context_parallel_size != 1
                or context.cuda_graph or context.sliding_window):
            raise ValueError('requires one eager causal prefill request with Q<=8192, CP=1, no sliding window')
        if not 0 <= context.layer_id < self.num_layers:
            raise ValueError('layer_id outside configured num_layers')
        if self.variant == 'shared4' and prepared.q.shape[1] != 64:
            raise ValueError('shared4 requires 64 query heads')
        if out is None:
            out = torch.empty((rows, 2048), dtype=torch.int32, device=prepared.q.device)
        validate_output(prepared, out)
        if any(out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr() for x in (starts, ends)):
            raise ValueError('output must not alias ranges')
        with self._lock, torch.cuda.device(prepared.q.device):
            if self._broken:
                raise RuntimeError('previous kernel call failed; reset the indexer before reuse')
            if self._request != context.request_id:
                if self.variant == 'shared4':
                    raise RuntimeError('call begin_forward(request_id) before shared4 selection')
                self._reset_locked()
                self._request = context.request_id
            stream = torch.cuda.current_stream(prepared.q.device)
            identity = (prepared.q.device, stream.cuda_stream)
            if self._stream is not None and self._stream != identity:
                raise RuntimeError('Indexer owns one CUDA stream; reset before changing stream/device')
            self._stream, self._device = identity, prepared.q.device
            from .kernels import causal_ranges
            import triton
            causal_ranges[(triton.cdiv(rows, 256),)](starts, ends, rows, n, context.query_start, 256, debug=True)
            try:
                if self.variant == 'linear':
                    from .kernels.linear import LinearRuntime
                    if self._runtime is None:
                        self._runtime = LinearRuntime(self.max_n)
                    self._runtime.select(prepared, starts, ends, out,
                        layer_id=context.layer_id, query_start=context.query_start)
                    self.last_stats = self._runtime.last_stats
                else:
                    from .kernels.threshold import native, recovery, score
                    p = self._init_threshold(prepared)
                    if self.variant == 'threshold':
                        if self._workspace is None or self._workspace.rows != rows:
                            if self._done is not None:
                                self._done.synchronize()
                            self._workspace = recovery.Workspace(rows, p.q.device, score.Budget(), self.max_n)
                        result = native.checked(p, starts, ends, workspace=self._workspace, max_n=self.max_n)
                        out.copy_(result.indices)
                    else:
                        if self._runtime is None:
                            import json
                            from pathlib import Path
                            from .kernels.threshold.pool import PoolEngine
                            policy = json.loads((Path(__file__).parent / 'kernels/threshold/policy_g4.json').read_text())
                            self._runtime = PoolEngine(max_n=self.max_n, policy=policy, num_layers=self.num_layers)
                            if self._forward_pending:
                                self._runtime.begin_forward()
                                self._forward_pending = False
                        result = self._runtime.select(p, starts, ends, out, layer=context.layer_id,
                            chunk=(context.query_start, rows))
                    self.last_stats = dict(route=self.variant, gpu_stats=result.gpu_stats,
                                           status=result.fast.candidates.status,
                                           counts=result.fast.candidates.counts,
                                           candidate_ids=result.fast.candidates.ids)
                    from .kernels import threshold_status
                    threshold_status[(triton.cdiv(rows, 256),)](result.fast.candidates.status,
                        result.fast.candidates.counts, ends, rows, 256, debug=True)
            except Exception:
                self._broken = True
                raise
            finally:
                if self._done is None:
                    self._done = torch.cuda.Event()
                self._done.record(stream)
        return out
