"""FP32 minmax scoring, using explicit local kernel calls.

The optimized contract is seven residual heads: FP8 Q, H64/D128, pool1024,
M <= 8192 and N <= 1M. The score is an FP32 minmax bound with the accepted
near-tie repair. It can differ from the colleague's shared_bf16 approximation.
All buffers are local to a call; this helper has no numerical request cache.
"""

import torch
import triton

from ..maxmin_head_score import (
    _boundary_lookup,
    _pool,
    deterministic_heads,
    maxmin_head_scores as _received_head_scores,
)
from .pooling import boundary_extrema, prefix_extrema
from .repair import repair
from .scoring import launch


def configuration(m, n):
    """Fixed minmax launch configuration, without runtime rebinding."""
    if m == 8192:
        return (64, 32, 4, 1, 1) if n <= 131072 else (64, 32, 8, 1, 2)
    return (64, 32, 4, 1, 2)


def maxmin_head_scores(q, k, scales, weights, starts, ends, pool_size):
    """Fixed minmax prefill scorer; retain bounds checks and geometry fallback."""
    eligible = (
        q.ndim == 3 and tuple(q.shape[1:]) == (64, 128)
        and 0 < q.shape[0] <= 8192 and q.dtype == torch.float8_e4m3fn
        and k.ndim == 2 and k.shape[1] == 128 and 0 < k.shape[0] <= 1048576
        and pool_size == 1024
    )
    if not eligible:
        return _received_head_scores(
            q, k, scales, weights, starts, ends, pool_size, "maxmin", "auto",
            validate_bounds=True, score_backend="shared_bf16")

    m, h, d = q.shape
    n = k.shape[0]
    if scales.dtype != torch.float32 or scales.numel() != n:
        raise ValueError("one FP32 dequantization scale per K token required")
    if weights.shape != (m, h) or starts.numel() != m or ends.numel() != m:
        raise ValueError("weights/bounds shape mismatch")
    if not all(t.is_cuda and t.device == q.device
               for t in (q, k, scales, weights, starts, ends)):
        raise ValueError("all inputs must be on the same CUDA device")
    if starts.dtype not in (torch.int32, torch.int64) or ends.dtype not in (torch.int32, torch.int64):
        raise ValueError("integer token bounds required")
    if bool(((starts < 0) | (ends < starts) | (ends > n)).any()):
        raise ValueError("require 0 <= start <= end <= N")

    starts = starts.reshape(-1).int().contiguous()
    ends = ends.reshape(-1).int().contiguous()
    nb = triton.cdiv(n, pool_size)
    bounds = torch.empty((nb + 2 * m, 3, d), device=q.device, dtype=torch.float32)
    scores = torch.empty((m, h), device=q.device, dtype=torch.float32)
    key, key_scales = k.contiguous(), scales.contiguous()
    use_prefix = m >= nb
    if use_prefix:
        prefix = torch.empty((n, 2, d), device=q.device, dtype=torch.float32)
        suffix = torch.empty_like(prefix)
        prefix_extrema(
            key, key_scales, starts, ends, prefix, suffix, bounds,
            n, d, pool_size, triton.next_power_of_2(pool_size), 8, num_warps=8)
        _boundary_lookup[(triton.cdiv(m, 16),)](
            starts, ends, prefix, suffix, bounds, m, n, d, nb, pool_size,
            triton.next_power_of_2(d), 16, num_warps=4)
        boundary_extrema(
            key, key_scales, starts, ends, bounds,
            n, d, nb, pool_size, triton.next_power_of_2(pool_size), 32, num_warps=8)
    else:
        _pool[(nb + 2 * m, triton.cdiv(d, 16))](
            key, key_scales, starts, ends, bounds, n, d, nb, pool_size,
            triton.next_power_of_2(pool_size), 16, d, num_warps=8)
    query, effective_weights = q.contiguous(), weights.contiguous()
    launch(query, effective_weights, bounds, starts, ends, scores,
           nb, pool_size, configuration(m, n))
    repair(query, effective_weights, bounds, starts, ends, scores,
           nb, pool_size, relative=1.e-5)
    return scores
