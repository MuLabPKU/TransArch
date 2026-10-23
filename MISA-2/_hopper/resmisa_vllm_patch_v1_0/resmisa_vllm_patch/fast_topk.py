"""Capacity-safe radix TopK and the sparse-attention token ID contract.

The CUDA operator selects exact FP32 score values, with arbitrary equal-score
tie IDs. Valid-domain scores must be finite; masked positions use -infinity.
The overflow path refines the full input row instead of truncating a bucket.
"""
from pathlib import Path
from threading import Lock

import torch
from torch.utils.cpp_extension import load

MAX_TOPK = 8192
_LOADED = False
_LOAD_LOCK = Lock()


def _ensure_loaded():
    global _LOADED
    if _LOADED:
        return
    with _LOAD_LOCK:
        if not _LOADED:
            load(
                name="resmisa_fast_topk_fixed_v1",
                sources=[str(Path(__file__).parent / "csrc/fast_topk_runtime.cu")],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                is_python_module=False,
                verbose=False,
            )
            _LOADED = True


def resmisa_fast_topk(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Return unique absolute column IDs; order is unspecified until normalized.

    The CUDA implementation supports K <= 8192. Larger K explicitly uses
    torch.topk.
    """
    if scores.ndim != 2 or not scores.is_cuda or topk <= 0:
        raise ValueError("Expected CUDA scores [rows, columns] and positive topk")
    k = min(topk, scores.shape[1])
    if not k or not scores.shape[0]:
        return torch.empty((scores.shape[0], k), device=scores.device, dtype=torch.int32)
    if k > MAX_TOPK:
        return torch.topk(scores, k, dim=-1, sorted=False).indices.to(torch.int32)
    _ensure_loaded()
    if scores.dtype != torch.float32:
        scores = scores.float()
    if scores.stride(1) != 1:
        scores = scores.contiguous()
    out = torch.empty((scores.shape[0], k), dtype=torch.int32, device=scores.device)
    torch.ops.resmisa_topk_v1.fast_topk_runtime(scores, out)
    return out


def normalize_ids(ids: torch.Tensor) -> torch.Tensor:
    """Sort valid nonnegative IDs ascending, then pad with -1.

    Only order changes: duplicate IDs, if supplied by a broken selector, are not
    silently removed. Selectors must return unique valid IDs.
    """
    sentinel = torch.iinfo(torch.int32).max
    ordered = torch.where(ids >= 0, ids, sentinel).sort(dim=-1).values
    return torch.where(ordered == sentinel, -1, ordered).to(torch.int32)


def select_window(scores, starts, ends, topk):
    """Return sorted window-relative IDs for half-open [starts, ends) rows.

    The scorer masks other columns to -infinity. Short rows include every valid
    token, independent of how the radix selector breaks masked-score ties.
    """
    absolute = resmisa_fast_topk(scores, topk).long()
    k = absolute.shape[1]
    starts = starts.reshape(-1, 1)
    ends = ends.reshape(-1, 1)
    sequential = starts + torch.arange(k, device=scores.device)[None, :]
    absolute = torch.where(ends - starts <= k, sequential, absolute)
    relative = torch.where((absolute >= starts) & (absolute < ends), absolute - starts, -1)
    return normalize_ids(relative)
