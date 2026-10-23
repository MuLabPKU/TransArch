"""Attention contract: ascending valid window-relative IDs, then -1 padding.

The selection multiset is unchanged. Invalid/duplicate IDs are not silently
repaired here: native status recovery owns selection correctness.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _ordered_ids(Ids, Out, S0: tl.constexpr, S1: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, K)
    values = tl.load(Ids + row * S0 + columns * S1)
    # Qualified native indices are >=0 or -1 and bounded by the <=1M domain.
    keys = tl.where(values >= 0, values, 2147483647)
    ordered = tl.sort(keys, descending=False)
    result = tl.where(ordered == 2147483647, -1, ordered)
    tl.store(Out + row * K + columns, result)


def normalize_ids(ids):
    if not (ids.is_cuda and ids.dtype == torch.int32 and ids.ndim == 2
            and ids.shape[1] == 2048 and 0 < ids.shape[0] <= 8192):
        raise ValueError("ID normalization requires CUDA int32 [1..8192,2048]")
    out = torch.empty(ids.shape, dtype=torch.int32, device=ids.device)
    _ordered_ids[(ids.shape[0],)](ids, out, ids.stride(0), ids.stride(1), 2048, num_warps=8)
    return out
