"""Hopper-named variant: September minmax in the colleague prefill pipeline.

Only the qualified seven-residual-head route changes. The fused query and
final merged scorer are the colleague implementations, imported normally.
Paged decode remains in resmisa_ops_fp8.py and never imports this module.
"""

import torch

from .hopper.head_selection import deterministic_heads, maxmin_head_scores
from .maxmin_head_score import build_fused_head_q_exact
from .resmisa_ops_fp8 import resmisa_mqa_logits as _received_mqa_logits
from .triton_kernel import resmisa_mqa_fused_merged_8h_v2_logits_triton


def resmisa_mqa_logits(q, kv, weights, q_head_topk_indices, cu_seqlen_ks,
                      cu_seqlen_ke, use_qnorm, topk_heads, k_block_size=None,
                      resmisa_fuse=True, resmisa_minmax=False):
    """Return dense prefill logits, with the original callable's arguments."""
    k_fp8, k_scales = kv
    eligible = (
        q.ndim == 3 and tuple(q.shape[1:]) == (64, 128)
        and 0 < q.shape[0] <= 8192 and q.dtype == torch.float8_e4m3fn
        and k_fp8.ndim == 2 and k_fp8.shape[1] == 128
        and 0 < k_fp8.shape[0] <= 1048576 and topk_heads == 7
        and k_block_size == 1024 and resmisa_fuse and resmisa_minmax
    )
    if not eligible:
        return _received_mqa_logits(
            q, kv, weights, q_head_topk_indices, cu_seqlen_ks, cu_seqlen_ke,
            use_qnorm, topk_heads, k_block_size=k_block_size,
            resmisa_fuse=resmisa_fuse, resmisa_minmax=resmisa_minmax)

    if not k_scales.is_contiguous():
        k_scales = k_scales.contiguous()
    k_scales = k_scales.view(torch.float32)
    if k_scales.ndim == 2:
        k_scales = k_scales.squeeze(-1)
    weights = weights.bfloat16()
    starts = cu_seqlen_ks if cu_seqlen_ks.dtype == torch.int32 else cu_seqlen_ks.to(torch.int32)
    ends = cu_seqlen_ke if cu_seqlen_ke.dtype == torch.int32 else cu_seqlen_ke.to(torch.int32)
    scores = maxmin_head_scores(
        q, k_fp8, k_scales, weights, starts, ends, k_block_size)
    heads = deterministic_heads(scores, topk_heads)
    q_fused, q_scale = build_fused_head_q_exact(q, weights)
    expanded = heads.unsqueeze(-1).expand(-1, -1, q.size(-1))
    sparse_q = torch.gather(q, dim=1, index=expanded)
    sparse_weights = torch.gather(weights, dim=1, index=heads)
    logits = resmisa_mqa_fused_merged_8h_v2_logits_triton(
        q_fused=q_fused, q_scale=q_scale, q_sparse=sparse_q,
        sparse_weights=sparse_weights, kv=k_fp8, k_scales=k_scales,
        cu_seqlen_ks=starts, cu_seqlen_ke=ends)
    return logits

from .fast_topk import select_window

def select_prefill(q, kv, weights, q_head_topk_indices, starts, ends,
                   use_qnorm, topk_heads, *, k_block_size=None,
                   resmisa_fuse=True, resmisa_minmax=False,
                   topk_tokens=2048):
    """Selected minmax prefill and radix token TopK."""
    logits = resmisa_mqa_logits(
        q, kv, weights, q_head_topk_indices, starts, ends, use_qnorm,
        topk_heads, k_block_size=k_block_size, resmisa_fuse=resmisa_fuse,
        resmisa_minmax=resmisa_minmax,
    )
    return select_window(logits, starts, ends, topk_tokens)


