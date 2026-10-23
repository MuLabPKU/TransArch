import os
import torch
from .block_ops_fp8 import (block_mean_pooling_interface, 
                        block_mqa_attn_return_logits_interface,
                        paged_mean_pooling_interface,
                        paged_mean_pooling_incremental_interface,
                        batch_block_mqa_attn_return_logits_interface)
from . import pooling_cache
from .paged_minmax import paged_minmax_head_indices
from .maxmin_head_score import (
    maxmin_head_scores, deterministic_heads, build_fused_head_q_exact,
)
from .triton_kernel import (
        paged_mqa_sparse_head_return_logits_triton,
        resmisa_mqa_sparse_head_return_logits_triton,
        paged_mqa_fused_head_linear_logits_triton,
        resmisa_mqa_fused_head_linear_logits_triton,
        resmisa_mqa_fused_merged_logits_triton,
        resmisa_mqa_fused_merged_8h_v2_logits_triton,
        paged_mqa_fused_merged_logits_triton,
        paged_mqa_fused_merged_8h_v2_logits_triton,
    )


_FP8_MAX = 448.0


# Static head pruning and head-selection frequency calibration.
# ResMISA_CALIB_HEAD=1 saves histograms for per-layer fixed-head selection.
# Nonempty config.resmisa_fixed_topk_heads supplies fixed_head_indices to the ops,
# bypassing pooling and top-k scoring while reusing gathering and logits kernels.


def _layer_num_from_prefix(cache_prefix):
    """Extract the layer number from cache_prefix, or return 'unknown' if absent."""
    import re as _re
    if cache_prefix is None:
        return "unknown"
    m = _re.search(r"layers\.(\d+)", cache_prefix)
    return m.group(1) if m else "unknown"


def _build_fused_head_q(q_fp8: torch.Tensor, weights: torch.Tensor):
    """Build q_fused = sum_h(w_h * q_h) and quantize to FP8 with per-token scales.

    The linear term of relu(x) = x + relu(-x) is
        sum_h(w_h * scale * (k @ q_h)) = scale * (k @ q_fused).
    Sum weighted heads in FP32 before quantizing to avoid per-head rounding.

    Args:
        q_fp8: [B, S, H, D] FP8; q_scale is absorbed into weights.
        weights: [B, H] or [B, S, H] head weights, including q_scale.
    Returns:
        q_fused_fp8: [B, S, D] float8_e4m3fn.
        q_scale: [B, S] FP32 per-token dequantization factors.
    """
    q_f32 = q_fp8.to(torch.float32)                       # [B, S, H, D]
    # Broadcast weights according to their layout:
    # decode: [B, H] shares head weights over S, expanding to [B, 1, H, 1].
    # prefill: [B, S, H] has per-token weights, expanding to [B, S, H, 1].
    w_f32 = weights.to(torch.float32)
    if w_f32.dim() == q_f32.dim() - 2:        # [B, H]: add S and D axes
        w_f32 = w_f32[:, None, :, None]       # [B, 1, H, 1]
    elif w_f32.dim() == q_f32.dim() - 1:      # [B, S, H]: add the D axis
        w_f32 = w_f32[..., None]              # [B, S, H, 1]
    else:
        raise ValueError(f"unexpected weights shape {tuple(weights.shape)} for q {tuple(q_fp8.shape)}")
    q_fused = (q_f32 * w_f32).sum(dim=-2)                 # [B, S, D]
    amax = torch.clamp(q_fused.abs().amax(dim=-1, keepdim=True), min=1e-12)  # [B, S, 1]
    q_scale = amax / _FP8_MAX                            # [B, S, 1]
    q_fused_fp8 = (q_fused / q_scale).to(torch.float8_e4m3fn)  # [B, S, D]
    return q_fused_fp8, q_scale.squeeze(-1)               # [B,S,D], [B,S]


def _pooling_with_cache(
    kv_cache_fp8: torch.Tensor,
    context_lens_i32: torch.Tensor,
    block_tables_i32: torch.Tensor,
    k_block_size: int,
    dim: int,
    variant: str,
    cache_prefix: str | None,
):
    """Select full pooling or incremental pooling with a persistent cache.

    Incremental caching reuses buffers keyed by prefix, variant, and shape for
    CUDA graphs. Only the trailing incr_blocks pooling blocks are recomputed;
    completed blocks retain their cached values. Initialization or invalidation
    (a changed first physical block or decreased sequence length) triggers a full
    recomputation to populate the buffer.

    Returns (blocked_k, num_pooling_blocks), as paged_mean_pooling_interface does.
    """
    batch, max_blocks = block_tables_i32.shape
    num_blocks, paged_block_size, head, _ = kv_cache_fp8.shape
    max_num_pooling_blocks = (max_blocks * paged_block_size - 1) // k_block_size + 1

    if cache_prefix is None or not pooling_cache.should_use_cache(
        batch, max_num_pooling_blocks
    ):
        # Use full pooling as the numerical reference when caching is disabled.
        return paged_mean_pooling_interface(
            kv_cache_fp8, context_lens_i32, block_tables_i32,
            k_block_size, dim, variant=variant,
        )

    entry = pooling_cache.get_pooling_cache(
        cache_prefix, variant, max_num_pooling_blocks, dim,
        kv_cache_fp8.device,
    )
    # Allocate the persistent buffer for MAX_BATCH and use its first batch rows.
    # The slice retains a stable base address across CUDA graph batch sizes.
    assert batch <= entry.blocked_k.shape[0], (
        f"runtime batch={batch} larger than ResMISA_POOL_MAX_BATCH={entry.blocked_k.shape[0]}, "
        f"please set the environment variable ResMISA_POOL_MAX_BATCH >= vllm max_num_seqs"
    )
    entry_blocked_k = entry.blocked_k[:batch]

    if pooling_cache.CUDAGRAPH_MODE:
        # Choose by capture/replay state rather than data-dependent Python
        # fallback branches, keeping the captured kernel sequence stable.
        if pooling_cache.is_cudagraph_replaying():
            incr = 1 if variant == "decode" else pooling_cache.INCR_BLOCKS
            start_block = pooling_cache.compute_start_block_pure(
                context_lens_i32, k_block_size, incr_blocks=incr,
            )
            return paged_mean_pooling_incremental_interface(
                kv_cache_fp8, context_lens_i32, block_tables_i32,
                k_block_size, dim, entry_blocked_k, start_block,
                incr_blocks=incr, variant=variant,
            )
        # Eager execution, including pre-capture warmup, fills the persistent buffer
        # so subsequent incremental replay can reuse completed blocks.
        blocked_k_full, num_pool = paged_mean_pooling_interface(
            kv_cache_fp8, context_lens_i32, block_tables_i32,
            k_block_size, dim, variant=variant,
        )
        if blocked_k_full.shape == entry_blocked_k.shape:
            entry_blocked_k.copy_(blocked_k_full)
            return entry_blocked_k, num_pool
        return blocked_k_full, num_pool

    start_block, fallback_full = pooling_cache.compute_start_block_and_invalidate(
        entry, context_lens_i32, block_tables_i32, k_block_size,
    )

    if fallback_full:
        # Initialization or invalidation requires full pooling.
        blocked_k_full, num_pool = paged_mean_pooling_interface(
            kv_cache_fp8, context_lens_i32, block_tables_i32,
            k_block_size, dim, variant=variant,
        )
        # Copy in place when shapes match to preserve the persistent buffer address.
        if blocked_k_full.shape == entry_blocked_k.shape:
            entry_blocked_k.copy_(blocked_k_full)
            return entry_blocked_k, num_pool
        return blocked_k_full, num_pool

    # Update trailing blocks in the persistent buffer in place.
    return paged_mean_pooling_incremental_interface(
        kv_cache_fp8, context_lens_i32, block_tables_i32,
        k_block_size, dim, entry_blocked_k, start_block,
        incr_blocks=pooling_cache.INCR_BLOCKS, variant=variant,
    )


def resmisa_page_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    q_head_topk_indices: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    decode_topk_heads: int,
    use_qnorm: bool,
    k_block_size: int | None = None, 
    cache_prefix: str | None = None,
    resmisa_fuse: bool = True,
    resmisa_minmax: bool = False,
    ):
    '''
    Compute inference logits with paged KV caches and two-stage FP8 selection.
    q_fp8: [B, 1, H, D] FP8.
    kv_cache_fp8: [num_blocks, block_size, 1, D+4], packed FP8 data and FP32 scales.
    weights: [B, H] bfloat16 with q_scale absorbed.
    cache_prefix: Enables per-layer incremental pooling when non-None and
        ResMISA_POOL_CACHE=1.
    resmisa_fuse: True selects heads by relu(-y) and adds the fused-head linear term
        to sparse-head relu(-y) residuals, using a single dot when supported.
        False selects heads by relu(+y) and computes sparse-head logits directly.
    '''
    dim = q_fp8.shape[-1]

    weights = weights.bfloat16()

    context_lens_i32 = context_lens if context_lens.dtype == torch.int32 else context_lens.to(torch.int32)
    block_tables_i32 = block_tables if block_tables.dtype == torch.int32 else block_tables.to(torch.int32)

    if resmisa_fuse and resmisa_minmax:
        q_head_topk_indices = paged_minmax_head_indices(
            q_fp8, kv_cache_fp8, weights, context_lens_i32,
            block_tables_i32, k_block_size, decode_topk_heads,
            cache_prefix=cache_prefix,
        )
    elif not use_qnorm:
        assert k_block_size is not None, "k_block_size must be provided when use_qnorm is False"
        blocked_k, num_pooling_blocks = _pooling_with_cache(
            kv_cache_fp8, context_lens_i32, block_tables_i32, k_block_size, dim,
            variant="decode", cache_prefix=cache_prefix,
        )  # blocked_k: [B, num_pooling_blocks, D] fp32

        q_head_score = batch_block_mqa_attn_return_logits_interface(
            q=q_fp8, blocked_kv=blocked_k, weights=weights,
            context_lens=num_pooling_blocks,
            kv_block_size=k_block_size,
            relu_neg=resmisa_fuse,
        )
        if decode_topk_heads > 0:
            _, q_head_topk_indices = torch.topk(q_head_score, k=decode_topk_heads, dim=-1, largest=True, sorted=False)

    if not resmisa_fuse:
        # Compute sparse-head relu(+y) logits directly, without fused-head decomposition.
        assert decode_topk_heads > 0, "resmisa_fuse=False 需要 decode_topk_heads > 0"
        topk_indices_expanded = (q_head_topk_indices.view(q_fp8.size(0), q_fp8.size(1), -1)
                                    .unsqueeze(-1).expand(-1, -1, -1, q_fp8.size(-1)))
        sparse_head_q = torch.gather(q_fp8, dim=2, index=topk_indices_expanded)
        sparse_head_weights = torch.gather(weights, dim=1, index=q_head_topk_indices)
        logits = paged_mqa_sparse_head_return_logits_triton(
            q=sparse_head_q,
            kv_cache=kv_cache_fp8,
            weights=sparse_head_weights,
            context_lens=context_lens_i32,
            block_tables=block_tables_i32,
            max_model_len=max_model_len,
            dim=dim,
            relu_neg=False,
        )
        return logits.view(-1, logits.shape[-1])

    # ReLU decomposition: all-head fused linear term + sparse-head relu(-y) residual.
    # Linear term: sum_h(w_h * scale * (k @ q_h)) = scale * (k @ q_fused).
    # Reuse the FP8 tensor-core kernel.
    q_fused_fp8, q_scale = _build_fused_head_q(q_fp8, weights)  # [B,S,D] fp8, [B,S] fp32

    if decode_topk_heads > 0:
        topk_indices_expanded = (q_head_topk_indices.view(q_fp8.size(0), q_fp8.size(1), -1)
                                    .unsqueeze(-1).expand(-1, -1, -1, q_fp8.size(-1)))
        sparse_head_q = torch.gather(q_fp8, dim=2, index=topk_indices_expanded)
        sparse_head_weights = torch.gather(weights, dim=1, index=q_head_topk_indices)

        # Compute the linear term and relu(-y) residual in one launch.
        # A single dot combines the fused head (column 0) and sparse heads (1..H)
        # when H+1 is a power of two; otherwise use separate dot products.
        _num_sparse = sparse_head_q.shape[2]
        _hp1 = _num_sparse + 1
        if (_hp1 & (_hp1 - 1)) == 0:
            logits = paged_mqa_fused_merged_8h_v2_logits_triton(
                q_fused=q_fused_fp8,
                q_scale=q_scale,
                q_sparse=sparse_head_q,
                weights=sparse_head_weights,
                kv_cache=kv_cache_fp8,
                context_lens=context_lens_i32,
                block_tables=block_tables_i32,
                max_model_len=max_model_len,
                dim=dim,
            )
        else:
            logits = paged_mqa_fused_merged_logits_triton(
                q_fused=q_fused_fp8,
                q_scale=q_scale,
                q_sparse=sparse_head_q,
                weights=sparse_head_weights,
                kv_cache=kv_cache_fp8,
                context_lens=context_lens_i32,
                block_tables=block_tables_i32,
                max_model_len=max_model_len,
                dim=dim,
            )
        return logits.view(-1, logits.shape[-1])
    else:
        fused_logits = paged_mqa_fused_head_linear_logits_triton(
            q_fused=q_fused_fp8,
            q_scale=q_scale,
            kv_cache=kv_cache_fp8,
            context_lens=context_lens_i32,
            block_tables=block_tables_i32,
            max_model_len=max_model_len,
            dim=dim,
        )
        return fused_logits.view(-1, fused_logits.shape[-1])


def resmisa_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    q_head_topk_indices: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    use_qnorm: bool,
    topk_heads: int,
    k_block_size: int | None = None,
    resmisa_fuse: bool = True,
    resmisa_minmax: bool = False,
) -> torch.Tensor:
    """Compute MQA logits from FP8 queries and keys with optional head selection.

    Args:
        q: [M, H, D], dtype float8_e4m3fn.
        kv: (k_fp8 [N, D] float8_e4m3fn, k_scales [N] or [N,1]).
        weights: [M, H], bfloat16 (q_scale already absorbed).
        resmisa_fuse: True selects heads by relu(-y) and combines the fused-head
            linear term with sparse-head relu(-y) residuals via ReLU decomposition.
            False selects heads by relu(+y) and computes sparse-head logits directly.
    """
    k_fp8, k_scales = kv

    if not k_scales.is_contiguous():
        k_scales = k_scales.contiguous()
    k_scales = k_scales.view(torch.float32)
    if k_scales.ndim == 2:
        k_scales = k_scales.squeeze(-1)  # [N]

    q_fp8 = q # [M, H, D] fp8
    weights = weights.bfloat16()

    cu_seqlen_ks = cu_seqlen_ks if cu_seqlen_ks.dtype == torch.int32 else cu_seqlen_ks.to(torch.int32)
    cu_seqlen_ke = cu_seqlen_ke if cu_seqlen_ke.dtype == torch.int32 else cu_seqlen_ke.to(torch.int32)

    if resmisa_fuse and resmisa_minmax:
        # Attention metadata supplies per-query legal half-open token bounds.
        # No CPU readback: bounds may change on CUDA Graph replay.
        q_head_score = maxmin_head_scores(
            q_fp8, k_fp8, k_scales, weights, cu_seqlen_ks, cu_seqlen_ke,
            k_block_size, score_backend="shared_bf16"
        )
        q_head_topk_indices = deterministic_heads(q_head_score, topk_heads)
    elif use_qnorm:
        if topk_heads < q_head_topk_indices.shape[1]:
            q_head_topk_indices = q_head_topk_indices[:, :topk_heads]
    else:
        assert k_block_size is not None, "k_block_size must be provided when block_topk is set"
        cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
        cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size

        blocked_k_fp32 = block_mean_pooling_interface(
            k_fp8, k_scales, k_block_size,
        )

        # Score with relu(-y) to select heads contributing most to the residual.
        # ResMISA_RELU_NEG=0 uses relu(+y); resmisa_fuse=False always uses relu(+y).
        relu_neg = resmisa_fuse
        q_head_score = block_mqa_attn_return_logits_interface(
            q=q_fp8, 
            blocked_kv=blocked_k_fp32, 
            kv_block_size=k_block_size, 
            weights=weights, 
            cu_seqlen_blocked_ks=cu_seqlen_blocked_ks, 
            cu_seqlen_blocked_ke=cu_seqlen_blocked_ke,
            relu_neg=relu_neg,
        )
        if topk_heads > 0:
            _, q_head_topk_indices = torch.topk(q_head_score, k=topk_heads, dim=-1, largest=True, sorted=False)

    if not resmisa_fuse:
        # Compute sparse-head relu(+y) logits directly, without fused-head decomposition.
        assert topk_heads > 0, "resmisa_fuse=False need topk_heads > 0"
        topk_indices_expanded = q_head_topk_indices.unsqueeze(-1).expand(-1, -1, q_fp8.size(-1))
        sparse_head_q_fp8 = torch.gather(q_fp8, dim=1, index=topk_indices_expanded)
        sparse_head_weights = torch.gather(weights, dim=1, index=q_head_topk_indices)
        logits = resmisa_mqa_sparse_head_return_logits_triton(
            q=sparse_head_q_fp8,
            kv=k_fp8,
            weights=sparse_head_weights,
            k_scales=k_scales,
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
            relu_neg=False,
        )
        return logits

    # ReLU decomposition: logit = linear_term + residual_term.
    # linear_term: scale * (k @ q_fused), using all head weights.
    # residual_term: sum_h(w_h * relu(-y_h)), using sparse top-k heads.
    
    # Build the fused head using all head weights.
    # _build_fused_head_q expects [B, S, H, D]; prefill queries have [M, H, D].
    # unsqueeze(0) yields queries [1, M, H, D] and weights [1, M, H].
    if resmisa_minmax:
        q_fused_fp8, q_scale = build_fused_head_q_exact(q_fp8, weights)
    else:
        # Quantize the weighted head sum with a per-token FP8 scale.
        q_fused_fp8, q_scale = _build_fused_head_q(
            q_fp8.unsqueeze(0), weights.unsqueeze(0)
        )
        q_fused_fp8 = q_fused_fp8.squeeze(0)
        q_scale = q_scale.squeeze(0)
    
    # Gather queries and weights for the selected top-k heads.
    if topk_heads > 0:
        topk_indices_expanded = q_head_topk_indices.unsqueeze(-1).expand(-1, -1, q_fp8.size(-1))
        sparse_head_q_fp8 = torch.gather(q_fp8, dim=1, index=topk_indices_expanded)
        sparse_head_weights = torch.gather(weights, dim=1, index=q_head_topk_indices)

        # Compute the linear term and relu(-y) residual in one launch.
        # A single dot combines the fused head (column 0) and sparse heads (1..H)
        # when H+1 is a power of two; otherwise use separate dot products.
        _num_sparse = sparse_head_q_fp8.shape[1]
        _hp1 = _num_sparse + 1
        if (_hp1 & (_hp1 - 1)) == 0:
            logits = resmisa_mqa_fused_merged_8h_v2_logits_triton(
                q_fused=q_fused_fp8,
                q_scale=q_scale,
                q_sparse=sparse_head_q_fp8,
                sparse_weights=sparse_head_weights,
                kv=k_fp8,
                k_scales=k_scales,
                cu_seqlen_ks=cu_seqlen_ks,
                cu_seqlen_ke=cu_seqlen_ke,
            )
        else:
            logits = resmisa_mqa_fused_merged_logits_triton(
                q_fused=q_fused_fp8,
                q_scale=q_scale,
                q_sparse=sparse_head_q_fp8,
                sparse_weights=sparse_head_weights,
                kv=k_fp8,
                k_scales=k_scales,
                cu_seqlen_ks=cu_seqlen_ks,
                cu_seqlen_ke=cu_seqlen_ke,
            )
        return logits
    else:
        # Without sparse heads, compute only the fused linear term, without ReLU.
        fused_logits = resmisa_mqa_fused_head_linear_logits_triton(
            q_fused=q_fused_fp8,
            q_scale=q_scale,
            kv=k_fp8,
            k_scales=k_scales,
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
        )
    
        return fused_logits