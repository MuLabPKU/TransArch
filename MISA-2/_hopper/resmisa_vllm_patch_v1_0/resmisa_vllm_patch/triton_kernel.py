from __future__ import annotations

import torch
import triton
import triton.language as tl


# =====================================================================
# Kernel 1: sparse-head stage-1  (dense KV window)
# =====================================================================
@triton.jit
def _resmisa_mqa_sparse_head_return_logits_kernel(
    IndexQ,           # [seq_len*HEADS, INDEX_DIM]      fp8
    IndexK,           # [seq_len_kv,    INDEX_DIM]      fp8
    Logits,           # [seq_len,       seq_len_kv]     fp32
    Weights,          # [seq_len,       HEADS]          bf16
    KScales,          # [seq_len_kv]                    fp32
    CuSeqLenS,        # [seq_len]  int32
    CuSeqLenE,        # [seq_len]  int32
    stride_q_row,
    stride_k_row,
    stride_l_row,
    stride_w_row,
    seq_len,
    seq_len_kv,
    HEADS: tl.constexpr,
    INDEX_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NS_INNER: tl.constexpr,
    RELU_NEG: tl.constexpr = False,
):
    pid = tl.program_id(0)
    seq_row_base = pid * BLOCK_Q

    q_offs = seq_row_base + tl.arange(0, BLOCK_Q)
    q_mask = q_offs < seq_len
    cu_s_vec = tl.load(CuSeqLenS + q_offs, mask=q_mask, other=0).to(tl.int32)
    cu_e_vec = tl.load(CuSeqLenE + q_offs, mask=q_mask, other=0).to(tl.int32)

    big = tl.full((BLOCK_Q,), 2147483647, dtype=tl.int32)
    small = tl.full((BLOCK_Q,), -2147483648, dtype=tl.int32)
    cu_s_min = tl.min(tl.where(q_mask, cu_s_vec, big), axis=0)
    cu_e_max = tl.max(tl.where(q_mask, cu_e_vec, small), axis=0)

    cu_seq = cu_e_max - cu_s_min

    qh_cols = tl.arange(0, BLOCK_Q * HEADS)
    qh_offs_row = seq_row_base * HEADS + qh_cols
    qh_mask_col = qh_offs_row < (seq_len * HEADS)
    d_offs = tl.arange(0, INDEX_DIM)
    q_ptrs_T = (IndexQ
                + qh_offs_row[None, :] * stride_q_row
                + d_offs[:, None])
    q_tile_T = tl.load(q_ptrs_T, mask=qh_mask_col[None, :], other=0.0)  # [D, BQ*H] fp8

    # Weights bf16 -> fp32 in registers
    h_offs = tl.arange(0, HEADS)
    w_ptrs = Weights + q_offs[:, None] * stride_w_row + h_offs[None, :]
    w_tile = tl.load(w_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    num_blocks = tl.cdiv(cu_seq, BLOCK_N)
    for b_i in tl.range(0, num_blocks, num_stages=NS_INNER):
        block_s_i = cu_s_min + b_i * BLOCK_N
        n_offs = block_s_i + tl.arange(0, BLOCK_N)
        n_mask = n_offs < seq_len_kv

        k_ptrs = IndexK + n_offs[:, None] * stride_k_row + d_offs[None, :]
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)  # fp8

        k_scale = tl.load(KScales + n_offs, mask=n_mask, other=0.0).to(tl.float32)

        # FP8 tensor-core dot: (BN, D) @ (D, BQ*H) -> (BN, BQ*H)
        s = tl.dot(k_tile, q_tile_T, out_dtype=tl.float32)

        if RELU_NEG:
            s = tl.maximum(-s * k_scale[:, None], 0.0)
        else:
            s = tl.maximum(s * k_scale[:, None], 0.0)
        s = tl.reshape(s, (BLOCK_N, BLOCK_Q, HEADS))
        s = s * w_tile[None, :, :]
        logits_tile = tl.sum(s, axis=2)  # [BN, BQ]

        n_bcast = n_offs[:, None]
        s_bcast = cu_s_vec[None, :]
        e_bcast = cu_e_vec[None, :]
        in_range = ((n_bcast >= s_bcast) & (n_bcast < e_bcast)
                    & q_mask[None, :] & n_mask[:, None])
        logits_tile = tl.where(in_range, logits_tile, float("-inf"))

        # Use int64 indices: seq_len * seq_len_kv may exceed 2^31.
        out_ptrs = (Logits
                    + q_offs[None, :].to(tl.int64) * stride_l_row
                    + n_offs[:, None].to(tl.int64))
        tl.store(out_ptrs, logits_tile,
                 mask=q_mask[None, :] & n_mask[:, None])


def resmisa_mqa_sparse_head_return_logits_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    k_scales: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    relu_neg: bool = False,
) -> torch.Tensor:
    """Use the same inputs and outputs as the tilelang interface."""
    assert q.dtype == torch.float8_e4m3fn
    assert kv.dtype == torch.float8_e4m3fn
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]

    if not q.is_contiguous():
        q = q.contiguous()
    if not kv.is_contiguous():
        kv = kv.contiguous()
    if weights.dtype != torch.bfloat16:
        weights = weights.to(torch.bfloat16)
    if not weights.is_contiguous():
        weights = weights.contiguous()
    if k_scales.dtype != torch.float32:
        k_scales = k_scales.to(torch.float32)
    if not k_scales.is_contiguous():
        k_scales = k_scales.contiguous()
    if cu_seqlen_ks.dtype != torch.int32:
        cu_seqlen_ks = cu_seqlen_ks.to(torch.int32)
    if cu_seqlen_ke.dtype != torch.int32:
        cu_seqlen_ke = cu_seqlen_ke.to(torch.int32)

    q_flat = q.view(seq_len * heads, index_dim)

    logits = torch.full(
        (seq_len, seq_len_kv), float("-inf"),
        dtype=torch.float32, device=q.device,
    )

    # Assert a unit innermost stride for static specialization.
    assert q_flat.stride(1) == 1, f"q innermost stride must be 1, got {q_flat.stride(1)}"
    assert kv.stride(1) == 1,     f"kv innermost stride must be 1, got {kv.stride(1)}"
    assert logits.stride(1) == 1, f"logits innermost stride mustbe 1, got {logits.stride(1)}"
    assert weights.stride(1) == 1

    if heads <= 16:
        if seq_len_kv >= 65536:
            _M_TARGET = 64
            BN, NW, NS, NS_INNER = 256, 4, 4, 3   # M=64 -> BQ=8
        elif seq_len_kv >= 4096:
            _M_TARGET = 64
            BN, NW, NS, NS_INNER = 128, 4, 3, 4   # M=64
        else:
            _M_TARGET = 64
            BN, NW, NS, NS_INNER = 64, 4, 3, 2
    else:
        _M_TARGET = 128
        BN, NW, NS, NS_INNER = 128, 8, 3, 3

    if heads <= 128 and (_M_TARGET % heads == 0):
        BQ = _M_TARGET // heads          # heads=8, M=64 → BQ=8
    else:
        BQ = max(1, _M_TARGET // heads)  # Guard against a zero-sized tile.

    grid = (triton.cdiv(seq_len, BQ),)

    _resmisa_mqa_sparse_head_return_logits_kernel[grid](
        q_flat, kv, logits, weights, k_scales,
        cu_seqlen_ks, cu_seqlen_ke,
        q_flat.stride(0),
        kv.stride(0),
        logits.stride(0),
        weights.stride(0),
        seq_len,
        seq_len_kv,
        HEADS=heads,
        INDEX_DIM=index_dim,
        BLOCK_Q=BQ,
        BLOCK_N=BN,
        NS_INNER=NS_INNER,
        RELU_NEG=relu_neg,
        num_warps=NW,
        num_stages=NS,
    )
    return logits


# =====================================================================
# Kernel 1-fused: prefill fused-head LINEAR term (ReLU identity)
#   logit_linear(m, n) = q_scale[m] * k_scale[n] * Σ_d ( k[n,d] * q_fused[m,d] )
# q_fused[m] = sum_h w_h * q_h[m], weighted in host floating point, then quantized to fp8.
# Variable-length windows (cu_seqlen_ks/ke), one fused head, no ReLU, tensor-core dot.
# =====================================================================
@triton.jit
def _resmisa_mqa_fused_head_linear_logits_kernel(
    IndexQ,           # [seq_len, INDEX_DIM] fp8 (single fused head)
    IndexK,           # [seq_len_kv, INDEX_DIM] fp8
    Logits,           # [seq_len, seq_len_kv]   fp32
    QScale,           # [seq_len]               fp32 (q_fused per-token scale)
    KScales,          # [seq_len_kv]            fp32
    CuSeqLenS,       # [seq_len] int32
    CuSeqLenE,        # [seq_len] int32
    stride_q_row,
    stride_k_row,
    stride_l_row,
    seq_len,
    seq_len_kv,
    INDEX_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NS_INNER: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_row_base = pid * BLOCK_Q

    q_offs = seq_row_base + tl.arange(0, BLOCK_Q)
    q_mask = q_offs < seq_len
    cu_s_vec = tl.load(CuSeqLenS + q_offs, mask=q_mask, other=0).to(tl.int32)
    cu_e_vec = tl.load(CuSeqLenE + q_offs, mask=q_mask, other=0).to(tl.int32)

    big = tl.full((BLOCK_Q,), 2147483647, dtype=tl.int32)
    small = tl.full((BLOCK_Q,), -2147483648, dtype=tl.int32)
    cu_s_min = tl.min(tl.where(q_mask, cu_s_vec, big), axis=0)
    cu_e_max = tl.max(tl.where(q_mask, cu_e_vec, small), axis=0)
    cu_seq = cu_e_max - cu_s_min

    d_offs = tl.arange(0, INDEX_DIM)
    # Fused Q: [D, BQ] (one head, no HEADS dimension)
    q_ptrs_T = (IndexQ
                + q_offs[None, :] * stride_q_row
                + d_offs[:, None])
    q_tile_T = tl.load(q_ptrs_T, mask=q_mask[None, :], other=0.0)  # [D, BQ] fp8

    # q_scale per-token
    q_scale = tl.load(QScale + q_offs, mask=q_mask, other=0.0).to(tl.float32)  # [BQ]

    num_blocks = tl.cdiv(cu_seq, BLOCK_N)
    for b_i in tl.range(0, num_blocks, num_stages=NS_INNER):
        block_s_i = cu_s_min + b_i * BLOCK_N
        n_offs = block_s_i + tl.arange(0, BLOCK_N)
        n_mask = n_offs < seq_len_kv

        k_ptrs = IndexK + n_offs[:, None] * stride_k_row + d_offs[None, :]
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)  # fp8

        k_scale = tl.load(KScales + n_offs, mask=n_mask, other=0.0).to(tl.float32)

        # tensor-core dot: (BN, D) @ (D, BQ) -> (BN, BQ)
        s = tl.dot(k_tile, q_tile_T, out_dtype=tl.float32)
        # Linear term: no ReLU; multiply by k_scale and q_scale.
        logits_tile = s * k_scale[:, None] * q_scale[None, :]  # [BN, BQ]

        n_bcast = n_offs[:, None]
        s_bcast = cu_s_vec[None, :]
        e_bcast = cu_e_vec[None, :]
        in_range = ((n_bcast >= s_bcast) & (n_bcast < e_bcast)
                    & q_mask[None, :] & n_mask[:, None])
        logits_tile = tl.where(in_range, logits_tile, float("-inf"))

        out_ptrs = (Logits
                    + q_offs[None, :].to(tl.int64) * stride_l_row
                    + n_offs[:, None].to(tl.int64))
        tl.store(out_ptrs, logits_tile,
                 mask=q_mask[None, :] & n_mask[:, None])


def resmisa_mqa_fused_head_linear_logits_triton(
    q_fused: torch.Tensor,     # [seq_len, D] fp8 (single fused head)
    q_scale: torch.Tensor,     # [seq_len] fp32 (per-token scale)
    kv: torch.Tensor,          # [seq_len_kv, D] fp8
    k_scales: torch.Tensor,    # [seq_len_kv] fp32
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Prefill fused-head linear term: logit_linear = q_scale * k_scale * (k · q_fused).

    Output shape matches resmisa_mqa_sparse_head_return_logits_triton:
    logits[seq_len, seq_len_kv] fp32 (-inf outside the valid window).
    """
    assert q_fused.dtype == torch.float8_e4m3fn
    assert kv.dtype == torch.float8_e4m3fn
    seq_len, index_dim = q_fused.shape
    seq_len_kv = kv.shape[0]

    if not q_fused.is_contiguous():
        q_fused = q_fused.contiguous()
    if not kv.is_contiguous():
        kv = kv.contiguous()
    if q_scale.dtype != torch.float32:
        q_scale = q_scale.to(torch.float32)
    q_scale = q_scale.contiguous()
    if k_scales.dtype != torch.float32:
        k_scales = k_scales.to(torch.float32)
    k_scales = k_scales.contiguous()
    if cu_seqlen_ks.dtype != torch.int32:
        cu_seqlen_ks = cu_seqlen_ks.to(torch.int32)
    if cu_seqlen_ke.dtype != torch.int32:
        cu_seqlen_ke = cu_seqlen_ke.to(torch.int32)

    logits = torch.full(
        (seq_len, seq_len_kv), float("-inf"),
        dtype=torch.float32, device=q_fused.device,
    )

    assert q_fused.stride(1) == 1
    assert kv.stride(1) == 1
    assert logits.stride(1) == 1

    # With one fused head, use a larger query tile BQ for dot (BN,D)@(D,BQ).
    BQ = 64
    BN, NW, NS, NS_INNER = 128, 4, 3, 3

    grid = (triton.cdiv(seq_len, BQ),)

    _resmisa_mqa_fused_head_linear_logits_kernel[grid](
        q_fused, kv, logits, q_scale, k_scales,
        cu_seqlen_ks, cu_seqlen_ke,
        q_fused.stride(0),
        kv.stride(0),
        logits.stride(0),
        seq_len,
        seq_len_kv,
        INDEX_DIM=index_dim,
        BLOCK_Q=BQ,
        BLOCK_N=BN,
        NS_INNER=NS_INNER,
        num_warps=NW,
        num_stages=NS,
    )
    return logits


# =====================================================================
# Kernel 1-merged: prefill fused-head linear term + sparse-head ReLU residual
#   logit(m, n) = q_scale[m]*k_scale[n]*Σ_d(k[n,d]*q_fused[m,d]) (linear, no ReLU)
#               + Σ_{h∈sparse} w_h[m] * relu(-k_scale[n]*Σ_d(k[n,d]*q_h[m,d])) (residual)
#   Compute and add both terms in one launch, reducing launches and intermediate-logit memory traffic.
# =====================================================================
@triton.jit
def _resmisa_mqa_fused_merged_logits_kernel(
    IndexQFused,      # [seq_len, INDEX_DIM] fp8 (single fused head)
    IndexQSparse,     # [seq_len*HEADS, INDEX_DIM] fp8 (sparse top-k heads)
    IndexK,           # [seq_len_kv,     INDEX_DIM]  fp8
    Logits,           # [seq_len,        seq_len_kv] fp32
    QScale,           # [seq_len]                    fp32 (q_fused per-token scale)
    Weights,          # [seq_len, HEADS] bf16 (sparse-head weights)
    KScales,          # [seq_len_kv]                 fp32
    CuSeqLenS,        # [seq_len]  int32
    CuSeqLenE,        # [seq_len]  int32
    stride_qf_row,
    stride_qs_row,
    stride_k_row,
    stride_l_row,
    stride_w_row,
    seq_len,
    seq_len_kv,
    HEADS: tl.constexpr,
    INDEX_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NS_INNER: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_row_base = pid * BLOCK_Q

    q_offs = seq_row_base + tl.arange(0, BLOCK_Q)
    q_mask = q_offs < seq_len
    cu_s_vec = tl.load(CuSeqLenS + q_offs, mask=q_mask, other=0).to(tl.int32)
    cu_e_vec = tl.load(CuSeqLenE + q_offs, mask=q_mask, other=0).to(tl.int32)

    big = tl.full((BLOCK_Q,), 2147483647, dtype=tl.int32)
    small = tl.full((BLOCK_Q,), -2147483648, dtype=tl.int32)
    cu_s_min = tl.min(tl.where(q_mask, cu_s_vec, big), axis=0)
    cu_e_max = tl.max(tl.where(q_mask, cu_e_vec, small), axis=0)
    cu_seq = cu_e_max - cu_s_min

    d_offs = tl.arange(0, INDEX_DIM)

    # ---- Fused Q: [D, BQ] (single head) ----
    qf_ptrs_T = (IndexQFused
                 + q_offs[None, :] * stride_qf_row
                 + d_offs[:, None])
    qf_tile_T = tl.load(qf_ptrs_T, mask=q_mask[None,:], other=0.0)  # [D, BQ] fp8
    q_scale = tl.load(QScale + q_offs, mask=q_mask, other=0.0).to(tl.float32)  # [BQ]

    # ---- Sparse Q: [D, BQ*H] ----
    qh_cols = tl.arange(0, BLOCK_Q * HEADS)
    qh_offs_row = seq_row_base * HEADS + qh_cols
    qh_mask_col = qh_offs_row < (seq_len * HEADS)
    qs_ptrs_T = (IndexQSparse
                 + qh_offs_row[None, :] * stride_qs_row
                 + d_offs[:, None])
    qs_tile_T = tl.load(qs_ptrs_T, mask=qh_mask_col[None, :], other=0.0)  # [D, BQ*H] fp8

    # Sparse-head weights
    h_offs = tl.arange(0, HEADS)
    w_ptrs = Weights + q_offs[:, None] * stride_w_row + h_offs[None, :]
    w_tile = tl.load(w_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)  # [BQ, H]

    num_blocks = tl.cdiv(cu_seq, BLOCK_N)
    for b_i in tl.range(0, num_blocks, num_stages=NS_INNER):
        block_s_i = cu_s_min + b_i * BLOCK_N
        n_offs = block_s_i + tl.arange(0, BLOCK_N)
        n_mask = n_offs < seq_len_kv

        k_ptrs = IndexK + n_offs[:, None] * stride_k_row + d_offs[None, :]
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)  # [BN, D] fp8
        k_scale = tl.load(KScales + n_offs, mask=n_mask, other=0.0).to(tl.float32)  # [BN]

        # Linear term: (BN, D) @ (D, BQ) -> (BN, BQ)
        s_lin = tl.dot(k_tile, qf_tile_T, out_dtype=tl.float32)
        lin_tile = s_lin * k_scale[:, None] * q_scale[None, :]  # [BN, BQ]

        # Residual term: (BN, D) @ (D, BQ*H) -> (BN, BQ*H)
        s_res = tl.dot(k_tile, qs_tile_T, out_dtype=tl.float32)
        s_res = tl.maximum(-s_res * k_scale[:, None], 0.0)   # relu(-y)
        s_res = tl.reshape(s_res, (BLOCK_N, BLOCK_Q, HEADS))
        s_res = s_res * w_tile[None, :, :]
        res_tile = tl.sum(s_res, axis=2)  # [BN, BQ]

        logits_tile = lin_tile + res_tile

        n_bcast = n_offs[:, None]
        s_bcast = cu_s_vec[None, :]
        e_bcast = cu_e_vec[None, :]
        in_range = ((n_bcast >= s_bcast) & (n_bcast < e_bcast)
                    & q_mask[None, :] & n_mask[:, None])
        logits_tile = tl.where(in_range, logits_tile, float("-inf"))

        out_ptrs = (Logits
                    + q_offs[None, :].to(tl.int64) * stride_l_row
                    + n_offs[:, None].to(tl.int64))
        tl.store(out_ptrs, logits_tile,
                 mask=q_mask[None, :] & n_mask[:, None])


def resmisa_mqa_fused_merged_logits_triton(
    q_fused: torch.Tensor,     # [seq_len, D] fp8 (single fused head)
    q_scale: torch.Tensor,     # [seq_len] fp32 (per-token scale)
    q_sparse: torch.Tensor,    # [seq_len, H, D] fp8 (sparse top-k heads)
    sparse_weights: torch.Tensor,  # [seq_len, H] bf16
    kv: torch.Tensor,          # [seq_len_kv, D] fp8
    k_scales: torch.Tensor,    # [seq_len_kv] fp32
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Combine the prefill fused-head linear term and sparse-head relu(-y) residual in one kernel.

    Numerically equivalent to:
        resmisa_mqa_fused_head_linear_logits_triton(q_fused, ...)
      + resmisa_mqa_sparse_head_return_logits_triton(q_sparse, ..., relu_neg=True)
    Return logits[seq_len, seq_len_kv] fp32 (-inf outside the valid window).
    """
    assert q_fused.dtype == torch.float8_e4m3fn
    assert q_sparse.dtype == torch.float8_e4m3fn
    assert kv.dtype == torch.float8_e4m3fn
    seq_len, index_dim = q_fused.shape
    _seq_len_s, heads, _index_dim_s = q_sparse.shape
    assert _seq_len_s == seq_len and _index_dim_s == index_dim
    seq_len_kv = kv.shape[0]

    if not q_fused.is_contiguous():
        q_fused = q_fused.contiguous()
    if not q_sparse.is_contiguous():
        q_sparse = q_sparse.contiguous()
    if not kv.is_contiguous():
        kv = kv.contiguous()
    if q_scale.dtype != torch.float32:
        q_scale = q_scale.to(torch.float32)
    q_scale = q_scale.contiguous()
    if sparse_weights.dtype != torch.bfloat16:
        sparse_weights = sparse_weights.to(torch.bfloat16)
    sparse_weights = sparse_weights.contiguous()
    if k_scales.dtype != torch.float32:
        k_scales = k_scales.to(torch.float32)
    k_scales = k_scales.contiguous()
    if cu_seqlen_ks.dtype != torch.int32:
        cu_seqlen_ks = cu_seqlen_ks.to(torch.int32)
    if cu_seqlen_ke.dtype != torch.int32:
        cu_seqlen_ke = cu_seqlen_ke.to(torch.int32)

    # Both tl.arange and the head reduction require power-of-two extents.
    # Zero Q and zero weights make padded heads contribute exactly zero.
    padded_heads = triton.next_power_of_2(max(1, heads))
    if heads != padded_heads:
        qs_padded = torch.zeros(
            (seq_len, padded_heads, index_dim),
            dtype=q_sparse.dtype, device=q_sparse.device,
        )
        qs_padded[:, :heads, :] = q_sparse
        weights_padded = torch.zeros(
            (seq_len, padded_heads),
            dtype=sparse_weights.dtype, device=sparse_weights.device,
        )
        weights_padded[:, :heads] = sparse_weights
        q_sparse, sparse_weights = qs_padded, weights_padded
        heads = padded_heads
    q_sparse_flat = q_sparse.view(seq_len * heads, index_dim)

    logits = torch.full(
        (seq_len, seq_len_kv), float("-inf"),
        dtype=torch.float32, device=q_fused.device,
    )

    assert q_fused.stride(1) == 1
    assert q_sparse_flat.stride(1) == 1
    assert kv.stride(1) == 1
    assert logits.stride(1) == 1
    assert sparse_weights.stride(1) == 1

    # BQ must support reshape (BN, BQ, HEADS): BQ*HEADS is the sparse-dot N dimension.
    if heads <= 16:
        _M_TARGET = 64
        BN, NW, NS, NS_INNER = 128, 4, 3, 3
    else:
        _M_TARGET = 128
        BN, NW, NS, NS_INNER = 128, 8, 3, 3

    if heads <= 128 and (_M_TARGET % heads == 0):
        BQ = _M_TARGET // heads
    else:
        BQ = max(1, _M_TARGET // heads)

    grid = (triton.cdiv(seq_len, BQ),)

    _resmisa_mqa_fused_merged_logits_kernel[grid](
        q_fused, q_sparse_flat, kv, logits, q_scale, sparse_weights, k_scales,
        cu_seqlen_ks, cu_seqlen_ke,
        q_fused.stride(0),
        q_sparse_flat.stride(0),
        kv.stride(0),
        logits.stride(0),
        sparse_weights.stride(0),
        seq_len,
        seq_len_kv,
        HEADS=heads,
        INDEX_DIM=index_dim,
        BLOCK_Q=BQ,
        BLOCK_N=BN,
        NS_INNER=NS_INNER,
        num_warps=NW,
        num_stages=NS,
    )
    return logits


# =====================================================================
# Kernel 1-merged-8h: prefill fused and sparse heads in one dot (8-head parallelism)
#   Concatenate fused head 0 and top-H sparse heads into H+1 heads for one tensor-core dot:
#     s_all = dot(k, q_all_T) -> [BN, BQ*(H+1)] (H+1 heads in parallel, no second serial dot)
#   Apply a head-dependent activation after the dot:
#     head 0 (fused, linear): out = s0 * k_scale * q_scale (no ReLU)
#     head j>=1 (sparse, residual): out = relu(-s_j * k_scale) * w_j (relu(-y))
#   Sum over H+1 heads for the final logit, numerically equivalent to the two-dot formulation.
# =====================================================================
@triton.jit
def _resmisa_mqa_fused_merged_8h_logits_kernel(
    IndexQAll,        # [seq_len*(HEADS+1), INDEX_DIM] fp8 (head0=fused, 1..H=sparse)
    IndexK,           # [seq_len_kv,     INDEX_DIM]  fp8
    Logits,           # [seq_len,        seq_len_kv] fp32
    Coef,             # [seq_len, HEADS+1] fp32 (head0=q_scale, 1..H=w_h)
    KScales,          # [seq_len_kv]                 fp32
    CuSeqLenS,        # [seq_len]  int32
    CuSeqLenE,        # [seq_len]  int32
    stride_qa_row,
    stride_k_row,
    stride_l_row,
    stride_c_row,
    seq_len,
    seq_len_kv,
    HP1: tl.constexpr,      # HEADS+1 padded to a power of two
    INDEX_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,  # Power of two
    BLOCK_N: tl.constexpr,
    NS_INNER: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_row_base = pid * BLOCK_Q

    q_offs = seq_row_base + tl.arange(0, BLOCK_Q)
    q_mask = q_offs < seq_len
    cu_s_vec = tl.load(CuSeqLenS + q_offs, mask=q_mask, other=0).to(tl.int32)
    cu_e_vec = tl.load(CuSeqLenE + q_offs, mask=q_mask, other=0).to(tl.int32)

    big = tl.full((BLOCK_Q,), 2147483647, dtype=tl.int32)
    small = tl.full((BLOCK_Q,), -2147483648, dtype=tl.int32)
    cu_s_min = tl.min(tl.where(q_mask, cu_s_vec, big), axis=0)
    cu_e_max = tl.max(tl.where(q_mask, cu_e_vec, small), axis=0)
    cu_seq = cu_e_max - cu_s_min

    d_offs = tl.arange(0, INDEX_DIM)

    # ---- Concatenated Q: [D, BQ*(H+1)], column c -> token=c//HP1, head=c%HP1 ----
    qa_cols = tl.arange(0, BLOCK_Q * HP1)
    qa_offs_row = seq_row_base * HP1 + qa_cols
    qa_mask_col = qa_offs_row < (seq_len * HP1)
    qa_ptrs_T = (IndexQAll
                 + qa_offs_row[None, :] * stride_qa_row
                 + d_offs[:, None])
    qa_tile_T = tl.load(qa_ptrs_T, mask=qa_mask_col[None, :], other=0.0)  # [D, BQ*(H+1)]

    # ---- Coefficients [BQ, H+1]: head0=q_scale, head>=1=w_h ----
    hp1_offs = tl.arange(0, HP1)
    c_ptrs = Coef + q_offs[:, None] * stride_c_row + hp1_offs[None, :]
    c_tile = tl.load(c_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)  # [BQ, H+1]
    # Head flag: 0=fused (linear), 1=sparse (relu(-y))
    is_sparse = (hp1_offs > 0)  # [H+1] bool

    num_blocks = tl.cdiv(cu_seq, BLOCK_N)
    for b_i in tl.range(0, num_blocks, num_stages=NS_INNER):
        block_s_i = cu_s_min + b_i * BLOCK_N
        n_offs = block_s_i + tl.arange(0, BLOCK_N)
        n_mask = n_offs < seq_len_kv

        k_ptrs = IndexK + n_offs[:, None] * stride_k_row + d_offs[None, :]
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)  # [BN, D] fp8
        k_scale = tl.load(KScales + n_offs, mask=n_mask, other=0.0).to(tl.float32)  # [BN]

        # Single dot: (BN, D) @ (D, BQ*(H+1)) -> (BN, BQ*(H+1))
        s_all = tl.dot(k_tile, qa_tile_T, out_dtype=tl.float32)
        s_all = tl.reshape(s_all, (BLOCK_N, BLOCK_Q, HP1))
        y = s_all * k_scale[:, None, None]  # Multiply by k_scale; [BN, BQ, H+1]

        # Linear term (head0): y * q_scale; residual (head>=1): relu(-y)
        lin = y
        res = tl.maximum(-y, 0.0)
        contrib = tl.where(is_sparse[None, None, :], res, lin)
        contrib = contrib * c_tile[None, :, :]  # Multiply by coefficients (q_scale / w_h)
        logits_tile = tl.sum(contrib, axis=2)   # [BN, BQ]

        n_bcast = n_offs[:, None]
        s_bcast = cu_s_vec[None, :]
        e_bcast = cu_e_vec[None, :]
        in_range = ((n_bcast >= s_bcast) & (n_bcast < e_bcast)
                    & q_mask[None, :] & n_mask[:, None])
        logits_tile = tl.where(in_range, logits_tile, float("-inf"))

        out_ptrs = (Logits
                    + q_offs[None, :].to(tl.int64) * stride_l_row
                    + n_offs[:, None].to(tl.int64))
        tl.store(out_ptrs, logits_tile,
                 mask=q_mask[None, :] & n_mask[:, None])


# =====================================================================
# Prefill single-dot kernel: read q_fused and q_sparse through separate pointers.
#   Avoid host concatenation: no torch.empty or three copies for q_fused/q_sparse/coef.
#     - head 0 (fused, linear): read IndexQFused at row=token, coefficient=q_scale
#     - head j>=1 (sparse, residual): read IndexQSparse at row=token*NSPARSE+(j-1), coefficient=w_j
#   Two loads and tl.where assemble Q [D, BQ*HP1] by head inside the kernel before one dot.
#   Requires HP1 = NSPARSE + 1 to be a power of two (HP1=8 needs no padding).
# =====================================================================
@triton.jit
def _resmisa_mqa_fused_merged_8h_v2_kernel(
    IndexQFused,      # [seq_len, INDEX_DIM] fp8 (fused head)
    IndexQSparse,     # [seq_len*NSPARSE, INDEX_DIM] fp8 (flattened sparse heads)
    IndexK,           # [seq_len_kv,       INDEX_DIM] fp8
    Logits,           # [seq_len,          seq_len_kv] fp32
    QScale,           # [seq_len] fp32 (fused-head per-token scale)
    SparseW,          # [seq_len, NSPARSE] fp32 (sparse-head weights)
    KScales,          # [seq_len_kv]       fp32
    CuSeqLenS,        # [seq_len] int32
    CuSeqLenE,        # [seq_len] int32
    stride_qf_row,
    stride_qs_row,
    stride_k_row,
    stride_l_row,
    stride_sw_row,
    seq_len,
    seq_len_kv,
    HP1: tl.constexpr,       # NSPARSE + 1, a power of two
    NSPARSE: tl.constexpr,   # Number of sparse heads (HP1 - 1)
    INDEX_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,   # Power of two
    BLOCK_N: tl.constexpr,
    NS_INNER: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_row_base = pid * BLOCK_Q

    q_offs = seq_row_base + tl.arange(0, BLOCK_Q)
    q_mask = q_offs < seq_len
    cu_s_vec = tl.load(CuSeqLenS + q_offs, mask=q_mask, other=0).to(tl.int32)
    cu_e_vec = tl.load(CuSeqLenE + q_offs, mask=q_mask, other=0).to(tl.int32)

    big = tl.full((BLOCK_Q,), 2147483647, dtype=tl.int32)
    small = tl.full((BLOCK_Q,), -2147483648, dtype=tl.int32)
    cu_s_min = tl.min(tl.where(q_mask, cu_s_vec, big), axis=0)
    cu_e_max = tl.max(tl.where(q_mask, cu_e_vec, small), axis=0)
    cu_seq = cu_e_max - cu_s_min

    d_offs = tl.arange(0, INDEX_DIM)

    # ---- Assemble Q [D, BQ*HP1] without concatenation: column c -> token=c//HP1, head=c%HP1 ----
    qa_cols = tl.arange(0, BLOCK_Q * HP1)
    col_token = seq_row_base + (qa_cols // HP1)   # Global token index
    col_head = qa_cols % HP1                       # 0=fused head, 1..NSPARSE=sparse heads
    col_valid = col_token < seq_len
    is_fused_col = col_head == 0

    # Fused-head pointer: row = token
    qf_row = col_token
    qf_ptrs_T = (IndexQFused
                 + qf_row[None, :] * stride_qf_row
                 + d_offs[:, None])
    qf_tile_T = tl.load(qf_ptrs_T,
                        mask=(col_valid & is_fused_col)[None, :], other=0.0)  # [D, BQ*HP1]

    # Sparse-head pointer: row = token*NSPARSE + (head-1)
    qs_row = col_token * NSPARSE + (col_head - 1)
    qs_valid = col_valid & (~is_fused_col)
    qs_ptrs_T = (IndexQSparse
                 + qs_row[None, :] * stride_qs_row
                 + d_offs[:, None])
    qs_tile_T = tl.load(qs_ptrs_T,
                        mask=qs_valid[None, :], other=0.0)  # [D, BQ*HP1]

    # Select by head: head0 uses fused Q; head>=1 uses sparse Q.
    qa_tile_T = tl.where(is_fused_col[None, :], qf_tile_T, qs_tile_T)

    # ---- Coefficients [BQ, HP1]: head0=q_scale, head>=1=w_h (two loads + where, no concatenation) ----
    hp1_offs = tl.arange(0, HP1)
    is_sparse = hp1_offs > 0  # [HP1] bool

    q_scale_vec = tl.load(QScale + q_offs, mask=q_mask, other=0.0).to(tl.float32)  # [BQ]
    # Sparse weights [BQ, HP1]: column j>=1 reads SparseW[:, j-1]; column 0 is masked to zero.
    sw_col = hp1_offs - 1  # Column 0 -> -1 (invalid)
    sw_ptrs = (SparseW
               + q_offs[:, None] * stride_sw_row
               + sw_col[None, :])
    sw_mask = q_mask[:, None] & is_sparse[None, :]
    sw_tile = tl.load(sw_ptrs, mask=sw_mask, other=0.0).to(tl.float32)  # [BQ, HP1]
    # Fill head0 with q_scale and head>=1 with sparse weights.
    c_tile = tl.where(is_sparse[None, :], sw_tile, q_scale_vec[:, None])  # [BQ, HP1]

    num_blocks = tl.cdiv(cu_seq, BLOCK_N)
    for b_i in tl.range(0, num_blocks, num_stages=NS_INNER):
        block_s_i = cu_s_min + b_i * BLOCK_N
        n_offs = block_s_i + tl.arange(0, BLOCK_N)
        n_mask = n_offs < seq_len_kv

        k_ptrs = IndexK + n_offs[:, None] * stride_k_row + d_offs[None, :]
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)  # [BN, D] fp8
        k_scale = tl.load(KScales + n_offs, mask=n_mask, other=0.0).to(tl.float32)  # [BN]

        s_all = tl.dot(k_tile, qa_tile_T, out_dtype=tl.float32)  # [BN, BQ*HP1]
        s_all = tl.reshape(s_all, (BLOCK_N, BLOCK_Q, HP1))
        y = s_all * k_scale[:, None, None]

        lin = y
        res = tl.maximum(-y, 0.0)
        contrib = tl.where(is_sparse[None, None, :], res, lin)
        contrib = contrib * c_tile[None, :, :]
        logits_tile = tl.sum(contrib, axis=2)  # [BN, BQ]

        n_bcast = n_offs[:, None]
        s_bcast = cu_s_vec[None, :]
        e_bcast = cu_e_vec[None, :]
        in_range = ((n_bcast >= s_bcast) & (n_bcast < e_bcast)
                    & q_mask[None, :] & n_mask[:, None])
        logits_tile = tl.where(in_range, logits_tile, float("-inf"))

        out_ptrs = (Logits
                    + q_offs[None, :].to(tl.int64) * stride_l_row
                    + n_offs[:, None].to(tl.int64))
        tl.store(out_ptrs, logits_tile,
                 mask=q_mask[None, :] & n_mask[:, None])


def resmisa_mqa_fused_merged_8h_logits_triton(
    q_fused: torch.Tensor,     # [seq_len, D] fp8 (single fused head)
    q_scale: torch.Tensor,     # [seq_len] fp32 (per-token scale)
    q_sparse: torch.Tensor,    # [seq_len, H, D] fp8 (sparse top-k heads)
    sparse_weights: torch.Tensor,  # [seq_len, H] bf16/fp32
    kv: torch.Tensor,          # [seq_len_kv, D] fp8
    k_scales: torch.Tensor,    # [seq_len_kv] fp32
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Combine prefill fused and sparse heads in one dot (8-head parallelism).

    Numerically equivalent to resmisa_mqa_fused_merged_logits_triton; both terms
    use H+1 parallel heads in one dot, eliminating the second serial dot.
    """
    assert q_fused.dtype == torch.float8_e4m3fn
    assert q_sparse.dtype == torch.float8_e4m3fn
    assert kv.dtype == torch.float8_e4m3fn
    seq_len, index_dim = q_fused.shape
    _seq_len_s, n_sparse, _index_dim_s = q_sparse.shape
    assert _seq_len_s == seq_len and _index_dim_s == index_dim
    seq_len_kv = kv.shape[0]
    # Total heads = one fused head + n_sparse sparse heads; choose n_sparse = total_heads-1
    # so hp1 = n_sparse+1 is a power of two (e.g. 8=1 fused+7 sparse), without padding.
    hp1 = n_sparse + 1
    # Pad non-power-of-two hp1 for arange/reshape; zero coefficients make extra columns contribute zero.
    hp1_pad = 1 << (hp1 - 1).bit_length()

    # ---- Host concatenation q_all[seq_len, HP1_PAD, D]: head0=fused, 1..n_sparse=sparse, remaining=0 ----
    if hp1_pad == hp1:
        q_all = torch.empty(seq_len, hp1_pad, index_dim, dtype=q_fused.dtype, device=q_fused.device)
    else:
        q_all = torch.zeros(seq_len, hp1_pad, index_dim, dtype=q_fused.dtype, device=q_fused.device)
    q_all[:, 0, :] = q_fused
    q_all[:, 1:hp1, :] = q_sparse
    q_all_flat = q_all.view(seq_len * hp1_pad, index_dim)

    # ---- Coefficients coef[seq_len, HP1_PAD]: head0=q_scale, 1..n_sparse=w_h, remaining=0 ----
    if q_scale.dtype != torch.float32:
        q_scale = q_scale.to(torch.float32)
    if hp1_pad == hp1:
        coef = torch.empty(seq_len, hp1_pad, dtype=torch.float32, device=q_fused.device)
    else:
        coef = torch.zeros(seq_len, hp1_pad, dtype=torch.float32, device=q_fused.device)
    coef[:, 0] = q_scale.reshape(seq_len)
    coef[:, 1:hp1] = sparse_weights.to(torch.float32)
    coef = coef.contiguous()

    if not kv.is_contiguous():
        kv = kv.contiguous()
    if k_scales.dtype != torch.float32:
        k_scales = k_scales.to(torch.float32)
    k_scales = k_scales.contiguous()
    if cu_seqlen_ks.dtype != torch.int32:
        cu_seqlen_ks = cu_seqlen_ks.to(torch.int32)
    if cu_seqlen_ke.dtype != torch.int32:
        cu_seqlen_ke = cu_seqlen_ke.to(torch.int32)

    logits = torch.full(
        (seq_len, seq_len_kv), float("-inf"),
        dtype=torch.float32, device=q_fused.device,
    )

    assert q_all_flat.stride(1) == 1
    assert kv.stride(1) == 1
    assert logits.stride(1) == 1
    assert coef.stride(1) == 1

    if hp1 <= 16:
        _M_TARGET = 64
        BN, NW, NS, NS_INNER = 128, 4, 3, 3
    else:
        _M_TARGET = 128
        BN, NW, NS, NS_INNER = 128, 8, 3, 3

    if hp1_pad <= 128 and (_M_TARGET % hp1_pad == 0):
        BQ = _M_TARGET // hp1_pad
    else:
        BQ = max(1, _M_TARGET // hp1_pad)
    # BLOCK_Q must be a power of two for tl.arange in the kernel.
    if BQ < 1:
        BQ = 1
    BQ = 1 << (BQ.bit_length() - 1)  # Round down to a power of two

    grid = (triton.cdiv(seq_len, BQ),)

    _resmisa_mqa_fused_merged_8h_logits_kernel[grid](
        q_all_flat, kv, logits, coef, k_scales,
        cu_seqlen_ks, cu_seqlen_ke,
        q_all_flat.stride(0),
        kv.stride(0),
        logits.stride(0),
        coef.stride(0),
        seq_len,
        seq_len_kv,
        HP1=hp1_pad,
        INDEX_DIM=index_dim,
        BLOCK_Q=BQ,
        BLOCK_N=BN,
        NS_INNER=NS_INNER,
        num_warps=NW,
        num_stages=NS,
    )
    return logits


def prep_8h_buffers(q_fused, q_scale, q_sparse, sparse_weights):
    """Prebuild q_all_flat and coef for the 8h kernel by host-side concatenation."""
    seq_len, index_dim = q_fused.shape
    _sl, n_sparse, _id = q_sparse.shape
    hp1 = n_sparse + 1
    hp1_pad = 1 << (hp1 - 1).bit_length()
    if hp1_pad == hp1:
        q_all = torch.empty(seq_len, hp1_pad, index_dim, dtype=q_fused.dtype, device=q_fused.device)
    else:
        q_all = torch.zeros(seq_len, hp1_pad, index_dim, dtype=q_fused.dtype, device=q_fused.device)
    q_all[:, 0, :] = q_fused
    q_all[:, 1:hp1, :] = q_sparse
    q_all_flat = q_all.view(seq_len * hp1_pad, index_dim)
    if q_scale.dtype != torch.float32:
        q_scale = q_scale.to(torch.float32)
    if hp1_pad == hp1:
        coef = torch.empty(seq_len, hp1_pad, dtype=torch.float32, device=q_fused.device)
    else:
        coef = torch.zeros(seq_len, hp1_pad, dtype=torch.float32, device=q_fused.device)
    coef[:, 0] = q_scale.reshape(seq_len)
    coef[:, 1:hp1] = sparse_weights.to(torch.float32)
    coef = coef.contiguous()
    return q_all_flat, coef, hp1, hp1_pad, index_dim


def launch_8h_kernel(q_all_flat, coef, hp1, hp1_pad, index_dim, kv, k_scales,
                     cu_seqlen_ks, cu_seqlen_ke, seq_len, seq_len_kv, logits=None):
    """Launch the 8h kernel with prebuilt buffers, without host-side concatenation."""
    if logits is None:
        logits = torch.full((seq_len, seq_len_kv), float("-inf"),
                            dtype=torch.float32, device=kv.device)
    if hp1 <= 16:
        _M_TARGET = 64
        BN, NW, NS, NS_INNER = 128, 4, 3, 3
    else:
        _M_TARGET = 128
        BN, NW, NS, NS_INNER = 128, 8, 3, 3
    BQ = max(1, _M_TARGET // hp1_pad)
    BQ = 1 << (BQ.bit_length() - 1)
    grid = (triton.cdiv(seq_len, BQ),)
    _resmisa_mqa_fused_merged_8h_logits_kernel[grid](
        q_all_flat, kv, logits, coef, k_scales,
        cu_seqlen_ks, cu_seqlen_ke,
        q_all_flat.stride(0), kv.stride(0), logits.stride(0), coef.stride(0),
        seq_len, seq_len_kv,
        HP1=hp1_pad, INDEX_DIM=index_dim, BLOCK_Q=BQ, BLOCK_N=BN,
        NS_INNER=NS_INNER, num_warps=NW, num_stages=NS,
    )
    return logits


def resmisa_mqa_fused_merged_8h_v2_logits_triton(
    q_fused: torch.Tensor,     # [seq_len, D] fp8 (single fused head)
    q_scale: torch.Tensor,     # [seq_len] fp32 (per-token scale)
    q_sparse: torch.Tensor,    # [seq_len, H, D] fp8 (sparse top-k heads)
    sparse_weights: torch.Tensor,  # [seq_len, H] bf16/fp32
    kv: torch.Tensor,          # [seq_len_kv, D] fp8
    k_scales: torch.Tensor,    # [seq_len_kv] fp32
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    logits: torch.Tensor = None,  # Optional reusable buffer; must already be filled with -inf
) -> torch.Tensor:
    """Prefill 8-head single dot: read q_fused and q_sparse directly through two pointers.

    Assemble Q inside the kernel without host-side concatenation, avoiding
    torch.empty and three copies for q_fused/q_sparse/coef.
    Requires HP1 = H + 1 to be a power of two (H=7 gives HP1=8).
    Reuse a logits buffer prefilled with -inf to avoid a torch.full allocation.
    """
    assert q_fused.dtype == torch.float8_e4m3fn
    assert q_sparse.dtype == torch.float8_e4m3fn
    assert kv.dtype == torch.float8_e4m3fn
    seq_len, index_dim = q_fused.shape
    _sl, n_sparse, _id = q_sparse.shape
    assert _sl == seq_len and _id == index_dim
    seq_len_kv = kv.shape[0]

    hp1 = n_sparse + 1
    assert (hp1 & (hp1 - 1)) == 0, f"HP1={hp1} 必须是 2 的幂 (稀疏头数 H={n_sparse} 应为 2^k-1)"

    if not q_fused.is_contiguous():
        q_fused = q_fused.contiguous()
    if not q_sparse.is_contiguous():
        q_sparse = q_sparse.contiguous()
    if not kv.is_contiguous():
        kv = kv.contiguous()
    if q_scale.dtype != torch.float32:
        q_scale = q_scale.to(torch.float32)
    q_scale = q_scale.contiguous()
    if sparse_weights.dtype != torch.float32:
        sparse_weights = sparse_weights.to(torch.float32)
    sparse_weights = sparse_weights.contiguous()
    if k_scales.dtype != torch.float32:
        k_scales = k_scales.to(torch.float32)
    k_scales = k_scales.contiguous()
    if cu_seqlen_ks.dtype != torch.int32:
        cu_seqlen_ks = cu_seqlen_ks.to(torch.int32)
    if cu_seqlen_ke.dtype != torch.int32:
        cu_seqlen_ke = cu_seqlen_ke.to(torch.int32)

    q_sparse_flat = q_sparse.view(seq_len * n_sparse, index_dim)

    if logits is None:
        logits = torch.full(
            (seq_len, seq_len_kv), float("-inf"),
            dtype=torch.float32, device=q_fused.device,
        )

    assert q_fused.stride(1) == 1
    assert q_sparse_flat.stride(1) == 1
    assert kv.stride(1) == 1
    assert logits.stride(1) == 1
    assert sparse_weights.stride(1) == 1

    if hp1 <= 16:
        _M_TARGET = 64
        BN, NW, NS, NS_INNER = 128, 4, 3, 3
    else:
        _M_TARGET = 128
        BN, NW, NS, NS_INNER = 128, 8, 3, 3
    BQ = max(1, _M_TARGET // hp1)
    BQ = 1 << (BQ.bit_length() - 1)
    grid = (triton.cdiv(seq_len, BQ),)

    _resmisa_mqa_fused_merged_8h_v2_kernel[grid](
        q_fused, q_sparse_flat, kv, logits,
        q_scale, sparse_weights, k_scales,
        cu_seqlen_ks, cu_seqlen_ke,
        q_fused.stride(0),
        q_sparse_flat.stride(0),
        kv.stride(0),
        logits.stride(0),
        sparse_weights.stride(0),
        seq_len,
        seq_len_kv,
        HP1=hp1,
        NSPARSE=n_sparse,
        INDEX_DIM=index_dim,
        BLOCK_Q=BQ,
        BLOCK_N=BN,
        NS_INNER=NS_INNER,
        num_warps=NW,
        num_stages=NS,
    )
    return logits


# =====================================================================
# Kernel 2: stage-2 attn (sparse index gather) — PERSISTENT KERNEL
# =====================================================================
@triton.jit
def _resmisa_mqa_attn_return_logits_persistent_kernel(
    IndexQ,             # [seq_len*HEADS, INDEX_DIM]  fp8
    IndexK,             # [seq_len_kv,    INDEX_DIM]  fp8
    Logits,             # [seq_len,       stage_1_topk] fp32
    Weights,            # [seq_len,       HEADS]      fp32
    KScales,            # [seq_len_kv]                fp32
    SparseTopKIndices,  # [seq_len,       stage_1_topk] int32
    CuSeqLenS,
    CuSeqLenE,
    stride_q_row,
    stride_k_row,
    stride_l_row,
    stride_w_row,
    stride_idx_row,
    seq_len,
    seq_len_kv,
    stage_1_topk,
    HEADS: tl.constexpr,
    INDEX_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    NS_INNER: tl.constexpr,
):
    """Persistent kernel: each CTA handles GROUP_M query rows."""
    pid = tl.program_id(0)
    row_base = pid * GROUP_M

    d_offs = tl.arange(0, INDEX_DIM)
    h_offs = tl.arange(0, HEADS)

    # Statically unroll GROUP_M rows; GROUP_M is constexpr.
    for gm in tl.static_range(0, GROUP_M):
        row_id = row_base + gm
        row_valid = row_id < seq_len

        # Skip invalid rows in the statically unrolled loop without affecting the pipeline.
        if row_valid:
            cu_s = tl.load(CuSeqLenS + row_id).to(tl.int32)
            cu_e = tl.load(CuSeqLenE + row_id).to(tl.int32)

            q_row_offs = row_id * HEADS + h_offs
            # Load directly in transposed [INDEX_DIM, HEADS] layout to avoid tl.trans.
            q_ptrs_T = (IndexQ
                        + q_row_offs[None, :] * stride_q_row
                        + d_offs[:, None])
            q_tile_T = tl.load(q_ptrs_T)  # fp8, [D, H]

            w_ptrs = Weights + row_id * stride_w_row + h_offs
            w_vec = tl.load(w_ptrs).to(tl.float32)  # [H]

            num_blocks = tl.cdiv(stage_1_topk, BLOCK_N)
            for b_i in tl.range(0, num_blocks, num_stages=NS_INNER):
                block_s_i = b_i * BLOCK_N
                k_col_offs = block_s_i + tl.arange(0, BLOCK_N)
                col_mask = k_col_offs < stage_1_topk

                idx_ptrs = (SparseTopKIndices
                            + row_id * stride_idx_row
                            + k_col_offs)
                idx = tl.load(idx_ptrs, mask=col_mask, other=0).to(tl.int32)
                idx_valid = (idx >= 0) & (idx < seq_len_kv) & col_mask
                idx_safe = tl.where(idx_valid, idx, 0)

                k_ptrs = IndexK + idx_safe[:, None] * stride_k_row + d_offs[None, :]
                k_tile = tl.load(k_ptrs, mask=idx_valid[:, None], other=0.0)  # fp8

                k_scale = tl.load(KScales + idx_safe,
                                  mask=idx_valid, other=0.0).to(tl.float32)

                # FP8 tensor-core dot: (BN, D) @ (D, H) -> (BN, H)
                s = tl.dot(k_tile, q_tile_T, out_dtype=tl.float32)

                s = tl.maximum(s * k_scale[:, None], 0.0) * w_vec[None, :]
                logits_tile = tl.sum(s, axis=1)

                in_range = (idx >= cu_s) & (idx < cu_e) & idx_valid
                logits_tile = tl.where(in_range, logits_tile, float("-inf"))

                out_ptrs = (Logits
                            + tl.cast(row_id, tl.int64) * stride_l_row
                            + k_col_offs.to(tl.int64))
                tl.store(out_ptrs, logits_tile, mask=col_mask)


def resmisa_mqa_attn_return_logits_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    k_scales: torch.Tensor,
    sparse_topk_indices: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    assert q.dtype == torch.float8_e4m3fn
    assert kv.dtype == torch.float8_e4m3fn
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]
    stage_1_topk = sparse_topk_indices.shape[-1]

    if not q.is_contiguous():
        q = q.contiguous()
    if not kv.is_contiguous():
        kv = kv.contiguous()
    if weights.dtype != torch.float32:
        weights = weights.to(torch.float32)
    if not weights.is_contiguous():
        weights = weights.contiguous()
    if k_scales.dtype != torch.float32:
        k_scales = k_scales.to(torch.float32)
    if not k_scales.is_contiguous():
        k_scales = k_scales.contiguous()
    if sparse_topk_indices.dtype != torch.int32:
        sparse_topk_indices = sparse_topk_indices.to(torch.int32)
    if not sparse_topk_indices.is_contiguous():
        sparse_topk_indices = sparse_topk_indices.contiguous()
    if cu_seqlen_ks.dtype != torch.int32:
        cu_seqlen_ks = cu_seqlen_ks.to(torch.int32)
    if cu_seqlen_ke.dtype != torch.int32:
        cu_seqlen_ke = cu_seqlen_ke.to(torch.int32)

    q_flat = q.view(seq_len * heads, index_dim)
    logits = torch.empty(
        (seq_len, stage_1_topk),
        dtype=torch.float32, device=q.device,
    )

    assert q_flat.stride(1) == 1
    assert kv.stride(1) == 1
    assert logits.stride(1) == 1
    assert weights.stride(1) == 1
    assert sparse_topk_indices.stride(1) == 1

    if heads <= 16:
        BN, GM, NW, NS, NS_INNER = 128, 4, 4, 3, 3
    else:
        if stage_1_topk >= 4096:
            BN, GM, NW, NS, NS_INNER = 128, 1, 4, 3, 3
        elif stage_1_topk >= 256:
            BN, GM, NW, NS, NS_INNER = 128, 1, 4, 3, 3
        else:
            BN, GM, NW, NS, NS_INNER = 64,  1, 4, 3, 2

    grid = (triton.cdiv(seq_len, GM),)

    _resmisa_mqa_attn_return_logits_persistent_kernel[grid](
        q_flat, kv, logits, weights, k_scales,
        sparse_topk_indices, cu_seqlen_ks, cu_seqlen_ke,
        q_flat.stride(0),
        kv.stride(0),
        logits.stride(0),
        weights.stride(0),
        sparse_topk_indices.stride(0),
        seq_len,
        seq_len_kv,
        stage_1_topk,
        HEADS=heads,
        INDEX_DIM=index_dim,
        BLOCK_N=BN,
        GROUP_M=GM,
        NS_INNER=NS_INNER,
        num_warps=NW,
        num_stages=NS,
    )
    return logits


# =====================================================================
# Kernel 3: paged sparse-head stage-1 (DECODE path)
# =====================================================================
@triton.jit
def _paged_mqa_sparse_head_return_logits_kernel(
    IndexQ,           # [batch, seq_len, HEADS, INDEX_DIM]  fp8
    KvCache,          # [num_phys_blocks, BLOCK_N, 1, D+4]  uint8
    KvCacheS4,        # Int32 view of the same buffer for reading the scale segment
    Logits,           # [batch, seq_len, max_model_len]     fp32
    Weights,          # [batch, seq_len, HEADS]             fp32
    ContextLens,      # [batch]                             int32
    BlockTables,      # [batch, max_blocks]                 int32
    stride_q_b, stride_q_s, stride_q_h,
    stride_kv_blk, stride_kv_tok,
    stride_kv_s4,
    stride_l_b, stride_l_s,
    stride_w_b, stride_w_s,
    stride_bt_b,
    max_blocks,
    max_model_len,
    SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    INDEX_DIM: tl.constexpr,
    D_PLUS_4: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLKS_PER_CTA: tl.constexpr,
    NS_INNER: tl.constexpr,
    RELU_NEG: tl.constexpr,
):
    # grid = (batch * seq_len, num_block_tiles)
    bs = tl.program_id(0)
    b = bs // SEQ_LEN
    s = bs % SEQ_LEN
    tile = tl.program_id(1)

    context_len = tl.load(ContextLens + b).to(tl.int32)
    num_blocks = tl.cdiv(context_len, BLOCK_N)

    blk_start = tile * BLKS_PER_CTA
    # Return if this CTA has no KV blocks; logits are prefilled with -inf.
    if blk_start >= num_blocks:
        return

    d_offs = tl.arange(0, INDEX_DIM)
    h_offs = tl.arange(0, HEADS)

    # Load Q in transposed layout: q_T[d, h] = IndexQ[b, s, h, d]
    q_ptrs_T = (IndexQ
                + b * stride_q_b
                + s * stride_q_s
                + h_offs[None, :] * stride_q_h
                + d_offs[:, None])
    q_tile_T = tl.load(q_ptrs_T)  # fp8, [D, H]

    # weights [H] fp32
    w_ptrs = Weights + b * stride_w_b + s * stride_w_s + h_offs
    w_vec = tl.load(w_ptrs).to(tl.float32)  # [H]

    n_intra = tl.arange(0, BLOCK_N)
    scale_seg_off = (BLOCK_N * INDEX_DIM) // 4  # Scale-segment start offset in int32 elements

    blk_end = tl.minimum(blk_start + BLKS_PER_CTA, num_blocks)
    for b_i in tl.range(blk_start, blk_end, num_stages=NS_INNER):
        phys = tl.load(BlockTables + b * stride_bt_b + b_i).to(tl.int32)

        # ---- Load fp8 keys from the data segment: D contiguous bytes per token ----
        k_ptrs = (KvCache
                  + phys * stride_kv_blk
                  + n_intra[:, None] * INDEX_DIM
                  + d_offs[None, :])
        k_u8 = tl.load(k_ptrs)  # uint8, [BLOCK_N, D]
        k_tile = k_u8.to(tl.uint8, bitcast=True).to(tl.float8e4nv, bitcast=True)

        # ---- Load per-token fp32 scales together from the scale segment via the int32 view ----
        scale_ptrs = (KvCacheS4
                      + phys * stride_kv_s4
                      + scale_seg_off
                      + n_intra)
        k_scale = tl.load(scale_ptrs).to(tl.float32, bitcast=True)  # [BLOCK_N]

        # FP8 tensor-core dot: (BN, D) @ (D, H) -> (BN, H)
        sc = tl.dot(k_tile, q_tile_T, out_dtype=tl.float32)

        sc = sc * k_scale[:, None]
        if RELU_NEG:
            # Residual term: relu(-y) = max(-y, 0)
            sc = tl.maximum(-sc, 0.0) * w_vec[None, :]
        else:
            # Positive-ReLU term: relu(y) = max(y, 0)
            sc = tl.maximum(sc, 0.0) * w_vec[None, :]
        logits_tile = tl.sum(sc, axis=1)  # [BN]

        # Bounds: cu_s_min is always zero; check the context upper bound and max_blocks.
        k_i = b_i * BLOCK_N + n_intra
        in_range = (k_i < context_len) & (b_i < max_blocks)
        logits_tile = tl.where(in_range, logits_tile, float("-inf"))

        # Store Logits[b, s, b_i*BLOCK_N + n]
        col = b_i * BLOCK_N + n_intra
        out_ptrs = (Logits
                    + b.to(tl.int64) * stride_l_b
                    + s.to(tl.int64) * stride_l_s
                    + col.to(tl.int64))
        tl.store(out_ptrs, logits_tile, mask=col < max_model_len)


def paged_mqa_sparse_head_return_logits_triton(
    q: torch.Tensor,          # [B, seq_len, H, D] fp8
    kv_cache: torch.Tensor,   # [num_blocks, block_size, 1, D+4] uint8
    weights: torch.Tensor,    # [B, seq_len, H] or [B, H]
    context_lens: torch.Tensor,   # [B] int32
    block_tables: torch.Tensor,   # [B, max_blocks] int32
    max_model_len: int,
    dim: int,
    blks_per_cta: int = 4,
    relu_neg: bool = False,
) -> torch.Tensor:
    """Match the inputs and outputs of the tilelang paged_mqa_sparse_head_return_logits_interface.

    relu_neg=False (default): positive-ReLU term Σ_h w_h·relu(y_h).
    relu_neg=True: residual Σ_h w_h·relu(-y_h), used in the ReLU identity.
    """
    assert q.dtype == torch.float8_e4m3fn
    assert kv_cache.dtype == torch.uint8
    batch, seq_len, heads, index_dim = q.shape
    paged_block_size = int(kv_cache.shape[1])
    D_plus_4 = int(kv_cache.shape[3])
    max_blocks = int(block_tables.shape[1])

    block_H = heads if heads >= 16 else 16

    if weights.ndim == 2:
        weights = weights.view(batch, seq_len, heads)

    # Pad heads to block_H on the host and convert weights to fp32, matching the tilelang interface.
    if heads < block_H:
        q_padded = torch.zeros(
            batch, seq_len, block_H, index_dim, dtype=q.dtype, device=q.device,
        )
        q_padded[:, :, :heads, :] = q
        weights_padded = torch.zeros(
            batch, seq_len, block_H, dtype=torch.float32, device=weights.device,
        )
        weights_padded[:, :, :heads] = weights.float()
    else:
        q_padded = q if q.is_contiguous() else q.contiguous()
        weights_padded = weights.float()

    if not q_padded.is_contiguous():
        q_padded = q_padded.contiguous()
    if not weights_padded.is_contiguous():
        weights_padded = weights_padded.contiguous()
    if not kv_cache.is_contiguous():
        kv_cache = kv_cache.contiguous()
    if context_lens.dtype != torch.int32:
        context_lens = context_lens.to(torch.int32)
    if not context_lens.is_contiguous():
        context_lens = context_lens.contiguous()
    if block_tables.dtype != torch.int32:
        block_tables = block_tables.to(torch.int32)
    if not block_tables.is_contiguous():
        block_tables = block_tables.contiguous()

    logits = torch.full(
        (batch, seq_len, max_model_len), float("-inf"),
        dtype=torch.float32, device=q.device,
    )

    # ---- Occupancy-oriented configuration ----
    BLOCK_N = paged_block_size    # Must equal paged_block_size, as assumed by the kernel
    NW, NS, NS_INNER = 4, 2, 2

    # Int32 view for scales; D_plus_4 must be 4-byte aligned.
    assert D_plus_4 % 4 == 0, "D_plus_4 must be 4-byte aligned for int32 view to load scale segment"
    kv_cache_s4 = kv_cache.view(torch.int32)  # Logical layout: [num_blocks, block_size*(D+4)//4]

    # Parallelize over blocks: each CTA handles BLKS_PER_CTA pages to increase the CTA count.
    # Decode usually has seq_len=1 and possibly a small batch; splitting KV blocks improves occupancy.
    import math as _math
    max_num_blocks = _math.ceil(max_model_len / paged_block_size)
    BLKS_PER_CTA = blks_per_cta
    num_block_tiles = _math.ceil(max_num_blocks / BLKS_PER_CTA)

    grid = (batch * seq_len, num_block_tiles)

    # kv_cache strides count uint8 elements; the s4 view counts int32 elements.
    stride_kv_blk = kv_cache.stride(0)
    stride_kv_tok = kv_cache.stride(1)
    stride_kv_s4 = stride_kv_blk // 4  # Block stride in int32 elements

    _paged_mqa_sparse_head_return_logits_kernel[grid](
        q_padded, kv_cache, kv_cache_s4, logits, weights_padded,
        context_lens, block_tables,
        q_padded.stride(0), q_padded.stride(1), q_padded.stride(2),
        stride_kv_blk, stride_kv_tok,
        stride_kv_s4,
        logits.stride(0), logits.stride(1),
        weights_padded.stride(0), weights_padded.stride(1),
        block_tables.stride(0),
        max_blocks,
        max_model_len,
        SEQ_LEN=seq_len,
        HEADS=block_H,
        INDEX_DIM=dim,
        D_PLUS_4=D_plus_4,
        BLOCK_N=BLOCK_N,
        BLKS_PER_CTA=BLKS_PER_CTA,
        NS_INNER=NS_INNER,
        RELU_NEG=relu_neg,
        num_warps=NW,
        num_stages=NS,
    )
    return logits


# =====================================================================
# Kernel 3-fused: paged fused-head LINEAR term (DECODE path, ReLU identity)
#   logit_linear(s_token) = q_scale * Σ_d ( k_scale * k[n,d] * q_fused[d] )
#   q_fused = Σ_h w_h * q_h, weighted in host floating point before fp8 quantization.
#   One fused head, no ReLU; elementwise multiply-reduce avoids underutilizing a tensor-core dot with N=1.
# =====================================================================
@triton.jit
def _paged_mqa_fused_head_linear_logits_kernel(
    IndexQ,           # [batch, seq_len, INDEX_DIM] fp8 (single fused head)
    KvCache,          # [num_phys_blocks, BLOCK_N, 1, D+4]  uint8
    KvCacheS4,        # Int32 view of the same buffer for reading the scale segment
    Logits,           # [batch, seq_len, max_model_len]     fp32
    QScale,           # [batch, seq_len]                    fp32 (q_fused per-token scale)
    ContextLens,      # [batch]                             int32
    BlockTables,      # [batch, max_blocks]                 int32
    stride_q_b, stride_q_s,
    stride_kv_blk,
    stride_kv_s4,
    stride_l_b, stride_l_s,
    stride_qs_b, stride_qs_s,
    stride_bt_b,
    max_blocks,
    max_model_len,
    SEQ_LEN: tl.constexpr,
    INDEX_DIM: tl.constexpr,
    D_PLUS_4: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLKS_PER_CTA: tl.constexpr,
    NS_INNER: tl.constexpr,
):
    # grid = (batch * seq_len, num_block_tiles)
    bs = tl.program_id(0)
    b = bs // SEQ_LEN
    s = bs % SEQ_LEN
    tile = tl.program_id(1)

    context_len = tl.load(ContextLens + b).to(tl.int32)
    num_blocks = tl.cdiv(context_len, BLOCK_N)

    blk_start = tile * BLKS_PER_CTA
    if blk_start >= num_blocks:
        return

    d_offs = tl.arange(0, INDEX_DIM)

    # Load fused Q: q[d] = IndexQ[b, s, d], then convert to fp32.
    q_ptrs = IndexQ + b * stride_q_b + s * stride_q_s + d_offs
    q_vec = tl.load(q_ptrs).to(tl.float32)  # [D]

    q_scale = tl.load(QScale + b * stride_qs_b + s * stride_qs_s).to(tl.float32)

    n_intra = tl.arange(0, BLOCK_N)
    scale_seg_off = (BLOCK_N * INDEX_DIM) // 4

    blk_end = tl.minimum(blk_start + BLKS_PER_CTA, num_blocks)
    for b_i in tl.range(blk_start, blk_end, num_stages=NS_INNER):
        phys = tl.load(BlockTables + b * stride_bt_b + b_i).to(tl.int32)

        k_ptrs = (KvCache
                  + phys * stride_kv_blk
                  + n_intra[:, None] * INDEX_DIM
                  + d_offs[None, :])
        k_u8 = tl.load(k_ptrs)  # uint8, [BLOCK_N, D]
        k_tile = k_u8.to(tl.uint8, bitcast=True).to(tl.float8e4nv, bitcast=True).to(tl.float32)

        scale_ptrs = (KvCacheS4
                      + phys * stride_kv_s4
                      + scale_seg_off
                      + n_intra)
        k_scale = tl.load(scale_ptrs).to(tl.float32, bitcast=True)  # [BLOCK_N]

        # Elementwise multiply-reduce: s[n] = Σ_d k[n,d] * q_fused[d]
        prod = k_tile * q_vec[None, :]           # [BLOCK_N, D]
        raw = tl.sum(prod, axis=1)               # [BLOCK_N]
        # Linear term: no ReLU; multiply by k_scale and q_scale.
        logits_tile = raw * k_scale * q_scale    # [BLOCK_N]

        k_i = b_i * BLOCK_N + n_intra
        in_range = (k_i < context_len) & (b_i < max_blocks)
        logits_tile = tl.where(in_range, logits_tile, float("-inf"))

        col = b_i * BLOCK_N + n_intra
        out_ptrs = (Logits
                    + b.to(tl.int64) * stride_l_b
                    + s.to(tl.int64) * stride_l_s
                    + col.to(tl.int64))
        tl.store(out_ptrs, logits_tile, mask=col < max_model_len)


def paged_mqa_fused_head_linear_logits_triton(
    q_fused: torch.Tensor,     # [B, seq_len, D] fp8 (single fused head)
    q_scale: torch.Tensor,     # [B, seq_len] fp32 (per-token dynamic scale)
    kv_cache: torch.Tensor,    # [num_blocks, block_size, 1, D+4] uint8
    context_lens: torch.Tensor,   # [B] int32
    block_tables: torch.Tensor,   # [B, max_blocks] int32
    max_model_len: int,
    dim: int,
    blks_per_cta: int = 4,
) -> torch.Tensor:
    """Fused-head linear term: logit_linear = q_scale * scale * (k_s · q_fused).

    Output shape matches paged_mqa_sparse_head_return_logits_triton:
    logits[B, seq_len, max_model_len] fp32 (-inf outside the valid range).
    """
    assert q_fused.dtype == torch.float8_e4m3fn
    assert kv_cache.dtype == torch.uint8
    batch, seq_len, index_dim = q_fused.shape
    paged_block_size = int(kv_cache.shape[1])
    D_plus_4 = int(kv_cache.shape[3])
    max_blocks = int(block_tables.shape[1])

    q_fused = q_fused if q_fused.is_contiguous() else q_fused.contiguous()
    if q_scale.dtype != torch.float32:
        q_scale = q_scale.to(torch.float32)
    q_scale = q_scale.contiguous()
    if not kv_cache.is_contiguous():
        kv_cache = kv_cache.contiguous()
    if context_lens.dtype != torch.int32:
        context_lens = context_lens.to(torch.int32)
    context_lens = context_lens.contiguous()
    if block_tables.dtype != torch.int32:
        block_tables = block_tables.to(torch.int32)
    block_tables = block_tables.contiguous()

    logits = torch.full(
        (batch, seq_len, max_model_len), float("-inf"),
        dtype=torch.float32, device=q_fused.device,
    )

    BLOCK_N = paged_block_size
    NW, NS, NS_INNER = 4, 2, 2

    assert D_plus_4 % 4 == 0, "D_plus_4 must be 4-byte aligned for int32 view to load scale segment"
    kv_cache_s4 = kv_cache.view(torch.int32)

    import math as _math
    max_num_blocks = _math.ceil(max_model_len / paged_block_size)
    BLKS_PER_CTA = blks_per_cta
    num_block_tiles = _math.ceil(max_num_blocks / BLKS_PER_CTA)

    grid = (batch * seq_len, num_block_tiles)

    stride_kv_blk = kv_cache.stride(0)
    stride_kv_s4 = stride_kv_blk // 4

    _paged_mqa_fused_head_linear_logits_kernel[grid](
        q_fused, kv_cache, kv_cache_s4, logits, q_scale,
        context_lens, block_tables,
        q_fused.stride(0), q_fused.stride(1),
        stride_kv_blk,
        stride_kv_s4,
        logits.stride(0), logits.stride(1),
        q_scale.stride(0), q_scale.stride(1),
        block_tables.stride(0),
        max_blocks,
        max_model_len,
        SEQ_LEN=seq_len,
        INDEX_DIM=dim,
        D_PLUS_4=D_plus_4,
        BLOCK_N=BLOCK_N,
        BLKS_PER_CTA=BLKS_PER_CTA,
        NS_INNER=NS_INNER,
        num_warps=NW,
        num_stages=NS,
    )
    return logits


# =====================================================================
# Kernel 3-merged: paged DECODE fused logits using the ReLU identity
#   logits(b,s,n) = linear(b,s,n) + residual(b,s,n)
#     linear = q_scale[b,s] * Σ_d(k_scale[n] * k[n,d] * q_fused[b,s,d]) (no ReLU)
#     residual = Σ_h w_h * relu(-k_scale[n] * (k[n] · q_sparse[b,s,h])) (top-k heads)
#   ReLU identity: relu(y)=y+relu(-y); the linear term contains Σ_h w_h·y_h.
#   Load each KV block once: fused-head multiply-reduce plus sparse-head tensor-core dot.
# =====================================================================
@triton.jit
def _paged_mqa_fused_merged_logits_kernel(
    IndexQFused,      # [batch, seq_len, INDEX_DIM] fp8 (single fused head)
    IndexQSparse,     # [batch, seq_len, HEADS, INDEX_DIM] fp8 (sparse top-k heads)
    KvCache,          # [num_phys_blocks, BLOCK_N, 1, D+4]   uint8
    KvCacheS4,        # Int32 view of the same buffer for reading the scale segment
    Logits,           # [batch, seq_len, max_model_len]      fp32
    QScale,           # [batch, seq_len]                     fp32 (q_fused per-token scale)
    Weights,          # [batch, seq_len, HEADS]              fp32
    ContextLens,      # [batch]                              int32
    BlockTables,      # [batch, max_blocks]                  int32
    stride_qf_b, stride_qf_s,
    stride_qs_b, stride_qs_s, stride_qs_h,
    stride_kv_blk,
    stride_kv_s4,
    stride_l_b, stride_l_s,
    stride_scl_b, stride_scl_s,
    stride_w_b, stride_w_s,
    stride_bt_b,
    max_blocks,
    max_model_len,
    SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    INDEX_DIM: tl.constexpr,
    D_PLUS_4: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLKS_PER_CTA: tl.constexpr,
    NS_INNER: tl.constexpr,
):
    # grid = (batch * seq_len, num_block_tiles)
    bs = tl.program_id(0)
    b = bs // SEQ_LEN
    s = bs % SEQ_LEN
    tile = tl.program_id(1)

    context_len = tl.load(ContextLens + b).to(tl.int32)
    num_blocks = tl.cdiv(context_len, BLOCK_N)

    blk_start = tile * BLKS_PER_CTA
    if blk_start >= num_blocks:
        return

    d_offs = tl.arange(0, INDEX_DIM)
    h_offs = tl.arange(0, HEADS)

    # Fused Q: q_f[d] fp32
    qf_ptrs = IndexQFused + b * stride_qf_b + s * stride_qf_s + d_offs
    qf_vec = tl.load(qf_ptrs).to(tl.float32)  # [D]
    q_scale = tl.load(QScale + b * stride_scl_b + s * stride_scl_s).to(tl.float32)

    # Transposed sparse Q: q_s_T[d, h] = IndexQSparse[b, s, h, d], fp8
    qs_ptrs_T = (IndexQSparse
                 + b * stride_qs_b
                 + s * stride_qs_s
                 + h_offs[None, :] * stride_qs_h
                 + d_offs[:, None])
    qs_tile_T = tl.load(qs_ptrs_T)  # fp8, [D, H]

    # weights [H] fp32
    w_ptrs = Weights + b * stride_w_b + s * stride_w_s + h_offs
    w_vec = tl.load(w_ptrs).to(tl.float32)  # [H]

    n_intra = tl.arange(0, BLOCK_N)
    scale_seg_off = (BLOCK_N * INDEX_DIM) // 4

    blk_end = tl.minimum(blk_start + BLKS_PER_CTA, num_blocks)
    for b_i in tl.range(blk_start, blk_end, num_stages=NS_INNER):
        phys = tl.load(BlockTables + b * stride_bt_b + b_i).to(tl.int32)

        # ---- Load each KV block only once ----
        k_ptrs = (KvCache
                  + phys * stride_kv_blk
                  + n_intra[:, None] * INDEX_DIM
                  + d_offs[None, :])
        k_u8 = tl.load(k_ptrs)  # uint8, [BLOCK_N, D]
        k_fp8 = k_u8.to(tl.uint8, bitcast=True).to(tl.float8e4nv, bitcast=True)
        k_f32 = k_fp8.to(tl.float32)

        scale_ptrs = (KvCacheS4
                      + phys * stride_kv_s4
                      + scale_seg_off
                      + n_intra)
        k_scale = tl.load(scale_ptrs).to(tl.float32, bitcast=True)  # [BLOCK_N]

        # ---- Linear term: fused-head elementwise multiply-reduce, no ReLU ----
        prod = k_f32 * qf_vec[None, :]           # [BLOCK_N, D]
        raw_lin = tl.sum(prod, axis=1)           # [BLOCK_N]
        lin_tile = raw_lin * k_scale * q_scale   # [BLOCK_N]

        # ---- Residual: sparse-head tensor-core dot, relu(-y) ----
        sc = tl.dot(k_fp8, qs_tile_T, out_dtype=tl.float32)  # [BLOCK_N, H]
        sc = tl.maximum(-sc * k_scale[:, None], 0.0) * w_vec[None, :]
        res_tile = tl.sum(sc, axis=1)            # [BLOCK_N]

        logits_tile = lin_tile + res_tile        # [BLOCK_N]

        k_i = b_i * BLOCK_N + n_intra
        in_range = (k_i < context_len) & (b_i < max_blocks)
        logits_tile = tl.where(in_range, logits_tile, float("-inf"))

        col = b_i * BLOCK_N + n_intra
        out_ptrs = (Logits
                    + b.to(tl.int64) * stride_l_b
                    + s.to(tl.int64) * stride_l_s
                    + col.to(tl.int64))
        tl.store(out_ptrs, logits_tile, mask=col < max_model_len)


def paged_mqa_fused_merged_logits_triton(
    q_fused: torch.Tensor,     # [B, seq_len, D] fp8 (single fused head)
    q_scale: torch.Tensor,     # [B, seq_len] fp32
    q_sparse: torch.Tensor,    # [B, seq_len, H, D] fp8 (sparse top-k heads)
    weights: torch.Tensor,     # [B, seq_len, H] or [B, H] fp32
    kv_cache: torch.Tensor,    # [num_blocks, block_size, 1, D+4] uint8
    context_lens: torch.Tensor,   # [B] int32
    block_tables: torch.Tensor,   # [B, max_blocks] int32
    max_model_len: int,
    dim: int,
    blks_per_cta: int = 4,
) -> torch.Tensor:
    """DECODE: combine the fused linear term and sparse relu(-y) residual in one kernel.

    Return fp32 logits[B, seq_len, max_model_len], with -inf outside valid bounds.
    Numerically equivalent to the sum of separate linear and relu_neg sparse outputs.
    """
    assert q_fused.dtype == torch.float8_e4m3fn
    assert q_sparse.dtype == torch.float8_e4m3fn
    assert kv_cache.dtype == torch.uint8
    batch, seq_len, index_dim = q_fused.shape
    _, _, heads, _ = q_sparse.shape
    paged_block_size = int(kv_cache.shape[1])
    D_plus_4 = int(kv_cache.shape[3])
    max_blocks = int(block_tables.shape[1])

    block_H = triton.next_power_of_2(max(16, heads))

    if weights.ndim == 2:
        weights = weights.view(batch, seq_len, heads)

    # Pad heads to block_H with zero weights for padded heads.
    if heads < block_H:
        qs_padded = torch.zeros(
            batch, seq_len, block_H, index_dim, dtype=q_sparse.dtype, device=q_sparse.device,
        )
        qs_padded[:, :, :heads, :] = q_sparse
        weights_padded = torch.zeros(
            batch, seq_len, block_H, dtype=torch.float32, device=weights.device,
        )
        weights_padded[:, :, :heads] = weights.float()
    else:
        qs_padded = q_sparse if q_sparse.is_contiguous() else q_sparse.contiguous()
        weights_padded = weights.float()

    q_fused = q_fused if q_fused.is_contiguous() else q_fused.contiguous()
    if q_scale.dtype != torch.float32:
        q_scale = q_scale.to(torch.float32)
    q_scale = q_scale.contiguous()
    if not qs_padded.is_contiguous():
        qs_padded = qs_padded.contiguous()
    if not weights_padded.is_contiguous():
        weights_padded = weights_padded.contiguous()
    if not kv_cache.is_contiguous():
        kv_cache = kv_cache.contiguous()
    if context_lens.dtype != torch.int32:
        context_lens = context_lens.to(torch.int32)
    context_lens = context_lens.contiguous()
    if block_tables.dtype != torch.int32:
        block_tables = block_tables.to(torch.int32)
    block_tables = block_tables.contiguous()

    logits = torch.full(
        (batch, seq_len, max_model_len), float("-inf"),
        dtype=torch.float32, device=q_fused.device,
    )

    BLOCK_N = paged_block_size
    NW, NS, NS_INNER = 4, 2, 2

    assert D_plus_4 % 4 == 0, "D_plus_4 must be 4-byte aligned for int32 view to load scale segment"
    kv_cache_s4 = kv_cache.view(torch.int32)

    import math as _math
    max_num_blocks = _math.ceil(max_model_len / paged_block_size)
    BLKS_PER_CTA = blks_per_cta
    num_block_tiles = _math.ceil(max_num_blocks / BLKS_PER_CTA)

    grid = (batch * seq_len, num_block_tiles)

    stride_kv_blk = kv_cache.stride(0)
    stride_kv_s4 = stride_kv_blk // 4

    _paged_mqa_fused_merged_logits_kernel[grid](
        q_fused, qs_padded, kv_cache, kv_cache_s4, logits, q_scale, weights_padded,
        context_lens, block_tables,
        q_fused.stride(0), q_fused.stride(1),
        qs_padded.stride(0), qs_padded.stride(1), qs_padded.stride(2),
        stride_kv_blk,
        stride_kv_s4,
        logits.stride(0), logits.stride(1),
        q_scale.stride(0), q_scale.stride(1),
        weights_padded.stride(0), weights_padded.stride(1),
        block_tables.stride(0),
        max_blocks,
        max_model_len,
        SEQ_LEN=seq_len,
        HEADS=block_H,
        INDEX_DIM=dim,
        D_PLUS_4=D_plus_4,
        BLOCK_N=BLOCK_N,
        BLKS_PER_CTA=BLKS_PER_CTA,
        NS_INNER=NS_INNER,
        num_warps=NW,
        num_stages=NS,
    )
    return logits


# =====================================================================
# Paged DECODE: dual-pointer, single-dot merged 8-head logits without host concatenation
#   Column head 0 is fused; heads 1..HEADS are sparse, combined in one tl.dot.
#   Compute linear and residual terms together; HP1=HEADS+1 must be a power of two (7 -> 8).
# =====================================================================
@triton.jit
def _paged_mqa_fused_merged_8h_v2_kernel(
    IndexQFused,      # [batch, seq_len, INDEX_DIM] fp8 (single fused head)
    IndexQSparse,     # [batch, seq_len, HEADS, INDEX_DIM] fp8 (sparse top-k heads)
    KvCache,          # [num_phys_blocks, BLOCK_N, 1, D+4]   uint8
    KvCacheS4,        # Int32 view of the same buffer for reading the scale segment
    Logits,           # [batch, seq_len, max_model_len]      fp32
    QScale,           # [batch, seq_len]                     fp32 (q_fused per-token scale)
    Weights,          # [batch, seq_len, HEADS]              fp32
    ContextLens,      # [batch]                              int32
    BlockTables,      # [batch, max_blocks]                  int32
    stride_qf_b, stride_qf_s,
    stride_qs_b, stride_qs_s, stride_qs_h,
    stride_kv_blk,
    stride_kv_s4,
    stride_l_b, stride_l_s,
    stride_scl_b, stride_scl_s,
    stride_w_b, stride_w_s,
    stride_bt_b,
    max_blocks,
    max_model_len,
    SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,       # Number of sparse heads
    HP1: tl.constexpr,         # HEADS + 1, a power of two
    INDEX_DIM: tl.constexpr,
    D_PLUS_4: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLKS_PER_CTA: tl.constexpr,
    NS_INNER: tl.constexpr,
):
    # grid = (batch * seq_len, num_block_tiles)
    bs = tl.program_id(0)
    b = bs // SEQ_LEN
    s = bs % SEQ_LEN
    tile = tl.program_id(1)

    context_len = tl.load(ContextLens + b).to(tl.int32)
    num_blocks = tl.cdiv(context_len, BLOCK_N)

    blk_start = tile * BLKS_PER_CTA
    if blk_start >= num_blocks:
        return

    d_offs = tl.arange(0, INDEX_DIM)

    # ---- Build Q [D, HP1] without host concatenation: column 0 fused, 1..HEADS sparse ----
    hp1_cols = tl.arange(0, HP1)
    is_sparse = hp1_cols > 0  # [HP1] bool (column 0 is fused)

    # Fused pointers: broadcast across columns; only column 0 is valid.
    qf_ptrs_T = (IndexQFused
                 + b * stride_qf_b + s * stride_qf_s
                 + d_offs[:, None])  # [D, 1], broadcast
    qf_tile_T = tl.load(qf_ptrs_T + tl.zeros((1, HP1), dtype=tl.int32),
                        mask=(~is_sparse)[None, :], other=0.0)  # [D, HP1]

    # Sparse pointers: column j (j>=1) reads head j-1.
    qs_head = hp1_cols - 1  # Column 0 maps to -1 (masked out)
    qs_ptrs_T = (IndexQSparse
                 + b * stride_qs_b + s * stride_qs_s
                 + qs_head[None, :] * stride_qs_h
                 + d_offs[:, None])
    qs_tile_T = tl.load(qs_ptrs_T, mask=is_sparse[None, :], other=0.0)  # [D, HP1]

    qa_tile_T = tl.where(is_sparse[None, :], qs_tile_T, qf_tile_T)  # [D, HP1] fp8

    # ---- Coefficients [HP1]: column 0=q_scale, column j>=1=w_(j-1) ----
    q_scale = tl.load(QScale + b * stride_scl_b + s * stride_scl_s).to(tl.float32)
    w_ptrs = Weights + b * stride_w_b + s * stride_w_s + qs_head
    w_vec = tl.load(w_ptrs, mask=is_sparse, other=0.0).to(tl.float32)  # [HP1]
    c_vec = tl.where(is_sparse, w_vec, q_scale)  # [HP1]

    n_intra = tl.arange(0, BLOCK_N)
    scale_seg_off = (BLOCK_N * INDEX_DIM) // 4

    blk_end = tl.minimum(blk_start + BLKS_PER_CTA, num_blocks)
    for b_i in tl.range(blk_start, blk_end, num_stages=NS_INNER):
        phys = tl.load(BlockTables + b * stride_bt_b + b_i).to(tl.int32)

        k_ptrs = (KvCache
                  + phys * stride_kv_blk
                  + n_intra[:, None] * INDEX_DIM
                  + d_offs[None, :])
        k_u8 = tl.load(k_ptrs)
        k_fp8 = k_u8.to(tl.uint8, bitcast=True).to(tl.float8e4nv, bitcast=True)

        scale_ptrs = (KvCacheS4
                      + phys * stride_kv_s4
                      + scale_seg_off
                      + n_intra)
        k_scale = tl.load(scale_ptrs).to(tl.float32, bitcast=True)  # [BLOCK_N]

        # ---- Single dot: (BN,D)@(D,HP1) -> (BN,HP1) ----
        s_all = tl.dot(k_fp8, qa_tile_T, out_dtype=tl.float32)  # [BN, HP1]
        y = s_all * k_scale[:, None]                            # [BN, HP1]

        lin = y
        res = tl.maximum(-y, 0.0)
        contrib = tl.where(is_sparse[None, :], res, lin) * c_vec[None, :]
        logits_tile = tl.sum(contrib, axis=1)  # [BN]

        k_i = b_i * BLOCK_N + n_intra
        in_range = (k_i < context_len) & (b_i < max_blocks)
        logits_tile = tl.where(in_range, logits_tile, float("-inf"))

        col = b_i * BLOCK_N + n_intra
        out_ptrs = (Logits
                    + b.to(tl.int64) * stride_l_b
                    + s.to(tl.int64) * stride_l_s
                    + col.to(tl.int64))
        tl.store(out_ptrs, logits_tile, mask=col < max_model_len)


def paged_mqa_fused_merged_8h_v2_logits_triton(
    q_fused: torch.Tensor,     # [B, seq_len, D] fp8 (single fused head)
    q_scale: torch.Tensor,     # [B, seq_len] fp32
    q_sparse: torch.Tensor,    # [B, seq_len, H, D] fp8 (sparse top-k heads)
    weights: torch.Tensor,     # [B, seq_len, H] or [B, H] fp32
    kv_cache: torch.Tensor,    # [num_blocks, block_size, 1, D+4] uint8
    context_lens: torch.Tensor,   # [B] int32
    block_tables: torch.Tensor,   # [B, max_blocks] int32
    max_model_len: int,
    dim: int,
    blks_per_cta: int = 4,
) -> torch.Tensor:
    """DECODE: combine fused column 0 and sparse columns 1..H in one dot without host concatenation.

    Numerically equivalent to paged_mqa_fused_merged_logits_triton; linear and residual
    terms use H+1 parallel heads in the same dot. HP1=H+1 must be a power of two
    (H=7 gives HP1=8).
    """
    assert q_fused.dtype == torch.float8_e4m3fn
    assert q_sparse.dtype == torch.float8_e4m3fn
    assert kv_cache.dtype == torch.uint8
    batch, seq_len, index_dim = q_fused.shape
    _, _, heads, _ = q_sparse.shape
    paged_block_size = int(kv_cache.shape[1])
    D_plus_4 = int(kv_cache.shape[3])
    max_blocks = int(block_tables.shape[1])

    hp1 = heads + 1
    assert (hp1 & (hp1 - 1)) == 0, f"HP1={hp1} 必须是 2 的幂 (稀疏头数 H={heads} 应为 2^k-1)"

    if weights.ndim == 2:
        weights = weights.view(batch, seq_len, heads)

    q_fused = q_fused if q_fused.is_contiguous() else q_fused.contiguous()
    if q_scale.dtype != torch.float32:
        q_scale = q_scale.to(torch.float32)
    q_scale = q_scale.contiguous()
    q_sparse = q_sparse if q_sparse.is_contiguous() else q_sparse.contiguous()
    weights = weights.float().contiguous()
    if not kv_cache.is_contiguous():
        kv_cache = kv_cache.contiguous()
    if context_lens.dtype != torch.int32:
        context_lens = context_lens.to(torch.int32)
    context_lens = context_lens.contiguous()
    if block_tables.dtype != torch.int32:
        block_tables = block_tables.to(torch.int32)
    block_tables = block_tables.contiguous()

    logits = torch.full(
        (batch, seq_len, max_model_len), float("-inf"),
        dtype=torch.float32, device=q_fused.device,
    )

    BLOCK_N = paged_block_size
    NW, NS, NS_INNER = 4, 2, 2

    assert D_plus_4 % 4 == 0, "D_plus_4 must be 4-byte aligned for int32 view to load scale segment"
    kv_cache_s4 = kv_cache.view(torch.int32)

    import math as _math
    max_num_blocks = _math.ceil(max_model_len / paged_block_size)
    BLKS_PER_CTA = blks_per_cta
    num_block_tiles = _math.ceil(max_num_blocks / BLKS_PER_CTA)

    grid = (batch * seq_len, num_block_tiles)

    stride_kv_blk = kv_cache.stride(0)
    stride_kv_s4 = stride_kv_blk // 4

    _paged_mqa_fused_merged_8h_v2_kernel[grid](
        q_fused, q_sparse, kv_cache, kv_cache_s4, logits, q_scale, weights,
        context_lens, block_tables,
        q_fused.stride(0), q_fused.stride(1),
        q_sparse.stride(0), q_sparse.stride(1), q_sparse.stride(2),
        stride_kv_blk,
        stride_kv_s4,
        logits.stride(0), logits.stride(1),
        q_scale.stride(0), q_scale.stride(1),
        weights.stride(0), weights.stride(1),
        block_tables.stride(0),
        max_blocks,
        max_model_len,
        SEQ_LEN=seq_len,
        HEADS=heads,
        HP1=hp1,
        INDEX_DIM=dim,
        D_PLUS_4=D_plus_4,
        BLOCK_N=BLOCK_N,
        BLKS_PER_CTA=BLKS_PER_CTA,
        NS_INNER=NS_INNER,
        num_warps=NW,
        num_stages=NS,
    )
    return logits

# =====================================================================
# Kernel 4: decode paged stage-2 attn (sparse index gather, paged KV)
# =====================================================================
@triton.jit
def _paged_mqa_attn_return_logits_kernel(
    IndexQ,             # [batch, seq_len, HEADS, INDEX_DIM]  fp8
    KvCache,            # [num_phys_blocks, PBS, 1, D+4]      uint8
    KvCacheS4,          # Int32 view of the same buffer for reading the scale segment
    Logits,             # [batch, seq_len, PADDED_TOPK]       fp32
    Weights,            # [batch, seq_len, HEADS]             fp32
    SparseTopKIndices,  # [batch, seq_len, STAGE1_TOPK]       int32
    ContextLens,        # [batch]                             int32
    BlockTables,        # [batch, max_blocks]                 int32
    stride_q_b, stride_q_s, stride_q_h,
    stride_kv_blk, stride_kv_tok,
    stride_kv_s4,
    stride_l_b, stride_l_s,
    stride_w_b, stride_w_s,
    stride_idx_b, stride_idx_s,
    stride_bt_b,
    max_blocks,
    stage1_topk,
    SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    INDEX_DIM: tl.constexpr,
    D_PLUS_4: tl.constexpr,
    PAGED_BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NS_INNER: tl.constexpr,
):
    # grid = (batch*seq_len, num_topk_tiles)
    bs = tl.program_id(0)
    b = bs // SEQ_LEN
    s = bs % SEQ_LEN
    tile = tl.program_id(1)

    col_start = tile * BLOCK_N
    if col_start >= stage1_topk:
        return

    context_len = tl.load(ContextLens + b).to(tl.int32)

    d_offs = tl.arange(0, INDEX_DIM)
    h_offs = tl.arange(0, HEADS)

    # Load Q transposed: q_T[d, h] = IndexQ[b, s, h, d]
    q_ptrs_T = (IndexQ
                + b * stride_q_b
                + s * stride_q_s
                + h_offs[None, :] * stride_q_h
                + d_offs[:, None])
    q_tile_T = tl.load(q_ptrs_T)  # fp8, [D, H]

    # weights [H] fp32
    w_ptrs = Weights + b * stride_w_b + s * stride_w_s + h_offs
    w_vec = tl.load(w_ptrs).to(tl.float32)  # [H]

    n_intra = tl.arange(0, BLOCK_N)
    # ---- Block-segmented layout ----
    # page block buffer: [tok0(D) tok1(D) ... | tok0_scale(4) tok1_scale(4) ...]
    scale_seg_off = (PAGED_BLOCK_SIZE * INDEX_DIM) // 4  # Scale segment start in int32 units

    k_col_offs = col_start + n_intra
    col_mask = k_col_offs < stage1_topk

    # gather sparse indices
    idx = tl.load(SparseTopKIndices + b * stride_idx_b + s * stride_idx_s + k_col_offs,
                  mask=col_mask, other=-1).to(tl.int32)

    # Logical block index and token offset within the block
    p = idx // PAGED_BLOCK_SIZE
    tok = idx % PAGED_BLOCK_SIZE
    idx_valid = (idx >= 0) & (idx < context_len) & (p < max_blocks) & col_mask
    p_safe = tl.where(idx_valid, p, 0)
    tok_safe = tl.where(idx_valid, tok, 0)

    # phys = BlockTables[b, p]
    phys = tl.load(BlockTables + b * stride_bt_b + p_safe,
                   mask=idx_valid, other=0).to(tl.int32)

    # ---- Load fp8 keys: D contiguous bytes per token in the data segment ----
    k_ptrs = (KvCache
              + phys[:, None] * stride_kv_blk
              + tok_safe[:, None] * INDEX_DIM
              + d_offs[None, :])
    k_u8 = tl.load(k_ptrs, mask=idx_valid[:, None], other=0)  # uint8, [BN, D]
    k_tile = k_u8.to(tl.uint8, bitcast=True).to(tl.float8e4nv, bitcast=True)

    # ---- Load per-token fp32 scales once from the scale segment via the int32 view ----
    scale_ptrs = (KvCacheS4
                  + phys * stride_kv_s4
                  + scale_seg_off
                  + tok_safe)
    k_scale = tl.load(scale_ptrs, mask=idx_valid, other=0).to(tl.float32, bitcast=True)

    # FP8 tensor-core dot: (BN, D) @ (D, H) -> (BN, H)
    sc = tl.dot(k_tile, q_tile_T, out_dtype=tl.float32)
    sc = tl.maximum(sc * k_scale[:, None], 0.0) * w_vec[None, :]
    logits_tile = tl.sum(sc, axis=1)  # [BN]

    logits_tile = tl.where(idx_valid, logits_tile, float("-inf"))

    out_ptrs = (Logits
                + b.to(tl.int64) * stride_l_b
                + s.to(tl.int64) * stride_l_s
                + k_col_offs.to(tl.int64))
    tl.store(out_ptrs, logits_tile, mask=col_mask)


def paged_mqa_attn_return_logits_triton(
    q: torch.Tensor,               # [B, seq_len, H, D] fp8
    kv_cache: torch.Tensor,        # [num_blocks, block_size, 1, D+4] uint8
    weights: torch.Tensor,         # [B, seq_len, H] or [B, H]
    context_lens: torch.Tensor,    # [B] int32
    block_tables: torch.Tensor,    # [B, max_blocks] int32
    sparse_topk_indices: torch.Tensor,  # [B, seq_len, stage1_topk] int32
    dim: int,
) -> torch.Tensor:
    """Match the inputs and outputs of tilelang paged_mqa_attn_return_logits_interface.
    Return fp32 logits[B, seq_len, stage1_topk] without padding.
    """
    assert q.dtype == torch.float8_e4m3fn
    assert kv_cache.dtype == torch.uint8
    batch, seq_len, heads, index_dim = q.shape
    paged_block_size = int(kv_cache.shape[1])
    D_plus_4 = int(kv_cache.shape[3])
    max_blocks = int(block_tables.shape[1])
    stage1_topk = int(sparse_topk_indices.shape[-1])

    assert D_plus_4 % 4 == 0, "D_plus_4 must be 4-byte aligned for int32 view to load scale segment"

    if weights.ndim == 2:
        weights = weights.view(batch, seq_len, heads)

    # fp8 tl.dot requires output N >= 16; pad heads with zero weights to preserve the sum.
    block_H = heads if heads >= 16 else 16
    if heads < block_H:
        q_padded = torch.zeros(
            batch, seq_len, block_H, index_dim, dtype=q.dtype, device=q.device,
        )
        q_padded[:, :, :heads, :] = q
        weights_padded = torch.zeros(
            batch, seq_len, block_H, dtype=torch.float32, device=weights.device,
        )
        weights_padded[:, :, :heads] = weights.float()
    else:
        q_padded = q if q.is_contiguous() else q.contiguous()
        weights_padded = weights.float()

    q_c = q_padded if q_padded.is_contiguous() else q_padded.contiguous()
    weights_f = weights_padded if weights_padded.is_contiguous() else weights_padded.contiguous()
    if not kv_cache.is_contiguous():
        kv_cache = kv_cache.contiguous()
    if context_lens.dtype != torch.int32:
        context_lens = context_lens.to(torch.int32)
    context_lens = context_lens.contiguous()
    if block_tables.dtype != torch.int32:
        block_tables = block_tables.to(torch.int32)
    block_tables = block_tables.contiguous()
    if sparse_topk_indices.dtype != torch.int32:
        sparse_topk_indices = sparse_topk_indices.to(torch.int32)
    sparse_topk_indices = sparse_topk_indices.contiguous()

    logits = torch.full(
        (batch, seq_len, stage1_topk), float("-inf"),
        dtype=torch.float32, device=q.device,
    )

    kv_cache_s4 = kv_cache.view(torch.int32)

    import math as _math
    BLOCK_N = paged_block_size
    NW, NS, NS_INNER = 4, 2, 2
    num_topk_tiles = _math.ceil(stage1_topk/ BLOCK_N)

    grid = (batch * seq_len, num_topk_tiles)

    stride_kv_blk = kv_cache.stride(0)
    stride_kv_tok = kv_cache.stride(1)
    stride_kv_s4 = stride_kv_blk // 4

    _paged_mqa_attn_return_logits_kernel[grid](
        q_c, kv_cache, kv_cache_s4, logits, weights_f,
        sparse_topk_indices, context_lens, block_tables,
        q_c.stride(0), q_c.stride(1), q_c.stride(2),
        stride_kv_blk, stride_kv_tok,
        stride_kv_s4,
        logits.stride(0), logits.stride(1),
        weights_f.stride(0), weights_f.stride(1),
        sparse_topk_indices.stride(0), sparse_topk_indices.stride(1),
        block_tables.stride(0),
        max_blocks,
        stage1_topk,
        SEQ_LEN=seq_len,
        HEADS=block_H,
        INDEX_DIM=dim,
        D_PLUS_4=D_plus_4,
        PAGED_BLOCK_SIZE=paged_block_size,
        BLOCK_N=BLOCK_N,
        NS_INNER=NS_INNER,
        num_warps=NW,
        num_stages=NS,
    )
    return logits


def report_best_configs():
    """Return placeholder strings for static dispatch without an autotune cache."""
    return ("[stage1] static dispatch (no autotune cache)\n"
            "[stage2] static dispatch (persistent, no autotune cache)")


__all__ = [
    "resmisa_mqa_sparse_head_return_logits_triton",
    "resmisa_mqa_attn_return_logits_triton",
    "paged_mqa_sparse_head_return_logits_triton",
    "paged_mqa_attn_return_logits_triton",
    "report_best_configs",
]