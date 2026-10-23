import torch
import tilelang
from tilelang import language as T

global_pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }

@tilelang.jit(
    pass_configs=global_pass_configs,
)
def batch_decode_block_mqa_attn_return_logits(
    heads: int,
    index_dim: int,
    block_N: int = 128,
    block_H: int = 64,
    num_stages: int = 1,
    threads: int = 256,
    q_dtype: str = "float8_e4m3fn",
    k_dtype: str = "float32",
    relu_neg: bool = True,
):
    """Score decode heads with FP8 Q and FP32 pooled keys.

    Cast both inputs to BF16 when loading shared memory for GEMM, avoiding
    separate interface-level conversion kernels.

    relu_neg selects the same scoring rule as the prefill kernel:
        True (default): use relu(-y) = max(-s, 0) to rank heads by the
            magnitude of their weighted ReLU-decomposition residuals.
        False: use relu(+y) = max(s, 0) for positive-axis attention mass.
    """
    compute_dtype = "bfloat16"
    accum_dtype = T.float32
    index_dtype = T.int32

    batch = T.dynamic("batch")
    nb = T.dynamic("seq_len_blocked_kv")

    q_shape = [batch, heads, index_dim]
    k_shape = [batch, nb, index_dim]
    w_shape = [batch, heads]

    block_H_pad = T.ceildiv(block_H, 16) * 16
    assert block_H_pad == heads

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, q_dtype),  # type: ignore  # fp8
        BlockedK: T.Tensor(k_shape, k_dtype),  # type: ignore  # fp32
        QScores: T.Tensor(w_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor(w_shape, "bfloat16"),  # type: ignore
        ContextLens: T.Tensor([batch], index_dtype),  # type: ignore
    ):
        with T.Kernel(batch, 1, threads=threads) as (bx, by):
            k_shared = T.alloc_shared([block_N, index_dim], compute_dtype)
            q_shared = T.alloc_shared([block_H_pad, index_dim], compute_dtype)
            q_scores = T.alloc_fragment([block_H_pad], accum_dtype)

            s = T.alloc_fragment([block_N, block_H_pad], accum_dtype)
            w = T.alloc_fragment([block_H_pad], accum_dtype)

            k_e = T.min(ContextLens[bx], nb)

            # Load FP8 Q from global memory and cast to BF16 in shared memory.
            for h_i, d_i in T.Parallel(block_H_pad, index_dim):
                q_shared[h_i, d_i] = T.cast(Q[bx, h_i, d_i], compute_dtype)

            T.copy(Weights[bx, 0], w)

            for k_i in T.Pipelined(T.ceildiv(nb, block_N), num_stages=num_stages):
                k_start = k_i * block_N
                # Load FP32 BlockedK from global memory and cast to BF16 in shared memory.
                for bn_i, d in T.Parallel(block_N, index_dim):
                    k_shared[bn_i, d] = T.cast(BlockedK[bx, k_start + bn_i, d], compute_dtype)

                T.gemm(
                    k_shared,
                    q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for kn_i, hn_i in T.Parallel(block_N, block_H_pad):
                    k_col = k_start + kn_i
                    if k_col < k_e:
                        if relu_neg:
                            # Score the relu(-y) residual using max(-s, 0).
                            s[kn_i, hn_i] = T.max(-s[kn_i, hn_i], 0) * w[hn_i]
                        else:
                            s[kn_i, hn_i] = T.max(s[kn_i, hn_i], 0) * w[hn_i]
                    else:
                        s[kn_i, hn_i] = T.cast(0, accum_dtype)

                T.reduce_abssum(s, q_scores, dim=0)
                for hn_i in T.Parallel(block_H_pad):
                    QScores[bx, hn_i] += q_scores[hn_i]

    return kernel


def batch_block_mqa_attn_return_logits_interface(
    q: torch.Tensor,
    blocked_kv: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    kv_block_size: int,
    block_N: int = 64,
    relu_neg: bool = True,
):
    """Return decode head scores, matching the prefill scoring interface.

    Args:
        q: [B, 1, H, D] FP8.
        blocked_kv: [B, Nb, D] FP32.
        weights: [B, 1, H] or [B, H] BF16.
        context_lens: [B] valid pooled-block counts.
        relu_neg: True (default) scores weighted relu(-y) residual magnitudes;
            False scores weighted relu(+y) magnitudes.
    Returns:
        q_head_score: [B, H] FP32.

    The kernel casts FP8 queries and FP32 keys to BF16 in shared memory,
    avoiding separate interface-level conversion kernels.
    """

    assert len(q.shape) == 4
    B, seq_len_q, H, D = q.shape
    B, seq_len_kv, D = blocked_kv.shape

    assert seq_len_q == 1, "decode expects q_len=1"

    q = q.squeeze(1)  # [B, H, D] fp8
    weights = weights.squeeze(1)

    q_head_score = torch.zeros((B, H), device=q.device, dtype=torch.float32)

    kernel = batch_decode_block_mqa_attn_return_logits(
        heads=H,
        index_dim=D,
        block_N=block_N,
        block_H=H,
        q_dtype="float8_e4m3fn",
        k_dtype="float32",
        relu_neg=relu_neg,
    )
    kernel(
        q,              # fp8, cast to BF16 inside the kernel
        blocked_kv,     # fp32, cast to BF16 inside the kernel
        q_head_score,
        weights,
        context_lens if context_lens.dtype == torch.int32 else context_lens.to(torch.int32),
    )

    return q_head_score


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def paged_block_sparse_mqa_attn_return_logits(
    paged_block_size,
    kv_block_size,
    topk,
    heads,
    index_dim,
    num_stages=1,
    threads=256,
    dtype="float8_e4m3fn",
):
    accum_dtype = T.float32
    index_dtype = T.int32

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    max_blocks = T.dynamic("max_blocks")
    num_phys_blocks = T.dynamic("num_phys_blocks")

    index_q_shape = [batch, seq_len, heads, index_dim]
    kv_cache_shape = [num_phys_blocks, paged_block_size, 1, index_dim]
    kv_scale_shape = [num_phys_blocks, paged_block_size]
    logits_shape = [batch, seq_len, topk * kv_block_size]

    block_H = heads if heads >= 16 else 16
    weights_shape = [batch, seq_len, block_H]
    block_N = paged_block_size
    assert block_N > 0, "block_N must be positive"
    assert kv_block_size >= block_N and kv_block_size % block_N == 0, "block_N must divide kv_block_size"
    assert paged_block_size >= block_N and paged_block_size % block_N == 0, "block_N must divide paged_block_size"
    assert paged_block_size == block_N, "for simplicity we require paged_block_size == block_N in this kernel"

    @T.prim_func
    def paged_block_sparse_mqa_attn_return_logits_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        KvCache: T.Tensor(kv_cache_shape, dtype),  # type: ignore
        KvScale: T.Tensor(kv_scale_shape, accum_dtype),  # type: ignore
        TopKBlockIndex: T.Tensor([batch, seq_len, topk], index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor(weights_shape, "bfloat16"),  # type: ignore
        ContextLens: T.Tensor([batch], index_dtype),  # type: ignore
        BlockTables: T.Tensor([batch, max_blocks], index_dtype),  # type: ignore
    ):
        with T.Kernel(batch, seq_len, threads=threads) as (bx, by):
            b = bx
            seq_len_i = by

            index_q_shared = T.alloc_shared([block_H, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, block_H], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, 1, block_H))
            logits = T.alloc_fragment([block_N, 1], accum_dtype)
            weights = T.alloc_fragment([1, block_H], accum_dtype)
            k_scale_frag = T.alloc_fragment([block_N], accum_dtype)

            cu_k_s_min = T.cast(0, index_dtype)
            cu_k_e_max = ContextLens[b]

            for h_i, dim in T.Parallel(block_H, index_dim):
                if h_i < heads:
                    index_q_shared[h_i, dim] = IndexQ[b, seq_len_i, h_i, dim]
                else:
                    index_q_shared[h_i, dim] = 0

            for h_i in T.Parallel(block_H):
                if h_i < heads:
                    weights[0, h_i] = Weights[b, seq_len_i, h_i]
                else:
                    weights[0, h_i] = 0

            for n_i in T.serial(topk):
                topk_block_id = TopKBlockIndex[b, seq_len_i, n_i]
                block_s = topk_block_id * kv_block_size
                for b_i in T.Pipelined(kv_block_size // block_N, num_stages=num_stages):
                    block_s_i = block_s + b_i * block_N

                    if block_s_i // paged_block_size >= 0 and block_s_i // paged_block_size < max_blocks:
                        phys = BlockTables[b, block_s_i // paged_block_size]
                        for bn_i, d in T.Parallel(block_N, index_dim):
                            index_k_shared[bn_i, d] = KvCache[phys, bn_i, 0, d]
                        for bn_i in T.Parallel(block_N):
                            k_scale_frag[bn_i] = KvScale[phys, bn_i]

                    T.gemm(
                        index_k_shared,
                        index_q_shared,
                        s,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    for bn_i, bq_i, h_i in T.Parallel(block_N, 1, block_H):
                        if h_i < heads:
                            s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i] * k_scale_frag[bn_i], 0) * weights[bq_i, h_i])
                        else:
                            s_reshaped[bn_i, bq_i, h_i] = 0
                    
                    T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                    for i_i in T.Parallel(block_N):
                        k_i = block_s_i + i_i
                        p = k_i // paged_block_size
                        if (k_i < cu_k_s_min) or (k_i >= cu_k_e_max) or (p < 0) or (p >= max_blocks):
                            logits[i_i, 0] = -T.infinity(accum_dtype)

                    for bn_i in T.Parallel(block_N):
                        Logits[b, seq_len_i, n_i * kv_block_size + b_i * block_N + bn_i] = logits[bn_i, 0]
    
    @T.prim_func
    def paged_block_sparse_mqa_attn_return_logits_kernel_for_small_pooling_size(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        KvCache: T.Tensor(kv_cache_shape, dtype),  # type: ignore
        KvScale: T.Tensor(kv_scale_shape, accum_dtype),  # type: ignore
        TopKBlockIndex: T.Tensor([batch, seq_len, topk], index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor(weights_shape, "bfloat16"),  # type: ignore
        ContextLens: T.Tensor([batch], index_dtype),  # type: ignore
        BlockTables: T.Tensor([batch, max_blocks], index_dtype),  # type: ignore
    ):
        with T.Kernel(batch, seq_len, threads=threads) as (bx, by):
            b = bx
            seq_len_i = by

            index_q_shared = T.alloc_shared([block_H, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, block_H], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, 1, block_H))
            logits = T.alloc_fragment([block_N, 1], accum_dtype)
            weights = T.alloc_fragment([1, block_H], accum_dtype)
            k_scale_frag = T.alloc_fragment([block_N], accum_dtype)

            cu_k_s_min = T.cast(0, index_dtype)
            cu_k_e_max = ContextLens[b]

            for h_i, dim in T.Parallel(block_H, index_dim):
                if h_i < heads:
                    index_q_shared[h_i, dim] = IndexQ[b, seq_len_i, h_i, dim]
                else:
                    index_q_shared[h_i, dim] = 0

            for h_i in T.Parallel(block_H):
                if h_i < heads:
                    weights[0, h_i] = Weights[b, seq_len_i, h_i]
                else:
                    weights[0, h_i] = 0

            for n_i in T.serial(topk):
                topk_block_id = TopKBlockIndex[b, seq_len_i, n_i]
                block_s_i = topk_block_id * kv_block_size

                if block_s_i // paged_block_size >= 0 and block_s_i // paged_block_size < max_blocks:
                    phys = BlockTables[b, block_s_i // paged_block_size]
                    for bn_i, d in T.Parallel(block_N, index_dim):
                        index_k_shared[bn_i, d] = KvCache[phys, bn_i, 0, d]
                    for bn_i in T.Parallel(block_N):
                        k_scale_frag[bn_i] = KvScale[phys, bn_i]

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, 1, block_H):
                    if h_i < heads:
                        s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i] * k_scale_frag[bn_i], 0) * weights[bq_i, h_i])
                    else:
                        s_reshaped[bn_i, bq_i, h_i] = 0
                        
                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for i_i in T.Parallel(block_N):
                    k_i = block_s_i + i_i
                    p = k_i // paged_block_size
                    if (k_i < cu_k_s_min) or (k_i >= cu_k_e_max) or (p < 0) or (p >= max_blocks):
                        logits[i_i, 0] = -T.infinity(accum_dtype)

                for bn_i in T.Parallel(block_N):
                    Logits[b, seq_len_i, n_i * kv_block_size + bn_i] = logits[bn_i, 0]

    if kv_block_size == block_N:
        return paged_block_sparse_mqa_attn_return_logits_kernel_for_small_pooling_size
    else:
        return paged_block_sparse_mqa_attn_return_logits_kernel

def paged_block_sparse_mqa_attn_return_logits_interface(
    q,
    kv_cache,
    kv_scale,
    topk_block_index,
    kv_block_size,
    weights,
    context_lens,
    block_tables,
):
    """
    Args:
        q: [B, seq_len, H, D] fp8
        kv_cache: [num_blocks, block_size, 1, D] fp8
        kv_scale: [num_blocks, block_size] fp32
        weights: [B, seq_len, H] or [B, H] bfloat16
    """
    batch, seq_len, heads, index_dim = q.shape
    topk = int(topk_block_index.shape[-1])
    paged_block_size = int(kv_cache.shape[1])

    if weights.ndim == 2:
        weights = weights.view(batch, seq_len, heads)

    block_H = heads if heads >= 16 else 16
    if heads < block_H:
        pad_size = block_H - heads
        weights = torch.nn.functional.pad(weights, (0, pad_size), value=0.0)

    # The kernel writes all topk*kv_block_size positions, using -inf out of bounds; no prefill is needed.
    logits = torch.empty(
        (batch, seq_len, topk * kv_block_size),
        device=q.device,
        dtype=torch.float32,
    )

    kernel = paged_block_sparse_mqa_attn_return_logits(
        paged_block_size=paged_block_size,
        kv_block_size=kv_block_size,
        topk=topk,
        heads=heads,
        index_dim=index_dim,
        dtype="float8_e4m3fn",
    )
    kernel(
        q,
        kv_cache,
        kv_scale,
        topk_block_index.to(torch.int32),
        logits,
        weights,
        context_lens if context_lens.dtype == torch.int32 else context_lens.to(torch.int32),
        block_tables if block_tables.dtype == torch.int32 else block_tables.to(torch.int32),
    )
    return logits


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def paged_mean_pooling(
    paged_block_size: int,
    pooling_block_size: int,
    max_num_pooling_blocks: int,
    dim: int,
    D_plus_4: int,
    threads=512,
    dtype="float8_e4m3fn",
    variant: str = "decode",
):
    # variant separates decode and verify JIT signatures so differing max_blocks
    # values use independent kernels with matching host-side stride constraints.
    # It does not affect kernel computation.
    _ = variant
    accum_dtype = T.float32
    index_dtype = T.int32

    num_blocks = T.dynamic("num_blocks")
    max_blocks = T.dynamic("max_blocks")
    batch = T.dynamic("batch")

    block_nbytes = paged_block_size * D_plus_4
    kv_cache_shape = [num_blocks, block_nbytes]
    block_tables_shape = [batch, max_blocks]
    context_lens_shape = [batch]
    blocked_k_shape = [batch, max_num_pooling_blocks, dim]

    # Byte offset of the FP32 scale segment within each physical block.
    scale_seg_off = paged_block_size * dim

    block_N = paged_block_size
    assert pooling_block_size % block_N == 0, "For simplicity, we require pooling_block_size to be a multiple of paged_block_size"

    @T.prim_func
    def paged_mean_pooling_kernel(
        KvCache: T.Tensor(kv_cache_shape, "uint8"), # type: ignore
        BlockTables: T.Tensor(block_tables_shape, index_dtype), # type: ignore
        ContextLens: T.Tensor(context_lens_shape, index_dtype), # type: ignore
        BlockedK: T.Tensor(blocked_k_shape, accum_dtype), # type: ignore
    ):
        with T.Kernel(batch, max_num_pooling_blocks, threads=threads) as (bx, by):
            b = bx
            seq_len = ContextLens[b]
            k_start = by * pooling_block_size
            k_end = T.min(k_start + pooling_block_size, seq_len)
            cur_pooling_block_size = k_end - k_start

            index_k_frag = T.alloc_fragment([block_N, dim], accum_dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            T.fill(acc, 0.0)

            if cur_pooling_block_size > 0:
                for b_i in T.Serial(T.ceildiv(cur_pooling_block_size, block_N)):
                    paged_block_s = k_start + b_i * block_N
                    T.fill(index_k_frag, 0.0)

                    if paged_block_s // paged_block_size < max_blocks:
                        paged_block_phys_id = BlockTables[b, paged_block_s // paged_block_size]
                        for n_i, d_i in T.Parallel(block_N, dim):
                            tl_block_idx = paged_block_s + n_i
                            if tl_block_idx < k_end:
                                # FP8 data segment: byte d_i of token n_i is at n_i*dim + d_i.
                                fp8_val = T.reinterpret(dtype, KvCache[paged_block_phys_id, n_i * dim + d_i])
                                # Scale segment: read four little-endian bytes at scale_seg_off + n_i*4.
                                sb0 = KvCache[paged_block_phys_id, scale_seg_off + n_i * 4 + 0].astype(T.uint32)
                                sb1 = KvCache[paged_block_phys_id, scale_seg_off + n_i * 4 + 1].astype(T.uint32)
                                sb2 = KvCache[paged_block_phys_id, scale_seg_off + n_i * 4 + 2].astype(T.uint32)
                                sb3 = KvCache[paged_block_phys_id, scale_seg_off + n_i * 4 + 3].astype(T.uint32)
                                scale_u32 = sb0 | (sb1 << 8) | (sb2 << 16) | (sb3 << 24)
                                scale_val = T.reinterpret("float32", scale_u32)
                                index_k_frag[n_i, d_i] = T.cast(fp8_val, accum_dtype) * scale_val

                    T.reduce_sum(index_k_frag, acc, dim=0, clear=False)

                for d_i in T.Parallel(dim):
                    acc[d_i] = acc[d_i] / T.cast(cur_pooling_block_size, accum_dtype)

            T.copy(acc, BlockedK[b, by, :])

    return paged_mean_pooling_kernel

def paged_mean_pooling_interface(
        kv_cache: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        k_block_size: int,
        dim: int,
        variant: str = "decode",
        threads: int = 512,
    ):
    """Pool paged FP8 keys with per-token FP32 scales.

    Args:
        kv_cache: [num_blocks, block_size, 1, D+4] uint8 wrapper. Each physical
            block stores FP8 data followed by little-endian FP32 scales.
        context_lens: [B] token counts.
        block_tables: [B, max_blocks] physical block indices.
        variant: JIT signature discriminator (decode / verify) that keeps
            kernels separate when max_blocks and host stride constraints differ.
    Returns:
        blocked_k: [B, max_num_pooling_blocks, D] FP32.
        num_pooling_blocks: [B] valid pooled-block counts.
    """
    num_blocks, paged_block_size, head, D_plus_4 = kv_cache.shape
    batch, max_blocks = block_tables.shape
    assert head == 1, "Only support head=1 for now"

    max_num_pooling_blocks = (max_blocks * paged_block_size - 1) // k_block_size + 1

    blocked_k = torch.empty((batch, max_num_pooling_blocks, dim), device=kv_cache.device, dtype=torch.float32)

    kernel = paged_mean_pooling(
        paged_block_size=paged_block_size,
        pooling_block_size=k_block_size,
        max_num_pooling_blocks=max_num_pooling_blocks,
        dim=dim,
        D_plus_4=D_plus_4,
        threads=threads,
        dtype="float8_e4m3fn",
        variant=variant,
    )
    kv_cache_flat = kv_cache.reshape(num_blocks, -1)
    # The host requires block_tables strides (shape[1], 1) and context_lens stride 1.
    # Row or column slices may not satisfy these exact constraints.
    # With batch == 1, is_contiguous() may accept a nonstandard first stride;
    # contiguous() then returns the same tensor, and clone() may preserve it.
    # Allocate standard row-major storage and copy by logical indices to enforce
    # the required strides independently of the source layout.
    if block_tables.stride(0) != block_tables.shape[1] or block_tables.stride(-1) != 1:
        _bt_new = torch.empty(
            block_tables.shape, dtype=block_tables.dtype, device=block_tables.device
        )
        _bt_new.copy_(block_tables)
        block_tables = _bt_new
    if context_lens.stride(-1) != 1:
        _ctx_new = torch.empty(
            context_lens.shape, dtype=context_lens.dtype, device=context_lens.device
        )
        _ctx_new.copy_(context_lens)
        context_lens = _ctx_new
    kernel(
        kv_cache_flat,
        block_tables,
        context_lens,
        blocked_k,
    )

    num_pooling_blocks = (context_lens + k_block_size - 1) // k_block_size
    return blocked_k, num_pooling_blocks


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def paged_mean_pooling_incremental(
    paged_block_size: int,
    pooling_block_size: int,
    incr_blocks: int,
    dim: int,
    D_plus_4: int,
    threads=512,
    dtype="float8_e4m3fn",
    variant: str = "decode",
):
    """Update pooled keys with a CUDA-graph-compatible incremental grid.

    The grid is (batch, incr_blocks), with a compile-time second dimension.
    CTA (b, by) updates pooling block StartBlock[b] + by, launching only
    batch * incr_blocks CTAs rather than covering the full cache capacity.
    Completed blocks can be retained in the persistent BlockedK cache while
    partial tail blocks are recomputed from their currently available tokens.
    Each mean depends only on the tokens covered by its pooling block.
    """
    _ = variant
    accum_dtype = T.float32
    index_dtype = T.int32

    num_blocks = T.dynamic("num_blocks")
    max_blocks = T.dynamic("max_blocks")
    batch = T.dynamic("batch")
    max_num_pooling_blocks = T.dynamic("max_num_pooling_blocks")

    block_nbytes = paged_block_size * D_plus_4
    kv_cache_shape = [num_blocks, block_nbytes]
    block_tables_shape = [batch, max_blocks]
    context_lens_shape = [batch]
    start_block_shape = [batch]
    blocked_k_shape = [batch, max_num_pooling_blocks, dim]

    scale_seg_off = paged_block_size * dim
    block_N = paged_block_size
    assert pooling_block_size % block_N == 0, (
        "pooling_block_size must be a multiple of paged_block_size"
    )

    @T.prim_func
    def paged_mean_pooling_incremental_kernel(
        KvCache: T.Tensor(kv_cache_shape, "uint8"),  # type: ignore
        BlockTables: T.Tensor(block_tables_shape, index_dtype),  # type: ignore
        ContextLens: T.Tensor(context_lens_shape, index_dtype),  # type: ignore
        StartBlock: T.Tensor(start_block_shape, index_dtype),  # type: ignore
        BlockedK: T.Tensor(blocked_k_shape, accum_dtype),  # type: ignore
    ):
        # Grid: (batch, incr_blocks), with a compile-time second dimension for CUDA graphs.
        with T.Kernel(batch, incr_blocks, threads=threads) as (bx, by):
            b = bx
            # This CTA updates the starting pooling-block index plus its local offset.
            pool_blk = StartBlock[b] + by
            seq_len = ContextLens[b]
            k_start = pool_blk * pooling_block_size
            k_end = T.min(k_start + pooling_block_size, seq_len)
            cur_pooling_block_size = k_end - k_start

            index_k_frag = T.alloc_fragment([block_N, dim], accum_dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            T.fill(acc, 0.0)

            # Skip padded CTAs beyond the sequence length or output cache capacity.
            if (cur_pooling_block_size > 0) and (pool_blk < max_num_pooling_blocks):
                for b_i in T.Serial(T.ceildiv(cur_pooling_block_size, block_N)):
                    paged_block_s = k_start + b_i * block_N
                    T.fill(index_k_frag, 0.0)

                    if paged_block_s // paged_block_size < max_blocks:
                        paged_block_phys_id = BlockTables[b, paged_block_s // paged_block_size]
                        for n_i, d_i in T.Parallel(block_N, dim):
                            tl_block_idx = paged_block_s + n_i
                            if tl_block_idx < k_end:
                                fp8_val = T.reinterpret(dtype, KvCache[paged_block_phys_id, n_i * dim + d_i])
                                sb0 = KvCache[paged_block_phys_id, scale_seg_off + n_i * 4 + 0].astype(T.uint32)
                                sb1 = KvCache[paged_block_phys_id, scale_seg_off + n_i * 4 + 1].astype(T.uint32)
                                sb2 = KvCache[paged_block_phys_id, scale_seg_off + n_i * 4 + 2].astype(T.uint32)
                                sb3 = KvCache[paged_block_phys_id, scale_seg_off + n_i * 4 + 3].astype(T.uint32)
                                scale_u32 = sb0 | (sb1 << 8) | (sb2 << 16) | (sb3 << 24)
                                scale_val = T.reinterpret("float32", scale_u32)
                                index_k_frag[n_i, d_i] = T.cast(fp8_val, accum_dtype) * scale_val

                    T.reduce_sum(index_k_frag, acc, dim=0, clear=False)

                for d_i in T.Parallel(dim):
                    acc[d_i] = acc[d_i] / T.cast(cur_pooling_block_size, accum_dtype)

                T.copy(acc, BlockedK[b, pool_blk, :])

    return paged_mean_pooling_incremental_kernel


def paged_mean_pooling_incremental_interface(
        kv_cache: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        k_block_size: int,
        dim: int,
        blocked_k: torch.Tensor,
        start_block: torch.Tensor,
        incr_blocks: int = 2,
        variant: str = "decode",
        threads: int = 512,
    ):
    """Update the persistent pooled-key cache with a fixed incremental grid.

    Args:
        kv_cache: [num_blocks, paged_block_size, 1, D+4] uint8 wrapper storing
            FP8 data followed by per-token FP32 scales in each physical block.
        context_lens: [B] int32 sequence lengths, including new tokens.
        block_tables: [B, max_blocks] int32 physical block indices.
        k_block_size: Number of tokens per pooling block.
        dim: FP8 data dimension D.
        blocked_k: [B, max_num_pooling_blocks, D] FP32 persistent output buffer,
            preallocated and reused across steps at a fixed address for CUDA
            graphs. Completed blocks outside the update range are preserved.
        start_block: [B] int32 index of each sequence's first block to update.
            CTA (b, by) updates start_block[b] + by.
        incr_blocks: Compile-time number of update CTAs per sequence; must
            cover all tail blocks requiring recomputation in this step.
    Returns:
        blocked_k (updated in place), num_pooling_blocks [B].
    """
    num_blocks, paged_block_size, head, D_plus_4 = kv_cache.shape
    batch, max_blocks = block_tables.shape
    assert head == 1, "Only support head=1 for now"

    max_num_pooling_blocks = blocked_k.shape[1]

    kernel = paged_mean_pooling_incremental(
        paged_block_size=paged_block_size,
        pooling_block_size=k_block_size,
        incr_blocks=incr_blocks,
        dim=dim,
        D_plus_4=D_plus_4,
        threads=threads,
        dtype="float8_e4m3fn",
        variant=variant,
    )
    kv_cache_flat = kv_cache.reshape(num_blocks, -1)

    # Normalize strides to the same host layout constraints as the full pooling interface.
    if block_tables.stride(0) != block_tables.shape[1] or block_tables.stride(-1) != 1:
        _bt_new = torch.empty(
            block_tables.shape, dtype=block_tables.dtype, device=block_tables.device
        )
        _bt_new.copy_(block_tables)
        block_tables = _bt_new
    if context_lens.stride(-1) != 1:
        _ctx_new = torch.empty(
            context_lens.shape, dtype=context_lens.dtype, device=context_lens.device
        )
        _ctx_new.copy_(context_lens)
        context_lens = _ctx_new
    if start_block.stride(-1) != 1:
        _sb_new = torch.empty(
            start_block.shape, dtype=start_block.dtype, device=start_block.device
        )
        _sb_new.copy_(start_block)
        start_block = _sb_new

    kernel(
        kv_cache_flat,
        block_tables,
        context_lens,
        start_block,
        blocked_k,
    )

    num_pooling_blocks = (context_lens + k_block_size - 1) // k_block_size
    return blocked_k, num_pooling_blocks


@tilelang.jit(
        pass_configs=global_pass_configs
    )
def block_mqa_attn_return_logits(
    heads,
    index_dim,
    block_N=128,
    num_stages=1,
    threads=512,
    block_Q=None,
    dtype="bfloat16",
    relu_neg=False,
):
    """Score prefill heads using BF16 queries and pooled keys.

    Convert FP32 pooled keys to BF16 rather than requantizing them to FP8.
    Scores sum the absolute values of weighted, rectified dot products.
    relu_neg selects the scoring rule:
        False (default): use relu(+y) = max(s, 0) for positive-axis scores.
        True: use relu(-y) = max(-s, 0) to rank heads by the magnitude of
            their weighted ReLU-decomposition residuals.
    """
    if block_Q is None:
        block_Q = 128 // heads
    accum_dtype = "float32"
    index_dtype = "int32"

    seq_len = T.dynamic("seq_len")
    seq_len_blocked_kv = T.dynamic("seq_len_blocked_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_blocked_kv, index_dim]

    @T.prim_func
    def block_mqa_attn_return_logits_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexBlockedK: T.Tensor(index_k_shape, dtype),  # type: ignore
        QScores: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], "bfloat16"),  # type: ignore
        CuSeqLenBlockedKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenBlockedKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)

            s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, block_Q, heads))
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)
            q_scores = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s_min = T.alloc_var(index_dtype)
            cu_k_e_max = T.alloc_var(index_dtype)

            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenBlockedKS[seq_len_i + bq_i], seq_len_blocked_kv))
            for bq_i in T.serial(block_Q):
                cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenBlockedKE[seq_len_i + bq_i], seq_len_blocked_kv))

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for nbn_i in T.Pipelined(T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages):
                ks_i = cu_k_s_min + nbn_i * block_N
                T.copy(IndexBlockedK[ks_i, 0], index_k_shared)

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    if (ks_i + bn_i) < CuSeqLenBlockedKS[seq_len_i + bq_i] or (ks_i + bn_i) >= CuSeqLenBlockedKE[seq_len_i + bq_i]:
                        s_reshaped[bn_i, bq_i, h_i] = T.cast(0, accum_dtype)
                    elif relu_neg:
                        # Score the relu(-y) residual using max(-s, 0).
                        s_reshaped[bn_i, bq_i, h_i] = (T.max(-s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i])
                    else:
                        s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]) 

                T.reduce_abssum(s_reshaped, q_scores, dim=0)

                for bq_i, h_i in T.Parallel(block_Q, heads):
                    QScores[seq_len_i + bq_i, h_i] += q_scores[bq_i, h_i]

    return block_mqa_attn_return_logits_kernel


def block_mqa_attn_return_logits_interface(q, blocked_kv, kv_block_size, weights, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke, relu_neg=False):
    """Return FP32 prefill head scores using BF16 coarse screening.

    Args:
        q: [seq_len, heads, index_dim] FP8, converted to BF16 by the interface.
        blocked_kv: [seq_len_blocked_kv, index_dim] FP32, converted to BF16.
        weights: [seq_len, heads] BF16.
        cu_seqlen_blocked_ks, cu_seqlen_blocked_ke: [seq_len] int32 bounds.
        relu_neg: True scores weighted relu(-y) residual magnitudes;
            False (default) scores weighted relu(+y) magnitudes.
    Returns:
        q_head_score: [seq_len, heads] FP32.

    Pooled keys are not requantized to FP8 before screening.
    """
    seq_len, heads, index_dim = q.shape

    q_bf16 = q.to(torch.bfloat16)
    blocked_kv_bf16 = blocked_kv.bfloat16()

    block_mqa_attn_return_logits_kernel = block_mqa_attn_return_logits(heads=heads, index_dim=index_dim, dtype="bfloat16", relu_neg=relu_neg)
    q_head_score = torch.zeros([seq_len, heads], device=q.device, dtype=torch.float32)
    block_mqa_attn_return_logits_kernel(
        q_bf16.view(seq_len * heads, index_dim),
        blocked_kv_bf16,
        q_head_score,
        weights,
        cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke,
    )

    return q_head_score


@tilelang.jit(
    pass_configs=global_pass_configs,
)
def block_mean_pooling(
    pooling_block_size: int, 
    dim: int,
    block_N: int=64,
    num_stages=1,
    threads=256,
    dtype="float8_e4m3fn",
):
    """Pool FP8 input by casting to FP32 and applying scales inside the kernel.

    Avoids materializing a full FP32 input tensor of seq_len_k * dim * 4 bytes
    and the corresponding interface-level conversion.
    """
    accum_dtype = T.float32
    index_dtype = T.int32
    
    seq_len_k = T.dynamic("seq_len_k")
    max_num_pooling_blocks = T.dynamic("max_num_pooling_blocks")
    k_size = [seq_len_k, dim]
    k_scales_size = [seq_len_k]
    blocked_k_size = [max_num_pooling_blocks, dim]

    @T.prim_func
    def block_mean_pooling_kernel(
        K: T.Tensor(k_size, dtype=dtype), # type: ignore  [fp8 input]
        KScales: T.Tensor(k_scales_size, dtype=accum_dtype), # type: ignore
        BlockedK: T.Tensor(blocked_k_size, dtype=accum_dtype), # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len_k, pooling_block_size), threads=threads) as bx:
            index_k = T.alloc_fragment([block_N, dim], accum_dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            T.fill(acc, 0.0)

            k_start = bx * pooling_block_size
            k_end = T.min(k_start + pooling_block_size, seq_len_k)
            cur_pooling_block_size = k_end - k_start

            for b_i in T.serial(T.ceildiv(cur_pooling_block_size, block_N)):
                T.fill(index_k, 0.0)

                tl_block_s = k_start + b_i * block_N
                tl_block_e = T.min(k_start + (b_i + 1) * block_N, k_end)
                cur_tl_block_size = tl_block_e - tl_block_s

                for n_i, d_i in T.Parallel(block_N, dim):
                    if n_i < cur_tl_block_size:
                        index_k[n_i, d_i] = T.cast(K[tl_block_s + n_i, d_i], accum_dtype) * KScales[tl_block_s + n_i]

                T.reduce_sum(index_k, acc, dim=0, clear=False)

            for d_i in T.parallel(dim):
                acc[d_i] = acc[d_i] / T.cast(cur_pooling_block_size, accum_dtype)

            T.copy(acc, BlockedK[bx, :])

    return block_mean_pooling_kernel

def block_mean_pooling_interface(k_fp8, k_scales, k_block_size):
    """
    Args:
        k_fp8: [seq_len_k, dim], dtype float8_e4m3fn
        k_scales: [seq_len_k], dtype float32
        k_block_size: int
    Returns:
        blocked_k_fp32: [max_num_pooling_blocks, dim], dtype float32
    Return FP32 pooled keys without FP8 requantization or its elementwise
    operations. Decode screening casts keys to BF16 inside the kernel;
    prefill screening casts them to BF16 at the interface.
    """
    seq_len_k, d = k_fp8.shape
    max_num_pooling_blocks = (seq_len_k + k_block_size - 1) // k_block_size

    blocked_k_fp32 = torch.empty((max_num_pooling_blocks, d), device=k_fp8.device, dtype=torch.float32)
    kernel = block_mean_pooling(
        pooling_block_size=k_block_size,
        dim=d,
        dtype="float8_e4m3fn",
    )
    kernel(
        k_fp8,
        k_scales,
        blocked_k_fp32,
    )

    return blocked_k_fp32
