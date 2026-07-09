import torch
import tilelang
from tilelang import language as T

global_pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        # tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        # tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def batch_decode_block_mqa_attn_return_logits(
    heads: int,
    index_dim: int,
    block_N: int = 64,
    block_H: int = 64,
    num_stages: int = 3,
    threads: int = 256,
    dtype: str = "bfloat16",
):
    """
    Decode 专用：q_len 固定为 1，不在 Q 维分块，只在 H 和 Nb 上分块。

    Shapes:
      Q:          [B, 1, H, D]
      BlockedK:   [B, Nb, D]
      Logits:     [B, 1, Nb] fp32
      Weights:    [B, 1, H]
      ContextLens:[B]  (有效 Nb)
    """
    accum_dtype = T.float32
    index_dtype = T.int32

    batch = T.dynamic("batch")
    nb = T.dynamic("seq_len_blocked_kv")

    q_shape = [batch, heads, index_dim]
    k_shape = [batch, nb, index_dim]
    w_shape = [batch, heads]

    # padding 到 16 对齐，避免 gemm 列维过小/不合法
    block_H_pad = T.ceildiv(block_H, 16) * 16
    assert block_H_pad == heads

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        BlockedK: T.Tensor(k_shape, dtype),  # type: ignore
        QScores: T.Tensor(w_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor(w_shape, dtype),  # type: ignore
        ContextLens: T.Tensor([batch], index_dtype),  # type: ignore
    ):
        with T.Kernel(batch, 1, threads=threads) as (bx, by):
            # shared tiles
            k_shared = T.alloc_shared([block_N, index_dim], dtype)
            q_shared = T.alloc_shared([block_H_pad, index_dim], dtype)
            q_scores = T.alloc_fragment([block_H_pad], accum_dtype)

            # fragments
            s = T.alloc_fragment([block_N, block_H_pad], accum_dtype)
            w = T.alloc_fragment([block_H_pad], accum_dtype)

            # valid kv range
            k_e = T.min(ContextLens[bx], nb)
            T.copy(Q[bx, 0, 0], q_shared)
            T.copy(Weights[bx, 0], w)

            for k_i in T.Pipelined(T.ceildiv(nb, block_N), num_stages=num_stages):
                k_start = k_i * block_N
                T.copy(BlockedK[bx, k_start, 0], k_shared)

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
                        s[kn_i, hn_i] = T.max(s[kn_i, hn_i], 0) * w[hn_i]
                    else:
                        s[kn_i, hn_i] = T.cast(0, accum_dtype)

                T.reduce_abssum(s, q_scores, dim=0)
                for hn_i in T.Parallel(block_H_pad):
                    QScores[bx, hn_i] += q_scores[hn_i]

            # T.copy(q_scores, QScores[bx, 0])

    return kernel


def batch_block_mqa_attn_return_logits_interface(
    q: torch.Tensor,
    blocked_kv: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    kv_block_size: int,
    clean_logits: bool = True,
    force_maintain: bool = True,
    dtype: str = "bfloat16",
    block_N: int = 64,
):
    """
    Decode 接口：
      q:          [B, 1, H, D]
      blocked_kv: [B, Nb, D]
      weights:    [B,1,H]
      context_lens:[B] (有效 Nb)
    Return:
      logits: [B, Nb] fp32
    """

    assert len(q.shape) == 4
    B, seq_len_q, H, D = q.shape
    B, seq_len_kv, D = blocked_kv.shape

    assert seq_len_q == 1, "decode expects q_len=1"

    q = q.squeeze(1)
    weights = weights.squeeze(1)

    q_head_score = torch.zeros(weights.shape, device=q.device, dtype=torch.float32)

    kernel = batch_decode_block_mqa_attn_return_logits(
        heads=H,
        index_dim=D,
        block_N=block_N,
        block_H=H,
        dtype=dtype,
    )
    kernel(
        q,
        blocked_kv,
        q_head_score,
        weights,
        context_lens.to(torch.int32),
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
    dtype="bfloat16",
):
    accum_dtype = T.float32
    index_dtype = T.int32

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    max_blocks = T.dynamic("max_blocks")
    num_phys_blocks = T.dynamic("num_phys_blocks")

    index_q_shape = [batch, seq_len, heads, index_dim]
    kv_cache_shape = [num_phys_blocks, paged_block_size, 1, index_dim]
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
        TopKBlockIndex: T.Tensor([batch, seq_len, topk], index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor(weights_shape, dtype),  # type: ignore
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

            cu_k_s_min = T.cast(0, index_dtype)
            cu_k_e_max = ContextLens[b]

            # T.copy(IndexQ[b, seq_len_i, :, :], index_q_shared)
            # T.copy(Weights[b, seq_len_i, :], weights)
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
                        T.copy(KvCache[phys, :, 0, :], index_k_shared)

                    T.gemm(
                        index_k_shared,
                        index_q_shared,
                        s,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    for bn_i, bq_i, h_i in T.Parallel(block_N, 1, block_H):
                        s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i])

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
        TopKBlockIndex: T.Tensor([batch, seq_len, topk], index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor(weights_shape, dtype),  # type: ignore
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

            cu_k_s_min = T.cast(0, index_dtype)
            cu_k_e_max = ContextLens[b]

            # T.copy(IndexQ[b, seq_len_i, :, :], index_q_shared)
            # T.copy(Weights[b, seq_len_i, :], weights)
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
                    T.copy(KvCache[phys, :, 0, :], index_k_shared)

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, 1, block_H):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i])

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
    topk_block_index,
    kv_block_size,
    weights,
    context_lens,
    block_tables,
    dtype="bfloat16",
):
    batch, seq_len, heads, index_dim = q.shape
    topk = int(topk_block_index.shape[-1])
    paged_block_size = int(kv_cache.shape[1])

    if weights.ndim == 2:
        weights = weights.view(batch, seq_len, heads)

    # Pad weights to block_H to avoid tilelang fragment layout issues
    block_H = heads if heads >= 16 else 16
    if heads < block_H:
        pad_size = block_H - heads
        weights = torch.nn.functional.pad(weights, (0, pad_size), value=0.0)

    logits = torch.full(
        (batch, seq_len, topk * kv_block_size),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )

    kernel = paged_block_sparse_mqa_attn_return_logits(
        paged_block_size=paged_block_size,
        kv_block_size=kv_block_size,
        topk=topk,
        heads=heads,
        index_dim=index_dim,
        dtype=dtype,
    )
    kernel(
        q,
        kv_cache,
        topk_block_index.to(torch.int32),
        logits,
        weights,
        context_lens.to(torch.int32),
        block_tables.to(torch.int32),
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
    num_stages=1,
    threads=256,
    dtype="bfloat16",
):
    accum_dtype = T.float32
    index_dtype = T.int32

    num_blocks = T.dynamic("num_blocks")
    max_blocks = T.dynamic("max_blocks")
    batch = T.dynamic("batch")

    kv_cache_shape = [num_blocks, paged_block_size, 1, dim]
    block_tables_shape = [batch, max_blocks]
    context_lens_shape = [batch]
    blocked_k_shape = [batch, max_num_pooling_blocks, dim]

    block_N = paged_block_size
    assert pooling_block_size % block_N == 0, "For simplicity, we require pooling_block_size to be a multiple of paged_block_size"

    @T.prim_func
    def paged_mean_pooling_kernel(
        KvCache: T.Tensor(kv_cache_shape, dtype), # type: ignore
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

            index_k_shared = T.alloc_fragment([block_N, dim], dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            T.fill(acc, 0.0)

            if cur_pooling_block_size > 0:
                for b_i in T.Serial(T.ceildiv(cur_pooling_block_size, block_N)):
                    paged_block_s = k_start + b_i * block_N
                    T.fill(index_k_shared, 0.0)

                    if paged_block_s // paged_block_size < max_blocks:
                        paged_block_phys_id = BlockTables[b, paged_block_s // paged_block_size]
                        T.copy(KvCache[paged_block_phys_id, :, 0, :], index_k_shared)

                    for n_i, d_i in T.Parallel(block_N, dim):
                        tl_block_idx = paged_block_s + n_i
                        if tl_block_idx >= k_end:
                            index_k_shared[n_i, d_i] = T.cast(0, accum_dtype)

                    T.reduce_sum(index_k_shared, acc, dim=0, clear=False)

                for d_i in T.Parallel(dim):
                    acc[d_i] = acc[d_i] / T.cast(cur_pooling_block_size, accum_dtype)

            T.copy(acc, BlockedK[b, by, :])

    return paged_mean_pooling_kernel

def paged_mean_pooling_interface(
        kv_cache: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        k_block_size: int,
    ):
    """
    Args:
        kv_cache: [num_blocks, block_size, 1, D]
        context_lens: [B]
        block_tables: [B, max_blocks]  (逻辑 paged block -> 物理 block)
    Returns:
        blocked_k: [B, max_num_pooling_blocks, D]  (已 padding)
        num_pooling_blocks: [B]  (每个样本真实的 pooling block 数，用于 mask)
    """
    num_blocks, paged_block_size, head, dim = kv_cache.shape
    batch, max_blocks = block_tables.shape
    assert head == 1, "Only support head=1 for now"

    # TODO calculation of max_num_pooling_blocks
    # max_num_pooling_blocks = ((context_lens.max().item() + k_block_size - 1) // k_block_size)
    max_num_pooling_blocks = (max_blocks * paged_block_size - 1) // k_block_size + 1

    blocked_k = torch.empty((batch, max_num_pooling_blocks, dim), device=kv_cache.device, dtype=torch.float32)

    kernel = paged_mean_pooling(paged_block_size=paged_block_size, pooling_block_size=k_block_size, max_num_pooling_blocks=max_num_pooling_blocks, dim=dim)
    kernel(
        kv_cache,
        block_tables,
        context_lens,
        blocked_k,
    )

    blocked_k = blocked_k.to(kv_cache.dtype)
    num_pooling_blocks = (context_lens + k_block_size - 1) // k_block_size
    return blocked_k, num_pooling_blocks

@tilelang.jit(
        pass_configs=global_pass_configs
    )
def block_mqa_attn_return_logits(
    heads,
    index_dim,
    block_N=128,
    num_stages=3,
    threads=256,
    block_Q=None,
    dtype="bfloat16",
):
    if block_Q is None:
        block_Q = 128 // heads
    accum_dtype = "float32"
    index_dtype = "int32"

    seq_len = T.dynamic("seq_len")
    seq_len_blocked_kv = T.dynamic("seq_len_blocked_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_blocked_kv, index_dim]
    logits_shape = [seq_len, seq_len_blocked_kv]

    @T.prim_func
    def block_mqa_attn_return_logits_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexBlockedK: T.Tensor(index_k_shape, dtype),  # type: ignore
        QScores: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], dtype),  # type: ignore
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
                    else:
                        s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i])

                # T.reduce_sum(s_reshaped, q_scores, dim=0, clear=False)
                T.reduce_abssum(s_reshaped, q_scores, dim=0)

                for bq_i,h_i in T.Parallel(block_Q, heads):
                    QScores[seq_len_i + bq_i, h_i] += q_scores[bq_i, h_i]

            # T.copy(q_scores, QScores[seq_len_i, 0])

    return block_mqa_attn_return_logits_kernel


@tilelang.jit
def clean_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx < cu_k_s or idx >= cu_k_e:
                        Logits[bx, idx] = -T.infinity(dtype)

    return clean_logits_kernel

@tilelang.jit
def force_maintain_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def force_maintain_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx == cu_k_s or idx == cu_k_e - 1:
                        Logits[bx, idx] = T.infinity(dtype)

    return force_maintain_logits_kernel

@tilelang.jit
def clean_and_maintain_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_and_maintain_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx == cu_k_s or idx == cu_k_e - 1:
                        Logits[bx, idx] = T.infinity(dtype)
                    if idx < cu_k_s or idx >= cu_k_e:
                        Logits[bx, idx] = -T.infinity(dtype)

    return clean_and_maintain_logits_kernel

def block_mqa_attn_return_logits_interface(q, blocked_kv, kv_block_size, weights, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke, clean_logits=True, force_maintain=True, dtype="bfloat16"):
    seq_len, heads, index_dim = q.shape
    seq_len_blocked_kv = blocked_kv.shape[0]

    block_mqa_attn_return_logits_kernel = block_mqa_attn_return_logits(heads=heads, index_dim=index_dim, dtype=dtype)
    q_head_score = torch.zeros([seq_len, heads], device=q.device, dtype=torch.float32)
    block_mqa_attn_return_logits_kernel(
        q.view(seq_len * heads, index_dim),
        blocked_kv,
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
    threads=128,
    dtype="bfloat16",
):
    accum_dtype = T.float32

    seq_len_k = T.dynamic("seq_len_k")
    max_num_pooling_blocks = T.dynamic("max_num_pooling_blocks")
    k_size = [seq_len_k, dim]
    blocked_k_size = [max_num_pooling_blocks, dim]

    @T.prim_func
    def block_mean_pooling_kernel(
        K: T.Tensor(k_size, dtype=dtype), # type: ignore
        BlockedK: T.Tensor(blocked_k_size, dtype=accum_dtype), # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len_k, pooling_block_size), threads=threads) as bx:
            index_k = T.alloc_fragment([block_N, dim], dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            T.fill(acc, 0.0)

            k_start = bx * pooling_block_size
            k_end = T.min(k_start + pooling_block_size, seq_len_k)
            cur_pooling_block_size = k_end - k_start

            for b_i in T.serial(T.ceildiv(cur_pooling_block_size, block_N)):
                T.fill(index_k, 0.0)

                tl_block_s = k_start + b_i * block_N
                tl_block_e = T.min(k_start + (b_i + 1) * block_N, k_end)
                T.copy(K[tl_block_s:tl_block_s + block_N, :], index_k)

                cur_tl_block_size = tl_block_e - tl_block_s
                for n_i in T.parallel(block_N):
                    for d_i in T.parallel(dim):
                        if n_i >= cur_tl_block_size:
                            index_k[n_i, d_i] = T.cast(0, accum_dtype)

                T.reduce_sum(index_k, acc, dim=0, clear=False)

            for d_i in T.parallel(dim):
                acc[d_i] = acc[d_i] / T.cast(cur_pooling_block_size, accum_dtype)

            T.copy(acc, BlockedK[bx, :])

    return block_mean_pooling_kernel

def block_mean_pooling_interface(k, k_block_size):
    seq_len_k, d = k.shape
    max_num_pooling_blocks = (seq_len_k + k_block_size - 1) // k_block_size

    blocked_k = torch.empty((max_num_pooling_blocks, d), device=k.device, dtype=torch.float32)
    kernel = block_mean_pooling(
        pooling_block_size=k_block_size,
        dim=d,
    )
    kernel(
        k,
        blocked_k,
    )
    blocked_k = blocked_k.to(k.dtype)

    return blocked_k

@tilelang.jit(
    pass_configs=global_pass_configs,
)
def block_sparse_mqa_attn_return_logits(
    kv_block_size,
    heads,
    index_dim,
    block_N=128,
    block_Q=None,
    num_stages=1,
    threads=256,
    dtype="bfloat16",
):
    accum_dtype = T.float32
    index_dtype = T.int32

    if block_Q is None:
        block_Q = 128 // heads # 多少q放在一个block里

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    topk = T.dynamic("topk")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    logits_shape = [seq_len, topk * kv_block_size]

    # TODO check padded H in sparse_mla_fwd
    # does it matter here?
    block_H = heads if heads >= 16 else 16
    block_N = T.min(block_N, kv_block_size)
    assert kv_block_size % block_N == 0, "block_N must divide kv_block_size"

    @T.prim_func
    def block_sparse_mqa_attn_return_logits_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        TopKBlockIndex: T.Tensor([seq_len, topk], index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            index_q_shared = T.alloc_shared([block_H, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, block_H], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, 1, block_H))
            logits = T.alloc_fragment([block_N, 1], accum_dtype)
            weights = T.alloc_shared([1, block_H], accum_dtype)

            seq_len_i = bx

            cu_k_s_min = CuSeqLenKS[seq_len_i]
            cu_k_e_max = CuSeqLenKE[seq_len_i]

            for h_i, dim in T.Parallel(block_H, index_dim):
                if h_i < heads:
                    index_q_shared[h_i, dim] = IndexQ[seq_len_i * heads + h_i, dim]
                else:
                    index_q_shared[h_i, dim] = 0

            for h_i in T.Parallel(block_H):
                if h_i < heads:
                    weights[0, h_i] = Weights[seq_len_i, h_i]
                else:
                    weights[0, h_i] = 0

            for n_i in T.serial(topk):
                topk_block_id = TopKBlockIndex[seq_len_i, n_i]
                block_s = topk_block_id * kv_block_size
                for b_i in T.Pipelined(kv_block_size // block_N, num_stages=num_stages):
                    block_s_i = block_s + b_i * block_N

                    T.copy(IndexK[block_s_i, 0], index_k_shared)

                    T.gemm(
                        index_k_shared,
                        index_q_shared,
                        s,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    for bn_i, bq_i, h_i in T.Parallel(block_N, 1, heads):
                        s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i])

                    T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                    for i_i in T.Parallel(block_N):
                        k_i = block_s_i + i_i
                        if k_i < cu_k_s_min or k_i >= cu_k_e_max:
                            logits[i_i, 0] = -T.infinity(accum_dtype)

                    for bn_i in T.Parallel(block_N):
                        Logits[seq_len_i, n_i * kv_block_size + b_i * block_N + bn_i] = logits[bn_i, 0]

    @T.prim_func
    def block_sparse_mqa_attn_return_logits_kernel_for_small_pooling_size(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        TopKBlockIndex: T.Tensor([seq_len, topk], index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            index_q_shared = T.alloc_shared([block_H, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, block_H], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, 1, block_H))
            logits = T.alloc_fragment([block_N, 1], accum_dtype)
            weights = T.alloc_shared([1, block_H], accum_dtype)

            seq_len_i = bx

            cu_k_s_min = CuSeqLenKS[seq_len_i]
            cu_k_e_max = CuSeqLenKE[seq_len_i]

            for h_i, dim in T.Parallel(block_H, index_dim):
                if h_i < heads:
                    index_q_shared[h_i, dim] = IndexQ[seq_len_i * heads + h_i, dim]
                else:
                    index_q_shared[h_i, dim] = 0

            for h_i in T.Parallel(heads):
                if h_i < heads:
                    weights[0, h_i] = Weights[seq_len_i, h_i]
                else:
                    weights[0, h_i] = 0

            for n_i in T.Pipelined(topk, num_stages=num_stages):
                topk_block_id = TopKBlockIndex[seq_len_i, n_i]
                block_s_i = topk_block_id * kv_block_size

                T.copy(IndexK[block_s_i, 0], index_k_shared)

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, 1, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i])

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for i_i in T.Parallel(block_N):
                    k_i = block_s_i + i_i
                    if k_i < cu_k_s_min or k_i >= cu_k_e_max:
                        logits[i_i, 0] = -T.infinity(accum_dtype)

                for bn_i in T.Parallel(block_N):
                    Logits[seq_len_i, n_i * kv_block_size + bn_i] = logits[bn_i, 0]

    if kv_block_size == block_N:
        return block_sparse_mqa_attn_return_logits_kernel_for_small_pooling_size
    else:
        return block_sparse_mqa_attn_return_logits_kernel

def block_sparse_mqa_attn_return_logits_interface(q, kv, weights, topk_block_index, kv_block_size, cu_seqlen_ks, cu_seqlen_ke, dtype="bfloat16"):
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]
    topk = topk_block_index.shape[-1]

    block_sparse_mqa_attn_return_logits_kernel = block_sparse_mqa_attn_return_logits(heads=heads, index_dim=index_dim, kv_block_size=kv_block_size)
    logits = torch.full([seq_len, topk * kv_block_size], fill_value=float("-inf"), device=q.device, dtype=torch.float32)
    block_sparse_mqa_attn_return_logits_kernel(
        q.view(seq_len * heads, index_dim),
        kv,
        topk_block_index,
        logits,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    return logits