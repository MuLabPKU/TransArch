import torch
import tilelang
from tilelang import language as T
from block_ops import (block_mean_pooling_interface,
                        block_mqa_attn_return_logits_interface,
                        block_sparse_mqa_attn_return_logits_interface,
                        paged_mean_pooling_interface,
                        paged_block_sparse_mqa_attn_return_logits_interface,
                        batch_block_mqa_attn_return_logits_interface)


def calc_diff(x, y):
    """Calculate cosine dissimilarity (1 - cosine_similarity)"""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

def calc_diff_with_window_vectorized(x, y, cu_seqlen_ks, cu_seqlen_ke):
    """
    向量化版本：根据每行的窗口范围创建掩码
    """
    N = x.shape[0]

    # 生成列索引矩阵
    col_indices = torch.arange(N, device=x.device).unsqueeze(0).expand(N, N)

    # 创建掩码：每行的列索引在 [cu_seqlen_ks[i], cu_seqlen_ke[i]) 范围内
    row_indices = torch.arange(N, device=x.device)
    start_indices = cu_seqlen_ks[row_indices].unsqueeze(1)
    end_indices = cu_seqlen_ke[row_indices].unsqueeze(1)

    mask = (col_indices >= start_indices) & (col_indices < end_indices)

    # 使用 torch.where 避免 -inf * 0
    x_masked = torch.where(mask, x, torch.tensor(0.0, device=x.device))
    y_masked = torch.where(mask, y, torch.tensor(0.0, device=y.device))

    # 计算余弦差异
    x_masked, y_masked = x_masked.double(), y_masked.double()

    denominator = (x_masked * x_masked + y_masked * y_masked).sum()
    if denominator == 0:
        return 1.0

    sim = 2 * (x_masked * y_masked).sum() / denominator

    return 1 - sim

global_pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        # tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        # tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }

@tilelang.jit(
    pass_configs=global_pass_configs
)
def paged_mqa_sparse_head_return_logits(
    paged_block_size,
    max_model_len,
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
    logits_shape = [batch, seq_len, max_model_len]

    block_H = heads if heads >= 16 else 16
    block_N = paged_block_size
    weights_shape = [batch, seq_len, block_H]

    assert block_N > 0, "block_N must be positive"
    assert paged_block_size >= block_N and paged_block_size % block_N == 0, "block_N must divide paged_block_size"
    assert paged_block_size == block_N, "for simplicity we require paged_block_size == block_N in this kernel"

    @T.prim_func
    def paged_mqa_sparse_head_return_logits_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        KvCache: T.Tensor(kv_cache_shape, dtype),  # type: ignore
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

            cu_s_min = T.cast(0, index_dtype)
            cu_e_max = ContextLens[b]
            cu_len = cu_e_max - cu_s_min

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

            for b_i in T.Pipelined(T.ceildiv(cu_len ,block_N), num_stages=num_stages):
                block_s_i = b_i * block_N

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
                    if h_i < heads:
                        s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i])
                    else:
                        s_reshaped[bn_i, bq_i, h_i] = 0

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for i_i in T.Parallel(block_N):
                    k_i = block_s_i + i_i
                    p = k_i // paged_block_size # b_i
                    if (k_i < cu_s_min) or (k_i >= cu_e_max) or (p < 0) or (p >= max_blocks):
                        logits[i_i, 0] = -T.infinity(accum_dtype)

                for bn_i in T.Parallel(block_N):
                    Logits[b, seq_len_i, b_i * block_N + bn_i] = logits[bn_i, 0]

    return paged_mqa_sparse_head_return_logits_kernel

def paged_mqa_sparse_head_return_logits_interface(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    max_model_len,
    dtype="bfloat16",
):
    batch, seq_len, heads, index_dim = q.shape
    paged_block_size = int(kv_cache.shape[1])

    if weights.ndim == 2:
        weights = weights.view(batch, seq_len, heads)

    # Pad weights to block_H to avoid tilelang fragment layout issues
    block_H = heads if heads >= 16 else 16
    if heads < block_H:
        pad_size = block_H - heads
        weights = torch.nn.functional.pad(weights, (0, pad_size), value=0.0)

    logits = torch.full([batch, seq_len, max_model_len], float('-inf'), device=q.device, dtype=torch.float32)

    kernel = paged_mqa_sparse_head_return_logits(
        paged_block_size=paged_block_size,
        max_model_len=max_model_len,
        heads=heads,
        index_dim=index_dim,
        dtype=dtype,
    )
    kernel(
        q,
        kv_cache,
        logits,
        weights,
        context_lens.to(torch.int32),
        block_tables.to(torch.int32),
    )
    return logits

@tilelang.jit(
    pass_configs=global_pass_configs
)
def paged_mqa_attn_return_logits(
    paged_block_size,
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
    stage_1_topk = T.dynamic("stage_1_topk")
    padded_topk = T.dynamic("padded_topk")
    max_blocks = T.dynamic("max_blocks")
    num_phys_blocks = T.dynamic("num_phys_blocks")

    index_q_shape = [batch, seq_len, heads, index_dim]
    kv_cache_shape = [num_phys_blocks, paged_block_size, 1, index_dim]
    logits_shape = [batch, seq_len, padded_topk]
    weights_shape = [batch, seq_len, heads]
    sparse_topk_indices_shape = [batch, seq_len, stage_1_topk]

    H_per_block = heads
    block_N = paged_block_size
    assert block_N > 0, "block_N must be positive"
    assert paged_block_size >= block_N and paged_block_size % block_N == 0, "block_N must divide paged_block_size"
    assert paged_block_size == block_N, "for simplicity we require paged_block_size == block_N in this kernel"

    @T.prim_func
    def paged_mqa_attn_return_logits_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        KvCache: T.Tensor(kv_cache_shape, dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor(weights_shape, dtype),  # type: ignore
        SparseTopKIndices: T.Tensor(sparse_topk_indices_shape, index_dtype),  # type: ignore
        ContextLens: T.Tensor([batch], index_dtype),  # type: ignore
        BlockTables: T.Tensor([batch, max_blocks], index_dtype),  # type: ignore
    ):
        with T.Kernel(batch, seq_len, threads=threads) as (bx, by):
            b = bx
            seq_len_i = by

            index_q_shared = T.alloc_shared([H_per_block, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, H_per_block], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, H_per_block // heads, heads))
            logits = T.alloc_fragment([block_N, H_per_block // heads], accum_dtype)
            weights = T.alloc_fragment([H_per_block // heads, heads], accum_dtype)

            cu_k_s_min = T.cast(0, index_dtype)
            cu_k_e_max = ContextLens[b]

            T.copy(IndexQ[b, seq_len_i, :, :], index_q_shared)
            T.copy(Weights[b, seq_len_i, :], weights)

            for b_i in T.Pipelined(T.ceildiv(stage_1_topk, block_N), num_stages=num_stages):
                block_s_i = b_i * block_N

                for i, j in T.Parallel(block_N, index_dim):
                    idx = SparseTopKIndices[b, seq_len_i, block_s_i + i]
                    phys = BlockTables[b, idx // paged_block_size]
                    index_k_shared[i, j] = KvCache[phys, idx % paged_block_size, 0, j]

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, H_per_block // heads, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i])

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for i_i in T.Parallel(block_N):
                    idx = SparseTopKIndices[b, seq_len_i, block_s_i + i_i]
                    p = idx // paged_block_size # b_i
                    if (idx < cu_k_s_min) or (idx >= cu_k_e_max) or (p < 0) or (p >= max_blocks):
                        logits[i_i, 0] = -T.infinity(accum_dtype)

                for bn_i in T.Parallel(block_N):
                    Logits[b, seq_len_i, block_s_i + bn_i] = logits[bn_i, 0]

    return paged_mqa_attn_return_logits_kernel

def paged_mqa_attn_return_logits_interface(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    sparse_topk_indices,
    dtype="bfloat16",
):
    batch, seq_len, heads, index_dim = q.shape
    paged_block_size = int(kv_cache.shape[1])
    stage_1_topk = sparse_topk_indices.shape[-1]

    block_size = 128
    padded_topk = ((stage_1_topk + block_size - 1) // block_size) * block_size
    padded_size = padded_topk - stage_1_topk

    if padded_size > 0:
        pad_value = -1
        padded_sparse_topk_indices = torch.cat([sparse_topk_indices,
                                                torch.full((batch, seq_len, padded_size), pad_value,
                                                        dtype=sparse_topk_indices.dtype, device=sparse_topk_indices.device)], dim=-1)
        logits = torch.full((batch, seq_len, padded_topk), float('-inf'), device=q.device, dtype=torch.float32)
    else:
        padded_sparse_topk_indices = sparse_topk_indices
        logits = torch.full((batch, seq_len, stage_1_topk), float('-inf'), device=q.device, dtype=torch.float32)


    if weights.ndim == 2:
        weights = weights.view(batch, seq_len, heads)



    kernel = paged_mqa_attn_return_logits(
        paged_block_size=paged_block_size,
        heads=heads,
        index_dim=index_dim,
        dtype=dtype,
    )
    kernel(
        q,
        kv_cache,
        logits,
        weights,
        padded_sparse_topk_indices,
        context_lens.to(torch.int32),
        block_tables.to(torch.int32),
    )
    return logits[:, :, :stage_1_topk]

def batch_selfk_mqa_attn_return_logits_interface(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    dtype="bfloat16",
):
    """
    Decode阶段：每个q只和自己同位置的k计算head score。
    q: [B, 1, H, D]
    kv_cache: [num_blocks, block_size, 1, D]
    weights: [B, 1, H]
    context_lens: [B], 每个样本的有效kv长度
    block_tables: [B, max_blocks]

    decode时，当前token的k位置就是 context_lens[b] - 1。

    Returns: q_head_score [B, 1, H]
    """
    B, seq_len_q, H, D = q.shape
    assert seq_len_q == 1
    paged_block_size = kv_cache.shape[1]

    self_k_pos = context_lens - 1  # [B], 每个样本当前token的k位置

    # 从paged kv_cache中取出每个样本自身位置的k
    logical_block_idx = self_k_pos // paged_block_size  # [B]
    offset_in_block = self_k_pos % paged_block_size  # [B]

    phys_block_idx = block_tables[torch.arange(B, device=block_tables.device), logical_block_idx]  # [B]
    self_k = kv_cache[phys_block_idx, offset_in_block, 0, :]  # [B, D]

    # q: [B, 1, H, D] -> [B, H, D]
    q_squeezed = q.squeeze(1)  # [B, H, D]
    weights_squeezed = weights.squeeze(1)  # [B, H]

    # score[b, h] = relu(self_k[b] @ q[b, h]) * weights[b, h]
    scores = torch.einsum('bd,bhd->bh', self_k.float(), q_squeezed.float())  # [B, H]
    scores = torch.relu(scores) * weights_squeezed.float()  # [B, H]

    return scores.float()

def misa_page_mqa_logits(
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
    ):
    '''
    推理时计算logits，使用page attention形式的kv cache
    '''
    q = q_fp8.float()

    fp8_dtype = q_fp8.dtype
    dim = q.shape[-1]
    num_blocks, block_size, _, D_plus_4 = kv_cache_fp8.shape
    kv_cache = kv_cache_fp8.view(num_blocks, -1)

    scale = kv_cache[:, block_size * dim:]  # [num_blocks]
    kv_cache = kv_cache[:, :block_size * dim].view(fp8_dtype)      # [num_blocks, block_size * dim]
    scale = scale.contiguous().view(torch.float32)

    kv_cache = kv_cache.view(num_blocks, block_size, 1, dim)  # [num_blocks, block_size, dim]
    scale = scale.view(num_blocks, block_size)

    kv_cache = kv_cache.float() * scale[:, :, None, None]  # [num_blocks, block_size, 1, D]

    q = q.bfloat16()
    kv_cache = kv_cache.bfloat16()
    weights = weights.bfloat16()

    if not use_qnorm:
        # blocked_k, num_pooling_blocks = paged_mean_pooling_interface(kv_cache, context_lens, block_tables, k_block_size)  # [B, num_pooling_blocks, D], [B]

        # q_head_score = batch_block_mqa_attn_return_logits_interface(q=q, blocked_kv=blocked_k, kv_block_size=k_block_size, weights=weights, context_lens=num_pooling_blocks)  # [B, next_n, num_pooling_blocks]
        q_head_score = batch_selfk_mqa_attn_return_logits_interface(q=q, kv_cache=kv_cache, weights=weights, context_lens=context_lens, block_tables=block_tables)  # [B, next_n, num_pooling_blocks]

        _, q_head_topk_indices = torch.topk(q_head_score, k=decode_topk_heads, dim=-1, largest=True, sorted=False)
    topk_indices_expanded = (q_head_topk_indices.view(q.size(0), q.size(1), -1)
                                .unsqueeze(-1).expand(-1, -1, -1, q.size(-1)))
    sparse_head_q = torch.gather(q, dim=2, index=topk_indices_expanded)
    sparse_head_weights = torch.gather(weights, dim=1, index=q_head_topk_indices)
    # print(sparse_head_q.shape, sparse_head_weights.shape)

    logits = paged_mqa_sparse_head_return_logits_interface(
        q=sparse_head_q,
        kv_cache=kv_cache,
        weights=sparse_head_weights,
        context_lens=context_lens,
        block_tables=block_tables,
        max_model_len=max_model_len,
        dtype="bfloat16",
    )

    return logits.view(-1, logits.shape[-1])

@tilelang.jit(
    pass_configs=global_pass_configs,
)
def misa_mqa_sparse_head_return_logits(
    heads,
    index_dim,
    block_N=64,
    block_Q=None,
    num_stages=2,
    threads=128,
    dtype="bfloat16",
):
    accum_dtype = T.float32
    index_dtype = T.int32

    if block_Q is None:
        block_Q = 128 // heads # 多少q放在一个block里

    seq_len = T.dynamic("seq_len")
    seq_len_kv  = T.dynamic("seq_len_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    logits_shape = [seq_len, seq_len_kv]

    @T.prim_func
    def misa_mqa_sparse_head_return_logits_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        CuSeqLenS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, block_Q, heads))
            logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_s_min = T.alloc_var(index_dtype)
            cu_e_max = T.alloc_var(index_dtype)

            cu_s_min = 2147483647
            cu_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                cu_s_min = T.min(cu_s_min, CuSeqLenS[seq_len_i + bq_i])
            for bq_i in T.serial(block_Q):
                cu_e_max = T.max(cu_e_max, CuSeqLenE[seq_len_i + bq_i])
            cu_seq = cu_e_max - cu_s_min

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for b_i in T.Pipelined(T.ceildiv(cu_seq, block_N), num_stages=num_stages):
                block_s_i = cu_s_min + b_i * block_N
                T.copy(IndexK[block_s_i, 0], index_k_shared)
                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i])

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    n_i = block_s_i + bn_i
                    if n_i < CuSeqLenS[seq_len_i + bq_i] or n_i >= CuSeqLenE[seq_len_i + bq_i]:
                        logits[bn_i, bq_i] = -T.infinity(accum_dtype)

                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    Logits[seq_len_i + bq_i, block_s_i + bn_i] = logits[bn_i, bq_i]

    return misa_mqa_sparse_head_return_logits_kernel

def misa_mqa_sparse_head_return_logits_interface(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, dtype="bfloat16"):
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]

    if heads==64:
        misa_mqa_sparse_head_return_logits_kernel = misa_mqa_sparse_head_return_logits(heads=heads, index_dim=index_dim, block_N=128, num_stages=3,threads=256)
    else:
        misa_mqa_sparse_head_return_logits_kernel = misa_mqa_sparse_head_return_logits(heads=heads, index_dim=index_dim)
    logits = torch.full([seq_len, seq_len_kv], float('-inf'), device=q.device, dtype=torch.float32)
    misa_mqa_sparse_head_return_logits_kernel(
        q.view(seq_len * heads, index_dim),
        kv,
        logits,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    return logits

@tilelang.jit(
    pass_configs=global_pass_configs,
)
def misa_mqa_attn_return_logits(
    heads,
    index_dim,
    block_N=64,
    num_stages=1,
    threads=128,
    dtype="bfloat16",
):
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len = T.dynamic("seq_len")
    seq_len_kv  = T.dynamic("seq_len_kv")
    stage_1_topk = T.dynamic("stage_1_topk")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    logits_shape = [seq_len, stage_1_topk]
    sparse_topk_indices_shape = [seq_len, stage_1_topk]

    # TODO check padded H in sparse_mla_fwd
    # does it matter here?
    H_per_block = heads

    @T.prim_func
    def misa_mqa_attn_return_logits_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        SparseTopKIndices: T.Tensor(sparse_topk_indices_shape, index_dtype),  # type: ignore
        CuSeqLenS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            index_q_shared = T.alloc_shared([H_per_block, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, H_per_block], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, H_per_block // heads, heads))
            logits = T.alloc_fragment([block_N, H_per_block // heads], accum_dtype)
            weights = T.alloc_fragment([H_per_block // heads, heads], accum_dtype)
            # sparse_topk_indices = T.alloc_fragment([block_N], index_dtype)

            seq_len_i = bx

            cu_s_min = CuSeqLenS[seq_len_i]
            cu_e_max = CuSeqLenE[seq_len_i]

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for b_i in T.Pipelined(T.ceildiv(stage_1_topk, block_N), num_stages=num_stages):
                block_s_i = b_i * block_N

                # T.copy(SparseTopKIndices[seq_len_i, block_s_i], sparse_topk_indices)
                # T.print(sparse_topk_indices)
                for i,j in T.Parallel(block_N, index_dim):
                    idx = SparseTopKIndices[seq_len_i, block_s_i + i]
                    index_k_shared[i, j] = IndexK[idx, j]

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, H_per_block, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i])

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for i_i in T.Parallel(block_N):
                    k_i = SparseTopKIndices[seq_len_i, block_s_i + i_i]
                    if k_i < cu_s_min or k_i >= cu_e_max:
                        logits[i_i, 0] = -T.infinity(accum_dtype)

                for bn_i in T.Parallel(block_N):
                    Logits[seq_len_i, block_s_i + bn_i] = logits[bn_i, 0]

    return misa_mqa_attn_return_logits_kernel

def selfk_mqa_attn_return_logits_interface(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, dtype="bfloat16"):
    """
    Prefill阶段：每个q只和自己同位置的k计算head score。
    q: [M, H, D]
    kv: [N, D]
    weights: [M, H]
    cu_seqlen_ks: [M] int32, 每个q对应的k起始位置（用于确定自身k位置）
    cu_seqlen_ke: [M] int32, 每个q对应的k结束位置

    对于位置i的q，其对应的k位置就是i自身在kv序列中的位置。
    在prefill场景中，q的第i个token对应kv的第(cu_seqlen_ke[i]-1)个位置
    （即当前token自身的k）。

    score[i, h] = relu(k_i @ q[i, h]) * weights[i, h]

    Returns: q_head_score [M, H]
    """
    seq_len, heads, index_dim = q.shape

    self_k_indices = (cu_seqlen_ke - 1).long()  # [M], 每个q对应自身位置的k索引
    self_k = kv[self_k_indices]  # [M, D]

    # score = (self_k @ q^T) => [M, H]
    # self_k: [M, D], q: [M, H, D] => einsum: [M, H]
    scores = torch.einsum('md,mhd->mh', self_k.float(), q.float())  # [M, H]
    scores = torch.relu(scores) * weights.float()  # [M, H]

    return scores.float()



def misa_mqa_attn_return_logits_interface(q, kv, weights, sparse_topk_indices, cu_seqlen_ks, cu_seqlen_ke, dtype="bfloat16"):
    seq_len, heads, index_dim = q.shape
    stage_1_topk = sparse_topk_indices.shape[-1]

    # 计算需要的对齐长度（128的倍数）
    block_size = 128
    padded_topk = ((stage_1_topk + block_size - 1) // block_size) * block_size
    padding_size = padded_topk - stage_1_topk

    # 获取原始 kv 序列长度
    kv_seq_len = kv.shape[0]

    # Padding kv：用0填充
    if padding_size > 0:
        kv_padded = torch.cat([kv, torch.zeros(padding_size, kv.shape[1],
                                                 device=kv.device, dtype=kv.dtype)], dim=0)
    else:
        kv_padded = kv

    # Padding sparse_topk_indices：用 kv_seq_len-1 填充
    # 这样 kernel 内部会因为索引超出范围而赋为 -inf
    if padding_size > 0:
        pad_value = kv_seq_len - 1
        sparse_topk_indices_padded = torch.cat([
            sparse_topk_indices,
            torch.full((seq_len, padding_size), pad_value,
                      device=sparse_topk_indices.device, dtype=sparse_topk_indices.dtype)
        ], dim=-1)
    else:
        sparse_topk_indices_padded = sparse_topk_indices

    # 调用 kernel
    misa_mqa_attn_return_logits_kernel = misa_mqa_attn_return_logits(heads=heads, index_dim=index_dim, block_N=128)
    logits_padded = torch.empty([seq_len, padded_topk], device=q.device, dtype=torch.float32)
    misa_mqa_attn_return_logits_kernel(
        q.view(seq_len * heads, index_dim),
        kv_padded,
        logits_padded,
        weights,
        sparse_topk_indices_padded,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )

    # 去掉 padding 部分，返回原始大小
    logits = logits_padded[:, :stage_1_topk]

    return logits

def misa_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    q_head_topk_indices: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    use_qnorm: bool,
    topk_heads: int,
    k_block_size: int | None = None,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    q = q.float()  # [M, H, D]
    k_fp8, k_scales = kv
    k_scales = k_scales.contiguous().view(torch.float32)
    if k_scales.ndim == 1:
        k_scales = k_scales.unsqueeze(-1)  # [N, 1]
    k = k_fp8.float() * k_scales  # [N, D]
    q = q.bfloat16()
    k = k.bfloat16()
    weights = weights.bfloat16()
    if use_qnorm:
        if topk_heads < q_head_topk_indices.shape[1]:
            q_head_topk_indices = q_head_topk_indices[:, :topk_heads]
    else:
        assert k_block_size is not None, "k_block_size must be provided when block_topk is set"
        cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
        cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size

        blocked_k = block_mean_pooling_interface(k, k_block_size)  # [num_block, D]

        q_head_score = block_mqa_attn_return_logits_interface(q=q, blocked_kv=blocked_k, kv_block_size=k_block_size, weights=weights, cu_seqlen_blocked_ks=cu_seqlen_blocked_ks, cu_seqlen_blocked_ke=cu_seqlen_blocked_ke)
        # q_head_score = selfk_mqa_attn_return_logits_interface(q=q, kv=k, weights=weights, cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)
        # print(q_head_score)
        _, q_head_topk_indices = torch.topk(q_head_score, k=topk_heads, dim=-1, largest=True, sorted=False)
    topk_indices_expanded = q_head_topk_indices.unsqueeze(-1).expand(-1, -1, q.size(-1))
    sparse_head_q = torch.gather(q, dim=1, index=topk_indices_expanded)
    sparse_head_weights = torch.gather(weights, dim=1, index=q_head_topk_indices)

    logits = misa_mqa_sparse_head_return_logits_interface(q=sparse_head_q, kv=k, weights=sparse_head_weights, cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)

    return logits
