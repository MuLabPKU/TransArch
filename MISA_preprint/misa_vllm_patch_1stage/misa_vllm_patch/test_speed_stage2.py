from typing import Optional
from einops import einsum, repeat
import torch
import torch.nn.functional as F
from misa_vllm_patch_1stage.misa_vllm_patch.misa_ops import misa_mqa_sparse_head_return_logits_interface, misa_mqa_attn_return_logits_interface
from block_ops import (block_mean_pooling_interface,
                        block_mqa_attn_return_logits_interface,
                        block_sparse_mqa_attn_return_logits_interface,
                        paged_mean_pooling_interface,
                        paged_block_sparse_mqa_attn_return_logits_interface,
                        batch_block_mqa_attn_return_logits_interface)
from tilelang_utils import get_abs_err, get_err_ratio, prepare_ks_ke_from_cu_seqlens_qk, per_custom_dims_cast_to_fp8
import argparse

def misa_indexer_topk_reducesum_interface(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    topk_heads: int,
    topk_tokens: int,
    offsets_q: torch.Tensor,
    offsets_k: torch.Tensor,
    k_block_size: Optional[int] = None,
):
    # TODO dtype not used
    seq_len, heads, dim = q.shape
    cu_seqlen_ks, cu_seqlen_ke = prepare_ks_ke_from_cu_seqlens_qk(offsets_q, offsets_k)

    if len(cu_seqlen_ke) > seq_len:
        raise ValueError("Length of cu_seqlen_ke is greater than seq_len")
        assert len(offsets_q) == 2
        cu_seqlen_ke = cu_seqlen_ke[-seq_len:]
        cu_seqlen_ks = cu_seqlen_ks[-seq_len:]

    if k_block_size is None:
        # 两阶段筛选： moe token
        # 第一阶段：稀疏topk heads选出8192个token / 选出1/4token
        q_norm = q_norm = torch.norm(q, p=2, dim=-1) # (seq, heads)
        _, q_head_topk_indices = torch.topk(q_norm, k=topk_heads, dim=-1, largest=True, sorted=False)
        if topk_heads < q_head_topk_indices.shape[1]:
            q_head_topk_indices = q_head_topk_indices[:, :topk_heads]
    else:
        cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
        cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size

        blocked_k = block_mean_pooling_interface(kv, k_block_size)  # [num_block, D]

        q_head_score = block_mqa_attn_return_logits_interface(q=q, blocked_kv=blocked_k, kv_block_size=k_block_size, weights=weights, cu_seqlen_blocked_ks=cu_seqlen_blocked_ks, cu_seqlen_blocked_ke=cu_seqlen_blocked_ke)
        _, q_head_topk_indices = torch.topk(q_head_score, k=topk_heads, dim=-1, largest=True, sorted=False)
    topk_indices_expanded = q_head_topk_indices.unsqueeze(-1).expand(-1, -1, q.size(-1))
    sparse_head_q = torch.gather(q, dim=1, index=topk_indices_expanded)
    sparse_head_weights = torch.gather(weights, dim=1, index=q_head_topk_indices)

    stage_1_logits = misa_mqa_sparse_head_return_logits_interface(q=sparse_head_q, kv=kv, weights=sparse_head_weights, cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)
    sparse_topk_indices = stage_1_logits.topk(k=8192, dim=-1, largest=True, sorted=False).indices.to(torch.int32)  # [M, 8192]

    # 第二阶段：8192个token中选出最终topk tokens
    logits = misa_mqa_attn_return_logits_interface(q=q, kv=kv, weights=weights, sparse_topk_indices=sparse_topk_indices, cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)  # [M, 8192]

    # Select top-K tokens from second-stage logits
    topk_logits_indices = torch.topk(logits, k=min(topk_tokens, logits.shape[-1]), dim=-1, largest=True).indices  # [M, topk]
    topk_indices = torch.gather(sparse_topk_indices, dim=-1, index=topk_logits_indices)
    topk_indices = torch.where((topk_indices >= cu_seqlen_ks.unsqueeze(-1)) & (topk_indices < cu_seqlen_ke.unsqueeze(-1)) , topk_indices, -1)

    # make absolute indices to relative indices within the sequence
    topk_indices = topk_indices - cu_seqlen_ks.unsqueeze(-1)
    return topk_indices

def deepseek_sparse_attention_indexer(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    offsets_q: torch.Tensor,
    offsets_k: torch.Tensor,
    topk: int,
):
    cu_seqlen_ks, cu_seqlen_ke = prepare_ks_ke_from_cu_seqlens_qk(offsets_q, offsets_k)
    index_score = misa_mqa_sparse_head_return_logits_interface(q=q, kv=kv, weights=weights, cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)

    topk_indices = torch.topk(index_score, k=min(topk, index_score.shape[-1]), dim=-1).indices
    topk_indices = topk_indices - cu_seqlen_ks.unsqueeze(-1)

    return topk_indices


def test_kernel(
    B=1,
    SQ=1024,
    SK=1024,
    H=16,
    D=512,
    tail_D=64,
    index_D=128,
    topk_heads: int=16,
    topk_tokens: int=2048,
    WARMUP=100,
    REPEAT=100,
    k_block_size: Optional[int] = 1024,
):
    torch.manual_seed(42)
    q = torch.randn((SQ, H, D + tail_D)).cuda().bfloat16().requires_grad_()
    kv = torch.randn((SK, D + tail_D)).cuda().bfloat16().requires_grad_()
    index_q = torch.randn((SQ, H, index_D)).cuda().bfloat16().requires_grad_()
    weights = torch.randn((SQ, H)).cuda().bfloat16().requires_grad_()
    index_k = torch.randn((SK, index_D)).cuda().bfloat16().requires_grad_()

    offsets_q = list(range(0, SQ, SQ // B))[:B] + [SQ]
    offsets_k = list(range(0, SK, SK // B))[:B] + [SK]
    offsets_q = torch.tensor(offsets_q, dtype=torch.int32).cuda()
    offsets_k = torch.tensor(offsets_k, dtype=torch.int32).cuda()

    # 计算交集
    topk_indices = misa_indexer_topk_reducesum_interface(index_q, index_k, weights, topk_heads , topk_tokens,offsets_q, offsets_k,
                                                          k_block_size=k_block_size)

    ref_tilelang_topk_indices = deepseek_sparse_attention_indexer(index_q, index_k, weights, offsets_q, offsets_k, topk_tokens)

    intersections = []
    for j in range(SQ):
        ref_np = ref_tilelang_topk_indices[j].cpu().to(torch.int32).numpy()
        trt_np = topk_indices[j].cpu().to(torch.int32).numpy()

        mask = (trt_np != -1)

        set_ref = set(ref_np[mask])
        set_trt = set(trt_np[mask])
        intersection = set_ref & set_trt
        intersections.append(len(intersection) / len(set_ref))
    print("average intersections: {:.4f}".format(sum(intersections) / len(intersections)))

    from tilelang.profiler import do_bench

    # 测试 deepseek_sparse_attention_indexer 的时间分解
    print("\n=== deepseek_sparse_attention_indexer 时间分解 ===")

    def ref_index_score_only():
        cu_seqlen_ks, cu_seqlen_ke = prepare_ks_ke_from_cu_seqlens_qk(offsets_q, offsets_k)
        index_score = misa_mqa_sparse_head_return_logits_interface(q=index_q, kv=index_k, weights=weights, cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)
        return index_score

    def ref_topk_only():
        cu_seqlen_ks, cu_seqlen_ke = prepare_ks_ke_from_cu_seqlens_qk(offsets_q, offsets_k)
        index_score = misa_mqa_sparse_head_return_logits_interface(q=index_q, kv=index_k, weights=weights, cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)
        topk_indices = index_score.topk(k=topk_tokens, dim=-1, largest=True, sorted=False).indices.to(torch.int32)
        topk_indices = torch.where((topk_indices >= cu_seqlen_ks.unsqueeze(-1)) & (topk_indices < cu_seqlen_ke.unsqueeze(-1)) ,
                                  topk_indices, -1)
        topk_indices = topk_indices - cu_seqlen_ks.unsqueeze(-1)
        return topk_indices

    def ref_complete():
        return deepseek_sparse_attention_indexer(index_q, index_k, weights, offsets_q, offsets_k, topk_tokens)

    # 测量index_score时间
    ref_index_time = do_bench(ref_index_score_only, warmup=WARMUP, rep=REPEAT)
    print(f"deepseek_sparse_attention_indexer - index_score时间: {ref_index_time:.4f} ms")

    # 测量topk时间（包含index_score）
    ref_topk_time = do_bench(ref_topk_only, warmup=WARMUP, rep=REPEAT)
    print(f"deepseek_sparse_attention_indexer - 总时间: {ref_topk_time:.4f} ms")
    print(f"deepseek_sparse_attention_indexer - topk计算时间: {ref_topk_time - ref_index_time:.4f} ms")

    # 测试 misa_indexer_topk_reducesum_interface 的时间分解
    print("\n=== misa_indexer_topk_reducesum_interface 时间分解 ===")

    def misa_head_score_only():
        seq_len, heads, dim = index_q.shape
        cu_seqlen_ks, cu_seqlen_ke = prepare_ks_ke_from_cu_seqlens_qk(offsets_q, offsets_k)

        if k_block_size is None:
            q_norm = torch.norm(index_q, p=2, dim=-1)
            _, q_head_topk_indices = torch.topk(q_norm, k=topk_heads, dim=-1, largest=True, sorted=False)
        else:
            cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
            cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size
            blocked_k = block_mean_pooling_interface(index_k, k_block_size)
            q_head_score = block_mqa_attn_return_logits_interface(q=index_q, blocked_kv=blocked_k, kv_block_size=k_block_size,
                                                                 weights=weights, cu_seqlen_blocked_ks=cu_seqlen_blocked_ks,
                                                                 cu_seqlen_blocked_ke=cu_seqlen_blocked_ke)
            _, q_head_topk_indices = torch.topk(q_head_score, k=topk_heads, dim=-1, largest=True, sorted=False)
        return q_head_topk_indices

    def misa_index_only():
        seq_len, heads, dim = index_q.shape
        cu_seqlen_ks, cu_seqlen_ke = prepare_ks_ke_from_cu_seqlens_qk(offsets_q, offsets_k)

        if k_block_size is None:
            q_norm = torch.norm(index_q, p=2, dim=-1)
            _, q_head_topk_indices = torch.topk(q_norm, k=topk_heads, dim=-1, largest=True, sorted=False)
        else:
            cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
            cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size
            blocked_k = block_mean_pooling_interface(index_k, k_block_size)
            q_head_score = block_mqa_attn_return_logits_interface(q=index_q, blocked_kv=blocked_k, kv_block_size=k_block_size,
                                                                 weights=weights, cu_seqlen_blocked_ks=cu_seqlen_blocked_ks,
                                                                 cu_seqlen_blocked_ke=cu_seqlen_blocked_ke)
            _, q_head_topk_indices = torch.topk(q_head_score, k=topk_heads, dim=-1, largest=True, sorted=False)

        topk_indices_expanded = q_head_topk_indices.unsqueeze(-1).expand(-1, -1, index_q.size(-1))
        sparse_head_q = torch.gather(index_q, dim=1, index=topk_indices_expanded)
        sparse_head_weights = torch.gather(weights, dim=1, index=q_head_topk_indices)

        logits = misa_mqa_sparse_head_return_logits_interface(q=sparse_head_q, kv=index_k, weights=sparse_head_weights,
                                                             cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)
        return logits

    def misa_stage1_topk_only():
        seq_len, heads, dim = index_q.shape
        cu_seqlen_ks, cu_seqlen_ke = prepare_ks_ke_from_cu_seqlens_qk(offsets_q, offsets_k)

        if k_block_size is None:
            q_norm = torch.norm(index_q, p=2, dim=-1)
            _, q_head_topk_indices = torch.topk(q_norm, k=topk_heads, dim=-1, largest=True, sorted=False)
        else:
            cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
            cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size
            blocked_k = block_mean_pooling_interface(index_k, k_block_size)
            q_head_score = block_mqa_attn_return_logits_interface(q=index_q, blocked_kv=blocked_k, kv_block_size=k_block_size,
                                                                 weights=weights, cu_seqlen_blocked_ks=cu_seqlen_blocked_ks,
                                                                 cu_seqlen_blocked_ke=cu_seqlen_blocked_ke)
            _, q_head_topk_indices = torch.topk(q_head_score, k=topk_heads, dim=-1, largest=True, sorted=False)

        topk_indices_expanded = q_head_topk_indices.unsqueeze(-1).expand(-1, -1, index_q.size(-1))
        sparse_head_q = torch.gather(index_q, dim=1, index=topk_indices_expanded)
        sparse_head_weights = torch.gather(weights, dim=1, index=q_head_topk_indices)

        logits = misa_mqa_sparse_head_return_logits_interface(q=sparse_head_q, kv=index_k, weights=sparse_head_weights,
                                                             cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)
        topk_indices = logits.topk(k=8192, dim=-1, largest=True, sorted=False).indices.to(torch.int32)

        return topk_indices

    def misa_stage2_index_only():
        seq_len, heads, dim = index_q.shape
        cu_seqlen_ks, cu_seqlen_ke = prepare_ks_ke_from_cu_seqlens_qk(offsets_q, offsets_k)

        if k_block_size is None:
            q_norm = torch.norm(index_q, p=2, dim=-1)
            _, q_head_topk_indices = torch.topk(q_norm, k=topk_heads, dim=-1, largest=True, sorted=False)
        else:
            cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
            cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size
            blocked_k = block_mean_pooling_interface(index_k, k_block_size)
            q_head_score = block_mqa_attn_return_logits_interface(q=index_q, blocked_kv=blocked_k, kv_block_size=k_block_size,
                                                                 weights=weights, cu_seqlen_blocked_ks=cu_seqlen_blocked_ks,
                                                                 cu_seqlen_blocked_ke=cu_seqlen_blocked_ke)
            _, q_head_topk_indices = torch.topk(q_head_score, k=topk_heads, dim=-1, largest=True, sorted=False)

        topk_indices_expanded = q_head_topk_indices.unsqueeze(-1).expand(-1, -1, index_q.size(-1))
        sparse_head_q = torch.gather(index_q, dim=1, index=topk_indices_expanded)
        sparse_head_weights = torch.gather(weights, dim=1, index=q_head_topk_indices)

        stage_1_logits = misa_mqa_sparse_head_return_logits_interface(q=sparse_head_q, kv=index_k, weights=sparse_head_weights,
                                                             cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)
        sparse_topk_indices = stage_1_logits.topk(k=8192, dim=-1, largest=True, sorted=False).indices.to(torch.int32)
        logits = misa_mqa_attn_return_logits_interface(q=index_q, kv=index_k, weights=weights, sparse_topk_indices=sparse_topk_indices, cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)

        return logits

    def misa_stage2_topk_only():
        seq_len, heads, dim = index_q.shape
        cu_seqlen_ks, cu_seqlen_ke = prepare_ks_ke_from_cu_seqlens_qk(offsets_q, offsets_k)

        if k_block_size is None:
            q_norm = torch.norm(index_q, p=2, dim=-1)
            _, q_head_topk_indices = torch.topk(q_norm, k=topk_heads, dim=-1, largest=True, sorted=False)
        else:
            cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
            cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size
            blocked_k = block_mean_pooling_interface(index_k, k_block_size)
            q_head_score = block_mqa_attn_return_logits_interface(q=index_q, blocked_kv=blocked_k, kv_block_size=k_block_size,
                                                                 weights=weights, cu_seqlen_blocked_ks=cu_seqlen_blocked_ks,
                                                                 cu_seqlen_blocked_ke=cu_seqlen_blocked_ke)
            _, q_head_topk_indices = torch.topk(q_head_score, k=topk_heads, dim=-1, largest=True, sorted=False)

        topk_indices_expanded = q_head_topk_indices.unsqueeze(-1).expand(-1, -1, index_q.size(-1))
        sparse_head_q = torch.gather(index_q, dim=1, index=topk_indices_expanded)
        sparse_head_weights = torch.gather(weights, dim=1, index=q_head_topk_indices)

        stage_1_logits = misa_mqa_sparse_head_return_logits_interface(q=sparse_head_q, kv=index_k, weights=sparse_head_weights,
                                                             cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)
        sparse_topk_indices = stage_1_logits.topk(k=8192, dim=-1, largest=True, sorted=False).indices.to(torch.int32)
        logits = misa_mqa_attn_return_logits_interface(q=index_q, kv=index_k, weights=weights, sparse_topk_indices=sparse_topk_indices, cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)
        topk_logits_indices = torch.topk(logits, k=min(topk_tokens, logits.shape[-1]), dim=-1, largest=True).indices
        topk_indices = torch.gather(sparse_topk_indices, dim=-1, index=topk_logits_indices)
        topk_indices = torch.where((topk_indices >= cu_seqlen_ks.unsqueeze(-1)) & (topk_indices < cu_seqlen_ke.unsqueeze(-1)) , topk_indices, -1)
        topk_indices = topk_indices - cu_seqlen_ks.unsqueeze(-1)
        return topk_indices


    def misa_complete():
        return misa_indexer_topk_reducesum_interface(index_q, index_k, weights, topk_heads , topk_tokens,
                                                   offsets_q, offsets_k, k_block_size=k_block_size)

    # 测量head_score时间
    misa_head_score_time = do_bench(misa_head_score_only, warmup=WARMUP, rep=REPEAT)
    print(f"misa_indexer - head_score时间: {misa_head_score_time:.4f} ms")

    # 测量stage1 index时间（包含head_score）
    misa_index_time = do_bench(misa_index_only, warmup=WARMUP, rep=REPEAT)
    print(f"misa_indexer - stage1 index_score时间: {misa_index_time:.4f} ms")
    print(f"misa_indexer - stage1 index计算时间(增量): {misa_index_time - misa_head_score_time:.4f} ms")

    # 测量stage1 topk时间
    misa_stage1_topk_time = do_bench(misa_stage1_topk_only, warmup=WARMUP, rep=REPEAT)
    print(f"misa_indexer - stage1 topk时间: {misa_stage1_topk_time:.4f} ms")
    print(f"misa_indexer - stage1 topk计算时间(增量): {misa_stage1_topk_time - misa_index_time:.4f} ms")

    # 测量stage2 index时间
    misa_stage2_index_time = do_bench(misa_stage2_index_only, warmup=WARMUP, rep=REPEAT)
    print(f"misa_indexer - stage2 index时间: {misa_stage2_index_time:.4f} ms")
    print(f"misa_indexer - stage2 index计算时间(增量): {misa_stage2_index_time - misa_stage1_topk_time:.4f} ms")

    # 测量stage2 topk时间（完整流程）
    misa_stage2_topk_time = do_bench(misa_stage2_topk_only, warmup=WARMUP, rep=REPEAT)
    print(f"misa_indexer - stage2 topk时间: {misa_stage2_topk_time:.4f} ms")
    print(f"misa_indexer - stage2 topk计算时间(增量): {misa_stage2_topk_time - misa_stage2_index_time:.4f} ms")

    # 完整函数对比
    print("\n=== 完整函数性能对比 ===")
    ref_complete_time = do_bench(ref_complete, warmup=WARMUP, rep=REPEAT)
    misa_complete_time = do_bench(misa_complete, warmup=WARMUP, rep=REPEAT)

    print(f"deepseek_sparse_attention_indexer 总时间: {ref_complete_time:.4f} ms")
    print(f"misa_indexer_topk_reducesum_interface 总时间: {misa_complete_time:.4f} ms")
    print(f"性能提升: {ref_complete_time/misa_complete_time:.2f}x")

    # 时间分解汇总
    print("\n=== misa_indexer 时间分解汇总 ===")
    print(f"  head_score:               {misa_head_score_time:.4f} ms")
    print(f"  stage1 index (增量):      {misa_index_time - misa_head_score_time:.4f} ms")
    print(f"  stage1 topk  (增量):      {misa_stage1_topk_time - misa_index_time:.4f} ms")
    print(f"  stage2 index (增量):      {misa_stage2_index_time - misa_stage1_topk_time:.4f} ms")
    print(f"  stage2 topk  (增量):      {misa_stage2_topk_time - misa_stage2_index_time:.4f} ms")
    print(f"  端到端总时间:             {misa_stage2_topk_time:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--SQ", type=int, default=2048)
    parser.add_argument("--SK", type=int, default=8192)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--D", type=int, default=512)
    parser.add_argument("--tail_D", type=int, default=64)
    parser.add_argument("--index_D", type=int, default=128)
    parser.add_argument("--topk_heads", type=int, default=8)
    parser.add_argument("--topk_tokens", type=int, default=2048)
    args = parser.parse_args()

    test_kernel(B=args.B, SQ=args.SQ, SK=args.SK, H=args.H, D=args.D, tail_D=args.tail_D, index_D=args.index_D, topk_heads=args.topk_heads, topk_tokens=args.topk_tokens)
