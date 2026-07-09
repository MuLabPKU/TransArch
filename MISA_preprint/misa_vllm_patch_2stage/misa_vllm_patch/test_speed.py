from typing import Optional
from einops import einsum, repeat
import torch
import torch.nn.functional as F
from misa_vllm_patch_2stage.misa_vllm_patch.misa_ops import misa_mqa_sparse_head_return_logits_interface, misa_mqa_attn_return_logits_interface
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
    stage_1_topk_tokens: int,
    topk_tokens: int,
    offsets_q: torch.Tensor,
    offsets_k: torch.Tensor,
    block_topk: Optional[int] = None,
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

    logits = misa_mqa_sparse_head_return_logits_interface(q=sparse_head_q, kv=kv, weights=sparse_head_weights, cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)
    topk_indices = logits.topk(k=topk_tokens, dim=-1, largest=True, sorted=False).indices.to(torch.int32)  # [M, 8192]

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

    topk_indices = torch.topk(index_score, k=min(topk, index_score.shape[-1]), dim=-1).indices  # [seq_len, block_topk]
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
    stage_1_topk_tokens: int=8192,
    topk_tokens: int=2048,
    WARMUP=100,
    REPEAT=100,
    block_topk: Optional[int] = 128,
    k_block_size: Optional[int] = 128,
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

    topk_indices = misa_indexer_topk_reducesum_interface(index_q, index_k, weights, topk_heads , stage_1_topk_tokens, topk_tokens,offsets_q, offsets_k,
                                                         block_topk=block_topk, k_block_size=k_block_size)

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

    def ref_tilelang_fn():
        return deepseek_sparse_attention_indexer(index_q, index_k, weights,
                                                            offsets_q, offsets_k, topk_tokens)
    ref_tilelang_ms = do_bench(ref_tilelang_fn, warmup=WARMUP, rep=REPEAT)
    print(f"dsa_tilelang_ms: {ref_tilelang_ms}")

    def tilelang_fn():
        return misa_indexer_topk_reducesum_interface(index_q, index_k, weights, topk_heads , stage_1_topk_tokens, topk_tokens,offsets_q, offsets_k,
                                                     block_topk=block_topk, k_block_size=k_block_size)
    tilelang_ms = do_bench(tilelang_fn, warmup=WARMUP, rep=REPEAT)
    print(f"misa_tilelang_ms: {tilelang_ms}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--SQ", type=int, default=2048)
    parser.add_argument("--SK", type=int, default=32768)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--D", type=int, default=512)
    parser.add_argument("--tail_D", type=int, default=64)
    parser.add_argument("--index_D", type=int, default=128)
    parser.add_argument("--topk_heads", type=int, default=8)
    parser.add_argument("--stage_1_topk_tokens", type=int, default=2048)
    parser.add_argument("--topk_tokens", type=int, default=2048)
    args = parser.parse_args()

    test_kernel(B=args.B, SQ=args.SQ, SK=args.SK, H=args.H, D=args.D, tail_D=args.tail_D, index_D=args.index_D, topk_heads=args.topk_heads, stage_1_topk_tokens=args.stage_1_topk_tokens, topk_tokens=args.topk_tokens)
