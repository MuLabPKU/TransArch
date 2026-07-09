"""test_cp_equiv.py — warmup CP numerical equivalence (Gate 2).

Runs ONE patched DeepSeek-V2 attention layer's warmup path and asserts that the
context-parallel (cp=world) result equals the single-GPU (cp=1) reference for BOTH
the indexer KL loss and the indexer gradients.

  torchrun --nproc_per_node=2 tests/test_cp_equiv.py

Each rank:
  1. builds the SAME tiny 1-layer model + indexer (seeded identically),
  2. rank 0 computes the cp=1 full-sequence reference (loss + indexer grads),
  3. all ranks compute the cp=world sharded warmup (local queries, all-gathered keys),
     eager-backward gives per-rank indexer grads; we all-reduce(SUM) them over the CP
     group, then compare on rank 0 to the reference.
"""
import _path  # noqa: F401
import os
import torch
from transformers import AutoModelForCausalLM, AutoConfig

from opendsa.dist import init_parallel, cp_size, cp_rank, local_shard, cp_group, is_dist
from opendsa.modeling import (patch_model_with_dsa, IndexerConfig,
                              freeze_backbone_train_indexer, indexer_parameters,
                              dsa_loss_registry, set_dsa_mode, set_cu_seqlens, set_cp_size)
import torch.distributed as dist

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"


def build(dtype):
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg.num_hidden_layers = 1
    cfg.first_k_dense_replace = 1
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True,
                                         attn_implementation="eager").cuda().to(dtype)
    patch_model_with_dsa(m, IndexerConfig(n_heads=8, head_dim=128, topk=64, tile=64),
                         mode="warmup", dtype=dtype)
    freeze_backbone_train_indexer(m)
    return cfg, m


def main():
    S = init_parallel()
    dev = f"cuda:{S.local_rank}"
    torch.cuda.set_device(dev)
    dtype = torch.float64                        # fp64 for a tight numerical gate
    cfg, m = build(dtype)
    attn = m.model.layers[0].self_attn
    n = cp_size()

    L = 128
    torch.manual_seed(123)
    hs_full = torch.randn(1, L, cfg.hidden_size, device=dev, dtype=dtype)
    cu_full = torch.tensor([0, L], device=dev)

    # --- reference: absorbed cp=1 full sequence (like-for-like with the CP path,
    #     which also uses the absorbed teacher). Build it directly via the same ops. ---
    from opendsa.modeling.dsa_attention import (_compute_teacher_absorbed,
                                                _indexer_rope_cossin, _warmup_forward)
    from opendsa.ops import flashkl_warmup_loss
    attn.cp_size = 1
    attn.cu_seqlens = None
    m.zero_grad(set_to_none=True)
    gpos_full = torch.arange(L, device=dev).unsqueeze(0)
    with torch.no_grad():
        q_lat, latent, W_UV, cos, sin = _compute_teacher_absorbed(attn, hs_full, gpos_full,
                                                                  rope_seq_len=L)
    Qlat = q_lat.transpose(1, 2)                                   # [1,L,H,Dl]
    qin = hs_full.detach()
    c_idx, s_idx = _indexer_rope_cossin(attn, cos, sin, gpos_full)
    q_i, k_i, w_i = attn.indexer(hs_full.detach(), qin, c_idx, s_idx)
    ref_L = flashkl_warmup_loss(q_i[0], k_i[0], w_i[0], Qlat[0].float(), latent[0].float(),
                                cu_full, sm_scale_teacher=float(attn.softmax_scale),
                                sm_scale_index=attn.indexer.softmax_scale, tile=64,
                                q_global_pos=gpos_full[0], nrow_global=L)
    ref_loss = float(ref_L.detach())
    ref_L.backward()
    ref_g = {nm: p.grad.clone() for nm, p in attn.indexer.named_parameters()
             if p.grad is not None}

    # --- CP: shard queries, all-gather keys, eager backward, all-reduce grads ---
    attn.cp_size = n
    attn.cu_seqlens = cu_full
    hs_loc = local_shard(hs_full, dim=1)         # [1, L/n, hidden]
    reg = dsa_loss_registry()
    reg.reset(); reg.set_eager_backward(True, scale=1.0)
    m.zero_grad(set_to_none=True)
    _warmup_forward(attn, None, hs_loc, None, None, None, None, use_cache=False)
    cp_loss_local = reg.eager_loss_sum()
    # sum loss + grads across CP group
    cp_g = {}
    for nm, p in attn.indexer.named_parameters():
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        if is_dist():
            dist.all_reduce(g, op=dist.ReduceOp.SUM, group=cp_group())
        cp_g[nm] = g.clone()
    loss_t = torch.tensor(cp_loss_local, device=dev)
    if is_dist():
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM, group=cp_group())
    cp_loss = float(loss_t)

    if cp_rank() == 0:
        rel = lambda a, b: (a - b).norm().item() / (b.norm().item() + 1e-30)
        dloss = abs(cp_loss - ref_loss)
        gerr = {nm: rel(cp_g[nm], ref_g[nm]) for nm in ref_g}
        maxg = max(gerr.values())
        ok = dloss < 1e-6 and maxg < 1e-4
        print(f"[cp_equiv] n={n}  ref_loss={ref_loss:.6f}  cp_loss={cp_loss:.6f}  |Δ|={dloss:.2e}")
        print(f"[cp_equiv] indexer grad max rel-err={maxg:.2e}")
        for nm, e in gerr.items():
            print(f"    {nm}: {e:.2e}")
        print("CP EQUIV PASSED" if ok else "CP EQUIV FAILED")


if __name__ == "__main__":
    try:
        main()
    finally:
        _path.shutdown_dist()
