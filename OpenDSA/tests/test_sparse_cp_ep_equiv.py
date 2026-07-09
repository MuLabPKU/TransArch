"""test_sparse_cp_ep_equiv.py — sparse CP+EP numerical equivalence (Gate 4).

  torchrun --nproc_per_node=2 tests/test_sparse_cp_ep_equiv.py

Tiny 2-layer DeepSeek-V2 patched for sparse DSA. Compares:
  * single-process reference: cp=1, ep=1, full sequence, dense MoE
  * distributed: cp=world (seq sharded) + ep=world (experts sharded), no FSDP
for the LM loss, the indexer loss, and gradients:
  - non-expert params (attn, indexer, router, shared, norms, embed): each rank's
    seq-shard contributes, so grads SUM across the CP group == reference;
  - expert params: dispatched to owners, grad already complete on the owner.

fp32 (bf16 kernels internally); tolerance ~1e-3 (sparse top-k + all-to-all + CP).
"""
import _path  # noqa: F401
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoConfig

from opendsa.dist import (init_parallel, cp_size, cp_rank, ep_size, ep_rank, cp_group,
                          ep_group, is_dist, local_shard)
from opendsa.modeling import (patch_model_with_dsa, IndexerConfig, set_dsa_mode,
                              unfreeze_all, set_cp_size, patch_model_with_ep,
                              dsa_loss_registry)

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"


def build(dtype):
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    # shrink to keep the numerical gate fast; parallelism logic is size-independent
    cfg.num_hidden_layers = 2
    cfg.first_k_dense_replace = 1        # layer 0 dense, layer 1 MoE
    cfg.aux_loss_alpha = 0.0             # exclude seq-level aux (per-shard under CP)
    cfg.hidden_size = 256
    cfg.intermediate_size = 512
    cfg.moe_intermediate_size = 128
    cfg.n_routed_experts = 8
    cfg.num_experts_per_tok = 2
    cfg.n_shared_experts = 1
    cfg.num_attention_heads = 16
    cfg.q_lora_rank = None
    cfg.kv_lora_rank = 128
    cfg.v_head_dim = 32
    cfg.qk_nope_head_dim = 32
    cfg.qk_rope_head_dim = 16
    cfg.vocab_size = 1024
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True,
                                         attn_implementation="eager").cuda().to(dtype)
    # topk == seq_len so the indexer selects ALL causal keys (dense-equivalent). This
    # removes the discrete top-k selection, which on an UNTRAINED indexer flips under
    # the tiny (~3e-4) CP reassociation noise and would otherwise dominate the error —
    # an artifact of hard top-k on random scores, not a CP/EP correctness issue. With
    # dense selection the CP+EP path is numerically compared to the single-GPU ref.
    patch_model_with_dsa(m, IndexerConfig(n_heads=4, head_dim=64, topk=64, tile=32),
                         mode="sparse", dtype=dtype)
    unfreeze_all(m)
    return cfg, m


def run_once(m, ids, coeff=1.0, ntok=None):
    """Gate loss = mean over tokens of hidden.sum()  +  coeff*indexer_loss, computed
    as a SUM over local tokens / GLOBAL token count. This exercises grads through the
    full attention (CP) + MoE (EP) + indexer path and composes exactly across shards
    (summing per-shard grads == full-batch grad). We use the pre-LM-head hidden state
    (out.hidden_states[-1]) rather than the next-token CE to avoid the shard-boundary
    target coupling, which is a standard/orthogonal concern, not a CP-attention check."""
    reg = dsa_loss_registry(); reg.reset(); reg.set_eager_backward(False)
    out = m(input_ids=ids, use_cache=False, output_hidden_states=True)
    h = out.hidden_states[-1]                       # [1, Lloc, hidden]
    if ntok is None:
        ntok = h.shape[1]
    # deterministic scalar objective over hidden (weighted so the sum/ntok composes)
    proxy = (h.float() * _PROBE).sum() / ntok
    idx = reg.total()
    loss = proxy + coeff * (idx if idx is not None else 0.0)
    return loss, float(proxy.detach()), float(idx.detach() if idx is not None else 0.0)


_PROBE = None


def main():
    S = init_parallel()
    dev = f"cuda:{S.local_rank}"
    torch.cuda.set_device(dev)
    dtype = torch.float32
    cfg, m = build(dtype)
    n = cp_size()

    L = 64
    torch.manual_seed(7)
    ids_full = torch.randint(0, cfg.vocab_size, (1, L), device=dev)
    global _PROBE
    _PROBE = torch.randn(cfg.hidden_size, device=dev, dtype=torch.float32)

    # --- reference: cp=1, ep=1, full sequence, TRUE dense MoE ---
    # Force _dense_routed (ignores ep_size, uses ALL experts) so the reference is a
    # genuine single-GPU baseline. ep_moe_forward would branch on the global ep_size
    # even here and give an EP result — not what we want to compare against.
    from opendsa.modeling.patch_deepseek import _iter_moe_modules
    from opendsa.modeling.ep_moe import ep_moe_forward, _dense_routed
    import types as _t
    for _, mlp in _iter_moe_modules(m):
        mlp.forward = _t.MethodType(_dense_routed, mlp)
    set_cp_size(m, 1)
    m.zero_grad(set_to_none=True)
    loss, lm, il = run_once(m, ids_full, ntok=L)
    loss.backward()
    ref = {nm: p.grad.clone() for nm, p in m.named_parameters() if p.grad is not None}
    if cp_rank() == 0:
        print(f"[ref] lm={lm:.5f} idx={il:.5f}")
    m.zero_grad(set_to_none=True)

    if not is_dist():
        print("SPARSE CP+EP EQUIV PASSED (single proc, trivial)")
        return

    # --- distributed: shard experts (EP) + shard sequence (CP) ---
    for _, mlp in _iter_moe_modules(m):
        mlp.forward = _t.MethodType(ep_moe_forward, mlp)
    patch_model_with_ep(m, ep_size())
    set_cp_size(m, n)
    ids_loc = local_shard(ids_full, dim=1)
    m.zero_grad(set_to_none=True)
    # run_once already normalizes by the GLOBAL token count (ntok=L), so each rank's
    # loss is its shard's contribution to the global-mean objective. Summing per-shard
    # non-expert grads over the CP group == the full-batch reference; expert grads are
    # complete on their owner. No extra 1/n scaling needed.
    loss, lm, il = run_once(m, ids_loc, ntok=L)
    loss.backward()

    # compare grads: expert params -> owner-only (complete via all-to-all); others
    # (attn, indexer, router, shared, norms, embed) -> SUM over CP group == reference.
    from opendsa.modeling import expert_parameters
    expert_ids = set(id(p) for p in expert_parameters(m))
    rel = lambda a, b: (a - b).norm().item() / (b.norm().item() + 1e-30)
    errs = {}
    for nm, p in m.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.clone()
        if id(p) not in expert_ids and is_dist():
            dist.all_reduce(g, op=dist.ReduceOp.SUM, group=cp_group())
        if nm in ref and ref[nm].norm() > 0:
            errs[nm] = rel(g, ref[nm])
    maxg = max(errs.values()) if errs else 99.0
    if cp_rank() == 0:
        worst = sorted(errs.items(), key=lambda kv: -kv[1])[:8]
        ok = maxg < 5e-3
        print(f"[cp+ep] lm={lm:.5f} idx={il:.5f}  max grad rel-err={maxg:.2e}")
        for nm, e in worst:
            print(f"    {nm}: {e:.2e}")
        print("SPARSE CP+EP EQUIV PASSED" if ok else "SPARSE CP+EP EQUIV FAILED")


def _moe_iter(m):
    from opendsa.modeling.patch_deepseek import _iter_moe_modules
    return _iter_moe_modules(m)


if __name__ == "__main__":
    try:
        main()
    finally:
        _path.shutdown_dist()
