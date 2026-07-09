"""test_ep_equiv.py — expert-parallel MoE numerical equivalence (Gate 3).

  torchrun --nproc_per_node=2 tests/test_ep_equiv.py

Correct EP semantics: a GLOBAL batch of tokens is sharded across ranks (each rank
holds N_local = N_global/ep tokens, like the CP sequence shard). We check:
  * forward: gathering the per-rank EP outputs reproduces the dense full-batch output;
  * expert grads: each expert's tokens are all-to-all'd to its owner, so its weight
    grad already equals the dense grad (NO cross-rank sum needed);
  * router / shared grads: computed per-shard, so they SUM across ranks == dense.
All checked in fp64 against a single-process dense reference on the full batch.
"""
import _path  # noqa: F401
import torch
import importlib
from transformers import AutoConfig
import torch.distributed as dist

from opendsa.dist import (init_parallel, ep_size, ep_rank, ep_group, is_dist,
                          local_shard, all_gather_seq)
from opendsa.modeling.ep_moe import ep_moe_forward, _dense_routed

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"


def _deepseek_modeling_module(cfg):
    """Resolve the remote-code modeling module matching this config cache."""
    cfg_mod = type(cfg).__module__
    base = cfg_mod.rsplit(".", 1)[0]
    candidates = [
        f"{base}.modeling_deepseek",
        cfg_mod.replace("configuration_deepseek", "modeling_deepseek"),
    ]
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(mod, "DeepseekV2MoE"):
            return mod
    raise ImportError(f"could not resolve DeepseekV2MoE module from {cfg_mod}")


def main():
    S = init_parallel()
    dev = f"cuda:{S.local_rank}"
    torch.cuda.set_device(dev)
    dtype = torch.float64
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    # The MoE router aux loss uses seq-level token statistics; under token sharding
    # each rank sees only its shard, so the summed aux grad differs from a full-batch
    # aux by design (it's a tiny α=1e-3 load-balancing regularizer, computed per
    # microbatch in normal training too). Disable it here so the gate grad is an exact
    # numerical check of the EP routing/dispatch itself.
    cfg.aux_loss_alpha = 0.0
    m = _deepseek_modeling_module(cfg)
    torch.manual_seed(0)
    moe = m.DeepseekV2MoE(cfg).to(dev).to(dtype)
    moe.train()
    ep = ep_size()

    # global batch (identical on all ranks); each rank will take its shard
    torch.manual_seed(42)
    Ng = 16 * max(ep, 1)
    x_full = torch.randn(1, Ng, cfg.hidden_size, device=dev, dtype=dtype)

    # --- dense reference on the FULL batch ---
    xr = x_full.clone().requires_grad_(True)
    y_ref = _dense_routed(moe, xr)
    y_ref.sum().backward()
    gx_ref = xr.grad.clone()
    gp_ref = {nm: p.grad.clone() for nm, p in moe.named_parameters() if p.grad is not None}
    moe.zero_grad(set_to_none=True)

    # --- EP: shard tokens across ranks, drop non-owned experts ---
    E = len(moe.experts)
    per = E // ep
    lo, hi = ep_rank() * per, (ep_rank() + 1) * per
    if is_dist():
        for e in range(E):
            if not (lo <= e < hi):
                moe.experts[e] = None
    x_loc = local_shard(x_full, dim=1).clone().requires_grad_(True)  # [1,Nloc,h]
    y_loc = ep_moe_forward(moe, x_loc)
    y_loc.sum().backward()

    # gather EP forward outputs -> full, compare to dense
    y_gathered = all_gather_seq(y_loc[0].detach(), grad=False) if is_dist() else y_loc[0].detach()
    rel = lambda a, b: (a - b).norm().item() / (b.norm().item() + 1e-30)
    e_y = rel(y_gathered, y_ref[0])

    # router/shared grads: sum across ranks (per-shard contributions) -> compare dense
    # expert grads: already complete on the owner -> compare owned experts directly
    errs = {}
    for nm, p in moe.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.clone()
        if nm.startswith("experts."):
            eid = int(nm.split(".")[1])
            if not (lo <= eid < hi):
                continue
        else:
            if is_dist():
                dist.all_reduce(g, op=dist.ReduceOp.SUM, group=ep_group())
        if nm in gp_ref:
            errs[nm] = rel(g, gp_ref[nm])
    maxg = max(errs.values()) if errs else 0.0
    n_exp = sum(1 for nm in errs if nm.startswith("experts."))
    n_rt = sum(1 for nm in errs if not nm.startswith("experts."))
    if ep_rank() == 0:
        ok = e_y < 1e-9 and maxg < 1e-6 and n_exp > 0 and n_rt > 0
        print(f"[ep_equiv] ep={ep}  fwd(gathered) rel-err={e_y:.2e}  "
              f"max grad rel-err={maxg:.2e}  (experts checked={n_exp}, router/shared={n_rt})")
        worst = sorted(errs.items(), key=lambda kv: -kv[1])[:5]
        for nm, e in worst:
            print(f"    {nm}: {e:.2e}")
        print("EP EQUIV PASSED" if ok else "EP EQUIV FAILED")


if __name__ == "__main__":
    try:
        main()
    finally:
        _path.shutdown_dist()
