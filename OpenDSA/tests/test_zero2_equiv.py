"""test_zero2_equiv.py — Zero2Adam numerical equivalence gate (ZeRO-2 fusion with CP).

  torchrun --nproc_per_node=2 tests/test_zero2_equiv.py

Trajectory equivalence: 3 AdamW steps under Zero2Adam (2 ranks, sharded non-expert
params, local expert params) MUST match 3 steps of plain torch.optim.AdamW using the
SUMMED grad. This validates:
  - reduce_scatter fuses the CP grad-sum with the ZeRO shard step correctly
  - fp32 Adam on the shard yields the same param update as plain AdamW on the full
  - all_gather reassembles the updated param without corrupting the untouched slices
  - local (expert) params behave as plain AdamW (unique per rank)

Tolerance: max ||a-b|| / ||b|| < 1e-5 (fp32). Slight looseness from expression order.
"""
import _path  # noqa: F401
import torch
import torch.distributed as dist
from opendsa.dist import (init_parallel, cp_size, cp_rank, cp_group, is_dist,
                          all_gather_into_tensor)
from opendsa.train.zero2 import Zero2Adam


def clone_params(shapes, dev, seed):
    """Make requires-grad params with a deterministic seed (identical across ranks)."""
    torch.manual_seed(seed)
    return [torch.randn(*s, device=dev, dtype=torch.float32, requires_grad=True)
            for s in shapes]


def rel_err(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-30)


def main():
    S = init_parallel()
    dev = f"cuda:{S.local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(dev)
    world = cp_size()
    rank = cp_rank()

    # Deliberate mix of shapes:
    #  * (37,) — NOT divisible by world (needs padding under Zero2Adam)
    #  * (128, 64) — cleanly divisible
    #  * (17, 5) — NOT divisible (padding again)
    #  * (256,) — cleanly divisible
    # Local (expert-like) shapes: no comm, no shard, just plain Adam.
    sharded_shapes = [(37,), (128, 64), (17, 5), (256,)]
    local_shapes = [(64, 32), (10,)]
    lr, betas, eps, wd = 1e-3, (0.9, 0.999), 1e-8, 0.01

    # --- reference: plain AdamW on ALL params (sharded + local), full grads ---
    ref_sh = clone_params(sharded_shapes, dev, seed=42)
    ref_lo = clone_params(local_shapes, dev, seed=43)
    ref_opt = torch.optim.AdamW(ref_sh + ref_lo, lr=lr, betas=betas, eps=eps,
                                weight_decay=wd)

    # --- test: Zero2Adam over the same starting tensors ---
    z2_sh = clone_params(sharded_shapes, dev, seed=42)   # same init as ref
    z2_lo = clone_params(local_shapes, dev, seed=43)
    z2_opt = Zero2Adam(sharded_params=z2_sh, local_params=z2_lo,
                       group=cp_group() if is_dist() else None,
                       lr=lr, betas=betas, eps=eps, weight_decay=wd)

    for step in range(3):
        # deterministic identical "full" grad on every rank
        torch.manual_seed(1000 + step)
        full_grads_sh = [torch.randn_like(p) for p in ref_sh]
        # local grads: identical on all ranks (in real EP they'd differ per rank
        # since each rank owns different experts; here we just check the Adam math)
        torch.manual_seed(2000 + step)
        full_grads_lo = [torch.randn_like(p) for p in ref_lo]

        # reference step: full grads, plain AdamW
        for p, g in zip(ref_sh + ref_lo, full_grads_sh + full_grads_lo):
            p.grad = g.clone()
        ref_opt.step()
        ref_opt.zero_grad(set_to_none=True)

        # ZeRO-2 step: each rank contributes 1/world of the sharded grad, so
        # reduce_scatter(SUM) reconstructs the full grad on the owned shard.
        # (Mirrors the sparse loss being scaled by 1/cp so per-rank grads sum to
        # the global-mean grad.)
        for p, g in zip(z2_sh, full_grads_sh):
            p.grad = (g / world).clone()
        # local grads: identical (single-rank behaviour under our test)
        for p, g in zip(z2_lo, full_grads_lo):
            p.grad = g.clone()
        z2_opt.step()

    # --- compare after 3 steps ---
    errs = {}
    for i, (r, z) in enumerate(zip(ref_sh, z2_sh)):
        # z is replicated (all-gathered) across ranks after step; compare rank-local.
        errs[f"sharded[{i}] {tuple(r.shape)}"] = rel_err(z.detach(), r.detach())
    for i, (r, z) in enumerate(zip(ref_lo, z2_lo)):
        errs[f"local[{i}] {tuple(r.shape)}"] = rel_err(z.detach(), r.detach())

    max_err = max(errs.values())
    if rank == 0:
        for k, v in errs.items():
            print(f"  {k:30s}  rel-err = {v:.2e}")
        ok = max_err < 1e-5
        print(f"[zero2] max rel-err after 3 steps = {max_err:.2e}")
        print("ZERO2 EQUIV PASSED" if ok else "ZERO2 EQUIV FAILED")
        assert ok, f"Zero2Adam trajectory diverged from plain AdamW: {max_err:.2e}"


if __name__ == "__main__":
    try:
        main()
    finally:
        _path.shutdown_dist()
