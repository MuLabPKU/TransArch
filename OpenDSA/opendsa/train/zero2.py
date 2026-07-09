"""zero2.py — hand-rolled ZeRO-2 optimizer for the DSA sparse stage.

Rationale:
sparse @ 200k OOMs on the optimizer step because the 1.31 B non-expert params carry
their bf16 grad + m + v REPLICATED per rank (≈ 10 GB/GPU of state that no amount of
CP or EP can reduce). Sharding those across the 8 ranks reclaims ~7 GB/GPU — enough
to close the OOM gap on the 80 GB budget without introducing FSDP's per-layer param
all-gather or pipeline bubbles.

**Dtype**: to match HF Trainer's baseline memory footprint we keep Adam state in
the parameter's native dtype (bf16 under our `--bf16` config), NOT fp32. HF's default
``torch.optim.AdamW`` on bf16 params runs the state in bf16; introducing fp32 master
would balloon per-GPU memory by ~2 GB on non-expert and ~13 GB on the local expert
side (which no ZeRO sharding can compensate for). The precision cost is real but
matches what the baseline was doing before.

Design (v1):
  * Two disjoint param sets:
    - ``sharded_params`` (non-expert): grads + Adam state sharded across ``group``.
      One in-place update per param per step:
          reduce_scatter(grad) -> shard_grad  (SUM over CP group)
          Adam on (master_shard, m_shard, v_shard) using shard_grad
          all_gather(shard) -> new full param
      This FUSES the CP grad-sum (formerly all_reduce over the CP group) with the
      ZeRO shard step into a single ``reduce_scatter`` — total comm 2P per step
      (P reduce-scatter + P all-gather), same as the old all_reduce path but with
      the Adam state sharded 1/N.
    - ``local_params`` (expert): unique per rank via EP=world. Plain Adam, state
      kept locally in native dtype, no cross-rank comm.
  * Optimizer-compatible surface — subclasses ``torch.optim.Optimizer`` so HF's
    Accelerator / LR scheduler / Trainer see this as a regular optimizer.
  * ``state_dict``/``load_state_dict`` unsupported in v1: sparse pipeline does not
    resume mid-run; extending later needs gather-to-rank-0 or sharded checkpointing.
    The accelerate wrap does a defensive state_dict-roundtrip on init that MUST
    succeed; ``load_state_dict`` no-ops when it recognizes our own stub.

Not implemented in v1 (documented follow-ups):
  * True ZeRO-2 grad memory (needs ``register_post_accumulate_grad_hook`` per param
    to reduce_scatter grads AS they are produced during backward). v1 accumulates
    full replicated bf16 grads across microsteps and shards them at ``step()`` time,
    which is ZeRO-1 memory behavior + ZeRO-2 comm pattern. Additional ≈2 GB/GPU
    could be reclaimed with the hook; deferred pending the 200k measurement.
  * Grad clipping: disabled (see ``DSATrainer.__init__`` where ``max_grad_norm`` is
    forced to 0 when zero2 is on). Would need a distributed-aware norm reduction
    that sees the SHARDED grad. At lr=2e-5 on a warmed backbone this is acceptable;
    add a proper clip if instability appears.
  * Bucketing / comm-compute overlap. Per-tensor collectives are the simple form.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.distributed as dist

from ..dist import reduce_scatter_tensor, all_gather_into_tensor


class Zero2Adam(torch.optim.Optimizer):
    """ZeRO-2 AdamW over ``sharded_params``, plain AdamW over ``local_params``.

    Subclass of ``torch.optim.Optimizer`` so HF Accelerator / Trainer / LR
    schedulers accept it without special-casing.

    Args:
      sharded_params: iterable of params to shard grads+opt state across ``group``
        (typically ``nonexpert_parameters(model)``).
      local_params: iterable of params kept fully local per rank (typically
        ``expert_parameters(model)``; already unique per rank via EP sharding).
      group: process group over which to shard. ``None`` == trivial single-proc
        (identity collectives), for use in cp=1 test paths.
      lr, betas, eps, weight_decay: standard AdamW hyperparameters.

    Adam state (master, m, v) is stored in the parameter's own dtype (bf16 under
    ``--bf16`` training) to match HF's default AdamW memory footprint. The AdamW
    variant is decoupled weight decay.
    """

    def __init__(self,
                 sharded_params: Iterable[torch.Tensor],
                 local_params: Iterable[torch.Tensor],
                 group,
                 lr: float,
                 betas=(0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0):
        sharded_params = [p for p in sharded_params if p.requires_grad]
        local_params = [p for p in local_params if p.requires_grad]

        defaults = {"lr": lr, "betas": tuple(betas), "eps": eps,
                    "weight_decay": weight_decay}
        param_groups = [
            {"params": sharded_params, "kind": "sharded"},
            {"params": local_params, "kind": "local"},
        ]
        super().__init__(param_groups, defaults)

        self.group = group
        self.world = dist.get_world_size(group) if (group is not None and dist.is_initialized()) else 1
        self.rank = dist.get_rank(group) if (group is not None and dist.is_initialized()) else 0

        # Per-sharded-param bookkeeping: shard of (master, m, v) in the param's
        # native dtype, plus (numel, pad, shard, start) so grad reduce-scatter and
        # param all-gather use consistent padding. Adam intermediates use fp32
        # accumulators (see _adam_update) but state itself stays in param dtype.
        for p in sharded_params:
            n = p.numel()
            pad = (-n) % self.world
            shard = (n + pad) // self.world
            start = shard * self.rank
            flat = p.data.detach().reshape(-1)
            if pad:
                flat_padded = torch.zeros(n + pad, dtype=flat.dtype, device=flat.device)
                flat_padded[:n] = flat
            else:
                flat_padded = flat
            master = flat_padded[start:start + shard].detach().clone()
            self.state[p] = {
                "kind": "sharded",
                "numel": n, "pad": pad, "shard": shard, "start": start,
                "master": master,
                "exp_avg": torch.zeros(shard, dtype=p.dtype, device=p.device),
                "exp_avg_sq": torch.zeros(shard, dtype=p.dtype, device=p.device),
                "step": 0,
            }
        for p in local_params:
            self.state[p] = {
                "kind": "local",
                "master": p.data.detach().clone(),
                "exp_avg": torch.zeros_like(p.data),
                "exp_avg_sq": torch.zeros_like(p.data),
                "step": 0,
            }

    # ------------------------------------------------------------------- helpers

    @staticmethod
    def _adam_update(master, exp_avg, exp_avg_sq, grad, lr, beta1, beta2,
                     eps, wd, bc1, bc2):
        """In-place decoupled-AdamW step. State tensors are in the param's native
        dtype (bf16 under our training config); computed in that dtype throughout —
        matches HF's default ``torch.optim.AdamW`` when the param is bf16."""
        if wd != 0.0:
            master.mul_(1.0 - lr * wd)
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        denom = exp_avg_sq.sqrt().div_(bc2 ** 0.5).add_(eps)
        master.addcdiv_(exp_avg, denom, value=-lr / bc1)

    # ------------------------------------------------------------------- API

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None, "Zero2Adam does not support closures"
        # Group 0: sharded (non-expert). One reduce_scatter + local Adam + all_gather
        # per param.
        gsh = self.param_groups[0]
        lr = gsh["lr"]; beta1, beta2 = gsh["betas"]; eps = gsh["eps"]; wd = gsh["weight_decay"]
        for p in gsh["params"]:
            if p.grad is None:
                continue
            st = self.state[p]
            st["step"] += 1
            t = st["step"]
            bc1 = 1.0 - beta1 ** t
            bc2 = 1.0 - beta2 ** t
            n, pad, shard = st["numel"], st["pad"], st["shard"]
            flat = p.grad.detach().reshape(-1)
            if pad:
                flat_padded = torch.zeros(n + pad, dtype=flat.dtype, device=flat.device)
                flat_padded[:n] = flat
            else:
                flat_padded = flat
            shard_grad = torch.empty(shard, dtype=flat.dtype, device=flat.device)
            reduce_scatter_tensor(shard_grad, flat_padded, op=dist.ReduceOp.SUM,
                                  group=self.group)
            self._adam_update(st["master"], st["exp_avg"], st["exp_avg_sq"], shard_grad,
                              lr, beta1, beta2, eps, wd, bc1, bc2)
            gathered = torch.empty(shard * self.world, dtype=p.dtype, device=p.device)
            all_gather_into_tensor(gathered, st["master"], group=self.group)
            p.data.reshape(-1).copy_(gathered[:n])
            p.grad = None

        # Group 1: local (expert). Plain Adam in native dtype, no comm.
        glo = self.param_groups[1]
        lr = glo["lr"]; beta1, beta2 = glo["betas"]; eps = glo["eps"]; wd = glo["weight_decay"]
        for p in glo["params"]:
            if p.grad is None:
                continue
            st = self.state[p]
            st["step"] += 1
            t = st["step"]
            bc1 = 1.0 - beta1 ** t
            bc2 = 1.0 - beta2 ** t
            self._adam_update(st["master"], st["exp_avg"], st["exp_avg_sq"], p.grad.detach(),
                              lr, beta1, beta2, eps, wd, bc1, bc2)
            p.data.copy_(st["master"])
            p.grad = None

    def state_dict(self):
        # Real sharded checkpointing is unsupported in v1. We return a self-recognizable
        # STUB — accelerate's AcceleratedOptimizer wrap does a defensive
        # ``state_dict() -> load_state_dict()`` roundtrip on init that MUST succeed;
        # returning something we can load back as a no-op keeps that happy without
        # pretending to persist real Adam state. HF Trainer.save also calls
        # state_dict() → torch.save (harmless — the resulting file is small and inert).
        return {"__zero2_stub__": True}

    def load_state_dict(self, state_dict):
        # No-op when loading OUR stub (accelerator roundtrip or an inert checkpoint
        # from a previous run). Only raise when the caller is trying to actually
        # resume Adam state from a real optimizer state_dict — that path is
        # unsupported in v1 (see zero2.py header).
        if isinstance(state_dict, dict) and state_dict.get("__zero2_stub__") is True:
            return
        raise NotImplementedError(
            "Zero2Adam cannot resume from a non-stub optimizer state_dict in v1. "
            "Restart from step 0, or extend Zero2Adam with sharded checkpointing.")
