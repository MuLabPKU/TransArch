"""pg.py — process groups + collectives for OpenDSA context/expert parallelism.

On a single 8-GPU node we run **CP and EP on the SAME ranks** (CP=EP=world, DP=1):
attention treats the group as context-parallel (each rank holds L/N tokens of one
sequence), the MoE treats it as expert-parallel (each rank owns 64/N experts). This
keeps the long-context path to a single data-parallel replica with no FSDP.

Init from the launcher env (torchrun / accelerate set RANK, WORLD_SIZE, LOCAL_RANK).
When WORLD_SIZE is unset or 1 we run in a trivial single-process mode where every
collective is the identity — so the exact same code path runs on 1 GPU (the
numerical-equivalence baseline) and on N.

Sequence sharding is **contiguous**: rank r owns global positions
    [r * Lloc, (r+1) * Lloc),  Lloc = L / cp_size
(zigzag load-balancing is a later speed tweak; with KV all-gather correctness is
identical). The autograd-aware ``AllGatherSeq`` gathers a sharded [Lloc, ...] tensor
to the full [L, ...] on every rank (forward) and reduce-scatters the gradient back to
each rank's shard (backward), so student tensors that require grad can be gathered.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


# --------------------------------------------------------------------------- #
#  global state (set once by init_parallel)
# --------------------------------------------------------------------------- #
class _State:
    initialized = False
    world = 1
    rank = 0
    local_rank = 0
    cp_group = None      # process group for context parallel (None == trivial)
    ep_group = None      # process group for expert parallel
    cp_size = 1
    ep_size = 1


_S = _State()


def init_parallel(cp_size: Optional[int] = None, ep_size: Optional[int] = None):
    """Initialize torch.distributed (if launched with >1 process) and build the CP
    and EP groups. Idempotent. If ``WORLD_SIZE`` is 1/unset, stays in trivial mode.

    cp_size / ep_size default to the full world (single-node CP=EP=world). We only
    support the CP==EP==world, DP=1 layout for now (assert otherwise).
    """
    if _S.initialized:
        return _S
    world = int(os.environ.get("WORLD_SIZE", "1"))
    _S.world = world
    _S.rank = int(os.environ.get("RANK", "0"))
    _S.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world <= 1:
        # trivial single-process mode: collectives are identities
        _S.cp_size = 1
        _S.ep_size = 1
        _S.initialized = True
        return _S

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(_S.local_rank)

    cp = cp_size or world
    ep = ep_size or world
    assert cp == world and ep == world, (
        f"only CP==EP==world (DP=1) supported for now; got cp={cp} ep={ep} world={world}")
    # single group spanning all ranks serves as both CP and EP
    _S.cp_group = dist.new_group(ranks=list(range(world)))
    _S.ep_group = _S.cp_group
    _S.cp_size = cp
    _S.ep_size = ep
    _S.initialized = True
    return _S


def is_dist() -> bool:
    return _S.world > 1


def cp_size() -> int:
    return _S.cp_size


def cp_rank() -> int:
    return _S.rank if _S.cp_size > 1 else 0


def ep_size() -> int:
    return _S.ep_size


def ep_rank() -> int:
    return _S.rank if _S.ep_size > 1 else 0


def cp_group():
    return _S.cp_group


def ep_group():
    return _S.ep_group


# --------------------------------------------------------------------------- #
#  sequence sharding (contiguous)
# --------------------------------------------------------------------------- #
def local_shard(x: torch.Tensor, dim: int = 0):
    """Return this CP rank's contiguous shard of ``x`` along ``dim``. No-op if cp==1.
    Requires x.shape[dim] % cp_size == 0."""
    n = cp_size()
    if n == 1:
        return x
    L = x.shape[dim]
    assert L % n == 0, f"seq len {L} not divisible by cp_size {n}"
    Lloc = L // n
    start = cp_rank() * Lloc
    return x.narrow(dim, start, Lloc).contiguous()


def seq_offset(L_local: int) -> int:
    """Global position of this rank's first local token (contiguous split)."""
    return cp_rank() * L_local


# --------------------------------------------------------------------------- #
#  collectives
# --------------------------------------------------------------------------- #
def _all_gather_dim0(x: torch.Tensor) -> torch.Tensor:
    """Plain all-gather of an [Lloc, ...] tensor along dim0 -> [Lloc*cp, ...].
    Assumes every rank contributes the same shape (contiguous even split)."""
    n = cp_size()
    if n == 1:
        return x
    xc = x.contiguous()
    out = [torch.empty_like(xc) for _ in range(n)]
    dist.all_gather(out, xc, group=cp_group())
    return torch.cat(out, dim=0)


class AllGatherSeq(torch.autograd.Function):
    """Autograd-aware all-gather along the sequence dim (dim0). Forward gathers the
    local [Lloc,...] shard into the full [L,...] on every rank; backward reduce-
    scatters the full-sequence gradient back to each rank's own shard (summing the
    contributions this rank's tokens received as keys on every rank).

    Use for the STUDENT indexer key that must carry grad. For detached/no-grad
    tensors (teacher latent) just call ``all_gather_seq`` (no autograd needed)."""

    @staticmethod
    def forward(ctx, x):
        ctx.cp = cp_size()
        ctx.Lloc = x.shape[0]
        return _all_gather_dim0(x)

    @staticmethod
    def backward(ctx, grad_full):
        n = ctx.cp
        if n == 1:
            return grad_full
        # sum-reduce the full-seq grad across ranks, then take this rank's slice
        g = grad_full.contiguous()
        dist.all_reduce(g, op=dist.ReduceOp.SUM, group=cp_group())
        start = cp_rank() * ctx.Lloc
        return g.narrow(0, start, ctx.Lloc)


def all_gather_seq(x: torch.Tensor, grad: bool = False) -> torch.Tensor:
    """All-gather along the sequence dim. ``grad=True`` routes through AllGatherSeq
    (autograd-aware); otherwise a plain no-grad gather."""
    if grad:
        return AllGatherSeq.apply(x)
    return _all_gather_dim0(x)


def all_to_all(x: torch.Tensor, group=None) -> torch.Tensor:
    """all-to-all over ``group`` (default ep_group). Splits ``x`` along dim0 into N
    equal chunks, sends chunk i to rank i, returns the concatenation of received
    chunks. Requires x.shape[0] % N == 0. No-op if N==1."""
    g = group or ep_group()
    n = ep_size()
    if n == 1:
        return x
    assert x.shape[0] % n == 0, f"all_to_all dim0 {x.shape[0]} not divisible by {n}"
    xc = x.contiguous()
    out = torch.empty_like(xc)
    dist.all_to_all_single(out, xc, group=g)
    return out


def reduce_grads(params, group=None, op=dist.ReduceOp.SUM, scale: float = 1.0):
    """In-place all-reduce of .grad over ``group`` for the given params (skips None).
    Optional ``scale`` multiplies each grad after reduction (e.g. 1/global_tokens)."""
    if not is_dist():
        if scale != 1.0:
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(scale)
        return
    g = group or cp_group()
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=op, group=g)
            if scale != 1.0:
                p.grad.mul_(scale)


def reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor,
                          op=dist.ReduceOp.SUM, group=None):
    """Thin wrapper over ``torch.distributed.reduce_scatter_tensor``. Requires
    ``input.numel() == output.numel() * world_size``. Used by ZeRO-2 to fuse
    "sum-across-ranks" and "shard-to-owner" into one collective — replaces the
    all_reduce that the CP grad-sum used to do."""
    g = group or cp_group()
    if not is_dist():
        assert input.numel() == output.numel(), (
            f"reduce_scatter identity: input {input.numel()} != output {output.numel()}")
        output.copy_(input)
        return
    dist.reduce_scatter_tensor(output, input.contiguous(), op=op, group=g)


def all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor, group=None):
    """Thin wrapper over ``torch.distributed.all_gather_into_tensor``. Requires
    ``output.numel() == input.numel() * world_size``. Used by ZeRO-2 to reassemble
    the updated bf16 param from per-rank shards after the local Adam step."""
    g = group or cp_group()
    if not is_dist():
        assert input.numel() == output.numel(), (
            f"all_gather identity: input {input.numel()} != output {output.numel()}")
        output.copy_(input)
        return
    dist.all_gather_into_tensor(output, input.contiguous(), group=g)


def barrier():
    if is_dist():
        dist.barrier(group=cp_group())
