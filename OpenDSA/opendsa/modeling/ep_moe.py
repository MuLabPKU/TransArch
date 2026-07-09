"""ep_moe.py — expert-parallel MoE for DeepSeek-V2, drop-in for DeepseekV2MoE.

Replaces the routed-expert computation of a patched ``DeepseekV2MoE`` with an
expert-parallel version: rank r owns experts ``[r*E/ep : (r+1)*E/ep]``. The router
(``gate``) and ``shared_experts`` are reused UNCHANGED (so routing is bit-identical
to the base model), only the routed expert MLPs are sharded + dispatched via
all-to-all.

Correctness contract: the output equals the single-rank dense MoE
(``_dense_routed`` mirrors the HF training path exactly), verified numerically by
``tests/test_ep_equiv.py`` (ep2 == ep1).

Dispatch (per forward):
  1. gate -> topk_idx [N,k], topk_weight [N,k]  (N = local tokens)
  2. flatten to N*k (token, expert) assignments; sort by OWNER RANK
  3. all-to-all the sorted tokens so each rank receives exactly the assignments whose
     expert it owns
  4. run local experts on received tokens
  5. all-to-all the results back; unsort; weight by topk_weight; sum over k
  6. add shared-expert output (local, all tokens)

Everything is autograd-transparent (all_to_all_single has a backward); expert params
get grads only from their routed tokens, which is exactly right — no cross-rank
reduction of expert grads. The router/shared params are non-expert and reduced with
the rest of the backbone by the trainer.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from ..dist import ep_size, ep_rank, ep_group, is_dist


class _AllToAllVar(torch.autograd.Function):
    """Autograd-aware all-to-all with per-rank variable split sizes along dim0.
    in_splits[i] rows go to rank i; out_splits[i] rows come from rank i."""

    @staticmethod
    def forward(ctx, x, out_splits, in_splits, group):
        ctx.out_splits = out_splits
        ctx.in_splits = in_splits
        ctx.group = group
        out = x.new_empty((sum(out_splits),) + tuple(x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(),
                               output_split_sizes=out_splits,
                               input_split_sizes=in_splits, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # reverse: swap the split roles
        grad_in = grad_out.new_empty((sum(ctx.in_splits),) + tuple(grad_out.shape[1:]))
        dist.all_to_all_single(grad_in, grad_out.contiguous(),
                               output_split_sizes=ctx.in_splits,
                               input_split_sizes=ctx.out_splits, group=ctx.group)
        return grad_in, None, None, None


def _all_to_all_var(x, out_splits, in_splits, group):
    return _AllToAllVar.apply(x, out_splits, in_splits, group)


def ep_moe_forward(self, hidden_states):
    """Expert-parallel replacement for DeepseekV2MoE.forward. ``self`` is the
    original MoE module (gate, experts ModuleList, shared_experts all present on this
    rank, but only this rank's owned experts are actually used)."""
    if not is_dist() or ep_size() == 1:
        return _dense_routed(self, hidden_states)

    identity = hidden_states
    orig_shape = hidden_states.shape
    topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
    x = hidden_states.view(-1, hidden_states.shape[-1])                # [N,h]
    N, h = x.shape
    k = topk_idx.shape[-1]
    ep = ep_size()
    E = len(self.experts)
    per_rank = E // ep
    grp = ep_group()
    dev = x.device

    flat_expert = topk_idx.reshape(-1)                                 # [N*k]
    # gather each of the N*k routed copies of its token
    tok_index = torch.arange(N, device=dev).repeat_interleave(k)       # [N*k]
    owner = (flat_expert // per_rank).clamp(max=ep - 1)                # [N*k] dest rank
    order = torch.argsort(owner)                                       # group by dest rank
    sorted_owner = owner[order]
    sorted_expert = flat_expert[order]
    sorted_tokrow = tok_index[order]
    sent = x[sorted_tokrow]                                            # [N*k,h] sorted by dest

    in_splits = torch.bincount(sorted_owner, minlength=ep).tolist()    # rows to each rank
    # exchange counts so every rank knows how many rows it will receive from each
    in_counts = torch.tensor(in_splits, device=dev, dtype=torch.long)
    out_counts = torch.empty_like(in_counts)
    dist.all_to_all_single(out_counts, in_counts, group=grp)
    out_splits = out_counts.tolist()

    recv = _all_to_all_var(sent, out_splits, in_splits, grp)           # [M,h] tokens I own
    # which expert each received row targets: send the expert ids too
    recv_expert = _all_to_all_var(sorted_expert.to(torch.int64).unsqueeze(-1).float(),
                                  out_splits, in_splits, grp).squeeze(-1).long()
    # run my local experts
    y_local = torch.zeros_like(recv)
    lo = ep_rank() * per_rank
    for e in range(per_rank):
        gid = lo + e
        mask = recv_expert == gid
        if mask.any():
            y_local[mask] = self.experts[gid](recv[mask])
    # send results back (reverse splits)
    y_back = _all_to_all_var(y_local, in_splits, out_splits, grp)      # [N*k,h] sorted order
    # unsort to original (token,k) order
    y_sorted = torch.zeros_like(y_back)
    y_sorted[order] = y_back
    y = (y_sorted.view(N, k, h) * topk_weight.unsqueeze(-1)).sum(1)    # weight + sum over k
    y = y.to(x.dtype).view(*orig_shape)
    try:
        from transformers.models.auto import modeling_auto  # noqa
    except Exception:
        pass
    y = _apply_aux(self, y, aux_loss)
    if getattr(self.config, "n_shared_experts", None) is not None:
        y = y + self.shared_experts(identity)
    return y


def _apply_aux(self, y, aux_loss):
    """Attach the router aux loss the same way HF does (AddAuxiliaryLoss)."""
    mod = type(self).__module__
    import importlib
    m = importlib.import_module(mod)
    AddAux = getattr(m, "AddAuxiliaryLoss", None)
    if AddAux is not None and self.training and aux_loss is not None:
        return AddAux.apply(y, aux_loss)
    return y


def _dense_routed(self, hidden_states):
    """Single-rank dense MoE, mirroring the HF training forward exactly. Used when
    ep==1 (the numerical baseline)."""
    identity = hidden_states
    orig_shape = hidden_states.shape
    topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
    x = hidden_states.view(-1, hidden_states.shape[-1])
    flat = topk_idx.view(-1)
    xr = x.repeat_interleave(topk_idx.shape[-1], dim=0)
    y = torch.empty_like(xr)
    for i, expert in enumerate(self.experts):
        if expert is not None:
            m = flat == i
            if m.any():
                y[m] = expert(xr[m])
    y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(1)
    y = y.to(x.dtype).view(*orig_shape)
    y = _apply_aux(self, y, aux_loss)
    if getattr(self.config, "n_shared_experts", None) is not None:
        y = y + self.shared_experts(identity)
    return y
