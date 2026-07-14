"""topk_select.py — indexer top-k key selection + sparse-stage indexer KL.

Two pieces used by the DSA sparse-training stage:

  1. ``indexer_select_topk`` — compute the Lightning-Indexer score I[t,j] and
     return, per query, the top-k causal key indices (int32, -1 padded). This
     drives which keys the sparse MLA attends to. No grad (a hard selection).

  2. ``sparse_kl_chunked`` — the indexer keeps learning during sparse training:
     KL( teacher_over_selected ‖ softmax(I over selected) ), i.e. the distillation
     KL restricted to each query's selected top-k keys (not the full causal range).
     Chunked over the query axis + gradient-checkpointed so the per-chunk gathered
     ``[q_chunk, K, ...]`` tensors are never all resident.

The indexer score matches the warmup student exactly:
    I[t,j] = ( Σ_h w[t,h] · ReLU(<qf[t,h], kf[j]>) ) · sm_scale_index
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    from .flashkl_warmup import prepare_ks_ke
except ImportError:  # allow importing this module directly
    from flashkl_warmup import prepare_ks_ke


@torch.no_grad()
def indexer_select_topk(
    index_q, index_k, index_w, cu_seqlens, topk, *,
    sm_scale_index: Optional[float] = None,
    q_chunk: int = 4096,
    q_global_pos=None,
) -> torch.Tensor:
    """Per-query top-k causal key indices via the indexer score. Returns
    [Lq, topk] int32 (-1 padded where fewer than topk causal keys exist); indices are
    absolute key positions over [0, Lk].

    Non-CP: index_q and index_k share L, causal from cu_seqlens.
    CP: index_q holds Lq local queries, index_k holds the FULL Lk gathered keys; pass
    ``q_global_pos`` [Lq] (true global position) so causal bound = pos+1 and indices
    are GLOBAL. Chunked over queries to bound the [q_chunk, Lk] score buffer."""
    Lq, Hf, df = index_q.shape
    Lk = index_k.shape[0]
    dev = index_q.device
    sf = sm_scale_index if sm_scale_index is not None else df ** -0.5
    if q_global_pos is None:
        ks, ke = prepare_ks_ke(cu_seqlens)
    else:
        from .flashkl_warmup import prepare_ks_ke_cp
        ks, ke = prepare_ks_ke_cp(cu_seqlens, q_global_pos)
    jj = torch.arange(Lk, device=dev)
    out = torch.full((Lq, topk), -1, dtype=torch.int32, device=dev)

    for t0 in range(0, Lq, q_chunk):
        t1 = min(t0 + q_chunk, Lq)
        qc = index_q[t0:t1]                                   # [c,Hf,df]
        sh = torch.einsum("thd,jd->thj", qc, index_k)         # [c,Hf,Lk]
        sh.relu_()
        sh.mul_(index_w[t0:t1].unsqueeze(-1))
        I = sh.sum(1).mul_(sf)                                # [c,Lk]
        causal = (jj.view(1, Lk) >= ks[t0:t1].view(-1, 1)) & (jj.view(1, Lk) < ke[t0:t1].view(-1, 1))
        I = I.masked_fill(~causal, float("-inf"))
        k_eff = min(topk, Lk)
        vals, idx = torch.topk(I, k_eff, dim=-1)              # [c,k_eff]
        # invalidate picks that were -inf (fewer than topk causal keys)
        idx = torch.where(torch.isfinite(vals), idx.to(torch.int32),
                          torch.full_like(idx, -1, dtype=torch.int32))
        out[t0:t1, :k_eff] = idx
    return out


# --------------------------------------------------------------------------- #
#  memory-bounded sparse KL (query-chunked + gradient-checkpointed)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _teacher_pbar_chunk(Qm_c, Km, ids_c, sm_tea):
    """Head-averaged teacher distribution p̄[c,K] over the selected keys, computed
    under no_grad (teacher is frozen/detached). Kept OUTSIDE the checkpointed student
    region so its key-gather + einsum runs ONCE (fwd only), not again on every
    backward recompute — this is the main sparse-KL speedup. ``Km`` may be shared
    latent [Lk,D] (the absorbed default) or per-head [Lk,H,D]."""
    c, K = ids_c.shape
    valid = ids_c >= 0
    idc = ids_c.clamp(min=0)
    if Km.dim() == 3:
        a = torch.einsum("thd,tphd->thp", Qm_c, Km[idc]) * sm_tea   # per-head K
    else:
        a = torch.einsum("thd,tpd->thp", Qm_c, Km[idc]) * sm_tea    # shared K
    a = a.masked_fill(~valid.view(c, 1, K), float("-inf"))
    pbar = torch.nan_to_num(torch.softmax(a, dim=-1), nan=0.0).mean(1)  # [c,K]
    return torch.where(valid, pbar, torch.zeros_like(pbar))


def _sparse_kl_chunk_student(qf_c, wf_c, kf, ids_c, pbar, sm_f):
    """Trainable sparse-KL for one query chunk given the precomputed teacher p̄.
    Only the student score depends on the indexer params, so this is the sole part
    that must be (re)differentiated; teacher entropy is dropped (constant), so the
    value differs from the true KL by a constant but has identical indexer grads.

    qf_c [c,Hf,df], wf_c [c,Hf], kf [L,df], ids_c [c,K] int64 (-1 pad), pbar [c,K].
    The [c,K,df] student gather is the only big intermediate recomputed on backward
    under checkpoint — small (Hf indexer heads at df, e.g. 8×128) vs the teacher
    gather kept outside."""
    c, K = ids_c.shape
    valid = ids_c >= 0
    idc = ids_c.clamp(min=0)
    kg = kf[idc]                                                    # [c,K,df]
    dd = torch.einsum("thd,tpd->tph", qf_c, kg)                     # [c,K,Hf]
    S = (torch.relu(dd) * wf_c.unsqueeze(1)).sum(-1) * sm_f         # [c,K]
    Sm = torch.where(valid, S, torch.full_like(S, float("-inf")))
    lse = torch.logsumexp(Sm, dim=-1)                              # [c]
    cross = (pbar * torch.where(valid, S, torch.zeros_like(S))).sum(-1)  # [c]
    vr = valid.any(-1)
    rows = torch.where(vr, lse - cross, torch.zeros_like(lse))
    return rows.sum()


def sparse_kl_chunked(index_q, index_k, index_w, Q_main, K_main, ids, *,
                      sm_scale_teacher=None, sm_scale_index=None,
                      q_chunk: int = 512, checkpoint: bool = True, nrow_global=None):
    """Memory-bounded sparse-stage indexer KL over each query's selected top-k keys.
    Streams the query axis so the per-chunk gathered ``[q_chunk, K, ...]`` tensors are
    never all resident. Teacher (Q_main/K_main)
    is detached; its head-averaged distribution is computed once per chunk under
    no_grad (not recomputed on backward). Only the cheap student score is
    gradient-checkpointed.

    ``nrow_global``: normalize by this global query count instead of the local valid
    row count (CP: so summing per-shard losses/grads over the CP group == the true
    global mean).

    Returns the mean-over-valid-rows trainable loss (teacher entropy dropped)."""
    from torch.utils.checkpoint import checkpoint as _ckpt

    L = index_q.shape[0]
    st = sm_scale_teacher if sm_scale_teacher is not None else Q_main.shape[-1] ** -0.5
    sf = sm_scale_index if sm_scale_index is not None else index_q.shape[-1] ** -0.5
    Qm = Q_main.detach()
    Km = K_main.detach()
    ids64 = ids.to(torch.int64)
    valid = ids64 >= 0
    nrow = (float(nrow_global) if nrow_global is not None
            else valid.any(-1).to(index_q.dtype).sum().clamp_min(1.0))

    use_ckpt = checkpoint and torch.is_grad_enabled() and index_q.requires_grad
    total = index_q.new_zeros(())
    for t0 in range(0, L, q_chunk):
        t1 = min(t0 + q_chunk, L)
        ids_c = ids64[t0:t1]
        pbar = _teacher_pbar_chunk(Qm[t0:t1], Km, ids_c, float(st))
        args = (index_q[t0:t1], index_w[t0:t1], index_k, ids_c, pbar, float(sf))
        if use_ckpt:
            total = total + _ckpt(_sparse_kl_chunk_student, *args, use_reentrant=False)
        else:
            total = total + _sparse_kl_chunk_student(*args)
    return total / nrow

