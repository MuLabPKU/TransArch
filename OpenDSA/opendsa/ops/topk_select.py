"""topk_select.py — indexer top-k key selection + sparse-stage indexer KL.

Two pieces used by the DSA sparse-training stage:

  1. ``indexer_select_topk`` — compute the Lightning-Indexer score I[t,j] and
     return, per query, the top-k causal key indices (int32, -1 padded). This
     drives which keys the sparse MLA attends to. No grad (a hard selection).

  2. ``flashkl_sparse_loss`` — the indexer keeps learning during sparse training:
     KL( teacher_over_selected ‖ softmax(I over selected) ). Same FlashKL trick
     as warmup, but the normalization set is the per-query selected top-k rather
     than the full causal range. Reuses the streaming loss on gathered columns.

The indexer score matches the warmup student exactly:
    I[t,j] = ( Σ_h w[t,h] · ReLU(<qf[t,h], kf[j]>) ) · sm_scale_index
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    from .flashkl_warmup import prepare_ks_ke, _acc_dtype
except ImportError:  # allow running this file directly for the self-test
    from flashkl_warmup import prepare_ks_ke, _acc_dtype


def indexer_scores(index_q, index_k, index_w, *, sm_scale_index=None):
    """Dense indexer logits I[t,j] = (Σ_h w·ReLU(<q,k>))·scale. [L,L]. O(L²) —
    fine for short ctx / reference; the training path uses the streaming loss and
    a chunked selection below."""
    Hf, df = index_q.shape[1], index_q.shape[2]
    sf = sm_scale_index if sm_scale_index is not None else df ** -0.5
    sh = torch.einsum("thd,jd->thj", index_q, index_k)      # [L,Hf,L]
    return (torch.relu(sh) * index_w.unsqueeze(-1)).sum(1) * sf


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
#  sparse-stage indexer KL loss (FlashKL over selected columns)
# --------------------------------------------------------------------------- #
class _RefineSparse(torch.autograd.Function):
    """FlashKL over per-query selected keys ``ids`` [L,k] (int, -1 padded).
    teacher/student both normalized over the selected set. Mirrors _RefineWarmup
    but streams the k gathered columns instead of the full causal range."""

    @staticmethod
    def forward(ctx, qf, kf, wf, Qm, Km, ids, sm_tea, sm_f, tile):
        L, H, D = Qm.shape
        df = qf.shape[2]
        dev = Qm.device
        acc = _acc_dtype(Qm.dtype)
        k_eff = ids.shape[1]
        valid = ids >= 0
        vr = valid.any(-1)
        nrow = vr.to(acc).sum().clamp_min(1.0)
        Qmf, Kmf = Qm.to(acc), Km.to(acc)
        qfd, kfd, wfd = qf.to(acc), kf.to(acc), wf.to(acc)

        M = torch.full((L, H), float("-inf"), dtype=acc, device=dev)
        Dt = torch.zeros(L, H, dtype=acc, device=dev)
        N = torch.zeros(L, H, dtype=acc, device=dev)
        m = torch.full((L,), float("-inf"), dtype=acc, device=dev)
        ell = torch.zeros(L, dtype=acc, device=dev)
        for c0 in range(0, k_eff, tile):
            c1 = min(c0 + tile, k_eff)
            b = c1 - c0
            jj = ids[:, c0:c1]
            vb = valid[:, c0:c1]
            jc = jj.long().clamp(min=0)
            kg = kfd[jc]                                       # [L,b,df]
            dd = torch.einsum("thd,tpd->tph", qfd, kg)         # [L,b,Hf]
            S = (torch.relu(dd) * wfd.unsqueeze(1)).sum(-1) * sm_f   # [L,b]
            km = Kmf[jc] if Kmf.dim() == 2 else Kmf[jc]              # [L,b,D] or [L,b,H,D]
            if Kmf.dim() == 3:
                a = torch.einsum("thd,tphd->thp", Qmf, Kmf[jc]) * sm_tea  # per-head K
            else:
                a = torch.einsum("thd,tpd->thp", Qmf, km) * sm_tea       # shared K
            a = a.masked_fill(~vb.view(L, 1, b), float("-inf"))
            Sm = torch.where(vb, S, torch.full_like(S, float("-inf")))
            mn = torch.maximum(m, Sm.max(-1).values)
            beta = torch.exp(m - mn)
            beta = torch.where(torch.isfinite(beta), beta, torch.zeros_like(beta))
            add = torch.where(vb, torch.exp(Sm - mn.unsqueeze(-1)), torch.zeros_like(Sm))
            ell = ell * beta + add.sum(-1)
            m = mn
            Mn = torch.maximum(M, a.max(-1).values)
            rho = torch.exp(M - Mn)
            rho = torch.where(torch.isfinite(rho), rho, torch.zeros_like(rho))
            e = torch.where(vb.view(L, 1, b), torch.exp(a - Mn.unsqueeze(-1)),
                            torch.zeros_like(a))
            Dt = Dt * rho + e.sum(-1)
            N = N * rho + (e * S.unsqueeze(1)).sum(-1)
            M = Mn
        cross = (N / Dt.clamp_min(1e-30)).sum(-1) / H
        lse = m + torch.log(ell.clamp_min(1e-30))
        L_ref = torch.where(vr, lse - cross, torch.zeros(L, dtype=acc, device=dev)).sum() / nrow
        ctx.save_for_backward(qf, kf, wf, Qm, Km, ids, M, Dt, m, ell)
        ctx.hyper = (sm_tea, sm_f, tile, nrow.item(), H, df, acc)
        return L_ref.to(qf.dtype)

    @staticmethod
    def backward(ctx, gL):
        qf, kf, wf, Qm, Km, ids, M, Dt, m, ell = ctx.saved_tensors
        sm_tea, sm_f, tile, nrow, H, df, acc = ctx.hyper
        L = Qm.shape[0]
        g0 = float(gL)
        k_eff = ids.shape[1]
        valid = ids >= 0
        Qmf, Kmf = Qm.to(acc), Km.to(acc)
        qfd, kfd, wfd = qf.to(acc), kf.to(acc), wf.to(acc)
        dqf = torch.zeros_like(qfd)
        dkf = torch.zeros_like(kfd)
        dwf = torch.zeros_like(wfd)
        for c0 in range(0, k_eff, tile):
            c1 = min(c0 + tile, k_eff)
            b = c1 - c0
            jj = ids[:, c0:c1]
            vb = valid[:, c0:c1]
            jc = jj.long().clamp(min=0)
            kfb = kfd[jc]
            dd = torch.einsum("thd,tpd->tph", qfd, kfb)
            relu = torch.relu(dd)
            S = (relu * wfd.unsqueeze(1)).sum(-1) * sm_f
            phat = torch.where(vb, torch.exp(S - m.unsqueeze(-1)) /
                               ell.clamp_min(1e-30).unsqueeze(-1), torch.zeros_like(S))
            if Kmf.dim() == 3:
                a = torch.einsum("thd,tphd->thp", Qmf, Kmf[jc]) * sm_tea
            else:
                a = torch.einsum("thd,tpd->thp", Qmf, Kmf[jc]) * sm_tea
            a = a.masked_fill(~vb.view(L, 1, b), float("-inf"))
            pbar = torch.exp(a - M.unsqueeze(-1)) / Dt.clamp_min(1e-30).unsqueeze(-1)
            pbar = torch.where(vb.view(L, 1, b), pbar, torch.zeros_like(pbar)).mean(1)
            g = torch.where(vb, (phat - pbar) * (g0 / nrow), torch.zeros_like(phat))
            gs = g * sm_f
            gate = (dd > 0).to(acc)
            dwf += (gs.unsqueeze(-1) * relu).sum(1)
            coef = gs.unsqueeze(-1) * wfd.unsqueeze(1) * gate         # [L,b,Hf]
            dqf += torch.einsum("tph,tpd->thd", coef, kfb)
            contrib = torch.einsum("tph,thd->tpd", coef, qfd)        # [L,b,df]
            contrib = torch.where(vb.unsqueeze(-1), contrib, torch.zeros_like(contrib))
            dkf.index_add_(0, jc.reshape(-1), contrib.reshape(-1, df))
        return (dqf.to(qf.dtype), dkf.to(kf.dtype), dwf.to(wf.dtype),
                None, None, None, None, None, None)


def flashkl_sparse_loss(index_q, index_k, index_w, Q_main, K_main, ids, *,
                        sm_scale_teacher=None, sm_scale_index=None, tile=128):
    """Sparse-stage indexer KL over selected keys ``ids`` [L,k] (int32, -1 pad)."""
    st = sm_scale_teacher if sm_scale_teacher is not None else Q_main.shape[-1] ** -0.5
    sf = sm_scale_index if sm_scale_index is not None else index_q.shape[-1] ** -0.5
    return _RefineSparse.apply(index_q, index_k, index_w, Q_main, K_main,
                               ids.to(torch.int64), float(st), float(sf), int(tile))


# --------------------------------------------------------------------------- #
#  memory-bounded sparse KL (query-chunked + gradient-checkpointed)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _teacher_pbar_chunk(Qm_c, Km, ids_c, sm_tea):
    """Head-averaged teacher distribution p̄[c,K] over the selected keys, computed
    under no_grad (teacher is frozen/detached). Kept OUTSIDE the checkpointed student
    region so its [c,K,H,D] key-gather + per-head einsum runs ONCE (fwd only), not
    again on every backward recompute — this is the main sparse-KL speedup."""
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
    under checkpoint — H_indexer (8) heads at df=128, far smaller than the teacher's
    H(16)×192 per-head gather we now keep outside."""
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
    """Memory-bounded sparse-stage indexer KL. Same gradients as
    :func:`flashkl_sparse_loss` but streams query chunks so the gathered
    ``[q_chunk, K, H, D]`` tensors are never all resident. Teacher (Q_main/K_main)
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


# --------------------------------------------------------------------------- #
#  self-test: sparse FlashKL vs dense autograd restricted to selected keys
# --------------------------------------------------------------------------- #
def _dense_sparse_ref(qf, kf, wf, Qm, Km, ids, sm_tea, sm_f):
    L, H, D = Qm.shape
    dt = Qm.dtype
    valid = ids >= 0
    idc = ids.long().clamp(min=0)
    # student
    kg = kf[idc]
    dd = torch.einsum("thd,tpd->tph", qf, kg)
    S = (torch.relu(dd) * wf.unsqueeze(1)).sum(-1) * sm_f            # [L,k]
    neg = torch.finfo(dt).min
    logq = torch.log_softmax(torch.where(valid, S, torch.full_like(S, neg)), -1)
    # teacher
    km = Km[idc]
    a = torch.einsum("thd,tpd->thp", Qm, km) * sm_tea               # [L,H,k]
    a = a.masked_fill(~valid.view(L, 1, -1), float("-inf"))
    p = torch.nan_to_num(torch.softmax(a, -1), nan=0.0).mean(1)     # [L,k]
    p = torch.where(valid, p, torch.zeros_like(p))
    vr = valid.any(-1)
    kl = (p * (torch.log(p.clamp_min(1e-30)) - logq)).sum(-1)
    return torch.where(vr, kl, torch.zeros_like(kl)).sum() / vr.to(dt).sum().clamp_min(1.0)


def _selftest():
    torch.manual_seed(0)
    f64 = torch.float64
    L, H, D, Hf, df, K = 20, 8, 16, 4, 12, 5

    def rel(a, b):
        return (a - b).norm().item() / (b.norm().item() + 1e-30)

    qf = torch.randn(L, Hf, df, dtype=f64, requires_grad=True)
    kf = torch.randn(L, df, dtype=f64, requires_grad=True)
    wf = torch.rand(L, Hf, dtype=f64, requires_grad=True)
    Qm = torch.randn(L, H, D, dtype=f64)
    Km = torch.randn(L, D, dtype=f64)
    cu = torch.tensor([0, 11, 20])

    ids = indexer_select_topk(qf.detach(), kf.detach(), wf.detach(), cu, K)
    st, sf = D ** -0.5, df ** -0.5

    Lref = _dense_sparse_ref(qf, kf, wf, Qm, Km, ids.long(), st, sf)
    Lref.backward()
    ref = {k: v.grad.clone() for k, v in [("qf", qf), ("kf", kf), ("wf", wf)]}
    qf.grad = kf.grad = wf.grad = None

    Lfl = flashkl_sparse_loss(qf, kf, wf, Qm, Km, ids, tile=2)
    Lfl.backward()
    fu = {k: v.grad.clone() for k, v in [("qf", qf), ("kf", kf), ("wf", wf)]}
    errs = {k: rel(fu[k], ref[k]) for k in ref}
    # FlashKL drops the teacher-entropy CONSTANT, so its loss value differs from
    # the full-KL reference by that constant — but the GRADIENTS are identical.
    # Correctness is judged on the gradients only.
    ok = all(v < 1e-9 for v in errs.values())
    print("=" * 70)
    print("sparse FlashKL vs dense autograd (selected keys, float64)")
    print("=" * 70)
    print(f"  loss flashkl={Lfl.item():.6f} (trainable, entropy dropped)  "
          f"dense_fullKL={Lref.item():.6f}")
    print("  grad rel-err  " + "  ".join(f"{k}={v:.2e}" for k, v in errs.items())
          + f"  -> OK(<1e-9): {ok}")

    # --- chunked sparse KL vs FlashKL (grads must match; loss must match too,
    #     since both drop the same teacher-entropy constant) ---
    qf.grad = kf.grad = wf.grad = None
    Lch = sparse_kl_chunked(qf, kf, wf, Qm, Km, ids, q_chunk=3)
    Lch.backward()
    ch = {k: v.grad.clone() for k, v in [("qf", qf), ("kf", kf), ("wf", wf)]}
    errs_c = {k: rel(ch[k], ref[k]) for k in ref}
    dloss = abs(Lch.item() - Lfl.item())
    ok_c = dloss < 1e-9 and all(v < 1e-9 for v in errs_c.values())
    ok = ok and ok_c
    print("-" * 70)
    print("sparse_kl_chunked vs FlashKL (query-chunked + checkpoint, float64)")
    print(f"  loss chunked={Lch.item():.6f}  |Δloss|={dloss:.2e}")
    print("  grad rel-err  " + "  ".join(f"{k}={v:.2e}" for k, v in errs_c.items())
          + f"  -> OK(<1e-9): {ok_c}")
    print("=" * 70)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
