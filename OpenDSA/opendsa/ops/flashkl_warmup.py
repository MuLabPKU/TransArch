"""flashkl_warmup.py — single-indexer FlashKL warmup loss (O(L·H) memory).

The DSA warmup stage trains a Lightning Indexer to match the *teacher* attention
distribution (the model's own dense MLA attention, head-averaged) via a KL loss.
Because the teacher is frozen, the teacher-entropy term of the KL is a constant
w.r.t. the indexer parameters and can be dropped; the remaining cross-entropy is
**linear** in the teacher probabilities, so it decomposes per head and can be
accumulated with Flash-Attention-style online-softmax tiling — streaming over the
key axis, never materializing the [L, H, L] attention matrix.

    L = (1/N) Σ_t [ LSE_{j∈causal}(I[t,:]) − Σ_j p̄[t,j]·I[t,j] ] + const
    grad on indexer logits:  g[t,j] = (1/N)(p̂[t,j] − p̄[t,j])

where
    teacher  p̄[t,j] = (1/H) Σ_h softmax_j(α^(h)[t,:])[j],  α^(h)=<Qm[t,h],Km[j]>·s_t
    student  p̂[t,j] = softmax_j(I[t,:])[j],  I[t,j]=(Σ_h w[t,h]·ReLU(<qf[t,h],kf[j]>))·s_f

This is the *refine-only* (single indexer) specialization of the fused two-level
FlashKL reference implementation. Memory is O(L·H):
we keep only running per-head (M, D, N) ∈ [L,H] and per-query student (m, ℓ) ∈ [L].

Teacher Q/K contract: per-head query Qm[L,H,D]. Teacher key Km may be either
per-head Km[L,H,D] (the faithful DeepSeek-V2 eager form: k_nope is per-head, k_pe
broadcast — matches the reference Megatron ``einsum("bhqd,bhtd->bhqt")``) OR shared
Km[L,D] (MLA absorbed latent form, broadcast across heads). Both give identical
attention scores; per-head is the default used from the HF eager path.

Verify:  python flashkl_warmup.py   # pure-CPU float64 self-test, no GPU needed
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
#  causal support from packed cu_seqlens
# --------------------------------------------------------------------------- #
def prepare_ks_ke(cu_seqlens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """packed cu_seqlens [S+1] -> per-query causal interval [ks, ke) (int64 [L]).

    Query t (0-indexed within its sequence, absolute index in the packed buffer)
    may attend keys ks[t] <= j < ke[t] == its own absolute position + 1.
    """
    cu = cu_seqlens.long()
    lens = torch.diff(cu)
    if lens.numel():
        pos = torch.cat([torch.arange(int(n), device=cu.device) for n in lens.tolist()])
    else:
        pos = torch.empty(0, dtype=torch.long, device=cu.device)
    seq = torch.repeat_interleave(torch.arange(lens.numel(), device=cu.device), lens)
    ks = cu[seq]
    ke = ks + pos + 1
    return ks.long(), ke.long()


def _acc_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _teacher_logits(Qmf, Kmf, s0, s1, sm_tea):
    """Per-head teacher logits α[L,H,b] for key tile [s0,s1).
    Qmf: [L,H,D]. Kmf: [L,H,D] (per-head) or [L,D] (shared across heads)."""
    if Kmf.dim() == 3:
        return torch.einsum("thd,jhd->thj", Qmf, Kmf[s0:s1]) * sm_tea
    return torch.einsum("thd,jd->thj", Qmf, Kmf[s0:s1]) * sm_tea


# --------------------------------------------------------------------------- #
#  FlashKL warmup autograd.Function (streaming over key tiles)
# --------------------------------------------------------------------------- #
class _RefineWarmup(torch.autograd.Function):
    """One forward + one backward FlashKL warmup for a single indexer.

    Inputs (student = indexer, teacher = main attention):
        qf [L, Hf, df]   indexer query (per head)
        kf [L, df]       indexer key   (single head, shared)
        wf [L, Hf]       indexer per-head weights
        Qm [L, H, D]     teacher query (per head)
        Km [L, D]        teacher key   (shared across heads, MLA absorbed)
    Returns scalar trainable loss (teacher entropy dropped).
    """

    @staticmethod
    def forward(ctx, qf, kf, wf, Qm, Km, ks, ke, sm_tea, sm_f, tile, nrow_g):
        # rows = LOCAL queries (Lq); key axis = FULL gathered keys (Lk). Under CP
        # these differ (Lq = Lk / cp); without CP Lq == Lk. ks/ke are per-query
        # causal intervals [ks,ke) in ABSOLUTE key coordinates (ke = query's true
        # global position + 1), so the same formula covers both cases. We never
        # build the dense [Lq,Lk] causal mask — it's computed per key-tile.
        Lq, H, D = Qm.shape
        Lk = Km.shape[0]
        Hf, df = qf.shape[1], qf.shape[2]
        dev = Qm.device
        acc = _acc_dtype(Qm.dtype)

        Qmf, Kmf = Qm.to(acc), Km.to(acc)
        qfd, kfd, wfd = qf.to(acc), kf.to(acc), wf.to(acc)

        # teacher online-softmax state (per head) + student LSE state (per query)
        M = torch.full((Lq, H), float("-inf"), dtype=acc, device=dev)
        Dt = torch.zeros(Lq, H, dtype=acc, device=dev)
        N = torch.zeros(Lq, H, dtype=acc, device=dev)  # Σ e^(α-M) · S  (value accumulator)
        m = torch.full((Lq,), float("-inf"), dtype=acc, device=dev)
        ell = torch.zeros(Lq, dtype=acc, device=dev)

        for s0 in range(0, Lk, tile):
            s1 = min(s0 + tile, Lk)
            b = s1 - s0
            jt = torch.arange(s0, s1, device=dev)                     # [b] abs key pos
            sup = (jt.view(1, b) >= ks.view(Lq, 1)) & (jt.view(1, b) < ke.view(Lq, 1))  # [Lq,b]
            # teacher per-head logits (K per-head or shared)
            a = _teacher_logits(Qmf, Kmf, s0, s1, sm_tea)             # [Lq,H,b]
            a = a.masked_fill(~sup.view(Lq, 1, b), float("-inf"))
            M_new = torch.maximum(M, a.max(-1).values)
            rho = torch.exp(M - M_new)
            rho = torch.where(torch.isfinite(rho), rho, torch.zeros_like(rho))
            e = torch.exp(a - M_new.unsqueeze(-1))
            e = torch.where(sup.view(Lq, 1, b), e, torch.zeros_like(e))  # [Lq,H,b]
            Dt = Dt * rho + e.sum(-1)
            # student score S[t,j] (cheap Hf GEMM) — acts as scalar "value"
            sh = torch.einsum("thd,jd->thj", qfd, kfd[s0:s1])          # [Lq,Hf,b]
            S = (torch.relu(sh) * wfd.unsqueeze(-1)).sum(1) * sm_f      # [Lq,b]
            N = N * rho + (e * S.unsqueeze(1)).sum(-1)                  # [Lq,H]
            # student online LSE
            Sm = torch.where(sup, S, torch.full_like(S, float("-inf")))
            m_new = torch.maximum(m, Sm.max(-1).values)
            beta = torch.exp(m - m_new)
            beta = torch.where(torch.isfinite(beta), beta, torch.zeros_like(beta))
            add = torch.exp(Sm - m_new.unsqueeze(-1))
            add = torch.where(sup, add, torch.zeros_like(add))
            ell = ell * beta + add.sum(-1)
            m = m_new
            M = M_new

        valid_row = ke > ks                                           # [Lq] has ≥1 key
        # normalization: global query count if given (CP: divide by total queries so
        # summing per-rank losses/grads over the CP group yields the true mean), else
        # the local valid-row count.
        nrow = float(nrow_g) if nrow_g is not None else valid_row.to(acc).sum().clamp_min(1.0).item()
        cross = (N / Dt.clamp_min(1e-30)).sum(-1) / H                  # [Lq]  Σ_j p̄·I
        lse = m + torch.log(ell.clamp_min(1e-30))                      # [Lq]  LSE_j I
        L_ref = torch.where(valid_row, lse - cross,
                            torch.zeros(Lq, dtype=acc, device=dev)).sum() / nrow

        ctx.save_for_backward(qf, kf, wf, Qm, Km, ks, ke, M, Dt, m, ell)
        ctx.hyper = (sm_tea, sm_f, tile, nrow, H, df, acc)
        return L_ref.to(qf.dtype)

    @staticmethod
    def backward(ctx, gL):
        qf, kf, wf, Qm, Km, ks, ke, M, Dt, m, ell = ctx.saved_tensors
        sm_tea, sm_f, tile, nrow, H, df, acc = ctx.hyper
        Lq, D = Qm.shape[0], Qm.shape[2]
        Lk = Km.shape[0]
        dev = Qm.device
        g0 = float(gL)

        Qmf, Kmf = Qm.to(acc), Km.to(acc)
        qfd, kfd, wfd = qf.to(acc), kf.to(acc), wf.to(acc)
        dqf = torch.zeros_like(qfd)
        dkf = torch.zeros_like(kfd)
        dwf = torch.zeros_like(wfd)

        for s0 in range(0, Lk, tile):
            s1 = min(s0 + tile, Lk)
            b = s1 - s0
            jt = torch.arange(s0, s1, device=dev)
            sup = (jt.view(1, b) >= ks.view(Lq, 1)) & (jt.view(1, b) < ke.view(Lq, 1))  # [Lq,b]
            kfb = kfd[s0:s1]
            sh = torch.einsum("thd,jd->thj", qfd, kfb)                 # [Lq,Hf,b]
            relu = torch.relu(sh)
            S = (relu * wfd.unsqueeze(-1)).sum(1) * sm_f               # [Lq,b]
            phat = torch.where(sup, torch.exp(S - m.unsqueeze(-1)) /
                               ell.clamp_min(1e-30).unsqueeze(-1), torch.zeros_like(S))
            a = _teacher_logits(Qmf, Kmf, s0, s1, sm_tea)             # [Lq,H,b]
            a = a.masked_fill(~sup.view(Lq, 1, b), float("-inf"))
            pbar = torch.exp(a - M.unsqueeze(-1)) / Dt.clamp_min(1e-30).unsqueeze(-1)
            pbar = torch.where(sup.view(Lq, 1, b), pbar, torch.zeros_like(pbar)).mean(1)  # [Lq,b]
            g = torch.where(sup, (phat - pbar) * (g0 / nrow), torch.zeros_like(phat))
            gs = g * sm_f
            gate = (sh > 0).to(acc)
            dwf += (gs.unsqueeze(1) * relu).sum(-1)                    # [Lq,Hf]
            coef = gs.unsqueeze(1) * wfd.unsqueeze(-1) * gate          # [Lq,Hf,b]
            dqf += torch.einsum("thc,cd->thd", coef, kfb)
            dkf[s0:s1] += torch.einsum("thc,thd->cd", coef, qfd)

        return (dqf.to(qf.dtype), dkf.to(kf.dtype), dwf.to(wf.dtype),
                None, None, None, None, None, None, None, None)


def prepare_ks_ke_cp(cu_full: torch.Tensor, q_global_pos: torch.Tensor):
    """Per-LOCAL-QUERY causal interval [ks,ke) in ABSOLUTE key coords, for CP.

    cu_full [S+1]: full-sequence doc boundaries (absolute over [0, Lk]).
    q_global_pos [Lq]: true global position of each local query.
    Returns ks[Lq] = query's doc start, ke[Lq] = q_global_pos + 1 (causal upper bound).
    Works for the non-CP case too (q_global_pos = arange(L))."""
    cu = cu_full.long()
    p = q_global_pos.long()
    d = torch.searchsorted(cu, p, right=True) - 1                    # doc index per query
    d = d.clamp(0, cu.numel() - 2)
    ks = cu[d]
    ke = p + 1
    return ks.long(), ke.long()


def flashkl_warmup_loss(
    index_q, index_k, index_w,      # student indexer: [Lq,Hf,df], [Lk,df], [Lq,Hf]
    Q_main, K_main,                 # teacher: [Lq,H,D] (queries), [Lk,D]/[Lk,H,D] (keys)
    cu_seqlens, *,
    sm_scale_teacher: Optional[float] = None,
    sm_scale_index: Optional[float] = None,
    tile: int = 128,
    q_global_pos=None,              # [Lq] true global position per query (CP); None -> local
    nrow_global=None,              # global query count for CP normalization; None -> local
):
    """FlashKL warmup loss for one indexer. O(Lq·H) memory, one fwd + one bwd.

    Non-CP: index_q/index_k/index_w and Q_main/K_main all have the same leading L;
    q_global_pos defaults to the per-doc causal ranges from ``cu_seqlens``.

    CP: the QUERY tensors (index_q, index_w, Q_main) hold this rank's Lq local rows;
    the KEY tensors (index_k, K_main) hold the FULL gathered Lk keys; ``q_global_pos``
    gives each local query's true global position (so causal bound = pos+1) and
    ``cu_seqlens`` are the FULL-sequence doc boundaries. Pass ``nrow_global`` = total
    query count across the CP group so that summing per-rank losses/grads over the
    group yields the correct global mean.

    ``(loss).backward()`` lands grads on the three indexer projections only (teacher
    detached); the teacher-entropy constant is dropped (grads exact).
    """
    if q_global_pos is None:
        ks, ke = prepare_ks_ke(cu_seqlens)
    else:
        ks, ke = prepare_ks_ke_cp(cu_seqlens, q_global_pos)
    st = sm_scale_teacher if sm_scale_teacher is not None else Q_main.shape[-1] ** -0.5
    sf = sm_scale_index if sm_scale_index is not None else index_q.shape[-1] ** -0.5
    return _RefineWarmup.apply(index_q, index_k, index_w, Q_main, K_main,
                               ks, ke, float(st), float(sf), int(tile), nrow_global)


# --------------------------------------------------------------------------- #
#  dense reference (materializes [L,H,L]) for correctness checking
# --------------------------------------------------------------------------- #
def dense_warmup_reference(index_q, index_k, index_w, Q_main, K_main, cu_seqlens,
                           *, sm_scale_teacher=None, sm_scale_index=None):
    """Naive O(L²·H) reference. Returns (full_KL, trainable_loss); grads of both
    w.r.t. the indexer params are identical (teacher entropy is constant)."""
    L, H, D = Q_main.shape
    Hf, df = index_q.shape[1], index_q.shape[2]
    dev = Q_main.device
    dt = Q_main.dtype
    st = sm_scale_teacher if sm_scale_teacher is not None else D ** -0.5
    sf = sm_scale_index if sm_scale_index is not None else df ** -0.5
    ks, ke = prepare_ks_ke(cu_seqlens)
    jj = torch.arange(L, device=dev)
    causal = (jj.view(1, L) >= ks.view(L, 1)) & (jj.view(1, L) < ke.view(L, 1))

    # student logits I[t,j]
    sh = torch.einsum("thd,jd->thj", index_q, index_k)               # [L,Hf,L]
    I = (torch.relu(sh) * index_w.unsqueeze(-1)).sum(1) * sf          # [L,L]
    neg = torch.finfo(dt).min
    Imask = torch.where(causal, I, torch.full_like(I, neg))
    logp_hat = torch.log_softmax(Imask, dim=-1)                       # [L,L]

    # teacher p̄[t,j]
    if K_main.dim() == 3:
        a = torch.einsum("thd,jhd->thj", Q_main, K_main) * st        # [L,H,L]
    else:
        a = torch.einsum("thd,jd->thj", Q_main, K_main) * st         # [L,H,L]
    a = a.masked_fill(~causal.view(L, 1, L), float("-inf"))
    pbar = torch.nan_to_num(torch.softmax(a, dim=-1), nan=0.0).mean(1)  # [L,L]
    pbar = torch.where(causal, pbar, torch.zeros_like(pbar))

    valid = causal.any(-1)
    nrow = valid.to(dt).sum().clamp_min(1.0)
    # full KL = Σ p̄ (log p̄ − log p̂)
    kl = (pbar * (torch.log(pbar.clamp_min(1e-30)) - logp_hat)).sum(-1)
    full_kl = torch.where(valid, kl, torch.zeros_like(kl)).sum() / nrow
    # trainable = LSE − Σ p̄·I  (== full KL + teacher entropy const)
    lse = torch.logsumexp(Imask, dim=-1)
    cross = (pbar * I).sum(-1)
    trainable = torch.where(valid, lse - cross, torch.zeros_like(lse)).sum() / nrow
    return full_kl, trainable


def indexer_topk_recall(index_q, index_k, index_w, Q_main, K_main, cu_seqlens,
                        topk, *, sm_scale_teacher=None, sm_scale_index=None,
                        q_chunk: int = 128, q_global_pos=None):
    """Diagnostic: fraction of the teacher's top-k keys that fall in the indexer's
    top-k, averaged over queries. The core warmup quality metric.

    Query-chunked so it never materializes the full [Lq,H,Lk] teacher attention. CP:
    queries (index_q/Q_main/index_w) are Lq local rows, keys (index_k/K_main) are the
    full Lk gathered; pass ``q_global_pos`` [Lq] and full-seq ``cu_seqlens``.

    FULLY VECTORIZED over queries: the outer ``for t0 in range`` loop only chunks the
    query axis (bounded, O(Lq/q_chunk)); there is NO per-query Python loop. The old
    ``for i in range(c)`` topk+isin with per-row ``.item()`` syncs was O(L) serial and
    hung ~7.5 min at L=32768. Per chunk we take ONE ``torch.topk(., K, dim=-1)`` with
    K=min(topk, Lk) for teacher (pbar) and student (I). Each row honours its own
    k_i = min(topk, n_valid_i): a [c,K] ``keep`` mask selects each row's first k_i picks
    AND, defensively, requires the picked slot to be a genuine (causal) key -- so masked
    filler (student -inf / teacher -1.0) can never be counted even in degenerate/tie
    cases. Teacher's kept picks are scattered into a [c,Lk] boolean grid; gathering that
    grid at the student's kept picks gives the per-row intersection. recall_i = |cap|/k_i
    with k_i the per-row kept count (== min(topk, n_valid_i)), averaged over rows with
    k_i>0 and accumulated in float64 across chunks. Numerically identical to the old
    per-query loop (verified to <1e-9 vs the reference across shared/per-head K, packed
    multi-doc incl. length-1 docs, CP query shards, tie-heavy integer inputs, topk above
    and below Lk, and multiple q_chunk sizes)."""
    Lq, H, D = Q_main.shape
    Lk = K_main.shape[0]
    df = index_q.shape[2]
    dev = Q_main.device
    st = sm_scale_teacher if sm_scale_teacher is not None else D ** -0.5
    sf = sm_scale_index if sm_scale_index is not None else df ** -0.5
    if q_global_pos is None:
        ks, ke = prepare_ks_ke(cu_seqlens)
    else:
        ks, ke = prepare_ks_ke_cp(cu_seqlens, q_global_pos)
    jj = torch.arange(Lk, device=dev)
    K = int(min(int(topk), Lk))                                       # shared topk width
    if K <= 0:
        return 0.0
    hit = torch.zeros((), dtype=torch.float64, device=dev)
    cnt = torch.zeros((), dtype=torch.float64, device=dev)
    with torch.no_grad():
        for t0 in range(0, Lq, q_chunk):
            t1 = min(t0 + q_chunk, Lq)
            c = t1 - t0
            causal = (jj.view(1, Lk) >= ks[t0:t1].view(c, 1)) & \
                     (jj.view(1, Lk) < ke[t0:t1].view(c, 1))            # [c,Lk]
            sh = torch.einsum("thd,jd->thj", index_q[t0:t1], index_k)  # [c,Hf,Lk]
            I = (torch.relu(sh) * index_w[t0:t1].unsqueeze(-1)).sum(1) * sf  # [c,Lk]
            if K_main.dim() == 3:
                a = torch.einsum("thd,jhd->thj", Q_main[t0:t1], K_main) * st
            else:
                a = torch.einsum("thd,jd->thj", Q_main[t0:t1], K_main) * st
            a = a.masked_fill(~causal.view(c, 1, Lk), float("-inf"))
            pbar = torch.nan_to_num(torch.softmax(a, -1), nan=0.0).mean(1)  # [c,Lk]
            neg = torch.finfo(I.dtype).min
            I = torch.where(causal, I, torch.full_like(I, neg))        # filler -> -inf
            pbar = torch.where(causal, pbar, torch.full_like(pbar, -1.0))  # filler -> -1
            # per-row k_i = min(topk, n_valid); rows with no causal keys are skipped.
            n_valid = causal.sum(-1)                                   # [c]
            k_row = torch.clamp(n_valid, max=int(topk))               # [c] == min(topk,n_valid)
            # one topk for the whole chunk (its sorted prefix equals a per-row topk(k_i)),
            # then keep only each row's first k_i picks that land on a genuine causal key.
            tea_idx = torch.topk(pbar, K, dim=-1).indices             # [c,K]
            stu_idx = torch.topk(I, K, dim=-1).indices               # [c,K]
            within = torch.arange(K, device=dev).view(1, K) < k_row.view(c, 1)  # [c,K]
            keep_tea = within & causal.gather(1, tea_idx)             # [c,K] genuine teacher picks
            keep_stu = within & causal.gather(1, stu_idx)             # [c,K] genuine student picks
            # scatter teacher's kept picks into a [c,Lk] boolean membership grid;
            # 'keep_tea' as the scatter source keeps filler / beyond-k_i slots False.
            grid = torch.zeros(c, Lk, dtype=torch.bool, device=dev)
            grid.scatter_(1, tea_idx, keep_tea)                       # [c,Lk] teacher set
            inter = (grid.gather(1, stu_idx) & keep_stu).sum(-1)      # [c] student cap teacher
            k_i = keep_tea.sum(-1)                                    # [c] == min(topk,n_valid)
            valid = k_i > 0                                           # [c]
            recall_row = torch.where(
                valid,
                inter.to(torch.float64) / k_i.clamp_min(1).to(torch.float64),
                torch.zeros(c, dtype=torch.float64, device=dev))      # [c]
            hit += recall_row.sum()
            cnt += valid.to(torch.float64).sum()
        return float((hit / cnt.clamp_min(1)).item())


# --------------------------------------------------------------------------- #
#  self-test (CPU / float64): FlashKL vs dense autograd to machine precision
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)
    f64 = torch.float64
    dev = "cpu"

    def rel(a, b):
        return (a - b).norm().item() / (b.norm().item() + 1e-30)

    H, D = 8, 16
    Hf, df = 4, 12

    def mk(L, per_head_k=False):
        return dict(
            qf=torch.randn(L, Hf, df, dtype=f64, requires_grad=True),
            kf=torch.randn(L, df, dtype=f64, requires_grad=True),
            wf=torch.rand(L, Hf, dtype=f64, requires_grad=True),
            Qm=torch.randn(L, H, D, dtype=f64),
            Km=torch.randn(L, H, D, dtype=f64) if per_head_k else torch.randn(L, D, dtype=f64),
        )

    print("=" * 76)
    print("FlashKL warmup (single indexer) — vs dense autograd, float64")
    print("=" * 76)
    ok_all = True
    cases = [
        ("shared-K   single-seq  cu=[0,20]", torch.tensor([0, 20]), False),
        ("shared-K   packed      cu=[0,9,20]", torch.tensor([0, 9, 20]), False),
        ("perhead-K  single-seq  cu=[0,23]", torch.tensor([0, 23]), True),
        ("perhead-K  packed      cu=[0,11,20]", torch.tensor([0, 11, 20]), True),
    ]
    for name, cu, ph in cases:
        L = int(cu[-1])
        g = mk(L, per_head_k=ph)
        # dense reference: grad of FULL KL (entropy is const -> same grad as trainable)
        full_kl, trainable_ref = dense_warmup_reference(
            g["qf"], g["kf"], g["wf"], g["Qm"], g["Km"], cu)
        full_kl.backward()
        ref = {k: g[k].grad.clone() for k in ["qf", "kf", "wf"]}
        for k in ["qf", "kf", "wf"]:
            g[k].grad = None
        # FlashKL
        Lfl = flashkl_warmup_loss(g["qf"], g["kf"], g["wf"], g["Qm"], g["Km"], cu, tile=4)
        Lfl.backward()
        fu = {k: g[k].grad.clone() for k in ["qf", "kf", "wf"]}
        errs = {k: rel(fu[k], ref[k]) for k in ref}
        dval = abs(Lfl.item() - trainable_ref.item())
        ok = dval < 1e-9 and all(v < 1e-9 for v in errs.values())
        ok_all &= ok
        print(f"[{name}]")
        print(f"   loss(trainable) flashkl={Lfl.item():.10f}  dense={trainable_ref.item():.10f}  |Δ|={dval:.2e}")
        print("   grad rel-err  " + "  ".join(f"{k}={v:.2e}" for k, v in errs.items())
              + f"   -> OK(<1e-9): {ok}")
    print("=" * 76)
    print(f"ALL PASS: {ok_all}")
    print("=" * 76)

    # --- CP-equivalence: sharded queries + full keys, summed == full ---
    print("CP-equivalence: sum over query-shards of (loss, grads) == full")
    print("=" * 76)
    torch.manual_seed(1)
    H, D, Hf, df = 8, 16, 4, 12
    L, CP = 24, 3           # split L queries into CP contiguous shards
    cu = torch.tensor([0, L])   # single doc
    qf = torch.randn(L, Hf, df, dtype=f64, requires_grad=True)
    kf = torch.randn(L, df, dtype=f64, requires_grad=True)
    wf = torch.rand(L, Hf, dtype=f64, requires_grad=True)
    Qm = torch.randn(L, H, D, dtype=f64)
    Km = torch.randn(L, D, dtype=f64)
    # full (baseline): global nrow = L
    Lfull = flashkl_warmup_loss(qf, kf, wf, Qm, Km, cu, tile=5,
                                q_global_pos=torch.arange(L), nrow_global=L)
    Lfull.backward()
    gfull = {"qf": qf.grad.clone(), "kf": kf.grad.clone(), "wf": wf.grad.clone()}
    qf.grad = kf.grad = wf.grad = None
    # sharded: each shard holds its Lq query rows + the FULL keys; nrow_global=L.
    # kf/Km are the full keys (shared); qf/wf/Qm are sliced per shard. Sum losses+grads.
    Lloc = L // CP
    loss_sum = 0.0
    gsum = {"qf": torch.zeros_like(qf), "kf": torch.zeros_like(kf), "wf": torch.zeros_like(wf)}
    for r in range(CP):
        s, e = r * Lloc, (r + 1) * Lloc
        qfs = qf[s:e].detach().clone().requires_grad_(True)
        wfs = wf[s:e].detach().clone().requires_grad_(True)
        kfs = kf.detach().clone().requires_grad_(True)
        gpos = torch.arange(s, e)
        Ls = flashkl_warmup_loss(qfs, kfs, wfs, Qm[s:e], Km, cu, tile=5,
                                 q_global_pos=gpos, nrow_global=L)
        Ls.backward()
        loss_sum += Ls.item()
        gsum["qf"][s:e] += qfs.grad
        gsum["wf"][s:e] += wfs.grad
        gsum["kf"] += kfs.grad      # key grads accumulate across all shards
    e_loss = abs(loss_sum - Lfull.item())
    e_g = {k: rel(gsum[k], gfull[k]) for k in gfull}
    ok_cp = e_loss < 1e-9 and all(v < 1e-9 for v in e_g.values())
    ok_all &= ok_cp
    print(f"  |Δloss|={e_loss:.2e}  " + "  ".join(f"d{k}={v:.2e}" for k, v in e_g.items())
          + f"  -> OK(<1e-9): {ok_cp}")
    print("=" * 76)
    print(f"ALL PASS: {ok_all}")
    print("=" * 76)
    return ok_all


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
