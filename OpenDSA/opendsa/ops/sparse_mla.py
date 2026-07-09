"""sparse_mla.py — sparse MLA attention for the DSA sparse-training stage.

At sparse-training time each query attends only to the top-k keys chosen by the
Lightning Indexer. This module provides:

  * ``sparse_mla_ref``   — pure-torch, autograd-differentiable reference. Gathers
    the selected keys per query and runs softmax attention. Framework-agnostic,
    works on CPU/GPU, correct by construction (backward via autograd). This is
    the default path and the correctness baseline.

  * ``sparse_mla_kernel`` — thin wrapper around optional TileLang sparse-MLA
    kernels (sm_90). Set ``OPENDSA_SPARSE_MLA_KERNEL_PATH`` if they live outside
    the import path. Falls back to the reference automatically if TileLang is
    unavailable or shapes are unsupported. Currently exposes forward; use the
    reference for backward until the kernel bwd is wired (see note below).

MLA absorbed contract (matches the teacher used in warmup):
    q  : [L, H, Dqk]     per-head query in latent space (Dqk = kv_lora + rope, e.g. 576)
    kv : [L, Dqk]        shared K==softmax key; the first d_v dims (=kv_lora, 512)
                         also serve as V (MLA fuses K and V in the latent space)
    indices : [L, topk]  int32, per-query selected key positions (-1 padded)
    output  : [L, H, d_v]

Verify:  python sparse_mla.py    # CPU float64: ref vs dense-masked attention
"""
from __future__ import annotations

from typing import Optional

import torch


def _causal_topk_mask(indices: torch.Tensor, q_pos: torch.Tensor) -> torch.Tensor:
    """[L,topk] bool: valid iff index in [0, q_pos] (causal) and != -1."""
    L, K = indices.shape
    valid = (indices >= 0) & (indices <= q_pos.view(L, 1))
    return valid


def sparse_mla_ref(
    q: torch.Tensor,          # [L, H, Dqk]
    kv: torch.Tensor,         # [L, Dqk]
    indices: torch.Tensor,    # [L, topk] int (-1 padded)
    *,
    d_v: int = 512,
    sm_scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure-torch differentiable sparse MLA attention.

    For each query t, softmax over the selected causal keys of <q_t, k_j>·scale,
    then weighted sum of v_j (v_j = kv[j, :d_v]). Returns [L, H, d_v].
    """
    L, H, Dqk = q.shape
    K = indices.shape[-1]
    dev = q.device
    if sm_scale is None:
        sm_scale = Dqk ** -0.5

    # per-query absolute position (causal upper bound)
    if cu_seqlens is not None:
        try:
            from .flashkl_warmup import prepare_ks_ke
        except ImportError:
            from flashkl_warmup import prepare_ks_ke
        ks, ke = prepare_ks_ke(cu_seqlens)
        q_pos = ke - 1
    else:
        q_pos = torch.arange(L, device=dev)

    idx_clamped = indices.long().clamp(min=0)                       # [L,K]
    valid = _causal_topk_mask(indices, q_pos)                       # [L,K]

    k_sel = kv[idx_clamped]                                         # [L,K,Dqk]
    v_sel = k_sel[..., :d_v]                                        # [L,K,d_v]

    scores = torch.einsum("thd,tkd->thk", q, k_sel) * sm_scale      # [L,H,K]
    scores = scores.masked_fill(~valid.view(L, 1, K), float("-inf"))
    # rows with no valid key (shouldn't happen: self is always selectable) -> 0
    all_masked = ~valid.any(-1)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.einsum("thk,tkd->thd", probs, v_sel)               # [L,H,d_v]
    if all_masked.any():
        out = out.masked_fill(all_masked.view(L, 1, 1), 0.0)
    return out


def sparse_attend_ref(
    q: torch.Tensor,          # [L, H, Dqk]  per-head query
    k: torch.Tensor,          # [L, H, Dqk]  per-head key
    v: torch.Tensor,          # [L, H, Dv]   per-head value
    indices: torch.Tensor,    # [L, topk] int (-1 padded), per-query selected keys
    *,
    sm_scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure-torch differentiable sparse attention with SEPARATE per-head Q/K/V.

    This is the DeepSeek-V2 eager layout (K has q_head_dim=192, V has v_head_dim
    =128, per head). For each query t: softmax over selected causal keys of
    <q_t, k_j>·scale, then weighted sum of v_j. Returns [L, H, Dv].
    """
    L, H, Dqk = q.shape
    Dv = v.shape[-1]
    K = indices.shape[-1]
    dev = q.device
    if sm_scale is None:
        sm_scale = Dqk ** -0.5
    if cu_seqlens is not None:
        try:
            from .flashkl_warmup import prepare_ks_ke
        except ImportError:
            from flashkl_warmup import prepare_ks_ke
        _, ke = prepare_ks_ke(cu_seqlens)
        q_pos = ke - 1
    else:
        q_pos = torch.arange(L, device=dev)

    idx_clamped = indices.long().clamp(min=0)                       # [L,K]
    valid = (indices >= 0) & (indices <= q_pos.view(L, 1))          # [L,K]

    k_sel = k[idx_clamped]                                          # [L,K,H,Dqk]
    v_sel = v[idx_clamped]                                          # [L,K,H,Dv]
    scores = torch.einsum("thd,tkhd->thk", q, k_sel) * sm_scale     # [L,H,K]
    scores = scores.masked_fill(~valid.view(L, 1, K), float("-inf"))
    probs = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    out = torch.einsum("thk,tkhd->thd", probs, v_sel)              # [L,H,Dv]
    all_masked = ~valid.any(-1)
    if all_masked.any():
        out = out.masked_fill(all_masked.view(L, 1, 1), 0.0)
    return out


def _attend_chunk(q_c, k, v, idx_c, valid_c, sm_scale):
    """One query-chunk of sparse attention. q_c [c,H,Dqk]; k/v full [L,H,*];
    idx_c [c,K] clamped>=0; valid_c [c,K] bool. Returns [c,H,Dv].

    The big intermediate here is k_sel/v_sel [c,K,H,*]; when this fn is wrapped in
    ``torch.utils.checkpoint`` it is recomputed in backward instead of stored, so
    peak memory is O(c·K·H·D) with c=q_chunk << L rather than O(L·K·H·D)."""
    c, K = idx_c.shape
    H = q_c.shape[1]
    k_sel = k[idx_c]                                                # [c,K,H,Dqk]
    v_sel = v[idx_c]                                                # [c,K,H,Dv]
    scores = torch.einsum("thd,tkhd->thk", q_c, k_sel) * sm_scale  # [c,H,K]
    scores = scores.masked_fill(~valid_c.view(c, 1, K), float("-inf"))
    probs = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    out = torch.einsum("thk,tkhd->thd", probs, v_sel)              # [c,H,Dv]
    all_masked = ~valid_c.any(-1)
    if all_masked.any():
        out = out.masked_fill(all_masked.view(c, 1, 1), 0.0)
    return out


def sparse_attend_chunked(
    q: torch.Tensor,          # [L, H, Dqk]  per-head query
    k: torch.Tensor,          # [L, H, Dqk]  per-head key
    v: torch.Tensor,          # [L, H, Dv]   per-head value
    indices: torch.Tensor,    # [L, topk] int (-1 padded)
    *,
    sm_scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    q_chunk: int = 512,
    checkpoint: bool = True,
) -> torch.Tensor:
    """Memory-bounded, autograd-differentiable sparse attention (per-head Q/K/V).

    Identical result to :func:`sparse_attend_ref` but streams over query chunks and
    (by default) gradient-checkpoints each chunk, so the per-query gathered key/value
    tensors ``[q_chunk, K, H, D]`` are never all resident at once. This is what makes
    the sparse stage fit: at q_chunk=512, K=2048, H=16, D=192 the gather is ~6GB
    regardless of the full sequence length L.
    """
    from torch.utils.checkpoint import checkpoint as _ckpt

    L, H, Dqk = q.shape
    Dv = v.shape[-1]
    dev = q.device
    if sm_scale is None:
        sm_scale = Dqk ** -0.5
    if cu_seqlens is not None:
        try:
            from .flashkl_warmup import prepare_ks_ke
        except ImportError:
            from flashkl_warmup import prepare_ks_ke
        _, ke = prepare_ks_ke(cu_seqlens)
        q_pos = ke - 1
    else:
        q_pos = torch.arange(L, device=dev)

    idx_clamped = indices.long().clamp(min=0)                       # [L,K]
    valid = (indices >= 0) & (indices <= q_pos.view(L, 1))          # [L,K]

    outs = []
    use_ckpt = checkpoint and torch.is_grad_enabled() and q.requires_grad
    for t0 in range(0, L, q_chunk):
        t1 = min(t0 + q_chunk, L)
        q_c = q[t0:t1]
        idx_c = idx_clamped[t0:t1]
        valid_c = valid[t0:t1]
        if use_ckpt:
            o = _ckpt(_attend_chunk, q_c, k, v, idx_c, valid_c, sm_scale,
                      use_reentrant=False)
        else:
            o = _attend_chunk(q_c, k, v, idx_c, valid_c, sm_scale)
        outs.append(o)
    return torch.cat(outs, dim=0)                                   # [L,H,Dv]


# --------------------------------------------------------------------------- #
#  absorbed-MLA sparse attention (latent space; ~9x smaller gather)
# --------------------------------------------------------------------------- #
def _attend_absorbed_chunk(q_lat_c, latent, W_UV, idx_c, valid_c, sm_scale, d_c):
    """One query-chunk of absorbed-space sparse MLA attention.

    q_lat_c [c,H,Dl]  per-head query in latent space = concat(q_nope·W_UK, q_pe)
    latent  [L,Dl]    shared key latent = concat(c_kv, k_pe), Dl = d_c + rope
    W_UV    [H,Dv,d_c] value up-projection (per head), applied to the context latent
    idx_c   [c,K] clamped>=0 ; valid_c [c,K] bool ; d_c = kv_lora_rank

    The only big intermediate is the SHARED latent gather ``[c,K,Dl]`` (Dl≈576),
    vs the per-head path's ``[c,K,H,Dqk]+[c,K,H,Dv]`` (≈ H·320). Recomputed in
    backward under checkpoint. Value needs no separate gather: it is the first d_c
    dims of the same gathered latent, up-projected by W_UV after the softmax."""
    c, K = idx_c.shape
    lat_sel = latent[idx_c]                                         # [c,K,Dl]
    scores = torch.einsum("thd,tkd->thk", q_lat_c, lat_sel) * sm_scale  # [c,H,K]
    scores = scores.masked_fill(~valid_c.view(c, 1, K), float("-inf"))
    probs = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    ctx = torch.einsum("thk,tkl->thl", probs, lat_sel[..., :d_c])   # [c,H,d_c]
    out = torch.einsum("thl,hvl->thv", ctx, W_UV)                   # [c,H,Dv]
    all_masked = ~valid_c.any(-1)
    if all_masked.any():
        out = out.masked_fill(all_masked.view(c, 1, 1), 0.0)
    return out


def sparse_attend_absorbed_chunked(
    q_lat: torch.Tensor,      # [L, H, Dl]  per-head query in latent space
    latent: torch.Tensor,     # [L, Dl]     shared key latent (c_kv | k_pe)
    W_UV: torch.Tensor,       # [H, Dv, d_c] value up-projection
    indices: torch.Tensor,    # [L, topk] int (-1 padded)
    *,
    d_c: int,                 # kv_lora_rank (the value-carrying latent width)
    sm_scale: float,
    cu_seqlens: Optional[torch.Tensor] = None,
    q_chunk: int = 512,
    checkpoint: bool = True,
    q_global_pos=None,        # [Lq] true global query position (CP); None -> local
) -> torch.Tensor:
    """Memory-bounded sparse MLA attention in DeepSeek-V2 absorbed latent space.

    Mathematically identical to the per-head :func:`sparse_attend_chunked` (verified
    to ~1e-15 in fp64), but gathers only the shared latent ``[q_chunk,K,Dl]`` per
    chunk instead of per-head K and V — ~9x less gather traffic and the dominant
    saving in the sparse stage's backward. Returns [Lq, H, Dv].

    CP: q_lat holds Lq local queries, latent holds the FULL Lk gathered keys, indices
    are GLOBAL key positions; pass ``q_global_pos`` [Lq] so a query attends keys up to
    its true global position."""
    from torch.utils.checkpoint import checkpoint as _ckpt

    L, H, Dl = q_lat.shape
    dev = q_lat.device
    if q_global_pos is not None:
        q_pos = q_global_pos.long()
    elif cu_seqlens is not None:
        try:
            from .flashkl_warmup import prepare_ks_ke
        except ImportError:
            from flashkl_warmup import prepare_ks_ke
        _, ke = prepare_ks_ke(cu_seqlens)
        q_pos = ke - 1
    else:
        q_pos = torch.arange(L, device=dev)

    idx_clamped = indices.long().clamp(min=0)                       # [L,K]
    valid = (indices >= 0) & (indices <= q_pos.view(L, 1))          # [L,K]

    outs = []
    use_ckpt = checkpoint and torch.is_grad_enabled() and q_lat.requires_grad
    for t0 in range(0, L, q_chunk):
        t1 = min(t0 + q_chunk, L)
        args = (q_lat[t0:t1], latent, W_UV, idx_clamped[t0:t1], valid[t0:t1],
                sm_scale, d_c)
        if use_ckpt:
            o = _ckpt(_attend_absorbed_chunk, *args, use_reentrant=False)
        else:
            o = _attend_absorbed_chunk(*args)
        outs.append(o)
    return torch.cat(outs, dim=0)                                   # [L,H,Dv]
_KERNEL_CACHE = {"loaded": False, "fn": None}


def _load_kernel():
    if _KERNEL_CACHE["loaded"]:
        return _KERNEL_CACHE["fn"]
    _KERNEL_CACHE["loaded"] = True
    try:
        import os
        import sys
        kpath = os.environ.get("OPENDSA_SPARSE_MLA_KERNEL_PATH")
        if kpath and kpath not in sys.path:
            sys.path.insert(0, kpath)
        from kernels.sparse_mla_wrapper import sparse_mla_attn  # noqa
        _KERNEL_CACHE["fn"] = sparse_mla_attn
    except Exception as e:  # pragma: no cover - kernel optional
        _KERNEL_CACHE["fn"] = None
        _KERNEL_CACHE["err"] = repr(e)
    return _KERNEL_CACHE["fn"]


def sparse_mla_kernel(
    q: torch.Tensor,          # [L, H, 576]
    kv: torch.Tensor,         # [L, 576]
    indices: torch.Tensor,    # [L, topk] int32
    *,
    d_v: int = 512,
    sm_scale: Optional[float] = None,
    block_I: int = 32,
) -> torch.Tensor:
    """Forward-only TileLang sparse MLA (bf16, sm_90). Returns [L,H,d_v].
    Falls back to ``sparse_mla_ref`` if the kernel can't be loaded/run."""
    fn = _load_kernel()
    if fn is None or not q.is_cuda:
        return sparse_mla_ref(q, kv, indices, d_v=d_v, sm_scale=sm_scale)
    try:
        return fn(q, kv, indices.to(torch.int32), sm_scale=sm_scale, block_I=block_I)
    except Exception:
        return sparse_mla_ref(q, kv, indices, d_v=d_v, sm_scale=sm_scale)


# --------------------------------------------------------------------------- #
#  dense-masked reference (materializes [L,H,L]) — ground truth for the ref
# --------------------------------------------------------------------------- #
def dense_masked_attn(q, kv, allow_mask, *, d_v=512, sm_scale=None):
    """Full [L,H,L] masked softmax attention over an arbitrary boolean key mask.
    Used only to validate ``sparse_mla_ref`` for small sizes."""
    L, H, Dqk = q.shape
    if sm_scale is None:
        sm_scale = Dqk ** -0.5
    v = kv[..., :d_v]
    scores = torch.einsum("thd,jd->thj", q, kv) * sm_scale         # [L,H,L]
    scores = scores.masked_fill(~allow_mask.view(L, 1, L), float("-inf"))
    probs = torch.nan_to_num(torch.softmax(scores, -1), nan=0.0)
    return torch.einsum("thj,jd->thd", probs, v)


def _selftest():
    torch.manual_seed(0)
    f64 = torch.float64
    L, H, Dqk, d_v, K = 24, 4, 32, 20, 6

    def rel(a, b):
        return (a - b).norm().item() / (b.norm().item() + 1e-30)

    q = torch.randn(L, H, Dqk, dtype=f64, requires_grad=True)
    kv = torch.randn(L, Dqk, dtype=f64, requires_grad=True)

    # random causal top-k selection (self always included)
    q_pos = torch.arange(L)
    indices = torch.full((L, K), -1, dtype=torch.long)
    for t in range(L):
        cand = torch.arange(t + 1)
        perm = cand[torch.randperm(t + 1)][: min(K, t + 1)]
        if t not in perm.tolist():
            perm[-1] = t
        indices[t, : perm.numel()] = perm

    out = sparse_mla_ref(q, kv, indices, d_v=d_v)
    out.sum().backward()
    gq, gkv = q.grad.clone(), kv.grad.clone()
    q.grad = None; kv.grad = None

    # dense-masked equivalent: allow exactly the selected indices
    allow = torch.zeros(L, L, dtype=torch.bool)
    for t in range(L):
        sel = indices[t][indices[t] >= 0]
        allow[t, sel] = True
    out_d = dense_masked_attn(q, kv, allow, d_v=d_v)
    out_d.sum().backward()
    gq_d, gkv_d = q.grad.clone(), kv.grad.clone()

    e_out = rel(out, out_d)
    e_gq = rel(gq, gq_d)
    e_gkv = rel(gkv, gkv_d)
    ok = max(e_out, e_gq, e_gkv) < 1e-12
    print("=" * 70)
    print("sparse_mla_ref vs dense-masked attention (float64)")
    print("=" * 70)
    print(f"  out rel-err={e_out:.2e}  dq={e_gq:.2e}  dkv={e_gkv:.2e}  -> OK(<1e-12): {ok}")

    # --- chunked per-head attention vs sparse_attend_ref (separate Q/K/V) ---
    Hh, Dqk2, Dv2 = 4, 16, 12
    qh = torch.randn(L, Hh, Dqk2, dtype=f64, requires_grad=True)
    kh = torch.randn(L, Hh, Dqk2, dtype=f64, requires_grad=True)
    vh = torch.randn(L, Hh, Dv2, dtype=f64, requires_grad=True)
    cu = torch.tensor([0, 9, L])  # packed: two docs
    ref = sparse_attend_ref(qh, kh, vh, indices, cu_seqlens=cu)
    ref.sum().backward()
    gref = {n: t.grad.clone() for n, t in [("q", qh), ("k", kh), ("v", vh)]}
    qh.grad = kh.grad = vh.grad = None
    ch = sparse_attend_chunked(qh, kh, vh, indices, cu_seqlens=cu, q_chunk=5)
    ch.sum().backward()
    gch = {n: t.grad.clone() for n, t in [("q", qh), ("k", kh), ("v", vh)]}
    e_fwd = rel(ch, ref)
    e_bwd = {n: rel(gch[n], gref[n]) for n in gref}
    ok_ch = e_fwd < 1e-12 and all(v < 1e-12 for v in e_bwd.values())
    ok = ok and ok_ch
    print("-" * 70)
    print("sparse_attend_chunked vs sparse_attend_ref (packed causal, float64)")
    print(f"  out rel-err={e_fwd:.2e}  " +
          "  ".join(f"d{n}={v:.2e}" for n, v in e_bwd.items()) +
          f"  -> OK(<1e-12): {ok_ch}")

    # --- absorbed-MLA attention vs per-head reference (constructed MLA tensors) ---
    Hh, nope, rope, d_c, Dv3 = 4, 8, 6, 20, 10
    Dl = d_c + rope
    c_kv = torch.randn(L, d_c, dtype=f64)
    k_pe = torch.randn(L, rope, dtype=f64)
    latent = torch.cat([c_kv, k_pe], dim=-1).requires_grad_(True)   # [L,Dl]
    W_UK = torch.randn(Hh, nope, d_c, dtype=f64)                    # up-proj key
    W_UV = torch.randn(Hh, Dv3, d_c, dtype=f64, requires_grad=True) # up-proj value
    q_nope = torch.randn(L, Hh, nope, dtype=f64)
    q_pe = torch.randn(L, Hh, rope, dtype=f64)
    # per-head Q/K/V implied by this MLA decomposition
    k_nope = torch.einsum("ld,hnd->lhn", c_kv, W_UK)               # [L,H,nope]
    Km3 = torch.cat([k_nope, k_pe.unsqueeze(1).expand(L, Hh, rope)], dim=-1)
    Qm3 = torch.cat([q_nope, q_pe], dim=-1)                         # [L,H,nope+rope]
    Vm3 = torch.einsum("ld,hvd->lhv", c_kv, W_UV)                   # [L,H,Dv]
    sm3 = (nope + rope) ** -0.5
    ref3 = sparse_attend_ref(Qm3, Km3, Vm3, indices, sm_scale=sm3, cu_seqlens=cu)
    # absorbed query: [q_nope·W_UK, q_pe]
    q_absorb = torch.einsum("lhn,hnd->lhd", q_nope, W_UK)           # [L,H,d_c]
    q_lat = torch.cat([q_absorb, q_pe], dim=-1)                     # [L,H,Dl]
    abs_out = sparse_attend_absorbed_chunked(q_lat, latent, W_UV, indices, d_c=d_c,
                                             sm_scale=sm3, cu_seqlens=cu, q_chunk=5)
    e_abs = rel(abs_out, ref3)
    # grad check: differentiate the absorbed path w.r.t. latent and W_UV
    abs_out.sum().backward()
    ok_abs = e_abs < 1e-12 and latent.grad is not None and W_UV.grad is not None
    ok = ok and ok_abs
    print("-" * 70)
    print("sparse_attend_absorbed_chunked vs per-head ref (MLA latent, float64)")
    print(f"  out rel-err={e_abs:.2e}  grad(latent,W_UV) present={latent.grad is not None and W_UV.grad is not None}"
          f"  -> OK(<1e-12): {ok_abs}")
    print("=" * 70)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
