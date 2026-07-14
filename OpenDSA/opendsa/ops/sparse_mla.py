"""sparse_mla.py — sparse MLA attention for the DSA sparse-training stage.

At sparse-training time each query attends only to the top-k keys chosen by the
Lightning Indexer. This module provides ``sparse_attend_absorbed_chunked``: a
memory-bounded, autograd-differentiable sparse MLA attention that runs in
DeepSeek-V2's absorbed latent space (gathers the shared latent once per query
chunk instead of per-head K/V), query-chunked + gradient-checkpointed.

MLA absorbed contract (matches the teacher used in warmup):
    q  : [L, H, Dqk]     per-head query in latent space (Dqk = kv_lora + rope, e.g. 576)
    kv : [L, Dqk]        shared K==softmax key; the first d_v dims (=kv_lora, 512)
                         also serve as V (MLA fuses K and V in the latent space)
    indices : [L, topk]  int32, per-query selected key positions (-1 padded)
    output  : [L, H, d_v]
"""
from __future__ import annotations

from typing import Optional

import torch


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

    Gathers only the shared latent ``[q_chunk,K,Dl]`` per chunk instead of per-head
    K and V — ~9x less gather traffic and the dominant saving in the sparse stage's
    backward. Returns [Lq, H, Dv].

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
