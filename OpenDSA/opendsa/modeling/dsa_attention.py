"""dsa_attention.py — monkeypatch DeepSeek-V2 MLA attention into DSA.

We do NOT rewrite the model. We attach a ``LightningIndexer`` to each
``DeepseekV2Attention`` module and replace its ``forward`` with a DSA-aware one
that has two modes (selected by ``self.dsa_mode``):

  * "warmup": run the ORIGINAL dense MLA attention unchanged (so the model output
    and LM loss are identical to the base model), AND compute the indexer's KL
    distillation loss against the model's own attention distribution. The teacher
    (query_states, key_states) are detached; indexer inputs are detached so no
    gradient flows into the frozen backbone. The per-layer indexer loss is stashed
    on a module-level registry for the trainer to sum. Uses FlashKL -> O(L·H).

  * "sparse": the indexer selects top-k keys; attention runs only over them
    (sparse MLA). The LM loss flows through the sparse output; the indexer keeps
    training via a sparse-set KL. (Backbone is trainable in this stage.)

Teacher faithfulness: teacher Q/K are exactly the model's ``query_states`` /
``key_states`` ([B,H,L,q_head_dim]); teacher softmax_scale is the model's own
``self.softmax_scale`` (includes YaRN mscale). This matches the base attention.

Currently supports batch size handling per-sequence via an optional ``cu_seqlens``
carried on the module (set by the collator/trainer for packed data); if absent, a
single causal sequence per batch row is assumed.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .indexer import LightningIndexer
from ..ops import (
    flashkl_warmup_loss,
    auto_warmup_tile,
    flashkl_sparse_loss,
    sparse_kl_chunked,
    indexer_select_topk,
    sparse_attend_ref,
    sparse_attend_chunked,
    sparse_attend_absorbed_chunked,
    indexer_topk_recall,
)


# --------------------------------------------------------------------------- #
#  per-forward indexer-loss registry (trainer reads + clears each step)
# --------------------------------------------------------------------------- #
class IndexerLossRegistry:
    """Collects per-layer indexer losses during a forward.

    Two modes:
      * default: stash the (graph-carrying) loss tensors; the trainer sums them and
        does one backward. Simple, but keeps ALL layers' graphs alive at once ->
        O(num_layers · L · H) peak activation (80GB at 32k across 27 layers).
      * eager-backward (``set_eager_backward``): the moment a layer produces its
        loss we backward it immediately and keep only the detached scalar. Because
        the per-layer indexer losses are INDEPENDENT (hidden states flow under
        no_grad, teacher detached), summing-then-backward and per-layer-backward
        give identical indexer grads — but the latter frees each layer's graph
        before the next layer runs, so peak activation is ~1 layer, not all of them.
        A ``scale`` (e.g. 1/grad_accum) is applied to each layer's backward.
    """

    def __init__(self):
        self._losses = []
        self._recalls = []
        self._entropies = []    # per-layer detached teacher entropy (logging-only)
        self._eager = False
        self._scale = 1.0
        self._eager_sum = 0.0   # running detached sum of scaled losses (for logging)

    def reset(self):
        self._losses = []
        self._recalls = []
        self._entropies = []
        self._eager_sum = 0.0

    def set_eager_backward(self, flag: bool, scale: float = 1.0):
        self._eager = flag
        self._scale = scale

    def add(self, loss, recall=None, entropy=None):
        if self._eager and loss is not None and loss.requires_grad:
            (loss * self._scale).backward()
            self._eager_sum += float(loss.detach())
        else:
            self._losses.append(loss)
        if recall is not None:
            self._recalls.append(recall)
        # teacher entropy is a logging-only diagnostic (no grad); the per-layer
        # trainable loss is CE = KL(teacher||student) + H(teacher), so true KL is
        # recovered as loss - entropy. Stored with the same normalization as loss.
        if entropy is not None:
            self._entropies.append(float(entropy))

    def total(self):
        if not self._losses:
            return None
        return torch.stack(self._losses).sum()

    def eager_loss_sum(self):
        """Detached sum of the per-layer losses backwarded in eager mode (logging)."""
        return self._eager_sum

    def mean_recall(self):
        if not self._recalls:
            return None
        return sum(self._recalls) / len(self._recalls)

    def entropy_sum(self):
        """Sum of per-layer teacher entropies (matches how losses are summed in
        total()/eager_loss_sum()); None if no entropy was recorded this step.
        true_KL = (indexer_loss) - (this), reported on logging steps only."""
        if not self._entropies:
            return None
        return sum(self._entropies)


REGISTRY = IndexerLossRegistry()


def _cu_from_batch(B, L, device, cu_seqlens=None):
    """Return per-row cu_seqlens list. If cu_seqlens given (1D packed for a single
    row), use it; else one causal sequence [0,L] per row."""
    if cu_seqlens is not None:
        return [cu_seqlens.to(device)] * B
    base = torch.tensor([0, L], dtype=torch.long, device=device)
    return [base] * B


@torch.no_grad()
def _warmup_teacher_entropy(Qm, Km, cu_seqlens, sm_tea, q_chunk=1024,
                            q_global_pos=None, nrow_global=None):
    """Per-row-mean teacher entropy H(p̄) for the FULL causal range (warmup).

    p̄ = head-averaged softmax of the dense teacher attention (same distribution
    flashkl_warmup_loss distills against). The trainable warmup loss equals
    KL(p̄||p̂) + H(p̄), so logging (loss - this) recovers the true KL. Reduction
    matches the loss: mean over valid rows. Under CP pass ``nrow_global`` so each
    rank records its entropy contribution divided by the global row count; summing
    ranks then matches FlashKL's CP-normalized loss. Query-chunked, no_grad,
    teacher only — runs on logging steps only. Qm [Lq,H,D], Km [Lk,H,D] or [Lk,D]."""
    Lq, H, D = Qm.shape
    Lk = Km.shape[0]
    dev = Qm.device
    from ..ops.flashkl_warmup import prepare_ks_ke, prepare_ks_ke_cp
    if q_global_pos is None:
        ks, ke = prepare_ks_ke(cu_seqlens)
    else:
        ks, ke = prepare_ks_ke_cp(cu_seqlens, q_global_pos)
    jj = torch.arange(Lk, device=dev)
    ent_sum = torch.zeros((), dtype=torch.float64, device=dev)
    nrow = torch.zeros((), dtype=torch.float64, device=dev)
    for t0 in range(0, Lq, q_chunk):
        t1 = min(t0 + q_chunk, Lq)
        c = t1 - t0
        causal = (jj.view(1, Lk) >= ks[t0:t1].view(c, 1)) & \
                 (jj.view(1, Lk) < ke[t0:t1].view(c, 1))            # [c,Lk]
        if Km.dim() == 3:
            a = torch.einsum("thd,jhd->thj", Qm[t0:t1], Km) * sm_tea
        else:
            a = torch.einsum("thd,jd->thj", Qm[t0:t1], Km) * sm_tea
        a = a.masked_fill(~causal.view(c, 1, Lk), float("-inf"))
        pbar = torch.nan_to_num(torch.softmax(a, -1), nan=0.0).mean(1)  # [c,Lk]
        safe = pbar.clamp_min(1e-30)
        ent = -(pbar * torch.log(safe)).sum(-1)                    # [c]
        vr = causal.any(-1)
        ent_sum += ent[vr].to(torch.float64).sum()
        nrow += vr.to(torch.float64).sum()
    denom = (torch.as_tensor(float(nrow_global), dtype=torch.float64, device=dev)
             if nrow_global is not None else nrow.clamp_min(1))
    return float((ent_sum / denom.clamp_min(1)).item())


@torch.no_grad()
def _sparse_teacher_entropy(Qm, Km, ids, sm_tea, q_chunk=512, nrow_global=None):
    """Per-row-mean teacher entropy H(p̄) over the SELECTED top-k set (sparse).

    Mirrors _teacher_pbar_chunk's distribution (head-averaged softmax over the
    selected keys), so the sparse trainable loss = KL + this entropy and
    (loss - this) is the true KL. Qm [L,H,Dl], Km [L,Dl] (absorbed latent),
    ids [L,K] int (-1 pad). Reduction matches sparse_kl_chunked; under CP pass
    nrow_global so all ranks' contributions sum to the global mean."""
    from ..ops.topk_select import _teacher_pbar_chunk
    L = Qm.shape[0]
    ids64 = ids.to(torch.int64)
    ent_sum = torch.zeros((), dtype=torch.float64, device=Qm.device)
    nrow = torch.zeros((), dtype=torch.float64, device=Qm.device)
    for t0 in range(0, L, q_chunk):
        t1 = min(t0 + q_chunk, L)
        ids_c = ids64[t0:t1]
        pbar = _teacher_pbar_chunk(Qm[t0:t1], Km, ids_c, float(sm_tea))  # [c,K]
        safe = pbar.clamp_min(1e-30)
        ent = -(pbar * torch.log(safe)).sum(-1)                    # [c]
        vr = (ids_c >= 0).any(-1)
        ent_sum += ent[vr].to(torch.float64).sum()
        nrow += vr.to(torch.float64).sum()
    denom = (torch.as_tensor(float(nrow_global), dtype=torch.float64, device=Qm.device)
             if nrow_global is not None else nrow.clamp_min(1))
    return float((ent_sum / denom.clamp_min(1)).item())


# --------------------------------------------------------------------------- #
#  the patched forward
# --------------------------------------------------------------------------- #
def make_dsa_forward(orig_forward):
    def dsa_forward(self, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False,
                    **kwargs):
        mode = getattr(self, "dsa_mode", "warmup")
        if mode == "warmup":
            return _warmup_forward(self, orig_forward, hidden_states, attention_mask,
                                   position_ids, past_key_value, output_attentions,
                                   use_cache, **kwargs)
        elif mode == "sparse":
            return _sparse_forward(self, orig_forward, hidden_states, attention_mask,
                                   position_ids, past_key_value, output_attentions,
                                   use_cache, **kwargs)
        else:  # "dense" — pristine base attention (eval baseline)
            return orig_forward(self, hidden_states, attention_mask, position_ids,
                                past_key_value, output_attentions, use_cache, **kwargs)
    return dsa_forward


def _compute_teacher_qk(self, hidden_states, position_ids):
    """Reproduce DeepSeek-V2 MLA query_states/key_states (per-head, [B,H,L,qhd])
    and the rope cos/sin, WITHOUT running attention. Mirrors the eager forward."""
    bsz, q_len, _ = hidden_states.size()
    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
          .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
          .transpose(1, 2))
    k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    from transformers.models.auto import modeling_auto  # noqa (ensure transformers loaded)
    cos, sin = self.rotary_emb(value_states, seq_len=q_len)
    # apply_rotary_pos_emb is defined in the model's module; fetch via the class
    apply_rope = _get_apply_rotary(self)
    q_pe, k_pe = apply_rope(q_pe, k_pe, cos, sin, position_ids)

    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim:] = q_pe
    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim:] = k_pe
    return query_states, key_states, value_states, cos, sin


def _compute_teacher_absorbed(self, hidden_states, position_ids, rope_seq_len=None):
    """DeepSeek-V2 MLA teacher in ABSORBED latent space, WITHOUT materializing
    per-head K/V. Returns:
      q_lat  [B,H,L,Dl]  per-head query = concat(q_nope·W_UK, roped q_pe)
      latent [B,L,Dl]    shared key latent = concat(kv_a_layernorm(c_kv), roped k_pe)
      W_UV   [H,v_head,kv_lora]  value up-projection (from kv_b_proj)
      cos,sin            rope tables (for the indexer)
    with Dl = kv_lora_rank + qk_rope_head_dim. Attention/value in this space equal
    the per-head path exactly (verified fp64 ~1e-15). This is the fast sparse path:
    it gathers the shared [.,K,Dl] latent once instead of per-head K[.,K,H,192]+V.

    ``rope_seq_len`` overrides the rotary-table length: under CP the local tokens'
    global positions run up to L_global, so the table must span the full sequence,
    not just the local q_len. Defaults to q_len (non-CP)."""
    bsz, q_len, _ = hidden_states.size()
    H = self.num_heads
    nope, rope = self.qk_nope_head_dim, self.qk_rope_head_dim
    d_c, v_head = self.kv_lora_rank, self.v_head_dim

    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, H, self.q_head_dim).transpose(1, 2)          # [B,H,L,qhd]
    q_nope, q_pe = torch.split(q, [nope, rope], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    c_kv, k_pe = torch.split(compressed_kv, [d_c, rope], dim=-1)
    c_kv = self.kv_a_layernorm(c_kv)                                    # [B,L,d_c]
    k_pe = k_pe.view(bsz, q_len, 1, rope).transpose(1, 2)              # [B,1,L,rope]

    cos, sin = self.rotary_emb(hidden_states, seq_len=rope_seq_len or q_len)
    apply_rope = _get_apply_rotary(self)
    q_pe, k_pe = apply_rope(q_pe, k_pe, cos, sin, position_ids)         # [B,H,L,rope],[B,1,L,rope]

    # absorb q_nope into latent via W_UK (kv_b_proj rows for the nope part)
    W = self.kv_b_proj.weight.view(H, nope + v_head, d_c)               # [H, nope+v_head, d_c]
    W_UK = W[:, :nope, :]                                               # [H,nope,d_c]
    W_UV = W[:, nope:, :]                                               # [H,v_head,d_c]
    q_absorb = torch.einsum("bhln,hnd->bhld", q_nope, W_UK)            # [B,H,L,d_c]
    q_lat = torch.cat([q_absorb, q_pe], dim=-1)                        # [B,H,L,Dl]
    latent = torch.cat([c_kv, k_pe[:, 0]], dim=-1)                     # [B,L,Dl]
    return q_lat, latent, W_UV, cos, sin


_APPLY_ROPE_CACHE = {}


def _get_apply_rotary(self):
    mod = type(self).__module__
    if mod not in _APPLY_ROPE_CACHE:
        import importlib
        m = importlib.import_module(mod)
        _APPLY_ROPE_CACHE[mod] = m.apply_rotary_pos_emb
    return _APPLY_ROPE_CACHE[mod]


def _indexer_rope_cossin(self, cos, sin, position_ids):
    """The model's rope cos/sin are [seq, rope_dim] (or [1,seq,rope_dim]); gather
    to [B,L,rope_dim] at the given positions for the indexer's own rope apply."""
    # DeepseekV2RotaryEmbedding returns cos,sin of shape [seq_len, dim]; the model
    # indexes them by position_ids inside apply_rotary_pos_emb. Reproduce that.
    if cos.dim() == 2:
        c = cos[position_ids]  # [B,L,rope_dim]
        s = sin[position_ids]
    else:
        c = cos
        s = sin
    return c, s


def _warmup_forward(self, orig_forward, hidden_states, attention_mask, position_ids,
                    past_key_value, output_attentions, use_cache, **kwargs):
    if getattr(self, "cp_size", 1) > 1:
        return _warmup_forward_cp(self, hidden_states, position_ids)
    B, L, _ = hidden_states.shape
    dev = hidden_states.device
    if position_ids is None:
        position_ids = torch.arange(L, device=dev).unsqueeze(0).expand(B, -1)

    # --- teacher Q/K/V (detached; no grad into frozen backbone) ---
    with torch.no_grad():
        qs, ks_, vs, cos, sin = _compute_teacher_qk(self, hidden_states, position_ids)
    # per-head teacher: [B,H,L,qhd] -> [B,L,H,qhd]
    Qm = qs.transpose(1, 2)
    Km = ks_.transpose(1, 2)

    # --- indexer (inputs detached) ---
    q_input = hidden_states.detach()
    if self.q_lora_rank is not None:
        # indexer q reads the q_lora latent for fidelity to the reference; recompute cheaply
        q_input = self.q_a_layernorm(self.q_a_proj(hidden_states)).detach()
    c_idx, s_idx = _indexer_rope_cossin(self, cos, sin, position_ids)
    q_idx, k_idx, w_idx = self.indexer(hidden_states.detach(), q_input, c_idx, s_idx)

    # --- per-row FlashKL warmup loss ---
    cus = _cu_from_batch(B, L, dev, getattr(self, "cu_seqlens", None))
    sm_tea = float(self.softmax_scale)
    # memory-adaptive key-tile: shrink from the configured cap when free memory is
    # tight (peak is O(Lq·H·tile) fp32 scratch), never exceeding it.
    warmup_tile = auto_warmup_tile(L, Qm.shape[2], getattr(self, "dsa_tile", 512), dev)
    losses = []
    with torch.no_grad():
        do_recall = getattr(self, "dsa_log_recall", False)
        recalls = []
    for b in range(B):
        lb = flashkl_warmup_loss(
            q_idx[b], k_idx[b], w_idx[b],
            Qm[b].to(torch.float32), Km[b].to(torch.float32),
            cus[b], sm_scale_teacher=sm_tea,
            sm_scale_index=self.indexer.softmax_scale, tile=warmup_tile,
        )
        losses.append(lb)
    loss = torch.stack(losses).mean()
    recall = None
    entropy = None
    if getattr(self, "dsa_log_recall", False) or getattr(self, "dsa_log_kl", False):
        with torch.no_grad():
            if getattr(self, "dsa_log_recall", False):
                recall = indexer_topk_recall(
                    q_idx[0].float(), k_idx[0].float(), w_idx[0].float(),
                    Qm[0].float(), Km[0].float(), cus[0],
                    getattr(self, "dsa_topk", 2048),
                    sm_scale_teacher=sm_tea, sm_scale_index=self.indexer.softmax_scale)
            entropy = _warmup_teacher_entropy(
                Qm[0].float(), Km[0].float(), cus[0], sm_tea)
    REGISTRY.add(loss, recall, entropy)

    # Model output for the (unchanged, frozen) LM path. We DON'T call orig_forward:
    # the base eager attention materializes [B,H,L,L] (32GB at L=32k). Since the
    # backbone is frozen in warmup, we only need faithful hidden states for the next
    # layer's indexer — so we run the identical causal MLA attention via flash-attn
    # (O(L) memory, per-doc masking through cu_seqlens), then o_proj.
    with torch.no_grad():
        out = _warmup_attn_output(self, qs, ks_, vs, B, L, dev)
        out = out.reshape(B, L, self.num_heads * self.v_head_dim)
        out = self.o_proj(out.to(self.o_proj.weight.dtype))
    return out, None, past_key_value


def _warmup_attn_output(self, qs, ks_, vs, B, L, dev):
    """Memory-efficient causal MLA attention output for the warmup LM path.
    qs/ks_ [B,H,L,qhd], vs [B,H,L,vhd]. Uses flash-attn varlen so no [L,L] matrix is
    built; honors packed per-doc boundaries via cu_seqlens when present. Returns
    [B,L,H,vhd]."""
    from flash_attn import flash_attn_varlen_func
    H = self.num_heads
    qhd, vhd = self.q_head_dim, self.v_head_dim
    # flash varlen wants [total, H, D] with a cu_seqlens int32 vector
    cu = getattr(self, "cu_seqlens", None)
    q = qs.transpose(1, 2).reshape(B * L, H, qhd).to(torch.bfloat16)
    k = ks_.transpose(1, 2).reshape(B * L, H, qhd).to(torch.bfloat16)
    v = vs.transpose(1, 2).reshape(B * L, H, vhd).to(torch.bfloat16)
    # flash-attn requires head_dim_v == head_dim_qk; DS-V2 has vhd(128) < qhd(192).
    # Pad V to qhd, run flash, then slice the first vhd dims of the output back.
    if vhd < qhd:
        v = F.pad(v, (0, qhd - vhd))
    if cu is not None and B == 1:
        cuq = cu.to(dev).to(torch.int32)
        max_s = int((cuq[1:] - cuq[:-1]).max().item())
    else:
        cuq = torch.arange(0, B * L + 1, L, device=dev, dtype=torch.int32)
        max_s = L
    out = flash_attn_varlen_func(q, k, v, cuq, cuq, max_s, max_s,
                                 softmax_scale=float(self.softmax_scale), causal=True)
    return out[..., :vhd].view(B, L, H, vhd)


def _warmup_forward_cp(self, hidden_states, position_ids):
    """Context-parallel warmup for one attention layer. This rank holds Lloc local
    query tokens; keys are all-gathered to the full Lk = Lloc*cp.

    Frozen backbone: teacher latent all-gathered (no grad); student indexer key
    all-gathered with autograd (AllGatherSeq). FlashKL runs local-Q vs full-K with
    true global positions; its per-layer loss is eager-backwarded (grads on this
    rank's indexer params; the CP-group all-reduce happens in the trainer). The LM
    path reconstructs full per-head K/V from the gathered latent and runs local-Q
    vs full-K/V causal flash attention -> local hidden for the next layer."""
    from ..dist import cp_size, cp_rank, zigzag_local_gpos, all_gather_seq
    B, Lloc, _ = hidden_states.shape
    assert B == 1, "CP warmup assumes batch size 1 (packed long-context)"
    dev = hidden_states.device
    n = cp_size()
    Lk = Lloc * n
    gpos = zigzag_local_gpos(Lk, dev).unsqueeze(0)  # [1,Lloc] global (zigzag order)

    # local absorbed teacher (rope table spans the full global sequence)
    with torch.no_grad():
        q_lat, latent_loc, W_UV, cos, sin = _compute_teacher_absorbed(
            self, hidden_states, gpos, rope_seq_len=Lk)
    Qlat_loc = q_lat.transpose(1, 2)                                    # [1,Lloc,H,Dl]

    # indexer (student): local q/k/w; inputs detached (frozen backbone)
    q_input = hidden_states.detach()
    if self.q_lora_rank is not None:
        q_input = self.q_a_layernorm(self.q_a_proj(hidden_states)).detach()
    c_idx, s_idx = _indexer_rope_cossin(self, cos, sin, gpos)
    q_idx, k_idx, w_idx = self.indexer(hidden_states.detach(), q_input, c_idx, s_idx)

    # all-gather keys to full sequence: teacher latent (no grad), indexer k (autograd)
    latent_full = all_gather_seq(latent_loc[0].detach(), grad=False)    # [Lk,Dl]
    k_idx_full = all_gather_seq(k_idx[0], grad=True)                    # [Lk,df]

    # FlashKL: local queries vs full keys; teacher query = absorbed q_lat, shared key
    # = latent_full (Km.dim()==2 branch); causal by true global position.
    cu_full = getattr(self, "cu_seqlens", None)
    if cu_full is None:
        cu_full = torch.tensor([0, Lk], device=dev)
    else:
        cu_full = cu_full.to(dev)
    sm_tea = float(self.softmax_scale)
    # memory-adaptive key-tile (CP: local Lloc queries, full Lk key axis).
    warmup_tile = auto_warmup_tile(Lloc, Qlat_loc.shape[2], getattr(self, "dsa_tile", 512), dev)
    loss = flashkl_warmup_loss(
        q_idx[0], k_idx_full, w_idx[0],
        Qlat_loc[0].float(), latent_full.float(),
        cu_full, sm_scale_teacher=sm_tea,
        sm_scale_index=self.indexer.softmax_scale, tile=warmup_tile,
        q_global_pos=gpos[0], nrow_global=Lk)

    recall = None
    entropy = None
    if getattr(self, "dsa_log_recall", False) or getattr(self, "dsa_log_kl", False):
        with torch.no_grad():
            if getattr(self, "dsa_log_recall", False):
                recall = indexer_topk_recall(
                    q_idx[0].float(), k_idx_full.float(), w_idx[0].float(),
                    Qlat_loc[0].float(), latent_full.float(), cu_full,
                    getattr(self, "dsa_topk", 2048), sm_scale_teacher=sm_tea,
                    sm_scale_index=self.indexer.softmax_scale, q_global_pos=gpos[0])
            entropy = _warmup_teacher_entropy(
                Qlat_loc[0].float(), latent_full.float(), cu_full, sm_tea,
                q_global_pos=gpos[0], nrow_global=Lk)
    REGISTRY.add(loss, recall, entropy)

    # LM path (frozen, no grad): reconstruct full per-head K/V from gathered latent,
    # run local-Q (per-head, from q_lat's original q) vs full-K/V causal attention.
    with torch.no_grad():
        out = _warmup_attn_output_cp(self, Qlat_loc[0], latent_full, gpos[0], cu_full, dev)
        out = out.reshape(B, Lloc, self.num_heads * self.v_head_dim)
        out = self.o_proj(out.to(self.o_proj.weight.dtype))
    return out, None, None


def _warmup_attn_output_cp(self, q_lat_loc, latent_full, gpos, cu_full, dev):
    """LM-path DENSE causal attention for CP warmup, in absorbed latent space,
    memory-bounded by online-softmax tiling (NO [Lloc,Lk] matrix, NO dense index
    list — the earlier dense-index form was O(Lloc·maxw) and blew up at 200k).

    q_lat_loc [Lloc,H,Dl] local queries; latent_full [Lk,Dl] gathered keys; gpos
    [Lloc] global query positions; cu_full full-seq doc boundaries. Each query t
    attends keys in [ks[t], gpos[t]] (its document, causal). Streams key tiles with
    a flash-style running (max, denom, acc). Returns [Lloc,H,vhd]."""
    H = self.num_heads
    d_c, v_head = self.kv_lora_rank, self.v_head_dim
    W = self.kv_b_proj.weight.view(H, self.qk_nope_head_dim + v_head, d_c)
    W_UV = W[:, self.qk_nope_head_dim:, :].to(torch.float32)            # [H,vhd,d_c]
    Lloc, Dl = q_lat_loc.shape[0], q_lat_loc.shape[-1]
    Lk = latent_full.shape[0]
    sm = float(self.softmax_scale)
    ks, ke = _cp_doc_causal(cu_full, gpos)                              # [Lloc] abs key range
    qf = q_lat_loc.float()
    latf = latent_full.float()
    tile = 4096
    qc = getattr(self, "dsa_qchunk", 512)
    outs = []
    for t0 in range(0, Lloc, qc):
        t1 = min(t0 + qc, Lloc)
        q = qf[t0:t1]                                                   # [c,H,Dl]
        c = t1 - t0
        ksc, kec = ks[t0:t1], ke[t0:t1]
        kmax = int(kec.max().item())                                   # no key beyond this
        m = torch.full((c, H), float("-inf"), device=dev)
        denom = torch.zeros(c, H, device=dev)
        acc = torch.zeros(c, H, d_c, device=dev)                       # value latent accum
        for s0 in range(0, kmax, tile):
            s1 = min(s0 + tile, kmax)
            kt = latf[s0:s1]                                           # [b,Dl]
            jt = torch.arange(s0, s1, device=dev)
            valid = (jt.view(1, 1, -1) >= ksc.view(c, 1, 1)) & (jt.view(1, 1, -1) < kec.view(c, 1, 1))  # [c,1,b]
            sc = torch.einsum("chd,jd->chj", q, kt) * sm               # [c,H,b]
            sc = sc.masked_fill(~valid, float("-inf"))
            m_new = torch.maximum(m, sc.amax(-1))
            alpha = torch.exp(m - m_new)
            alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
            p = torch.exp(sc - m_new.unsqueeze(-1))                    # [c,H,b]
            p = torch.where(valid, p, torch.zeros_like(p))
            denom = denom * alpha + p.sum(-1)
            acc = acc * alpha.unsqueeze(-1) + torch.einsum("chj,jd->chd", p, kt[:, :d_c])
            m = m_new
        ctx = acc / denom.clamp_min(1e-30).unsqueeze(-1)              # [c,H,d_c]
        out = torch.einsum("chl,hvl->chv", ctx, W_UV)                 # [c,H,vhd]
        outs.append(out)
    return torch.cat(outs, dim=0)


def _cp_doc_causal(cu_full, gpos):
    """Per-local-query causal interval [ks,ke) in absolute key coords from full-seq
    doc boundaries and global query positions. ks = query's doc start, ke = gpos+1."""
    cu = cu_full.long()
    p = gpos.long()
    d = (torch.searchsorted(cu, p, right=True) - 1).clamp(0, cu.numel() - 2)
    return cu[d], p + 1


def _sparse_forward(self, orig_forward, hidden_states, attention_mask, position_ids,
                    past_key_value, output_attentions, use_cache, **kwargs):
    if getattr(self, "cp_size", 1) > 1:
        return _sparse_forward_cp(self, hidden_states)
    B, L, _ = hidden_states.shape
    dev = hidden_states.device
    if position_ids is None:
        position_ids = torch.arange(L, device=dev).unsqueeze(0).expand(B, -1)

    # teacher in ABSORBED latent space (grad flows: backbone trainable in sparse).
    # We never materialize per-head K/V — the sparse ops gather the shared latent.
    q_lat, latent, W_UV, cos, sin = _compute_teacher_absorbed(self, hidden_states, position_ids)
    Qlat = q_lat.transpose(1, 2)                                  # [B,L,H,Dl]
    d_c = self.kv_lora_rank

    q_input = hidden_states
    if self.q_lora_rank is not None:
        q_input = self.q_a_layernorm(self.q_a_proj(hidden_states))
    c_idx, s_idx = _indexer_rope_cossin(self, cos, sin, position_ids)
    q_idx, k_idx, w_idx = self.indexer(hidden_states.detach(), q_input.detach(), c_idx, s_idx)

    cus = _cu_from_batch(B, L, dev, getattr(self, "cu_seqlens", None))
    topk = getattr(self, "dsa_topk", 2048)
    sm_tea = float(self.softmax_scale)
    q_chunk = getattr(self, "dsa_qchunk", 512)

    outs = []
    idx_losses = []
    ids0 = None
    for b in range(B):
        with torch.no_grad():
            ids = indexer_select_topk(q_idx[b].detach(), k_idx[b].detach(),
                                      w_idx[b].detach(), cus[b], topk,
                                      sm_scale_index=self.indexer.softmax_scale,
                                      q_chunk=q_chunk)
        if b == 0:
            ids0 = ids
        # sparse MLA in absorbed space: gather shared latent [q_chunk,K,Dl] once
        # (query-chunked + gradient-checkpointed) — ~9x less traffic than per-head.
        out_b = sparse_attend_absorbed_chunked(Qlat[b], latent[b], W_UV, ids,
                                               d_c=d_c, sm_scale=sm_tea,
                                               cu_seqlens=cus[b], q_chunk=q_chunk)  # [L,H,vhd]
        outs.append(out_b)
        # indexer KL on the selected set; teacher = absorbed per-head query vs the
        # shared latent (Km.dim()==2 branch inside), so it too gathers latent once.
        lb = sparse_kl_chunked(q_idx[b], k_idx[b], w_idx[b],
                               Qlat[b], latent[b], ids,
                               sm_scale_teacher=sm_tea,
                               sm_scale_index=self.indexer.softmax_scale,
                               q_chunk=q_chunk)
        idx_losses.append(lb)
    entropy = None
    if (getattr(self, "dsa_log_recall", False) or getattr(self, "dsa_log_kl", False)) and ids0 is not None:
        with torch.no_grad():
            entropy = _sparse_teacher_entropy(Qlat[0], latent[0], ids0, sm_tea,
                                              q_chunk=q_chunk)
    REGISTRY.add(torch.stack(idx_losses).mean(), entropy=entropy)

    attn_output = torch.stack(outs, 0)                            # [B,L,H,vhd]
    attn_output = attn_output.reshape(B, L, self.num_heads * self.v_head_dim)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def _sparse_forward_cp(self, hidden_states):
    """Context-parallel sparse attention for one layer. Backbone TRAINABLE, so the
    gathered latent + indexer key carry autograd (AllGatherSeq: fwd all-gather / bwd
    reduce-scatter → each rank's latent/indexer-k grads land on its own shard).

    This rank holds Lloc local queries; keys all-gathered to Lk = Lloc*cp. Top-k is
    selected over the full gathered indexer keys (indices are GLOBAL positions);
    sparse MLA attends the gathered latent; the indexer KL trains on the selected set.
    LM output is the local [Lloc,...] hidden for the next layer."""
    from ..dist import cp_size, cp_rank, zigzag_local_gpos, all_gather_seq
    B, Lloc, _ = hidden_states.shape
    assert B == 1, "CP sparse assumes batch size 1 (packed long-context)"
    dev = hidden_states.device
    n = cp_size()
    Lk = Lloc * n
    gpos = zigzag_local_gpos(Lk, dev).unsqueeze(0)  # [1,Lloc] global (zigzag order)

    # absorbed teacher (grad flows); rope table spans the full global sequence
    q_lat, latent_loc, W_UV, cos, sin = _compute_teacher_absorbed(
        self, hidden_states, gpos, rope_seq_len=Lk)
    Qlat = q_lat.transpose(1, 2)                                    # [1,Lloc,H,Dl]
    d_c = self.kv_lora_rank

    q_input = hidden_states
    if self.q_lora_rank is not None:
        q_input = self.q_a_layernorm(self.q_a_proj(hidden_states))
    c_idx, s_idx = _indexer_rope_cossin(self, cos, sin, gpos)
    q_idx, k_idx, w_idx = self.indexer(hidden_states.detach(), q_input.detach(), c_idx, s_idx)

    # all-gather keys to full sequence (autograd: latent + indexer-k are trainable)
    latent_full = all_gather_seq(latent_loc[0], grad=True)         # [Lk,Dl]
    k_idx_full = all_gather_seq(k_idx[0], grad=True)               # [Lk,df]

    cu_full = getattr(self, "cu_seqlens", None)
    cu_full = torch.tensor([0, Lk], device=dev) if cu_full is None else cu_full.to(dev)
    topk = getattr(self, "dsa_topk", 2048)
    sm_tea = float(self.softmax_scale)
    q_chunk = getattr(self, "dsa_qchunk", 512)

    # top-k over the FULL gathered indexer keys, causal by true global position ->
    # GLOBAL key indices in [0,Lk).
    with torch.no_grad():
        ids = indexer_select_topk(q_idx[0].detach(), k_idx_full.detach(),
                                  w_idx[0].detach(), cu_full, topk,
                                  sm_scale_index=self.indexer.softmax_scale,
                                  q_chunk=q_chunk, q_global_pos=gpos[0])
    # sparse MLA over gathered latent (global indices), local queries
    out = sparse_attend_absorbed_chunked(Qlat[0], latent_full, W_UV, ids, d_c=d_c,
                                         sm_scale=sm_tea, cu_seqlens=cu_full,
                                         q_chunk=q_chunk, q_global_pos=gpos[0])  # [Lloc,H,vhd]
    # indexer KL on the selected set (teacher = absorbed query vs gathered latent)
    lb = sparse_kl_chunked(q_idx[0], k_idx_full, w_idx[0], Qlat[0], latent_full, ids,
                           sm_scale_teacher=sm_tea,
                           sm_scale_index=self.indexer.softmax_scale, q_chunk=q_chunk,
                           nrow_global=Lk)
    entropy = None
    if getattr(self, "dsa_log_recall", False) or getattr(self, "dsa_log_kl", False):
        with torch.no_grad():
            entropy = _sparse_teacher_entropy(Qlat[0], latent_full, ids, sm_tea,
                                              q_chunk=q_chunk, nrow_global=Lk)
    REGISTRY.add(lb, entropy=entropy)

    out = out.reshape(B, Lloc, self.num_heads * self.v_head_dim)
    out = self.o_proj(out)
    return out, None, None
