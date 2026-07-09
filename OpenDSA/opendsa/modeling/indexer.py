"""indexer.py — the DSA Lightning Indexer for DeepSeek-V2 MLA.

Produces, per token, the ingredients of the indexer score
    I[t,j] = ( Σ_h w[t,h] · ReLU(<q_idx[t,h], k_idx[j]>) ) · softmax_scale
which is distilled to match the model's own (head-averaged) MLA attention during
warmup, and then drives top-k key selection during sparse training/inference.

Design mirrors the reference (agi-megatron-lm .../dsa.py::DSAIndexer):
  * q projection: q_input -> H_I * d_I   (q_input = q_lora latent if present, else
    hidden_states — DeepSeek-V2-Lite has q_lora_rank=None so it reads hidden_states)
  * k projection: hidden_size -> d_I      (single head, shared across indexer heads)
  * k LayerNorm (non-RMS) over d_I
  * per-head weights: hidden_size -> H_I
  * RoPE applied to the last ``rope_dim`` dims of q_idx and k_idx (rope_dim =
    qk_rope_head_dim = 64), using the model's cos/sin so positions match the teacher.

Shapes returned (per batch): q_idx [B,L,H_I,d_I], k_idx [B,L,d_I], w [B,L,H_I].
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, cos, sin):
    """x: [B, L, n, rope_dim]; cos/sin: [B, L, rope_dim]. DeepSeek rotate_half."""
    cos = cos.unsqueeze(2)  # [B,L,1,rope_dim]
    sin = sin.unsqueeze(2)
    return x * cos + _rotate_half(x) * sin


class LightningIndexer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 8,
        head_dim: int = 128,
        rope_dim: int = 64,
        q_input_dim: int | None = None,   # None -> use hidden_size (q_lora_rank=None)
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.nope_dim = head_dim - rope_dim
        assert self.nope_dim >= 0, "head_dim must be >= rope_dim"
        q_in = q_input_dim if q_input_dim is not None else hidden_size
        self.q_input_dim = q_in
        self.softmax_scale = head_dim ** -0.5

        self.wq = nn.Linear(q_in, n_heads * head_dim, bias=False, dtype=dtype)
        self.wk = nn.Linear(hidden_size, head_dim, bias=False, dtype=dtype)
        self.k_norm = nn.LayerNorm(head_dim, dtype=dtype)
        self.w_proj = nn.Linear(hidden_size, n_heads, bias=False, dtype=dtype)

    def forward(self, hidden_states, q_input, cos, sin):
        """hidden_states [B,L,hidden]; q_input [B,L,q_in]; cos/sin [B,L,rope_dim].
        Returns q_idx [B,L,H_I,d_I], k_idx [B,L,d_I], w [B,L,H_I]."""
        B, L, _ = hidden_states.shape
        H, d, rd = self.n_heads, self.head_dim, self.rope_dim

        q = self.wq(q_input).view(B, L, H, d)                      # [B,L,H,d]
        k = self.k_norm(self.wk(hidden_states)).view(B, L, 1, d)   # [B,L,1,d]

        if rd > 0:
            q_nope, q_pe = q[..., : d - rd], q[..., d - rd:]
            k_nope, k_pe = k[..., : d - rd], k[..., d - rd:]
            q_pe = _apply_rope(q_pe, cos, sin)
            k_pe = _apply_rope(k_pe, cos, sin)
            q = torch.cat([q_nope, q_pe], dim=-1)
            k = torch.cat([k_nope, k_pe], dim=-1)

        k = k.squeeze(2)                                           # [B,L,d]
        w = self.w_proj(hidden_states) * (self.n_heads ** -0.5)    # [B,L,H]
        return q, k, w
