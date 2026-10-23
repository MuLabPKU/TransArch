"""Window compressor that builds each layer's compressed K/V entries."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import Qwen3KVCConfig
from .grouped_linear import PositionGroupedLinear

UNIT_RMS_EPS = 1e-6


def _unit_rms(v: torch.Tensor, eps: float = UNIT_RMS_EPS) -> torch.Tensor:
    """Gain-free unit RMS along the last dim, computed in fp32 (fp64 kept)."""

    compute_dtype = torch.float64 if v.dtype == torch.float64 else torch.float32
    v_acc = v.to(dtype=compute_dtype)
    return v_acc * torch.rsqrt(v_acc.pow(2).mean(dim=-1, keepdim=True) + eps)


@dataclass
class CompressionOutput:
    """Closed compressed entries and their source-coordinate metadata.

    ``values`` is ``None`` under KV sharing, where the keys double as values.
    """

    keys: torch.Tensor
    values: torch.Tensor | None
    entry_positions: torch.Tensor
    closure_positions: torch.Tensor


class WindowCompressor(nn.Module):
    """Right-aligned causal K/V compressor with explicit startup masking.

    Entry ``i`` closes at token ``(i + 1) * S - 1`` and pools the ``W`` tokens
    ending there; positions before zero are masked out of the pool.
    """

    z_key: PositionGroupedLinear | None
    z_value: PositionGroupedLinear | None
    tau_k: nn.Parameter | None
    tau_v: nn.Parameter | None

    def __init__(self, config: Qwen3KVCConfig) -> None:
        super().__init__()
        self.config = config
        self.window = config.compression_window
        self.stride = config.compression_stride
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.projection_size = config.num_key_value_heads * config.head_dim
        self.tau_k = None
        self.tau_v = None
        if config.post_projection_entry_normalization:
            self.tau_k = nn.Parameter(torch.ones((), dtype=torch.float32))
            if not config.kv_sharing:
                self.tau_v = nn.Parameter(torch.ones((), dtype=torch.float32))

        self.c_key = self._make_c()
        self.c_value = None if config.kv_sharing else self._make_c()
        if config.compression_pooling == "adaptive_softmax":
            z_dim = (
                config.num_key_value_heads if config.gate_mode == "scalar" else self.projection_size
            )
            self.z_key = self._make_z(z_dim)
            self.z_value = None if config.kv_sharing else self._make_z(z_dim)
        else:
            self.z_key = None
            self.z_value = None

    def _make_c(self) -> PositionGroupedLinear:
        return PositionGroupedLinear(
            self.config.hidden_size,
            self.projection_size,
            self.config.c_group_ids,
            bias=self.config.c_bias,
            init_std=self.config.c_init_std,
            trainable=self.config.c_trainable,
        )

    def _make_z(self, output_size: int) -> PositionGroupedLinear:
        return PositionGroupedLinear(
            self.config.hidden_size,
            output_size,
            self.config.z_group_ids,
            bias=self.config.z_bias,
            init_std=self.config.z_init_std,
            trainable=self.config.z_trainable,
        )

    def make_windows(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather every closed window: ``[batch, entries, W, hidden]`` plus coordinates."""

        sequence = hidden_states.shape[1]
        device = hidden_states.device
        # Only closed entries exist; a trailing partial stride block forms none. Entry i closes
        # at (i + 1) * S - 1 and takes RoPE position i * S, the start of its newest stride block.
        entry_indices = torch.arange(sequence // self.stride, device=device)
        entry_positions = entry_indices * self.stride
        closure_positions = (entry_indices + 1) * self.stride - 1
        # Offset 0 is the oldest source token, W - 1 the closing one; for W = S, offset = t mod S.
        offsets = torch.arange(self.window, device=device)
        source_positions = closure_positions[:, None] - self.window + 1 + offsets[None, :]
        # [entries, W]; sources before position 0 occur only when W > S and are masked.
        valid_mask = source_positions >= 0
        safe = source_positions.clamp(0, max(sequence - 1, 0))
        windows = hidden_states[:, safe].masked_fill(~valid_mask[None, :, :, None], 0)
        return windows, valid_mask, entry_positions, closure_positions

    def _branch(
        self,
        windows: torch.Tensor,
        valid_mask: torch.Tensor,
        c_proj: PositionGroupedLinear,
        z_proj: PositionGroupedLinear | None,
        tau: nn.Parameter | None,
    ) -> torch.Tensor:
        """Pool one K or V branch to ``[batch, entries, kv_heads, head_dim]``."""

        batch, entries, window, _ = windows.shape
        # C^B_i payloads: [batch, entries, W, kv_heads, head_dim].
        payload = c_proj(windows).view(
            batch, entries, window, self.num_key_value_heads, self.head_dim
        )
        if tau is not None:  # unit RMS per head and offset, then a learned scalar gain
            payload = (tau * _unit_rms(payload)).to(dtype=payload.dtype)
        payload_float = payload.float()
        if z_proj is None:  # mean pooling over the valid window positions
            counts = valid_mask.sum(dim=1).clamp_min(1).float()
            gates = (valid_mask.float() / counts[:, None])[None, :, :, None, None]
            pooled = (payload_float * gates).sum(dim=2)
        else:
            # Gate scores from Z^B_i and b^B_i; softmax over each entry's W offsets in fp32.
            gate_channels = 1 if self.config.gate_mode == "scalar" else self.head_dim
            scores = (
                z_proj(windows)
                .view(batch, entries, window, self.num_key_value_heads, gate_channels)
                .float()
            )
            scores = scores.masked_fill(~valid_mask[None, :, :, None, None], float("-inf"))
            gates = torch.softmax(scores, dim=2)
            pooled = (payload_float * gates.expand_as(payload_float)).sum(dim=2)
        return pooled.to(payload.dtype)

    def forward(self, hidden_states: torch.Tensor) -> CompressionOutput:
        windows, valid, entries, closures = self.make_windows(hidden_states)
        keys = self._branch(windows, valid, self.c_key, self.z_key, self.tau_k)
        values = None
        if self.c_value is not None:
            values = self._branch(windows, valid, self.c_value, self.z_value, self.tau_v)
        return CompressionOutput(
            keys=keys, values=values, entry_positions=entries, closure_positions=closures
        )


__all__ = ["CompressionOutput", "WindowCompressor"]
