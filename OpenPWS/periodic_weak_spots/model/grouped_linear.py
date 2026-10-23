"""Per-window-offset linear maps whose weights can be tied across offsets."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class PositionGroupedLinear(nn.Module):
    """Logical per-offset projections backed by compact shared weight banks.

    The forward path applies each physical matrix only to offsets assigned to
    that bank. It never creates a ``[window, in_features, out_features]``
    expanded weight tensor.
    """

    group_ids: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_ids: Sequence[int],
        *,
        bias: bool,
        init_std: float,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        ids = tuple(group_ids)  # contiguous bank IDs 0..G-1, one per window offset
        self.window = len(ids)
        self.num_groups = len(set(ids))
        # _group_offsets[g] lists the offsets served by bank g; _inverse_group_order maps each
        # offset to its row in forward()'s bank-ordered concatenation.
        self._group_offsets = tuple(
            tuple(offset for offset, group_id in enumerate(ids) if group_id == group)
            for group in range(self.num_groups)
        )
        grouped_order = tuple(offset for offsets in self._group_offsets for offset in offsets)
        self._inverse_group_order = tuple(
            grouped_order.index(offset) for offset in range(self.window)
        )
        self.register_buffer("group_ids", torch.tensor(ids, dtype=torch.long), persistent=True)
        self.weight_bank = nn.Parameter(
            torch.empty(self.num_groups, in_features, out_features),
            requires_grad=trainable,
        )
        # Biases are per offset even when weights are tied (for gates, the b^B_i).
        self.bias = (
            nn.Parameter(torch.zeros(self.window, out_features), requires_grad=trainable)
            if bias
            else None
        )
        nn.init.normal_(self.weight_bank, mean=0.0, std=init_std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map ``[..., window, in_features]`` to ``[..., window, out_features]``."""

        # One GEMM per physical bank preserves compact tied weights while
        # avoiding one Python-launched GEMM per logical offset.
        grouped = [
            torch.matmul(
                inputs[..., list(offsets), :],
                self.weight_bank[group_id],
            )
            for group_id, offsets in enumerate(self._group_offsets)
        ]
        result = torch.cat(grouped, dim=-2)[..., list(self._inverse_group_order), :]
        if self.bias is not None:
            result = result + self.bias
        return result


__all__ = ["PositionGroupedLinear"]
