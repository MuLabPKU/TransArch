"""FlexAttention block masks that reproduce the eager KVC visibility rules."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

MaskMod = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def make_kvc_mask_mod(
    *,
    compressed_length: int,
    stride: int,
    local_attention_mode: str,
    swa_window_size: int | None,
) -> MaskMod:
    """Exact KVC visibility over ``[compressed entries ; raw tokens]`` key coordinates."""

    def mask_mod(
        batch: torch.Tensor,
        head: torch.Tensor,
        query_index: torch.Tensor,
        kv_index: torch.Tensor,
    ) -> torch.Tensor:
        del batch, head
        # Entry i closes at token (i + 1) * stride - 1.
        compressed_visible = (kv_index + 1) * stride - 1 <= query_index
        raw_index = kv_index - compressed_length
        if local_attention_mode == "fixed_swa":
            # The swa_window_size raw tokens ending at the query, inclusive.
            raw_visible = (raw_index >= query_index - swa_window_size + 1) & (
                raw_index <= query_index
            )
        else:
            # tail_concat: raw tokens of the query's still-open stride block.
            raw_visible = (
                (query_index.remainder(stride) != stride - 1)
                & (
                    raw_index.div(stride, rounding_mode="floor")
                    == query_index.div(stride, rounding_mode="floor")
                )
                & (raw_index <= query_index)
            )
        return torch.where(kv_index < compressed_length, compressed_visible, raw_visible)

    return mask_mod


def _causal_mask_mod(
    batch: torch.Tensor, head: torch.Tensor, query_index: torch.Tensor, kv_index: torch.Tensor
) -> torch.Tensor:
    del batch, head
    return kv_index <= query_index


# Cached so that layers with the same geometry share one mask.
@lru_cache(maxsize=32)
def create_kvc_block_mask(
    *,
    sequence_length: int,
    compressed_length: int,
    stride: int,
    local_attention_mode: str,
    swa_window_size: int | None,
    device: torch.device,
) -> BlockMask:
    """One batch/head-broadcast block mask for a fixed training shape."""

    mask_mod = make_kvc_mask_mod(
        compressed_length=compressed_length,
        stride=stride,
        local_attention_mode=local_attention_mode,
        swa_window_size=swa_window_size,
    )
    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=compressed_length + sequence_length,
        device=device,
        _compile=True,
    )


@lru_cache(maxsize=32)
def create_full_attn_block_mask(*, sequence_length: int, device: torch.device) -> BlockMask:
    """One batch/head-broadcast causal mask for a fixed training shape."""

    return create_block_mask(
        _causal_mask_mod,
        B=None,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=device,
        _compile=True,
    )


__all__ = ["create_full_attn_block_mask", "create_kvc_block_mask", "make_kvc_mask_mod"]
