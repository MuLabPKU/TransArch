"""KVC attention: one softmax over closed compressed entries and raw local tokens."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb

from .compressor import CompressionOutput, WindowCompressor
from .config import Qwen3KVCConfig
from .flex_attention import create_full_attn_block_mask, create_kvc_block_mask


def apply_partial_rotary(
    tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dims: int
) -> torch.Tensor:
    """Rotate the last ``rope_dims`` of ``tensor``; leave the prefix unchanged.

    ``cos`` and ``sin`` come from a ``rope_dims``-dimensional rotary table.
    Full-head RoPE is identical to Qwen3; ``rope_dims=0`` is NoPE.
    """

    if rope_dims == 0:
        return tensor
    if rope_dims == tensor.shape[-1]:
        return apply_rotary_pos_emb(tensor, tensor, cos, sin)[0]
    tail = tensor[..., -rope_dims:]
    rotated = apply_rotary_pos_emb(tail, tail, cos, sin)[0]
    return torch.cat((tensor[..., :-rope_dims], rotated), dim=-1)


def repeat_kv(states: torch.Tensor, num_query_groups: int) -> torch.Tensor:
    """Repeat native KV heads to query heads without mixing head order."""

    return states.repeat_interleave(num_query_groups, dim=1)


def _flex_dtype(query: torch.Tensor) -> torch.dtype:
    # flex_attention needs one q/k/v dtype; follow CUDA autocast when it is active.
    return torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else query.dtype


class Qwen3KVCAttention(Qwen3Attention):
    """Qwen3 attention over compressed entries plus one exact local-memory policy.

    ``fixed_swa`` attends to every causally closed compressed entry plus the
    last ``swa_window_size`` raw tokens (overlap is intentional). ``tail_concat``
    attends to closed entries plus the raw tokens of the open stride block.
    ``full`` has no compressor and is ordinary causal attention.
    """

    def __init__(self, config: Qwen3KVCConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        if not config.qk_norm:
            self.q_norm = torch.nn.Identity()  # type: ignore[assignment]
            self.k_norm = torch.nn.Identity()  # type: ignore[assignment]
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.groups = config.query_groups_per_kv_head
        self._flex_block_mask: BlockMask | None = None
        self._flex_mask_signature: tuple[int, torch.device] | None = None
        if config.local_attention_mode == "full":
            return
        self.compressor = WindowCompressor(config)
        if config.kv_sharing:  # values reuse the normalized, rotated keys
            del self.v_proj

    def _queries(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, sequence, self.num_heads, self.head_dim)
        query = self.q_norm(query).transpose(1, 2)
        cos, sin = position_embeddings
        return apply_partial_rotary(query, cos, sin, self.config.rope_dims)

    def _compressed_kv(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, CompressionOutput]:
        """Entry keys and values as ``[batch, kv_heads, entries, head_dim]``."""

        compressed = self.compressor(hidden_states)
        # Entries get the raw keys' k_norm and RoPE, at their entry positions i * S.
        keys = self.k_norm(compressed.keys).transpose(1, 2)
        cos, sin = position_embeddings
        entry_cos = cos[:, compressed.entry_positions]
        entry_sin = sin[:, compressed.entry_positions]
        keys = apply_partial_rotary(keys, entry_cos, entry_sin, self.config.rope_dims)
        values = keys if compressed.values is None else compressed.values.transpose(1, 2)
        return keys, values, compressed

    def _tail_kv(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw per-token keys and values as ``[batch, kv_heads, sequence, head_dim]``."""

        batch, sequence, _ = hidden_states.shape
        raw_keys = self.k_proj(hidden_states).view(
            batch, sequence, self.num_kv_heads, self.head_dim
        )
        keys = self.k_norm(raw_keys).transpose(1, 2)
        cos, sin = position_embeddings
        keys = apply_partial_rotary(keys, cos, sin, self.config.rope_dims)
        if self.config.kv_sharing:
            # Shared KV uses one normalized, rotated vector as both K and V; the
            # output is inverse-rotated at each query in _restore_shared_output.
            return keys, keys
        values = self.v_proj(hidden_states).view(batch, sequence, self.num_kv_heads, self.head_dim)
        return keys, values.transpose(1, 2)

    def _use_flex_attention(self) -> bool:
        """Flex is the compiled training kernel for one static sequence length."""

        return self.config.attention_backend == "flex_attention" and self.training

    def prepare_flex_attention(self, sequence_length: int, device: torch.device) -> None:
        """Prepare the static block mask before Trainer compiles the model."""

        if self.config.attention_backend != "flex_attention":
            return
        if self.config.local_attention_mode == "full":
            self._flex_block_mask = create_full_attn_block_mask(
                sequence_length=sequence_length, device=device
            )
        else:
            self._flex_block_mask = create_kvc_block_mask(
                sequence_length=sequence_length,
                compressed_length=sequence_length // self.config.compression_stride,
                stride=self.config.compression_stride,
                local_attention_mode=self.config.local_attention_mode,
                swa_window_size=self.config.swa_window_size,
                device=device,
            )
        self._flex_mask_signature = (sequence_length, device)

    def _flex(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        # Keys and values are laid out as [entries ; raw tokens], matching make_kvc_mask_mod.
        signature = (query.shape[-2], query.device)
        if self._flex_mask_signature != signature:
            self.prepare_flex_attention(*signature)
        dtype = _flex_dtype(query)
        output = flex_attention(
            query.to(dtype),
            keys.to(dtype),
            values.to(dtype),
            block_mask=self._flex_block_mask,
            scale=self.scaling,
            enable_gqa=self.groups > 1,
        )
        return cast(torch.Tensor, output)

    def _restore_shared_output(
        self, output: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Undo query-position RoPE after attending to shared rotated K=V."""

        if not self.config.kv_sharing:
            return output
        cos, sin = position_embeddings
        return apply_partial_rotary(output, cos, -sin, self.config.rope_dims)

    def _split_probabilities(
        self,
        query: torch.Tensor,
        compressed_keys: torch.Tensor,
        tail_keys: torch.Tensor,
        compressed: CompressionOutput,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One softmax over [visible compressed entries ; visible raw local tokens]."""

        # Inputs use query heads ([b, h, q, d]). Returns probabilities [b, h, q, entries + local]
        # and tail_sources [q, local], the raw position of each local column.
        batch, heads, sequence, dim = query.shape
        stride = self.config.compression_stride
        compressed_scores = torch.matmul(query.float(), compressed_keys.transpose(-1, -2).float())
        compressed_scores = compressed_scores * self.scaling
        # Every causally closed entry is available immediately.
        visible = compressed.closure_positions <= positions.unsqueeze(-1)
        compressed_scores = compressed_scores.masked_fill(~visible[None, None], float("-inf"))
        if self.config.local_attention_mode == "fixed_swa":
            # The swa_window_size raw tokens ending at the query: keys [b, h, q, window, d].
            window = self.config.swa_window_size
            offsets = torch.arange(window, device=query.device)
            tail_sources = positions[:, None] - window + 1 + offsets[None]
            valid = tail_sources >= 0
            local_keys = tail_keys[:, :, tail_sources.clamp(0, sequence - 1), :]
            tail_scores = torch.einsum("bhqd,bhqwd->bhqw", query.float(), local_keys.float())
            tail_scores = (tail_scores * self.scaling).masked_fill(
                ~valid[None, None], float("-inf")
            )
        else:
            # tail_concat: causal attention inside the query's open stride block;
            # a block's last token closes it, so that query sees no raw tokens. Scores are
            # computed per block as [b, h, blocks, S, S] and flattened to [b, h, q, S].
            blocks = (sequence + stride - 1) // stride
            padded = blocks * stride
            padding = padded - sequence
            query_blocks = F.pad(query, (0, 0, 0, padding)).view(batch, heads, blocks, stride, dim)
            key_blocks = F.pad(tail_keys, (0, 0, 0, padding)).view(
                batch, heads, blocks, stride, dim
            )
            tail_scores = (
                torch.matmul(query_blocks.float(), key_blocks.transpose(-1, -2).float())
                * self.scaling
            )
            offsets = torch.arange(stride, device=query.device)
            block_positions = (
                torch.arange(blocks, device=query.device)[:, None] * stride + offsets[None]
            )
            valid = block_positions < sequence
            causal = offsets[None, :] <= offsets[:, None]
            open_query = offsets != stride - 1
            visible = (
                causal[None] & open_query[None, :, None] & valid[:, :, None] & valid[:, None, :]
            )
            tail_scores = tail_scores.masked_fill(~visible[None, None], float("-inf"))
            tail_scores = tail_scores.reshape(batch, heads, padded, stride)[:, :, :sequence]
            tail_sources = (
                torch.div(positions, stride, rounding_mode="floor")[:, None] * stride
                + offsets[None]
            )
        # No row is all -inf: each query sees its own raw token or the entry it closes.
        probabilities = torch.softmax(torch.cat((compressed_scores, tail_scores), dim=-1), dim=-1)
        return probabilities, tail_sources

    def _output_from_probabilities(
        self,
        probabilities: torch.Tensor,
        compressed_values: torch.Tensor,
        tail_values: torch.Tensor,
        tail_sources: torch.Tensor,
    ) -> torch.Tensor:
        """Apply split probabilities to compressed values and the matching raw local values."""

        sequence = probabilities.shape[-2]
        stride = self.config.compression_stride
        compressed_length = compressed_values.shape[-2]
        compressed_probability = probabilities[..., :compressed_length]
        tail_probability = probabilities[..., compressed_length:]
        output = torch.matmul(compressed_probability.to(compressed_values.dtype), compressed_values)
        if self.config.local_attention_mode == "fixed_swa":
            local_values = tail_values[:, :, tail_sources.clamp(0, sequence - 1), :]
            output = output + torch.einsum(
                "bhqw,bhqwd->bhqd", tail_probability.to(local_values.dtype), local_values
            )
        else:
            batch, heads, _, dim = tail_values.shape
            blocks = (sequence + stride - 1) // stride
            padded = blocks * stride
            padding = padded - sequence
            value_blocks = F.pad(tail_values, (0, 0, 0, padding)).view(
                batch, heads, blocks, stride, dim
            )
            probability_blocks = F.pad(tail_probability, (0, 0, 0, padding)).view(
                batch, heads, blocks, stride, stride
            )
            tail_output = torch.matmul(
                probability_blocks.to(value_blocks.dtype), value_blocks
            ).reshape(batch, heads, padded, dim)[:, :, :sequence]
            output = output + tail_output
        return output

    def _full_attention(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Ordinary causal attention for ``local_attention_mode: full``."""

        keys, values = self._tail_kv(hidden_states, position_embeddings)
        if self._use_flex_attention():
            return self._flex(query, keys, values)
        keys = repeat_kv(keys, self.groups)
        values = repeat_kv(values, self.groups)
        sequence = keys.shape[-2]
        scores = torch.matmul(query.float(), keys.transpose(-1, -2).float()) * self.scaling
        index = torch.arange(sequence, device=query.device)
        mask = (index[None, :] <= index[:, None])[None, None]
        probabilities = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        attention = F.dropout(
            probabilities.to(query.dtype), p=self.config.attention_dropout, training=self.training
        )
        return torch.matmul(attention.to(values.dtype), values)

    def _compressed_attention(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        query: torch.Tensor,
    ) -> torch.Tensor:
        compressed_keys, compressed_values, compressed = self._compressed_kv(
            hidden_states, position_embeddings
        )
        tail_keys, tail_values = self._tail_kv(hidden_states, position_embeddings)
        if self._use_flex_attention():
            output = self._flex(
                query,
                torch.cat((compressed_keys, tail_keys), dim=-2),
                torch.cat((compressed_values, tail_values), dim=-2),
            )
        else:
            positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
            probabilities, tail_sources = self._split_probabilities(
                query,
                repeat_kv(compressed_keys, self.groups),
                repeat_kv(tail_keys, self.groups),
                compressed,
                positions,
            )
            attention = F.dropout(
                probabilities.to(query.dtype),
                p=self.config.attention_dropout,
                training=self.training,
            )
            output = self._output_from_probabilities(
                attention,
                repeat_kv(compressed_values, self.groups),
                repeat_kv(tail_values, self.groups),
                tail_sources,
            )
        return self._restore_shared_output(output, position_embeddings)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        if past_key_values is not None or kwargs.get("use_cache") is True:
            raise ValueError("Qwen3-KVC does not support KV caching")
        batch, sequence, _ = hidden_states.shape
        query = self._queries(hidden_states, position_embeddings)
        if self.config.local_attention_mode == "full":
            output = self._full_attention(hidden_states, position_embeddings, query)
        else:
            output = self._compressed_attention(hidden_states, position_embeddings, query)
        # Head-major o_proj input [batch, sequence, heads * head_dim], as the knockouts assume.
        output = output.transpose(1, 2).reshape(
            batch, sequence, self.config.num_attention_heads * self.config.head_dim
        )
        return self.o_proj(output), None


__all__ = ["Qwen3KVCAttention", "apply_partial_rotary", "repeat_kv"]
