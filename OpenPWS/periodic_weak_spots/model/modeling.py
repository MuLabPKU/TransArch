"""Hugging Face causal-LM wrapper for Qwen3-KVC."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import torch
from transformers import Qwen3ForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from .attention import Qwen3KVCAttention
from .config import Qwen3KVCConfig


class Qwen3KVCForCausalLM(Qwen3ForCausalLM):
    """Hugging Face Qwen3 with every self-attention replaced by KVC attention."""

    config_class = Qwen3KVCConfig
    # KVC attention bypasses the HF attention interface; config.attention_backend picks its
    # kernel (FlexAttention in training, eager otherwise).
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_attention_backend = False

    def __init__(self, config: Qwen3KVCConfig) -> None:
        super().__init__(config)
        if 0 < config.rope_dims < config.head_dim:
            # Build the frequency ladder at dimension rope_dims rather than
            # cropping a full-head table.
            rotary_config = deepcopy(config)
            rotary_config.head_dim = config.rope_dims
            self.model.rotary_emb = Qwen3RotaryEmbedding(rotary_config)
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = Qwen3KVCAttention(config, layer_idx)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        if past_key_values is not None or use_cache is True:
            raise ValueError("Qwen3-KVC does not support KV caching")
        # KVC masks are built from positions inside attention, so only all-ones masks pass.
        if attention_mask is not None and not bool(torch.all(attention_mask == 1)):
            raise ValueError("Qwen3-KVC does not support padding or packed masks")
        return cast(
            CausalLMOutputWithPast,
            super().forward(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=False,
                logits_to_keep=logits_to_keep,
                **kwargs,
            ),
        )

    def prepare_flex_attention(self, sequence_length: int) -> None:
        """Build fixed-shape masks before Trainer applies torch.compile."""

        device = next(self.parameters()).device
        for layer in self.model.layers:
            cast(Qwen3KVCAttention, layer.self_attn).prepare_flex_attention(sequence_length, device)

    @torch.no_grad()
    def last_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Eager next-token logits at the final position, as used for scoring."""

        was_training = self.training
        self.eval()  # FlexAttention only runs in training mode
        try:
            return self(input_ids=input_ids, logits_to_keep=1).logits[:, -1]
        finally:
            self.train(was_training)


__all__ = ["Qwen3KVCForCausalLM"]
