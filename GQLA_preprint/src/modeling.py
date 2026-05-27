"""GQLA variant of Glm4MoeLite for GLM-4.7-Flash.

The kv_a chain is retained verbatim; only ``kv_b_proj`` is shrunk to size
``num_key_value_heads`` instead of ``num_attention_heads``. The upstream
attention forward already supports the GQA broadcast path through
``num_key_value_groups``.
"""

from __future__ import annotations

import torch.nn as nn

from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteModel,
)


class Glm4MoeLiteGQLAAttention(Glm4MoeLiteAttention):
    def __init__(self, config: Glm4MoeLiteConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            config.num_key_value_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )


class Glm4MoeLiteGQLADecoderLayer(Glm4MoeLiteDecoderLayer):
    def __init__(self, config: Glm4MoeLiteConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Glm4MoeLiteGQLAAttention(config, layer_idx)


class Glm4MoeLiteGQLAModel(Glm4MoeLiteModel):
    def __init__(self, config: Glm4MoeLiteConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Glm4MoeLiteGQLADecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.post_init()


class Glm4MoeLiteGQLAForCausalLM(Glm4MoeLiteForCausalLM):
    def __init__(self, config: Glm4MoeLiteConfig):
        super().__init__(config)
        self.model = Glm4MoeLiteGQLAModel(config)
        self.post_init()


__all__ = [
    "Glm4MoeLiteGQLAAttention",
    "Glm4MoeLiteGQLADecoderLayer",
    "Glm4MoeLiteGQLAModel",
    "Glm4MoeLiteGQLAForCausalLM",
]
