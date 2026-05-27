# SPDX-License-Identifier: Apache-2.0
"""vLLM model for the GQLA-converted GLM-4.7-Flash checkpoint.

Identical to ``vllm.model_executor.models.glm4_moe_lite`` except the attention
class: ``kv_b_proj`` is sized for ``num_key_value_heads`` and K/V are reshaped
with ``num_local_kv_heads`` so the KV cache is GQA-sized. Tensor parallel comes
from the ColumnParallel/RowParallel linear layers; the world size is read at
``__init__`` time via ``get_tensor_model_parallel_world_size``.
"""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.models.deepseek_v2 import (
    DeepSeekV2FusedQkvAProjLinear,
    yarn_get_mscale,
)
from vllm.model_executor.models.glm4_moe_lite import (
    Glm4MoeLite,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteMLP,
    Glm4MoeLiteModel,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig


class Glm4MoeLiteGQLAAttention(nn.Module):
    """GLM-4.7-Flash attention with GQA-shaped KV. Requires q LoRA (always true on GLM-4.7)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: "Glm4MoeLiteConfig",
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        assert num_kv_heads % tp_size == 0 or tp_size % num_kv_heads == 0, (
            f"num_kv_heads={num_kv_heads} incompatible with tp_size={tp_size}"
        )
        self.num_local_heads = num_heads // tp_size
        self.num_local_kv_heads = max(1, num_kv_heads // tp_size)
        self.scaling = self.qk_head_dim ** -0.5
        self.max_position_embeddings = max_position_embeddings

        self.q_a_proj = ReplicatedLinear(
            hidden_size, q_lora_rank, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank, num_heads * self.qk_head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.q_b_proj",
        )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
        # GQLA delta: kv_b_proj is sized for num_kv_heads (matches converted checkpoint).
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank, num_kv_heads * (qk_nope_head_dim + v_head_dim), bias=False,
            quant_config=quant_config, prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim, hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.o_proj",
        )

        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        if config.rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            m = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * m * m

        self.attn = Attention(
            self.num_local_heads, self.qk_head_dim, self.scaling,
            num_kv_heads=self.num_local_kv_heads,
            cache_config=cache_config, quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q_a_proj(hidden_states)[0]
        q = self.q_a_layernorm(q)
        q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)[0]
        # GQLA delta: reshape with num_local_kv_heads, not num_local_heads.
        kv = kv.view(-1, self.num_local_kv_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = latent_cache[:, :, self.kv_lora_rank:]                # (T, 1, qk_rope)

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim:] = q_pe

        # k_pe broadcasts across the kv-head dim.
        k = k_nope.new_empty((k_nope.shape[0], self.num_local_kv_heads, self.qk_head_dim))
        k[..., :self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        # Pad V to qk_head_dim (no-op for GLM-4.7-Flash where qk_head_dim == v_head_dim).
        v = torch.nn.functional.pad(
            v, [0, self.qk_head_dim - self.v_head_dim], value=0
        ).view(-1, self.num_local_kv_heads * self.qk_head_dim)
        attn_output = self.attn(q, k, v)
        attn_output = attn_output.view(-1, self.num_local_heads, self.qk_head_dim)[
            ..., :self.v_head_dim
        ].reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output


class Glm4MoeLiteGQLADecoderLayer(Glm4MoeLiteDecoderLayer):
    """Swap in GQLA attention; reuse parent forward + MoE/MLP wiring."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: "Glm4MoeLiteConfig | None" = None,
    ) -> None:
        nn.Module.__init__(self)
        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        layer_idx = int(prefix.split(".")[-1])
        self.layer_idx = layer_idx

        self.self_attn = Glm4MoeLiteGQLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        ):
            self.mlp = Glm4MoeLite(
                config=config, quant_config=quant_config, prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm4MoeLiteMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Glm4MoeLiteGQLAModel(nn.Module):
    """Standalone twin of Glm4MoeLiteModel.

    Cannot subclass it because the parent is @support_torch_compile-decorated;
    subclassing would break the compiled dispatch. Weight loading and expert
    mapping are delegated to the parent's unbound methods (~200 lines reused).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type
        self.vocab_size = config.vocab_size

        assert not hasattr(config, "index_topk"), "DeepSeek-V3.2 indexer not supported"

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size,
                quant_config=quant_config, prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Glm4MoeLiteGQLADecoderLayer(
                vllm_config=vllm_config, config=config, prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    # Reuse upstream weight loader / expert mapping — they only depend on
    # named_parameters() + config, which are layout-compatible.
    load_weights = Glm4MoeLiteModel.load_weights
    get_expert_mapping = Glm4MoeLiteModel.get_expert_mapping


class Glm4MoeLiteGQLAForCausalLM(Glm4MoeLiteForCausalLM):
    """Top-level causal LM. Same as parent except ``self.model``."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        # GQLA always uses MLA-shaped projections (separate q_a_proj / kv_a_proj).
        self.use_mha = False
        self.fuse_qkv_a_proj = False
        self.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

        self.model = Glm4MoeLiteGQLAModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size,
                quant_config=quant_config, prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self.num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace
        # Provided by the Glm4LiteMixtureOfExperts mixin via the parent's MRO.
        self.set_moe_parameters()


def _expand_gqla_kv_b_weight(
    weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    per_head_kv: int,
) -> torch.Tensor:
    """Expand a GQLA-packed kv_b_proj weight to MLA layout.

    Source: ``(num_kv_heads * per_head_kv, kv_lora_rank)`` — one (W_UK|W_UV)
    block per KV group (rows already laid out nope-then-v inside each block by
    ``compression.compress_and_absorb``).

    Target: ``(num_heads * per_head_kv, kv_lora_rank)`` — block repeated
    ``group_size`` times so the standard MLA absorb path treats each of the
    ``gs`` heads in a group as having its own (identical) W_UK / W_UV. KV cache
    still stores only the latent ``(kv_lora_rank + qk_rope)``.
    """
    G = num_kv_heads
    H = num_heads
    if H % G != 0:
        raise ValueError(f"num_heads={H} not divisible by num_kv_heads={G}")
    gs = H // G
    if weight.shape[0] != G * per_head_kv:
        raise ValueError(
            f"GQLA kv_b_proj weight has rows={weight.shape[0]}, expected "
            f"{G}*{per_head_kv}={G * per_head_kv}"
        )
    return (
        weight.view(G, per_head_kv, weight.shape[-1])
        .repeat_interleave(gs, dim=0)
        .reshape(H * per_head_kv, weight.shape[-1])
        .contiguous()
    )


class Glm4MoeLiteGQLAAbsorbAttention(nn.Module):
    """GLM-4.7-Flash attention serving a GQLA checkpoint via MLA absorb.

    Same shapes as vanilla MLA — ``kv_b_proj`` is sized for ``num_heads`` and
    weights are absorbed into Q (W_UK) / O (W_UV) at runtime. The GQLA
    checkpoint's per-group ``kv_b_proj`` is expanded to MLA layout in the model
    weight loader (see :func:`_expand_gqla_kv_b_weight`); Q and O were already
    absorbed by the conversion. KV cache stores only the latent
    ``kv_lora_rank + qk_rope_head_dim`` (MQA-sized).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: "Glm4MoeLiteConfig",
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads

        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim ** -0.5
        self.max_position_embeddings = max_position_embeddings

        self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            hidden_size,
            [q_lora_rank, kv_lora_rank + qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank, num_heads * self.qk_head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.q_b_proj",
        )

        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
        # MLA shape — full num_heads. The GQLA checkpoint's smaller kv_b_proj
        # is expanded at load time by the model's weight loader.
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False,
            quant_config=quant_config, prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim, hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.o_proj",
        )

        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        if config.rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            m = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * m * m

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=None,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=None,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states, llama_4_scaling)


class Glm4MoeLiteGQLAAbsorbDecoderLayer(Glm4MoeLiteDecoderLayer):
    """Swap in GQLA-absorb attention; reuse parent forward + MoE/MLP wiring."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: "Glm4MoeLiteConfig | None" = None,
    ) -> None:
        nn.Module.__init__(self)
        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        layer_idx = int(prefix.split(".")[-1])
        self.layer_idx = layer_idx

        self.self_attn = Glm4MoeLiteGQLAAbsorbAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        ):
            self.mlp = Glm4MoeLite(
                config=config, quant_config=quant_config, prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm4MoeLiteMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Glm4MoeLiteGQLAAbsorbModel(nn.Module):
    """Standalone twin of Glm4MoeLiteModel for the GQLA-absorb path.

    Cannot subclass Glm4MoeLiteModel directly (it is @support_torch_compile-
    decorated). Weight loading delegates to the upstream loader but first
    expands per-group kv_b_proj rows to MLA layout (see
    :func:`_expand_gqla_kv_b_weight`).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type
        self.vocab_size = config.vocab_size

        assert not hasattr(config, "index_topk"), "DeepSeek-V3.2 indexer not supported"

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size,
                quant_config=quant_config, prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Glm4MoeLiteGQLAAbsorbDecoderLayer(
                vllm_config=vllm_config, config=config, prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    get_expert_mapping = Glm4MoeLiteModel.get_expert_mapping

    def load_weights(self, weights):
        config = self.config
        H = config.num_attention_heads
        G = config.num_key_value_heads
        per_head_kv = config.qk_nope_head_dim + config.v_head_dim
        kv_lora = config.kv_lora_rank

        def _expanded(it):
            for name, w in it:
                if name.endswith("kv_b_proj.weight") and w.shape[0] == G * per_head_kv:
                    w = _expand_gqla_kv_b_weight(w, H, G, per_head_kv)
                yield name, w

        return Glm4MoeLiteModel.load_weights(self, _expanded(weights))


class Glm4MoeLiteGQLAAbsorbForCausalLM(Glm4MoeLiteForCausalLM):
    """Top-level causal LM serving a GQLA checkpoint via MLA absorb."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        # MLA absorb path: fuse q_a_proj + kv_a_proj_with_mqa from the checkpoint
        # into fused_qkv_a_proj at load time (handled by upstream stacked mapping).
        self.use_mha = False
        self.fuse_qkv_a_proj = True
        self.packed_modules_mapping = {
            "gate_up_proj": ["gate_proj", "up_proj"],
            "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        }

        self.model = Glm4MoeLiteGQLAAbsorbModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size,
                quant_config=quant_config, prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self.num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace
        self.set_moe_parameters()


__all__ = [
    "Glm4MoeLiteGQLAAttention",
    "Glm4MoeLiteGQLADecoderLayer",
    "Glm4MoeLiteGQLAModel",
    "Glm4MoeLiteGQLAForCausalLM",
    "Glm4MoeLiteGQLAAbsorbAttention",
    "Glm4MoeLiteGQLAAbsorbDecoderLayer",
    "Glm4MoeLiteGQLAAbsorbModel",
    "Glm4MoeLiteGQLAAbsorbForCausalLM",
]
