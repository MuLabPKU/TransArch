# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only DeepseekV2/DeepseekV3 model."""
import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (CacheConfig, VllmConfig)
from vllm.distributed import (get_ep_group, get_pp_group,
                              get_tensor_model_parallel_world_size)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.deepseek_v2 import (DeepseekV2ForCausalLM,
                                                    DeepSeekV2FusedQkvAProj,
                                                    DeepseekV32IndexerCache,
                                                    DeepseekV2MoE,
                                                    DeepseekAttention,
                                                    _get_llama_4_scaling,
                                                    Indexer)
from vllm.platforms import current_platform
from vllm.model_executor.models.utils import (PPMissingLayer,
                    make_empty_intermediate_tensors_factory, make_layers)
from .indexers import BaselineSparseAttnIndexer, MiSASparseAttnIndexer

logger = init_logger(__name__)


class DeepseekV2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           reduce_results=reduce_results,
                                           disable_tp=is_sequence_parallel,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class MiSAIndexer(nn.Module):

    def __init__(self,
                 vllm_config: VllmConfig,
                 config: Union[DeepseekV2Config, DeepseekV3Config],
                 hidden_size: int,
                 q_lora_rank: int,
                 quant_config: Optional[QuantizationConfig],
                 cache_config: Optional[CacheConfig],
                 topk_indices_buffer: Optional[torch.Tensor],
                 prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        # for misa
        self.use_misa = getattr(config, "use_misa", False)
        if self.use_misa:
            self.topk_heads = config.misa_topk_heads  # 16
            self.decode_topk_heads = getattr(config, "misa_decode_topk_heads", None)  # 16
            self.k_block_size = getattr(config, "misa_k_block_size", None)  # 128
            self.use_qnorm = getattr(config, "misa_use_qnorm", True)
            if not self.use_qnorm:
                assert  self.k_block_size is not None, "k_block_size must be set when not using qnorm for head selection in MiSA indexer."
            if 'layers.0' in prefix:
                print(prefix)
                print(f'Using misa indexer with topk_heads {self.topk_heads}, decode_topk_heads {self.decode_topk_heads}.')
                if self.use_qnorm:
                    print(f'Using qnorm for head selection in misa indexer.')
                else:
                    print('Using block attention score for head selection in misa indexer.')
                if self.k_block_size is not None:
                    print(f'Using block-based indexing with k_block_size {self.k_block_size}.')

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(self.q_lora_rank,
                                     self.head_dim * self.n_head,
                                     bias=False,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.wq_b")
        self.wk = ReplicatedLinear(hidden_size,
                                   self.head_dim,
                                   bias=False,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.wk")
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(hidden_size,
                                             self.n_head,
                                             quant_config=None,
                                             prefix=f"{prefix}.weights_proj")
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        # NOTE: (zyongye) we use fp8 naive cache,
        #       where we store value in fp8 and scale in fp32
        #       per self.quant_block_size element
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim +
            self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config)
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix
        from vllm.v1.attention.backends.mla.indexer import (
            get_max_prefill_buffer_size)
        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)

        if self.use_misa:
            if "layers.0" in prefix:
                logger.info(f"use_misa={self.use_misa}, use misa sparse attn indexer custom op")
            self.indexer_op = MiSASparseAttnIndexer(
                self.k_cache,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_heads,
                self.decode_topk_heads,
                self.use_qnorm,
                self.k_block_size,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )
        else:
            if "layers.0" in prefix:
                logger.info(f"use_misa={self.use_misa}, use patched baseline sparse attn indexer custom op")
            self.indexer_op = BaselineSparseAttnIndexer(
                self.k_cache,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )

    def forward(self, hidden_states: torch.Tensor, qr: torch.Tensor, positions,
                rotary_emb) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)

        k, _ = self.wk(hidden_states)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # Note: RoPE (NeoX) can introduce extra leading dimensions during compilation
        # so we need to reshape back to token-flattened shapes
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        # `rotary_emb` is shape-preserving; `q_pe` is already
        # [num_tokens, n_head, rope_dim].
        q = torch.cat([q_pe, q_nope], dim=-1)
        # `k_pe` is [num_tokens, 1, rope_dim] (MQA).
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

        q_head_topk_indices = None
        weights, _ = self.weights_proj(hidden_states)
        if self.use_misa and self.use_qnorm:
            if self.decode_topk_heads is not None:
                topk_heads = self.decode_topk_heads
            else:
                topk_heads = self.topk_heads
            q_norm = torch.norm(q, p=2, dim=-1) # (seq, heads)
            _, q_head_topk_indices = torch.topk(q_norm, k=topk_heads, dim=-1, largest=True, sorted=False)
            del q_norm
        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = per_token_group_quant_fp8(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
        )
        weights = weights.squeeze(-1)

        if self.use_misa:
            return self.indexer_op(hidden_states, q_fp8, k, weights, q_head_topk_indices)
        else:
            return self.indexer_op(hidden_states, q_fp8, k, weights)

class MiSAMLAAttention(nn.Module):
    """
    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

        For more info see MLACommonImpl in:
        vllm/v1/attention/backends/mla/utils.py
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Union[DeepseekV2Config, DeepseekV3Config],
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        topk_indices_buffer: Optional[torch.Tensor] = None,
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

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProj(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
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

        if (
            config.rope_parameters["rope_type"] != "default"
            and config.rope_parameters["rope_type"] == "deepseek_yarn"
        ):
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.is_v32 = hasattr(config, "index_topk")

        if self.is_v32:
            self.indexer_rope_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=not getattr(config, "indexer_rope_interleave", False),
            )
            self.indexer = MiSAIndexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )

            # self.indexer = Indexer(
            #     vllm_config,
            #     config,
            #     hidden_size,
            #     q_lora_rank,
            #     quant_config,
            #     cache_config,
            #     topk_indices_buffer,
            #     f"{prefix}.indexer",
            # )

        else:
            self.indexer_rope_emb = None
            self.indexer = None

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj
            if self.q_lora_rank is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa
            if self.q_lora_rank is None
            else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=self.indexer,
            indexer_rotary_emb=self.indexer_rope_emb,
            is_sparse=self.is_v32,
            topk_indices_buffer=topk_indices_buffer,
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
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states, llama_4_scaling)


class MiSADecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: DeepseekV2Config | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        # verify MLA attention specific fields
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        v_head_dim = getattr(config, "v_head_dim", 0)
        kv_lora_rank = getattr(config, "kv_lora_rank", 0)
        use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
        )

        self.use_mha = use_mha

        if use_mha:
            raise NotImplementedError("mha is not supported in hisa yet")
        elif model_config.use_mla:
            attn_cls = MiSAMLAAttention
        else:
            raise NotImplementedError("DeepseekV2Attention is not supported in hisa yet")

        self.self_attn = attn_cls(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=config.q_lora_rank if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
        )

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        ):
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_kwargs = {
            "positions": positions,
            "hidden_states": hidden_states,
        }
        if not self.use_mha:
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
        hidden_states = self.self_attn(**attn_kwargs)

        if (
            not isinstance(self.self_attn, DeepseekAttention)
            and hidden_states.dtype == torch.float16
        ):
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1.0 / self.routed_scaling_factor

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if isinstance(self.mlp, DeepseekV2MLP) and hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # Scaling the DeepseekV2MLP output, it is the input of
            # input_layernorm of next decoder layer.
            # The scaling of DeepseekV2MOE output would be done in the forward
            # of DeepseekV2MOE
            hidden_states *= 1.0 / self.routed_scaling_factor

        return hidden_states, residual


@support_torch_compile
class MiSAModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type

        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiSADecoderLayer(
                vllm_config, prefix, topk_indices_buffer=topk_indices_buffer
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
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Compute llama 4 scaling once per forward pass if enabled
        llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
        llama_4_scaling: torch.Tensor | None
        if llama_4_scaling_config is not None:
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config[
                    "original_max_position_embeddings"
                ],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )
        else:
            llama_4_scaling = None

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions, hidden_states, residual, llama_4_scaling
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MiSAForCausalLM(DeepseekV2ForCausalLM):
    model_cls = MiSAModel

    def set_moe_parameters(self):
        self.expert_weights = []

        self.num_expert_groups = getattr(self.config, "n_group", 1)

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, MiSADecoderLayer)
            if isinstance(layer.mlp, DeepseekV2MoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)
