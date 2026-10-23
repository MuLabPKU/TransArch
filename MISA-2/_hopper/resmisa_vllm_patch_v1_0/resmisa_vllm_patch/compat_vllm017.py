# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two explicit vLLM 0.17.0 compatibility methods for this package.

These ordinary source functions replace the two affected class methods in this
process. No installed file, model checkpoint, or model config is modified.
Unexpected versions/source revisions fail at model import instead of receiving
an unreviewed patch. Keep this file until upstream vLLM has equivalent fixes.
"""
import hashlib
import inspect
from importlib.metadata import version
from pathlib import Path

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

_INSTALLED = False
_ORIGINAL_MOE_SHA = "b636555bcf6084d5afffa25f15bc2ac8c68d4a74cd60fcaf34b1b2ae5c777c46"
_ORIGINAL_MLA_SHA = "524c990dd6e28ca01c268cedfc167931e2f65f00d6016eeb051f91cb0bae5f64"


def _apply_per_block(
    self,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor,
    activation: MoEActivation,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    a1q_scale: torch.Tensor | None,
    apply_router_weight_on_input: bool,
    # grouped topk + fused topk bias parameters
    num_expert_group: int | None = None,
    e_score_correction_bias: torch.Tensor | None = None,
    routed_scaling_factor: float | None = None,
    topk_group: int | None = None,
) -> torch.Tensor:
    # Delay import for non-CUDA.
    import flashinfer

    assert not apply_router_weight_on_input
    assert activation == MoEActivation.SILU

    if e_score_correction_bias is not None:
        e_score_correction_bias = e_score_correction_bias.to(torch.float32)

    if self.routing_method_type == RoutingMethodType.DeepSeekV3:
        router_logits = router_logits.to(torch.float32)

    assert self.topk <= global_num_experts
    assert self.topk <= 10
    assert global_num_experts % 4 == 0
    assert self.quant_config.block_shape == [128, 128]
    # Routing kernel expects #experts <= #threads 512
    assert global_num_experts <= 512

    # Kernel requires transposed hidden state scales
    # TODO: fuse into the quant kernel.
    assert a1q_scale is not None
    a1q_scale_t = a1q_scale.t().contiguous()

    return flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
        routing_logits=router_logits,
        routing_bias=e_score_correction_bias,
        hidden_states=hidden_states,
        hidden_states_scale=a1q_scale_t,
        gemm1_weights=w1,
        gemm1_weights_scale=self.quant_config.w1_scale,
        gemm2_weights=w2,
        gemm2_weights_scale=self.quant_config.w2_scale,
        num_experts=global_num_experts,
        top_k=self.topk,
        n_group=(num_expert_group or 0),
        topk_group=(topk_group or 0),
        intermediate_size=self.intermediate_size_per_partition,
        local_expert_offset=self.ep_rank * self.local_num_experts,
        local_num_experts=self.local_num_experts,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=self.routing_method_type,
        use_shuffled_weight=False,
    )


def _bf16_flash_mla_kernel(
    self,
    q: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    num_tokens = q.shape[0]
    kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
        -1, 1, kv_c_and_k_pe_cache.shape[-1]
    )

    # NOTE(Chen): kernel requires num_local_head to be a multiple of
    # 64 on hopper and 128 on blackwell
    if self.num_heads % self.prefill_padding != 0:
        assert self.prefill_padding % self.num_heads == 0
        q_padded = q.new_zeros((q.shape[0], self.prefill_padding, q.shape[2]))
        q_padded[:, : self.num_heads, :] = q
        q = q_padded

    topk_indices = topk_indices.view(num_tokens, 1, -1)
    output = flash_mla_sparse_fwd(
        q, kv_c_and_k_pe_cache, topk_indices, self.softmax_scale
    )[0]
    output = output[:, : self.num_heads, :]
    return output


def install():
    """Install the reviewed static methods once, before model construction."""
    global _INSTALLED
    if _INSTALLED:
        return
    installed = version("vllm")
    if installed.split("+")[0] != "0.17.0":
        raise RuntimeError(f"This compatibility module requires vLLM 0.17.0, got {installed}")
    from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import TrtLlmFp8Experts
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl
    replacements = (
        (TrtLlmFp8Experts, "_apply_per_block", _apply_per_block, _ORIGINAL_MOE_SHA),
        (FlashMLASparseImpl, "_bf16_flash_mla_kernel", _bf16_flash_mla_kernel, _ORIGINAL_MLA_SHA),
    )
    # Validate everything before replacing either method.
    for cls, name, fixed, expected in replacements:
        original = getattr(cls, name)
        path = Path(inspect.getsourcefile(original))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected or inspect.signature(original) != inspect.signature(fixed):
            raise RuntimeError(f"Unsupported vLLM source for {cls.__name__}.{name}: {actual}")
    for cls, name, fixed, _ in replacements:
        setattr(cls, name, fixed)
    _INSTALLED = True
    return


