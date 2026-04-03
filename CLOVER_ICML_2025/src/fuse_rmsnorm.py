import torch
import torch.nn as nn
import re
import logging

def fuse_to_attention_inputs(self_attn: nn.Module, input_layernorm: torch.Tensor) -> None:
    # Fuse input_layernorm into WQ, WK and WV matrices of the self-attention layer.
    for name in ["q_proj", "q_a_proj", "kv_a_proj_with_mqa"]:
        if hasattr(self_attn, name):
            W = getattr(self_attn, name)
            dtype = W.weight.dtype
            W_ = W.weight.to(torch.float64)
            W.weight.data = (W_*input_layernorm.weight.data[None,:]).to(dtype=dtype)
    torch.nn.init.ones_(input_layernorm.weight)

def fuse_to_mlp_input(mlp: nn.Module, post_attention_layernorm: torch.Tensor) -> None:
    # Fuse post_attention_layernorm into MLP & MoE input weights.
    for name, module in mlp.named_modules():
        if re.search(r"(up_proj|gate_proj|gate)$", name):
            dtype = module.weight.dtype
            W_ = module.weight.data.to(torch.float64)
            module.weight.data = (W_*post_attention_layernorm.weight.data[None,:]).to(dtype=dtype)
    torch.nn.init.ones_(post_attention_layernorm.weight)

def fuse_to_lm_head_input(lm_head: nn.Module, lm_norm: torch.Tensor) -> None:
    # Fuse lm_norm into lm_head weights.
        dtype = lm_head.weight.dtype
        W_ = lm_head.weight.data.to(torch.float64)
        lm_head.weight.data = (W_*lm_norm.weight.data[None,:]).to(dtype=dtype)
        torch.nn.init.ones_(lm_norm.weight)

def insert_shortcut_and_fuse_rmsnorm(model):
    for layer in model.model.layers:
        torch.nn.init.eye_(layer.attn_shortcut_Q)
        torch.nn.init.eye_(layer.mlp_shortcut_Q)

        fuse_to_attention_inputs(layer.self_attn, layer.input_layernorm)
        fuse_to_mlp_input(layer.mlp, layer.post_attention_layernorm)
    fuse_to_lm_head_input(model.lm_head, model.model.norm)