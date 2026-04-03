import torch
import torch.nn as nn
import re
import math

def slice_attention_inputs(self_attn: nn.Module, dim: int) -> None:
    # Slice the WQ, WK and WV matrices of the self-attention layer.
        for name in ["q_proj", "q_a_proj", "kv_a_proj_with_mqa"]:
            if hasattr(self_attn, name):
                W = getattr(self_attn, name)
                W.weight.data = W.weight.data[:, :dim]
                W.in_features = dim

def slice_attention_output(self_attn: nn.Module, dim: int) -> None:
    # Slice output matrix of the self-attention layer.
    for name in ["o_proj", ]:
        if hasattr(self_attn, name):
            W = getattr(self_attn, name)
            W.weight.data = W.weight.data[:dim, :]
            if W.bias is not None:
                W.bias.data = W.bias.data[:dim]
            W.out_features = dim

def slice_mlp_input(mlp: nn.Module, dim: int) -> None:
    # Slice the MLP input weights.
    for name, module in mlp.named_modules():
        if re.search(r"(up_proj|gate_proj|gate)$", name):
            dtype = module.weight.dtype
            module.weight.data = module.weight.data[:, :dim]
            module.in_features = dim
               
def slice_mlp_output(mlp: nn.Module, dim: int) -> None:
    # Slice the MLP output weights and bias.
    for name, module in mlp.named_modules():
        if re.search(r"(down_proj)$", name):
            dtype = module.weight.data.dtype
            module.weight.data = module.weight.data[:dim, :]
            if module.bias is not None:
                module.bias.data = module.bias.data[:dim]
            module.out_features = dim

def slice_embeddings(embed_tokens: nn.Module, dim: int) -> None:
    # Slice the embeddings.
    dtype = embed_tokens.weight.data.dtype
    embed_tokens.weight.data = embed_tokens.weight.data[:, : dim]
    embed_tokens.embedding_dim = dim

def slice_lm_head(lm_head: nn.Module, dim: int) -> None:
    # Slice the head.
    dtype = lm_head.weight.data.dtype
    lm_head.weight.data = lm_head.weight.data[:, :dim]
    lm_head.in_features = dim

def model_slice(model: nn.Module, dim: int, prune_lm_head=False):
    slice_embeddings(model.model.embed_tokens, dim)

    for layer_idx, layer in enumerate(model.model.layers):
        slice_attention_inputs(layer.self_attn, dim)
        slice_attention_output(layer.self_attn, dim)
        slice_mlp_input(layer.mlp, dim)
        layer.attn_shortcut_Q.data = layer.attn_shortcut_Q.data[:dim,:dim]
        if layer_idx < len(model.model.layers)-1 or prune_lm_head:
            slice_mlp_output(layer.mlp, dim)
            layer.mlp_shortcut_Q.data = layer.mlp_shortcut_Q.data[:dim,:dim]
        else:
            layer.mlp_shortcut_Q.data = layer.mlp_shortcut_Q.data[:dim]


        layer.input_layernorm.weight.data = layer.input_layernorm.weight.data[:dim] * math.sqrt(layer.input_layernorm.weight.data.shape[0]/dim)
        layer.post_attention_layernorm.weight.data = layer.post_attention_layernorm.weight.data[:dim] * math.sqrt(layer.post_attention_layernorm.weight.data.shape[0]/dim)

    if prune_lm_head:
        slice_lm_head(model.lm_head, dim)
        model.model.norm.weight.data = model.model.norm.weight.data[:dim]