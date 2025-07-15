import torch
import torch.nn as nn
import re
import logging

def rotate_attention_inputs(self_attn: nn.Module, Q: torch.Tensor) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for name in ["q_proj", "q_a_proj", "kv_a_proj_with_mqa"]:
        if hasattr(self_attn, name):
            W = getattr(self_attn, name)
            dtype = W.weight.dtype
            W_ = W.weight.to(dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)
            print(f"rotate {name} done.")
            
def rotate_attention_output(self_attn: nn.Module, Q: torch.Tensor) -> None:
    # Rotate output matrix of the self-attention layer.
    for name in ["o_proj", ]:
        if hasattr(self_attn, name):
            W = getattr(self_attn, name)
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(dtype=torch.float64)
            W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
            if W.bias is not None:
                b = W.bias.data.to(dtype=torch.float64)
                W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)
            print(f"rotate {name} done.")

def rotate_mlp_input(mlp: nn.Module, Q: torch.Tensor) -> None:
    # Rotate the MLP & MoE input weights.
    for name, module in mlp.named_modules():
        if re.search(r"(up_proj|gate_proj|gate)$", name):
            dtype = module.weight.dtype
            W_ = module.weight.data.to(dtype=torch.float64)
            module.weight.data = torch.matmul(W_, Q).to(dtype=dtype)
            print(f"rotate {name} done.")

def rotate_mlp_output(mlp: nn.Module, Q: torch.Tensor) -> None:
    # Rotate the MLP output weights and bias.
    for name, module in mlp.named_modules():
        if re.search(r"(down_proj)$", name):
            dtype = module.weight.data.dtype
            W_ = module.weight.data.to(dtype=torch.float64)
            module.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
            if module.bias is not None:
                b = module.bias.data.to(dtype=torch.float64)
                module.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)
            print(f"rotate {name} done.")

def rotate_embeddings(embed_tokens: nn.Module, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    dtype = embed_tokens.weight.data.dtype
    W_ = embed_tokens.weight.data.to(dtype=torch.float64)
    embed_tokens.weight.data = torch.matmul(W_, Q).to(dtype=dtype)
    print(f"rotate embed_tokens done.")

def rotate_lm_head(lm_head: nn.Module, Q: torch.Tensor) -> None:
    # Rotate the head.
    dtype = lm_head.weight.data.dtype
    W_ = lm_head.weight.data.to(dtype=torch.float64)
    lm_head.weight.data = torch.matmul(W_, Q).to(dtype=dtype)
    print(f"rotate lm_head done.")

def model_rotate(model: nn.Module, dtype, emb_Q: torch.Tensor, attn_Q: list[torch.Tensor], mlp_Q: list[torch.Tensor]):
    rotate_embeddings(model.model.embed_tokens, emb_Q)

    for layer_idx, layer in enumerate(model.model.layers):
        in_attn_Q = out_mlp_Q if layer_idx > 0 else emb_Q
        in_mlp_Q = out_attn_Q = attn_Q[layer_idx]
        out_mlp_Q = mlp_Q[layer_idx]

        layer.attn_shortcut_Q.data = torch.matmul(in_attn_Q.T.clone(), out_attn_Q.to(dtype=dtype),).to(torch.bfloat16)
        layer.mlp_shortcut_Q.data = torch.matmul(in_mlp_Q.T.clone().to(dtype=dtype), out_mlp_Q.to(dtype=dtype),).to(torch.bfloat16)

        rotate_attention_inputs(layer.self_attn, in_attn_Q)
        rotate_attention_output(layer.self_attn, out_attn_Q)

        rotate_mlp_input(layer.mlp, in_mlp_Q)
        rotate_mlp_output(layer.mlp, out_mlp_Q)

    rotate_lm_head(model.lm_head, out_mlp_Q)