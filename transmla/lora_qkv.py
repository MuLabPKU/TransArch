import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from utils import pca_calc, get_qkv_calibrate_outputs, evaluate_ppl, statistics_qkv_rmsnorm

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
 
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LoraQKV(nn.Module):
    def __init__(
        self, 
        self_attn, 
        query_outputs=None, 
        key_outputs=None, 
        value_outputs=None, 
        q_lora_rank=None, 
        qk_mqa_dim=64, 
        collapse=1,
        kv_lora_rank=896,
        use_qkv_norm=None, 
        balance_kv_ratio=None, 
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        assert qk_mqa_dim * collapse == self_attn.head_dim

        self.config = self_attn.config
        self.dtype = self_attn.q_proj.weight.dtype
        self.layer_idx = self_attn.layer_idx
        self.num_attention_heads = self_attn.num_attention_heads
        self.num_key_value_heads = self_attn.num_key_value_heads
        self.head_dim = self_attn.head_dim
        self.qk_mqa_dim = qk_mqa_dim
        self.collapse = collapse
        self.latent_dim = self_attn.latent_dim
        self.attention_dropout = self_attn.attention_dropout
        self.hidden_size = self_attn.hidden_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        assert self.kv_lora_rank <= 2 * self.latent_dim - self.qk_mqa_dim, f"kv_lora_rank ({self.kv_lora_rank}) must be less than 2 * latent_dim ({self.latent_dim}) - qk_mqa_dim ({self.qk_mqa_dim})"

        self.attention_function = ALL_ATTENTION_FUNCTIONS["sdpa"]
        # self.scaling = (self.head_dim + self.qk_mqa_dim)**(-0.5)
        self.scaling = self.head_dim**(-0.5)

        # -----------------Attributes for the bias-----------------
        q_bias = self_attn.q_proj.bias is not None
        k_bias = self_attn.k_proj.bias is not None
        v_bias = self_attn.v_proj.bias is not None
        assert q_bias == k_bias == v_bias, f"q_bias ({q_bias}), k_bias ({k_bias}), v_bias ({v_bias}) must be the same"
        self.attention_bias = q_bias

        # -----------------module definitions-----------------
        # q_a_proj & q_b_proj
        if q_lora_rank is not None:
            self.q_a_proj = nn.Linear(
                self.hidden_size, 
                q_lora_rank, 
                bias=None,
                device = self_attn.q_proj.weight.device,
                dtype = self.dtype,
            )
            if use_qkv_norm:
                self.q_a_layernorm = nn.RMSNorm(q_lora_rank, device=self_attn.q_proj.weight.device, dtype=self.dtype, eps=rms_norm_eps)
            self.q_b_proj = nn.Linear(
                q_lora_rank,
                self.num_attention_heads * (self.qk_mqa_dim + self.head_dim), 
                bias=self.attention_bias,
                device = self_attn.q_proj.weight.device,
                dtype = self.dtype,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size, 
                self.num_attention_heads * (self.qk_mqa_dim + self.head_dim), 
                bias=self.attention_bias,
                device = self_attn.q_proj.weight.device,
                dtype = self.dtype,
            )
        # kv_a_proj & kv_b_proj
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            kv_lora_rank + qk_mqa_dim,
            bias=self.attention_bias,
            device = self_attn.k_proj.weight.device,
            dtype = self.dtype,
        )
        if use_qkv_norm:
            self.kv_a_layernorm = nn.RMSNorm(kv_lora_rank, device=self_attn.k_proj.weight.device, dtype=self.dtype, eps=rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            self.num_attention_heads * self.head_dim * 2,
            bias=self.attention_bias,
            device = self_attn.k_proj.weight.device,
            dtype = self.dtype,
        )
        # nothing else to do for o_proj
        self.o_proj = self_attn.o_proj

        # -----------------apply bkv on the key and value outputs-----------------
        if balance_kv_ratio is not None:
            k_outputs_norm = torch.cat([key.reshape(-1, self.latent_dim)[:,self.qk_mqa_dim:] for key in key_outputs]).norm(p=2,dim=0).mean()
            v_outputs_norm = torch.cat([value.reshape(-1, self.latent_dim)[:,self.qk_mqa_dim:] for value in value_outputs]).norm(p=2,dim=0).mean()
            ratio = k_outputs_norm/(v_outputs_norm * balance_kv_ratio)
            self_attn.k_proj.weight.data[self.qk_mqa_dim:] = self_attn.k_proj.weight.data[self.qk_mqa_dim:] / ratio
            self_attn.k_up_proj.weight.data[:, self.qk_mqa_dim:] = self_attn.k_up_proj.weight.data[:, self.qk_mqa_dim:] * ratio
        else:
            ratio = 1
        kv_outputs = [torch.cat([key_outputs[i][:,:,qk_mqa_dim:] / ratio, value_outputs[i]],dim=-1) for i in range(len(key_outputs))]

        # -----------------apply pca on the query and key/value outputs-----------------
        if self.q_lora_rank is not None:
            if self.attention_bias:
                # If q_bias is not None, we need to remove the bias from the query_outputs, 
                # because the bias part does not need pca.
                R_q = pca_calc(
                    [x.to(dtype=self.dtype, device=self_attn.q_proj.bias.device) - self_attn.q_proj.bias.data for x in query_outputs], 
                    self_attn.q_proj.weight.device
                )
            else:
                R_q = pca_calc(query_outputs, self_attn.q_proj.weight.device)
        else:
            R_q = None
        if self.attention_bias:
            # If k_bias is not None, we need to remove the bias from the kv_outputs, 
            # because the bias part does not need pca.
            kv_bias = torch.cat([self_attn.k_proj.bias.data[qk_mqa_dim:] / ratio, self_attn.v_proj.bias.data])
            R_kv = pca_calc(
                [x.to(dtype=self.dtype, device=kv_bias.device) - kv_bias for x in kv_outputs], 
                self_attn.k_proj.weight.device
            )
        else:
            R_kv = pca_calc(kv_outputs, self_attn.k_proj.weight.device)

        # -----------------initialize the weights / bias-----------------
        self._init_weights(self_attn, R_q, R_kv)
        
    def _init_weights(self, self_attn, R_q, R_kv):
        # 0. Split the weights of k_proj and v_proj into rope / nope parts.
        k_a_rope_weight, k_a_nope_weight = self_attn.k_proj.weight.data.split([self.qk_mqa_dim, self.latent_dim - self.qk_mqa_dim],dim=0)
        k_b_rope_weight, k_b_nope_weight = self_attn.k_up_proj.weight.data.split([self.qk_mqa_dim, self.latent_dim - self.qk_mqa_dim], dim=1)
        k_b_rope_weight = k_b_rope_weight.view(self.num_attention_heads, self.head_dim, self.qk_mqa_dim)
        k_b_nope_weight = k_b_nope_weight.view(self.num_attention_heads, self.head_dim, self.latent_dim-self.qk_mqa_dim)
        if self.attention_bias:
            k_bias_rope, k_bias_nope = self_attn.k_proj.bias.data.split([self.qk_mqa_dim, self.latent_dim - self.qk_mqa_dim], dim=0)
            # k_bias = self_attn.k_proj.bias.data
        
        v_a_nope_weight  = self_attn.v_proj.weight.data
        v_b_nope_weight = self_attn.v_up_proj.weight.data
        v_b_nope_weight = v_b_nope_weight.view(self.num_attention_heads, self.head_dim, self.latent_dim)
        if self.attention_bias:
            v_bias = self_attn.v_proj.bias.data

        if self.attention_bias:
            q_bias = self_attn.q_proj.bias.data


        # 1. Initialize q_a_proj / q_b_proj if q_lora_rank is not None
        # 1.1 Initialize q_a_proj
        if self.q_lora_rank is not None:
            q_a_weight = (R_q.T @ self_attn.q_proj.weight.data.to(torch.float64))[: self.q_lora_rank].to(self.dtype)
            q_b_weight = R_q[:, :self.q_lora_rank].to(self.dtype)
            q_b_weight = q_b_weight.view(self.num_attention_heads, self.head_dim, self.q_lora_rank)
            assert self.q_a_proj.weight.data.shape == q_a_weight.shape
            self.q_a_proj.weight.data = q_a_weight.contiguous()

        # 1.2 Initialize q_b_proj
        # scaling = math.sqrt(self.head_dim + self.qk_mqa_dim) / math.sqrt(self.head_dim)
        scaling = 1
        if self.q_lora_rank is not None:
            # Absorb the rope part of k_b_proj into q_b_proj
            q_b_rope_weight = torch.einsum("hdq,hdk->hkq", q_b_weight, k_b_rope_weight)
            q_b_with_mqa_weight = torch.cat([q_b_weight, q_b_rope_weight], dim=1).reshape(
                self.num_attention_heads * (self.head_dim + self.qk_mqa_dim), self.q_lora_rank
            )

            # Scale the weight before initializing the q_b_proj
            # In the original GQA, attention scores are divided by sqrt(head_dim).
            # However, in the transformed MLA, the attention scores are divided by sqrt(head_dim + qk_mqa_dim).
            assert self.q_b_proj.weight.data.shape == q_b_with_mqa_weight.shape
            self.q_b_proj.weight.data = q_b_with_mqa_weight.contiguous() * scaling
            
            # Considering the bias
            if self.attention_bias:
                q_b_bias = q_bias.reshape(self.num_attention_heads, self.head_dim)
                q_b_rope_bias = torch.einsum("hd,hdk->hk", q_b_bias, k_b_rope_weight)
                q_b_with_mqa_bias = torch.cat([q_b_bias, q_b_rope_bias], dim=1).flatten()
                assert self.q_b_proj.bias.data.shape == q_b_with_mqa_bias.shape
                self.q_b_proj.bias.data = q_b_with_mqa_bias.contiguous() * scaling
        else:
            q_weight = self_attn.q_proj.weight.data.view(self.num_attention_heads, self.head_dim, self.hidden_size)
            q_rope_weight = torch.einsum("hdD,hdk->hkD", q_weight, k_b_rope_weight) 
            q_with_mqa_weight = torch.cat([q_weight, q_rope_weight], dim=1).reshape(
                self.num_attention_heads * (self.head_dim + self.qk_mqa_dim), self.hidden_size
            )
            assert self.q_proj.weight.data.shape == q_with_mqa_weight.shape
            self.q_proj.weight.data = q_with_mqa_weight.contiguous() * scaling

            if self.attention_bias:
                q_bias = q_bias.reshape(self.num_attention_heads, self.head_dim)
                q_rope_bias = torch.einsum("hd,hdk->hk", q_bias.to(torch.float64), k_b_rope_weight.to(torch.float64)).to(self.dtype)
                q_bias = torch.cat([q_bias, q_rope_bias], dim=1).flatten()
                assert self.q_proj.bias.data.shape == q_bias.shape
                self.q_proj.bias.data = q_bias.contiguous() * scaling
            
        
        # 2. Low-rank decomposing k_proj and v_proj
        # 2.1 Concatenate the nope parts of k_proj and v_proj
        kv_a_nope_weight = torch.cat([k_a_nope_weight, v_a_nope_weight], dim=0).to(torch.float64)
        kv_b_nope_weight = torch.cat(
            [
                torch.cat([k_b_nope_weight, torch.zeros_like(v_b_nope_weight)], dim=-1),
                torch.cat([torch.zeros_like(k_b_nope_weight), v_b_nope_weight], dim=-1)
            ], 
            dim=1
        ).reshape(2 * self.num_attention_heads * self.head_dim, 2 * self.latent_dim - self.qk_mqa_dim).to(torch.float64)

        # 2.2 Low-rank decomposing kv_a_nope_weight and kv_b_nope_weight
        kv_a_nope_weight = (R_kv.T @ kv_a_nope_weight)[: self.kv_lora_rank].to(self.dtype)
        kv_b_nope_weight = (kv_b_nope_weight @ R_kv)[:, :self.kv_lora_rank].to(self.dtype)
        self.kv_b_proj.weight.data = kv_b_nope_weight.contiguous()
        kv_a_proj_with_mqa_weight = torch.cat([kv_a_nope_weight, k_a_rope_weight], dim=0)
        assert self.kv_a_proj_with_mqa.weight.data.shape == kv_a_proj_with_mqa_weight.shape
        self.kv_a_proj_with_mqa.weight.data = kv_a_proj_with_mqa_weight.contiguous()

        # 2.3 Considering the bias of kv
        if self.attention_bias:
            kv_a_bias = torch.zeros(self.kv_lora_rank, dtype=self.dtype, device=self_attn.k_proj.bias.device)
            # kv_a_bias = torch.zeros_like(torch.cat([k_bias_nope, v_bias], dim=0), dtype=self.dtype, device=self_attn.k_proj.bias.device)
            kv_a_with_mqa_bias = torch.cat([kv_a_bias, k_bias_rope], dim=0)
            assert self.kv_a_proj_with_mqa.bias.data.shape == kv_a_with_mqa_bias.shape, f"{self.kv_a_proj_with_mqa.bias.data.shape} != {kv_a_with_mqa_bias.shape}"
            self.kv_a_proj_with_mqa.bias.data = kv_a_with_mqa_bias.contiguous()
            
            k_b_bias = k_b_nope_weight @ k_bias_nope
            v_b_bias = v_b_nope_weight @ v_bias
            kv_b_bias = torch.cat([k_b_bias, v_b_bias], dim=1).flatten()
            assert self.kv_b_proj.bias.data.shape == kv_b_bias.shape
            self.kv_b_proj.bias.data = kv_b_bias.contiguous()



    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # query
        if self.q_lora_rank is not None:
            query_states = self.q_a_proj(hidden_states)
            if hasattr(self, "q_a_layernorm"):
                query_states = self.q_a_layernorm(query_states)
            query_states = self.q_b_proj(query_states)
        else:
            query_states = self.q_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, -1).transpose(1,2)
        q_nope, q_rope = query_states.split([self.head_dim, self.qk_mqa_dim], dim=-1)

        # key and value
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, k_rope = compressed_kv.split([self.kv_lora_rank, self.qk_mqa_dim], dim=-1)
        kv_nope = kv_nope.view(bsz, 1, q_len, self.kv_lora_rank)
        k_rope = k_rope.view(bsz, 1, q_len, self.qk_mqa_dim)

        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos[ :, :, : : self.collapse], sin[ :, :, : : self.collapse])
        query_states = torch.cat([q_nope, q_rope], dim=-1)

        if hasattr(self, "kv_a_layernorm"):
            kv_nope = self.kv_a_layernorm(kv_nope)
        kv_nope = self.kv_b_proj(kv_nope).view(bsz, q_len, self.num_attention_heads, self.head_dim * 2).transpose(1, 2)
        k_nope, value_states = kv_nope.split([self.head_dim, self.head_dim],dim=-1)
        key_states = torch.cat([k_nope, repeat_kv(k_rope, self.num_attention_heads)], dim=-1)

        attn_output, attn_weights = self.attention_function(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


def low_rank_qkv(model, tokenizer, train_loader, test_loader, **kwargs):

    message = "Calibrating rope-removed model's qkv outputs"
    rm_rope_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader, message)

    for layer_idx, layer in enumerate(model.model.layers):
        setattr(layer, "self_attn", LoraQKV(
            layer.self_attn,
            rm_rope_qkv_outputs["query"][layer_idx], 
            rm_rope_qkv_outputs["key"][layer_idx], 
            rm_rope_qkv_outputs["value"][layer_idx], 
            q_lora_rank=kwargs["q_lora_rank"], 
            qk_mqa_dim=kwargs["qk_mqa_dim"], 
            collapse=kwargs["collapse"],
            kv_lora_rank=kwargs["kv_lora_rank"],
            use_qkv_norm=kwargs["use_qkv_norm"],
            balance_kv_ratio=kwargs["balance_kv_ratio"],
            rms_norm_eps=model.config.rms_norm_eps,
        ))
    
    if kwargs["use_qkv_norm"]:
        lora_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader)
        for layer_idx, layer in enumerate(model.model.layers):
            statistics_qkv_rmsnorm(
                layer.self_attn, 
                lora_qkv_outputs["q_a_proj"][layer_idx] if len(lora_qkv_outputs["q_a_proj"]) > layer_idx else None, 
                lora_qkv_outputs["kv_a_proj"][layer_idx]
            )

    if test_loader:
        message = "Evaluating lora-qkv model's ppl"
        dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader, message)
        print(f'Low rank approximate QKV ppl: {dataset_ppl:.4f}')
    
    return model