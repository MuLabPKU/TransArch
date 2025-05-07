import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F
from .utils import pca_calc

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
    def __init__(self, self_attn, query_outputs=None, key_outputs=None, value_outputs=None, qk_mqa_dim=64, q_lora_rank=2048, kv_lora_rank=896):
        super().__init__()
        assert qk_mqa_dim == self_attn.head_dim
        self.config = self_attn.config
        self.dtype = self_attn.q_proj.weight.dtype
        self.layer_idx = self_attn.layer_idx
        self.num_attention_heads = self_attn.num_attention_heads
        self.num_key_value_heads = self_attn.num_key_value_heads
        self.head_dim = self_attn.head_dim
        self.qk_mqa_dim = qk_mqa_dim
        self.latent_dim = self_attn.latent_dim
        self.attention_dropout = self_attn.attention_dropout
        self.hidden_size = self_attn.hidden_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.q_a_proj = nn.Linear(
            self.hidden_size, 
            q_lora_rank, 
            bias=self_attn.q_proj.bias is not None,
            device = self_attn.q_proj.weight.device,
            dtype = self.dtype,
        )
        self.q_b_proj = nn.Linear(
            q_lora_rank,
            self.num_attention_heads * (self.qk_mqa_dim + self.head_dim), 
            bias=self_attn.q_proj.bias is not None,
            device = self_attn.q_proj.weight.device,
            dtype = self.dtype,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            kv_lora_rank + qk_mqa_dim,
            bias=self_attn.k_proj.bias is not None,
            device = self_attn.k_proj.weight.device,
            dtype = self.dtype,
        )
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            self.num_attention_heads * self.head_dim * 2,
            bias=self_attn.k_proj.bias is not None,
            device = self_attn.k_proj.weight.device,
            dtype = self.dtype,
        )
        self.o_proj = self_attn.o_proj
        kv_outputs = [torch.cat([key_outputs[i][:,:,qk_mqa_dim:], value_outputs[i]],dim=-1) for i in range(len(key_outputs))]
        R_q = pca_calc(query_outputs, self_attn.q_proj.weight.device)
        R_kv = pca_calc(kv_outputs, self_attn.k_proj.weight.device)
        self.__init_deepseek__(self_attn, R_q, R_kv)
        
    def __init_deepseek__(self, self_attn, R_q, R_kv):
        # query svd
        q_a_weight = (R_q.T@self_attn.q_proj.weight.data.to(torch.float64))[:self.q_lora_rank].to(self.dtype)
        q_b_weight = R_q[:,:self.q_lora_rank].to(self.dtype)
        q_b_weight = q_b_weight.view(self.num_attention_heads, self.head_dim, self.q_lora_rank)
        assert self.q_a_proj.weight.data.shape == q_a_weight.shape
        self.q_a_proj.weight.data = q_a_weight.contiguous()
        
        # key split mqa rope
        k_a_rope_weight, k_a_nope_weight = self_attn.k_proj.weight.data.split([self.qk_mqa_dim, self.latent_dim-self.qk_mqa_dim],dim=0)
        k_b_rope_weight, k_b_nope_weight = self_attn.k_up_proj.weight.data.split([self.qk_mqa_dim, self.latent_dim-self.qk_mqa_dim], dim=1)
        k_b_rope_weight = k_b_rope_weight.view(self.num_attention_heads, self.head_dim, self.qk_mqa_dim)
        
        # query split mqa rope
        q_b_rope_weight = torch.einsum("hdq,hdk->hkq", q_b_weight, k_b_rope_weight) 
        q_b_with_mqa_weight = torch.cat([q_b_weight, q_b_rope_weight],dim=1).reshape(self.num_attention_heads*(self.head_dim+self.qk_mqa_dim), self.q_lora_rank)
        assert self.q_b_proj.weight.data.shape == q_b_with_mqa_weight.shape
        self.q_b_proj.weight.data = q_b_with_mqa_weight.contiguous() * math.sqrt(self.head_dim+self.q_lora_rank) / math.sqrt(self.head_dim)
        
        # value split mqa
        v_a_nope_weight  = self_attn.v_proj.weight.data
        v_b_nope_weight = self_attn.v_up_proj.weight.data
        
        # key & value svd 
        kv_a_nope_weight = torch.cat([k_a_nope_weight, v_a_nope_weight],dim=0).to(torch.float64)
        k_b_nope_weight = k_b_nope_weight.view(self.num_attention_heads, self.head_dim, self.latent_dim-self.qk_mqa_dim)
        v_b_nope_weight = v_b_nope_weight.view(self.num_attention_heads, self.head_dim, self.latent_dim)
        kv_b_nope_weight = torch.cat(
            [
                torch.cat([k_b_nope_weight, torch.zeros_like(v_b_nope_weight)], dim=-1),
                torch.cat([torch.zeros_like(k_b_nope_weight), v_b_nope_weight],dim=-1)
            ], 
            dim=1
        ).reshape(2*self.num_attention_heads*self.head_dim, 2*self.latent_dim-self.qk_mqa_dim).to(torch.float64)
        kv_a_nope_weight = (R_kv.T@kv_a_nope_weight)[:self.kv_lora_rank].to(self.dtype)
        kv_b_nope_weight = (kv_b_nope_weight@R_kv)[:,:self.kv_lora_rank].to(self.dtype)
        assert self.kv_b_proj.weight.data.shape == kv_b_nope_weight.shape
        self.kv_b_proj.weight.data = kv_b_nope_weight.contiguous()
        kv_a_proj_with_mqa_weight = torch.cat([kv_a_nope_weight, k_a_rope_weight],dim=0)
        assert self.kv_a_proj_with_mqa.weight.data.shape == kv_a_proj_with_mqa_weight.shape
        self.kv_a_proj_with_mqa.weight.data = kv_a_proj_with_mqa_weight.contiguous()

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
        query_states = self.q_b_proj(self.q_a_proj(hidden_states))
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim+self.qk_mqa_dim).transpose(1,2)
        q_nope, q_rope = query_states.split([self.head_dim, self.qk_mqa_dim], dim=-1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, k_rope = compressed_kv.split([self.kv_lora_rank, self.qk_mqa_dim], dim=-1)
        k_rope = k_rope.view(bsz, 1, q_len, self.qk_mqa_dim)
        kv_nope = kv_nope.view(bsz, 1, q_len, self.kv_lora_rank)
        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        query_states = torch.cat([q_nope, q_rope], dim=-1)
        kv_nope = self.kv_b_proj(kv_nope).view(bsz, q_len, self.num_attention_heads, self.head_dim*2).transpose(1, 2)
        k_nope, v_nope = kv_nope.split([self.head_dim, self.head_dim],dim=-1)
        key_states = torch.cat([k_nope, repeat_kv(k_rope, self.num_attention_heads)], dim=-1)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim+self.q_lora_rank)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_nope_output = torch.matmul(attn_weights, v_nope)
        attn_nope_output = attn_nope_output.transpose(1, 2).contiguous()
        attn_nope_output = attn_nope_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_nope_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights
