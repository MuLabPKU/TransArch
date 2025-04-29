import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F
from .lora_qkv import apply_rotary_pos_emb

class AbsorbQKVO(nn.Module):
    def __init__(self, self_attn):
        super().__init__()
        self.config = self_attn.config
        self.dtype = self_attn.q_a_proj.weight.dtype
        self.layer_idx = self_attn.layer_idx
        self.num_attention_heads = self_attn.num_attention_heads
        self.num_key_value_heads = self_attn.num_key_value_heads
        self.qk_mqa_dim = self_attn.qk_mqa_dim
        self.v_mqa_dim = self_attn.v_mqa_dim
        self.head_dim = self_attn.head_dim
        self.q_lora_rank = self_attn.q_lora_rank
        self.kv_lora_rank = self_attn.kv_lora_rank
        self.hidden_size = self_attn.hidden_size
        self.attention_dropout = self_attn.attention_dropout
        self.latent_dim = self_attn.latent_dim
        self.q_a_proj = self_attn.q_a_proj
        self.q_b_proj = nn.Linear(
            self.q_lora_rank,
            self.num_attention_heads * (self.qk_mqa_dim + self.kv_lora_rank), 
            bias=self_attn.q_b_with_mqa_proj.bias is not None,
            device = self_attn.q_b_with_mqa_proj.weight.device,
            dtype = self.dtype,
        )
        self.kv_proj = self_attn.kv_a_with_mqa_proj 
        self.o_proj = nn.Linear(
            self.num_attention_heads*(self.v_mqa_dim+self.kv_lora_rank),
            self.hidden_size,
            bias=self_attn.o_proj.bias is not None,
            device = self_attn.o_proj.weight.device,
            dtype = self.dtype,
        )
        self.__init_deepseek__(self_attn)
        
    def __init_deepseek__(self, self_attn):
        q_b_with_mqa_weight = self_attn.q_b_with_mqa_proj.weight.data
        q_b_rope_weight, q_b_nope_weight = q_b_with_mqa_weight.view(self.num_attention_heads, self.qk_mqa_dim+self.head_dim, self.q_lora_rank).split([self.qk_mqa_dim, self.head_dim], dim=1)
        kv_b_weight = self_attn.kv_b_proj.weight.data
        k_b_nope_weight, v_b_nope_weight = kv_b_weight.split(self.num_attention_heads*self.head_dim, dim=0)
        k_b_nope_weight = k_b_nope_weight.view(self.num_attention_heads, self.head_dim, self.kv_lora_rank)
        q_b_nope_weight = torch.einsum('hdq,hdk->hkq',q_b_nope_weight, k_b_nope_weight)
        q_b_weight = torch.cat([q_b_rope_weight, q_b_nope_weight],dim=1).reshape(self.num_attention_heads*(self.qk_mqa_dim+self.kv_lora_rank), self.q_lora_rank)
        assert self.q_b_proj.weight.data.shape == q_b_weight.shape
        self.q_b_proj.weight.data = q_b_weight
        v_b_rope_weight = self_attn.v_b_rope_proj.weight.data
        v_b_rope_weight = v_b_rope_weight.view(self.num_attention_heads, self.head_dim, self.v_mqa_dim)
        v_b_nope_weight = v_b_nope_weight.view(self.num_attention_heads, self.head_dim, self.kv_lora_rank)
        v_b_weight = torch.cat([v_b_rope_weight, v_b_nope_weight], dim=-1)
        o_weight = self_attn.o_proj.weight.data.view(self.hidden_size, self.num_attention_heads, self.head_dim)
        o_weight = torch.einsum('Dhd,hdv->Dhv', o_weight, v_b_weight)
        o_weight = o_weight.reshape(self.hidden_size, self.num_attention_heads*(self.v_mqa_dim+self.kv_lora_rank))
        assert self.o_proj.weight.data.shape == o_weight.shape
        self.o_proj.weight.data = o_weight

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
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.qk_mqa_dim+self.kv_lora_rank)
        q_rope, q_nope = query_states.split([self.qk_mqa_dim, self.kv_lora_rank], dim=-1)
        key_outputs = self.kv_proj(hidden_states).view(bsz, 1, q_len, self.qk_mqa_dim+self.v_mqa_dim+self.kv_lora_rank)
        k_rope, v_rope, kv_nope = key_outputs.split([self.qk_mqa_dim, self.v_mqa_dim, self.kv_lora_rank], dim=-1)
        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb(q_rope.transpose(1,2), k_rope, cos, sin)
        query_states = torch.cat([q_rope, q_nope.transpose(1,2)],dim=-1)
        key_states = torch.cat([k_rope, kv_nope], dim=-1)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            k_rope, kv_nope = past_key_value.update(k_rope, kv_nope, self.layer_idx, cache_kwargs)
            
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        value_states = torch.cat([v_rope, kv_nope],dim=-1)
        attn_nope_output = torch.matmul(attn_weights, value_states)
        attn_nope_output = attn_nope_output.transpose(1, 2).contiguous()
        attn_nope_output = attn_nope_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_nope_output)
        
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights