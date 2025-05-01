import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F

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

class SeparateKV(nn.Module):
    def __init__(self, self_attn, qk_mqa_dim=64, q_lora_rank=2048, k_lora_rank=448, v_lora_rank=512):
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
        self.k_lora_rank = k_lora_rank
        self.v_lora_rank = v_lora_rank
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
        self.k_a_proj = nn.Linear(
            self.hidden_size,
            qk_mqa_dim + k_lora_rank,
            bias=self_attn.k_proj.bias is not None,
            device = self_attn.k_proj.weight.device,
            dtype = self.dtype,
        )
        self.k_b_proj = nn.Linear(
            k_lora_rank,
            self.num_attention_heads * self.head_dim,
            bias=self_attn.k_proj.bias is not None,
            device = self_attn.k_proj.weight.device,
            dtype = self.dtype,
        )
        self.v_a_proj = nn.Linear(
            self.hidden_size,
            v_lora_rank,
            bias=self_attn.v_proj.bias is not None,
            device = self_attn.v_proj.weight.device,
            dtype = self.dtype,
        )
        self.v_b_proj = nn.Linear(
            v_lora_rank,
            self.num_attention_heads * self.head_dim,
            bias=self_attn.v_up_proj.bias is not None,
            device = self_attn.v_up_proj.weight.device,
            dtype = self.dtype,
        )
        self.o_proj = self_attn.o_proj
        self.__init_deepseek__(self_attn)
        
    def __init_deepseek__(self, self_attn, niter=16):
        # query svd
        Uq,Sq,Vq = torch.svd_lowrank(self_attn.q_proj.weight.data.to(torch.float64), self.q_lora_rank, niter=niter)
        q_a_weight = (torch.diag(torch.sqrt(Sq))@Vq.T).to(self.dtype)
        q_b_weight = (Uq@torch.diag(torch.sqrt(Sq))).to(self.dtype)
        q_b_weight = q_b_weight.view(self.num_attention_heads, self.head_dim, self.q_lora_rank)
        assert self.q_a_proj.weight.data.shape == q_a_weight.shape
        self.q_a_proj.weight.data = q_a_weight
        
        # key split mqa rope
        k_a_rope_weight, k_a_nope_weight = self_attn.k_proj.weight.data.split([self.qk_mqa_dim, self.latent_dim-self.qk_mqa_dim],dim=0)
        k_b_rope_weight, k_b_nope_weight = self_attn.k_up_proj.weight.data.split([self.qk_mqa_dim, self.latent_dim-self.qk_mqa_dim], dim=1)
        k_b_rope_weight = k_b_rope_weight.view(self.num_attention_heads, self.head_dim, self.qk_mqa_dim)
        
        # query split mqa rope
        q_b_rope_weight = torch.einsum("hdq,hdk->hkq", q_b_weight, k_b_rope_weight) 
        q_b_with_mqa_weight = torch.cat([q_b_rope_weight, q_b_weight],dim=1).reshape(self.num_attention_heads*(self.qk_mqa_dim+self.head_dim), self.q_lora_rank)
        assert self.q_b_proj.weight.data.shape == q_b_with_mqa_weight.shape
        self.q_b_proj.weight.data = q_b_with_mqa_weight
        
        # value split mqa
        v_a_weight, _ = self_attn.v_proj.weight.data.split([self.v_lora_rank, self.latent_dim-self.v_lora_rank],dim=0)
        v_b_weight, _ = self_attn.v_up_proj.weight.data.split([self.v_lora_rank, self.latent_dim-self.v_lora_rank], dim=1)

        assert self.v_a_proj.weight.data.shape == v_a_weight.shape
        assert self.v_b_proj.weight.data.shape == v_b_weight.shape
        self.v_a_proj.weight.data = v_a_weight
        self.v_b_proj.weight.data = v_b_weight
        
        # key & value svd 
        Ukv,Skv,Vkv = torch.svd_lowrank((k_b_nope_weight@k_a_nope_weight), self.k_lora_rank, niter=niter)
        k_a_nope_weight = (torch.diag(torch.sqrt(Skv))@Vkv.T).to(self.dtype)
        k_b_nope_weight = (Ukv@torch.diag(torch.sqrt(Skv))).to(self.dtype)
        assert self.k_b_proj.weight.data.shape == k_b_nope_weight.shape
        self.k_b_proj.weight.data = k_b_nope_weight
        k_a_proj_with_mqa_weight = torch.cat([k_a_rope_weight, k_a_nope_weight],dim=0)
        assert self.k_a_proj.weight.data.shape == k_a_proj_with_mqa_weight.shape
        self.k_a_proj.weight.data = k_a_proj_with_mqa_weight

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
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.qk_mqa_dim+self.head_dim).transpose(1,2)
        q_rope, q_nope = query_states.split([self.qk_mqa_dim, self.head_dim], dim=-1)
        k_rope, k_nope = self.k_a_proj(hidden_states).view(bsz, 1, q_len, self.qk_mqa_dim+self.k_lora_rank).split([self.qk_mqa_dim, self.k_lora_rank], dim=-1)
        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        query_states = torch.cat([q_rope, q_nope], dim=-1)

        if past_key_value is not None:
            pass
            #cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            #k_rope, k_nope_v = past_key_value.update(k_rope, k_nope_v, self.layer_idx, cache_kwargs)

        k_nope = self.k_b_proj(k_nope)
        k_nope = k_nope.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1,2)
        key_states = torch.cat([repeat_kv(k_rope, self.num_attention_heads), k_nope], dim=-1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        value_states = self.v_b_proj(self.v_a_proj(hidden_states))
        value_states = value_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1,2)
        attn_nope_output = torch.matmul(attn_weights, value_states)
        attn_nope_output = attn_nope_output.transpose(1, 2).contiguous()
        attn_nope_output = attn_nope_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_nope_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights
