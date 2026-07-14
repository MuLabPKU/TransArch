"""patch_deepseek.py — attach DSA to a loaded DeepSeek-V2 HF model.

Usage:
    model = AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True,
                                                 attn_implementation="eager")
    patch_model_with_dsa(model, indexer_cfg=..., mode="warmup")

What it does per DeepseekV2Attention module:
  * instantiate a LightningIndexer sized from the model config
  * bind DSA hyperparams (topk, tile, mode flags) onto the module
  * replace ``forward`` with the DSA-aware forward (make_dsa_forward)

Stage control:
  * set_dsa_mode(model, "warmup"|"sparse"|"dense")
  * freeze_backbone_train_indexer(model)  — warmup: only indexers require grad
  * unfreeze_all(model)                    — sparse: everything trainable

The attention modules are located structurally (model.model.layers[i].self_attn)
so this works regardless of the remote code's class name, as long as it exposes
the DeepSeek-V2 MLA attributes (q_lora_rank, kv_lora_rank, num_heads, ...).
"""
from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Optional

import torch

from .indexer import LightningIndexer
from .dsa_attention import make_dsa_forward, REGISTRY


@dataclass
class IndexerConfig:
    n_heads: int = 8
    head_dim: int = 128        # must be >= qk_rope_head_dim (64)
    topk: int = 2048
    tile: int = 512
    q_chunk: int = 512         # query-chunk size for the memory-bounded sparse ops
    log_recall: bool = False
    log_kl: bool = False


def _find_layers(model):
    """Locate the decoder ``layers`` ModuleList, descending through any nesting of
    ``.module`` (DDP/FSDP wrappers) and ``.model`` (CausalLM -> decoder). Returns
    the ModuleList or None."""
    seen = set()
    node = model
    for _ in range(8):  # bounded descent; real nesting is <=3 deep
        if id(node) in seen or node is None:
            break
        seen.add(id(node))
        layers = getattr(node, "layers", None)
        if layers is not None and hasattr(layers, "__len__"):
            return layers
        # prefer .model (decoder) then .module (wrapper)
        nxt = getattr(node, "model", None)
        if nxt is None:
            nxt = getattr(node, "module", None)
        node = nxt
    return None


def _iter_attn_modules(model):
    """Yield (layer_idx, MLA self-attention module). Robust to DDP/FSDP wrapping."""
    layers = _find_layers(model)
    assert layers is not None, "could not locate decoder layers"
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "kv_a_proj_with_mqa"):
            yield i, attn


def patch_model_with_dsa(model, indexer_cfg: Optional[IndexerConfig] = None,
                         mode: str = "warmup", dtype: torch.dtype = torch.bfloat16):
    cfg = indexer_cfg or IndexerConfig()
    hidden = model.config.hidden_size
    rope_dim = model.config.qk_rope_head_dim
    q_lora = getattr(model.config, "q_lora_rank", None)
    q_input_dim = q_lora if q_lora is not None else hidden

    n_patched = 0
    for i, attn in _iter_attn_modules(model):
        idx = LightningIndexer(
            hidden_size=hidden, n_heads=cfg.n_heads, head_dim=cfg.head_dim,
            rope_dim=rope_dim, q_input_dim=q_input_dim, dtype=dtype,
        ).to(next(attn.parameters()).device)
        attn.indexer = idx
        attn.dsa_mode = mode
        attn.dsa_topk = cfg.topk
        attn.dsa_tile = cfg.tile
        attn.dsa_qchunk = cfg.q_chunk
        attn.dsa_log_recall = cfg.log_recall
        attn.dsa_log_kl = cfg.log_kl
        attn.cu_seqlens = None
        attn.cp_size = 1
        # keep a handle to the pristine forward and install the DSA one
        if not hasattr(attn, "_orig_forward_fn"):
            attn._orig_forward_fn = attn.__class__.forward
        attn.forward = types.MethodType(make_dsa_forward(attn._orig_forward_fn), attn)
        n_patched += 1
    model._dsa_num_layers = n_patched
    model._dsa_indexer_cfg = cfg
    return model


def set_dsa_mode(model, mode: str):
    for _, attn in _iter_attn_modules(model):
        attn.dsa_mode = mode


def set_cu_seqlens(model, cu_seqlens):
    for _, attn in _iter_attn_modules(model):
        attn.cu_seqlens = cu_seqlens


def set_cp_size(model, cp_size: int):
    """Enable context-parallel DSA on every attention module (cp_size>1 activates
    the CP branch in the warmup/sparse forward)."""
    for _, attn in _iter_attn_modules(model):
        attn.cp_size = cp_size


def set_log_recall(model, flag: bool):
    for _, attn in _iter_attn_modules(model):
        attn.dsa_log_recall = flag


def set_log_kl(model, flag: bool):
    for _, attn in _iter_attn_modules(model):
        attn.dsa_log_kl = flag


def indexer_parameters(model):
    ps = []
    for _, attn in _iter_attn_modules(model):
        ps += list(attn.indexer.parameters())
    return ps


def freeze_backbone_train_indexer(model):
    """Warmup: freeze everything, then unfreeze only the indexers."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in indexer_parameters(model):
        p.requires_grad_(True)


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad_(True)


def dsa_loss_registry():
    return REGISTRY


# --------------------------------------------------------------------------- #
#  expert parallel (EP) for the MoE
# --------------------------------------------------------------------------- #
def _iter_moe_modules(model):
    """Yield (layer_idx, MoE module) for layers that use a routed MoE (have `.gate`
    and `.experts`). Dense-MLP layers (first_k_dense_replace) are skipped."""
    layers = _find_layers(model)
    assert layers is not None, "could not locate decoder layers"
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "experts") and hasattr(mlp, "gate"):
            yield i, mlp


def patch_model_with_ep(model, ep_size: int):
    """Install expert-parallel MoE on every routed-MoE layer and DROP the experts
    this rank does not own (free their params/memory). Rank r keeps experts
    [r*E/ep : (r+1)*E/ep]; others are set to None. The gate + shared_experts are
    untouched. No-op if ep_size==1."""
    import types
    from .ep_moe import ep_moe_forward
    if ep_size <= 1:
        return model
    from ..dist import ep_rank
    r = ep_rank()
    n_patched = 0
    for _, mlp in _iter_moe_modules(model):
        E = len(mlp.experts)
        assert E % ep_size == 0, f"n_experts {E} not divisible by ep_size {ep_size}"
        per = E // ep_size
        lo, hi = r * per, (r + 1) * per
        for e in range(E):
            if not (lo <= e < hi):
                mlp.experts[e] = None      # free non-owned expert params
        mlp.forward = types.MethodType(ep_moe_forward, mlp)
        n_patched += 1
    model._dsa_ep_size = ep_size
    return model


def expert_parameters(model):
    """Params of the routed experts owned by this rank (unique per rank; NOT reduced
    across the EP group during backward)."""
    ps = []
    for _, mlp in _iter_moe_modules(model):
        for e in mlp.experts:
            if e is not None:
                ps += list(e.parameters())
    return ps


def nonexpert_parameters(model):
    """All trainable params EXCEPT the routed experts — these are replicated across
    ranks and must be all-reduced during backward (CP/DP)."""
    expert_ids = set()
    for _, mlp in _iter_moe_modules(model):
        for e in mlp.experts:
            if e is not None:
                expert_ids.update(id(p) for p in e.parameters())
    return [p for p in model.parameters() if p.requires_grad and id(p) not in expert_ids]
