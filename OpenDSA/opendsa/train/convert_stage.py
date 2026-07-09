"""convert_stage.py — carry warmup-trained indexer weights into the sparse stage.

In OpenDSA the indexer module is attached identically in both stages (unlike the
Megatron reference which uses different key prefixes for warmup vs train), so the
"conversion" is simply: load the warmup checkpoint, keep the indexer weights, and
switch the model to sparse mode. This utility extracts just the indexer weights
from a warmup checkpoint into a small file, and loads them onto a fresh patched
model for sparse training.
"""
from __future__ import annotations

import os
import shutil

import torch
import torch.distributed as dist

from ..modeling import patch_model_with_dsa, IndexerConfig, set_dsa_mode


def extract_indexer_state(model):
    """Return {layer_idx: state_dict} for all indexers."""
    from ..modeling.patch_deepseek import _iter_attn_modules
    return {i: attn.indexer.state_dict() for i, attn in _iter_attn_modules(model)}


def save_indexer(model, path):
    torch.save(extract_indexer_state(model), path)
    print(f"[convert_stage] saved indexer weights -> {path}")


def load_indexer(model, path, strict=True):
    from ..modeling.patch_deepseek import _iter_attn_modules
    state = torch.load(path, map_location="cpu")
    n = 0
    for i, attn in _iter_attn_modules(model):
        if i in state:
            attn.indexer.load_state_dict(state[i], strict=strict)
            n += 1
    print(f"[convert_stage] loaded indexer weights into {n} layers from {path}")
    return model


def _unwrap_model(model):
    node = model
    for _ in range(8):
        if hasattr(node, "save_pretrained"):
            return node
        nxt = getattr(node, "module", None) or getattr(node, "model", None)
        if nxt is None:
            break
        node = nxt
    return model


def _is_expert_key(name: str) -> bool:
    return ".mlp.experts." in name


def save_ep_merged_model(model, output_dir: str, tokenizer=None, tmp_dir: str | None = None):
    """Save a full HF checkpoint from an EP-sharded model.

    Under EP, each rank owns only a slice of routed experts and non-owned experts are
    set to None. Rank 0 therefore cannot directly save a complete checkpoint. Each
    rank writes its local expert tensors to a temporary part file; rank 0 merges
    those parts with its replicated non-expert tensors and exports a normal
    `save_pretrained` checkpoint.
    """
    from ..dist import ep_size, ep_rank, ep_group

    base = _unwrap_model(model)
    world = ep_size()
    rank = ep_rank()
    if world <= 1:
        base.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        return

    tmp_dir = tmp_dir or os.path.join(os.path.dirname(output_dir), "_ep_save_parts")
    if rank == 0:
        os.makedirs(tmp_dir, exist_ok=True)
    dist.barrier(group=ep_group())

    local_state = base.state_dict()
    expert_state = {
        k: v.detach().cpu()
        for k, v in local_state.items()
        if _is_expert_key(k)
    }
    part_path = os.path.join(tmp_dir, f"rank{rank}.pt")
    torch.save(expert_state, part_path)
    dist.barrier(group=ep_group())

    if rank == 0:
        full_state = {k: v.detach().cpu() for k, v in local_state.items()}
        for r in range(world):
            part = torch.load(os.path.join(tmp_dir, f"rank{r}.pt"), map_location="cpu")
            full_state.update(part)
            del part
        os.makedirs(output_dir, exist_ok=True)
        base.save_pretrained(
            output_dir,
            state_dict=full_state,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[convert_stage] saved EP-merged full model -> {output_dir}")

    dist.barrier(group=ep_group())
