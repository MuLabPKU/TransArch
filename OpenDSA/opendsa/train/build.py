"""build.py — shared model/data/config construction for the DSA training stages."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from datasets import load_from_disk

from ..modeling import (patch_model_with_dsa, IndexerConfig,
                        freeze_backbone_train_indexer, unfreeze_all)

MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite-Chat"


@dataclass
class DSAArgs:
    model_id: str = MODEL_ID
    data_dir: str = ""                 # packed dataset (from long_pack)
    output_dir: str = "runs/dsa"
    stage: str = "warmup"              # warmup | sparse
    # indexer
    idx_heads: int = 8
    idx_head_dim: int = 128
    topk: int = 2048
    tile: int = 512
    q_chunk: int = 512
    indexer_loss_coeff: float = 1.0
    # training
    lr: float = 1e-3                   # warmup indexer lr; sparse overrides lower
    epochs: float = 1.0
    max_steps: int = -1
    per_device_bs: int = 1
    grad_accum: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 5
    save_steps: int = 200
    seq_len: int = 8192
    grad_ckpt: bool = True
    num_layers: int = -1              # -1 = full; else slice for debug
    log_recall_every: int = 0
    log_kl_every: int = 0             # compute true KL diagnostics every N steps
    indexer_init: str = ""            # path to indexer weights to preload (sparse)
    save_final: bool = True           # rank0 saves indexer + HF final after training
    bf16: bool = True


def build_model(args: DSAArgs, mode: str, device: str | None = None):
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    if args.num_layers and args.num_layers > 0:
        cfg.num_hidden_layers = args.num_layers
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, config=cfg, trust_remote_code=True,
        attn_implementation="eager", torch_dtype=dtype,
    )
    if device is not None:
        model = model.to(device)
    icfg = IndexerConfig(n_heads=args.idx_heads, head_dim=args.idx_head_dim,
                         topk=args.topk, tile=args.tile, q_chunk=args.q_chunk,
                         log_recall=(args.log_recall_every > 0))
    patch_model_with_dsa(model, icfg, mode=mode, dtype=dtype)
    if mode == "warmup":
        freeze_backbone_train_indexer(model)
        # NOTE: no gradient checkpointing in warmup — the backbone is frozen (its
        # activations aren't needed for backbone grads) and FlashKL keeps the
        # teacher in no_grad, so memory is already O(L·H). Grad-checkpointing a
        # frozen module also detaches the graph and breaks indexer grads.
    else:
        unfreeze_all(model)
        if args.grad_ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def build_tokenizer(args: DSAArgs):
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_packed(args: DSAArgs):
    return load_from_disk(args.data_dir)
