#!/usr/bin/env python3
"""Convert GLM-4.7-Flash (MLA) to GQLA via per-group K/V PCA + absorption.

Data-driven: a single MLA forward over the calibration set captures every
layer's attention input; per layer we then fit PCA on the per-group K/V
covariance and absorb the rotations into Q (nope rows) and O (per-head v cols).
``kv_b_proj`` shrinks from ``H * (qk_nope + v_dim)`` to ``G * (qk_nope + v_dim)``
rows; Q and O shapes are unchanged.

Multi-GPU: ``device_map="auto"`` shards layers across visible GPUs at load.
The calibration forward, the per-layer PCA fit, and the in-place mutation all
run on whatever device each layer sits on; the optional PPL eval forwards
through the sharded model normally.

Example::

    python -m src.convert \\
        --model_path huggingface/zai-org/GLM-4.7-Flash \\
        --save_path  outputs/glm47-gqla-g4 \\
        --num_kv_heads 4 --dtype bf16 \\
        --cal_dataset wikitext2 --cal_nsamples 128 --cal_seqlen 512 \\
        --eval_ppl --eval_ppl_dataset wikitext2 --eval_ppl_seqlen 2048
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compression import (
    GqlaLayout,
    collect_kv_grams,
    compress_and_absorb,
    evaluate_ppl,
    fit_per_group_pca,
    gather_calibration_hidden_states,
    inplace_apply_compressed,
    prepare_calibration_inputs,
    prepare_ppl_dataloader,
)


_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args():
    p = argparse.ArgumentParser(description="GLM-4.7-Flash MLA -> GQLA via PCA + absorption")
    p.add_argument("--model_path", required=True)
    p.add_argument("--save_path",  required=True)
    p.add_argument("--num_kv_heads", type=int, required=True,
                   help="GQA group count; must divide num_attention_heads.")
    p.add_argument("--dtype", choices=list(_DTYPES), default="bf16")
    p.add_argument("--device_map", default="auto",
                   help="HF device_map ('auto' shards across visible GPUs).")
    p.add_argument("--cal_dataset", choices=["wikitext2", "pg19", "alpaca"], default="wikitext2")
    p.add_argument("--cal_nsamples", type=int, default=128)
    p.add_argument("--cal_seqlen",   type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_ppl", action="store_true")
    p.add_argument("--eval_ppl_dataset", choices=["wikitext2", "pg19", "alpaca"], default="wikitext2")
    p.add_argument("--eval_ppl_seqlen",  type=int, default=2048)
    p.add_argument("--eval_ppl_batch_size", type=int, default=1)
    p.add_argument("--eval_ppl_max_chunks", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = _DTYPES[args.dtype]

    print(f"Loading {args.model_path} (dtype={args.dtype}, device_map={args.device_map}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, device_map=args.device_map, trust_remote_code=False,
    )
    model.eval()
    config = model.config
    layout = GqlaLayout.from_config(config, args.num_kv_heads)
    print(
        f"H={layout.num_heads} heads -> G={layout.num_kv_heads} KV groups "
        f"(group_size={layout.group_size}); KV cache reduction = {layout.group_size}x"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)

    ppl_loader = ppl_orig = ppl_gqla = None
    if args.eval_ppl:
        print(f"Building PPL loader ({args.eval_ppl_dataset}, seqlen={args.eval_ppl_seqlen}) ...")
        ppl_loader = prepare_ppl_dataloader(
            tokenizer, args.eval_ppl_dataset, args.eval_ppl_seqlen,
            args.eval_ppl_batch_size, max_chunks=args.eval_ppl_max_chunks,
        )
        print("Eval PPL on original MLA model ...")
        ppl_orig = evaluate_ppl(model, ppl_loader)
        print(f"  Original MLA PPL = {ppl_orig:.4f}")

    print(
        f"\nBuilding calibration ({args.cal_dataset}, n={args.cal_nsamples}, "
        f"len={args.cal_seqlen}) ..."
    )
    batches = prepare_calibration_inputs(
        tokenizer, args.cal_dataset, args.cal_nsamples, args.cal_seqlen, seed=args.seed,
    )

    # First-rank device for input_ids; HF device_map auto-routes through the rest.
    input_device = next(model.parameters()).device
    print(f"Gathering per-layer hidden states ({config.num_hidden_layers} layers) ...")
    calib = gather_calibration_hidden_states(model, batches, input_device)

    print("\nCompressing layers in place ...")
    for li, layer in enumerate(tqdm(model.model.layers, desc="Compress", unit="layer")):
        attn = layer.self_attn
        hidden_list = calib[li]
        cov_k, cov_v = collect_kv_grams(attn, hidden_list, layout)
        u_k = fit_per_group_pca(cov_k, layout.qk_nope)
        u_v = fit_per_group_pca(cov_v, layout.v_dim)
        compressed = compress_and_absorb(attn, layout, u_k, u_v)
        inplace_apply_compressed(attn, compressed, layout)
        calib[li] = []   # free CPU memory as we go

    config.num_key_value_heads = args.num_kv_heads

    if args.eval_ppl:
        print("\nEval PPL on converted GQLA model ...")
        ppl_gqla = evaluate_ppl(model, ppl_loader)
        delta = ppl_gqla - ppl_orig
        pct = 100.0 * (ppl_gqla / ppl_orig - 1.0)
        print(f"  Converted GQLA PPL = {ppl_gqla:.4f}  (Δ = {delta:+.4f}, {pct:+.2f}%)")

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {save_path} ...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Bundle modeling.py so reload with trust_remote_code=True picks up the GQLA class.
    pkg_root = Path(__file__).resolve().parent
    (save_path / "modeling.py").write_bytes((pkg_root / "modeling.py").read_bytes())

    # Patch config.json: keep upstream model_type, override architectures + auto_map.
    cfg_path = save_path / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["architectures"] = ["Glm4MoeLiteGQLAForCausalLM"]
    cfg["auto_map"] = {"AutoModelForCausalLM": "modeling.Glm4MoeLiteGQLAForCausalLM"}
    cfg_path.write_text(json.dumps(cfg, indent=2))

    meta = {
        "num_kv_heads": args.num_kv_heads,
        "group_size":   layout.group_size,
        "method":       "pca_neighbor_independent",
        "cal_dataset":  args.cal_dataset,
        "cal_nsamples": args.cal_nsamples,
        "cal_seqlen":   args.cal_seqlen,
        "seed":         args.seed,
    }
    if args.eval_ppl:
        meta["ppl_orig"] = ppl_orig
        meta["ppl_gqla"] = ppl_gqla
        meta["ppl_eval_config"] = {
            "dataset": args.eval_ppl_dataset, "seqlen": args.eval_ppl_seqlen,
            "batch_size": args.eval_ppl_batch_size,
        }
    (save_path / "gqla_meta.json").write_text(json.dumps(meta, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
