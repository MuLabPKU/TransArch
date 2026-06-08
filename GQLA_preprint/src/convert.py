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
    assemble_per_group_covs_from_full,
    collect_full_kv_grams,
    collect_kv_grams,
    compose_compressed_with_perm,
    compress_and_absorb,
    compute_head_similarity,
    compute_token_weights_nll,
    diagnose_weights,
    evaluate_ppl,
    fit_per_group_pca,
    gather_calibration_hidden_states,
    greedy_balanced_grouping,
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
    p.add_argument("--head_grouping", choices=["neighbor", "similarity"], default="neighbor",
                   help="Per-layer head grouping for PCA. 'neighbor' (default): heads "
                        "0..gs-1, gs..2gs-1, ... — byte-identical to legacy. "
                        "'similarity': full all-pair K/V cov + trace-normalised nuclear-norm "
                        "head similarity + seed-and-grow greedy balanced grouping. Costs an "
                        "extra O(H^2 d^2) per-layer cov collection but typically reduces PCA "
                        "truncation error noticeably.")
    p.add_argument("--sim_w_k", type=float, default=1.0,
                   help="K-side weight in nuclear-norm head similarity. w_k=1, w_v=0 groups "
                        "by K subspace only (default 1.0).")
    p.add_argument("--sim_w_v", type=float, default=1.0,
                   help="V-side weight in nuclear-norm head similarity (default 1.0).")
    p.add_argument("--hessian_pca", action="store_true", default=False,
                   help="Weight the calibration cov by per-token NLL from the teacher's "
                        "next-token prediction (SparseGPT/OBS-style Hessian-weighted PCA): "
                        "rare / surprising tokens contribute more than easy / predictable "
                        "ones. Requires one extra norm + lm_head forward over the cached "
                        "final hidden states; OFF path is byte-identical to legacy.")
    p.add_argument("--hessian_mode", choices=["nll"], default="nll",
                   help="Weighting scheme when --hessian_pca is set. 'nll' (default): per-"
                        "token NLL of the next token. Other modes from the research codebase "
                        "are not exposed in opensource.")
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
    if args.hessian_pca:
        calib, final_hiddens = gather_calibration_hidden_states(
            model, batches, input_device, capture_final=True,
        )
    else:
        calib = gather_calibration_hidden_states(model, batches, input_device)
        final_hiddens = None

    # Hessian-aware token weights (NLL of next token from the teacher's lm_head).
    # Computed once on the cached final hidden states; threaded into every layer's
    # cov accumulation. When --hessian_pca is OFF, token_weights stays None and the
    # cov path is byte-identical to the legacy code.
    token_weights = None
    if args.hessian_pca:
        print(f"[hessian_pca] computing per-token NLL weights ({args.hessian_mode}) ...")
        token_weights = compute_token_weights_nll(model, final_hiddens, batches)
        stats = diagnose_weights(token_weights)
        print(
            f"[hessian_pca] mode={args.hessian_mode}, token weight stats: "
            f"mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
            f"min={stats['min']:.3f}, max={stats['max']:.3f} "
            f"over {stats['n_tokens']} tokens"
        )
        del final_hiddens   # free CPU memory; weights are all we need

    print(
        f"\nCompressing layers in place "
        f"(head_grouping={args.head_grouping}, hessian_pca={args.hessian_pca}) ..."
    )
    for li, layer in enumerate(tqdm(model.model.layers, desc="Compress", unit="layer")):
        attn = layer.self_attn
        hidden_list = calib[li]
        if args.head_grouping == "similarity":
            full_cov_K, full_cov_V = collect_full_kv_grams(
                attn, hidden_list, layout, token_weights=token_weights,
            )
            sim = compute_head_similarity(
                full_cov_K, full_cov_V, layout, w_k=args.sim_w_k, w_v=args.sim_w_v,
            )
            groups = greedy_balanced_grouping(sim, layout.num_kv_heads, layout.group_size)
            perm = [h for grp in groups for h in grp]
            cov_k, cov_v = assemble_per_group_covs_from_full(
                full_cov_K, full_cov_V, perm, layout,
            )
            del full_cov_K, full_cov_V
            u_k = fit_per_group_pca(cov_k, layout.qk_nope)
            u_v = fit_per_group_pca(cov_v, layout.v_dim)
            compressed = compose_compressed_with_perm(attn, layout, groups, u_k, u_v)
        else:
            cov_k, cov_v = collect_kv_grams(
                attn, hidden_list, layout, token_weights=token_weights,
            )
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

    method = f"pca_{args.head_grouping}_independent"
    if args.hessian_pca:
        method += f"_hess_{args.hessian_mode}"
    meta = {
        "num_kv_heads":  args.num_kv_heads,
        "group_size":    layout.group_size,
        "method":        method,
        "head_grouping": args.head_grouping,
        "sim_w_k":       args.sim_w_k if args.head_grouping == "similarity" else None,
        "sim_w_v":       args.sim_w_v if args.head_grouping == "similarity" else None,
        "hessian_pca":   args.hessian_pca,
        "hessian_mode":  args.hessian_mode if args.hessian_pca else None,
        "cal_dataset":   args.cal_dataset,
        "cal_nsamples":  args.cal_nsamples,
        "cal_seqlen":    args.cal_seqlen,
        "seed":          args.seed,
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
