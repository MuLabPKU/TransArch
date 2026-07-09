"""longbench_hf.py — LongBench v2 (multiple-choice PPL) running OpenDSA's REAL DSA
sparse forward. Companion to niah_hf.py; same HF path, no sglang dependency.

LongBench v2 is a 4-way multiple-choice long-context QA benchmark. This script
expects length-bucketed, pre-rendered PPL files:

    LongBenchv2_{16k,32k,64k,...}_ppl.jsonl

Each record already contains a fully-rendered ``question`` prompt that ends with
``"... The correct answer is"``; ``choices`` is ``[" (A)"," (B)"," (C)"," (D)"]``
and ``answer`` is the gold letter (``"A".."D"``). This is the standard
``packed_multi_choice_ppl`` task: pick the choice with the highest continuation
likelihood given the prompt.

Scoring (faithful + cheap). Every choice tokenizes as ``" (X)" -> [tok(" ("),
tok(X), tok(")")]`` — the ``" ("`` prefix and ``")"`` suffix are identical across
the four choices, so the only term that discriminates the argmax is
``log p(letter | prompt + " (")``. We therefore run ONE forward on
``prompt + " ("`` per question and compare the four single-token letter logits.
This is exact for the argmax (the shared tokens cancel) and 4x cheaper than
scoring each choice with a separate forward. Metric = accuracy.

Dense (pristine MLA) vs sparse (DSA top-k) are scored on the SAME items so the
long-context number is directly comparable, exactly like niah_hf.py.

Usage:
    python -m opendsa.eval.longbench_hf \
        --model-dir runs/warmup/final --indexer runs/warmup/indexer.pt \
        --data-dir /path/to/longbenchv2 \
        --lengths 16k,32k --modes dense,sparse --samples 20 --topk 2048 \
        --out eval_results/longbench_hf.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from ..modeling import (patch_model_with_dsa, IndexerConfig, set_dsa_mode)
from ..train.convert_stage import load_indexer

DEFAULT_DATA_DIR = os.environ.get("OPENDSA_LONGBENCH_DIR", "data/longbenchv2")
LETTERS = ["A", "B", "C", "D"]


def _load_records(data_dir, length_tag, samples):
    """Read one length bucket and return raw LongBench records."""
    path = os.path.join(data_dir, f"LongBenchv2_{length_tag}_ppl.jsonl")
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
            if samples and len(recs) >= samples:
                break
    return recs


@torch.no_grad()
def _score_choice_logits(model, tok, prompt, letter_ids, prefix_ids, max_ctx):
    """Single forward on prompt + ' (' ; return log-softmax logits for the 4 letters.
    Left-truncates the prompt to fit max_ctx (keeps the tail, where the question is,
    plus enough head — LongBench answers can be anywhere, so we keep the whole thing
    up to max_ctx from the RIGHT is wrong; we keep head+tail. Simpler: truncate head
    only when over budget, preserving the question at the end)."""
    dev = next(model.parameters()).device
    ids = tok(prompt, add_special_tokens=True)["input_ids"]
    budget = max_ctx - len(prefix_ids)
    if len(ids) > budget:
        # keep the tail (question + format instruction live at the end)
        ids = ids[-budget:]
    ids = ids + prefix_ids
    inp = torch.tensor([ids], dtype=torch.long, device=dev)
    logits = model(input_ids=inp, use_cache=False).logits[:, -1, :].float()
    logp = torch.log_softmax(logits, dim=-1)[0]
    return [float(logp[i]) for i in letter_ids]


def run(args):
    dev = "cuda"
    dtype = torch.bfloat16
    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    if args.num_layers > 0:
        cfg.num_hidden_layers = args.num_layers
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model_dir,
                                        trust_remote_code=True)
    letter_ids = [tok(l, add_special_tokens=False)["input_ids"][0] for l in LETTERS]
    prefix_ids = tok(" (", add_special_tokens=False)["input_ids"]

    print(f"[longbench] loading model from {args.model_dir} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, config=cfg, trust_remote_code=True,
        attn_implementation="eager", torch_dtype=dtype).to(dev).eval()
    patch_model_with_dsa(
        model, IndexerConfig(n_heads=args.idx_heads, head_dim=args.idx_head_dim,
                             topk=args.topk, tile=args.tile),
        mode="dense", dtype=dtype)
    if args.indexer and os.path.exists(args.indexer):
        load_indexer(model, args.indexer, strict=True)
        print(f"[longbench] loaded trained indexer from {args.indexer}", flush=True)
    else:
        print("[longbench] WARNING: no indexer weights loaded (random indexer)", flush=True)

    results = {"model_dir": args.model_dir, "indexer": args.indexer,
               "topk": args.topk, "num_layers": cfg.num_hidden_layers,
               "max_ctx": args.max_ctx, "cases": []}

    for length_tag in args.lengths.split(","):
        recs = _load_records(args.data_dir, length_tag, args.samples)
        for mode in args.modes.split(","):
            set_dsa_mode(model, mode)
            hits, tot, t0, errs = 0, 0, time.time(), 0
            for rec in recs:
                gold = rec["answer"].strip().upper()
                if gold not in LETTERS:
                    continue
                try:
                    scores = _score_choice_logits(model, tok, rec["question"],
                                                  letter_ids, prefix_ids, args.max_ctx)
                except torch.cuda.OutOfMemoryError:
                    errs += 1
                    torch.cuda.empty_cache()
                    continue
                pred = LETTERS[int(max(range(4), key=lambda i: scores[i]))]
                hits += int(pred == gold); tot += 1
            acc = hits / max(tot, 1)
            dt = time.time() - t0
            rec_out = {"mode": mode, "length": length_tag, "acc": round(acc, 4),
                       "n": tot, "oom": errs, "sec": round(dt, 1)}
            results["cases"].append(rec_out)
            print(f"[longbench] mode={mode} L={length_tag} acc={acc:.3f} "
                  f"({hits}/{tot}) oom={errs} {dt:.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[longbench] wrote {args.out}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--indexer", default="runs/warmup/indexer.pt")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="directory containing LongBenchv2_{length}_ppl.jsonl files")
    ap.add_argument("--lengths", default="16k,32k",
                    help="comma list of LongBench v2 buckets: 16k,32k,64k,128k,256k,512k")
    ap.add_argument("--modes", default="dense,sparse")
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--idx-heads", type=int, default=8)
    ap.add_argument("--idx-head-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=-1)
    ap.add_argument("--samples", type=int, default=20, help="cap records per bucket")
    ap.add_argument("--max-ctx", type=int, default=32768,
                    help="left-truncate prompts longer than this (fits GPU memory)")
    ap.add_argument("--out", default="eval_results/longbench_hf.json")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
