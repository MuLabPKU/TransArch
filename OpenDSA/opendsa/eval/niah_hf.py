"""niah_hf.py — needle-in-a-haystack eval running OpenDSA's REAL DSA sparse forward.

This is the correctness-first fallback eval (plan P1 item 6): it does NOT depend on
sglang accepting a converted V2-Lite checkpoint. It loads the base DeepSeek-V2-Lite
+ the warmup-trained indexer weights, patches attention to DSA, and measures whether
the model can retrieve a "needle" (a random passkey) buried at various depths in a
long distractor "haystack" — the canonical long-context probe.

Two modes are compared on the SAME items so the number is interpretable:
  * dense  : pristine MLA attention (mode="dense") — the base-model ceiling
  * sparse : DSA top-k sparse attention (mode="sparse") — what training produced

We score exact-match retrieval of the passkey digits from a short greedy decode.

Usage:
    python -m opendsa.eval.niah_hf \
        --model-dir runs/warmup/final \
        --indexer runs/warmup/indexer.pt \
        --lengths 2048,4096,8192 --depths 0.1,0.5,0.9 --topk 2048 \
        --num-layers 27 --out eval_results/niah_hf.json

Notes:
  * model-dir may be the base HF id or any consolidated HF checkpoint (we only need
    the backbone weights; the indexer is loaded separately from --indexer).
  * runs on a single GPU; keep lengths modest for a quick, real number.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from ..modeling import (patch_model_with_dsa, IndexerConfig, set_dsa_mode)
from ..train.convert_stage import load_indexer


HAYSTACK = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    "There and back again. The city was quiet in the early morning light. "
    "People walked slowly to work, coffee in hand, thinking of the day ahead. "
)
NEEDLE = "The special passkey is {key}. Remember it well."
QUESTION = "\n\nWhat is the special passkey mentioned in the text above? The passkey is"


def _build_prompt(tok, length_tokens, depth, key):
    """Build a prompt of ~length_tokens with the needle at fractional `depth`."""
    filler_ids = tok(HAYSTACK, add_special_tokens=False)["input_ids"]
    needle_ids = tok(NEEDLE.format(key=key), add_special_tokens=False)["input_ids"]
    q_ids = tok(QUESTION, add_special_tokens=False)["input_ids"]
    budget = max(length_tokens - len(needle_ids) - len(q_ids), 0)
    # tile filler to budget
    reps = budget // max(len(filler_ids), 1) + 1
    body = (filler_ids * reps)[:budget]
    cut = int(len(body) * depth)
    ids = body[:cut] + needle_ids + body[cut:] + q_ids
    return torch.tensor([ids], dtype=torch.long)


@torch.no_grad()
def _greedy(model, input_ids, max_new=8):
    """Minimal greedy decode (no cache; DSA forward doesn't implement KV cache)."""
    dev = next(model.parameters()).device
    ids = input_ids.to(dev)
    out = []
    for _ in range(max_new):
        logits = model(input_ids=ids, use_cache=False).logits[:, -1, :]
        nxt = int(logits.argmax(-1))
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
    return out


def run(args):
    dev = "cuda"
    dtype = torch.bfloat16
    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    if args.num_layers > 0:
        cfg.num_hidden_layers = args.num_layers
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model_dir,
                                        trust_remote_code=True)
    print(f"[niah] loading model from {args.model_dir} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, config=cfg, trust_remote_code=True,
        attn_implementation="eager", torch_dtype=dtype).to(dev).eval()
    patch_model_with_dsa(
        model, IndexerConfig(n_heads=args.idx_heads, head_dim=args.idx_head_dim,
                             topk=args.topk, tile=args.tile),
        mode="dense", dtype=dtype)
    if args.indexer and os.path.exists(args.indexer):
        load_indexer(model, args.indexer, strict=True)
        print(f"[niah] loaded trained indexer from {args.indexer}", flush=True)
    else:
        print("[niah] WARNING: no indexer weights loaded (random indexer)", flush=True)

    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [float(x) for x in args.depths.split(",")]
    keys = [str(1000 + 137 * i % 8999) for i in range(args.samples)]

    results = {"model_dir": args.model_dir, "indexer": args.indexer,
               "topk": args.topk, "num_layers": cfg.num_hidden_layers, "cases": []}

    for mode in args.modes.split(","):
        set_dsa_mode(model, mode)
        for L in lengths:
            for d in depths:
                hits, tot, t0 = 0, 0, time.time()
                for key in keys:
                    prompt = _build_prompt(tok, L, d, key)
                    gen = _greedy(model, prompt, max_new=args.max_new)
                    text = tok.decode(gen)
                    found = re.search(r"\d{3,5}", text)
                    ok = bool(found and found.group() == key)
                    hits += int(ok); tot += 1
                acc = hits / max(tot, 1)
                dt = time.time() - t0
                rec = {"mode": mode, "length": L, "depth": d,
                       "acc": acc, "n": tot, "sec": round(dt, 1)}
                results["cases"].append(rec)
                print(f"[niah] mode={mode} L={L} depth={d} acc={acc:.2f} "
                      f"({hits}/{tot}) {dt:.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[niah] wrote {args.out}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--indexer", default="runs/warmup/indexer.pt")
    ap.add_argument("--lengths", default="2048,4096,8192")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--modes", default="dense,sparse")
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--idx-heads", type=int, default=8)
    ap.add_argument("--idx-head-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=-1)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--out", default="eval_results/niah_hf.json")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
