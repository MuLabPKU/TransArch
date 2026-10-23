"""``python -m periodic_weak_spots.niah_0p6b {build,score}``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m periodic_weak_spots.niah_0p6b")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build one suite as an HF dataset")
    build.add_argument("--config", required=True, help="configs/niah_0p6b/<suite>.json")
    build.add_argument("--output", required=True)
    build.add_argument("--cache-dir")
    build.add_argument("--workers", type=int)
    score = commands.add_parser("score", help="score a Qwen3-KVC checkpoint")
    score.add_argument("--suite", required=True, help="directory written by build")
    score.add_argument("--model", required=True, help="checkpoint directory")
    score.add_argument("--device", default="cuda")
    score.add_argument("--dtype", default="float32")
    score.add_argument("--batch-size", type=int, default=8)
    score.add_argument("--output", type=Path, help="write the summary JSON here")
    args = parser.parse_args(argv)

    if args.command == "build":
        from periodic_weak_spots.tokenization import load_tokenizer

        from .build import build_suite
        from .config import load_suite_config

        spec = load_suite_config(args.config).tokenizer
        tokenizer = load_tokenizer(spec.tokenizer_id, spec.revision, cache_dir=args.cache_dir)
        manifest = build_suite(
            args.config,
            args.output,
            tokenizer=tokenizer,
            cache_dir=args.cache_dir,
            workers=args.workers,
        )
        print(f"wrote {manifest.row_count} rows of {manifest.suite_id} to {args.output}")
        return

    import torch
    from transformers import AutoModelForCausalLM

    import periodic_weak_spots.model  # noqa: F401  (registers qwen3_kvc)

    from .build import load_suite
    from .score_hf import score_rows, summarize

    rows, manifest = load_suite(args.suite)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    model.to(args.device)
    summary = summarize(rows, score_rows(model, rows, batch_size=args.batch_size), manifest)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
