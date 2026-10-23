"""``python -m periodic_weak_spots.niah_128k {build,validate,score}``."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# The NIAH haystack of Paul Graham essays used for the paper (not redistributed).
PAPER_FILLER_SHA256 = "51eaf041f492d8fdfdcf87be8756d23888eda0d3cab6816b3c3f11e7682de337"


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m periodic_weak_spots.niah_128k")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="generate a balanced 128K NIAH dataset")
    build.add_argument("--config", type=Path, required=True, help="configs/niah_128k/<role>.json")
    build.add_argument(
        "--filler-file",
        type=Path,
        required=True,
        help="filler text for padding, alignment, and tail",
    )
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--cache-dir")
    build.add_argument("--tokenizer-path", type=Path, help="local copy of the pinned tokenizer")
    build.add_argument("--overwrite", action="store_true")
    validate = commands.add_parser("validate", help="independently retokenize a built dataset")
    validate.add_argument("--dataset", type=Path, required=True, help="directory written by build")
    validate.add_argument("--cache-dir")
    validate.add_argument("--tokenizer-path", type=Path)
    score = commands.add_parser("score", help="greedy vLLM decoding + answer-prefix accuracy")
    score.add_argument("--dataset", type=Path, required=True, help="directory written by build")
    score.add_argument("--model", required=True)
    score.add_argument("--tensor-parallel-size", type=int, default=1)
    score.add_argument("--max-tokens", type=int, default=8)
    score.add_argument("--results", type=Path, help="write completions + summary JSON here")
    args = parser.parse_args(argv)

    # Deferred imports keep each subcommand to its own dependencies (vLLM only for score).
    from periodic_weak_spots.tokenization import load_tokenizer

    from .config import ExperimentConfig, load_config

    if args.command == "score":
        from .score_vllm import generate, load_rows, summarize

        rows = load_rows(args.dataset)
        if args.results:
            # Before generation, so a bad path fails before the long vLLM run.
            args.results.parent.mkdir(parents=True, exist_ok=True)
        completions = generate(
            args.model,
            [row["prompt"] for row in rows],
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        summary = summarize(rows, completions)
        if args.results:
            args.results.write_text(
                json.dumps(
                    {
                        "summary": summary,
                        "completions": {
                            row["sample_id"]: text
                            for row, text in zip(rows, completions, strict=True)
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        _print(summary)
        return

    if args.command == "validate":
        from .validate import validate_dataset

        metadata = json.loads((args.dataset / "metadata.json").read_text(encoding="utf-8"))
        spec = ExperimentConfig.from_dict(metadata["config"]).tokenizer
        tokenizer = load_tokenizer(
            spec.model_id, spec.revision, local_path=args.tokenizer_path, cache_dir=args.cache_dir
        )
        _print(validate_dataset(args.dataset, tokenizer))
        return

    from .bindings import collect_context_stable_token_candidates, generate_bindings
    from .build import publish_dataset
    from .schedule import sentence_capacity_requirements
    from .source import curate_sentence_bins, load_pinned_source

    config = load_config(args.config)
    tokenizer = load_tokenizer(
        config.tokenizer.model_id,
        config.tokenizer.revision,
        local_path=args.tokenizer_path,
        cache_dir=args.cache_dir,
    )
    candidates = collect_context_stable_token_candidates(
        tokenizer,
        min_token_id=config.candidate_pool.min_token_id,
        max_characters=config.candidate_pool.max_characters,
    )
    # One binding per row index; every phase uses the full set, so residues share their records.
    bindings = generate_bindings(
        candidates,
        count=config.phase_design.rows_per_phase,
        pair_count=config.layout.pair_count,
        target_index=config.layout.target_index,
        seed=config.binding_seed,
    )
    bins, source_report = curate_sentence_bins(
        load_pinned_source(config.source, cache_dir=args.cache_dir),
        tokenizer,
        sentence_capacity_requirements(config.phase_design.rows_per_phase),
        spec=config.source,
    )
    filler = args.filler_file.read_bytes()
    metadata = publish_dataset(
        args.output,
        tokenizer,
        bins,
        bindings,
        filler=filler.decode("utf-8"),
        source_report=source_report,
        config=config,
        overwrite=args.overwrite,
    )
    # Any filler builds a valid dataset; the hash only reports fidelity to the paper's prompts.
    filler_sha256 = hashlib.sha256(filler).hexdigest()
    _print(
        {
            "dataset_id": metadata["dataset_id"],
            "total_rows": metadata["matrix"]["total_rows"],
            "filler_sha256": filler_sha256,
            "filler_matches_paper": filler_sha256 == PAPER_FILLER_SHA256,
        }
    )


if __name__ == "__main__":
    main()
