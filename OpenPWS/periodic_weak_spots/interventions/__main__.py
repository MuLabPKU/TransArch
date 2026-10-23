"""``python -m periodic_weak_spots.interventions <command>``.

Qwen3-KVC: ``head-sweep`` and ``gate-cycling``. DeepSeek-V4 (launch with
``torchrun``): ``suffix-means``, ``suffix-sweep``, and ``suffix-summary``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DTYPES = ("float32", "bfloat16", "float16")


def _write(value: Any, output: Path | None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


def _save_atomic(value: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_model(path: str, device: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM

    import periodic_weak_spots.model  # noqa: F401  (registers qwen3_kvc)

    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype))
    return model.to(device).eval()


def _head_means(args: argparse.Namespace, model, rows) -> Any:
    """Fit means on the calibration suite, or reuse a file fitted for the same inputs."""

    import torch

    from periodic_weak_spots.niah_0p6b.build import load_suite

    from .head_patching import fit_head_means

    calibration, manifest = load_suite(args.calibration)
    if {row.token_ids for row in calibration} & {row.token_ids for row in rows}:
        raise SystemExit("calibration and evaluation prompts must be disjoint")
    # Keyed on checkpoint and calibration suite, so `--layers` splits reuse one fit.
    provenance = {"model": str(Path(args.model).resolve()), "calibration": manifest.suite_id}
    if args.means is not None and args.means.exists():
        saved = torch.load(args.means)
        if saved["provenance"] != provenance:
            raise SystemExit(f"{args.means} was fitted for {saved['provenance']}, not {provenance}")
        return saved["means"]
    means = fit_head_means(model, calibration, batch_size=args.batch_size)
    if args.means is not None:
        _save_atomic({"provenance": provenance, "means": means}, args.means)
    return means


def _digest(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(token_ids)).encode()).hexdigest()


def _shard(text: str) -> tuple[int, int]:
    index, _, count = text.partition("/")
    try:
        shard = int(index), int(count)
    except ValueError:
        shard = (-1, 0)
    if not 0 <= shard[0] < shard[1]:
        raise argparse.ArgumentTypeError("shards look like INDEX/COUNT, e.g. 0/12")
    return shard


def _suffix_prompts(args: argparse.Namespace, root: Path):
    """Suffix prompts of a built 128K dataset and their leading special token IDs.

    Prompts are tokenized as pinned in the dataset metadata, with the tokenizer's default
    leading special tokens unless ``--no-special-tokens``.
    """

    from periodic_weak_spots.niah_128k.config import ExperimentConfig
    from periodic_weak_spots.tokenization import leading_special_ids, load_tokenizer

    from .suffix_patching import load_suffix_prompts

    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    spec = ExperimentConfig.from_dict(metadata["config"]).tokenizer
    tokenizer = load_tokenizer(spec.model_id, spec.revision, local_path=args.tokenizer_path)
    leading = [] if args.no_special_tokens else list(leading_special_ids(tokenizer))
    prompts = load_suffix_prompts(
        root, tokenizer, rows_per_residue=args.rows_per_residue, leading_ids=leading
    )
    return prompts, leading


def _suffix_command(args: argparse.Namespace) -> None:
    """DeepSeek-V4 stages; every rank joins each expert-parallel forward, rank zero writes."""

    import torch
    import torch.distributed as dist

    from .suffix_patching import evaluate_prompt, fit_suffix_means, load_model

    model = load_model(args.model)
    main = dist.get_rank() == 0
    if args.command == "suffix-means":
        prompts, leading = _suffix_prompts(args, args.calibration)
        means = fit_suffix_means(model, prompts, chunk_size=args.chunk_size)
        if main:
            # Digests let suffix-sweep check disjointness without the calibration dataset.
            _save_atomic(
                {
                    "model": args.model,
                    "calibration": str(args.calibration),
                    "leading_special_ids": leading,
                    "rows": len(prompts),
                    "prompt_digests": sorted(_digest(p.token_ids) for p in prompts),
                    "means": means.cpu(),
                },
                args.output,
            )
    else:
        saved = torch.load(args.means)
        if saved["model"] != args.model:
            raise SystemExit(f"{args.means} was fitted for {saved['model']}, not {args.model}")
        prompts, leading = _suffix_prompts(args, args.evaluation)
        # A means file without the field was fitted without special tokens.
        if saved.get("leading_special_ids", []) != leading:
            raise SystemExit(
                f"{args.means} was fitted with leading special tokens "
                f"{saved.get('leading_special_ids', [])}, not {leading}; "
                "match --no-special-tokens between suffix-means and suffix-sweep"
            )
        if {_digest(p.token_ids) for p in prompts} & set(saved["prompt_digests"]):
            raise SystemExit("calibration and evaluation prompts must be disjoint")
        index, count = args.shard
        # Every rank reads the same finished IDs, so all ranks skip the same prompts and the
        # collective forwards stay in lockstep.
        done = set()
        if args.output.exists():
            with args.output.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if record.get("leading_special_ids", []) != leading:
                        raise SystemExit(f"{args.output} was written with other special tokens")
                    done.add(record["sample_id"])
        means = saved["means"].to(model.device)
        # Prompts are sorted by residue, so strided shards get near-equal counts per residue.
        for prompt in prompts[index::count]:
            if prompt.sample_id in done:
                continue
            record = evaluate_prompt(
                model, prompt, means, layers=args.layers, chunk_size=args.chunk_size
            )
            record["leading_special_ids"] = leading
            if main:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
    dist.barrier()
    dist.destroy_process_group()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m periodic_weak_spots.interventions")
    commands = parser.add_subparsers(dest="command", required=True)

    sweep = commands.add_parser("head-sweep", help="Qwen3-KVC KV-head knockouts")
    sweep.add_argument("--model", required=True, help="Qwen3-KVC checkpoint directory")
    sweep.add_argument("--calibration", required=True, help="calibration suite directory")
    sweep.add_argument("--suite", required=True, help="evaluation suite directory")
    sweep.add_argument("--means", type=Path, help="reuse or save fitted means (.pt)")
    sweep.add_argument("--layers", type=int, nargs="+", help="default: every layer")

    cycling = commands.add_parser("gate-cycling", help="Qwen3-KVC gate cycling")
    cycling.add_argument("--model", required=True, help="Qwen3-KVC checkpoint directory")
    cycling.add_argument("--suite", required=True, help="evaluation suite directory")
    cycling.add_argument("--config", type=Path, required=True, help="configs/interventions/*.json")

    for command in (sweep, cycling):
        command.add_argument("--device", default="cuda")
        command.add_argument("--dtype", choices=DTYPES, default="float32")
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--output", type=Path)

    means = commands.add_parser("suffix-means", help="DeepSeek-V4 suffix calibration means")
    means.add_argument("--calibration", type=Path, required=True, help="built 128K dataset")
    means.add_argument("--output", type=Path, required=True, help="means file (.pt)")

    suffix = commands.add_parser("suffix-sweep", help="DeepSeek-V4 query-suffix knockouts")
    suffix.add_argument("--evaluation", type=Path, required=True, help="built 128K dataset")
    suffix.add_argument("--means", type=Path, required=True, help="written by suffix-means")
    suffix.add_argument("--layers", type=int, nargs="+", help="default: every layer")
    suffix.add_argument("--shard", type=_shard, default=(0, 1), help="INDEX/COUNT of the prompts")
    suffix.add_argument("--output", type=Path, required=True, help="per-prompt JSONL (resumes)")

    for command in (means, suffix):
        command.add_argument("--model", required=True, help="DeepSeek-V4 checkpoint or Hub ID")
        command.add_argument("--rows-per-residue", type=int, help="first N sample IDs per residue")
        command.add_argument("--chunk-size", type=int, default=1024, help="prefill chunk length")
        command.add_argument("--tokenizer-path", type=Path, help="local copy of the tokenizer")
        command.add_argument(
            "--no-special-tokens",
            action="store_true",
            help="omit the tokenizer's default leading special tokens, which vLLM adds",
        )

    summary = commands.add_parser("suffix-summary", help="merge suffix-sweep outputs")
    summary.add_argument("--rows", type=Path, nargs="+", required=True, help="suffix-sweep JSONL")
    summary.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.command in ("suffix-means", "suffix-sweep"):
        _suffix_command(args)
        return
    if args.command == "suffix-summary":
        from .suffix_patching import summarize

        # Keyed by sample ID, so prompts repeated across shard or resumed files count once.
        records = {}
        for path in args.rows:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    records[record["sample_id"]] = record
        _write(summarize(list(records.values())), args.output)
        return

    from periodic_weak_spots.niah_0p6b.build import load_suite

    model = _load_model(args.model, args.device, args.dtype)
    rows, manifest = load_suite(args.suite)
    if args.command == "gate-cycling":
        from .gate_cycling import GateCycle, evaluate_gate_cycles

        config = json.loads(args.config.read_text(encoding="utf-8"))
        cycles = [GateCycle.from_dict(value) for value in config["cycles"]]
        result = evaluate_gate_cycles(model, rows, manifest, cycles, batch_size=args.batch_size)
    else:
        from .head_patching import sweep_kv_heads

        means = _head_means(args, model, rows)
        result = sweep_kv_heads(
            model, rows, manifest, means, layers=args.layers, batch_size=args.batch_size
        )
    _write({"model": args.model, "suite_id": manifest.suite_id, **result}, args.output)


if __name__ == "__main__":
    main()
