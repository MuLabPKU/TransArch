"""One resumable GLM runner for block-indexer, HISA and DSA, with a shared runtime."""

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

from periodic_weak_spots.niah_128k.score_vllm import load_rows

from .utils import DEFAULT_METRIC, LABELS, METRICS, atomic, data_hashes, read_rows, score, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Result directory")
    parser.add_argument("--variants", nargs="+", choices=LABELS, default=list(LABELS))
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="First N prompts, positive multiple of eight")
    parser.add_argument("--audit", type=Path, help="Default: DATASET/validation.json")
    parser.add_argument(
        "--autotune-config", type=Path, help="Optional hardware-specific FlashInfer table"
    )
    args = parser.parse_args()
    if args.limit is not None and (args.limit < 8 or args.limit % 8):
        parser.error("--limit must be a positive multiple of eight")
    if len(set(args.variants)) != len(args.variants):
        parser.error("--variants must be unique")
    if importlib.metadata.version("vllm") != "0.31.0":
        raise ValueError(
            "This worker patch targets vLLM 0.31.0; revalidate before using another version"
        )
    audit = json.loads((args.audit or args.dataset / "validation.json").read_text())
    hashes = data_hashes(args.dataset)
    if not audit["passed"] or hashes != audit["data_sha256"]:
        raise ValueError("Dataset changed since validation")
    if sha(args.model / "tokenizer.json") != audit["tokenizer_sha256"]:
        raise ValueError("Engine tokenizer differs from the validated dataset")
    rows = sorted(load_rows(args.dataset), key=lambda r: (r["binding_id"], r["phase"]))
    row_map = {r["sample_id"]: r for r in rows}
    if len(row_map) != len(rows):
        raise ValueError("Duplicate dataset IDs")
    selected = rows[: args.limit]
    if not selected:
        raise ValueError("No input rows")
    expected_layers = 11
    options = dict(
        model=str(args.model.resolve()),
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=True,
        max_model_len=128016,
        max_num_seqs=1,
        max_num_batched_tokens=8192,
        block_size=128,
        gpu_memory_utilization=0.92,
        enforce_eager=True,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        seed=33377335,
        worker_extension_cls="extension.worker.Worker",
    )
    if args.autotune_config:
        os.environ["PWS_AUTOTUNE_CONFIG"] = str(args.autotune_config.resolve())
    elif os.environ.get("PWS_AUTOTUNE_CONFIG"):
        parser.error("Pass the inherited PWS_AUTOTUNE_CONFIG explicitly as --autotune-config")
    source = Path(__file__).parent
    identity = {
        "data_sha256": hashes,
        "engine_options": options,
        "versions": {
            n: importlib.metadata.version(n) for n in ("torch", "vllm", "transformers", "gigatoken")
        },
        "model_files_sha256": {
            n: sha(args.model / n)
            for n in (
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
            )
        },
        "source_sha256": {
            name: sha(source / name)
            for name in (
                "score.py",
                "utils.py",
                "prepare.py",
                "worker.py",
                "patch.py",
                "kernels.py",
            )
        },
        "autotune_sha256": sha(args.autotune_config) if args.autotune_config else None,
        "variants": args.variants,
        "temperature": 0,
        "max_tokens": 8,
        "default_metric": DEFAULT_METRIC,
        "order": "binding_id, phase, variant",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    records, done = {}, {}
    for mode in args.variants:
        path = args.output / f"{mode}.jsonl"
        if path.exists():
            previous = json.loads(path.with_suffix(".provenance.json").read_text())
            if previous["identity"] != identity:
                raise ValueError("Resume identity differs; choose a fresh result directory")
            records[mode] = read_rows(path)
        else:
            records[mode] = []
        done[mode] = {r["sample_id"] for r in records[mode]}
        if len(done[mode]) != len(records[mode]) or not done[mode] <= row_map.keys():
            raise ValueError("Duplicate or unknown saved IDs")
        for r in records[mode]:
            expected = row_map[r["sample_id"]]
            if (r["gold"], r["phase"], r["binding_id"]) != (
                expected["answer"][0],
                expected["phase"],
                expected["binding_id"],
            ) or r["correct"] != score(r["completion"], r["gold"]):
                raise ValueError("Saved result metadata or default score differs")
    if all(all(r["sample_id"] in done[m] for r in selected) for m in args.variants):
        print("All selected rows already complete")
        return
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    from vllm import LLM, SamplingParams

    llm = LLM(**options)
    tokenizer = llm.get_tokenizer()
    if tokenizer.encode("a", add_special_tokens=False) != tokenizer.encode(
        "a", add_special_tokens=True
    ):
        raise ValueError("GLM phase labels require no leading special tokens")
    initial = llm.collective_rpc("glm_indexer_audit")
    if len(initial) != args.tensor_parallel_size or any(
        len(w["layers"]) != expected_layers for w in initial
    ):
        raise ValueError("Unexpected worker/layer coverage")
    for mode in args.variants:
        atomic(
            args.output / f"{mode}.provenance.json",
            {"identity": identity, "variant": mode, "initial_worker_audit": initial},
        )
    params = SamplingParams(temperature=0, max_tokens=8)

    def generate(row, mode):
        changed = llm.collective_rpc("glm_indexer_set_mode", args=(mode,))
        if changed != [mode] * args.tensor_parallel_size:
            raise ValueError("Worker mode switch failed")
        before = llm.collective_rpc("glm_indexer_audit")
        start = time.monotonic()
        result = llm.generate([row["prompt"]], params, use_tqdm=False)[0]
        seconds = time.monotonic() - start
        pos = row["metadata"]["target_pair_token_start"]
        gold = row["answer"][0]
        gold_ids = tokenizer.encode(gold, add_special_tokens=False)
        if (
            len(result.prompt_token_ids) != 128000
            or len(gold_ids) != 1
            or result.prompt_token_ids[pos + 2] != gold_ids[0]
            or pos % 8 != row["phase"]
        ):
            raise ValueError("Actual prompt tokens do not match the validated geometry")
        after = llm.collective_rpc("glm_indexer_audit")
        for old, new in zip(before, after, strict=True):
            if new["active"] != (mode != "native") or len(new["layers"]) != expected_layers:
                raise ValueError("Incorrect active mode or sparse-layer coverage")
            for name, layer in new["layers"].items():
                delta = layer["selection_rows"] - old["layers"][name]["selection_rows"]
                if (mode == "native" and delta != 0) or (mode != "native" and delta <= 0):
                    raise ValueError("Incorrect native/experimental layer execution")
        atomic(args.output / f"{mode}.worker_audit.json", after)
        text = result.outputs[0].text
        return dict(
            sample_id=row["sample_id"],
            phase=row["phase"],
            binding_id=row["binding_id"],
            actual_source_position=pos,
            actual_source_residue=pos % 8,
            gold=gold,
            completion=text,
            correct=score(text, gold),
            scores={metric: score(text, gold, metric) for metric in METRICS},
            output_ids=list(result.outputs[0].token_ids),
            variant=mode,
            seconds=seconds,
        )

    # Repeating all modes on full requests checks cache reset and native-mode restoration.
    probe = {}
    for iteration in range(2):
        for mode in args.variants:
            print(f"Isolation check {iteration + 1}/2: {mode}", flush=True)
            record = generate(selected[0], mode)
            if iteration and record["output_ids"] != probe[mode]["output_ids"]:
                raise ValueError("Mode/request isolation check failed")
            probe[mode] = record
    atomic(args.output / "isolation.json", {"passed": True, "results": probe})
    for row in selected:
        for mode in args.variants:
            if row["sample_id"] in done[mode]:
                continue
            record = generate(row, mode)
            with (args.output / f"{mode}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            records[mode].append(record)
            done[mode].add(row["sample_id"])
            atomic(
                args.output / f"{mode}.summary.json",
                {
                    "default_metric": DEFAULT_METRIC,
                    "rows": len(records[mode]),
                    "expected_rows": len(rows),
                    "complete": len(records[mode]) == len(rows),
                    "accuracy": {
                        metric: sum(
                            score(r["completion"], r["gold"], metric) for r in records[mode]
                        )
                        / len(records[mode])
                        for metric in METRICS
                    },
                },
            )
            print(
                f"{mode}: {len(records[mode])}/{len(rows)}, correct={record['correct']}", flush=True
            )


if __name__ == "__main__":
    main()
