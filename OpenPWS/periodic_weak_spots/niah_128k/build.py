"""Deterministic 128K dataset generation with readable JSON and JSONL output."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from periodic_weak_spots.balanced_lengths import DesignA, conditional_pmf, exact_counts, raw_moment
from periodic_weak_spots.tokenization import Tokenizer

from .bindings import Binding
from .config import ExperimentConfig
from .layout import PAIR_FORM, build_prompt_row, prepare_scaffold
from .schedule import (
    FINITE_HISTOGRAM,
    build_prefill_schedule,
    finite_table_payload,
    sentence_capacity_requirements,
)
from .source import Sentence


def _fraction(value: Any) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def inspect_design(
    config: ExperimentConfig, *, available_capacity: dict[int, int]
) -> dict[str, Any]:
    """The design report stored in ``metadata.json``."""

    # Design A is the exact residue-balanced distribution (second moment 1811/3); the finite
    # table rounds it to 256 integer rows per residue with second moment 604.
    design = DesignA(modulus=8, start_multiple=1, n=2)
    counts = exact_counts(design)
    phases: dict[str, Any] = {}
    blocks = config.phase_design.rows_per_phase // 256
    for phase, base_histogram in FINITE_HISTOGRAM.items():
        histogram = {length: amount * blocks for length, amount in base_histogram.items()}
        count = sum(histogram.values())
        phases[str(phase)] = {
            "histogram": {str(length): amount for length, amount in histogram.items()},
            "count": count,
            "mean": sum(length * amount for length, amount in histogram.items()) // count,
            "second_raw_moment": (
                sum(length * length * amount for length, amount in histogram.items()) // count
            ),
        }
    # The target start is a constant plus the prefix length, so it inherits the prefix variance.
    mean = config.layout.target_pair_mean
    variance = config.phase_design.second_raw_moment - config.phase_design.mean**2
    capacity = {
        str(length): {
            "required": count,
            "available": available_capacity.get(length, 0),
            "sufficient": available_capacity.get(length, 0) >= count,
        }
        for length, count in sentence_capacity_requirements(
            config.phase_design.rows_per_phase
        ).items()
    }
    conditional = conditional_pmf(0, design)
    return {
        "experiment_id": config.experiment_id,
        "design_a": {
            "parameters": design.to_dict(),
            "counts": {str(length): count for length, count in counts.items()},
            "support": [min(counts), max(counts)],
            "residue_mass": _fraction(
                sum(
                    Fraction(count, design.denominator)
                    for length, count in counts.items()
                    if length % design.modulus == 0
                )
            ),
            "conditional_mean": _fraction(raw_moment(conditional, 1)),
            "conditional_second_raw_moment": _fraction(raw_moment(conditional, 2)),
        },
        "finite_table": {"phases": phases},
        "target_position_moments": {
            PAIR_FORM: {
                "mean": mean,
                "second_raw_moment": mean * mean + variance,
                "variance": variance,
                "variable_term": "natural_prefill_token_count",
            }
        },
        "source_bin_capacity": capacity,
        "spans": {
            "prompt_tokens": config.layout.prompt_tokens,
            "fixed": [
                "aligned_intro",
                "body_prefix",
                "pair_body",
                f"natural_prefill+natural_postfill={config.layout.natural_filler_tokens}",
                "tail",
                "query_separator",
                "query",
            ],
            "variable": ["natural_prefill", "natural_postfill", "target_position"],
        },
        "matrix": {
            "forms": list(config.matrix.forms),
            "phases": list(config.matrix.phases),
            "rows_per_cell": config.phase_design.rows_per_phase,
            "prompt_count": (
                len(config.matrix.forms)
                * len(config.matrix.phases)
                * config.phase_design.rows_per_phase
            ),
        },
        "candidate_pool": config.candidate_pool.to_dict(),
    }


def _write_json(path: Path, value: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def publish_dataset(
    output_root: str | Path,
    tokenizer: Tokenizer,
    bins: dict[int, list[Sentence]],
    bindings: list[Binding],
    *,
    filler: str,
    source_report: dict[str, Any],
    config: ExperimentConfig,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write bindings, the sentence schedule, ``rows/key_value/phase_*.jsonl``, and metadata."""

    rows_per_phase = config.phase_design.rows_per_phase
    if {binding.binding_id for binding in bindings} != set(range(rows_per_phase)) or len(
        bindings
    ) != rows_per_phase:
        raise ValueError(f"exactly {rows_per_phase} bindings with IDs 0..{rows_per_phase - 1}")
    if any(
        len(binding.pairs) != config.layout.pair_count
        or binding.target_index != config.layout.target_index
        for binding in bindings
    ):
        raise ValueError("binding dimensions differ from the fixed layout")
    by_binding = {binding.binding_id: binding for binding in bindings}
    schedule = build_prefill_schedule(
        bins,
        seed=config.phase_design.schedule_seed,
        rows_per_phase=rows_per_phase,
    )
    root = Path(output_root)
    _write_jsonl(
        root / "bindings.jsonl", [binding.to_dict() for binding in bindings], overwrite=overwrite
    )
    _write_jsonl(
        root / "sentence_schedule.jsonl",
        [assignment.to_dict() for assignment in schedule],
        overwrite=overwrite,
    )
    # Every row shares the intro, body prefix, and tail; only bindings and sentences vary.
    scaffold = prepare_scaffold(
        tokenizer, bindings[0], alignment_filler=filler, tail_filler=filler, spec=config.layout
    )
    for phase in config.matrix.phases:
        assignments = sorted(
            (row for row in schedule if row.phase == phase), key=lambda row: row.row_index
        )
        rows = [
            build_prompt_row(
                tokenizer,
                by_binding[assignment.binding_id],
                assignment,
                scaffold,
                config.layout,
                sample_prefix=config.sample_prefix,
            )
            for assignment in assignments
        ]
        _write_jsonl(root / "rows" / PAIR_FORM / f"phase_{phase}.jsonl", rows, overwrite=overwrite)

    metadata = {
        "schema_version": config.schema_version,
        "dataset_id": config.experiment_id,
        "status": "complete",
        "config": config.to_dict(),
        "source_report": source_report,
        "design": inspect_design(
            config, available_capacity={length: len(rows) for length, rows in bins.items()}
        ),
        "finite_table": finite_table_payload(rows_per_phase=rows_per_phase),
        "bindings_path": "bindings.jsonl",
        "sentence_schedule_path": "sentence_schedule.jsonl",
        "matrix": {
            "forms": list(config.matrix.forms),
            "phases": list(config.matrix.phases),
            "rows_per_cell": rows_per_phase,
            "total_rows": len(config.matrix.forms) * len(config.matrix.phases) * rows_per_phase,
            "row_path_template": "rows/{pair_form}/phase_{phase}.jsonl",
            "master_rows_per_cell": rows_per_phase,
        },
    }
    _write_json(root / "metadata.json", metadata, overwrite=overwrite)
    return metadata
