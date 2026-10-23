"""Independent retokenizing validator for a generated 128K dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from periodic_weak_spots.tokenization import Tokenizer, token_ids, token_ids_batch

from .bindings import Binding, BindingPair
from .config import ExperimentConfig
from .layout import PAIR_FORM, PAIR_TOKENS, VALUE_OFFSET, render_query
from .schedule import (
    PrefillAssignment,
    phase_lengths,
    validate_finite_table,
    validate_prefill_schedule,
)
from .source import Sentence


class ValidationError(ValueError):
    pass


# Prefix-length histogram, distinct (tail count, query start, prompt length), and row count.
CellResult = tuple[Counter[int], set[tuple[int, int, int]], int]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValidationError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def _binding(value: dict[str, Any]) -> Binding:
    expected = {
        "binding_id",
        "pairs",
        "target_index",
        "target_key",
        "target_value",
    }
    if set(value) != expected:
        raise ValidationError("binding record keys differ from schema")
    pairs: list[BindingPair] = []
    for pair in value["pairs"]:
        if set(pair) != {"key", "key_token_id", "value", "value_token_id"}:
            raise ValidationError("binding pair keys differ from schema")
        pairs.append(BindingPair(**pair))
    binding = Binding(
        binding_id=value["binding_id"],
        pairs=tuple(pairs),
        target_index=value["target_index"],
    )
    if binding.target.key != value["target_key"] or binding.target.value != value["target_value"]:
        raise ValidationError("binding target fields are inconsistent")
    return binding


def _assignment(value: dict[str, Any]) -> PrefillAssignment:
    expected = {
        "phase",
        "row_index",
        "binding_id",
        "sentence_pair_id",
        "prefill",
        "postfill",
        "role_reversal_phase",
        "role_reversal_row_index",
    }
    if set(value) != expected:
        raise ValidationError("sentence assignment keys differ from schema")
    return PrefillAssignment(
        phase=value["phase"],
        row_index=value["row_index"],
        binding_id=value["binding_id"],
        sentence_pair_id=value["sentence_pair_id"],
        prefill=Sentence.from_dict(value["prefill"]),
        postfill=Sentence.from_dict(value["postfill"]),
        role_reversal_phase=value["role_reversal_phase"],
        role_reversal_row_index=value["role_reversal_row_index"],
    )


def validate_prompt_row(
    row: dict[str, Any],
    tokenizer: Tokenizer,
    binding: Binding,
    assignment: PrefillAssignment,
    config: ExperimentConfig,
) -> None:
    """Retokenize and verify one row, reusing no layout helper except ``render_query``."""

    required = {
        "sample_id",
        "pair_form",
        "phase",
        "row_index",
        "prompt",
        "answer",
        "components",
        "metadata",
        "binding_id",
        "sentence_pair_id",
        "prefill_sentence_id",
        "postfill_sentence_id",
        "role_reversal_phase",
        "role_reversal_row_index",
    }
    if set(row) != required:
        raise ValidationError(
            f"row keys differ; missing={sorted(required - set(row))}, "
            f"unknown={sorted(set(row) - required)}"
        )
    if row["pair_form"] != PAIR_FORM:
        raise ValidationError("unsupported pair form")
    expected_sample_id = (
        f"{config.sample_prefix}_{PAIR_FORM}_phase{assignment.phase}_{assignment.row_index:03d}"
    )
    if row["sample_id"] != expected_sample_id:
        raise ValidationError("sample ID does not match its configured coordinate")
    if (
        row["phase"],
        row["row_index"],
        row["binding_id"],
        row["sentence_pair_id"],
    ) != (
        assignment.phase,
        assignment.row_index,
        assignment.binding_id,
        assignment.sentence_pair_id,
    ):
        raise ValidationError("row does not reference its schedule coordinate")
    if row["prefill_sentence_id"] != assignment.prefill.sentence_id:
        raise ValidationError("prefill sentence reference mismatch")
    if row["postfill_sentence_id"] != assignment.postfill.sentence_id:
        raise ValidationError("postfill sentence reference mismatch")
    if row["role_reversal_phase"] != assignment.role_reversal_phase:
        raise ValidationError("role-reversal phase mismatch")
    if row["role_reversal_row_index"] != assignment.role_reversal_row_index:
        raise ValidationError("role-reversal row mismatch")
    if row["answer"] != [binding.target.value]:
        raise ValidationError("answer does not equal the target value")

    component_order = [
        "intro",
        "natural_prefill",
        "body_prefix",
        "pair_body",
        "natural_postfill",
        "tail",
        "query_separator",
        "query",
    ]
    components = row["components"]
    if set(components) != set(component_order):
        raise ValidationError("component keys differ from layout schema")
    if components["natural_prefill"] != assignment.prefill.text:
        raise ValidationError("prefill text differs from sentence schedule")
    if components["natural_postfill"] != assignment.postfill.text:
        raise ValidationError("postfill text differs from sentence schedule")
    expected_query = render_query(binding.target.key)
    if components["query"] != expected_query:
        raise ValidationError("query text differs from the configured template")
    for label, guard in (
        ("body_prefix", config.layout.body_prefix_guard_token),
        ("tail", config.layout.tail_guard_token),
    ):
        if len(token_ids(tokenizer, guard)) != 1:
            raise ValidationError(f"{label} guard is not exactly one tokenizer token")
        if not (components[label].startswith(guard) and components[label].endswith(guard)):
            raise ValidationError(f"{label} does not contain both configured guards")
    expected_records: list[str] = []
    expected_body_ids: list[int] = []
    colon_ids, separator_ids = token_ids_batch(tokenizer, (":", config.layout.record_separator))
    if len(colon_ids) != 1 or len(separator_ids) != 1:
        raise ValidationError("pair punctuation pieces must each be one token")
    expected_record_ids: list[list[int]] = []
    for pair in binding.pairs:
        text = f"{pair.key}:{pair.value}{config.layout.record_separator}"
        expected = [pair.key_token_id, colon_ids[0], pair.value_token_id, separator_ids[0]]
        expected_records.append(text)
        expected_record_ids.append(expected)
        expected_body_ids.extend(expected)
    actual_record_ids = token_ids_batch(tokenizer, expected_records)
    if actual_record_ids != expected_record_ids:
        raise ValidationError("binding does not realize the declared pair form")
    if components["pair_body"] != "".join(expected_records):
        raise ValidationError("pair body text differs from binding")
    if token_ids(tokenizer, components["pair_body"]) != expected_body_ids:
        raise ValidationError("pair-body boundaries are not token exact")
    prompt = "".join(components[label] for label in component_order)
    if prompt != row["prompt"]:
        raise ValidationError("prompt is not the concatenation of declared components")
    prompt_ids = token_ids(tokenizer, prompt)
    if len(prompt_ids) != config.layout.prompt_tokens:
        raise ValidationError("prompt token count differs from the configured layout")
    component_ids = token_ids_batch(
        tokenizer,
        [components[label] for label in component_order],
    )
    # Additivity is what makes the per-component counts below valid prompt spans.
    separate_ids = [token for encoded in component_ids for token in encoded]
    if prompt_ids != separate_ids:
        raise ValidationError("declared prompt component boundary is non-additive")

    spans = row["metadata"].get("spans")
    if not isinstance(spans, dict) or set(spans) != set(component_order):
        raise ValidationError("span metadata is incomplete")
    cursor = 0
    for label, encoded in zip(component_order, component_ids, strict=True):
        encoded_count = len(encoded)
        if spans[label] != {"start": cursor, "count": encoded_count}:
            raise ValidationError(f"{label} span does not match retokenization")
        cursor += encoded_count
    if cursor != config.layout.prompt_tokens:
        raise ValidationError("component spans do not fill the prompt")

    body_start = spans["pair_body"]["start"]
    target_pair_start = body_start + binding.target_index * PAIR_TOKENS
    target_value_start = target_pair_start + VALUE_OFFSET
    query_count = spans["query"]["count"]
    query_start = config.layout.prompt_tokens - query_count
    metadata = row["metadata"]
    equations = {
        "prompt_token_count": config.layout.prompt_tokens,
        "model_prefill_token_count": config.layout.prompt_tokens,
        "phase_control_prefix_count": assignment.prefill.token_count,
        "fixed_natural_filler_count": config.layout.natural_filler_tokens,
        "pair_token_count": PAIR_TOKENS,
        "target_index": binding.target_index,
        "target_pair_token_start": target_pair_start,
        "target_value_token_start": target_value_start,
        "target_phase": target_pair_start % config.layout.alignment,
        "query_token_start": query_start,
        "query_token_count": query_count,
        "target_to_query_distance": query_start - target_value_start,
        "spans": spans,
    }
    if metadata != equations:
        raise ValidationError("metadata differs from independently computed equations")
    # Derived from the design mean rather than the layout's filler split, as a cross-check.
    expected_body_prefix_count = (
        config.layout.target_pair_mean
        - spans["intro"]["count"]
        - binding.target_index * PAIR_TOKENS
        - config.phase_design.mean
    )
    if spans["body_prefix"]["count"] != expected_body_prefix_count:
        raise ValidationError("body prefix does not realize the target mean")
    expected_target_start = (
        config.layout.target_pair_mean + assignment.prefill.token_count - config.phase_design.mean
    )
    if target_pair_start != expected_target_start:
        raise ValidationError("target position does not realize the configured shift")
    if target_pair_start % config.layout.alignment != assignment.phase:
        raise ValidationError("target phase does not equal requested phase")
    if (
        assignment.prefill.token_count + assignment.postfill.token_count
        != config.layout.natural_filler_tokens
    ):
        raise ValidationError("natural filler sum differs from the configured layout")


def _validate_cell(
    root: Path,
    pair_form: str,
    phase: int,
    rows_per_cell: int,
    config: ExperimentConfig,
    by_binding: dict[int, Binding],
    schedule_index: dict[tuple[int, int], PrefillAssignment],
    tokenizer: Tokenizer,
) -> CellResult:
    path = root / "rows" / pair_form / f"phase_{phase}.jsonl"
    rows = _read_jsonl(path)
    if len(rows) != rows_per_cell:
        raise ValidationError("matrix cell row count differs from published metadata")
    coordinates: set[int] = set()
    histogram: Counter[int] = Counter()
    fixed: set[tuple[int, int, int]] = set()
    for row in rows:
        if row["pair_form"] != pair_form or row["phase"] != phase:
            raise ValidationError("row is stored in the wrong matrix file")
        row_index = row["row_index"]
        coordinates.add(row_index)
        assignment = schedule_index[(phase, row_index)]
        validate_prompt_row(
            row,
            tokenizer,
            by_binding[assignment.binding_id],
            assignment,
            config,
        )
        histogram[assignment.prefill.token_count] += 1
        spans = row["metadata"]["spans"]
        fixed.add(
            (
                spans["tail"]["count"],
                spans["query"]["start"],
                row["metadata"]["prompt_token_count"],
            )
        )
    if coordinates != set(range(rows_per_cell)):
        raise ValidationError("matrix cell row indices are not 0..rows_per_cell-1")
    return histogram, fixed, len(rows)


def validate_dataset(output_root: str | Path, tokenizer: Tokenizer) -> dict[str, Any]:
    """Check every file of a built dataset; raise ``ValidationError`` on the first violation."""

    root = Path(output_root)
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        config = ExperimentConfig.from_dict(metadata["config"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"dataset metadata is missing or invalid: {exc}") from exc
    if metadata.get("schema_version") != config.schema_version:
        raise ValidationError("metadata schema_version differs from its config")
    if metadata.get("status") != "complete":
        raise ValidationError("dataset status must be complete")
    validate_finite_table()

    bindings = [_binding(row) for row in _read_jsonl(root / metadata["bindings_path"])]
    rows_per_phase = config.phase_design.rows_per_phase
    if len(bindings) != rows_per_phase or {item.binding_id for item in bindings} != set(
        range(rows_per_phase)
    ):
        raise ValidationError("binding IDs do not cover the full phase design")
    by_binding = {item.binding_id: item for item in bindings}

    schedule = [_assignment(row) for row in _read_jsonl(root / metadata["sentence_schedule_path"])]
    try:
        validate_prefill_schedule(schedule)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    schedule_index = {(item.phase, item.row_index): item for item in schedule}

    matrix = metadata.get("matrix")
    if (
        not isinstance(matrix, dict)
        or matrix.get("rows_per_cell") != rows_per_phase
        or matrix.get("master_rows_per_cell") != rows_per_phase
    ):
        raise ValidationError("matrix metadata must cover every scheduled row")
    fixed: set[tuple[int, int, int]] = set()
    checked = 0
    for phase in config.matrix.phases:
        histogram, cell_fixed, count = _validate_cell(
            root, PAIR_FORM, phase, rows_per_phase, config, by_binding, schedule_index, tokenizer
        )
        blocks = rows_per_phase // 256
        expected_histogram = Counter(
            {length: amount * blocks for length, amount in Counter(phase_lengths(phase)).items()}
        )
        if histogram != expected_histogram:
            raise ValidationError(f"phase {phase} finite histogram mismatch")
        fixed.update(cell_fixed)
        checked += count
    # Only the prefix/postfix split may vary across rows.
    if len(fixed) != 1:
        raise ValidationError("tail/query/prompt lengths vary across rows")
    expected_rows = len(config.matrix.phases) * rows_per_phase
    if matrix.get("total_rows") != expected_rows or checked != expected_rows:
        raise ValidationError(f"validated {checked} prompts, expected {expected_rows}")
    return {
        "status": "ok",
        "validated_rows": checked,
        "dataset_id": config.experiment_id,
        "tokenizer_id": config.tokenizer.model_id,
        "tokenizer_revision": config.tokenizer.revision,
    }
