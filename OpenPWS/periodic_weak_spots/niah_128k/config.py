"""The balanced 128K / 16,000-pair NIAH design used for DeepSeek-V4.

Every geometric constant is checked against the published design. Dataset size
may be increased in exact 256-row balanced blocks per phase.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar, cast

SOURCE_DATASET_ID = "agentlans/high-quality-english-sentences"
SOURCE_REVISION = "5dfcb23b31fa9b85ce90b5713fcbba29179f2401"
TOKENIZER_ID = "deepseek-ai/DeepSeek-V4-Flash"
TOKENIZER_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"

EVALUATION_ID = "niah-128k-balanced-target64k-16000pairs"
# Published profiles are pinned to their IDs and seeds. Evaluation and calibration are separated
# by seeds only (sentence source and token pool are shared); other identities use role "custom".
PROFILE_IDENTITIES = {
    "evaluation": {
        "experiment_id": EVALUATION_ID,
        "sample_prefix": "niah128k",
        "shuffle_seed": 20260720,
        "schedule_seed": 20260721,
        "binding_seed": 20260625,
    },
    "calibration": {
        "experiment_id": f"{EVALUATION_ID}-calibration",
        "sample_prefix": "niah128k-cal",
        "shuffle_seed": 20260902,
        "schedule_seed": 20260903,
        "binding_seed": 20260904,
    },
}

T = TypeVar("T")


def _is_default(spec: Any) -> bool:
    return all(getattr(spec, item.name) == item.default for item in fields(spec))


def _strict(cls: type[T], value: dict[str, Any]) -> T:
    """Construct ``cls`` from ``value``, rejecting missing or unknown keys."""
    expected = {item.name for item in fields(cast(Any, cls))}
    if set(value) != expected:
        raise ValueError(
            f"{cls.__name__} keys differ; missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )
    return cls(**value)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    dataset_id: str = SOURCE_DATASET_ID
    revision: str = SOURCE_REVISION
    split: str = "train"
    text_field: str = "text"
    shuffle_seed: int = 20260720

    def __post_init__(self) -> None:
        if (self.dataset_id, self.revision, self.split, self.text_field) != (
            SOURCE_DATASET_ID,
            SOURCE_REVISION,
            "train",
            "text",
        ):
            raise ValueError("the sentence source is pinned")


@dataclass(frozen=True, slots=True)
class TokenizerSpec:
    model_id: str = TOKENIZER_ID
    revision: str = TOKENIZER_REVISION

    def __post_init__(self) -> None:
        if (self.model_id, self.revision) != (TOKENIZER_ID, TOKENIZER_REVISION):
            raise ValueError("the DeepSeek-V4 tokenizer revision is pinned")


@dataclass(frozen=True, slots=True)
class CandidatePoolSpec:
    # Descriptive except min_token_id and max_characters; the rules are implemented in
    # bindings.collect_context_stable_token_candidates.
    min_token_id: int = 10_000
    max_characters: int = 16
    text_class: str = "alphabetic"
    require_record_stability: bool = True
    require_query_stability: bool = True
    master_pair_count: int = 16_000

    def __post_init__(self) -> None:
        if not _is_default(self):
            raise ValueError("the candidate pool is fixed")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LayoutSpec:
    prompt_tokens: int = 128_000
    pair_count: int = 16_000
    master_pair_count: int = 16_000
    target_index: int = 248
    natural_filler_tokens: int = 48
    alignment: int = 8
    query_separator: str = "\n\n"
    pad_token: str = "｜"
    record_separator: str = "。"
    target_pair_mean: int = 64_000
    body_prefix_guard_token: str = "｜"
    tail_guard_token: str = "｜"

    def __post_init__(self) -> None:
        if not _is_default(self):
            raise ValueError("the 128K / 16,000-pair layout is fixed")


@dataclass(frozen=True, slots=True)
class PhaseDesignSpec:
    # Prefix lengths in [8, 40] with mean 24 and second raw moment 604 (variance 28) in every
    # residue cell; each 256-row block needs 1,024 sentence pairs from 2,048 distinct sentences.
    modulus: int = 8
    rows_per_phase: int = 256
    prefill_min: int = 8
    prefill_max: int = 40
    mean: int = 24
    second_raw_moment: int = 604
    sentence_pair_count: int = 1_024
    distinct_sentence_count: int = 2_048
    schedule_seed: int = 20260721

    def __post_init__(self) -> None:
        if (
            self.modulus,
            self.prefill_min,
            self.prefill_max,
            self.mean,
            self.second_raw_moment,
        ) != (8, 8, 40, 24, 604):
            raise ValueError("the phase geometry and moments are fixed")
        if self.rows_per_phase <= 0 or self.rows_per_phase % 256:
            raise ValueError("rows_per_phase must be a positive multiple of 256")
        if self.sentence_pair_count != 4 * self.rows_per_phase:
            raise ValueError("sentence_pair_count must equal 4 * rows_per_phase")
        if self.distinct_sentence_count != 8 * self.rows_per_phase:
            raise ValueError("distinct_sentence_count must equal 8 * rows_per_phase")


@dataclass(frozen=True, slots=True)
class MatrixSpec:
    forms: tuple[str, ...] = ("key_value",)
    phases: tuple[int, ...] = tuple(range(8))

    def __post_init__(self) -> None:
        if self.forms != ("key_value",) or self.phases != tuple(range(8)):
            raise ValueError("the matrix is the key:value record form at all eight phases")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    schema_version: int = 1
    experiment_id: str = EVALUATION_ID
    source: SourceSpec = field(default_factory=SourceSpec)
    tokenizer: TokenizerSpec = field(default_factory=TokenizerSpec)
    candidate_pool: CandidatePoolSpec = field(default_factory=CandidatePoolSpec)
    layout: LayoutSpec = field(default_factory=LayoutSpec)
    phase_design: PhaseDesignSpec = field(default_factory=PhaseDesignSpec)
    matrix: MatrixSpec = field(default_factory=MatrixSpec)
    binding_seed: int = 20260625
    dataset_role: str = "evaluation"
    sample_prefix: str = "niah128k"

    def __post_init__(self) -> None:
        identity = {
            "experiment_id": self.experiment_id,
            "sample_prefix": self.sample_prefix,
            "shuffle_seed": self.source.shuffle_seed,
            "schedule_seed": self.phase_design.schedule_seed,
            "binding_seed": self.binding_seed,
        }
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.dataset_role in PROFILE_IDENTITIES:
            if PROFILE_IDENTITIES[self.dataset_role] != identity:
                raise ValueError("unknown 128K profile identity")
        elif self.dataset_role != "custom":
            raise ValueError("dataset_role must be evaluation, calibration, or custom")
        elif not self.experiment_id or not self.sample_prefix:
            raise ValueError("custom profiles require experiment_id and sample_prefix")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["matrix"]["forms"] = list(self.matrix.forms)
        value["matrix"]["phases"] = list(self.matrix.phases)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExperimentConfig:
        expected = {item.name for item in fields(cls)}
        if set(value) != expected:
            raise ValueError(f"config keys must be exactly {sorted(expected)}")
        matrix = dict(value["matrix"])
        matrix["forms"] = tuple(matrix["forms"])
        matrix["phases"] = tuple(matrix["phases"])
        return cls(
            schema_version=value["schema_version"],
            experiment_id=value["experiment_id"],
            source=_strict(SourceSpec, value["source"]),
            tokenizer=_strict(TokenizerSpec, value["tokenizer"]),
            candidate_pool=_strict(CandidatePoolSpec, value["candidate_pool"]),
            layout=_strict(LayoutSpec, value["layout"]),
            phase_design=_strict(PhaseDesignSpec, value["phase_design"]),
            matrix=_strict(MatrixSpec, matrix),
            binding_seed=value["binding_seed"],
            dataset_role=value["dataset_role"],
            sample_prefix=value["sample_prefix"],
        )


def load_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
