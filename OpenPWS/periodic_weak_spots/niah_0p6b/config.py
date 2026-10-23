"""Suite configuration and typed schemas for animal-name retrieval episodes."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

from .template import PromptTemplate

SCHEMA_VERSION = 1
QWEN_TOKENIZER_ID = "Qwen/Qwen3-0.6B"
# natural: Prefix Padding (complementary natural sentences before and after the body).
# synthetic: In-sequence Padding (fixed pad-token guards plus randomized gaps between pairs).
PostfillMode = Literal["natural", "synthetic"]
BodySpacingMode = Literal["packed", "randomized_gaps"]


def phase_base_for_strides(strides: tuple[int, ...]) -> int:
    """The residue modulus: the LCM of the strides (24 for strides 3, 4, 6, 8, 12)."""

    if not strides or any(type(stride) is not int or stride <= 0 for stride in strides):
        raise ValueError("suite strides must be a non-empty tuple of positive integers")
    return math.lcm(*strides)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A curated pool word (with its leading space) and its single token ID."""

    candidate_id: str
    text: str
    token_id: int


@dataclass(frozen=True, slots=True)
class TokenizerSpec:
    tokenizer_id: str
    revision: str

    def __post_init__(self) -> None:
        if self.tokenizer_id != QWEN_TOKENIZER_ID:
            raise ValueError(f"the 0.6B NIAH suites require {QWEN_TOKENIZER_ID}")
        if not self.revision:
            raise ValueError("tokenizer revision cannot be empty")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TokenizerSpec:
        if set(value) != {"tokenizer_id", "revision"}:
            raise ValueError("tokenizer fields do not match schema")
        return cls(tokenizer_id=str(value["tokenizer_id"]), revision=str(value["revision"]))


@dataclass(frozen=True, slots=True)
class Sentence:
    """A Prefix Padding filler sentence, including its suffix, and its exact token IDs."""

    sentence_id: str
    text: str
    token_ids: tuple[int, ...]
    source_index: int

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class BodySpacingSpec:
    """``packed`` pairs are contiguous; ``randomized_gaps`` inserts pad tokens between pairs.

    With randomized gaps the target pair lands where a Prefix Padding prefill of
    the balanced-length draw would put it: the pre-target gaps total that draw
    minus ``reference_filler_offset`` (the leading guard length). All gaps total
    ``total_gap_extras``.
    """

    mode: BodySpacingMode = "packed"
    pad_text: str | None = None
    pad_token_id: int | None = None
    outer_guard_tokens: int = 0
    total_gap_extras: int = 0
    max_gap_extras: int = 0
    reference_filler_offset: int = 0
    reference_suite_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "packed":
            if any(getattr(self, item.name) != item.default for item in fields(self)):
                raise ValueError("packed body spacing cannot declare padding controls")
        elif self.mode != "randomized_gaps":
            raise ValueError("unsupported body-spacing mode")
        elif self.reference_filler_offset != self.outer_guard_tokens:
            raise ValueError("reference filler offset must equal the leading guard length")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BodySpacingSpec:
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("body-spacing fields do not match schema")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """Prompt layout, sizes, and seeds for one suite; the gold pair sits at ``target_index``."""

    suite_id: str
    strides: tuple[int, ...]
    pair_count: int
    target_index: int
    samples_per_residue: int
    postfill_mode: PostfillMode
    fixed_filler_total: int
    seed: int
    k_shot: int
    few_shot_seed: int
    max_sequence_length: int
    prompt_template: PromptTemplate
    sample_index_start: int = 0
    body_spacing: BodySpacingSpec = BodySpacingSpec()

    def __post_init__(self) -> None:
        if not self.suite_id:
            raise ValueError("suite_id cannot be empty")
        phase_base_for_strides(self.strides)
        if self.pair_count < 2 or self.samples_per_residue <= 0:
            raise ValueError("episode requires at least two pairs and positive samples")
        if not 0 <= self.target_index < self.pair_count:
            raise ValueError("target index lies outside the pair body")
        if not 0 < self.k_shot < self.pair_count:
            raise ValueError("k_shot must be positive and leave a non-demonstration target")
        if self.postfill_mode not in ("natural", "synthetic"):
            raise ValueError("unsupported postfill mode")
        if (self.postfill_mode == "synthetic") != (self.body_spacing.mode == "randomized_gaps"):
            raise ValueError("synthetic guards go with randomized gap spacing")
        if self.postfill_mode == "natural":
            if not self.fixed_filler_total or self.fixed_filler_total % self.phase_base:
                raise ValueError("natural postfill needs a positive multiple of the phase base")
        else:
            if self.fixed_filler_total != 2 * self.body_spacing.outer_guard_tokens:
                raise ValueError("synthetic fixed filler total must equal both guards")
            before_capacity = self.target_index * self.body_spacing.max_gap_extras
            after_count = self.pair_count - self.target_index - 1
            after_capacity = after_count * self.body_spacing.max_gap_extras
            # The balanced-length support for modulus B ends at 5B (120 for B = 24); the
            # pre-target gaps must absorb that draw minus the leading guard.
            reference_support_max = 5 * self.phase_base
            maximum_x = reference_support_max - self.body_spacing.reference_filler_offset
            if before_capacity < maximum_x:
                raise ValueError("pre-gold gaps cannot realize the balanced-length support")
            if after_capacity < self.body_spacing.total_gap_extras:
                raise ValueError("post-gold gaps cannot realize the configured budget")

    @property
    def phase_base(self) -> int:
        return phase_base_for_strides(self.strides)


@dataclass(frozen=True, slots=True)
class EpisodeRow:
    """One prompt at one residue; ``phase`` is ``target_pair_start mod phase_base``."""

    sample_id: str
    logical_sample_id: str
    phase: int
    token_ids: tuple[int, ...]
    candidate_value_token_ids: tuple[int, ...]
    gold_candidate_index: int
    gold_key_token_id: int
    gold_value_token_id: int
    demonstration_pair_indices: tuple[int, ...]
    pair_region_start: int
    target_pair_start: int
    target_value_position: int
    postfill_start: int
    query_start: int
    scoring_position: int
    prefill_length: int
    postfill_length: int
    prefill_sentence_id: str
    postfill_sentence_id: str
    gap_extra_lengths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.sample_id or not self.logical_sample_id or not self.token_ids:
            raise ValueError("episode row must have identity and tokens")
        if not 0 <= self.gold_candidate_index < len(self.candidate_value_token_ids):
            raise ValueError("gold candidate index lies outside candidate values")
        if self.scoring_position != len(self.token_ids) - 1:
            raise ValueError("row must end at the scoring position without the answer token")
        if not (
            self.pair_region_start
            <= self.target_pair_start
            < self.target_value_position
            < self.postfill_start
            <= self.query_start
            <= self.scoring_position
        ):
            raise ValueError("episode component positions are reversed")
        if (
            not self.demonstration_pair_indices
            or len(set(self.demonstration_pair_indices)) != len(self.demonstration_pair_indices)
            or self.gold_candidate_index in self.demonstration_pair_indices
        ):
            raise ValueError("demonstration pair indices are empty, duplicated, or leak gold")
        if (
            self.prefill_length <= 0
            or self.postfill_length <= 0
            or self.prefill_sentence_id == self.postfill_sentence_id
            or self.query_start != self.postfill_start + self.postfill_length
        ):
            raise ValueError(
                "prefill and postfill must be distinct, non-empty, and end at the query"
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in (
            "token_ids",
            "candidate_value_token_ids",
            "demonstration_pair_indices",
            "gap_extra_lengths",
        ):
            value[name] = list(value[name])
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EpisodeRow:
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("episode row fields do not match schema")
        copied = dict(value)
        copied["token_ids"] = tuple(copied["token_ids"])
        copied["candidate_value_token_ids"] = tuple(copied["candidate_value_token_ids"])
        copied["demonstration_pair_indices"] = tuple(copied["demonstration_pair_indices"])
        copied["gap_extra_lengths"] = tuple(copied["gap_extra_lengths"])
        return cls(**copied)


@dataclass(frozen=True, slots=True)
class SuiteManifest:
    """Suite-level geometry saved as ``manifest.json``; rows are revalidated against it."""

    suite_id: str
    tokenizer_id: str
    tokenizer_revision: str
    phase_base: int
    strides: tuple[int, ...]
    row_count: int
    samples_per_residue: int
    candidate_count: int
    row_ids: tuple[str, ...]
    prompt_template: PromptTemplate
    target_index: int
    k_shot: int
    pair_token_count: int
    value_offset: int
    postfill_mode: PostfillMode
    fixed_filler_total: int
    task_descriptor_token_count: int
    mapping_start_token_count: int
    query_token_count: int
    query_residues: tuple[int, ...]
    scoring_residues: tuple[int, ...]
    sequence_length_min: int
    sequence_length_max: int
    maximum_sequence_length: int
    schema_version: int = SCHEMA_VERSION
    sample_index_start: int = 0
    body_spacing: BodySpacingSpec = BodySpacingSpec()

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if self.phase_base != phase_base_for_strides(self.strides):
            raise ValueError("phase base must be the LCM of suite strides")
        if self.row_count != self.phase_base * self.samples_per_residue:
            raise ValueError("manifest row count is not phase-derived")
        if len(self.row_ids) != self.row_count or len(set(self.row_ids)) != self.row_count:
            raise ValueError("manifest row IDs are incomplete or duplicated")
        if (
            len(self.query_residues) != self.phase_base
            or len(self.scoring_residues) != self.phase_base
            or any(
                not 0 <= residue < self.phase_base
                for residue in (*self.query_residues, *self.scoring_residues)
            )
        ):
            raise ValueError("manifest phase-indexed residues are incomplete or invalid")
        # Only the key's residue may vary across cells; the query and scoring token keep
        # one residue, so their phase cannot explain accuracy differences.
        if len(set(self.query_residues)) != 1 or len(set(self.scoring_residues)) != 1:
            raise ValueError("fixed filler totals require constant query/scoring residues")
        if not (
            self.sequence_length_min <= self.sequence_length_max <= self.maximum_sequence_length
        ):
            raise ValueError("manifest sequence bounds are invalid")
        TokenizerSpec(self.tokenizer_id, self.tokenizer_revision)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["strides"] = list(self.strides)
        value["row_ids"] = list(self.row_ids)
        value["query_residues"] = list(self.query_residues)
        value["scoring_residues"] = list(self.scoring_residues)
        value["prompt_template"] = self.prompt_template.to_dict()
        value["body_spacing"] = self.body_spacing.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SuiteManifest:
        if (
            set(value) != set(cls.__dataclass_fields__)
            or value.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError("manifest fields do not match schema")
        copied = dict(value)
        copied["strides"] = tuple(copied["strides"])
        copied["row_ids"] = tuple(copied["row_ids"])
        copied["query_residues"] = tuple(copied["query_residues"])
        copied["scoring_residues"] = tuple(copied["scoring_residues"])
        copied["prompt_template"] = PromptTemplate.from_dict(copied["prompt_template"])
        copied["body_spacing"] = BodySpacingSpec.from_dict(copied["body_spacing"])
        return cls(**copied)


@dataclass(frozen=True, slots=True)
class SentenceSourceSpec:
    dataset_id: str
    revision: str
    split: str
    text_field: str
    shuffle_seed: int
    maximum_rows_scanned: int
    minimum_token_length: int
    maximum_token_length: int
    maximum_per_bin: int


@dataclass(frozen=True, slots=True)
class SuiteConfig:
    suite_id: str
    tokenizer: TokenizerSpec
    token_pool_manifest: Path
    sentence_source: SentenceSourceSpec | None
    episode: EpisodeSpec

    def __post_init__(self) -> None:
        if self.suite_id != self.episode.suite_id:
            raise ValueError("suite ID differs between configuration and episode")
        if (self.sentence_source is None) != (self.episode.postfill_mode == "synthetic"):
            raise ValueError("only synthetic (In-sequence Padding) suites omit the sentence source")


_EPISODE_FIELDS = {
    "strides",
    "pair_count",
    "target_index",
    "samples_per_residue",
    "postfill_mode",
    "fixed_filler_total",
    "seed",
    "k_shot",
    "few_shot_seed",
    "max_sequence_length",
}
_OPTIONAL_EPISODE_FIELDS = {"sample_index_start", "body_spacing"}
_SUITE_FIELDS = {
    "schema_version",
    "suite_id",
    "tokenizer",
    "token_pool_manifest",
    "sentence_source",
    "episode",
    "template",
}


def load_suite_config(path: str | Path) -> SuiteConfig:
    """Load a suite JSON; ``token_pool_manifest`` is relative to the JSON file."""

    config_path = Path(path).expanduser().resolve()
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if set(value) != _SUITE_FIELDS or value["schema_version"] != 1:
        raise ValueError(f"suite configuration fields must be exactly {sorted(_SUITE_FIELDS)}")
    episode_value = value["episode"]
    if not _EPISODE_FIELDS <= set(episode_value) <= _EPISODE_FIELDS | _OPTIONAL_EPISODE_FIELDS:
        raise ValueError(
            f"episode fields must be {sorted(_EPISODE_FIELDS)} "
            f"(+ optional {sorted(_OPTIONAL_EPISODE_FIELDS)})"
        )
    suite_id = str(value["suite_id"])
    episode = EpisodeSpec(
        suite_id=suite_id,
        strides=tuple(int(item) for item in episode_value["strides"]),
        pair_count=int(episode_value["pair_count"]),
        target_index=int(episode_value["target_index"]),
        samples_per_residue=int(episode_value["samples_per_residue"]),
        postfill_mode=episode_value["postfill_mode"],
        fixed_filler_total=episode_value["fixed_filler_total"],
        seed=int(episode_value["seed"]),
        k_shot=int(episode_value["k_shot"]),
        few_shot_seed=int(episode_value["few_shot_seed"]),
        max_sequence_length=int(episode_value["max_sequence_length"]),
        prompt_template=PromptTemplate.from_dict(value["template"]),
        sample_index_start=int(episode_value.get("sample_index_start", 0)),
        body_spacing=BodySpacingSpec.from_dict(
            episode_value.get("body_spacing", BodySpacingSpec().to_dict())
        ),
    )
    source = value["sentence_source"]
    return SuiteConfig(
        suite_id=suite_id,
        tokenizer=TokenizerSpec.from_dict(value["tokenizer"]),
        token_pool_manifest=(config_path.parent / value["token_pool_manifest"]).resolve(),
        sentence_source=None if source is None else SentenceSourceSpec(**source),
        episode=episode,
    )
