"""Deterministic high-quality sentence curation into exact token-length bins."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from periodic_weak_spots.tokenization import Tokenizer, decode, token_ids

from .config import SourceSpec


class SentenceSource(Protocol):
    """Indexable rows, such as a Hugging Face ``Dataset``."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class Sentence:
    sentence_id: str
    source_index: int
    text: str
    token_ids: tuple[int, ...]

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["token_ids"] = list(self.token_ids)
        value["token_count"] = self.token_count
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Sentence:
        expected = {"sentence_id", "source_index", "text", "token_ids", "token_count"}
        if set(value) != expected:
            raise ValueError("invalid Sentence record keys")
        result = cls(
            sentence_id=str(value["sentence_id"]),
            source_index=int(value["source_index"]),
            text=str(value["text"]),
            token_ids=tuple(int(token) for token in value["token_ids"]),
        )
        if result.token_count != value["token_count"]:
            raise ValueError("Sentence token_count does not match token_ids")
        return result


class BinCapacityError(RuntimeError):
    """The source has too few sentences of some required token length."""


def curate_sentence_bins(
    source: SentenceSource,
    tokenizer: Tokenizer,
    required_counts: Mapping[int, int],
    *,
    spec: SourceSpec,
) -> tuple[dict[int, list[Sentence]], dict[str, Any]]:
    """Select exact, unique sentences in a deterministic shuffled source order.

    The text is never stripped or normalized.
    """

    indices = list(range(len(source)))
    random.Random(spec.shuffle_seed).shuffle(indices)
    bins: dict[int, list[Sentence]] = {length: [] for length in sorted(required_counts)}
    seen: set[str] = set()
    rejected: Counter[str] = Counter()
    scanned = 0

    for source_index in indices:
        if all(len(bins[length]) >= count for length, count in required_counts.items()):
            break
        scanned += 1
        row = source[source_index]
        text = row.get(spec.text_field) if isinstance(row, Mapping) else None
        if not isinstance(text, str):
            rejected["non_string"] += 1
            continue
        if not text:
            rejected["empty"] += 1
            continue
        # Deduplicate by text so distinct sentence IDs also mean distinct texts.
        if text in seen:
            rejected["duplicate_text"] += 1
            continue
        seen.add(text)
        encoded = token_ids(tokenizer, text)
        length = len(encoded)
        if length not in bins or len(bins[length]) >= required_counts[length]:
            rejected["length_not_needed"] += 1
            continue
        decoded = decode(tokenizer, encoded)
        if token_ids(tokenizer, decoded) != encoded:
            rejected["token_roundtrip"] += 1
            continue
        bins[length].append(
            Sentence(
                sentence_id=f"hq_train_{source_index:09d}",
                source_index=source_index,
                text=text,
                token_ids=tuple(encoded),
            )
        )

    missing = {
        length: required_counts[length] - len(rows)
        for length, rows in bins.items()
        if len(rows) < required_counts[length]
    }
    report = {
        "source": {
            "dataset_id": spec.dataset_id,
            "revision": spec.revision,
            "split": spec.split,
            "text_field": spec.text_field,
            "shuffle_seed": spec.shuffle_seed,
        },
        "source_rows": len(source),
        "rows_scanned": scanned,
        "rejected": dict(sorted(rejected.items())),
        "required_by_length": {str(key): value for key, value in sorted(required_counts.items())},
        "selected_by_length": {str(key): len(value) for key, value in sorted(bins.items())},
    }
    if missing:
        details = ", ".join(f"{length}: need {count} more" for length, count in missing.items())
        raise BinCapacityError(f"pinned sentence source has insufficient exact bins: {details}")
    return bins, report


def load_pinned_source(spec: SourceSpec, *, cache_dir: str | None = None) -> SentenceSource:
    from datasets import load_dataset

    return load_dataset(
        spec.dataset_id,
        name="default",
        split=spec.split,
        revision=spec.revision,
        cache_dir=cache_dir,
    )
