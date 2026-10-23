"""Checked-in animal-key / human-name-value single-token pool."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from periodic_weak_spots.tokenization import decode, token_ids

from .config import Candidate, TokenizerSpec
from .template import PromptTemplate


@dataclass(frozen=True, slots=True)
class AnimalNamePool:
    """The 64 animal keys and 64 name values, with the template they were checked against."""

    tokenizer: TokenizerSpec
    template: PromptTemplate
    k_shot: int
    keys: tuple[Candidate, ...]
    values: tuple[Candidate, ...]


def _candidates(
    items: list[dict[str, Any]], words: list[str], pattern: str
) -> tuple[Candidate, ...]:
    if [item["source_word"] for item in items] != words:
        raise ValueError("pool candidate order differs from source word order")
    compiled = re.compile(pattern)
    candidates = []
    for item in items:
        word, text = item["source_word"], item["text"]
        if text != f" {word}" or compiled.fullmatch(text) is None:
            raise ValueError(f"pool word {word!r} violates its text rule")
        candidates.append(
            Candidate(
                f"allowlist_{word.casefold()}_{item['token_id']}", text, int(item["token_id"])
            )
        )
    return tuple(candidates)


def load_pool(path: str | Path, *, tokenizer: Any | None = None) -> AnimalNamePool:
    """Load the pool; with a tokenizer, require every entry to be an exact single token."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 2 or value.get("curation") != "allowlist":
        raise ValueError("unsupported pool schema")
    selection, source = value["selection"], value["source"]
    keys = _candidates(value["keys"], source["animal_words"], selection["animal_text_pattern"])
    values = _candidates(value["values"], source["name_words"], selection["name_text_pattern"])
    if (len(keys), len(values)) != (selection["animal_count"], selection["name_count"]):
        raise ValueError("pool sizes differ from the configured counts")
    everything = keys + values
    if len({c.token_id for c in everything}) != len(everything):
        raise ValueError("pool token IDs must be distinct and key/value-disjoint")
    if {c.text.strip().casefold() for c in keys} & {c.text.strip().casefold() for c in values}:
        raise ValueError("animal and name words must be casefold-disjoint")
    if tokenizer is not None:
        special_ids = set(getattr(tokenizer, "all_special_ids", ()))
        for candidate in everything:
            if (
                candidate.token_id in special_ids
                or token_ids(tokenizer, candidate.text) != [candidate.token_id]
                or decode(tokenizer, [candidate.token_id]) != candidate.text
            ):
                raise ValueError(f"pool entry {candidate.text!r} is not an exact single token")
    return AnimalNamePool(
        tokenizer=TokenizerSpec.from_dict(value["tokenizer"]),
        template=PromptTemplate.from_dict(value["template"]),
        k_shot=int(selection["k_shot"]),
        keys=keys,
        values=values,
    )
