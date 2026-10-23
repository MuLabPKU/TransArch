"""Deterministic single-token key/value candidates and bindings."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from periodic_weak_spots.tokenization import Tokenizer, decode, token_ids

from .layout import QUERY_MIDDLE, QUERY_PREFIX, QUERY_SUFFIX, render_query


@dataclass(frozen=True, slots=True)
class ContextStableCandidate:
    text: str
    token_id: int


@dataclass(frozen=True, slots=True)
class BindingPair:
    key: str
    key_token_id: int
    value: str
    value_token_id: int


@dataclass(frozen=True, slots=True)
class Binding:
    """The records of one prompt; ``target_index`` selects the queried record."""

    binding_id: int
    pairs: tuple[BindingPair, ...]
    target_index: int

    @property
    def target(self) -> BindingPair:
        return self.pairs[self.target_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "pairs": [asdict(pair) for pair in self.pairs],
            "target_index": self.target_index,
            "target_key": self.target.key,
            "target_value": self.target.value,
        }


def collect_context_stable_token_candidates(
    tokenizer: Tokenizer,
    *,
    min_token_id: int,
    max_characters: int,
) -> list[ContextStableCandidate]:
    """Collect alphabetic single tokens stable in every prompt role."""

    punctuation = [token_ids(tokenizer, text) for text in (":", "。", "｜")]
    if any(len(ids) != 1 for ids in punctuation):
        raise ValueError("record punctuation must be exactly one tokenizer token")
    (colon,), (separator,), (guard,) = punctuation
    query_prefix = token_ids(tokenizer, QUERY_PREFIX)
    query_middle = token_ids(tokenizer, QUERY_MIDDLE)
    query_suffix = token_ids(tokenizer, QUERY_SUFFIX)

    candidates: list[ContextStableCandidate] = []
    for token_id in range(min_token_id, len(tokenizer)):
        text = decode(tokenizer, [token_id])
        if (
            not text
            or len(text) > max_characters
            or not text.isalpha()
            or token_ids(tokenizer, text) != [token_id]
        ):
            continue
        # Each record-boundary context must keep the candidate a single token.
        if (
            token_ids(tokenizer, "。" + text + ":") != [separator, token_id, colon]
            or token_ids(tokenizer, "｜" + text + ":") != [guard, token_id, colon]
            or token_ids(tokenizer, ":" + text + "。") != [colon, token_id, separator]
        ):
            continue
        # The key appears twice in the query and must stay one token in both places.
        if token_ids(tokenizer, render_query(text)) != [
            *query_prefix,
            token_id,
            *query_middle,
            token_id,
            *query_suffix,
        ]:
            continue
        candidates.append(ContextStableCandidate(text, token_id))
    return candidates


def generate_bindings(
    candidates: Sequence[ContextStableCandidate],
    *,
    count: int,
    pair_count: int,
    target_index: int,
    seed: int,
) -> list[Binding]:
    """Sample ``count`` bindings of ``pair_count`` records from the candidate pool."""

    if len(candidates) < 2 * pair_count:
        raise RuntimeError(f"candidate pool has {len(candidates)} entries; need {2 * pair_count}")
    if len({(candidate.text, candidate.token_id) for candidate in candidates}) != len(candidates):
        raise ValueError("candidate pool contains duplicates")
    rng = random.Random(seed)
    result: list[Binding] = []
    for binding_id in range(count):
        # Drawing keys and values without replacement keeps every token distinct within a
        # prompt, so the query key matches exactly one record.
        chosen = rng.sample(candidates, 2 * pair_count)
        pairs = tuple(
            BindingPair(
                key=chosen[index].text,
                key_token_id=chosen[index].token_id,
                value=chosen[pair_count + index].text,
                value_token_id=chosen[pair_count + index].token_id,
            )
            for index in range(pair_count)
        )
        result.append(
            Binding(
                binding_id=binding_id,
                pairs=pairs,
                target_index=target_index,
            )
        )
    return result
