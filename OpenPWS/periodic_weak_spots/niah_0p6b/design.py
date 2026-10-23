"""Residue-conditioned balanced-length draws and the complementary natural fillers."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

from periodic_weak_spots.balanced_lengths import balanced_length_pmf

from .config import Sentence


def design_a_conditional_pmf(*, base: int, residue: int) -> tuple[tuple[int, float], ...]:
    """The balanced-length PMF for modulus ``base`` conditioned on one residue."""

    raw = balanced_length_pmf(base)
    pairs = [(int(length), float(probability)) for length, probability in raw.items()]
    selected = [(length, probability) for length, probability in pairs if length % base == residue]
    total = sum(probability for _, probability in selected)
    return tuple((length, probability / total) for length, probability in selected)


def build_exact_sentence_bins(
    sentences: Sequence[Sentence],
) -> dict[int, tuple[Sentence, ...]]:
    """Group sentences by exact token count, rejecting duplicate IDs or text."""

    bins: dict[int, list[Sentence]] = defaultdict(list)
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for sentence in sentences:
        if sentence.sentence_id in seen_ids or sentence.text in seen_text:
            raise ValueError("sentence bins contain duplicate identity or exact text")
        seen_ids.add(sentence.sentence_id)
        seen_text.add(sentence.text)
        bins[sentence.token_count].append(sentence)
    return {length: tuple(rows) for length, rows in sorted(bins.items())}


def select_design_a_length(*, base: int, residue: int, rng: random.Random) -> int:
    """Draw a length from the residue-conditioned balanced-length PMF."""

    conditional = design_a_conditional_pmf(base=base, residue=residue)
    return rng.choices(
        [length for length, _ in conditional],
        weights=[probability for _, probability in conditional],
        k=1,
    )[0]


def select_complementary_fillers(
    bins: Mapping[int, Sequence[Sentence]],
    *,
    base: int,
    prefill_residue: int,
    fixed_total: int,
    rng: random.Random,
) -> tuple[Sentence, Sentence]:
    """Draw distinct (prefill, postfill) sentences whose lengths sum to ``fixed_total``.

    The prefill length follows the residue-conditioned balanced-length PMF
    (restricted to lengths both bins can realize); ``fixed_total`` is a
    multiple of ``base``, so the postfill has the complementary residue.
    """

    conditional = design_a_conditional_pmf(base=base, residue=prefill_residue)
    # With every bin from 24 to 120 non-empty and a total of 144, every support length is
    # feasible, so the draw follows the balanced PMF exactly.
    feasible = [
        (length, probability)
        for length, probability in conditional
        if bins.get(length) and bins.get(fixed_total - length)
    ]
    if not feasible:
        raise ValueError("sentence bins cannot realize the complementary filler control")
    chosen_length = rng.choices(
        [length for length, _ in feasible],
        weights=[weight for _, weight in feasible],
        k=1,
    )[0]
    prefill = rng.choice(tuple(bins[chosen_length]))
    post_candidates = tuple(
        sentence
        for sentence in bins[fixed_total - chosen_length]
        if sentence.sentence_id != prefill.sentence_id
    )
    if not post_candidates:
        raise ValueError("complementary filler requires two distinct sentences")
    return prefill, rng.choice(post_candidates)


__all__ = [
    "build_exact_sentence_bins",
    "design_a_conditional_pmf",
    "select_complementary_fillers",
    "select_design_a_length",
]
