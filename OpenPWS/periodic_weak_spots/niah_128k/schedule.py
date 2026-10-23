"""Exact finite schedule and complementary sentence-pair assignment."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

from .source import BinCapacityError, Sentence

# Prefix-length counts per residue for one 256-row block: lengths in [8, 40] congruent to the
# residue mod 8, with mean 24 and second raw moment 604 in every residue. Swapping prefix and
# postfix maps L to 48 - L, which carries residue r onto residue -r mod 8.
FINITE_HISTOGRAM: dict[int, dict[int, int]] = {
    0: {8: 1, 16: 52, 24: 150, 32: 52, 40: 1},
    1: {9: 2, 17: 68, 25: 146, 33: 40},
    2: {10: 4, 18: 84, 26: 140, 34: 28},
    3: {11: 7, 19: 101, 27: 129, 35: 19},
    4: {12: 12, 20: 116, 28: 116, 36: 12},
    5: {13: 19, 21: 129, 29: 101, 37: 7},
    6: {14: 28, 22: 140, 30: 84, 38: 4},
    7: {15: 40, 23: 146, 31: 68, 39: 2},
}

BALANCED_BLOCK_ROWS = 256
SENTENCE_PAIRS_PER_BLOCK = 1_024
DISTINCT_SENTENCES_PER_BLOCK = 2_048


class ScheduleError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PrefillAssignment:
    """One row's natural prefix and postfix and the row holding the same pair reversed."""

    phase: int
    row_index: int
    binding_id: int
    sentence_pair_id: int
    prefill: Sentence
    postfill: Sentence
    role_reversal_phase: int
    role_reversal_row_index: int = -1  # filled once every row is placed

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "row_index": self.row_index,
            "binding_id": self.binding_id,
            "sentence_pair_id": self.sentence_pair_id,
            "prefill": self.prefill.to_dict(),
            "postfill": self.postfill.to_dict(),
            "role_reversal_phase": self.role_reversal_phase,
            "role_reversal_row_index": self.role_reversal_row_index,
        }


def finite_table_payload(*, rows_per_phase: int = BALANCED_BLOCK_ROWS) -> dict[str, dict[str, int]]:
    blocks = _balanced_blocks(rows_per_phase)
    return {
        str(phase): {str(length): count * blocks for length, count in histogram.items()}
        for phase, histogram in FINITE_HISTOGRAM.items()
    }


def phase_lengths(phase: int) -> list[int]:
    """The 256 prefix lengths of one block at ``phase``."""

    if phase not in FINITE_HISTOGRAM:
        raise ValueError(f"unknown phase: {phase}")
    return [length for length, count in FINITE_HISTOGRAM[phase].items() for _ in range(count)]


def validate_finite_table() -> None:
    """Check the pinned histograms for counts, moments, residues, and reflection symmetry."""

    if set(FINITE_HISTOGRAM) != set(range(8)):
        raise ScheduleError("finite table must contain exactly phases 0..7")
    for phase in range(8):
        lengths = phase_lengths(phase)
        if (
            len(lengths) != 256
            or sum(lengths) != 256 * 24
            or sum(length * length for length in lengths) != 256 * 604
        ):
            raise ScheduleError(f"phase {phase} violates pinned finite moments")
        if any(length % 8 != phase for length in lengths):
            raise ScheduleError(f"phase {phase} contains a wrong-residue length")
        complement = (-phase) % 8
        reversed_histogram = Counter(48 - length for length in lengths)
        if reversed_histogram != Counter(phase_lengths(complement)):
            raise ScheduleError(f"phase {phase} is not the reversal of phase {complement}")


def _balanced_blocks(rows_per_phase: int) -> int:
    if rows_per_phase <= 0 or rows_per_phase % BALANCED_BLOCK_ROWS:
        raise ValueError(f"rows_per_phase must be a positive multiple of {BALANCED_BLOCK_ROWS}")
    return rows_per_phase // BALANCED_BLOCK_ROWS


def sentence_capacity_requirements(
    rows_per_phase: int = BALANCED_BLOCK_ROWS,
) -> dict[int, int]:
    """Exact source-sentence capacity for the requested balanced blocks."""

    blocks = _balanced_blocks(rows_per_phase)
    required: Counter[int] = Counter()
    for phase in range(8):
        required.update(phase_lengths(phase))
    required = Counter({length: count * blocks for length, count in required.items()})
    if sum(required.values()) != DISTINCT_SENTENCES_PER_BLOCK * blocks:
        raise AssertionError("finite table does not require exactly 2,048 sentences")
    return dict(sorted(required.items()))


@dataclass(frozen=True, slots=True)
class _Raw:
    pair_id: int
    prefill: Sentence
    postfill: Sentence
    partner_phase: int


def _build_base_prefill_schedule(
    bins: dict[int, list[Sentence]], *, seed: int = 20260721
) -> list[PrefillAssignment]:
    """Consume exactly 2,048 sentences and build all 2,048 oriented rows."""

    validate_finite_table()
    required = sentence_capacity_requirements()
    available: dict[int, list[Sentence]] = {}
    for length, count in required.items():
        selected = list(bins[length][:count])
        if any(sentence.token_count != length for sentence in selected):
            raise ScheduleError(f"length-{length} bin contains a mislabeled sentence")
        available[length] = selected
    all_selected = [sentence for rows in available.values() for sentence in rows]
    if len({sentence.sentence_id for sentence in all_selected}) != 2_048:
        raise ScheduleError("selected sentence IDs must be globally distinct")
    cursors = {length: 0 for length in available}

    def take(length: int) -> Sentence:
        cursor = cursors[length]
        sentence = available[length][cursor]
        cursors[length] += 1
        return sentence

    raw_by_phase: dict[int, list[_Raw]] = {phase: [] for phase in range(8)}
    pair_id = 0

    # Non-self complementary phases share 256 pairs in opposite orientations.
    for phase in (1, 2, 3):
        complement = (-phase) % 8
        for prefill_length in phase_lengths(phase):
            left = take(prefill_length)
            right = take(48 - prefill_length)
            raw_by_phase[phase].append(_Raw(pair_id, left, right, complement))
            raw_by_phase[complement].append(_Raw(pair_id, right, left, phase))
            pair_id += 1

    # Self-complementary phases contain both orientations internally.
    for phase in (0, 4):
        histogram = FINITE_HISTOGRAM[phase]
        for length, count in histogram.items():
            partner_length = 48 - length
            if length > partner_length:
                continue
            pair_count = count // 2 if length == partner_length else count
            if length == partner_length and count % 2:
                raise ScheduleError("center-bin count must be even")
            for _ in range(pair_count):
                left = take(length)
                right = take(partner_length)
                raw_by_phase[phase].append(_Raw(pair_id, left, right, phase))
                raw_by_phase[phase].append(_Raw(pair_id, right, left, phase))
                pair_id += 1

    if pair_id != 1_024:
        raise ScheduleError(f"built {pair_id} sentence pairs, expected 1,024")
    if any(cursors[length] != required[length] for length in required):
        raise ScheduleError("schedule did not consume exact sentence-bin capacity")

    # Apply one permutation per complementary group so opposite orientations align.
    ordered: dict[int, list[_Raw]] = {}
    for phase in (0, 1, 2, 3, 4):
        complement = (-phase) % 8
        permutation = list(range(256))
        random.Random(seed + phase).shuffle(permutation)
        ordered[phase] = [raw_by_phase[phase][index] for index in permutation]
        if complement != phase:
            ordered[complement] = [raw_by_phase[complement][index] for index in permutation]

    # Complementary phases share the permutation and the binding rotation, so when r != -r mod 8
    # a row and its partner share row index and binding and differ only in sentence order.
    rows: list[PrefillAssignment] = []
    by_orientation: dict[tuple[int, str], tuple[int, int]] = {}
    for phase in range(8):
        group = min(phase, (-phase) % 8)
        rotation = (37 * group) % 256
        for row_index, raw in enumerate(ordered[phase]):
            assignment = PrefillAssignment(
                phase=phase,
                row_index=row_index,
                binding_id=(row_index + rotation) % 256,
                sentence_pair_id=raw.pair_id,
                prefill=raw.prefill,
                postfill=raw.postfill,
                role_reversal_phase=raw.partner_phase,
            )
            rows.append(assignment)
            by_orientation[(raw.pair_id, raw.prefill.sentence_id)] = (phase, row_index)

    completed: list[PrefillAssignment] = []
    for row in rows:
        partner = by_orientation.get((row.sentence_pair_id, row.postfill.sentence_id))
        if partner is None:
            raise ScheduleError("role-reversal partner is missing")
        completed.append(replace(row, role_reversal_row_index=partner[1]))

    validate_prefill_schedule(completed)
    return completed


def build_prefill_schedule(
    bins: dict[int, list[Sentence]],
    *,
    seed: int = 20260721,
    rows_per_phase: int = BALANCED_BLOCK_ROWS,
) -> list[PrefillAssignment]:
    """Build one or more independent balanced 256-row blocks per phase.

    The first block is exactly the published schedule. Additional blocks use
    disjoint sentences, bindings, pair IDs, and deterministic seed offsets.
    """

    blocks = _balanced_blocks(rows_per_phase)
    base_required = sentence_capacity_requirements()
    required = sentence_capacity_requirements(rows_per_phase)
    missing = {
        length: count - len(bins.get(length, []))
        for length, count in required.items()
        if len(bins.get(length, [])) < count
    }
    if missing:
        raise BinCapacityError(
            "sentence-bin capacity failure: "
            + ", ".join(f"{length}: need {count} more" for length, count in missing.items())
        )

    combined: list[PrefillAssignment] = []
    for block in range(blocks):
        block_bins = {
            length: bins[length][block * count : (block + 1) * count]
            for length, count in base_required.items()
        }
        for row in _build_base_prefill_schedule(block_bins, seed=seed + block):
            combined.append(
                replace(
                    row,
                    row_index=row.row_index + block * BALANCED_BLOCK_ROWS,
                    binding_id=row.binding_id + block * BALANCED_BLOCK_ROWS,
                    sentence_pair_id=row.sentence_pair_id + block * SENTENCE_PAIRS_PER_BLOCK,
                    role_reversal_row_index=row.role_reversal_row_index
                    + block * BALANCED_BLOCK_ROWS,
                )
            )
    validate_prefill_schedule(combined)
    return combined


def validate_prefill_schedule(rows: list[PrefillAssignment]) -> None:
    """Check pairing, sentence use, binding bijectivity, histograms, and role reversal."""

    if not rows or len(rows) % 8:
        raise ScheduleError("schedule row count must be positive and divisible by eight")
    rows_per_phase = len(rows) // 8
    blocks = _balanced_blocks(rows_per_phase)
    expected_pairs = SENTENCE_PAIRS_PER_BLOCK * blocks
    expected_sentences = DISTINCT_SENTENCES_PER_BLOCK * blocks
    indexed = {(row.phase, row.row_index): row for row in rows}
    if len(indexed) != len(rows):
        raise ScheduleError("duplicate phase/row coordinates")
    pair_ids = {row.sentence_pair_id for row in rows}
    if len(pair_ids) != expected_pairs:
        raise ScheduleError(f"schedule must contain {expected_pairs} sentence pairs")
    pair_row_counts = Counter(row.sentence_pair_id for row in rows)
    if set(pair_row_counts.values()) != {2}:
        raise ScheduleError("every sentence pair must have two orientations")
    sentence_pairs: dict[str, int] = {}
    sentence_occurrences: Counter[str] = Counter()
    for row in rows:
        for sentence in (row.prefill, row.postfill):
            previous = sentence_pairs.setdefault(sentence.sentence_id, row.sentence_pair_id)
            if previous != row.sentence_pair_id:
                raise ScheduleError("a sentence is reused by multiple pairs")
            sentence_occurrences[sentence.sentence_id] += 1
    if len(sentence_occurrences) != expected_sentences or set(sentence_occurrences.values()) != {2}:
        raise ScheduleError(
            f"schedule must use {expected_sentences} distinct sentences exactly twice"
        )
    for phase in range(8):
        phase_rows = [row for row in rows if row.phase == phase]
        if len(phase_rows) != rows_per_phase:
            raise ScheduleError(f"phase {phase} does not contain {rows_per_phase} rows")
        if {row.binding_id for row in phase_rows} != set(range(rows_per_phase)):
            raise ScheduleError(f"phase {phase} binding assignment is not bijective")
        expected_histogram = Counter(
            {length: count * blocks for length, count in FINITE_HISTOGRAM[phase].items()}
        )
        if Counter(row.prefill.token_count for row in phase_rows) != expected_histogram:
            raise ScheduleError(f"phase {phase} has the wrong finite histogram")
        for row in phase_rows:
            if row.prefill.token_count + row.postfill.token_count != 48:
                raise ScheduleError("natural pre/post lengths do not sum to 48")
            partner = indexed[(row.role_reversal_phase, row.role_reversal_row_index)]
            if (
                partner.sentence_pair_id != row.sentence_pair_id
                or partner.prefill.sentence_id != row.postfill.sentence_id
                or partner.postfill.sentence_id != row.prefill.sentence_id
            ):
                raise ScheduleError("complementary role reversal is not exact")
