"""Score a Qwen3-KVC checkpoint on a 0.6B NIAH suite with last-token logits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

import torch

from .config import EpisodeRow, SuiteManifest


def exact_length_batches(
    rows: Sequence[EpisodeRow], *, batch_size: int
) -> Iterator[tuple[EpisodeRow, ...]]:
    """Batch rows of identical length; the model rejects padding."""

    groups: dict[int, list[EpisodeRow]] = defaultdict(list)
    for row in rows:
        groups[len(row.token_ids)].append(row)
    for length in sorted(groups):
        group = groups[length]
        for start in range(0, len(group), batch_size):
            yield tuple(group[start : start + batch_size])


@torch.no_grad()
def score_rows(
    model: Any, rows: Sequence[EpisodeRow], *, batch_size: int = 8
) -> dict[str, tuple[bool, bool]]:
    """Return ``sample_id -> (full_vocab_correct, candidate_correct)``."""

    device = next(model.parameters()).device
    scores: dict[str, tuple[bool, bool]] = {}
    for batch in exact_length_batches(rows, batch_size=batch_size):
        input_ids = torch.tensor([row.token_ids for row in batch], dtype=torch.long, device=device)
        logits = model.last_token_logits(input_ids).float()
        candidates = torch.tensor(
            [row.candidate_value_token_ids for row in batch], dtype=torch.long, device=device
        )
        gold_tokens = torch.tensor([row.gold_value_token_id for row in batch], device=device)
        gold_indices = torch.tensor([row.gold_candidate_index for row in batch], device=device)
        # Full-vocabulary accuracy takes the argmax over every token; candidate accuracy
        # restricts it to the 64 names bound in that prompt.
        full_vocab = logits.argmax(dim=1).eq(gold_tokens).tolist()
        candidate = logits.gather(1, candidates).argmax(dim=1).eq(gold_indices).tolist()
        for row, full, cand in zip(batch, full_vocab, candidate, strict=True):
            scores[row.sample_id] = (bool(full), bool(cand))
    if len(scores) != len(rows):
        raise RuntimeError("scoring dropped or duplicated rows")
    return scores


def summarize(
    rows: Sequence[EpisodeRow], scores: dict[str, tuple[bool, bool]], manifest: SuiteManifest
) -> dict[str, Any]:
    """Accuracy per source-key residue ``r = t_K mod 24`` and the best-minus-worst gap.

    A row's ``phase`` field stores this residue; a model with stride ``S`` sees
    phase ``r mod S`` (fold with ``interventions.gate_cycling.fold``).
    """

    totals = [[0, 0, 0] for _ in range(manifest.phase_base)]
    for row in rows:
        full, cand = scores[row.sample_id]
        cell = totals[row.phase]
        cell[0] += 1
        cell[1] += full
        cell[2] += cand
    full_by_residue = [cell[1] / cell[0] for cell in totals]
    candidate_by_residue = [cell[2] / cell[0] for cell in totals]
    return {
        "suite_id": manifest.suite_id,
        "residue_modulus": manifest.phase_base,
        "rows": len(rows),
        "full_vocab_accuracy": sum(cell[1] for cell in totals) / len(rows),
        "candidate_accuracy": sum(cell[2] for cell in totals) / len(rows),
        "full_vocab_accuracy_by_residue": full_by_residue,
        "candidate_accuracy_by_residue": candidate_by_residue,
        "full_vocab_gap": max(full_by_residue) - min(full_by_residue),
    }
