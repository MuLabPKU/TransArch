"""Greedy vLLM decoding and answer-prefix accuracy by target residue mod 8.

Prompts are passed as raw text, as with ``/v1/completions``: vLLM applies the
tokenizer's default special tokens and no chat template.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .layout import PAIR_FORM


def load_rows(output_root: str | Path) -> list[dict[str, Any]]:
    """All prompt rows of a built dataset, in phase order."""

    root = Path(output_root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for phase in metadata["matrix"]["phases"]:
        path = root / "rows" / PAIR_FORM / f"phase_{phase}.jsonl"
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle)
    return rows


def is_correct(completion: str, gold: str) -> bool:
    # Decoding continues past the value, so only the start of the completion is scored.
    return completion.strip().startswith(gold)


def summarize(rows: Sequence[dict[str, Any]], completions: Sequence[str]) -> dict[str, Any]:
    """Accuracy per residue mod 8 (a row's ``phase`` field) and the best-minus-worst gap."""

    by_residue: dict[int, list[bool]] = {}
    for row, completion in zip(rows, completions, strict=True):
        by_residue.setdefault(row["phase"], []).append(is_correct(completion, row["answer"][0]))
    accuracy = {residue: sum(hits) / len(hits) for residue, hits in sorted(by_residue.items())}
    total = [hit for hits in by_residue.values() for hit in hits]
    return {
        "rows": len(total),
        "answer_prefix_accuracy": sum(total) / len(total),
        "answer_prefix_accuracy_by_residue": {str(r): value for r, value in accuracy.items()},
        "gap": max(accuracy.values()) - min(accuracy.values()),
    }


def generate(
    model: str, prompts: Sequence[str], *, max_tokens: int = 8, **engine_args: Any
) -> list[str]:
    """Greedy completions of raw text prompts, in input order."""

    from vllm import LLM, SamplingParams

    llm = LLM(model=model, **engine_args)
    outputs = llm.generate(list(prompts), SamplingParams(temperature=0, max_tokens=max_tokens))
    return [output.outputs[0].text for output in outputs]
