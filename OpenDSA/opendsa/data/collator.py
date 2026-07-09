"""collator.py — batch packed DSA sequences.

Each example: input_ids [L], labels [L], cu_seqlens [n+1]. All packed examples in
a stage share the same L, so batching input_ids/labels is trivial. cu_seqlens has
variable length; we keep it as a python list per row (the DSA attention consumes
per-row cu_seqlens). For batch>1 with per-doc masking, each row carries its own
boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class DSADataCollator:
    pad_token_id: int = 0

    def __call__(self, features: List[dict]) -> dict:
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        cu = [torch.tensor(f["cu_seqlens"], dtype=torch.long) for f in features]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
            "cu_seqlens_list": cu,   # consumed by the trainer -> set_cu_seqlens
        }
