"""Mean-replacement knockout of attention heads at the scoring token (Appendix E).

Each layer's ``o_proj`` input is the concatenation of per-query-head attention
outputs, ``[batch, sequence, heads * head_dim]``. Calibration records these
outputs at the final (scoring) position and averages them over every
calibration prompt, giving one mean vector per ``(layer, query_head)``. A
knockout replaces the outputs of the selected heads at the scoring position
with those means and reruns the rest of the network. Knocking out KV head ``h``
replaces its whole query group; ``K_h`` and ``V_h`` themselves are untouched.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from periodic_weak_spots.model import Qwen3KVCForCausalLM
from periodic_weak_spots.niah_0p6b.config import EpisodeRow, SuiteManifest
from periodic_weak_spots.niah_0p6b.score_hf import exact_length_batches, score_rows, summarize


def _validate_units(model: Qwen3KVCForCausalLM, units: Mapping[int, Sequence[int]]) -> None:
    config = model.config
    if not units or not any(units.values()):
        raise ValueError("patching requires at least one (layer, query head) unit")
    for layer, heads in units.items():
        if not 0 <= layer < config.num_hidden_layers:
            raise ValueError(f"layer {layer} is outside [0, {config.num_hidden_layers})")
        if any(not 0 <= head < config.num_attention_heads for head in heads):
            raise ValueError(
                f"layer {layer}: query heads must lie in [0, {config.num_attention_heads})"
            )


@contextmanager
def head_output_hooks(
    model: Qwen3KVCForCausalLM,
    *,
    captures: list[torch.Tensor] | None = None,
    means: torch.Tensor | None = None,
    units: Mapping[int, Sequence[int]] | None = None,
) -> Iterator[None]:
    """Capture or replace scoring-position head outputs in every layer.

    With ``captures``, each forward appends one ``[batch, layers, heads,
    head_dim]`` CPU tensor. With ``means`` (``[layers, heads, head_dim]``) and
    ``units`` (``layer -> query heads``), the selected heads are replaced.
    Use with :meth:`Qwen3KVCForCausalLM.last_token_logits`, which runs the
    eager path whose final position is the scoring token.
    """

    config = model.config
    layers, heads, head_dim = config.num_hidden_layers, config.num_attention_heads, config.head_dim
    if captures is not None:
        if means is not None or units is not None:
            raise ValueError("capture mode takes no means or units")
    elif means is None or units is None:
        raise ValueError("pass captures, or both means and units")
    else:
        if tuple(means.shape) != (layers, heads, head_dim):
            raise ValueError(f"means must have shape {(layers, heads, head_dim)}")
        _validate_units(model, units)
    pending: list[torch.Tensor] = []

    def hook(layer: int):
        def pre_hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> Any:
            flat = args[0]
            # The o_proj input is head-major, so this view splits it into query heads.
            per_head = flat.view(*flat.shape[:-1], heads, head_dim)
            if captures is not None:
                pending.append(per_head[:, -1].detach().float().cpu())
                # Layers fire in order; the last one completes this forward's capture.
                if layer == layers - 1:
                    captures.append(torch.stack(pending, dim=1))
                    pending.clear()
                return None
            selected = list(units.get(layer, ()))
            if not selected:
                return None
            patched = per_head.clone()
            # Only the scoring position is replaced; every earlier position stays clean.
            patched[:, -1, selected] = means[layer, selected].to(flat.device, flat.dtype)
            return (patched.view_as(flat), *args[1:])

        return pre_hook

    handles = [
        decoder.self_attn.o_proj.register_forward_pre_hook(hook(layer))
        for layer, decoder in enumerate(model.model.layers)
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def fit_head_means(
    model: Qwen3KVCForCausalLM, rows: Sequence[EpisodeRow], *, batch_size: int = 8
) -> torch.Tensor:
    """Average scoring-position head outputs over calibration rows (float32, CPU)."""

    total: torch.Tensor | None = None
    device = next(model.parameters()).device
    for batch in exact_length_batches(rows, batch_size=batch_size):
        captures: list[torch.Tensor] = []
        input_ids = torch.tensor([row.token_ids for row in batch], device=device)
        with head_output_hooks(model, captures=captures):
            model.last_token_logits(input_ids)
        # Accumulate in float64 so the mean does not drift over many calibration rows.
        batch_sum = captures[0].sum(dim=0, dtype=torch.float64)
        total = batch_sum if total is None else total + batch_sum
    if total is None:
        raise ValueError("calibration rows are empty")
    return (total / len(rows)).float()


def kv_head_units(model: Qwen3KVCForCausalLM, layer: int, kv_head: int) -> dict[int, list[int]]:
    """The query heads that read KV head ``kv_head`` in ``layer``."""

    if not 0 <= kv_head < model.config.num_key_value_heads:
        raise ValueError(f"KV head {kv_head} is outside [0, {model.config.num_key_value_heads})")
    group = model.config.query_groups_per_kv_head
    # Matches repeat_kv, which serves KV head h to query heads [h * group, (h + 1) * group).
    return {layer: list(range(kv_head * group, (kv_head + 1) * group))}


def _relative(patched: Sequence[float], clean: Sequence[float]) -> list[float | None]:
    """``(patched - clean) / clean``; undefined (None) where clean accuracy is zero."""

    return [
        None if base == 0 else (value - base) / base
        for value, base in zip(patched, clean, strict=True)
    ]


def sweep_kv_heads(
    model: Qwen3KVCForCausalLM,
    rows: Sequence[EpisodeRow],
    manifest: SuiteManifest,
    means: torch.Tensor,
    *,
    layers: Sequence[int] | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Score the clean model and every single-KV-head knockout by residue mod 24."""

    # Each call scores its own clean baseline, so `--layers` splits are self-contained.
    clean = summarize(rows, score_rows(model, rows, batch_size=batch_size), manifest)
    conditions = []
    for layer in range(model.config.num_hidden_layers) if layers is None else layers:
        for kv_head in range(model.config.num_key_value_heads):
            units = kv_head_units(model, layer, kv_head)
            with head_output_hooks(model, means=means, units=units):
                scores = score_rows(model, rows, batch_size=batch_size)
            summary = summarize(rows, scores, manifest)
            conditions.append(
                {
                    "layer": layer,
                    "kv_head": kv_head,
                    "query_heads": units[layer],
                    "full_vocab_accuracy_by_residue": summary["full_vocab_accuracy_by_residue"],
                    "candidate_accuracy_by_residue": summary["candidate_accuracy_by_residue"],
                    "relative_change_full_vocab_by_residue": _relative(
                        summary["full_vocab_accuracy_by_residue"],
                        clean["full_vocab_accuracy_by_residue"],
                    ),
                    "relative_change_candidate_by_residue": _relative(
                        summary["candidate_accuracy_by_residue"],
                        clean["candidate_accuracy_by_residue"],
                    ),
                }
            )
    return {"clean": clean, "conditions": conditions}


__all__ = ["fit_head_means", "head_output_hooks", "kv_head_units", "sweep_kv_heads"]
