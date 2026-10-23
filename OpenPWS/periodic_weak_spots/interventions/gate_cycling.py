"""Gate cycling: move each compression gate's offset preference (Appendix H).

With adaptive-softmax pooling, window offset ``i`` has its own gate map
``Z_i`` and bias ``b_i`` for keys and values. Cycling by ``delta`` applies the
gate parameters of offset ``i`` at offset ``(i + delta) mod W``; maps and biases
move together and the payload maps stay fixed. If retrieval follows the gates,
accuracy by source-key phase shifts as ``A_delta(r) ~ A_0((r - delta) mod S)``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import torch

from periodic_weak_spots.model import Qwen3KVCForCausalLM
from periodic_weak_spots.model.grouped_linear import PositionGroupedLinear
from periodic_weak_spots.niah_0p6b.config import EpisodeRow, SuiteManifest
from periodic_weak_spots.niah_0p6b.score_hf import score_rows, summarize


@dataclass(frozen=True, slots=True)
class GateCycle:
    """Cycle the key and value gates of ``layers`` by ``delta`` window offsets."""

    name: str
    layers: tuple[int, ...]
    delta: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GateCycle:
        if set(value) != {"name", "layers", "delta"}:
            raise ValueError("a gate cycle has exactly name, layers, and delta")
        return cls(
            str(value["name"]), tuple(int(layer) for layer in value["layers"]), int(value["delta"])
        )


def _cycled_bank(projection: PositionGroupedLinear, delta: int) -> torch.Tensor:
    """``new[offset] = old[(offset - delta) mod W]`` expressed on the weight banks."""

    old = projection.weight_bank.detach()
    new = torch.empty_like(old)
    groups = projection.group_ids.tolist()
    for offset, group in enumerate(groups):
        new[group] = old[groups[(offset - delta) % projection.window]]
    return new


def _gates(model: Qwen3KVCForCausalLM, cycle: GateCycle) -> list[PositionGroupedLinear]:
    config = model.config
    if config.local_attention_mode == "full" or config.compression_pooling != "adaptive_softmax":
        raise ValueError("gate cycling requires adaptive-softmax compression")
    # A bank shared by several offsets cannot move to one new offset.
    if len(set(config.z_group_ids)) != config.compression_window:
        raise ValueError("gate cycling requires one gate bank per window offset")
    if not cycle.layers or any(not 0 <= layer < config.num_hidden_layers for layer in cycle.layers):
        raise ValueError(f"{cycle.name}: layers must lie in [0, {config.num_hidden_layers})")
    gates = []
    for layer in sorted(set(cycle.layers)):
        compressor = model.model.layers[layer].self_attn.compressor
        gates += [gate for gate in (compressor.z_key, compressor.z_value) if gate is not None]
    return gates


@contextmanager
def cycled_gates(model: Qwen3KVCForCausalLM, cycle: GateCycle) -> Iterator[None]:
    """Cycle the key and value gates in place; restore the exact parameters on exit."""

    gates = _gates(model, cycle)
    saved = [
        (
            gate.weight_bank.detach().clone(),
            None if gate.bias is None else gate.bias.detach().clone(),
        )
        for gate in gates
    ]
    try:
        with torch.no_grad():
            for gate in gates:
                gate.weight_bank.copy_(_cycled_bank(gate, cycle.delta))
                if gate.bias is not None:
                    # Biases are per offset; rolling moves them with their banks.
                    gate.bias.copy_(torch.roll(gate.bias, shifts=cycle.delta, dims=0))
        yield
    finally:
        with torch.no_grad():
            for gate, (weight, bias) in zip(gates, saved, strict=True):
                gate.weight_bank.copy_(weight)
                if bias is not None:
                    gate.bias.copy_(bias)


def fold(by_residue: Sequence[float], stride: int) -> list[float]:
    """Average balanced residues mod ``len(by_residue)`` into phases mod ``stride``."""

    if len(by_residue) % stride:
        raise ValueError("stride must divide the residue modulus")
    return [sum(by_residue[r::stride]) / len(by_residue[r::stride]) for r in range(stride)]


Exclusion = Literal["observed_or_source", "observed", "none"]


def translation_r2(
    clean: Sequence[float],
    cycled: Sequence[float],
    delta: int,
    *,
    exclude: Exclusion = "observed_or_source",
) -> float | None:
    """Squared Pearson correlation of predicted vs. observed accuracy changes by phase.

    The prediction is ``A_0((r - delta) mod S) - A_0(r)``. Comparisons involving
    the boundary phase ``S - 1`` are dropped as the observed or the source phase
    (H.2), as the observed phase only (H.3), or not at all. Returns None when the
    correlation is undefined (e.g. ``delta`` is a multiple of ``S``).
    """

    stride = len(clean)
    boundary = stride - 1
    # Observed phase r is predicted from source phase (r - delta) mod S of the clean model.
    skipped = {
        "observed_or_source": lambda r: boundary in (r, (r - delta) % stride),
        "observed": lambda r: r == boundary,
        "none": lambda r: False,
    }[exclude]
    pairs = [
        (clean[(r - delta) % stride] - clean[r], cycled[r] - clean[r])
        for r in range(stride)
        if not skipped(r)
    ]
    changes = torch.tensor(pairs, dtype=torch.float64).T
    # corrcoef is NaN for fewer than two points or a constant series.
    if changes.shape[-1] < 2 or bool((changes.std(dim=1) == 0).any()):
        return None
    return float(torch.corrcoef(changes)[0, 1] ** 2)


def evaluate_gate_cycles(
    model: Qwen3KVCForCausalLM,
    rows: Sequence[EpisodeRow],
    manifest: SuiteManifest,
    cycles: Sequence[GateCycle],
    *,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Full-vocabulary accuracy by phase (residues folded mod S), clean and cycled."""

    stride = model.config.compression_stride
    # Only with W = S does a window offset equal a phase, so moving gates predicts a phase shift.
    if model.config.compression_window != stride:
        raise ValueError("the translation prediction assumes non-overlapping windows (W = S)")

    def run() -> dict[str, Any]:
        summary = summarize(rows, score_rows(model, rows, batch_size=batch_size), manifest)
        summary["full_vocab_accuracy_by_phase"] = fold(
            summary["full_vocab_accuracy_by_residue"], stride
        )
        return summary

    clean = run()
    results: dict[str, Any] = {"stride": stride, "clean": clean, "cycles": []}
    clean_phases = clean["full_vocab_accuracy_by_phase"]
    for cycle in cycles:
        with cycled_gates(model, cycle):
            cycled = run()
        phases = cycled["full_vocab_accuracy_by_phase"]
        results["cycles"].append(
            {
                "name": cycle.name,
                "layers": list(cycle.layers),
                "delta": cycle.delta,
                "full_vocab_accuracy": cycled["full_vocab_accuracy"],
                "full_vocab_accuracy_by_residue": cycled["full_vocab_accuracy_by_residue"],
                "full_vocab_accuracy_by_phase": phases,
                # A_delta(r) ~ A_0((r - delta) mod S): weak phases move with their gates.
                "predicted_accuracy_by_phase": [
                    clean_phases[(r - cycle.delta) % stride] for r in range(stride)
                ],
                "translation_r2": {
                    exclude: translation_r2(clean_phases, phases, cycle.delta, exclude=exclude)
                    for exclude in ("observed_or_source", "observed", "none")
                },
            }
        )
    return results


__all__ = ["GateCycle", "cycled_gates", "evaluate_gate_cycles", "fold", "translation_r2"]
