#!/usr/bin/env python3
"""Aggregate the 2x2 ablation × {GQA, MQA-absorb} commonsense results into a table.

Reads ``{out_root}/<mode>/**/results_*.json`` for each of the 9 modes and prints:
  1. Per-task accuracy table (modes as columns, tasks as rows)
  2. Per-mode average + delta vs MLA
  3. GQA-vs-absorb consistency check (should be ≈identical since they share weights)

Usage: python -m scripts.summarize_commonsense_2x2 <out_root>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional


MODES = (
    "mla",
    "neigh_nohess_gqa", "neigh_nohess_absorb",
    "neigh_hess_gqa",   "neigh_hess_absorb",
    "sim_nohess_gqa",   "sim_nohess_absorb",
    "sim_hess_gqa",     "sim_hess_absorb",
)
TASKS = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "openbookqa", "boolq")
METRIC_PRIORITY = ("acc_norm", "acc")


def load_results(out_root: Path, mode: str) -> Optional[dict]:
    cands = sorted((out_root / mode).glob("**/results_*.json"))
    return json.loads(cands[-1].read_text()) if cands else None


def pick_metric(row: dict) -> Optional[float]:
    keys = {k.split(",")[0]: float(v) for k, v in row.items()
            if isinstance(v, (int, float)) and not k.endswith("stderr")}
    for m in METRIC_PRIORITY:
        if m in keys:
            return keys[m]
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <out_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    all_res = {m: load_results(root, m) for m in MODES}

    # Build (mode, task) -> acc grid.
    grid: dict[str, dict[str, Optional[float]]] = {m: {} for m in MODES}
    for m, r in all_res.items():
        if r is None:
            for t in TASKS:
                grid[m][t] = None
            continue
        results = r.get("results", {})
        for t in TASKS:
            grid[m][t] = pick_metric(results[t]) if t in results else None

    # Per-task table.
    print(f"\n{'='*120}")
    print(f"  Commonsense (2×2 ablation × {{GQA, MQA-absorb}}) from {root}")
    print(f"{'='*120}\n")
    col_w = 12
    header = f"{'task':<16}" + "".join(f"{m[:col_w]:>{col_w + 2}}" for m in MODES)
    print(header)
    print("-" * len(header))
    for t in TASKS:
        row = f"{t:<16}"
        for m in MODES:
            v = grid[m][t]
            row += f"{v:>{col_w + 2}.4f}" if v is not None else f"{'—':>{col_w + 2}}"
        print(row)
    # Averages (only over tasks present for ALL modes that have results).
    avg_row = f"{'AVG':<16}"
    mla_avg = None
    for m in MODES:
        present = [grid[m][t] for t in TASKS if grid[m][t] is not None]
        if not present:
            avg_row += f"{'—':>{col_w + 2}}"; continue
        avg = sum(present) / len(present)
        avg_row += f"{avg:>{col_w + 2}.4f}"
        if m == "mla":
            mla_avg = avg
    print("-" * len(header))
    print(avg_row)

    # Delta vs MLA.
    if mla_avg is not None:
        delta_row = f"{'Δpp vs MLA':<16}"
        for m in MODES:
            present = [grid[m][t] for t in TASKS if grid[m][t] is not None]
            if not present or m == "mla":
                delta_row += f"{'—' if m != 'mla' else '0.00':>{col_w + 2}}"
                continue
            avg = sum(present) / len(present)
            delta_row += f"{100*(avg - mla_avg):>+{col_w + 2}.2f}"
        print(delta_row)

    # GQA vs absorb consistency: per-config diff.
    print(f"\n{'='*60}")
    print("  GQA-vs-Absorb consistency (same weights, different decode path)")
    print(f"{'='*60}\n")
    print(f"{'config':<16} {'avg|GQA−Abs|':>15}  {'max|GQA−Abs|':>15}  per-task diffs")
    print("-" * 100)
    for cfg in ("neigh_nohess", "neigh_hess", "sim_nohess", "sim_hess"):
        gqa = grid.get(f"{cfg}_gqa", {})
        abs_ = grid.get(f"{cfg}_absorb", {})
        diffs = []
        per_task = []
        for t in TASKS:
            g, a = gqa.get(t), abs_.get(t)
            if g is not None and a is not None:
                d = abs(g - a)
                diffs.append(d)
                per_task.append(f"{t}={d:.4f}")
        if diffs:
            mean_d = sum(diffs) / len(diffs)
            max_d = max(diffs)
            print(f"{cfg:<16} {mean_d:>15.4f}  {max_d:>15.4f}  {' '.join(per_task)}")
        else:
            print(f"{cfg:<16} {'(no overlap)':>15}")

    # README-style baseline×winner table for quick read.
    winner = min(
        (m for m in MODES if m != "mla" and all_res.get(m) is not None),
        key=lambda m: sum(1 for t in TASKS if grid[m][t] is None) or (
            -sum(grid[m][t] for t in TASKS if grid[m][t] is not None)
        ),
        default=None,
    )
    if winner is not None and all_res.get("mla"):
        print(f"\n{'='*60}")
        print(f"  README-style: MLA vs winner ({winner})")
        print(f"{'='*60}\n")
        print(f"| Task          | MLA     | {winner:>15} | Δ (pp)  |")
        print(f"|---------------|---------|-----------------|---------|")
        for t in TASKS:
            mla_v = grid["mla"][t]
            w_v = grid[winner][t]
            if mla_v is None or w_v is None:
                continue
            d = 100 * (w_v - mla_v)
            print(f"| {t:<13} | {mla_v:.4f}  | {w_v:>15.4f} | {d:+7.2f} |")
        mla_avg_ = sum(grid["mla"][t] for t in TASKS if grid["mla"][t] is not None) / sum(
            1 for t in TASKS if grid["mla"][t] is not None)
        w_avg = sum(grid[winner][t] for t in TASKS if grid[winner][t] is not None) / sum(
            1 for t in TASKS if grid[winner][t] is not None)
        print(f"| **average**   | **{mla_avg_:.4f}** | **{w_avg:.4f}**      | "
              f"**{100*(w_avg - mla_avg_):+.2f}** |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
