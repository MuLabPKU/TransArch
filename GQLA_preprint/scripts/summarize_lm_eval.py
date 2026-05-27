"""Pretty-print lm-eval-harness results from a three-variant run.

Reads ``{out_root}/{mode}/**/results_*.json`` for mode in {baseline, gqla,
gqla-absorb} and prints a side-by-side table with deltas vs baseline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


MODES = ("baseline", "gqla", "gqla-absorb")
METRIC_PRIORITY = ("acc_norm", "acc")  # acc_norm preferred when available


def load_results(out_root: Path, mode: str):
    cands = sorted((out_root / mode).glob("**/results_*.json"))
    if not cands:
        return None
    return json.loads(cands[-1].read_text())


def pick_metric(row: dict) -> tuple[str, float, float | None]:
    """Return (metric_name, value, stderr) preferring acc_norm over acc."""
    keys = {k.split(",")[0]: float(v) for k, v in row.items()
            if isinstance(v, (int, float))}
    for m in METRIC_PRIORITY:
        if m in keys:
            return m, keys[m], keys.get(f"{m}_stderr")
    # fallback: first numeric key
    k, v = next(iter(keys.items()))
    return k, v, None


def main():
    root = Path(sys.argv[1])
    all_res = {m: load_results(root, m) for m in MODES}

    tasks: set[str] = set()
    for r in all_res.values():
        if r:
            tasks.update(r.get("results", {}).keys())
    tasks = sorted(tasks)

    print(f"\n{'='*88}")
    print(f"  Commonsense lm-eval (from {root})")
    print(f"{'='*88}\n")

    if all_res.get("baseline") is None:
        print("  (no baseline results found)")
    header = f"  {'task':18s}  {'metric':10s}  {'baseline':>11s}  {'gqla':>11s}  {'Δ_g':>8s}  {'gqla-absorb':>11s}  {'Δ_m':>8s}  {'|g-m|':>8s}"
    print(header)
    print("  " + "-" * (len(header)-2))

    avg_acc = {m: [] for m in MODES}
    for task in tasks:
        cells = {}
        metric_name = None
        for m in MODES:
            r = all_res.get(m)
            row = (r or {}).get("results", {}).get(task)
            if row is None:
                cells[m] = None
                continue
            mn, val, _ = pick_metric(row)
            cells[m] = val
            metric_name = mn
            avg_acc[m].append(val)
        base = cells.get("baseline")
        gqla = cells.get("gqla")
        mqa = cells.get("gqla-absorb")

        def fmt(v): return f"{v*100:7.2f}%" if v is not None else "    --  "
        def dlt(a, b): return f"{(a-b)*100:+6.2f}" if (a is not None and b is not None) else "   --  "

        gm_diff = (
            f"{abs(gqla - mqa)*100:7.4f}"
            if (gqla is not None and mqa is not None) else "   --  "
        )
        print(f"  {task:18s}  {metric_name or '--':10s}  "
              f"{fmt(base):>11s}  {fmt(gqla):>11s}  {dlt(gqla,base):>8s}  "
              f"{fmt(mqa):>11s}  {dlt(mqa,base):>8s}  {gm_diff:>8s}")

    print()
    print(f"  {'AVERAGE':18s}  {'(per-task)':10s}  "
          f"{(sum(avg_acc['baseline'])/max(len(avg_acc['baseline']),1))*100:10.2f}%  "
          f"{(sum(avg_acc['gqla'])/max(len(avg_acc['gqla']),1))*100:10.2f}%  "
          f"{'':>8s}  "
          f"{(sum(avg_acc['gqla-absorb'])/max(len(avg_acc['gqla-absorb']),1))*100:10.2f}%")
    print()


if __name__ == "__main__":
    main()
