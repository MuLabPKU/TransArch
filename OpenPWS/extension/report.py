"""Rescore saved completions and reproduce the three-model plot without model weights."""

import argparse
import csv
import json
from pathlib import Path

from .utils import DEFAULT_METRIC, LABELS, METRICS, atomic, read_rows, score

HERE = Path(__file__).parent


def summarize(rows, metric, actual_phase=True):
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicated observations")
    phase_key = "actual_source_residue" if actual_phase else "phase"
    cells = []
    for phase in range(8):
        selected = [r for r in rows if r[phase_key] == phase]
        if not selected:
            raise ValueError("All eight phases are required for a phase comparison")
        correct = sum(score(r["completion"], r["gold"], metric) for r in selected)
        cells.append(
            dict(phase=phase, n=len(selected), correct=correct, accuracy=correct / len(selected))
        )
    values = [c["accuracy"] for c in cells]
    return dict(
        metric=metric,
        rows=len(rows),
        phases=cells,
        accuracy=sum(c["correct"] for c in cells) / len(rows),
        gap=max(values) - min(values),
    )


def load(directory, name, dataset=None):
    plain = directory / f"{name}.jsonl"
    compressed = directory / f"{name}.jsonl.gz"
    if plain.exists() or compressed.exists():
        return read_rows(plain if plain.exists() else compressed)
    # Accept the unmodified upstream CLI's {summary, completions} output as well.
    if dataset is None:
        raise ValueError(f"Missing {name}.jsonl[.gz]; upstream JSON requires --deepseek-dataset")
    from periodic_weak_spots.niah_128k.score_vllm import load_rows

    completions = json.loads((directory / f"{name}.json").read_text())["completions"]
    source = load_rows(dataset)
    if set(completions) != {r["sample_id"] for r in source}:
        raise ValueError("Upstream completions must cover exactly the supplied dataset")
    return [
        dict(
            sample_id=r["sample_id"],
            phase=r["phase"],
            gold=r["answer"][0],
            completion=completions[r["sample_id"]],
        )
        for r in source
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm-results", type=Path, default=HERE / "reference")
    parser.add_argument("--deepseek-results", type=Path, default=HERE / "reference")
    parser.add_argument("--deepseek-dataset", type=Path, help="Dataset for upstream JSON results")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--glm-metric", choices=METRICS, default=DEFAULT_METRIC)
    parser.add_argument("--font", type=Path, help="Optional CJK font; default uses English labels")
    args = parser.parse_args()
    rows = {name: load(args.glm_results, name) for name in LABELS}
    native = {r["sample_id"]: r for r in rows["native"]}
    for arm in rows.values():
        if {r["sample_id"] for r in arm} != native.keys():
            raise ValueError("GLM variants must cover identical sample IDs")
        for r in arm:
            if any(
                r[k] != native[r["sample_id"]][k]
                for k in ("gold", "phase", "actual_source_residue", "binding_id")
            ):
                raise ValueError("GLM binding/phase identity differs")
    summaries = {
        name: {metric: summarize(arm, metric) for metric in METRICS} for name, arm in rows.items()
    }
    for name in ("v4", "v41"):
        summaries[name] = {
            "official": summarize(
                load(args.deepseek_results, name, args.deepseek_dataset), "official", False
            )
        }
    atomic(args.output / "metrics.json", {"glm_default_metric": args.glm_metric, "runs": summaries})
    with (args.output / "phase_accuracy.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model_or_variant", "metric", "phase", "correct", "n", "accuracy"])
        for name, metrics in summaries.items():
            for metric, summary in metrics.items():
                for c in summary["phases"]:
                    writer.writerow([name, metric, c["phase"], c["correct"], c["n"], c["accuracy"]])
    lines = [
        "# Results",
        "",
        f"GLM metric: `{args.glm_metric}`. DeepSeek metric: `official`.",
        "",
        "| Model / variant | n | Accuracy | Phase gap (pp) |",
        "|---|---:|---:|---:|",
    ]
    for name in ("v4", "v41", *LABELS):
        s = summaries[name][args.glm_metric if name in LABELS else "official"]
        lines.append(
            f"| {LABELS.get(name, name)} | {s['rows']} | "
            f"{100 * s['accuracy']:.2f}% | {100 * s['gap']:.2f} |"
        )
    lines += [
        "",
        "The JSON and CSV retain both GLM metrics. See extension/README.md for exact definitions.",
        "Saved historical observations are not a new inference run of the refactored code.",
        "",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import PercentFormatter

    font = FontProperties(fname=str(args.font)) if args.font else FontProperties()
    zh = bool(args.font)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharey=True)
    paper = json.loads((HERE / "reference/paper_table7.json").read_text())["models"]
    for ax, name in zip(axes[:2], ("v4", "v41"), strict=True):
        ax.plot(
            range(8),
            [100 * c["accuracy"] for c in summaries[name]["official"]["phases"]],
            "o-",
            color="#2166ac",
            lw=2.2,
            ms=5,
            label="本地复现（vLLM）" if zh else "Local (vLLM)",
        )
        ax.plot(
            range(8),
            paper[name]["accuracy_percent"],
            "s--",
            color="#d66b20",
            lw=2,
            ms=5,
            markerfacecolor="white",
            label="原文 Table 7" if zh else "Paper Table 7",
        )
        ax.set_title(paper[name]["name"], fontsize=13, pad=12)
        ax.legend(loc="lower right", frameon=False, prop=font)
    styles = {
        "native": ("#2166ac", "o", "-"),
        "hisa": ("#d66b20", "^", "--"),
        "dsa": ("#168576", "D", "-."),
    }
    for name, (color, marker, style) in styles.items():
        label = LABELS[name] if zh or name != "native" else "block-indexer (native)"
        axes[2].plot(
            range(8),
            [100 * c["accuracy"] for c in summaries[name][args.glm_metric]["phases"]],
            color=color,
            marker=marker,
            linestyle=style,
            lw=2.2,
            ms=5,
            label=label,
        )
    axes[2].set_title("GLM-5.3-Flash", fontsize=13, pad=12)
    axes[2].legend(loc="upper right", frameon=False, prop=font)
    lowest = min(
        100 * c["accuracy"]
        for name, metrics in summaries.items()
        for c in metrics[args.glm_metric if name in LABELS else "official"]["phases"]
    )
    lower = min(75, int(lowest // 5) * 5)
    for ax in axes:
        ax.set_xticks(range(8))
        ax.set_ylim(lower, 101)
        ax.set_yticks(range(lower, 101, 5))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel("相位" if zh else "Phase", fontproperties=font, labelpad=8)
        ax.set_ylabel("准确率" if zh else "Accuracy", fontproperties=font, labelpad=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.14, top=0.90, wspace=0.23)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.output / f"phase_comparison.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
