"""Small I/O helpers and the two explicitly named scoring rules."""

import gzip
import hashlib
import json
from pathlib import Path

from periodic_weak_spots.niah_128k.score_vllm import is_correct

DEFAULT_METRIC = "quote_tolerant"
METRICS = (DEFAULT_METRIC, "official")
LABELS = {"native": "block-indexer（原生）", "hisa": "HISA", "dsa": "DSA"}


def score(completion, gold, metric=DEFAULT_METRIC):
    if metric == "official":
        return is_correct(completion, gold)
    if metric != "quote_tolerant":
        raise ValueError(f"Unknown metric: {metric}")
    # Exact historical rule: one whitespace pass, one quote pass, one whitespace pass.
    return completion.lstrip().lstrip("\"'`“‘«").lstrip().startswith(gold)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def data_hashes(root):
    root = Path(root)
    return {
        str(p.relative_to(root)): sha(p)
        for p in sorted(root.rglob("*.jsonl")) + [root / "metadata.json"]
    }
