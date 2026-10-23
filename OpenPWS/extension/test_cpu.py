"""Regression checks for metric boundaries, archive scores and scoped adaptation."""

import json
import tempfile
import unittest
from pathlib import Path

from periodic_weak_spots.niah_128k import config, layout
from periodic_weak_spots.niah_128k.score_vllm import summarize as upstream_summary

from .prepare import QUERY, adapter
from .report import HERE, load, summarize
from .utils import DEFAULT_METRIC, score, sha


class ReleaseTests(unittest.TestCase):
    def test_quote_rule_boundaries(self):
        self.assertEqual(DEFAULT_METRIC, "quote_tolerant")
        for text in ("gold", "  gold.", '\n"gold"', "‘gold’", "```gold"):
            self.assertTrue(score(text, "gold"), text)
        for text in ("The answer is gold", "wrong gold", '"wrong gold"', "\" 'gold"):
            self.assertFalse(score(text, "gold"), text)
        self.assertFalse(score('"gold"', "gold", "official"))
        self.assertTrue(score("  gold.extra", "gold", "official"))
        with self.assertRaises(ValueError):
            score("gold", "gold", "unknown")

    def test_historical_counts(self):
        expected = {
            "native": (1573, 1721),
            "hisa": (1499, 1768),
            "dsa": (1561, 1769),
        }
        for name, counts in expected.items():
            rows = load(HERE / "reference", name)
            self.assertEqual(len(rows), 2048)
            for metric, total in zip(("official", "quote_tolerant"), counts, strict=True):
                summary = summarize(rows, metric)
                self.assertEqual(sum(c["correct"] for c in summary["phases"]), total)
                self.assertEqual({c["n"] for c in summary["phases"]}, {256})
        for name, total in (("v4", 1847), ("v41", 1875)):
            summary = summarize(load(HERE / "reference", name), "official", False)
            self.assertEqual(sum(c["correct"] for c in summary["phases"]), total)

    def test_reference_checksums(self):
        manifest = json.loads((HERE / "reference/provenance.json").read_text())
        for name, record in manifest["runs"].items():
            self.assertEqual(sha(HERE / f"reference/{name}.jsonl.gz"), record["export_sha256"])

    def test_upstream_result_format(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "rows/key_value").mkdir(parents=True)
            (root / "metadata.json").write_text(json.dumps({"matrix": {"phases": list(range(8))}}))
            rows = [{"sample_id": str(p), "phase": p, "answer": ["gold"]} for p in range(8)]
            completions = ["gold" if p % 2 == 0 else '"gold"' for p in range(8)]
            for row in rows:
                (root / f"rows/key_value/phase_{row['phase']}.jsonl").write_text(
                    json.dumps(row) + "\n"
                )
            output = {
                "summary": upstream_summary(rows, completions),
                "completions": {r["sample_id"]: c for r, c in zip(rows, completions, strict=True)},
            }
            (root / "v4.json").write_text(json.dumps(output))
            converted = summarize(load(root, "v4", root), "official", False)
            self.assertEqual(converted["accuracy"], output["summary"]["answer_prefix_accuracy"])
            self.assertEqual(converted["accuracy"], 0.5)

    def test_adapter_restores_upstream_even_on_error(self):
        original_spec, original_query = config.TokenizerSpec, layout.QUERY_PREFIX
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "tokenizer.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "test"), adapter(folder) as spec:
                cfg = config.ExperimentConfig(tokenizer=spec)
                self.assertEqual(config.ExperimentConfig.from_dict(cfg.to_dict()), cfg)
                self.assertEqual(layout.QUERY_PREFIX, QUERY["QUERY_PREFIX"])
                bad = cfg.to_dict()
                bad["tokenizer"]["revision"] = "wrong"
                with self.assertRaises(ValueError):
                    config.ExperimentConfig.from_dict(bad)
                raise RuntimeError("test")
        self.assertIs(config.TokenizerSpec, original_spec)
        self.assertEqual(layout.QUERY_PREFIX, original_query)


if __name__ == "__main__":
    unittest.main()
