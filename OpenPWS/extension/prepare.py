"""Adapt only tokenizer identity and query quoting; reuse the upstream 128K builder."""

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from periodic_weak_spots.niah_128k import bindings, config, layout
from periodic_weak_spots.niah_128k.build import publish_dataset
from periodic_weak_spots.niah_128k.schedule import sentence_capacity_requirements
from periodic_weak_spots.niah_128k.source import curate_sentence_bins, load_pinned_source
from periodic_weak_spots.niah_128k.validate import validate_dataset
from periodic_weak_spots.tokenization import load_tokenizer

from .utils import atomic, data_hashes, sha

QUERY = {
    "QUERY_PREFIX": "Now answer the question using the key-value pairs above. "
    'What is the value for key "',
    "QUERY_MIDDLE": '"? Answer with only the value. The value for key "',
    "QUERY_SUFFIX": '" is ',
}
PAPER_FILLER_SHA = "51eaf041f492d8fdfdcf87be8756d23888eda0d3cab6816b3c3f11e7682de337"


@contextmanager
def adapter(model):
    tokenizer_sha = sha(Path(model) / "tokenizer.json")

    @dataclass(frozen=True, slots=True)
    class TokenizerSpec:
        model_id: str = "zai-org/GLM-5.3-Flash"
        revision: str = "local-sha256:" + tokenizer_sha

        def __post_init__(self):
            if (self.model_id, self.revision) != (
                "zai-org/GLM-5.3-Flash",
                "local-sha256:" + tokenizer_sha,
            ):
                raise ValueError("Tokenizer differs from dataset identity")

    changes = [(config, "TokenizerSpec", TokenizerSpec)]
    changes += [
        (module, name, text) for module in (bindings, layout) for name, text in QUERY.items()
    ]
    original = [(module, name, getattr(module, name)) for module, name, _ in changes]
    try:
        for module, name, value in changes:
            setattr(module, name, value)
        yield TokenizerSpec()
    finally:
        for module, name, value in original:
            setattr(module, name, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local GLM model/tokenizer")
    parser.add_argument("--filler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-phase", type=int, default=256)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--audit-output", type=Path, help="Optional separate validation record")
    args = parser.parse_args()
    root = args.output
    with adapter(args.model) as spec:
        n = args.rows_per_phase
        cfg = config.ExperimentConfig(
            tokenizer=spec,
            dataset_role="custom",
            experiment_id="glm53flash-128k-balanced-target64k-16000pairs-quoted-v1",
            sample_prefix="glm53flash",
            phase_design=config.PhaseDesignSpec(
                rows_per_phase=n, sentence_pair_count=4 * n, distinct_sentence_count=8 * n
            ),
        )
        tok = load_tokenizer(spec.model_id, spec.revision, local_path=args.model)
        identity = {
            "config": cfg.to_dict(),
            "query": QUERY,
            "filler_sha256": sha(args.filler),
            "tokenizer_sha256": sha(args.model / "tokenizer.json"),
        }
        if args.validate_only:
            saved = json.loads((root / "metadata.json").read_text())
            if saved["config"] != cfg.to_dict():
                raise ValueError("Existing dataset config differs from requested profile")
        else:
            if root.exists():
                raise ValueError("Use a fresh output directory, or --validate-only")
            candidates = bindings.collect_context_stable_token_candidates(
                tok, min_token_id=10000, max_characters=16
            )
            print(f"Context-stable candidates: {len(candidates)}", flush=True)
            pairs = bindings.generate_bindings(
                candidates, count=n, pair_count=16000, target_index=248, seed=cfg.binding_seed
            )
            bins, report = curate_sentence_bins(
                load_pinned_source(cfg.source),
                tok,
                sentence_capacity_requirements(n),
                spec=cfg.source,
            )
            publish_dataset(
                root,
                tok,
                bins,
                pairs,
                filler=args.filler.read_text(),
                source_report=report,
                config=cfg,
            )
        result = validate_dataset(root, tok)
        for phase in range(8):
            with (root / f"rows/key_value/phase_{phase}.jsonl").open() as stream:
                prompt = json.loads(next(stream))["prompt"]
            if tok.encode(prompt) != tok.hf.encode(prompt, add_special_tokens=False):
                raise ValueError("GigaToken / HF mismatch")
        atomic(
            args.audit_output or root / "validation.json",
            {
                **identity,
                "passed": True,
                "upstream_result": result,
                "filler_matches_paper": identity["filler_sha256"] == PAPER_FILLER_SHA,
                "data_sha256": data_hashes(root),
                "prepare_sha256": sha(__file__),
            },
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
