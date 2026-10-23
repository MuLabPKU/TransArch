# Release validation

This file records validation of the compact extension, separately from the archived
full experiment observations. The upstream base is
`09c109910ba29b834f54441d96b442ab9e1634ae`.

Completed checks:

- `ruff check periodic_weak_spots extension` passes.
- All five CPU regression tests pass: quote-prefix boundaries, exact historical
  counts, archive checksums, upstream result-format compatibility, and restoration
  of upstream tokenizer/query definitions.
- All 2,048 existing GLM prompts pass the upstream independent validator under
  the new adapter. Dataset hashes match the original validation record; eight
  prompts (one per phase) also match between GigaToken and Hugging Face encoding.
- GPU FP8 score tests pass against an independently dequantized FP32 reference
  (atol 0.002, rtol 0.0003), including a 128,007-token context. Candidate gathering
  matches native scores bitwise; selected sets satisfy causality, uniqueness,
  candidate membership and the token budget.
- GPU runtime tests pass for prefill, single-token decode, cache reset, invalid
  continuation rejection, and a strict 2,048-token selection budget.
- The kernel functions and cache allocation/initialization preserve the original
  tested computation. The selection forward differs only in import ordering.
- Default and original-score reports regenerate from archived completions; PNG,
  PDF and SVG exports work with English labels and with an explicit CJK font.

The compact runner's real-model smoke run completed successfully (exit 0) using
eight 128K prompts covering all phases, in all three modes (24 saved observations).
Before these observations, two repetitions of the first prompt in each mode passed
the mode/request isolation check: identical output token IDs within each mode.
Worker audits covered all 11 sparse layers; native mode did not execute experimental
selection, and HISA/DSA did. The original L20Z tactic table loaded on all eight GPUs.

Comparison with the corresponding historical observations:

| Mode | Prompts | Both scoring rules agree | Exact output token IDs agree |
|---|---:|---:|---:|
| block-indexer (native) | 8 | 8/8 | 6/8 |
| HISA | 8 | 8/8 | 8/8 |
| DSA | 8 | 8/8 | 8/8 |

Two native completions changed from `γη.</think>γη` to `γη.` (phases 3 and 5).
The correct answer prefix is unchanged. The refactored runner shares one engine
across modes, unlike the historical separate native-control run; these observations
do not establish bitwise equivalence between runtime arrangements. No full 2,048-row
rerun of each arm was performed. This is a code validation run on one binding at
all eight phases, not a new statistical evaluation or evidence of improved accuracy.

Repeating the same `--limit 8` command passed the resume identity checks and exited
with `All selected rows already complete`, without loading the model or adding rows.
All GPU workers exited after the smoke run.

The full prompt dataset was not regenerated from the source sentence corpus
during this cleanup; the existing generated prompts were independently revalidated.
No claim of complete paper reproduction is made.
