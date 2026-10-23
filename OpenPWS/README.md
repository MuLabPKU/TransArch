# Periodic Weak Spots: Phase Sensitivity from Chunked KV-Cache Compression

**Local reproduction and GLM extension:** see [extension/README.md](extension/README.md)
for GLM-5.3-Flash, block-indexer / HISA / DSA, the exact default scoring rule,
saved observations, and reproducible plots. The original author implementation
and configurations below are unchanged from commit `09c1099`.

[![arXiv](https://img.shields.io/badge/arXiv-2609.36322-b31b1b.svg?style=flat)](https://arxiv.org/abs/2609.36322)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Xingyu Zhu\*, Pu (Luke) Yi\*, Ziheng Cheng\*, Ang Lv\*, Jing Liu, Lexing Ying,
Yiyuan Ma, and Xin Dong

\*Core contributors.

**Paper:** [arXiv:2609.36322](https://arxiv.org/abs/2609.36322) ·
[PDF](https://arxiv.org/pdf/2609.36322) · [Citation](#citation)

## Overview

**Chunked KV-cache compression.** To save memory and attention cost on long
contexts, models such as DeepSeek-V4 merge consecutive tokens into fewer cache
entries: each entry summarizes a window of `W` tokens, and a new entry is
formed every `S` tokens (the stride). A token's **phase** is its position
relative to the window boundaries, `t mod S`. With `W = S = 4`:

```text
position t      0 1 2 3 | 4 5 6 7 | 8 9 10 11 | ...
phase t mod 4   0 1 2 3 | 0 1 2 3 | 0 1  2  3 | ...
cache entry       e0    |   e1    |    e2     | ...
```

Windows may also overlap (`W > S`); DeepSeek-V4's CSA kernel uses `W = 8`,
`S = 4`.

**Finding.** Long-context retrieval accuracy varies sharply and periodically
with the phase of the information to be retrieved. In a controlled 128K
needle-in-a-haystack test, accuracy in DeepSeek-V4 models differs by up to 40
percentage points across phases, a gap that average benchmark scores conceal.

**Evidence.**

- *Controlled pretraining.* Models with the Qwen3-0.6B architecture, pretrained
  from scratch with chunked KV compression as the only substantial change,
  reproduce the effect across compression designs, with a period that follows
  the stride. Full-attention baselines show no comparable periodicity.
- *Causal interventions.* KV-head knockouts in the pretrained models and
  query-suffix knockouts in DeepSeek-V4-Flash-Base show that attention
  components contribute unequally to retrieval at different phases. Cycling
  the learned compression gates largely shifts the weak within-window phases.
- *Theory.* In idealized retrieval models, gradient flow can favor sharp phase
  specialization.

**Takeaway.** Evaluate models with chunked KV-cache compression across
compression phases: high average accuracy can coexist with systematic
positional failures.

## What's in This Repository

This is the minimal public release accompanying the paper. It contains:

- the balanced 128K NIAH constructor, validator, and vLLM scorer (Section 2, Appendix C.1);
- the Prefix Padding and In-sequence Padding NIAH constructors and scorer for the Qwen3-based models (Appendix C.2–C.3);
- the Qwen3-KVC model and the configurations of every pretrained model in the paper (Appendix D, Table 8);
- the causal interventions: KV-head knockouts in the Qwen3-based models (Appendix E), query-suffix knockouts in DeepSeek-V4 (Appendix E.6), and gate cycling (Appendix H).

It does **not** include checkpoints, a training recipe or pretraining data,
generated prompts, the code-completion examples of Section 2.1, the
gate-preference analysis of Section 4.2 and Appendix G, confidence intervals,
or plotting code. The pretraining setup is described in Appendix D.

```text
periodic_weak_spots/
├── periodic_weak_spots/
│   ├── model/              # Qwen3-KVC configuration and implementation
│   ├── niah_128k/          # balanced 128K constructor, validator, vLLM scorer
│   ├── niah_0p6b/          # Qwen3-based NIAH constructors and HF scorer
│   ├── interventions/      # head knockouts, suffix knockouts, and gate cycling
│   ├── balanced_lengths.py # residue-balanced length designs
│   └── tokenization.py     # pinned HF tokenizers with GigaToken encoding
├── configs/
│   ├── model/              # every pretrained model in the paper
│   ├── niah_128k/          # evaluation and calibration profiles
│   ├── niah_0p6b/          # evaluation and calibration suites for both protocols
│   └── interventions/      # gate-cycling conditions
```

Tokenization uses [GigaToken](https://pypi.org/project/gigatoken/), a parallel
Rust encoder that produces the same token IDs as the Hugging Face fast
tokenizer.

## Requirements and Install

- Linux with NVIDIA GPUs for scoring and interventions. The locked PyTorch
  2.14.1 is the default PyPI build for CUDA 13.0 (NVIDIA driver ≥ 580, Turing or
  newer); FlexAttention also needs a C compiler for Triton. Data construction
  runs on any machine, including macOS.
- DeepSeek-V4 over 128K prompts needs an 8-GPU node.
- All pinned Hugging Face models and datasets are public and ungated.

Tested with Python 3.11, torch 2.14.1, transformers 5.19.0, and datasets 5.1.0.

```bash
uv sync --locked --python 3.11
source .venv/bin/activate    # or prefix commands with `uv run`
```

vLLM pins its own PyTorch, so the 128K behavioral scorer (`niah_128k score`)
runs from a second environment; everything else uses the main one.

```bash
uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python -e . "vllm>=0.22"
```

## Quick Start

```bash
# Qwen3-based NIAH: 1,000 logical examples, each at all 24 residues.
python -m periodic_weak_spots.niah_0p6b build --config configs/niah_0p6b/prefix_padding.json --output data/prefix_padding
python -m periodic_weak_spots.niah_0p6b build --config configs/niah_0p6b/in_sequence_padding.json --output data/in_sequence_padding
python -m periodic_weak_spots.niah_0p6b score --suite data/prefix_padding --model runs/reference

# 128K NIAH: 8 residues x 256 prompts.
python -m periodic_weak_spots.niah_128k build --config configs/niah_128k/evaluation.json --filler-file filler.txt --output data/niah_128k
python -m periodic_weak_spots.niah_128k validate --dataset data/niah_128k
.venv-vllm/bin/python -m periodic_weak_spots.niah_128k score --dataset data/niah_128k \
  --model deepseek-ai/DeepSeek-V4-Flash-Base --tensor-parallel-size 8 --results results/niah_128k.json
```

`niah_0p6b score` reports full-vocabulary and candidate accuracy by residue
mod 24 and the best-minus-worst gap; `niah_128k score` reports answer-prefix
accuracy by residue mod 8 and the gap, and `--results` also saves completions.
`runs/reference` stands for any Qwen3-KVC checkpoint saved with
`save_pretrained`.

## Data

**128K NIAH** (`configs/niah_128k/`, Appendix C.1). Each prompt has exactly
128,000 tokens: 16,000 four-token `key:value。` records with single-token keys
and values, and the target is record 248 (zero-based). A 48-token
natural-sentence filler, split between a prefix and a postfix, sets the
target's residue mod 8 while its mean position stays at 64,000; every residue
cell has the same mean (24) and second moment (604) of prefix lengths, so
residues differ only in phase. The eight residues cover two cycles of
DeepSeek-V4's stride 4 and four of DeepSeek-V4.1-Flash's stride 2.
`evaluation.json` is the paper's set and `calibration.json` the same design
with disjoint seeds. Prompts are tokenized with `deepseek-ai/DeepSeek-V4-Flash`
at revision `60d8d70`. Scoring uses raw text prompts, greedy decoding of 8
tokens, and answer-prefix accuracy.

The paper evaluates `deepseek-ai/DeepSeek-V4-Flash-Base`,
`DeepSeek-V4-Flash-0731`, `DeepSeek-V4-Pro-Base`, `DeepSeek-V4-Pro-0813`, and
`DeepSeek-V4.1-Flash` (Table 7); its numbers may come from inference backends
other than vLLM. For larger sets, such as the 2,560 prompts per residue used for
DeepSeek-V4.1-Flash, copy a profile, set `dataset_role` to `custom` with unique
`experiment_id` and `sample_prefix`, and scale `rows_per_phase` (a multiple of
256), `sentence_pair_count` (4x), and `distinct_sentence_count` (8x).

The filler text for padding is not redistributed. The paper used the Paul
Graham essay haystack of the standard NIAH benchmark (sha256
`51eaf041f492d8fdfdcf87be8756d23888eda0d3cab6816b3c3f11e7682de337`); `build`
reports whether your `--filler-file` matches. Natural sentences come from
[`agentlans/high-quality-english-sentences`](https://huggingface.co/datasets/agentlans/high-quality-english-sentences)
(ODC-BY).

**Qwen3-based NIAH** (`configs/niah_0p6b/`, Appendix C.2–C.3). Each prompt has
64 animal-to-name pairs, the target at index 24, and 8 in-context
demonstrations, under the Qwen3-0.6B tokenizer. A residue-balanced length
design controls the source key's residue `r = t_K mod 24`.

- **Prefix Padding:** natural sentences of 24–120 tokens set the prefix; prefix and postfix total 144 tokens.
- **In-sequence Padding:** 24 `｜` guard tokens on each side, and 126 more spread over the gaps between records, at most 4 per gap.

Each protocol has an evaluation suite (logical examples `[0, 1000)`) and a
calibration suite (`[1000, 2000)`, `*_calibration.json`) for the knockouts.

## Models

`configs/model/` holds the 23 configurations behind the 26 runs of Table 8 (the
reference was trained with seeds 42–45) and the W12/S12 model with four KV
heads used for gate cycling (Appendix H.2).

| Folder | Models |
| --- | --- |
| `full_attention/` | Full attention with 1, 4, and 8 KV heads |
| `reference/` | W8/S8, 1 KV head, scalar gates, adaptive-softmax pooling, fixed SWA-16 |
| `kv_heads/` | Reference with 2, 4, and 8 KV heads |
| `geometry/` | W4/S4, W8/S4, W8/S6, W12/S12, and W12/S12 with 4 KV heads |
| `operators/` | Tied K/V, uniform averaging, uniform averaging with tied K/V and postnorm, vector gates |
| `position_norm/` | No Q/K norm, no gate biases, NoPE, RoPE on 16 dimensions |
| `local_memory/` | Uncompressed open tail (1 and 8 KV heads), DeepSeek-style vector and scalar gates |

The configurations match the trained models; the two DeepSeek-style models
apply RoPE to 16 of 128 head dimensions (`rope_dims: 16`).

```python
from periodic_weak_spots.model import Qwen3KVCForCausalLM, load_research_model_config

model = Qwen3KVCForCausalLM(load_research_model_config("configs/model/reference/reference.json"))
```

Importing `periodic_weak_spots.model` registers `qwen3_kvc` with the
Transformers Auto classes. Training mode uses FlexAttention (call
`model.prepare_flex_attention(sequence_length)` before `torch.compile`);
scoring uses the eager path through `model.last_token_logits`. KV caching and
padded batches are not supported.

## Causal Interventions

**KV-head knockouts** (Qwen3-KVC, Appendix E). A knockout replaces, at the
scoring token, the attention outputs of every query head that reads one KV head
(the `o_proj` input) with their mean over the calibration suite, then reruns
the rest of the network. The sweep covers every `(layer, KV head)` and reports
accuracy by residue mod 24 and the relative change from the clean model.
`--means` caches the fitted means, and `--layers` splits the sweep.

```bash
python -m periodic_weak_spots.niah_0p6b build --config configs/niah_0p6b/prefix_padding_calibration.json --output data/prefix_padding_calibration
python -m periodic_weak_spots.interventions head-sweep --model runs/reference \
  --calibration data/prefix_padding_calibration --suite data/prefix_padding \
  --means runs/reference/head_means.pt --output results/head_knockouts.json
```

**Query-suffix knockouts** (DeepSeek-V4, Appendix E.6). DeepSeek-V4 has one
shared KV head per layer, so a knockout replaces a layer's whole attention
readout, the output of `self_attn.o_b_proj`, at positions of the 16-token query
suffix, which starts at the first occurrence of the query key. Calibration
gives one mean per layer and suffix position; the four masks patch the first
key, the second key (offset 13), both, or the full suffix. Prompts are
tokenized as vLLM tokenizes raw text, with the tokenizer's default leading
special tokens (`--no-special-tokens` omits them), so the knockouts see the
same tokens as `niah_128k score`. Each prompt's 127,984-token prefix, plus any
leading special tokens, is prefilled once in 1,024-token chunks and its cache
forked for every suffix forward. Because outputs vary with batch size, the four
masks of a layer run as one batch, each compared with the same slot of a clean
batch of four. Scores are top-1 accuracy, logit, and NLL of the gold target
token. This runs in Hugging Face Transformers with expert parallelism on one
8-GPU node:

```bash
python -m periodic_weak_spots.niah_128k build --config configs/niah_128k/calibration.json --filler-file filler.txt --output data/niah_128k_calibration
torchrun --nproc-per-node 8 -m periodic_weak_spots.interventions suffix-means \
  --model deepseek-ai/DeepSeek-V4-Flash-Base --calibration data/niah_128k_calibration \
  --rows-per-residue 40 --output results/suffix_means.pt
torchrun --nproc-per-node 8 -m periodic_weak_spots.interventions suffix-sweep \
  --model deepseek-ai/DeepSeek-V4-Flash-Base --evaluation data/niah_128k \
  --means results/suffix_means.pt --output results/suffix_rows.jsonl
python -m periodic_weak_spots.interventions suffix-summary \
  --rows results/suffix_rows.jsonl --output results/suffix_summary.json
```

`suffix-sweep` resumes from its output file, and `--shard INDEX/COUNT` splits
prompts across nodes for `suffix-summary` to merge. Figure 4c uses 256
evaluation prompts per residue.

**Gate cycling** (Appendix H). In models with adaptive-softmax gates and
`W = S`, cycling by `delta` applies the gate map and bias of window offset `i`
at offset `(i + delta) mod W`. If retrieval follows the gates, accuracy by phase
shifts as `A_delta(r) ≈ A_0((r - delta) mod S)`. The config cycles the key and
value gates of layers 3–23 by ±1, ±2, and ±3; each condition reports accuracy
by phase, the prediction, and `translation_r2` excluding the boundary phase as
in H.2 (`observed_or_source`) or H.3 (`observed`), or over all phases (`none`).

```bash
python -m periodic_weak_spots.interventions gate-cycling --model runs/w12s12_4kv \
  --suite data/prefix_padding --config configs/interventions/gate_cycling.json --output results/gate_cycling.json
```

## Terminology

| Here | Paper |
| --- | --- |
| residue, a row's `phase` field | source-key residue `r = t_K mod 24` (Qwen3-based suites) or mod 8 (128K) |
| phase, `*_by_phase` outputs | `r mod S` for a model with stride `S` |
| `compression_window`, `compression_stride` | `W`, `S` |
| `c_*`, `z_*` | payload maps `C^B_i`; gate maps `Z^B_i` and positional gate biases `b^B_i` |
| `compression_pooling: mean` | uniform averaging |
| `kv_sharing` | tied K/V |
| `gate_mode` | scalar or vector gates |
| `rope_dims` | RoPE dimensions `d_r` |
| `local_attention_mode: fixed_swa`, `swa_window_size` | fixed sliding window `T` |
| `local_attention_mode: tail_concat` | uncompressed open tail |

## Citation

```bibtex
@article{zhu2026periodic,
  title   = {Periodic Weak Spots: Phase Sensitivity from Chunked {KV}-Cache Compression},
  author  = {Zhu, Xingyu and Yi, Pu and Cheng, Ziheng and Lv, Ang and Liu, Jing and Ying, Lexing and Ma, Yiyuan and Dong, Xin},
  journal = {arXiv preprint arXiv:2609.36322},
  year    = {2026}
}
```
