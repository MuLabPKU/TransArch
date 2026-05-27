# GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model Decoding [Preprint]

Multi-head Latent Attention (MLA), the attention used in DeepSeek-V2/V3, jointly compresses keys and values into a low-rank latent and matches the H100 roofline almost perfectly. Its trained weights, however, expose only one decoding path — an absorbed MQA form — which ties efficient inference to H100-class compute-bandwidth ratios, forfeits tensor parallelism along the head axis, and yields no Multi-Token Prediction (MTP) gain on commodity inference GPUs such as the export-restricted H20. **GQLA (Group-Query Latent Attention)** is a minimal modification of MLA whose trained weights expose **two algebraically equivalent decoding paths** over the same parameters: an MQA-absorb path identical to MLA's, and a GQA path with a per-group expanded cache.

**Highlights:**
- The runtime picks the path that matches the target hardware — **no retraining, no custom kernels** — so a single set of GQLA weights pins the rooflines of both **H100** (MQA-absorb, s_q=1) and **H20** (GQA + MTP, s_q=2).
- Supports up to **8-way zero-redundancy tensor parallelism** on the GQA path.
- To avoid pretraining from scratch we extend TransMLA into **TransGQLA**, which converts a pretrained MLA checkpoint into a GQLA model in minutes on a single node.
- On GLM-4.7-Flash, TransGQLA compresses per-token KV cache **5×** (`H=20 → G=4`) while staying within ~5pp of MLA average on seven commonsense tasks; wikitext-2 PPL is **lower** than the MLA baseline (12.07 vs 13.09) at production calibration scale.

[[Paper]](https://arxiv.org/abs/2605.15250)

---

## Repository layout

```
GQLA_preprint/
├── pyproject.toml             # package metadata + vLLM plugin entry point
├── README.md
├── src/                       # Python package (import src.X)
│   ├── __init__.py
│   ├── compression.py         # per-group K/V PCA + absorption (data-driven)
│   ├── convert.py             # CLI: MLA checkpoint → GQLA checkpoint
│   ├── modeling.py            # HF GQLA classes (kv_b_proj shrunk to G heads)
│   ├── vllm_model.py          # vLLM classes for both decoding paths
│   └── vllm_register.py       # registers both arches via vllm.general_plugins
└── scripts/
    ├── run_commonsense.sh     # one-click eval: baseline / gqla / gqla-absorb
    └── summarize_lm_eval.py   # pretty-prints the side-by-side results table
```

The package targets **GLM-4.7-Flash** (`glm4_moe_lite` family). Extension to DeepSeek-V2/V3 follows the same recipe with different shapes.

## Install

```bash
git clone https://github.com/MuLabPKU/TransArch
cd TransArch/GQLA_preprint
pip install -e . --no-build-isolation
```

`pip install -e .` registers the `vllm.general_plugins` entry point `register_gqla = src.vllm_register:register`, which vLLM calls automatically in its main process *and* every TP worker — required for `tensor_parallel_size >= 2`.

External dependencies (install separately as needed): `torch`, `transformers >= 4.46` (Glm4MoeLite support), `vllm >= 0.7`, `datasets`, `tqdm`, `lm_eval >= 0.4.5` (only for the scripts).

## Quick start

### 1. Convert MLA → GQLA

```bash
python -m src.convert \
    --model_path /path/to/GLM-4.7-Flash \
    --save_path  outputs/glm47-gqla-g4 \
    --num_kv_heads 4 --dtype bf16 --device_map auto \
    --cal_dataset wikitext2 --cal_nsamples 128 --cal_seqlen 512 \
    --eval_ppl --eval_ppl_dataset wikitext2 --eval_ppl_seqlen 1024
```

- `--num_kv_heads`: GQA group count. Must divide `num_attention_heads`. For GLM-4.7-Flash (H=20), valid values are 1, 2, 4, 5, 10, 20. `G=4` gives 5× KV reduction.
- `--device_map auto`: HF shards layers across visible GPUs at load. The calibration forward, per-layer PCA fit, and in-place mutation all run on whatever device each layer sits on.
- `--cal_nsamples × --cal_seqlen`: total calibration tokens (default 128 × 512 = 65k). Should comfortably exceed the per-group covariance dimension (`gs × qk_nope = 960` and `gs × v_dim = 1280` for G=4) to avoid undersampling.
- `--eval_ppl`: reports wikitext-2 PPL before and after conversion in the same run.

Output: an HF-loadable checkpoint with `config.json["architectures"] = ["Glm4MoeLiteGQLAForCausalLM"]`, bundled `modeling.py`, and `gqla_meta.json` recording layout + calibration provenance.

Walltime on 2 × L20 80GB: model load 12s, calibration forward ~10s, per-layer compress ~6s (47 layers), save 60s, PPL eval (282 chunks × 1024) ~80s × 2. Total ~5 minutes for cal=128×512.

### 2. Serve with vLLM

```python
from vllm import LLM, SamplingParams

# GQA path: KV cache stored at GQA shape (G heads × per_head_kv per token).
# Best for memory-bandwidth-bound hardware (H20, A10) and TP > 1.
llm = LLM(
    "outputs/glm47-gqla-g4",
    tensor_parallel_size=2,
    dtype="bfloat16",
    hf_overrides={"architectures": ["Glm4MoeLiteGQLAForCausalLM"]},
)

# MLA-absorb path: KV cache stored at MQA shape (kv_lora + qk_rope per token).
# Smallest cache; identical decode kernel as DeepSeek-MLA.
llm = LLM(
    "outputs/glm47-gqla-g4",
    tensor_parallel_size=2,
    dtype="bfloat16",
    hf_overrides={"architectures": ["Glm4MoeLiteGQLAAbsorbForCausalLM"]},
)
```

Both architectures load the **same on-disk weights**. The absorb path expands the per-group `kv_b_proj` to MLA layout once at weight-load time (`_expand_gqla_kv_b_weight` in `src/vllm_model.py`), then runs the standard MLA absorb kernel; decode-time cost is identical to native MLA.

### 3. Commonsense evaluation

```bash
BASELINE_PATH=/path/to/GLM-4.7-Flash \
GQLA_PATH=outputs/glm47-gqla-g4 \
OUT_ROOT=outputs/lm_eval \
TP=2 \
    bash scripts/run_commonsense.sh
```

Runs `hellaswag, arc_easy, arc_challenge, piqa, winogrande, openbookqa` for all three modes (baseline MLA / GQLA GQA-path / GQLA MLA-absorb-path) via lm-eval-harness + vLLM. Each mode launches a fresh python process so vLLM/CUDA state is released between runs. `GQLA_ABSORB_PATH` defaults to `$GQLA_PATH` — both paths share weights.

`scripts/summarize_lm_eval.py` is invoked at the end and prints a side-by-side accuracy table with deltas vs baseline.

## Method

For each layer and each KV group `g` (group size `gs = H / G`):

1. **Stream covariance.** Forward the calibration batches through the original `kv_a_proj_with_mqa → kv_a_layernorm → kv_b_proj` chain; accumulate per-group `(F × F)` Gram matrices in fp64 for stacked-per-group K (`F = gs × qk_nope`) and V (`F = gs × v_dim`). Memory is `O(G × F²)`, **independent of calibration size**.

2. **Per-group PCA.** Eigendecompose with 1% Tikhonov diagonal damping for numerical stability under sparse calibration. Retain the top `qk_nope` directions for K and `v_dim` for V — same rank as one head's slot, so dimensions match downstream.

3. **Absorb.** Build the compressed `kv_b_proj` by rotating the original K/V rows with `U_g^T`. Absorb the inverse rotation into Q (nope rows) and O (per-head v cols): because `U_g` is square (`d_k × d_k` and `d_v × d_v`), Q/O shapes are unchanged. The result is mathematically equivalent to the source MLA up to the rank-(`d_k`, `d_v`) PCA truncation per group.

After conversion: `kv_b_proj` has `G × (qk_nope + v_dim)` rows instead of `H × (qk_nope + v_dim)`; Q and O retain their original shapes; `num_key_value_heads = G`. The HF forward already broadcasts per-group K/V across `gs` query heads via `num_key_value_groups`.

## Results

GLM-4.7-Flash, G=4 (5× KV reduction), wikitext-2 calibration (128 × 512 = 65k tokens), bf16, evaluated at TP=2.

**wikitext-2 PPL** (concat-and-chunk, seqlen=1024, 282 chunks, full test split):

| Model                | PPL     | Δ vs MLA |
|----------------------|---------|----------|
| GLM-4.7-Flash (MLA)  | 13.09   | —        |
| GQLA G=4             | 12.07   | **−7.74%** |

**Zero-shot commonsense** (lm-eval-harness, full splits, no `--limit`):

| Task          | MLA     | GQLA G=4 | Δ        |
|---------------|---------|----------|----------|
| arc_easy      | 0.8228  | 0.7824   | −4.04pp  |
| arc_challenge | 0.5597  | 0.4906   | −6.91pp  |
| piqa          | 0.7992  | 0.7786   | −2.06pp  |
| winogrande    | 0.7340  | 0.6906   | −4.34pp  |
| hellaswag     | 0.6110  | 0.5335   | −7.75pp  |
| openbookqa    | 0.3340  | 0.2900   | −4.40pp  |
| boolq         | 0.8838  | 0.8196   | −6.42pp  |
| **average**   | **0.6778** | **0.6265** | **−5.13pp** |

The GQA and MLA-absorb paths share weights, so their outputs match up to floating-point reordering from cross-TP all-reduce.

## Reproduce the results table

```bash
# 1. Convert (~5 min on 2×A100)
python -m src.convert \
    --model_path /path/to/GLM-4.7-Flash \
    --save_path  outputs/glm47-gqla-g4-prod \
    --num_kv_heads 4 --dtype bf16 --device_map auto \
    --cal_dataset wikitext2 --cal_nsamples 128 --cal_seqlen 512 \
    --eval_ppl --eval_ppl_dataset wikitext2 --eval_ppl_seqlen 1024

# 2. Commonsense eval (~3 hours per mode at TP=2)
OUT_ROOT=outputs/lm_eval_prod \
BASELINE_PATH=/path/to/GLM-4.7-Flash \
GQLA_PATH=outputs/glm47-gqla-g4-prod \
GQLA_ABSORB_PATH=outputs/glm47-gqla-g4-prod \
    bash scripts/run_commonsense.sh
```

## Authors

Fanxu Meng

## Citation

```bibtex
@article{meng2026gqla,
  title={GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model Decoding},
  author={Meng, Fanxu},
  journal={arXiv preprint arXiv:2605.15250},
  year={2026}
}
```
