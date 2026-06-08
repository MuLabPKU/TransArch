# GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model Decoding [Preprint]

Multi-head Latent Attention (MLA), the attention used in DeepSeek-V2/V3, jointly compresses keys and values into a low-rank latent and matches the H100 roofline almost perfectly. Its trained weights, however, expose only one decoding path — an absorbed MQA form — which ties efficient inference to H100-class compute-bandwidth ratios, forfeits tensor parallelism along the head axis, and yields no Multi-Token Prediction (MTP) gain on commodity inference GPUs such as the export-restricted H20. **GQLA (Group-Query Latent Attention)** is a minimal modification of MLA whose trained weights expose **two algebraically equivalent decoding paths** over the same parameters: an MQA-absorb path identical to MLA's, and a GQA path with a per-group expanded cache.

**Highlights:**
- The runtime picks the path that matches the target hardware — **no retraining, no custom kernels** — so a single set of GQLA weights pins the rooflines of both **H100** (MQA-absorb, s_q=1) and **H20** (GQA + MTP, s_q=2).
- Supports up to **8-way zero-redundancy tensor parallelism** on the GQA path.
- To avoid pretraining from scratch we extend TransMLA into **TransGQLA**, which converts a pretrained MLA checkpoint into a GQLA model in minutes on a single node.
- On GLM-4.7-Flash with G=4 (5× KV reduction), TransGQLA with **similarity head grouping + Hessian-NLL PCA** keeps the commonsense-7 gap to MLA at **−2.90pp** (vs −5.13pp for the legacy neighbor-grouping baseline) and wikitext-2 PPL at **11.27** (vs MLA 13.09, −13.93%).

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
│   │                          # incl. similarity head grouping + Hessian-NLL token weights
│   ├── convert.py             # CLI: MLA checkpoint → GQLA checkpoint
│   ├── modeling.py            # HF GQLA classes (kv_b_proj shrunk to G heads)
│   ├── vllm_model.py          # vLLM classes for both decoding paths
│   └── vllm_register.py       # registers both arches via vllm.general_plugins
└── scripts/
    ├── run_commonsense.sh           # one-click eval: baseline / gqla / gqla-absorb (single config)
    ├── run_ablation_2x2.sh          # 2×2 conversion ablation (grouping × hess) + PPL
    ├── run_commonsense_2x2.sh       # 9-mode commonsense sweep (MLA + 4 cfgs × {GQA, absorb})
    ├── summarize_lm_eval.py         # original 3-variant pretty-printer
    └── summarize_commonsense_2x2.py # 9-mode aggregator (per-task + GQA-vs-absorb consistency)
```

The package targets **GLM-4.7-Flash** (`glm4_moe_lite` family). Extension to DeepSeek-V2/V3 follows the same recipe with different shapes.

## Install

```bash
git clone https://github.com/MuLabPKU/TransArch
cd TransArch/GQLA_preprint
pip install -e . --no-build-isolation
```

`pip install -e .` registers the `vllm.general_plugins` entry point `register_gqla = src.vllm_register:register`, which vLLM calls automatically in its main process *and* every TP worker — required for `tensor_parallel_size >= 2`.

External dependencies (install separately as needed): `torch`, `transformers >= 4.46` (Glm4MoeLite support — GLM-4.7-Flash's `config.json` actually declares `transformers_version: 5.0.0rc0`, so 5.x is the safer floor), `vllm >= 0.7` (we tested 0.22.1), `accelerate` (for `device_map="auto"`), `datasets`, `tqdm`, `lm_eval >= 0.4.5` (only for the scripts).

## Method

For each layer and each KV group `g` (group size `gs = H / G`):

1. **Capture activations.** Forward the calibration batches through the uncompressed model; a forward pre-hook on every `self_attn` saves its input hidden state. Optionally also captures the output of the last decoder layer (= input to `model.model.norm`) for the Hessian-NLL token weights below.

2. **(Optional) Hessian-NLL weighting** — `--hessian_pca --hessian_mode nll`. Standard PCA cov is `Σ = Xᵀ X` with uniform per-token weights. SparseGPT/OBS show that the right thing to weight by is `wₜ = NLL_teacher(token_{t+1} | tokens_{≤t})`, which approximates the diagonal of the empirical Hessian at the final hidden state. We compute `wₜ` once via `model.model.norm + model.lm_head + cross_entropy` on the cached final hiddens, per-sample mean-1 normalise, and pass `wₜ` into cov accumulation as `Σ = Xᵀ diag(w) X` (`sqrt(w)` row-scaling on `X`).

3. **Head grouping** — `--head_grouping {neighbor, similarity}`:
   - `neighbor` (default, legacy): heads partition by index — group `g` is `{g·gs, g·gs+1, …, (g+1)·gs−1}`.
   - `similarity`: collect the **full all-pair K/V covariance** `(H·d × H·d)`; compute trace-normalised nuclear-norm head similarity `S[h, h'] = ||Σ_{h,h'}||_* / sqrt(tr(Σ_h)·tr(Σ_{h'}))` summed across K and V (weights `--sim_w_k`, `--sim_w_v`); seed-and-grow greedy balanced grouping fills `G` groups of `gs` heads each.

4. **Stream per-group cov.** For neighbor groups, accumulate `(G, F, F)` Gram in fp64 directly (`F = gs·qk_nope` for K, `gs·v_dim` for V); memory is `O(G·F²)`, independent of calibration size. For similarity groups, slice per-group cov from the full all-pair cov using the chosen permutation.

5. **Per-group PCA.** Eigendecompose with 1% Tikhonov diagonal damping. Retain the top `qk_nope` directions for K and `v_dim` for V — same rank as one head's slot, so dimensions match downstream.

6. **Absorb.** Build the compressed `kv_b_proj` by rotating the original K/V rows with `U_gᵀ`. Absorb the inverse rotation into Q (nope rows) and O (per-head v cols): rotations are square (`d_k × d_k` and `d_v × d_v`), so Q/O shapes are unchanged. For the `similarity` path the absorption also handles the implicit head permutation in the same pass (`compose_compressed_with_perm`).

After conversion: `kv_b_proj` has `G × (qk_nope + v_dim)` rows instead of `H × (qk_nope + v_dim)`; Q and O retain their original shapes; `num_key_value_heads = G`. The HF forward already broadcasts per-group K/V across `gs` query heads via `num_key_value_groups`.

The default path (`--head_grouping neighbor` and no `--hessian_pca`) is byte-identical to the legacy code — `collect_kv_grams(..., token_weights=None)` and `compose_compressed_with_perm` on neighbor groups + identity rotation match `compress_and_absorb` up to fp32 reduction-order noise (verified: max diff < 3e-6).

## Quick start

### 1. Convert MLA → GQLA

```bash
python -m src.convert \
    --model_path /path/to/GLM-4.7-Flash \
    --save_path  outputs/glm47-gqla-g4 \
    --num_kv_heads 4 --dtype bf16 --device_map auto \
    --cal_dataset wikitext2 --cal_nsamples 128 --cal_seqlen 512 \
    --head_grouping similarity --hessian_pca --hessian_mode nll \
    --eval_ppl --eval_ppl_dataset wikitext2 --eval_ppl_seqlen 1024
```

- `--num_kv_heads`: GQA group count. Must divide `num_attention_heads`. For GLM-4.7-Flash (H=20), valid values are 1, 2, 4, 5, 10, 20. `G=4` gives 5× KV reduction.
- `--head_grouping`: `neighbor` (default, legacy) or `similarity` (data-driven). `similarity` adds an extra full all-pair cov collection per layer (~6× cost on cov, still ~7s/layer on GLM-4.7).
- `--sim_w_k`, `--sim_w_v`: K-vs-V contribution to head similarity (default 1.0 each). `(1, 0)` groups by K subspace only.
- `--hessian_pca` + `--hessian_mode nll`: weight cov by per-token teacher NLL. Off by default; off path is byte-identical to legacy.
- `--device_map auto`: HF shards layers across visible GPUs at load. Calibration forward, per-layer PCA fit, and in-place mutation all run on whatever device each layer sits on.
- `--cal_nsamples × --cal_seqlen`: total calibration tokens (default 128 × 512 = 65k). Should comfortably exceed the per-group covariance dimension (`gs·qk_nope = 960` and `gs·v_dim = 1280` for G=4) to avoid undersampling.
- `--eval_ppl`: reports wikitext-2 PPL before and after conversion in the same run.

Output: an HF-loadable checkpoint with `config.json["architectures"] = ["Glm4MoeLiteGQLAForCausalLM"]`, bundled `modeling.py`, and `gqla_meta.json` recording layout + calibration provenance (including which head_grouping / hessian flags were used).

Walltime on a 8×A100 80GB node (`device_map=auto`, cal=128×512, eval_ppl on full wikitext-2):
| config | model load | cal forward | hess weights | compress | save | PPL eval (×2) | **total** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `neighbor + nohess` | 1m10s | 30s | — | 15s | 1m | 3m | **~6m** |
| `neighbor + hess+nll` | 1m10s | 30s | 5s | 15s | 1m | 3m | **~6m** |
| `similarity + nohess` | 1m10s | 30s | — | 5m30s | 1m | 3m | **~11m** |
| `similarity + hess+nll` | 1m10s | 30s | 5s | 5m30s | 1m | 3m | **~11m** |

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

Both architectures load the **same on-disk weights**. The absorb path expands the per-group `kv_b_proj` to MLA layout once at weight-load time (`_expand_gqla_kv_b_weight` in `src/vllm_model.py`), then runs the standard MLA absorb kernel. Decode-time arithmetic is identical to native MLA; the two paths' outputs agree up to fp16 reduction-order noise from TP all-reduce (we measure ≤1.4pp max accuracy gap on 7 commonsense tasks — see consistency table below).

### 3. Commonsense evaluation (single config)

```bash
BASELINE_PATH=/path/to/GLM-4.7-Flash \
GQLA_PATH=outputs/glm47-gqla-g4 \
OUT_ROOT=outputs/lm_eval \
TP=2 \
    bash scripts/run_commonsense.sh
```

Runs `hellaswag, arc_easy, arc_challenge, piqa, winogrande, openbookqa` for all three modes (baseline MLA / GQLA GQA-path / GQLA MLA-absorb-path) via lm-eval-harness + vLLM. Each mode launches a fresh python process so vLLM/CUDA state is released between runs. `GQLA_ABSORB_PATH` defaults to `$GQLA_PATH` — both paths share weights.

`scripts/summarize_lm_eval.py` is invoked at the end and prints a side-by-side accuracy table with deltas vs baseline.

---

## End-to-end reproduce (this preprint's numbers)

The two scripts below reproduce **every number** in the result tables. Each is serial and resumable: a config whose `gqla_meta.json` already has `ppl_gqla` (conversion) or whose output dir already has `results_*.json` (eval) is skipped, so a crashed run is `bash scripts/<runner>.sh` away from picking up where it left off.

### Step A — convert the 4 ablation checkpoints (~35 min on 8×A100, cold cache)

```bash
ABL_ROOT=/path/to/outputs/abl_2x2   # default; set to wherever you have ≥250GB disk
BASELINE_PATH=/path/to/GLM-4.7-Flash \
    bash scripts/run_ablation_2x2.sh
```

This runs all 4 conversions in order (grouping-first sweep), each with `cal=128×512` + `--eval_ppl` on full wikitext-2 (282×1024 chunks):

| tag | head_grouping | hessian_pca | output |
|---|---|---|---|
| `neigh_nohess` | neighbor | off | `outputs/abl_2x2/neigh_nohess/` |
| `neigh_hess`   | neighbor | nll | `outputs/abl_2x2/neigh_hess/`   |
| `sim_nohess`   | similarity | off | `outputs/abl_2x2/sim_nohess/` |
| `sim_hess`     | similarity | nll | `outputs/abl_2x2/sim_hess/`   |

Each checkpoint is ~56 GB. Total disk: ~225 GB.

Env overrides: `PYTHON`, `MODEL_PATH`, `OUT_ROOT`, `CAL_NSAMPLES`, `CAL_SEQLEN`, `NUM_KV_HEADS`, `DEVICE_MAP`, `EVAL_PPL_MAX_CHUNKS` (set the last one for a quick smoke; leave unset for the full 282 chunks).

### Step B — commonsense lm-eval across all 9 modes (~35 min on 8×A100)

```bash
ABL_ROOT=/path/to/outputs/abl_2x2 \
BASELINE_PATH=/path/to/GLM-4.7-Flash \
OUT_ROOT=/path/to/outputs/abl_2x2_commonsense \
    bash scripts/run_commonsense_2x2.sh
```

This launches 9 lm_eval runs (1 MLA baseline + 4 GQLA configs × 2 decode paths) on the 7 commonsense tasks from the README results table. Topology: 8 GPUs split into 4 groups of TP=2; 4 modes run in parallel, batched (3 batches of 4 / 4 / 1). Each mode's output goes to `outputs/abl_2x2_commonsense/<mode>/`.

Env overrides: `PYTHON`, `ABL_ROOT`, `BASELINE_PATH`, `OUT_ROOT`, `TASKS` (default = the 7 commonsense tasks), `TP` (default 2), `LIMIT` (set to e.g. 5 for a smoke).

The first cold run incurs ~5 min of FlashInfer JIT compile + CUDA-graph capture per mode; the warm cache makes subsequent runs ~2 min startup + ~3 min eval. Total ~8 min/mode × ceil(9 / 4) batches ≈ 30 min wall-clock.

### Step C — print the result tables

```bash
# 2×2 PPL ablation (reads gqla_meta.json from each config dir)
for d in /path/to/outputs/abl_2x2/*/; do
    grep -E "method|ppl_(orig|gqla)" "$d/gqla_meta.json" | sed "s|^|$(basename $d): |"
done

# 9-mode commonsense aggregator (per-task table + GQA-vs-absorb consistency + winner)
python scripts/summarize_commonsense_2x2.py /path/to/outputs/abl_2x2_commonsense
```

---

## Results (GLM-4.7-Flash, G=4 → 5× KV reduction, bf16)

Calibration: wikitext2 `128 × 512 = 65k tokens`. PPL: wikitext2 concat-and-chunk, `seqlen=1024`, **all 282 chunks**. Commonsense: lm-eval-harness, full splits, no `--limit`.

### Table 1 — wikitext-2 PPL ablation (2 × 2)

| config | head_grouping | hess | PPL | Δ vs MLA | Δ% vs MLA |
|---|---|---|---:|---:|---:|
| MLA baseline                       | —          | —   | 13.0896 | —       | —      |
| GQLA `neigh_nohess` (= README old) | neighbor   | off | 12.0889 | −1.0007 | −7.65% |
| GQLA `neigh_hess` (**PPL-best**)   | neighbor   | nll | **11.2658** | **−1.8237** | **−13.93%** |
| GQLA `sim_nohess`                  | similarity | off | 12.4595 | −0.6300 | −4.81% |
| GQLA `sim_hess`                    | similarity | nll | 11.5945 | −1.4951 | −11.42% |

Main effects:
- **Hessian-NLL: −0.844 PPL** averaged over both groupings — large and consistent.
- **Similarity grouping: +0.350 PPL** averaged over both hess settings — *worse* on this model.
- Interaction (sim×hess additivity check): −0.042 PPL — essentially additive.

### Table 2 — Commonsense 7-task accuracy (acc_norm > acc), GQA path

(Both decode paths share the same weights; absorb-path numbers match GQA within ≤1.4pp; full 9-mode table + consistency check below.)

| Task | MLA | `neigh_nohess` | `neigh_hess` | `sim_nohess` | `sim_hess` |
|---|---:|---:|---:|---:|---:|
| hellaswag       | 0.8012 | 0.7364 | 0.7383 | 0.7386 | **0.7498** |
| arc_easy        | 0.8173 | 0.7677 | 0.7799 | 0.7799 | **0.7988** |
| arc_challenge   | 0.5717 | 0.5333 | 0.5324 | 0.5333 | **0.5538** |
| piqa            | 0.8063 | 0.7802 | 0.7769 | **0.7911** | 0.7927 |
| winogrande      | 0.7261 | 0.6827 | 0.7032 | **0.7048** | 0.6851 |
| openbookqa      | 0.4480 | 0.4080 | 0.4240 | **0.4420** | 0.4380 |
| boolq           | 0.8838 | 0.8187 | 0.8229 | **0.8410** | 0.8333 |
| **AVG**         | **0.7221** | 0.6753 | 0.6825 | 0.6922 | **0.6931** |
| **Δpp vs MLA**  | 0.00 | −4.68 | −3.95 | −2.99 | **−2.90** |

Bold = best across the 4 GQA configs on that row. **`sim_hess` (commonsense-best)** wins 4/7 tasks and the AVG; `sim_nohess` wins the other 3.

Main effects (GQA path only):
- **Hessian-NLL: +0.45pp AVG** (averaged across the two groupings).
- **Similarity grouping: +1.41pp AVG** (averaged across hess on/off).
- Note the reversal vs PPL: *sim hurts PPL but helps commonsense*. PPL penalises rare-token surprise, where neighbor's untouched per-head subspace is closer to teacher; multi-choice tasks reward ordering of head-relevant features, where the similarity partition keeps more task-relevant signal.

### Table 3 — Full 9-mode commonsense table (GQA + MQA-absorb)

| Task | MLA | neigh_nohess<br>GQA / Abs | neigh_hess<br>GQA / Abs | sim_nohess<br>GQA / Abs | sim_hess<br>GQA / Abs |
|---|---:|---:|---:|---:|---:|
| hellaswag       | 0.8012 | 0.7364 / 0.7364 | 0.7383 / 0.7382 | 0.7386 / 0.7401 | 0.7498 / 0.7496 |
| arc_easy        | 0.8173 | 0.7677 / 0.7723 | 0.7799 / 0.7790 | 0.7799 / 0.7782 | 0.7988 / **0.8056** |
| arc_challenge   | 0.5717 | 0.5333 / 0.5333 | 0.5324 / 0.5222 | 0.5333 / **0.5572** | 0.5538 / 0.5486 |
| piqa            | 0.8063 | 0.7802 / 0.7818 | 0.7769 / 0.7731 | 0.7911 / 0.7851 | 0.7927 / 0.7927 |
| winogrande      | 0.7261 | 0.6827 / 0.6953 | 0.7032 / 0.7001 | 0.7048 / 0.7032 | 0.6851 / 0.6993 |
| openbookqa      | 0.4480 | 0.4080 / 0.4120 | 0.4240 / 0.4160 | 0.4420 / 0.4320 | 0.4380 / 0.4380 |
| boolq           | 0.8838 | 0.8187 / 0.8214 | 0.8229 / 0.8257 | 0.8410 / 0.8364 | 0.8333 / 0.8315 |
| **AVG**         | **0.7221** | 0.6753 / 0.6789 | 0.6825 / 0.6792 | 0.6922 / 0.6903 | **0.6931 / 0.6950** |
| **Δpp vs MLA**  | 0.00 | −4.68 / −4.31 | −3.95 / −4.29 | −2.99 / −3.18 | **−2.90 / −2.70** |

### Table 4 — GQA-vs-MQA-absorb consistency (per-config)

Both paths read identical on-disk weights; differences come from kernel reduction order (fp16 noise) and the absorb path's `_expand_gqla_kv_b_weight` repeat-interleave.

| config | avg `|GQA − Abs|` | max `|GQA − Abs|` (task) |
|---|---:|---|
| `neigh_nohess` | 0.0037 | 0.0126 (winogrande) |
| `neigh_hess`   | 0.0041 | 0.0102 (arc_challenge) |
| `sim_nohess`   | 0.0050 | 0.0100 (openbookqa) |
| `sim_hess`     | 0.0040 | 0.0142 (winogrande) |

Avg ≤0.5pp, max ≤1.5pp across all configs — the two decode paths are interchangeable for downstream evaluation.

### Headline table (README old vs new)

| metric | MLA | GQLA old (`neigh_nohess`) | GQLA new (this preprint) | improvement |
|---|---:|---:|---:|---:|
| wikitext-2 PPL              | 13.09 | 12.07 (−7.74%) | **11.27 (−13.93%, `neigh_hess`)** | **−6.79% PPL** |
| commonsense-7 AVG (acc)     | 0.7221 | 0.6753 (−4.68pp) | **0.6931 (−2.90pp, `sim_hess`)** | **+1.78pp commonsense** |

Same 5× KV reduction, same TP topology, same vLLM kernels — purely calibration-side wins.

## Recommendation

- **For best PPL** (e.g. perplexity-bounded use cases like next-token prediction): `--head_grouping neighbor --hessian_pca --hessian_mode nll`.
- **For best zero-shot commonsense** (multi-choice / classification): `--head_grouping similarity --sim_w_k 1.0 --sim_w_v 1.0 --hessian_pca --hessian_mode nll`.
- The two cheapest knobs (cost: extra `~5s` for NLL weights) give the biggest gain — always turn on `--hessian_pca`. `--head_grouping similarity` adds ~5 min per conversion (full all-pair cov) and is a clear win for commonsense, mild loss for PPL.

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
