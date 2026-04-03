# CLOVER: Cross-Layer Orthogonal Vectors Pruning and Fine-Tuning [ICML 2025]

Memory-bounded inference in decoder-only language models is a major deployment challenge. **CLOVER** treats pairs of attention mechanism components (Q-K and V-O) as low-rank decompositions, applies Singular Value Decomposition (SVD) within each attention head, and uses the resulting singular values to guide pruning and serve as trainable parameters for efficient fine-tuning — all without increasing parameter count.

**Highlights:**
- Pruning 70% of the Q-K head dimension in GPT-2 XL achieves perplexity comparable to vanilla pruning of just 8%.
- Fine-tuning singular values outperforms LoRA, DoRA, HiRA, and PiSSA by 7.6%, 5.5%, 3.8%, and 0.7% respectively on commonsense tasks for LLaMA-2 7B.
- Combined with [TransMLA](../TransMLA_NeurIPS_2025), achieves up to **11.1× speedup** over LLaMA-2-7B.

[[Paper]](https://arxiv.org/abs/2411.17426) [[Standalone Repo]](https://github.com/fxmeng/CLOVER)

## Overview

CLOVER provides two pruning pipelines:

1. **`clover.py`** — SVD-based attention head pruning for DeepSeek-MLA models. Decomposes Q-K and V-O pairs per head via `svd_lowrank`, keeping the top singular components to reduce head dimensions while preserving model quality.

2. **`slicegpt.py`** — A SliceGPT-style structured pruning pipeline for general transformer models (e.g., DeepSeek-V2-Lite). The pipeline:
   - Inserts shortcut connections and fuses RMSNorm into weight matrices
   - Computes PCA on calibration data to find principal directions
   - Rotates all weight matrices into the PCA basis
   - Slices to the target dimension

## Usage

### CLOVER: Attention Head Pruning (MLA Models)

```bash
python clover.py \
    --model-path deepseek-ai/DeepSeek-V2-Lite \
    --save-path outputs/deepseek-v2-lite-clover \
    --dtype bf16 \
    --device auto \
    --pruned-dim 64
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--model-path` | Path or HuggingFace model ID | `deepseek-ai/DeepSeek-V2-Lite` |
| `--save-path` | Output path for the pruned model | `deepseek-ai/DeepSeek-V2-Lite` |
| `--dtype` | Data type: `fp32`, `fp16`, `bf16` | `bf16` |
| `--device` | Device: `cpu`, `cuda`, `auto` | `auto` |
| `--pruned-dim` | Target dimension for Q-K and V-O after SVD | `64` |
| `--cal-dataset` | Calibration dataset | `wikitext2` |
| `--ppl-eval-batch-size` | Batch size for perplexity evaluation | `1` |

### SliceGPT: Structured Width Pruning

```bash
python slicegpt.py \
    --model-path deepseek-ai/DeepSeek-V2-Lite \
    --save-path outputs/deepseek-v2-lite-sliced \
    --dtype bf16 \
    --device auto \
    --cal-dataset wikitext2 \
    --cal-nsamples 128 \
    --cal-max-seqlen 256 \
    --cal-batch-size 8 \
    --pruned-dim 2048
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--model-path` | Path or HuggingFace model ID | `deepseek-ai/DeepSeek-V2-Lite` |
| `--save-path` | Output path | `outputs` |
| `--dtype` | Data type: `fp32`, `fp16`, `bf16` | `bf16` |
| `--device` | Device: `cpu`, `cuda`, `auto` | `auto` |
| `--cal-dataset` | Calibration dataset: `wikitext2`, `ptb`, `c4`, `alpaca` | `wikitext2` |
| `--cal-nsamples` | Number of calibration samples | `128` |
| `--cal-max-seqlen` | Maximum sequence length for calibration | `256` |
| `--cal-batch-size` | Calibration batch size | `8` |
| `--pruned-dim` | Target hidden dimension after slicing | `2048` |
| `--ppl-eval-batch-size` | Batch size for perplexity eval (0 to skip) | `8` |

### Evaluate a Pruned Model

```bash
python test.py \
    --model-path outputs/deepseek-v2-lite-sliced \
    --cal-dataset wikitext2 \
    --pruned-dim 2048
```

## Citation

```bibtex
@inproceedings{meng2025clover,
  title={CLOVER: Cross-Layer Orthogonal Vectors Pruning and Fine-Tuning},
  author={Meng, Fanxu and Tang, Pingzhi and Jiang, Fan and Zhang, Muhan},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```
