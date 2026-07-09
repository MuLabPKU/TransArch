# MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference [Preprint]

DeepSeek Sparse Attention (DSA) sets the state of the art for fine-grained inference-time sparse attention by introducing a learned token-wise indexer that scores every prefix token and selects the most relevant ones for the main attention. To remain expressive, the indexer uses many query heads (e.g., 64 on DeepSeek-V3.2) that share the same selected token set; this multi-head design is precisely what makes the indexer the dominant cost on long contexts. **MISA (Mixture of Indexer Sparse Attention)** is a drop-in replacement for the DSA indexer that treats its indexer heads as a pool of mixture-of-experts. A lightweight router uses cheap block-level statistics to pick a query-dependent subset of only a few active heads, and only those heads run the heavy token-level scoring.

**Highlights:**
- Preserves the diversity of the original indexer pool while reducing the per-query cost from scoring every prefix token with every head to scoring it with only a handful of routed heads, plus a negligible router term computed on a small set of pooled keys.
- Hierarchical variant of MISA uses the routed pass to keep an enlarged candidate set and then re-ranks it with the original DSA indexer to recover the final selected tokens almost exactly.
- With only **8 active heads** and **no additional training**, MISA matches the dense DSA indexer on LongBench across DeepSeek-V3.2 and GLM-5 while running with **8× and 4× fewer indexer heads** respectively, and outperforms HISA on average.
- Preserves fully green Needle-in-a-Haystack heatmaps up to **128K-token context** and recovers **>92%** of the tokens selected by the DSA indexer per layer.
- TileLang kernel delivers roughly a **3.82× speedup** over DSA's original indexer kernel on a single NVIDIA H200 GPU.

[[Paper]](https://arxiv.org/abs/2605.07363)

## Code

This release provides vLLM patch implementations for both MISA variants:

- [`misa_vllm_patch_1stage/`](misa_vllm_patch_1stage/): MISA.
- [`misa_vllm_patch_2stage/`](misa_vllm_patch_2stage/): MISA†, the two-stage hierarchical variant.

### Environment

- `vllm == 0.17.0`
- `transformers == 5.2.0`
- `lm-eval == 0.4.11`
- `tilelang == 0.1.7.post3`

### Quick Start

Install one patch at a time, depending on the variant you want to run.

For MISA:

```bash
pip install -e ./misa_vllm_patch_1stage
```

For MISA†:

```bash
pip install -e ./misa_vllm_patch_2stage
```

Then add the following fields to the model `config.json`:

```json
{
  "use_misa": true,
  "misa_topk_heads": 8,
  "misa_decode_topk_heads": 8,
  "misa_k_block_size": 1024,
  "misa_use_qnorm": false
}
```

For MISA†, also add:

```json
{
  "misa_stage1_topk": 8192
}
```

## Authors

Ruijie Zhou, Fanxu Meng, Yufei Xu, Tongxuan Liu, Guangming Lu, Muhan Zhang, Wenjie Pei

## Citation

```bibtex
@article{zhou2026misa,
  title={MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference},
  author={Zhou, Ruijie and Meng, Fanxu and Xu, Yufei and Liu, Tongxuan and Lu, Guangming and Zhang, Muhan and Pei, Wenjie},
  journal={arXiv preprint arXiv:2605.07363},
  year={2026}
}
```
