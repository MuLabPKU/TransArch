# HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention [Under Review]

Token-level sparse attention mechanisms like DeepSeek Sparse Attention (DSA) need an indexer that scores every historical token for each query, creating an O(L²) per-layer bottleneck that becomes prohibitive as context lengths grow to 128K tokens and beyond. **HISA** replaces the flat token-scanning indexer with a two-stage hierarchical procedure:

1. **Block-level coarse filtering** — scores pooled block representatives to prune irrelevant regions and select top-*m* blocks.
2. **Token-level refinement** — applies the original indexer only within the surviving candidate blocks to select top-*k* tokens.

**Highlights:**
- Drop-in replacement for the sparse attention indexer that preserves the identical token-level top-*k* sparse pattern consumed by downstream Sparse MLA operators.
- Requires no additional training.
- Token selection sets between HISA and original DSA show **>99% mean IoU**, indicating minimal quality impact.
- Achieves **2× speedup at 32K** context and **4× speedup at 128K** context on kernel-level benchmarks.
- Successfully replaces the indexer in DeepSeek-V3.2 and GLM-5 without finetuning, matching original DSA quality on Needle-in-a-Haystack and LongBench benchmarks.

[[Paper]](https://arxiv.org/abs/2603.28458)

## Code

Code coming soon.

## Authors

Yufei Xu, Fanxu Meng, Fan Jiang, Yuxuan Wang, Ruijie Zhou, Zhaohui Wang, Jiexi Wu, Zhixin Pan, Xiaojuan Tang, Wenjie Pei, Tongxuan Liu, Di Yin, Xing Sun, Muhan Zhang

## Citation

```bibtex
@article{xu2026hisa,
  title={HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention},
  author={Xu, Yufei and Meng, Fanxu and Jiang, Fan and Wang, Yuxuan and Zhou, Ruijie and Wang, Zhaohui and Wu, Jiexi and Pan, Zhixin and Tang, Xiaojuan and Pei, Wenjie and Liu, Tongxuan and Yin, Di and Sun, Xing and Zhang, Muhan},
  journal={arXiv preprint arXiv:2603.28458},
  year={2026}
}
```
