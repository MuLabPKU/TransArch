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

Reference implementations are available in the following repositories:

- **GPU TileLang kernels:** [tile-ai/tilelang examples/dsa_hisa](https://github.com/tile-ai/tilelang/tree/main/examples/dsa_hisa)
- **Ascend TileLang kernels:** [leavelet/hisa-ascend](https://github.com/leavelet/hisa-ascend)
- **End-to-end SGLang integration:** [xuyufei-a/sglang_hisa, branch `hisa_pr`](https://github.com/xuyufei-a/sglang_hisa/tree/hisa_pr)


### End-to-End Usage

The end-to-end implementation is based on SGLang and should be used with the environment described in the [SGLang HISA PR](https://github.com/sgl-project/sglang/pull/24672). In particular, the implementation was validated on top of the official SGLang `v0.5.11` Docker image (`lmsysorg/sglang:latest`, CUDA 13). Inside that container, replace the bundled SGLang package with the HISA branch and install a CUDA-13-compatible TileLang release:

```bash
pip uninstall -y sglang
git clone https://github.com/xuyufei-a/sglang_hisa.git
cd sglang_hisa
git checkout hisa_pr
pip install -e python
pip install --upgrade 'tilelang==0.1.9'
```

HISA is enabled through model config overrides:

```bash
export SGLANG_NSA_FUSE_TOPK=0
python -m sglang.launch_server \
  --model /path/to/DeepSeek-V3.2 \
  --tp 8 \
  --json-model-override-args '{"use_hisa": true, "hisa_k_block_size": 128, "hisa_block_topk": 64}'
```

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
