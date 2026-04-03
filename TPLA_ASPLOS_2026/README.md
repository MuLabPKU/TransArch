# TPLA: Tensor Parallel Latent Attention [ASPLOS 2026]

Multi-Head Latent Attention (MLA) compresses the KV cache into low-rank latent vectors, but under tensor parallelism each device must still load the full latent cache — negating MLA's memory advantage. **TPLA** solves this by partitioning both the latent representation and each attention head's input dimension across devices, performing attention independently per shard, and combining results with a single all-reduce.

**Highlights:**
- Preserves compressed KV cache benefits while enabling efficient tensor parallelism.
- Every head still leverages the full latent representation, maintaining stronger representational capacity than Grouped Latent Attention (GLA).
- Drop-in compatible with models pre-trained using MLA — supports MLA-style prefilling and efficient tensor-parallel decoding without retraining.
- Orthogonal transforms (Hadamard or PCA) before TP slicing further reduce cross-shard interference.
- Achieves **1.79× and 1.93× speedups** on DeepSeek-V3 and Kimi-K2 respectively at 32K-token context length, with no degradation on commonsense and LongBench benchmarks.

[[Paper]](https://arxiv.org/abs/2508.15881)

## Code

Code coming soon.

## Authors

Xiaojuan Tang\*, Fanxu Meng\*, Pingzhi Tang, Yuxuan Wang, Di Yin, Xing Sun, Muhan Zhang

## Citation

```bibtex
@inproceedings{tang2026tpla,
  title={TPLA: Tensor Parallel Latent Attention for Efficient Disaggregated Prefill and Decode Inference},
  author={Tang, Xiaojuan and Meng, Fanxu and Tang, Pingzhi and Wang, Yuxuan and Yin, Di and Sun, Xing and Zhang, Muhan},
  booktitle={International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)},
  year={2026}
}
```
