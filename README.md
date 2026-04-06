# TransArch: Hardware-Aware Architecture Transfer for Foundation Models

Modern large language models are increasingly bottlenecked by communication rather than computation on today's hardware. **TransArch** is a collection of research projects that design hardware-friendly model architectures and migrate existing pre-trained models into these new architectures with minimal performance loss.

## 📦 Projects

| Project | Venue | Description | Paper |
|---------|-------|-------------|-------|
| **CLOVER** | ICML 2025 | Cross-layer SVD pruning of Q-K and V-O pairs in attention heads; combined with TransMLA achieves up to 11.1× speedup over LLaMA-2-7B | [arXiv](https://arxiv.org/abs/2411.17426) |
| **TransMLA** | NeurIPS 2025 Spotlight | Convert GQA models (LLaMA, Qwen, Mixtral, …) to DeepSeek-MLA with full Absorb compatibility and up to 10.6× inference speedup | [arXiv](https://arxiv.org/abs/2502.07864) |
| **TPLA** | ASPLOS 2026 | Tensor Parallel Latent Attention — partitions latent representations across devices, achieving 1.79×/1.93× speedup on DeepSeek-V3/Kimi-K2 | [arXiv](https://arxiv.org/abs/2508.15881) |
| **HISA** | Under Review | Hierarchical two-stage indexer for fine-grained sparse attention, achieving 2×–4× speedup at 32K–128K context | [arXiv](https://arxiv.org/abs/2603.28458) |

## 📰 News

- [2026.03] HISA preprint released: [arXiv:2603.28458](https://arxiv.org/abs/2603.28458).
- [2026.02] 🎉 TransMLA is adopted by Ant Group's latest 1T model [Ling-2.5-1T](https://huggingface.co/inclusionAI/Ling-2.5-1T)! This demonstrates the robust scalability of TransMLA in ultra-large-scale LLMs.
- [2025.11] TPLA accepted at ASPLOS 2026 (Summer cycle).
- [2025.09] TransMLA accepted at NeurIPS 2025 (Spotlight, Top 3.19%).
- [2025.05] CLOVER accepted at ICML 2025.

## 📋 To-Do

- [ ] Release TPLA code
- [ ] Release HISA code

## 📚 Citation

If you find our work useful, please cite the relevant paper(s):

```bibtex
@inproceedings{meng2025transmla,
  title={TransMLA: Multi-head Latent Attention Is All You Need},
  author={Meng, Fanxu and Tang, Pingzhi and Tang, Xiaojuan and Yao, Zengwei and Sun, Xing and Zhang, Muhan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}

@inproceedings{meng2025clover,
  title={CLOVER: Cross-Layer Orthogonal Vectors Pruning and Fine-Tuning},
  author={Meng, Fanxu and Tang, Pingzhi and Jiang, Fan and Zhang, Muhan},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}

@inproceedings{tang2026tpla,
  title={TPLA: Tensor Parallel Latent Attention for Efficient Disaggregated Prefill and Decode Inference},
  author={Tang, Xiaojuan and Meng, Fanxu and Tang, Pingzhi and Wang, Yuxuan and Yin, Di and Sun, Xing and Zhang, Muhan},
  booktitle={International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)},
  year={2026}
}

@article{xu2026hisa,
  title={HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention},
  author={Xu, Yufei and Meng, Fanxu and Jiang, Fan and Wang, Yuxuan and Zhou, Ruijie and Wang, Zhaohui and Wu, Jiexi and Pan, Zhixin and Tang, Xiaojuan and Pei, Wenjie and Liu, Tongxuan and Yin, Di and Sun, Xing and Zhang, Muhan},
  journal={arXiv preprint arXiv:2603.28458},
  year={2026}
}
```

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/chart?repos=MuLabPKU/TransArch&type=date&legend=top-left)](https://www.star-history.com/?repos=MuLabPKU%2FTransArch&type=date&legend=top-left)
