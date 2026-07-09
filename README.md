# TransArch: Hardware-Aware Architecture Transfer for Foundation Models

Modern large language models are increasingly bottlenecked by communication rather than computation on today's hardware. **TransArch** is a collection of research projects that design hardware-friendly model architectures and migrate existing pre-trained models into these new architectures with minimal performance loss.

## 📦 Projects

| Project | Venue | Description | Paper |
|---------|-------|-------------|-------|
| **CLOVER** | ICML 2025 | Cross-layer SVD pruning of Q-K and V-O pairs in attention heads; combined with TransMLA achieves up to 11.1× speedup over LLaMA-2-7B | [arXiv](https://arxiv.org/abs/2411.17426) |
| **TransMLA** | NeurIPS 2025 Spotlight | Convert GQA models (LLaMA, Qwen, Mixtral, …) to DeepSeek-MLA with full Absorb compatibility and up to 10.6× inference speedup | [arXiv](https://arxiv.org/abs/2502.07864) |
| **TPLA** | ASPLOS 2026 | Tensor Parallel Latent Attention — partitions latent representations across devices, achieving 1.79×/1.93× speedup on DeepSeek-V3/Kimi-K2 | [arXiv](https://arxiv.org/abs/2508.15881) |
| **HISA** | COLM 2026 | Hierarchical two-stage indexer for fine-grained sparse attention, achieving 2×–4× speedup at 32K–128K context | [arXiv](https://arxiv.org/abs/2603.28458) |
| **MISA** | Preprint | Mixture-of-experts routing over DSA indexer heads — matches dense DSA with 8×/4× fewer indexer heads and ~3.82× kernel speedup on H200 | [arXiv](https://arxiv.org/abs/2605.07363) |
| **GQLA** | Preprint | Group-Query Latent Attention — one set of weights, two decoding paths (MQA-absorb for H100, GQA+MTP for H20), with up to 8-way TP | [arXiv](https://arxiv.org/abs/2605.15250) |

## 📰 News

- [2026.07.09] HISA accepted at COLM 2026: see [HISA_COLM_2026/](HISA_COLM_2026/).
- [2026.06.30] HISA's hierarchical indexing is used in LongCat-2.0's LongCat Sparse Attention (LSA): see the [LongCat-2.0 tech blog](https://longcat.chat/blog/longcat-2.0/).
- [2026.05] GQLA preprint released: [arXiv:2605.15250](https://arxiv.org/abs/2605.15250).
- [2026.05] MISA preprint released: [arXiv:2605.07363](https://arxiv.org/abs/2605.07363).
- [2026.02] 🎉 TransMLA is adopted by Ant Group's latest 1T model [Ling-2.5-1T](https://huggingface.co/inclusionAI/Ling-2.5-1T)! This demonstrates the robust scalability of TransMLA in ultra-large-scale LLMs.
- [2025.11] TPLA accepted at ASPLOS 2026 (Summer cycle).
- [2025.09] TransMLA accepted at NeurIPS 2025 (Spotlight, Top 3.19%).
- [2025.05] CLOVER accepted at ICML 2025.

## 📋 To-Do

- [ ] Release TPLA code

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

@article{zhou2026misa,
  title={MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference},
  author={Zhou, Ruijie and Meng, Fanxu and Xu, Yufei and Liu, Tongxuan and Lu, Guangming and Zhang, Muhan and Pei, Wenjie},
  journal={arXiv preprint arXiv:2605.07363},
  year={2026}
}

@article{meng2026gqla,
  title={GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model Decoding},
  author={Meng, Fanxu},
  journal={arXiv preprint arXiv:2605.15250},
  year={2026}
}
```

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/chart?repos=MuLabPKU/TransArch&type=date&legend=top-left)](https://www.star-history.com/?repos=MuLabPKU%2FTransArch&type=date&legend=top-left)
