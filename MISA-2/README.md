# MISA-2: ResMiSA: Residual Mixture of Sparse Attention for Long-Context LLM Inference [Preprint]

This folder hosts MISA-2; the accompanying paper is titled **ResMiSA**.

ResMiSA extends MISA with an always-active shared expert that fuses the linear contributions of all indexer heads, while routed residual experts supply nonlinear corrections. The shared path preserves a direct gradient path through every head. Combined with LiteTopK, ResMiSA achieves a 2.43× end-to-end time-to-first-token speedup over DSA at 1M context.

[[Paper (PDF)]](resmisa.pdf)

## Code

The inference implementation is available in two variants. Internal package names
use `ResMISA`; this is the MISA-2 / ResMiSA method described in the paper.

| Variant | Implementation | Setup |
| --- | --- | --- |
| [`_hopper/`](_hopper/) | Minmax prefill head selection and `resmisa_fast_topk` for prefill/decode token selection | [Installation and launcher](_hopper/readme.md) |
| [`_blackwell/`](_blackwell/) | SM100 HOT prefill selection and exact decode TopK with paged minmax scoring | [Installation, pinned headers, and launcher](_blackwell/readme.md) |

This release includes the vLLM model plugin, indexer and CUDA/Triton/TileLang
kernels, runtime configuration, and serving launchers. It does not include model
weights, training code, or the complete paper evaluation suite.

### Environment and quick start

Use an isolated environment with **vLLM 0.17.0**, **TileLang 0.1.7.post3**,
**Triton 3.6.0**, Ninja, setuptools >= 68, and wheel. The source's recorded
environment used PyTorch 2.10.0+cu130, CUDA 13.0, and Transformers 4.57.6.
A CUDA C++ compiler is required for native compilation at first use. The plugin
checks vLLM source/signature compatibility; arbitrary vLLM versions are not
supported.

Install **one variant per environment**: both provide the same Python package
and vLLM plugin entry point. For the Blackwell variant, first follow its
[pinned DeepGEMM/CUTLASS header setup](_blackwell/readme.md).

From the repository root, after preparing the dependencies:

```bash
cd MISA-2/_blackwell  # alternatively: MISA-2/_hopper
python -m pip install --no-index --no-deps --no-build-isolation ./resmisa_vllm_patch_v1_0
bash ds_v32_vllm_serve.sh /path/to/DeepSeek-V3.2
```

The launcher defaults to tensor parallelism 8, BF16, seed 17, a maximum model
length of 135168, prefill chunks of 8192 tokens, one concurrent request, eager
execution, and no prefix caching. Both variants enable fused minmax selection
with eight selected heads in prefill and decode. The launcher reads the local
`config.json` as Hugging Face overrides without modifying the model checkpoint.
It listens on `127.0.0.1:8000`; `TP` and `PORT` can be set in the environment,
and additional vLLM arguments can be appended after the model path. GPU
placement and multi-node networking are configured by the caller.

The source's recorded hardware validation is **GB200 / SM100**. The `_hopper`
directory name alone does not establish H100 hardware qualification. The default
launcher is a serving example; it does not reproduce the paper's 1M-context
performance measurement.

### License

The code is released under [Apache-2.0](LICENSE). See
[third-party notices](THIRD_PARTY_NOTICES.md) for upstream attribution and the
licenses of external Blackwell headers.

## Authors

Ruijie Zhou, Fanxu Meng, Zhixin Pan, Jinbao Xue, Ke Zhang, Jun Yu, Wenjie Pei
