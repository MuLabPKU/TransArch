# LISA: Head-Fused Linear Indexing for Efficient Sparse Attention

[Paper](lisa.pdf)

LISA is a training-free sparse-attention indexer. This release contains its
core kernels and adapters for vLLM and SGLang.

| Variant | API name | Heads |
|---|---|---|
| LISA: head-fused linear indexing | `linear` | 32 / 64 |
| LISA†: candidate refinement | `threshold` | 32 / 64 |
| LISA‡: four-layer candidate sharing | `shared4` | 64 |

## Installation

Requires B200/GB200 (SM100) and a CUDA 13 toolkit with `nvcc`. Use separate
Python environments for vLLM and SGLang. SGLang's Linux wheels require glibc >=2.34.
The GB200 smoke tests used Linux aarch64, Python 3.12.3 and these versions:

| Framework | PyTorch build | Triton | Transformers | FlashInfer | CuTe DSL |
|---|---|---|---|---|---|
| vLLM 0.26.0 | 2.11.0+cu130 | 3.6.0 | 5.16.1 | 0.6.14 | 4.6.0, cu13 |
| SGLang 0.5.21 | 2.13.0+cu130 | 3.7.1 | 5.12.1 | 0.6.18, cu13 | 4.6.2, cu13 |

[requirements-vllm.txt](requirements-vllm.txt) and
[requirements-sglang.txt](requirements-sglang.txt) pin the key packages from these
environments; pip resolves the remaining framework dependencies. Both tested
environments passed `pip check`. Installation from an empty environment has not
yet been tested.

Clone LISA, then install one framework and LISA in its own environment:

```bash
git clone https://github.com/MuLabPKU/TransArch.git
cd TransArch/LISA
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install -r requirements-vllm.txt
python -m pip install --no-deps -e .
python -m pip check
```

For SGLang, use `.venv-sglang` and `requirements-sglang.txt` instead.

The linear CUDA path uses external CUTLASS and DeepGEMM headers. Keep these
outside this repository. The following versions plus the small compatibility
patch reproduce the headers used by the measured kernels:

```bash
git clone --depth 1 --branch v4.2.1 https://github.com/NVIDIA/cutlass.git /path/to/cutlass
git clone https://github.com/deepseek-ai/DeepGEMM.git /path/to/DeepGEMM
git -C /path/to/DeepGEMM checkout 7f2a703ed51ac1f7af07f5e1453b2d3267d37d50
git -C /path/to/DeepGEMM apply /absolute/path/to/LISA/lisa_indexer/deepgemm.patch
export CUTLASS_INCLUDE_DIR=/path/to/cutlass/include
export DEEPGEMM_INCLUDE_DIR=/path/to/DeepGEMM/deep_gemm/include
```

CUTLASS v4.2.1 is commit `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`.
Existing matching headers can be used through the same two environment variables.
Extensions compile on first use; `python -m lisa_indexer.build` builds them ahead
of time. Importing the package does not initialize CUDA or compile anything.

## Quick start

After installing dependencies and setting up the headers above, choose one
framework below. The smoke test uses one GPU and a five-layer DeepSeek-V3.2
configuration with dummy weights; it does not download model weights or a tokenizer.

vLLM:

```bash
python -m lisa_indexer.integrations.patch --framework vllm --apply
CUDA_VISIBLE_DEVICES=0 python -m lisa_indexer.smoke --framework vllm
```

SGLang:

```bash
python -m lisa_indexer.integrations.patch --framework sglang --apply
CUDA_VISIBLE_DEVICES=0 python -m lisa_indexer.smoke --framework sglang
```

The default runs `linear`, `threshold` and `shared4` with 64 indexer heads.
Add `--heads 32` to run the two supported 32-head variants, or `--variant linear`
to run just one case. Running both head counts in both environments covers ten
cases. `--dry-run` prints the model and launch settings without importing either
framework or initializing CUDA.

Each case sends 66,560 prompt tokens in 1,024-token chunks and generates two tokens.
Linear uses dense FP8x2 through 65,536 tokens; the final full chunk exercises HOT.
The script checks this length requirement before starting the model. A pass requires
actual LISA calls, zero prefill fallbacks, the linear HOT path when selected, and
a complete cross-layer schedule with no live pools for `shared4`.
The script enables and resets LISA inside the worker between cases.
Results and installed dependency versions are saved to
`results/smoke-<timestamp>/summary.json`; use `--output NEW_DIRECTORY` to choose
another location. A failed check exits nonzero and records the error once the
engine run has started.

First import and JIT compilation can take several minutes. Each variant prints
`START` before its worker setup and generation, then `PASS` after its checks.
The script supplies eager execution, FP8 KV, disabled prefix/radix caches and vLLM's
`kernel_config={"enable_cutedsl_warmup": False}`. If startup fails, check:

- **Missing libraries or symbols:** align dependencies with the chosen framework
  release in its own environment; the two releases use different versions.
- **Source hash mismatch:** the patch targets official release sources. Check
  the installed revision and `--framework-root` before applying it.
- **Missing headers or `nvcc`:** set the two include-directory variables above
  and point `CUDA_HOME` to a CUDA 13 toolkit.

GB200 testing of unmodified `61d5645` with the default prompt length passed
all ten cases: 3,200 LISA calls, zero fallbacks and five HOT calls per linear case.
This is a dummy-weight functional check; real-weight quality, GLM-specific
integration and paper performance require separate validation.

## Core usage

```python
from lisa_indexer import Indexer, PreparedFP8, PrefillContext

# q8: E4M3 [Q,H,128]; k8: E4M3 [N,128].
# scales: FP32 [N]; weights: FP32 [Q,H], including the query scale once.
indexer = Indexer("linear", max_n=1048576, num_layers=61)
prepared = PreparedFP8(q8, k8, scales, weights)
context = PrefillContext(request_id="request", layer_id=0,
                         query_start=query_offset, sequence_length=N)
indices = indexer.select(prepared, starts, ends, context=context)
indexer.reset()
```

Inputs are contiguous on one SM100 device. `starts` and `ends` are int32 `[Q]`:
all starts are zero and `ends[row] = query_offset + row + 1`. Output is int32
`[Q,2048]`, with request-relative indices and `-1` padding. Q is at most 8192;
N is at most 1,048,576. These are approximate indexers.

Keep one indexer across a request's layers/chunks. For `shared4`, call
`begin_forward(request_id)` before each model forward, execute all layers in
order, then call `end_forward()`. Layer 0 is independent; groups are 1–4,
5–8, and so on. Call `reset()` when the request ends or fails.

## vLLM and SGLang

Both adapters take gathered FP8 Q/K, K scales and effective weights before the
framework's scoring/Top-K step. They leave KV storage, attention and decode
with the framework. The patches target official
[vLLM 0.26.0](https://github.com/vllm-project/vllm/tree/v0.26.0) and
[SGLang 0.5.21](https://github.com/sgl-project/sglang/tree/v0.5.21).

For editable checkouts, pass the checkout root as `--framework-root`.
Add `--model-family glm52` for GLM. Use `--check` to verify source compatibility.
[sources.json](lisa_indexer/integrations/sources.json) records the upstream tags,
commits and source hashes. For other revisions, use
`VLLMAdapter` or `SGLangAdapter` from
`lisa_indexer.integrations.vllm` / `.sglang` at the equivalent call site.

For the offline hooks, set `LISA_INDEXER=linear` (or `threshold` /
`shared4`), `LISA_SINGLE_REQUEST=1`, `LISA_STRICT=1`, and a unique
`LISA_REQUEST_ID`. Set `LISA_NUM_LAYERS=61` for DeepSeek-V3.2 or `78` for GLM-5.2.
Use one eager request, CP=PP=1, FP8 KV, chunk <=8192, prefix cache and MTP off.
Reset worker state between requests; these hooks do not implement a concurrent
serving scheduler.

DeepSeek-V3.2 supports all three variants. GLM-5.2 supports `linear` and
`threshold`, retaining its native IndexCache reuse at 21 indexer layers; G4 is
unsupported for GLM.

## License

Original code is [Apache-2.0](LICENSE). Third-party attributions and licenses
are in [NOTICE](NOTICE). Please cite the accompanying [paper](lisa.pdf).

## Authors

Zhixin Pan, Fanxu Meng, Zhaohui Wang, Taosong Fang, Muhan Zhang
