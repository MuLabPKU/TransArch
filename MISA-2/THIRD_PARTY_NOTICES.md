# Third-party notices

MISA-2 is distributed under the Apache License, Version 2.0; see [LICENSE](LICENSE).
Existing upstream copyright and license headers are retained in both variants.

## vLLM model integration

The files `resmisa_deepseek_v2.py` and `compat_vllm017.py` in both
`resmisa_vllm_patch` packages contain adaptations of
[vLLM](https://github.com/vllm-project/vllm/tree/v0.17.0), licensed under Apache-2.0.
The model integration also retains notices for the DeepSeek-AI and HuggingFace
contributors. These files are modified for ResMISA model registration,
indexer integration, and vLLM 0.17 compatibility; they are not unmodified upstream
vLLM files. See their source headers for the original attribution.

## External Blackwell headers

These dependencies are downloaded separately, not bundled with this release:

| Dependency | Pinned revision | License |
| --- | --- | --- |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | `58af0be80213ac1bd2993c6ff6f1c97f66e574f8` | MIT |
| [CUTLASS / CuTe](https://github.com/NVIDIA/cutlass) | `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8` | BSD-3-Clause |

Follow the [Blackwell dependency instructions](_blackwell/readme.md) and retain
the upstream license files when redistributing those dependencies.
Runtime dependencies such as vLLM, PyTorch, TileLang, and Triton are distributed
under their respective licenses. Model weights are not included in this release.
