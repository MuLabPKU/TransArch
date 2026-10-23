# ResMISA Blackwell HOT

Selected algorithm source: resmisa_clean_v1. Complete HOT prefill uses select_prefill;
decode token TopK remains exact_topk with the colleague paged minmax scorer.
The shared correctness fixes remain. Unused diagnostic exports and experimental
interfaces have been removed from the native wrappers.

## Download external headers

Third-party headers are not bundled. Download the fixed public repositories below
once, before first use. No Python installation of this external DeepGEMM checkout
is needed; it supplies C++ headers only and does not replace vLLM's runtime dependency.

- DeepGEMM: https://github.com/deepseek-ai/DeepGEMM
  commit `58af0be80213ac1bd2993c6ff6f1c97f66e574f8` (MIT).
- CuTe and CUTLASS: https://github.com/NVIDIA/cutlass
  submodule commit `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8` (BSD-3-Clause).
  The DeepGEMM commit above pins this exact submodule.

Run in Bash using a new dependency directory (change the first path if desired):

```bash
export RESMISA_DEPS="$PWD/resmisa-deps"
mkdir -p "$RESMISA_DEPS"
git init -q "$RESMISA_DEPS/DeepGEMM"
git -C "$RESMISA_DEPS/DeepGEMM" remote add origin https://github.com/deepseek-ai/DeepGEMM.git
git -C "$RESMISA_DEPS/DeepGEMM" fetch --depth 1 origin 58af0be80213ac1bd2993c6ff6f1c97f66e574f8
git -C "$RESMISA_DEPS/DeepGEMM" checkout --detach FETCH_HEAD
git -C "$RESMISA_DEPS/DeepGEMM" submodule update --init --depth 1 third-party/cutlass
export RESMISA_DEEPGEMM_INCLUDE="$RESMISA_DEPS/DeepGEMM/deep_gemm/include"
export RESMISA_CUTLASS_INCLUDE="$RESMISA_DEPS/DeepGEMM/third-party/cutlass/include"
```

Preserve the two include environment variables in every worker process that builds
extensions. Keep these checkouts outside the installed Python package; on multiple
nodes provide the same pinned dependencies and valid local paths on each node.
The package neither downloads dependencies nor silently falls back to another version.
Missing include roots produce an error naming the environment variable to set.
The download commands were exercised in a temporary directory: all 166 previously
bundled headers matched byte-for-byte (107 CuTe, 49 CUTLASS, 10 DeepGEMM).
The two upstream repositories retain their own license notices; no third-party
header or dependency checkout is included in this source package.

## Install and run

Use an isolated environment with vLLM 0.17.0 and its compatible PyTorch/CUDA,
TileLang 0.1.7.post3, Triton 3.6.0, Ninja, setuptools>=68 and wheel already installed.
The recorded environment used PyTorch 2.10.0+cu130, CUDA 13.0 and Transformers 4.57.6.
A CUDA C++ compiler is required for first-use native compilation. Existing exact
vLLM source/signature checks still apply. Install only one variant per environment.

From this variant directory:

```bash
python -m pip install --no-index --no-deps --no-build-isolation ./resmisa_vllm_patch_v1_0
bash ds_v32_vllm_serve.sh /path/to/DeepSeek-V3.2
```

The launcher uses TP8, seed17, eager, chunk8192, max_len135168, one request,
no prefix caching, fuse/minmax=True and eight heads. `config.json` supplies hf_overrides;
it does not modify the checkpoint. The caller manages GPU placement and multi-node networking.

This source cleanup removes added diagnostics, unused experimental interfaces and
forwarding modules. The selected minmax/HOT and TopK paths remain fixed.
Recorded hardware validation uses GB200/SM100; the Hopper directory name does not
establish H100 hardware qualification.
Existing copyright and license notices in the source must be retained.
