# ResMISA Hopper

Selected algorithm source: resmisa_prefill_final_20261019_v2. September minmax prefill
head selection is retained; prefill and decode token TopK both use resmisa_fast_topk.
The colleague paged decode scorer and shared correctness fixes remain.
This variant does not require the separately downloaded Blackwell headers.

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
