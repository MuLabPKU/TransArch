# OpenDSA

Convert **DeepSeek-V2-Lite-Chat**'s MLA attention into **DSA** (DeepSeek Sparse
Attention) with a lightweight two-stage recipe on top of Hugging Face
Transformers/Accelerate, train on chat data, and evaluate long-context behavior.
Design goals: 8×H100, simple code, fast
training, and numerically checked distributed paths.

## Method (two stages, monkeypatch — no model rewrite)

A `LightningIndexer` is attached to every `DeepseekV2Attention` module and its
`forward` is replaced with a DSA-aware one (`opendsa/modeling/dsa_attention.py`).
The stage is selected per-module via `dsa_mode`:

1. **Warmup** (`dsa_mode="warmup"`, backbone frozen): the indexer is distilled to
   imitate the model's own (head-averaged) MLA attention distribution via a
   **FlashKL** KL loss that is `O(L·H)` in memory — it never materializes the
   `[L,H,L]` attention matrix. The dense LM path is unchanged. Only indexer params
   train.
2. **Sparse** (`dsa_mode="sparse"`, backbone trainable): the indexer selects the
   top-k keys per query; attention runs only over them (sparse MLA). Loss =
   `LM_loss + λ · indexer_KL(over selected keys)`.

All operators have pure-torch, CPU/float64-verifiable reference paths plus
memory-bounded (query-chunked + gradient-checkpointed) training paths. Every op
self-checks to ~1e-16 grad error against its dense reference:

```bash
python opendsa/ops/flashkl_warmup.py   # warmup FlashKL vs dense autograd
python opendsa/ops/topk_select.py      # sparse KL (FlashKL + chunked) vs dense
python opendsa/ops/sparse_mla.py       # sparse attention (ref + chunked) vs dense
python tests/test_patch_integration.py # tiny real DS-V2: warmup+sparse grad flow
```

## Layout

```
opendsa/
  modeling/  indexer.py, dsa_attention.py, patch_deepseek.py   # the DSA patch
  ops/       flashkl_warmup.py, topk_select.py, sparse_mla.py  # O(L·H) / chunked ops
  train/     build.py, warmup.py, sparse.py, trainer.py, convert_stage.py
  data/      long_pack.py (chat packing + per-doc cu_seqlens), collator.py
configs/     accelerate_ddp.yaml (warmup/CP), accelerate_fsdp.yaml (sparse FSDP)
scripts/     run_warmup.sh, run_sparse.sh
```

## Install

```bash
conda create -n opendsa python=3.12 -y
conda activate opendsa

# Install a CUDA-matched PyTorch build if your base image does not already have one.
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

Set `SKIP_FLASH_ATTN=1` only for CPU/operator checks; real warmup training needs
flash-attn for the dense LM path.

## Run

```bash
# Optional: create a local env.sh for site-specific CUDA/HF cache setup.
# The run scripts source it automatically if it exists.

# 1. build a packed dataset (UltraData-style JSONL, per-doc masking)
python -m opendsa.data.long_pack --seq-len 8192 --out data_cache/pack_8k \
    --data /path/to/UltraData-SFT-2605-split/32k-200k --max-packed 4000

# 2. train the indexer warmup stage
DATA=data_cache/pack_8k OUT=runs/warmup SEQ_LEN=8192 bash scripts/run_warmup.sh

# 3. train the sparse stage from the warmup indexer
DATA=data_cache/pack_8k OUT=runs/sparse INDEXER=runs/warmup/indexer.pt \
    SEQ_LEN=8192 bash scripts/run_sparse.sh
```

Distributed choice is dictated by memory, not preference:
- **Warmup → DDP** (`configs/accelerate_ddp.yaml`). The backbone is frozen (fits one
  H100), so we replicate and only sync the tiny indexer grads. The warmup loss is a
  module-registry side-effect rather than the model's output, so DDP's autograd
  reducer can't track it — the trainer runs the *unwrapped decoder stack* and
  **manually all-reduces the indexer grads** (`DSATrainer.training_step`).
- **Sparse, default → FSDP ZeRO-3** (`configs/accelerate_fsdp.yaml`). This is the
  standard sparse continued-training path for moderate context lengths. The full
  backbone trains, so params/grads/optimizer are sharded across GPUs; per-layer
  gradient-checkpointing + query chunking bound activation memory.
- **Long sparse → CP + EP + Zero2Adam** (`CP_MODE=1 bash scripts/run_sparse.sh`).
  FSDP shards model state, but it does not shard one very long sequence's context
  activations. For long contexts, CP splits the sequence across GPUs, EP shards the
  MoE routed experts, and `Zero2Adam` shards non-expert optimizer state without
  FSDP.

Tunables are env-overridable: `TOPK`, `GRAD_ACCUM`, `SEQ_LEN`, `STEPS`, `NPROC`,
`QCHUNK` (sparse), and `CP_MODE`.

## Notes

- **Absorbed-MLA sparse attention** (`sparse_attend_absorbed_chunked`): runs
  sparse MLA in DeepSeek-V2 latent space — absorb `q_nope` via `W_UK` into the
  `kv_lora` space and gather the *shared* latent `[chunk, K, 576]` once instead of
  per-head K `[chunk,K,H,192]` + V `[chunk,K,H,128]`; value comes from the same
  latent via `W_UV`.
- **Long-context data**: trains on `UltraData-SFT-2605-split/32k-200k` (real
  33k–200k-token docs). Warmup's LM path uses `flash_attn_varlen` and a
  query-chunked recall metric so no `[L,H,L]` matrix is built for diagnostics.
- **Context + Expert Parallel (CP + EP), no FSDP.** `CP_MODE=1 bash scripts/run_warmup.sh`
  splits each sequence across the 8 GPUs (KV all-gather in absorbed latent space);
  sparse `CP_MODE=1 bash scripts/run_sparse.sh` adds expert parallel (each rank owns
  `n_experts/NPROC` experts, all-to-all token dispatch). The parallel components
  are gated by numerical-equivalence tests: `tests/test_pg.py`,
  `tests/test_cp_equiv.py`, `tests/test_ep_equiv.py`, `tests/test_zero2_equiv.py`,
  and `tests/test_sparse_cp_ep_equiv.py`.

## Tests

```bash
# CPU/single-process checks
python opendsa/ops/flashkl_warmup.py
python opendsa/ops/topk_select.py
python opendsa/ops/sparse_mla.py
python tests/test_pg.py

# GPU integration and distributed equivalence checks
CUDA_VISIBLE_DEVICES=0 python tests/test_patch_integration.py
NCCL_DEBUG=WARN torchrun --nproc_per_node=2 tests/test_pg.py
NCCL_DEBUG=WARN torchrun --nproc_per_node=2 tests/test_zero2_equiv.py
NCCL_DEBUG=WARN torchrun --nproc_per_node=2 tests/test_cp_equiv.py
NCCL_DEBUG=WARN torchrun --nproc_per_node=2 tests/test_ep_equiv.py
NCCL_DEBUG=WARN torchrun --nproc_per_node=2 tests/test_sparse_cp_ep_equiv.py
```

The DeepSeek-based tests require `deepseek-ai/DeepSeek-V2-Lite-Chat` remote code
to be available through Hugging Face cache or network access.

## Acknowledgements

OpenDSA's FlashKL-based warmup and sparse KL operators reference the FlashKL
project: [XiaojuanTang/FlashKL](https://github.com/XiaojuanTang/FlashKL).
