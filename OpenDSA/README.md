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

## Performance & memory optimizations

The whole recipe is built around one constraint — long context (up to 200k) on an
80 GB budget — so every stage trades compute cleverly to stay memory-bounded and
fast:

- **Per-layer eager (immediate) warmup backward.** Because each layer's indexer KL
  is independent (hidden states flow under `no_grad`, teacher detached),
  `IndexerLossRegistry` can backward each layer's loss the moment it is produced and
  keep only a detached scalar (`set_eager_backward`, `opendsa/modeling/dsa_attention.py`;
  driven by `DSATrainer.training_step`). This frees each layer's graph before the next
  layer runs, dropping peak activation from `num_layers · O(L·H)` to `~1 · O(L·H)`
  and removing the need for activation recompute during warmup — while giving
  gradients numerically identical to the sum-then-backward path.
- **FlashKL `O(L·H)` warmup/sparse loss.** The KL against the frozen teacher is linear
  in the teacher probabilities, so it decomposes per head and is accumulated with
  Flash-Attention-style online-softmax tiling over the key axis — the `[L,H,L]`
  attention matrix is never materialized (`opendsa/ops/flashkl_warmup.py`,
  `opendsa/ops/topk_select.py`).
- **Memory-adaptive warmup key-tile.** FlashKL's peak is dominated by `O(Lq·H·tile)`
  fp32 scratch, so the key-tile — not the input dtype — is the real memory knob at
  long context. `auto_warmup_tile` sizes it from live free GPU memory (capped by the
  configured `dsa_tile`, falling back to the cap on CPU / if the driver can't report
  free memory), so short context keeps the fast wide tile while a memory-tight rank
  automatically shrinks it (e.g. ~2 GB free at 16k ctx: tile 512→256, peak −44%) to
  dodge OOM. The loss is bit-identical regardless of tile (the streaming sum is
  exact). Note: keeping these fp32 scratch reductions in fp32 is a numerical
  requirement (online-softmax stability), so — absent a fused kernel — running the
  q·k GEMMs in bf16 neither speeds this pure-torch path up nor meaningfully shrinks
  its peak; the tile is the lever.
- **No logits when we don't need them.** The `[B, L, vocab]` fp32 logits tensor is
  huge (~10 GB/GPU at 200k, CP=8). Warmup and diagnostics don't need it (their loss
  comes from the indexer, not next-token prediction), so those paths run the decoder
  stack alone and skip the LM head entirely — the logits are never built.
- **Sliced logits when we do.** The sparse stage needs a real next-token CE loss, so
  it can't skip the LM head — but instead of projecting all `L` tokens at once, it
  slices them into chunks, runs LM-head + CE one chunk at a time, and frees each
  chunk's logits before the next (recomputing them in the backward pass via
  `torch.utils.checkpoint`). Peak logits memory drops from `O(L · vocab)` to
  `O(chunk_size · vocab)`, and the loss is numerically identical to the one-shot
  version (`_chunked_shift_ce`).
- **Absorbed-MLA sparse attention.** Sparse MLA runs in DeepSeek-V2 latent space and
  gathers the *shared* latent `[chunk, K, 576]` once instead of per-head K/V, plus
  query chunking + gradient checkpointing (`sparse_attend_absorbed_chunked`).
- **Cross-document attention masking.** Packing carries per-doc `cu_seqlens`, and the
  dense LM path uses `flash_attn_varlen` so attention never crosses conversation
  boundaries — this both improves quality and speeds training (ProLong finding).
- **Distributed sharding matched to the bottleneck.** CP shards one long sequence's
  context activations (KV all-gather in absorbed latent space), EP shards the routed
  MoE experts (all-to-all dispatch), and `Zero2Adam` shards non-expert optimizer
  state — fusing the CP grad-sum into a single `reduce_scatter` — without FSDP's
  per-layer param all-gather.
- **Zigzag CP sequence sharding.** Causal attention makes late tokens attend more
  keys, so a contiguous split leaves the last rank doing ~2× the first rank's work.
  CP splits the sequence into `2·cp_size` chunks and gives rank r chunks `{r, 2n-1-r}`
  (one early, one late), equalizing per-rank causal work; gathered keys are reordered
  back to sequential order so causal masks and top-k are unchanged (`local_shard`,
  `zigzag_local_gpos` in `opendsa/dist/pg.py`, verified numerically equal to the
  single-GPU reference).

All operators are pure-torch and memory-bounded (query-chunked + gradient-
checkpointed). The warmup FlashKL loss is checked to ~1e-5 grad error against a
dense `[L,H,L]` reference (`dense_warmup_reference`) on a real DeepSeek-V2 teacher,
and the distributed paths are gated by numerical-equivalence tests:

```bash
python tests/test_patch_integration.py # tiny real DS-V2: warmup+sparse grad flow + FlashKL vs dense
python tests/test_pg.py                 # CP zigzag collectives round-trip
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

# 1. build a packed dataset from Hugging Face (UltraData-style messages, per-doc masking)
python -m opendsa.data.long_pack --seq-len 8192 --out data_cache/pack_8k \
    --data fxmeng/UltraData-SFT-2605-32k-200k --max-packed 4000

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
- **Long-context data**: trains on the Hugging Face dataset
  `fxmeng/UltraData-SFT-2605-32k-200k` (real 33k–200k-token docs). Warmup's LM
  path uses `flash_attn_varlen` and a
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
# single-process checks
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
