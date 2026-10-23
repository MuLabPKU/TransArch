"""Persistent pooling buffers managed by the indexer rather than vLLM cache specs.

Buffers are keyed by (prefix, variant, max_pool_blocks, dim). Batch sizes share
fixed-address storage allocated for MAX_BATCH, allowing CUDA graph reuse.
The incremental path recomputes trailing pooling blocks and retains complete
historical blocks. Start positions are derived from sequence lengths, so repeated
calls for the same length do not advance the update window.

The eager invalidation helper checks each row's first physical KV block and
sequence length. A changed block or shorter sequence requires full recomputation
when the update window cannot cover the historical blocks. The graph-compatible
helper skips invalidation and requires callers to initialize historical blocks.
"""
import os
import torch

# Number of trailing blocks updated per step; also the incremental kernel's
# second grid dimension. This must cover blocks added or modified in one step.
INCR_BLOCKS = 2

# Incremental caching is disabled by default.
CACHE_ENABLED = True

# Use batch * pool_blocks as a work estimate for the cache eligibility threshold.
_MIN_WORK = True

# In graph-compatible mode, callers use full pooling during eager execution and
# incremental pooling during capture/replay to keep the captured kernel sequence
# independent of tensor data. Historical blocks must be initialized beforehand;
# graph replay does not rerun Python invalidation checks.
CUDAGRAPH_MODE = True

# Allocate one maximum-batch buffer per key and reuse [:batch] views at a fixed
# base address. This bound must cover the largest runtime batch (max_num_seqs).
MAX_BATCH = int(os.environ.get("ResMISA_POOL_MAX_BATCH", "256"))


def is_cudagraph_replaying() -> bool:
    """Check whether the vLLM forward context uses a non-eager CUDA graph mode.

    NONE includes eager warmup; PIECEWISE and FULL indicate graph execution.
    Return False if the context or its graph mode cannot be obtained, allowing
    callers to use full pooling rather than assume initialized cache contents.
    """
    try:
        from vllm.forward_context import get_forward_context
        from vllm.config import CUDAGraphMode
        ctx = get_forward_context()
        return ctx.cudagraph_runtime_mode != CUDAGraphMode.NONE
    except Exception:
        return False


class _CacheEntry:
    """Persistent state for one (prefix, variant, max_pool_blocks, dim) key.

    Allocate for MAX_BATCH and share [:batch] views across batch sizes; the
    runtime batch size is not part of the cache key.
    """

    __slots__ = ("blocked_k", "prev_first_blk", "prev_seq_len",
                 "max_batch", "max_pool_blocks", "dim")

    def __init__(self, max_batch, max_pool_blocks, dim, device):
        # Fixed-address storage shared by runtime batch views and CUDA graphs.
        self.blocked_k = torch.zeros(
            (max_batch, max_pool_blocks, dim), device=device, dtype=torch.float32
        )
        # CPU snapshots used for invalidation, sized to the preceding batch.
        self.prev_first_blk = None  # torch.Tensor [batch] int64 on CPU
        self.prev_seq_len = None    # torch.Tensor [batch] int64 on CPU
        self.max_batch = max_batch
        self.max_pool_blocks = max_pool_blocks
        self.dim = dim


# Map (prefix, variant, max_pool_blocks, dim) to persistent state.
_CACHE: dict = {}


def reset_cache():
    """Clear all cache entries, for example between tests or execution contexts."""
    _CACHE.clear()


def get_pooling_cache(
    prefix: str,
    variant: str,
    max_pool_blocks: int,
    dim: int,
    device: torch.device,
) -> _CacheEntry:
    """Get or allocate fixed-address storage for the given cache key.

    Callers slice entry.blocked_k[:batch] from the MAX_BATCH allocation.
    All batch sizes share this storage rather than allocating separate buffers.
    """
    key = (prefix, variant, int(max_pool_blocks), int(dim))
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CacheEntry(MAX_BATCH, max_pool_blocks, dim, device)
        _CACHE[key] = entry
    return entry


def compute_start_block_and_invalidate(
    entry: _CacheEntry,
    context_lens: torch.Tensor,  # [B] int32, including tokens added this step
    block_tables: torch.Tensor,  # [B, max_blocks] int32
    k_block_size: int,
    incr_blocks: int = INCR_BLOCKS,
) -> torch.Tensor:
    """Derive each row's update start and detect invalid cached history.

    start = max(0, ceil(seq_len / k_block_size) - incr_blocks) depends only on
    the current length, not accumulated state. A changed first physical block
    or a shorter sequence invalidates a row. If its historical blocks extend
    before start, the fixed incremental grid cannot refill them and the entire
    batch requires full pooling. New entries or changed batch sizes also require
    full pooling when their history exceeds the incremental window.

    Returns:
        A pair of start_block ([B] int32 on context_lens.device) and
        fallback_full (bool indicating that the batch needs full pooling).
    """
    device = context_lens.device
    ctx_cpu = context_lens.detach().to("cpu", dtype=torch.int64)
    first_blk_cpu = block_tables[:, 0].detach().to("cpu", dtype=torch.int64)
    batch = ctx_cpu.shape[0]

    num_pool = (ctx_cpu + k_block_size - 1) // k_block_size  # ceil, [B]
    start = torch.clamp(num_pool - incr_blocks, min=0)       # [B] int64

    fallback_full = False
    if entry.prev_first_blk is not None and entry.prev_first_blk.shape[0] == batch:
        blk_changed = first_blk_cpu != entry.prev_first_blk
        sl_back = ctx_cpu < entry.prev_seq_len
        invalid = blk_changed | sl_back
        # Invalid history before the update window requires full-batch pooling.
        if bool((invalid & (start > 0)).any()):
            fallback_full = True
    else:
        # An uninitialized row may have more blocks than one update can fill.
        if bool((num_pool > incr_blocks).any()):
            fallback_full = True

    # Save snapshots for invalidation on the next call.
    entry.prev_first_blk = first_blk_cpu.clone()
    entry.prev_seq_len = ctx_cpu.clone()

    start_block = start.to(device=device, dtype=torch.int32)
    return start_block, fallback_full


def should_use_cache(batch: int, max_pool_blocks: int) -> bool:
    """Check the cache switch and the configured work-estimate threshold."""
    if not CACHE_ENABLED:
        return False
    return batch * max_pool_blocks >= _MIN_WORK


def compute_start_block_pure(
    context_lens: torch.Tensor,  # [B] int32, including tokens added this step
    k_block_size: int,
    incr_blocks: int = INCR_BLOCKS,
) -> torch.Tensor:
    """Derive [B] int32 update starts on context_lens.device for CUDA graphs.

    Compute max(0, ceil(seq_len / k_block_size) - incr_blocks) using device
    operations only. No invalidation, fallback flag, or cache history is used.
    Callers must initialize blocks preceding the update window through full
    pooling before relying on incremental updates during graph replay.
    """
    device = context_lens.device
    num_pool = (context_lens.to(torch.int64) + k_block_size - 1) // k_block_size
    start = torch.clamp(num_pool - incr_blocks, min=0)
    return start.to(device=device, dtype=torch.int32)