"""Normal SM100 prefill entry: minmax heads, fused Q, HOT and sorted token IDs.

This keeps the received q-scale absorption, BF16 gate conversion and fused-Q
construction. It calls September FP32 minmax and Torch-stable head ordering
directly. Decode belongs to the common indexer and is not changed here.
"""

import torch

from .hopper.head_selection import maxmin_head_scores, deterministic_heads
from .maxmin_head_score import build_fused_head_q_exact
from .blackwell import state
from .blackwell.hot import select_prepared
from .blackwell.normalize import normalize_ids


def select_prefill(q, kv, weights, q_head_topk_indices, cu_seqlen_ks,
                   cu_seqlen_ke, use_qnorm, topk_heads, *, k_block_size=1024,
                   resmisa_fuse=True, resmisa_minmax=True, topk_tokens=2048,
                   layer=None, first_chunk=None):
    k_fp8, k_scales = kv
    m, n = q.shape[0], k_fp8.shape[0]
    if not (q.ndim == 3 and q.shape[1:] == (64, 128)
            and k_fp8.shape == (n, 128) and weights.shape == (m, 64)
            and 0 < m <= 8192 and 0 < n <= 135168
            and q.dtype == k_fp8.dtype == torch.float8_e4m3fn
            and topk_tokens == 2048 and topk_heads == 7 and k_block_size == 1024
            and resmisa_fuse and resmisa_minmax and not use_qnorm):
        raise ValueError("Blackwell prefill requires FP8 H64/D128, 1+7 minmax, K2048, Q<=8192, context<=135168")
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) != (10, 0):
        raise RuntimeError("Blackwell prefill requires an SM100 device")
    if any(t.device != q.device for t in (k_fp8, k_scales, weights, cu_seqlen_ks, cu_seqlen_ke)):
        raise ValueError("Blackwell operands must share one device")
    if cu_seqlen_ks.shape != (m,) or cu_seqlen_ke.shape != (m,):
        raise ValueError("Blackwell causal bounds must have one entry per query")
    if first_chunk is None:
        first_chunk = n == m
    with state.chunk(layer, m, n, q.device, first_chunk):
        if not k_scales.is_contiguous():
            k_scales = k_scales.contiguous()
        k_scales = k_scales.view(torch.float32)
        if k_scales.ndim == 2:
            k_scales = k_scales.squeeze(-1)
        if k_scales.shape != (n,):
            raise ValueError("Blackwell K scale storage must contain one FP32 scale per key")
        weights = weights.bfloat16()
        starts = cu_seqlen_ks if cu_seqlen_ks.dtype == torch.int32 else cu_seqlen_ks.to(torch.int32)
        ends = cu_seqlen_ke if cu_seqlen_ke.dtype == torch.int32 else cu_seqlen_ke.to(torch.int32)
        head_scores = maxmin_head_scores(q, k_fp8, k_scales, weights, starts, ends,
                                         k_block_size)
        heads = deterministic_heads(head_scores, topk_heads)
        q_fused, q_scale = build_fused_head_q_exact(q, weights)
        expanded = heads.unsqueeze(-1).expand(-1, -1, q.size(-1))
        sparse_q = torch.gather(q, dim=1, index=expanded)
        sparse_weights = torch.gather(weights, dim=1, index=heads)
        ids = select_prepared(q_fused, q_scale, sparse_q, sparse_weights,
                              k_fp8, k_scales, starts, ends, layer=layer)
        # Exactly one ordering normalization after physical-to-logical ID map;
        # no external controller is needed before IDs enter main attention.
        out = normalize_ids(ids)
        return out


def validate_config(vllm_config):
    """Reject serving modes without this release's proven request lifecycle."""
    model = vllm_config.model_config
    scheduler = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    required = {
        "enforce_eager=True": bool(model.enforce_eager),
        "max_model_len<=135168": int(model.max_model_len) <= 135168,
        "max_num_seqs=1": scheduler.max_num_seqs == 1,
        "1<=max_num_batched_tokens<=8192": 1 <= scheduler.max_num_batched_tokens <= 8192,
        "enable_chunked_prefill=True": bool(scheduler.enable_chunked_prefill),
        "async_scheduling=False": scheduler.async_scheduling is False,
        "enable_prefix_caching=False": not cache.enable_prefix_caching,
        "no speculative decoding": vllm_config.speculative_config is None,
        "pipeline_parallel_size=1": parallel.pipeline_parallel_size == 1,
        "data_parallel_size=1": parallel.data_parallel_size == 1,
        "decode_context_parallel_size=1": getattr(parallel, "decode_context_parallel_size", 1) == 1,
        "prefill_context_parallel_size=1": getattr(parallel, "prefill_context_parallel_size", 1) == 1,
        "compilation mode=0": vllm_config.compilation_config.mode in (None, 0),
    }
    hf = model.hf_config
    required.update({
        "FP8 indexer H64/D128/Top2048": (hf.index_n_heads, hf.index_head_dim, hf.index_topk) == (64, 128, 2048),
        "fused minmax 1+7/pool1024": (
            getattr(hf, "resmisa_fuse", False) and getattr(hf, "resmisa_minmax", False)
            and hf.resmisa_topk_heads == 8 and hf.resmisa_k_block_size == 1024
            and not getattr(hf, "resmisa_use_qnorm", False)),
    })
    rejected = [name for name, ok in required.items() if not ok]
    if rejected:
        raise ValueError("Blackwell HOT requires " + "; ".join(rejected))


