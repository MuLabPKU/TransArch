"""Shared native token-score implementation for both experimental indexers.

The two-stage reference computes dense scores in bounded query chunks, then gathers
only stage-one candidates before top-k. This performs extra arithmetic but gives
both arms exactly the same scoring kernel and quantization, avoiding a kernel
change as a confound. It is not a performance-optimized candidate-only kernel.
"""

import torch
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits


def dense_scores(q, k, ks, weights, positions):
    pos = positions.to(torch.int32)
    return fp8_fp4_mqa_logits(
        (q, None), (k, ks), weights, torch.zeros_like(pos), pos + 1, clean_logits=True
    )


def candidate_scores(q, k, ks, weights, ids, positions):
    scores = dense_scores(q, k, ks, weights, positions)
    gathered = torch.gather(scores, 1, ids.long().clamp(min=0, max=k.shape[0] - 1))
    valid = (ids >= 0) & (ids < k.shape[0]) & (ids <= positions[:, None])
    return gathered.masked_fill(~valid, -torch.inf)


def select_candidates(scores, ids, budget=2048):
    values, indices = torch.topk(scores, min(budget, scores.shape[1]), dim=-1, sorted=False)
    selected = torch.gather(ids, 1, indices)
    return torch.where(torch.isfinite(values), selected, -1).to(torch.int32)
