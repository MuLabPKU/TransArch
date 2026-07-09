"""OpenDSA operators — FlashKL warmup/sparse KL losses + sparse MLA attention.

All ops have pure-torch, CPU/float64-verifiable reference paths (the default and
correctness baseline) plus optional TileLang GPU fast-paths on H100 (sm_90).
"""
from .flashkl_warmup import (
    flashkl_warmup_loss,
    dense_warmup_reference,
    indexer_topk_recall,
    prepare_ks_ke,
    prepare_ks_ke_cp,
)
from .topk_select import (
    indexer_scores,
    indexer_select_topk,
    flashkl_sparse_loss,
    sparse_kl_chunked,
)
from .sparse_mla import (
    sparse_mla_ref,
    sparse_mla_kernel,
    sparse_attend_ref,
    sparse_attend_chunked,
    sparse_attend_absorbed_chunked,
)

__all__ = [
    "flashkl_warmup_loss",
    "dense_warmup_reference",
    "indexer_topk_recall",
    "prepare_ks_ke",
    "prepare_ks_ke_cp",
    "indexer_scores",
    "indexer_select_topk",
    "flashkl_sparse_loss",
    "sparse_kl_chunked",
    "sparse_mla_ref",
    "sparse_mla_kernel",
    "sparse_attend_ref",
    "sparse_attend_chunked",
    "sparse_attend_absorbed_chunked",
]
