"""OpenDSA operators — FlashKL warmup/sparse KL losses + sparse MLA attention.

All ops are pure-torch, memory-bounded (query-chunked + gradient-checkpointed)
training paths.
"""
from .flashkl_warmup import (
    flashkl_warmup_loss,
    auto_warmup_tile,
    dense_warmup_reference,
    indexer_topk_recall,
    prepare_ks_ke,
    prepare_ks_ke_cp,
)
from .topk_select import (
    indexer_select_topk,
    sparse_kl_chunked,
)
from .sparse_mla import (
    sparse_attend_absorbed_chunked,
)

__all__ = [
    "flashkl_warmup_loss",
    "auto_warmup_tile",
    "dense_warmup_reference",
    "indexer_topk_recall",
    "prepare_ks_ke",
    "prepare_ks_ke_cp",
    "indexer_select_topk",
    "sparse_kl_chunked",
    "sparse_attend_absorbed_chunked",
]
