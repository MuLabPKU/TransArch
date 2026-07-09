"""OpenDSA modeling — DSA patch for DeepSeek-V2 MLA + Lightning Indexer."""
from .indexer import LightningIndexer
from .patch_deepseek import (
    IndexerConfig,
    patch_model_with_dsa,
    set_dsa_mode,
    set_cu_seqlens,
    set_cp_size,
    set_log_recall,
    set_log_kl,
    indexer_parameters,
    freeze_backbone_train_indexer,
    unfreeze_all,
    dsa_loss_registry,
    patch_model_with_ep,
    expert_parameters,
    nonexpert_parameters,
)
from .dsa_attention import REGISTRY

__all__ = [
    "LightningIndexer",
    "IndexerConfig",
    "patch_model_with_dsa",
    "set_dsa_mode",
    "set_cu_seqlens",
    "set_cp_size",
    "set_log_recall",
    "set_log_kl",
    "indexer_parameters",
    "freeze_backbone_train_indexer",
    "unfreeze_all",
    "dsa_loss_registry",
    "patch_model_with_ep",
    "expert_parameters",
    "nonexpert_parameters",
    "REGISTRY",
]
