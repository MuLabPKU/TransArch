"""MLA -> GQLA conversion for GLM-4.7-Flash via per-group K/V PCA + absorption.

Single conversion method (neighbor grouping + per-group PCA fit on calibration,
absorbed into Q and O); multi-GPU via HF ``device_map="auto"`` for conversion
and vLLM tensor parallel for serving.
"""

from .compression import (
    GqlaLayout,
    collect_kv_grams,
    compress_and_absorb,
    evaluate_ppl,
    fit_per_group_pca,
    gather_calibration_hidden_states,
    inplace_apply_compressed,
    prepare_calibration_inputs,
    prepare_ppl_dataloader,
)
from .modeling import (
    Glm4MoeLiteGQLAAttention,
    Glm4MoeLiteGQLADecoderLayer,
    Glm4MoeLiteGQLAForCausalLM,
    Glm4MoeLiteGQLAModel,
)

__all__ = [
    "GqlaLayout",
    "Glm4MoeLiteGQLAAttention",
    "Glm4MoeLiteGQLADecoderLayer",
    "Glm4MoeLiteGQLAModel",
    "Glm4MoeLiteGQLAForCausalLM",
    "collect_kv_grams",
    "compress_and_absorb",
    "evaluate_ppl",
    "fit_per_group_pca",
    "gather_calibration_hidden_states",
    "inplace_apply_compressed",
    "prepare_calibration_inputs",
    "prepare_ppl_dataloader",
]
