"""MLA -> GQLA conversion for GLM-4.7-Flash via per-group K/V PCA + absorption.

Default path: neighbor grouping + per-group PCA fit on calibration, absorbed
into Q and O. Optional flags enable data-driven similarity grouping
(``--head_grouping similarity``) and Hessian-weighted PCA via per-token NLL
(``--hessian_pca --hessian_mode nll``). Multi-GPU via HF ``device_map="auto"``
for conversion and vLLM tensor parallel for serving.
"""

from .compression import (
    GqlaLayout,
    assemble_per_group_covs_from_full,
    collect_full_kv_grams,
    collect_kv_grams,
    compose_compressed_with_perm,
    compress_and_absorb,
    compute_head_similarity,
    compute_token_weights_nll,
    diagnose_weights,
    evaluate_ppl,
    fit_per_group_pca,
    gather_calibration_hidden_states,
    greedy_balanced_grouping,
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
    "assemble_per_group_covs_from_full",
    "collect_full_kv_grams",
    "collect_kv_grams",
    "compose_compressed_with_perm",
    "compress_and_absorb",
    "compute_head_similarity",
    "compute_token_weights_nll",
    "diagnose_weights",
    "evaluate_ppl",
    "fit_per_group_pca",
    "gather_calibration_hidden_states",
    "greedy_balanced_grouping",
    "inplace_apply_compressed",
    "prepare_calibration_inputs",
    "prepare_ppl_dataloader",
]
