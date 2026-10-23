"""Qwen3-KVC configuration and the loader for ``configs/model`` JSON files."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from transformers import Qwen3Config

# Fields allowed in a model JSON's ``kvc`` object; all other fields belong to ``backbone``.
KVC_CONFIG_KEYS = frozenset(
    {
        "attention_backend",
        "c_bias",
        "c_group_ids",
        "c_init_std",
        "c_trainable",
        "compression_pooling",
        "compression_stride",
        "compression_window",
        "gate_mode",
        "kv_sharing",
        "local_attention_mode",
        "num_key_value_heads",
        "post_projection_entry_normalization",
        "qk_norm",
        "rope_dims",
        "swa_window_size",
        "z_bias",
        "z_group_ids",
        "z_init_std",
        "z_trainable",
    }
)
# The subset that is meaningful without a compressor (``local_attention_mode: full``).
FULL_ATTENTION_KVC_KEYS = frozenset(
    {
        "attention_backend",
        "kv_sharing",
        "local_attention_mode",
        "num_key_value_heads",
        "qk_norm",
        "rope_dims",
    }
)
# d_r used when a ``kvc`` object omits ``rope_dims``; every shipped config sets it.
DEFAULT_RESEARCH_ROPE_DIMS = 16


class Qwen3KVCConfig(Qwen3Config):
    """Qwen3 configuration extended only with global KVC geometry."""

    model_type = "qwen3_kvc"

    num_key_value_heads: int = 32
    use_cache: bool = False
    # W and S: entry i pools the W tokens ending at (i + 1) * S - 1.
    compression_window: int = 1
    compression_stride: int = 1
    # Weight bank of each window offset for C^B_i and Z^B_i; equal IDs tie offsets.
    c_group_ids: tuple[int, ...] = (0,)
    z_group_ids: tuple[int, ...] = (0,)
    # scalar: one gate per KV head and offset; vector: one gate per channel.
    gate_mode: str = "scalar"
    compression_pooling: str = "mean"
    c_bias: bool = False
    z_bias: bool = False
    c_trainable: bool = True
    z_trainable: bool = False
    c_init_std: float = 0.02
    z_init_std: float = 0.0
    kv_sharing: bool = False
    local_attention_mode: str = "tail_concat"
    swa_window_size: int | None = None
    attention_backend: str = "eager"
    rope_dims: int = DEFAULT_RESEARCH_ROPE_DIMS
    # Unit-RMS each projected payload and scale it by a learned scalar before pooling.
    post_projection_entry_normalization: bool = False
    qk_norm: bool = True

    def __post_init__(self, **kwargs: Any) -> None:
        self.c_group_ids = tuple(self.c_group_ids)
        self.z_group_ids = tuple(self.z_group_ids)
        super().__post_init__(**kwargs)

        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.rope_dims < 0 or self.rope_dims % 2 or self.rope_dims > self.head_dim:
            raise ValueError("rope_dims must be an even integer in [0, head_dim]")
        if self.compression_window < self.compression_stride:
            raise ValueError("compression_window must be at least compression_stride")
        if self.gate_mode not in {"scalar", "vector"}:
            raise ValueError("gate_mode must be 'scalar' or 'vector'")
        if self.compression_pooling not in {"adaptive_softmax", "mean"}:
            raise ValueError("compression_pooling must be adaptive_softmax or mean")
        if self.local_attention_mode not in {"tail_concat", "fixed_swa", "full"}:
            raise ValueError("local_attention_mode must be 'tail_concat', 'fixed_swa', or 'full'")
        if self.local_attention_mode == "full" and (
            self.kv_sharing or self.post_projection_entry_normalization
        ):
            raise ValueError("full attention has no compressor options")
        if self.attention_backend not in {"eager", "flex_attention"}:
            raise ValueError("attention_backend must be 'eager' or 'flex_attention'")
        if self.attention_backend == "flex_attention" and self.attention_dropout != 0:
            raise ValueError("flex_attention requires attention_dropout=0")
        if (self.local_attention_mode == "fixed_swa") != (self.swa_window_size is not None):
            raise ValueError("swa_window_size is required by, and only valid with, fixed_swa")
        for name in ("c_group_ids", "z_group_ids"):
            ids = getattr(self, name)
            if len(ids) != self.compression_window or set(ids) != set(range(len(set(ids)))):
                raise ValueError(f"{name} must be compression_window contiguous IDs from zero")
        if self.compression_pooling == "mean" and (
            self.z_bias or self.z_trainable or self.z_init_std != 0
        ):
            raise ValueError(
                "mean pooling requires z_bias=false, z_trainable=false, and z_init_std=0"
            )
        if self.use_sliding_window or any(t != "full_attention" for t in self.layer_types):
            raise ValueError(
                "KVC uses full-attention layers; configure its local path with "
                "local_attention_mode instead of Qwen sliding-window layers"
            )

    @property
    def query_groups_per_kv_head(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


def compose_qwen3_kvc_config(backbone: Mapping[str, Any], kvc: Mapping[str, Any]) -> Qwen3KVCConfig:
    """Merge isolated backbone and KVC objects into one HF config."""

    unknown = sorted(set(kvc) - KVC_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown kvc fields: {', '.join(unknown)}")
    leaked = sorted(set(backbone) & KVC_CONFIG_KEYS)
    if leaked:
        raise ValueError(f"backbone contains kvc fields: {', '.join(leaked)}")
    if "num_key_value_heads" not in kvc:
        raise ValueError("kvc must set num_key_value_heads")
    kvc_fields = {"rope_dims": DEFAULT_RESEARCH_ROPE_DIMS, **kvc}
    if kvc_fields.get("local_attention_mode") == "full":
        extra = sorted(set(kvc_fields) - FULL_ATTENTION_KVC_KEYS)
        if extra:
            raise ValueError("full attention forbids compressor fields: " + ", ".join(extra))
    return Qwen3KVCConfig.from_dict({**backbone, **kvc_fields})


def load_research_model_config(path: str | Path) -> Qwen3KVCConfig:
    """Load a model JSON made of exactly ``backbone`` and ``kvc`` objects."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"backbone", "kvc"}:
        raise ValueError(f"{path} must contain exactly backbone and kvc")
    return compose_qwen3_kvc_config(value["backbone"], value["kvc"])
