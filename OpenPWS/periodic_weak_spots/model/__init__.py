"""Qwen3 with chunk-compressed KV attention (KVC)."""

from transformers import AutoConfig, AutoModelForCausalLM

from .config import Qwen3KVCConfig, compose_qwen3_kvc_config, load_research_model_config
from .modeling import Qwen3KVCForCausalLM

AutoConfig.register(Qwen3KVCConfig.model_type, Qwen3KVCConfig, exist_ok=True)
AutoModelForCausalLM.register(Qwen3KVCConfig, Qwen3KVCForCausalLM, exist_ok=True)

__all__ = [
    "Qwen3KVCConfig",
    "Qwen3KVCForCausalLM",
    "compose_qwen3_kvc_config",
    "load_research_model_config",
]
