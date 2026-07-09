from vllm import ModelRegistry
# from .indexers import *

def register():
    """register MiSA to vLLM"""
    if "MiSAForCausalLM" not in ModelRegistry.get_supported_archs():
        # 延迟导入模型类（避免 CUDA 初始化问题）
        ModelRegistry.register_model(
            "MiSAForCausalLM",
            "misa_vllm_patch.misa_deepseek_v2:MiSAForCausalLM",
        )