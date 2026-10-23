from vllm import ModelRegistry


def register():
    """Register the architecture selected by the accompanying config overrides."""
    if "ResMISAForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "ResMISAForCausalLM",
            "resmisa_vllm_patch.resmisa_deepseek_v2:ResMISAForCausalLM",
        )
