import json

def modify_llama_config(model, config_path: str, args) -> None:
    with open(config_path, "r") as f:
        config = json.load(f)
        
    config["auto_map"] = {
        "AutoConfig": "configuration_llamamla.DeepseekV3Config",
        "AutoModel": "modeling_llamamla.DeepseekV3Model",
        "AutoModelForCausalLM": "modeling_llamamla.DeepseekV3ForCausalLM"
    }
    config["architectures"] = ["LlamaMLAForCausalLM"]
    config["model_type"] = "llama_mla"
    
    config["num_key_value_heads"] = config["num_attention_heads"]
    config["attention_bias"] = model.model.layers[0].self_attn.q_proj.bias is not None
    config["mlp_bias"] = model.model.layers[0].mlp.gate_proj.bias is not None
    config["first_k_dense_replace"] = config["num_hidden_layers"]

    config["qk_rope_head_dim"] = config["head_dim"] = args.qk_mqa_dim
    config["qk_nope_head_dim"] = config["v_head_dim"] = config["query_pre_attn_scalar"] = model.model.layers[0].self_attn.head_dim
    config["q_lora_rank"] = args.q_lora_rank
    config["kv_lora_rank"] = args.kv_lora_rank
    config["pretraining_tp"] = 1
    config["rope_interleave"] = False

    config["tie_word_embeddings"] = model.config.tie_word_embeddings

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def modify_gemma2_config(model, config_path: str, args) -> None:
    with open(config_path, "r") as f:
        config = json.load(f)
        
    config["auto_map"] = {
        "AutoConfig": "configuration_gemma2mla.DeepseekV3Config",
        "AutoModel": "modeling_gemma2mla.DeepseekV3Model",
        "AutoModelForCausalLM": "modeling_gemma2mla.DeepseekV3ForCausalLM"
    }
    config["architectures"] = ["Gemma2MLAForCausalLM"]
    config["model_type"] = "gemma2_mla"
    
    config["num_key_value_heads"] = config["num_attention_heads"]
    config["attention_bias"] = model.model.layers[0].self_attn.q_proj.bias is not None
    config["mlp_bias"] = model.model.layers[0].mlp.gate_proj.bias is not None
    config["first_k_dense_replace"] = config["num_hidden_layers"]

    config["qk_rope_head_dim"] = config["head_dim"] = args.qk_mqa_dim
    config["qk_nope_head_dim"] = config["v_head_dim"] = model.model.layers[0].self_attn.head_dim
    config["q_lora_rank"] = args.q_lora_rank
    config["kv_lora_rank"] = args.kv_lora_rank
    config["pretraining_tp"] = 1
    config["rope_interleave"] = False

    config["tie_word_embeddings"] = model.config.tie_word_embeddings

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)



ALL_CONFIG_MODIFIERS = {
    "llama": modify_llama_config,
    "qwen2": modify_llama_config,
    "gemma2": modify_gemma2_config,
}

__all__ = ["ALL_CONFIG_MODIFIERS"]