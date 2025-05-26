import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.utils import get_dataset, prepare_dataloader, prepare_test_dataloader, evaluate_ppl, get_qkv_calibrate_outputs, statistics_qkv_rmsnorm
from src.remove_rope import RemoveRope
from src.lora_qkv import LoraQKV
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="meta-llama/Llama-2-7b-hf", help="Model to load")
parser.add_argument("--save-path", type=str, default="outputs", help="output path.")
parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16", "bf16"], default="bf16")
parser.add_argument("--device", type=str, help="Device to use.", choices=["cpu", "cuda", "auto"], default="auto")
parser.add_argument("--cal-dataset", type=str, help="Dataset to calibrate and calculate perplexity on.", choices=["wikitext2", "ptb", "c4", "alpaca"], default="wikitext2")
parser.add_argument("--cal-nsamples", type=int, help="Number of samples of the calibration data to load.", default=128)
parser.add_argument("--cal-batch-size", type=int, default=8, help="Batch size for loading the calibration data.")
parser.add_argument("--cal-max-seqlen", type=int, default=256, help="Maximum sequence length for the calibration data.")
parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
parser.add_argument("--freqfold", type=int, default=8, help="")
parser.add_argument("--rope-head", type=int, default=1, help="")
parser.add_argument("--qk-mqa-dim", type=int, default=64, help="")
parser.add_argument("--collapse", type=int, default=2, help="")
parser.add_argument("--q-lora-rank", type=int, help="")
parser.add_argument("--kv-lora-rank", type=int, default=512, help="")
parser.add_argument("--balance-kv-ratio", type=float, default=1, help="")
parser.add_argument("--use-qkv-norm", action='store_true', default=False, help="")
args = parser.parse_args()

def main(args: argparse.Namespace) -> None:
    # get model, tokenizer and dataset
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16 if args.dtype == "bf16" else torch.float32,
        device_map=args.device,
        _attn_implementation="eager",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = get_dataset(args.cal_dataset)
    train_loader = prepare_dataloader(
        dataset=dataset["train"],
        tokenizer=tokenizer,
        max_seqlen=args.cal_max_seqlen,
        batch_size=args.cal_batch_size,
        nsamples=args.cal_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )
    if args.ppl_eval_batch_size > 0:
        test_loader = prepare_test_dataloader(
            dataset=dataset["test"], tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size
        )

    # evaluate original model
    print("-"*10+"Original Model:"+"-"*10)
    dataset_ppl = 0
    if args.ppl_eval_batch_size > 0:
        print("evaluate original model's ppl...")
        dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader)
        print(f'Original ppl: {dataset_ppl:.4f}')

    # remove rope
    print("-"*10+"RemoveRope Model:"+"-"*10)
    print("getting original model's qkv outputs...")
    ori_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader)
    for layer_idx, layer in enumerate(model.model.layers):
        setattr(layer, "self_attn", RemoveRope(
            layer.self_attn, 
            ori_qkv_outputs["key"][layer_idx], 
            freqfold=args.freqfold, 
            rope_head=args.rope_head,
            collapse=args.collapse,
        ))
    if args.ppl_eval_batch_size > 0:
        print("evaluate rope-removed model's ppl...")
        dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader)
        print(f'Remove RoPE ppl: {dataset_ppl:.4f}')

    # convert to deepseek-mla
    print("-"*10+"LoraQKV Model:"+"-"*10)
    print("getting rope-removed model's qkv outputs...")
    rm_rope_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader)
    for layer_idx, layer in enumerate(model.model.layers):
        assert args.rope_head == 1
        setattr(layer, "self_attn", LoraQKV(
            layer.self_attn, 
            rm_rope_qkv_outputs["query"][layer_idx], 
            rm_rope_qkv_outputs["key"][layer_idx], 
            rm_rope_qkv_outputs["value"][layer_idx], 
            q_lora_rank=args.q_lora_rank, 
            qk_mqa_dim=args.qk_mqa_dim, 
            collapse=args.collapse,
            kv_lora_rank=args.kv_lora_rank,
            use_qkv_norm=args.use_qkv_norm,
            balance_kv_ratio=args.balance_kv_ratio,
            rms_norm_eps=model.config.rms_norm_eps,
        ))
    
    if args.use_qkv_norm:
        lora_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader)
        for layer_idx, layer in enumerate(model.model.layers):
            statistics_qkv_rmsnorm(
                layer.self_attn, 
                lora_qkv_outputs["q_a_proj"][layer_idx] if len(lora_qkv_outputs["q_a_proj"])>layer_idx else None, 
                lora_qkv_outputs["kv_a_proj"][layer_idx]
            )

    if args.ppl_eval_batch_size > 0:
        print("evaluate lora-qkv model's ppl...")
        dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader)
        print(f'Low rank approximate QKV ppl: {dataset_ppl:.4f}')
    model.save_pretrained(os.path.join(args.save_path))
    tokenizer.save_pretrained(os.path.join(args.save_path))

    modify_config(model, os.path.join(args.save_path, "config.json"))


def modify_config(model, config_path: str) -> None:
    with open(config_path, "r") as f:
        config = json.load(f)
    
    config["attention_bias"] = model.model.layers[0].self_attn.q_proj.bias is not None
    config["mlp_bias"] = model.model.layers[0].mlp.gate_proj.bias is not None
    config["auto_map"] = {
        "AutoConfig": "configuration_deepseek_v3.DeepseekV3Config",
        "AutoModel": "modeling_deepseek_v3.DeepseekV3Model",
        "AutoModelForCausalLM": "modeling_deepseek_v3.DeepseekV3ForCausalLM"
    }
    config["architectures"] = ["DeepseekV3ForCausalLM"]
    config["model_type"] = "deepseek_v3"
    config["first_k_dense_replace"] = config["num_hidden_layers"]

    config["qk_rope_head_dim"] = config["head_dim"] = args.qk_mqa_dim
    config["qk_nope_head_dim"] = config["v_head_dim"] = model.model.layers[0].self_attn.head_dim
    config["q_lora_rank"] = args.q_lora_rank
    config["kv_lora_rank"] = args.kv_lora_rank
    config["pretraining_tp"] = 1
    config["rope_interleave"] = False

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)



    
if __name__ == "__main__":
    main(args)
