import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils import get_dataset, prepare_dataloader, prepare_test_dataloader, evaluate_ppl, get_qkv_calibrate_outputs, statistics_qkv_rmsnorm
from remove_rope import RemoveRope
from lora_qkv import LoraQKV
import json
from typing import Union, Literal
from copy import deepcopy

from config_modifiers import ALL_CONFIG_MODIFIERS

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="meta-llama/Llama-2-7b-hf", help="Model to load")
parser.add_argument("--save-path", type=str, default="outputs", help="output path.")
parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16", "bf16"], default="bf16")
parser.add_argument("--device", type=str, help="Device to use.", choices=["cpu", "cuda", "auto"], default="auto")
parser.add_argument("--cal-dataset", type=str, help="Dataset to calibrate and calculate perplexity on.", choices=["wikitext2", "ptb", "c4", "alpaca"], default="wikitext2")
parser.add_argument("--cal-nsamples", type=int, help="Number of samples of the calibration data to load.", default=128)
parser.add_argument("--cal-batch-size", type=int, default=8, help="Batch size for loading the calibration data.")
parser.add_argument("--cal-max-seqlen", type=int, default=256, help="Maximum sequence length for the calibration data.")
parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
parser.add_argument("--ppl-eval-batch-size", type=int, default=0, help="Batch size for evaluating the perplexity.")
parser.add_argument("--freqfold", type=str, default="auto", help="Freqfold for removing RoPE, int or auto")
parser.add_argument("--collapse", type=str, default="auto", help="Collapse for removing RoPE, int or auto")
parser.add_argument("--qk-mqa-dim", type=int, default=64, help="")
parser.add_argument("--q-lora-rank", type=int, help="")
parser.add_argument("--kv-lora-rank", type=int, default=512, help="")
parser.add_argument("--balance-kv-ratio", type=float, default=1, help="")
parser.add_argument("--use-qkv-norm", action='store_true', default=False, help="")
args = parser.parse_args()


def load_model_and_tokenizer(model_path: str):
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

    return model, tokenizer


def get_dataset_loader(args: argparse.Namespace, tokenizer: AutoTokenizer):
    dataset = get_dataset(args.cal_dataset)
    train_loader = prepare_dataloader(
        dataset=dataset["train"],
        tokenizer=tokenizer,
        max_seqlen=args.cal_max_seqlen,
        batch_size=args.cal_batch_size,
        nsamples=args.cal_nsamples,
        seed=args.seed,
    )
    if args.ppl_eval_batch_size > 0:
        test_loader = prepare_test_dataloader(
            dataset=dataset["test"], tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size
        )
    else:
        test_loader = None
    
    return train_loader, test_loader


def remove_rope(model, tokenizer, train_loader, test_loader, freqfold, collapse):

    print("Getting original model's qkv outputs...")
    ori_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader)

    def remove_rope_freqfold(model, ori_qkv_outputs, test_loader, freqfold: int, collapse):
        for layer_idx, layer in enumerate(model.model.layers):
            setattr(layer, "self_attn", RemoveRope(
                layer.self_attn, 
                ori_qkv_outputs["key"][layer_idx], 
                freqfold=freqfold,
                collapse=collapse,
            ))
            
        if test_loader:
            print(f"Evaluating rope-removed model's ppl, freqfold={freqfold}...")
            dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader)
            print(f'Remove RoPE ppl, freqfold={freqfold}: {dataset_ppl:.4f}')
            return model, dataset_ppl
        else:
            return model, None

    if type(freqfold) == int:
        return remove_rope_freqfold(model, ori_qkv_outputs, test_loader, freqfold, collapse)[0]
    else:
        assert test_loader is not None, "test_loader is required for auto freqfold detection"
        device = model.device
        model_original = model.to("cpu")

        print(f"Auto freqfold detection...")

        best_freqfold = freqfold = collapse
        best_ppl = float("inf")
        while freqfold <= model_original.config.head_dim // 2:
            model = deepcopy(model_original)
            model = model.to(device)
            model, ppl = remove_rope_freqfold(model, ori_qkv_outputs, test_loader, freqfold, collapse)
            if ppl < best_ppl:
                best_ppl = ppl
                best_freqfold = freqfold
                freqfold *= 2
            else:
                break

        model = deepcopy(model_original)
        model = model.to(device)
        model, _ = remove_rope_freqfold(model, ori_qkv_outputs, None, best_freqfold, collapse)

        return model, best_freqfold


def low_rank_qkv(model, tokenizer, train_loader, test_loader, args):

    print("Getting rope-removed model's qkv outputs...")
    rm_rope_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader)

    for layer_idx, layer in enumerate(model.model.layers):
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
                lora_qkv_outputs["q_a_proj"][layer_idx] if len(lora_qkv_outputs["q_a_proj"]) > layer_idx else None, 
                lora_qkv_outputs["kv_a_proj"][layer_idx]
            )

    if test_loader:
        print("Evaluating lora-qkv model's ppl...")
        dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader)
        print(f'Low rank approximate QKV ppl: {dataset_ppl:.4f}')
    
    return model

    
def main(args: argparse.Namespace) -> None:
    ##############################
    #       original model       #
    ##############################
    print("\n" + "="*60)
    print("Original Model".center(60))
    print("="*60 + "\n")

    # get model, tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    model_type = model.config.model_type

    # get dataset
    train_loader, test_loader = get_dataset_loader(args, tokenizer)

    if test_loader:
        print("Evaluating original model's ppl...")
        # dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader)
        # print(f'Original ppl: {dataset_ppl:.4f}')

    ##############################
    #        remove rope         #
    ##############################
    print("\n" + "="*60)
    print("Remove RoPE Model".center(60))
    print("="*60 + "\n")

    if args.collapse == "auto":
        args.collapse = model.config.head_dim // args.qk_mqa_dim
        print(f"Auto collapse: {args.collapse} (head_dim={model.config.head_dim} / qk_mqa_dim={args.qk_mqa_dim})")
    else:
        args.collapse = int(args.collapse)

    model = remove_rope(model, tokenizer, train_loader, test_loader, args.freqfold, args.collapse)
    if args.freqfold == "auto":
        args.freqfold = model[1]
        model = model[0]
        print(f"Best freqfold: {args.freqfold}")

    ##############################
    #     deepseek-mla model     #
    ##############################
    print("\n" + "="*60)
    print("LoraQKV Model".center(60))
    print("="*60 + "\n")
    model = low_rank_qkv(model, tokenizer, train_loader, test_loader, args)

    # save model
    model.save_pretrained(os.path.join(args.save_path))
    tokenizer.save_pretrained(os.path.join(args.save_path))

    # modify config
    config_modifier = ALL_CONFIG_MODIFIERS[model_type]
    config_modifier(model, os.path.join(args.save_path, "config.json"), args)

    
if __name__ == "__main__":
    main(args)
