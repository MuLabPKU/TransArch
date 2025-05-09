import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.utils import get_dataset, prepare_dataloader, prepare_test_dataloader, evaluate_ppl, get_qkv_calibrate_outputs, statistics_qkv_rmsnorm
from src.remove_rope import RemoveRope
from src.lora_qkv import LoraQKV

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="meta-llama/Llama-3.2-1B/", help="Model to load")
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
parser.add_argument("--dim2head", type=int, default=4, help="")
parser.add_argument("--rope-head", type=int, default=1, help="")
parser.add_argument("--qk-mqa-dim", type=int, default=64, help="")
parser.add_argument("--q-lora-rank", type=int, default=2048, help="")
parser.add_argument("--kv-lora-rank", type=int, default=960, help="")
parser.add_argument("--balance-kv-ratio", type=float, help="")
parser.add_argument("--use-qkv-norm", action='store_true', default=False, help="")
args = parser.parse_args()

def main(args: argparse.Namespace) -> None:
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

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if os.path.exists(os.path.join(args.save_path, "ori_qkv_outputs.pt")):
        print(f"load calculate feature from {args.save_path}")
        ori_qkv_outputs = torch.load(os.path.join(args.save_path, "ori_qkv_outputs.pt"), "cpu")
    else:
        print(f"generate calculate feature")
        ori_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader)
        torch.save(ori_qkv_outputs, os.path.join(args.save_path, "ori_qkv_outputs.pt"))

    print("+"*10+"Original Model:"+"+"*10)
    print(model)
    dataset_ppl = 0
    if args.ppl_eval_batch_size > 0:
        test_loader = prepare_test_dataloader(
            dataset=dataset["test"], tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size
        )
        dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader)
        print(f'Original ppl: {dataset_ppl:.4f}')

    print("+"*10+"RemoveRope Model:"+"+"*10)
    for layer_idx, layer in enumerate(model.model.layers):
        setattr(layer, "self_attn", RemoveRope(
            layer.self_attn, 
            ori_qkv_outputs["key"][layer_idx], 
            dim2head=args.dim2head, 
            rope_head=args.rope_head,
        ))
        
    if os.path.exists(os.path.join(args.save_path, "rm_rope_qkv_outputs.pt")):
        print(f"load calculate feature from {args.save_path}")
        rm_rope_qkv_outputs = torch.load(os.path.join(args.save_path, "rm_rope_qkv_outputs.pt"), "cpu")
    else:
        print(f"generate calculate feature")
        rm_rope_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader)
        torch.save(rm_rope_qkv_outputs, os.path.join(args.save_path, "rm_rope_qkv_outputs.pt"))

    print(model)
    if args.ppl_eval_batch_size > 0:
        dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader)
        print(f'Remove RoPE ppl: {dataset_ppl:.4f}')

    print("+"*10+"LoraQKV Model:"+"+"*10)
    for layer_idx, layer in enumerate(model.model.layers):
        assert args.rope_head == 1
        setattr(layer, "self_attn",LoraQKV(
            layer.self_attn, 
            rm_rope_qkv_outputs["query"][layer_idx], 
            rm_rope_qkv_outputs["key"][layer_idx], 
            rm_rope_qkv_outputs["value"][layer_idx], 
            qk_mqa_dim=args.qk_mqa_dim, 
            q_lora_rank=args.q_lora_rank, 
            kv_lora_rank=args.kv_lora_rank,
            use_qkv_norm=args.use_qkv_norm,
            balance_kv_ratio=args.balance_kv_ratio,
        ))
    
    if args.use_qkv_norm:
        if os.path.exists(os.path.join(args.save_path, "lora_qkv_outputs.pt")):
            print(f"load calculate feature from {args.save_path}")
            lora_qkv_outputs = torch.load(os.path.join(args.save_path, "lora_qkv_outputs.pt"), "cpu")
        else:
            print(f"generate calculate feature")
            lora_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader)
            torch.save(lora_qkv_outputs, os.path.join(args.save_path, "lora_qkv_outputs.pt"))

        for layer_idx, layer in enumerate(model.model.layers):
            statistics_qkv_rmsnorm(layer.self_attn, lora_qkv_outputs["q_a_proj"][layer_idx], lora_qkv_outputs["kv_a_proj"][layer_idx])
    print(model)

    if args.ppl_eval_batch_size > 0:
        dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader)
        print(f'Low rank approximate QKV ppl: {dataset_ppl:.4f}')
    model.save_pretrained(os.path.join(args.save_path))
    tokenizer.save_pretrained(os.path.join(args.save_path))
    
if __name__ == "__main__":
    main(args)
