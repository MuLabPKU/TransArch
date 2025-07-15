import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.data import get_dataset, prepare_test_dataloader, prepare_dataloader
from src.fuse_rmsnorm import insert_shortcut_and_fuse_rmsnorm
from src.pca_calc import get_calibrate_outputs, evaluate_ppl, model_pca_calc
from src.rotate import model_rotate
from src.slice import model_slice

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="deepseek-ai/DeepSeek-V2-Lite", help="Model to load")
parser.add_argument("--save-path", type=str, default="outputs", help="output path.")
parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16", "bf16"], default="bf16")
parser.add_argument("--device", type=str, help="Device to use.", choices=["cpu", "cuda", "auto"], default="auto")
parser.add_argument("--cal-dataset", type=str, help="Dataset to calibrate and calculate perplexity on.", choices=["wikitext2", "ptb", "c4", "alpaca"], default="wikitext2")
parser.add_argument("--cal-nsamples", type=int, help="Number of samples of the calibration data to load.", default=128)
parser.add_argument("--cal-batch-size", type=int, default=8, help="Batch size for loading the calibration data.")
parser.add_argument("--cal-max-seqlen", type=int, default=256, help="Maximum sequence length for the calibration data.")
parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
parser.add_argument("--pruned-dim", type=int, help="Data type to use.", default=2048)
parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
args = parser.parse_args()

def main(args: argparse.Namespace) -> None:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16 if args.dtype == "bf16" else torch.float32,
        device_map=args.device,
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

    insert_shortcut_and_fuse_rmsnorm(model)

    print("+"*10+"insert_shortcut_and_fuse_rmsnorm Model:"+"+"*10)
    print(model)
    if args.ppl_eval_batch_size > 0:
        test_loader = prepare_test_dataloader(
            dataset=dataset["test"], tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size
        )
        dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader)
        print(f'insert_shortcut_and_fuse_rmsnorm ppl: {dataset_ppl:.4f}')
    
    print(f"generate calculate feature")
    ori_outputs = get_calibrate_outputs(model, train_loader)

    emb_Q, attn_Q, mlp_Q = model_pca_calc(model, ori_outputs, model.model.embed_tokens.weight.device)
    model_rotate(model, torch.float64, emb_Q, attn_Q, mlp_Q)
    
    print("+"*10+"model_rotate Model:"+"+"*10)
    print(model)
    if args.ppl_eval_batch_size > 0:
        dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader)
        print(f'model_rotate ppl: {dataset_ppl:.4f}')
    
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    model_slice(model, args.pruned_dim)
    if args.ppl_eval_batch_size > 0:
        dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader)
        print(f'model_slice dim={args.pruned_dim} ppl: {dataset_ppl:.4f}')

if __name__ == "__main__":
    main(args)
