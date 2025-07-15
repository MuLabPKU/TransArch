import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.data import get_dataset, prepare_test_dataloader
from src.pca_calc import evaluate_ppl
from src.slice import model_slice

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="deepseek-ai/DeepSeek-V2-Lite", help="Model to load")
parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16", "bf16"], default="bf16")
parser.add_argument("--device", type=str, help="Device to use.", choices=["cpu", "cuda", "auto"], default="auto")
parser.add_argument("--cal-dataset", type=str, help="Dataset to calibrate and calculate perplexity on.", choices=["wikitext2", "ptb", "c4", "alpaca"], default="wikitext2")
parser.add_argument("--pruned-dim", type=int, help="Data type to use.")
parser.add_argument("--ppl-eval-batch-size", type=int, default=1, help="Batch size for evaluating the perplexity.")
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
    dataset_ppl = 0
    test_loader = prepare_test_dataloader(
        dataset=dataset["test"], tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size
    )
    if args.pruned_dim is not None:
        model_slice(model, args.pruned_dim)
        model.save_pretrained(f"outputs/slice_{args.pruned_dim}")
        tokenizer.save_pretrained(f"outputs/slice_{args.pruned_dim}")
    print(model)
    dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader)
    print(f'Original ppl: {dataset_ppl:.4f}')

if __name__ == "__main__":
    main(args)