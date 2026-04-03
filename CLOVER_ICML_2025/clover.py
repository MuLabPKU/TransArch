import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="deepseek-ai/DeepSeek-V2-Lite", help="Model to load")
parser.add_argument("--save-path", type=str, default="deepseek-ai/DeepSeek-V2-Lite", help="Model to load")
parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16", "bf16"], default="bf16")
parser.add_argument("--device", type=str, help="Device to use.", choices=["cpu", "cuda", "auto"], default="auto")
parser.add_argument("--cal-dataset", type=str, help="Dataset to calibrate and calculate perplexity on.", choices=["wikitext2", "ptb", "c4", "alpaca"], default="wikitext2")
parser.add_argument("--pruned-dim", type=int, help="Data type to use.", default=64)
parser.add_argument("--ppl-eval-batch-size", type=int, default=1, help="Batch size for evaluating the perplexity.")
args = parser.parse_args()

def main(args: argparse.Namespace) -> None:
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype = torch.bfloat16, trust_remote_code=True, device_map=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    hidden_size=model.config.hidden_size
    kv_lora_rank=model.config.kv_lora_rank
    num_attention_heads=model.config.num_attention_heads
    qk_nope_head_dim=model.config.qk_nope_head_dim
    qk_rope_head_dim=model.config.qk_rope_head_dim
    v_head_dim=model.config.v_head_dim
    pruned_dim = args.pruned_dim

    for layer in tqdm(range(model.config.num_hidden_layers)):
        self_attn = model.model.layers[layer].self_attn
        q_weight = self_attn.q_proj.weight.data.view(num_attention_heads, qk_nope_head_dim+qk_rope_head_dim, hidden_size)
        q_weight_pass, q_weight_rot = torch.split(q_weight, [qk_nope_head_dim, qk_rope_head_dim], dim=1)
        kv_b_weight = self_attn.kv_b_proj.weight.data.view(num_attention_heads, qk_nope_head_dim+v_head_dim, kv_lora_rank)
        k_b_weight, v_b_weight = torch.split(kv_b_weight, [qk_nope_head_dim, v_head_dim], dim=1)
        o_weight = self_attn.o_proj.weight.data.view(hidden_size, num_attention_heads, v_head_dim)
        Uqk,Sqk,Vqk = torch.svd_lowrank(torch.bmm(q_weight_pass.transpose(1,2), k_b_weight).to(torch.float32), q=pruned_dim, niter=10)
        q_weight_pass = (Uqk*Sqk[:,None]).transpose(1,2)
        k_b_weight = Vqk.transpose(1,2)
        Uvo,Svo,Vvo = torch.svd_lowrank(torch.einsum("hdl,Dhd->hlD",v_b_weight, o_weight).to(torch.float32), q=pruned_dim, niter=10)
        v_b_weight = (Uvo*Svo[:,None]).transpose(1,2)
        o_weight = Vvo.transpose(0,1)
        q_weight = torch.cat([q_weight_pass, q_weight_rot],dim=1).reshape(num_attention_heads*(pruned_dim+qk_rope_head_dim),hidden_size)
        kv_b_weight = torch.cat([k_b_weight, v_b_weight],dim=1).reshape(num_attention_heads*(pruned_dim+pruned_dim),kv_lora_rank)
        o_weight = o_weight.reshape(hidden_size, num_attention_heads*pruned_dim)

        self_attn.q_proj.weight.data = q_weight.contiguous()
        self_attn.kv_b_proj.weight.data = kv_b_weight.contiguous()
        self_attn.o_proj.weight.data = o_weight.contiguous()

    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

if __name__ == "__main__":
    main(args)