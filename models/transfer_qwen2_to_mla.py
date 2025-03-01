from qwen2.modeling_qwen2 import Qwen2MLAForCausalLM
from transformers import AutoTokenizer
import torch
from copy import deepcopy
from tqdm import tqdm
import argparse


def init_rope_repeat(model, hidden_size, n_heads, kv_heads, kv_dim, ori_kv_heads, ori_head_dim, latent_dim):
    # Insert identity matrice
    for name,module in model.named_modules():
        if 'k_up_proj' in name or "v_up_proj" in name:
            module.weight.data = torch.cat([torch.stack([torch.eye(kv_dim).reshape(-1, ori_head_dim, kv_dim)]*(n_heads//ori_kv_heads),dim=1)]*kv_heads).reshape(hidden_size, kv_dim).contiguous().to(module.weight.data.device,module.weight.data.dtype)

def init_rope_extend(model, hidden_size, n_heads, kv_heads, kv_dim, ori_kv_heads, ori_head_dim, latent_dim):
    # Insert identity matrice and reorder
    for name, module in model.named_modules():
        if 'k_up_proj' in name or "v_up_proj" in name:
            weight = torch.cat([torch.stack([torch.eye(kv_dim).reshape(-1, ori_head_dim, kv_dim)]*(n_heads//ori_kv_heads),dim=1)]*kv_heads).reshape(hidden_size, kv_dim).contiguous().to(module.weight.data.device,module.weight.data.dtype)
            if 'k_up_proj' in name:
                weight = weight.view(hidden_size, -1, ori_head_dim).transpose(1,2).reshape(hidden_size, kv_dim).contiguous()
            module.weight.data=weight
        elif 'k_proj' in name:
            module.weight.data = module.weight.data.view(kv_heads, -1, ori_head_dim, hidden_size).transpose(1,2).reshape(latent_dim, hidden_size).contiguous()
            module.bias.data = module.bias.data.view(kv_heads, -1, ori_head_dim).transpose(1,2).reshape(latent_dim).contiguous()

def absorb(model, hidden_size, n_heads, kv_heads, kv_dim, ori_kv_heads, ori_head_dim, latent_dim):
    for name,module in tqdm(model.named_modules()):
        if name.endswith("self_attn"):
            # Absorb k_up_proj into q_proj
            k_up_weight = deepcopy(module.k_up_proj.weight.data).reshape(n_heads, ori_head_dim, kv_dim) # (n_heads, head_dim, latent_dim)
            q_weight = deepcopy(module.q_proj.weight.data).reshape(n_heads, ori_head_dim, hidden_size) # (n_heads, head_dim, hidden_size)
            if module.q_proj.bias is not None:
                q_weight = torch.cat([q_weight,deepcopy(module.q_proj.bias.data).reshape(n_heads, ori_head_dim, 1)],dim=-1)
            q_k_up = torch.einsum("hdc,hdD->hcD",k_up_weight, q_weight) # (n_heads, latent_dim, hidden_size), rank<=head_dim
            q_proj = torch.nn.Linear(hidden_size, n_heads*kv_dim, bias=(module.q_proj.bias is not None))
            q_proj = q_proj.to(device=module.q_proj.weight.device, dtype=module.q_proj.weight.dtype)
            if module.q_proj.bias is not None:
                q_proj.bias.data = q_k_up[:,:,-1].reshape(-1).contiguous()
                q_k_up = q_k_up[:,:,:-1]
            q_proj.weight.data = q_k_up.reshape(n_heads*kv_dim, hidden_size).contiguous()
            setattr(module, "q_proj", q_proj)
            delattr(module, "k_up_proj")
            
            # Absorb v_up_proj into o_proj
            v_up_weight = deepcopy(module.v_up_proj.weight.data).reshape(n_heads, ori_head_dim, kv_dim) # (n_heads, head_dim, latent_dim)
            o_weight = deepcopy(module.o_proj.weight.data).reshape(hidden_size, n_heads, ori_head_dim) # (n_heads, head_dim, hidden_size)
            v_up_o = torch.einsum("hdc,Dhd->Dhc",v_up_weight, o_weight) # (hidden_size, n_heads, latent_dim), rank<=head_dim
            o_proj = torch.nn.Linear(n_heads*kv_dim, hidden_size, bias=(module.o_proj.bias is not None))
            o_proj = o_proj.to(device=module.o_proj.weight.device, dtype=module.o_proj.weight.dtype)
            o_proj.weight.data = v_up_o.reshape(hidden_size, n_heads*kv_dim).contiguous()
            if module.o_proj.bias is not None:
                o_proj.bias.data = module.o_proj.bias
            setattr(module, "o_proj", o_proj)
            delattr(module, "v_up_proj")
            module.absorb = True

def test_model(model, tokenizer, prompt="给我讲一个故事吧"):
    """Quick test of the model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
        )
    
    # Decode and display
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"\nGenerated text: {generated_text}")
    
    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert models with GQA to TransMLA")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B", 
                      help="Path or name of the model to convert")
    parser.add_argument("--output", type=str, default="mla-model", 
                      help="Directory to save the converted model")
    parser.add_argument("--num-kv-groups", type=int, required=True, 
                      help="Paged attention only support head_dim<=256, thus we seperate larger dimension to several groups")
    parser.add_argument("--head-dim", type=int, required=True, 
                      help="dimension for each kv group")
    parser.add_argument("--version", type=str, default='v2', 
                    help="""
                    `v1` for latent dimension compression (currently still cache the decompression key/value);
                    `v2` support absorb qk and vo;
                    `v3` upgrade `v1` to support kv_cache compression (TODO);
                    `v4` upgrade `v2` to support kv_cache compression (TODO).
                    """
                )
    parser.add_argument("--flash-attn", action="store_true", 
                      help="Use Flash Attention")
    parser.add_argument("--clover", action="store_true", 
                      help="Cross layer orthogonal vectors")
    parser.add_argument("--rope", type=str, default="extend",
                        help="repeat/extend/pruning")
    parser.add_argument("--absorb", action="store_true",
                      help="Absorb projection matrices (may affect stability)")
    parser.add_argument("--test", action="store_true",
                      help="Test the model after conversion")    
    args = parser.parse_args()
    
    model = Qwen2MLAForCausalLM.from_pretrained(
        args.model, 
        device_map='auto', 
        attn_implementation="flash_attention_2" if args.flash_attn else "eager", 
        num_key_value_heads=args.num_kv_groups,
        rope_repeat = args.rope=='repeat',
        head_dim=args.head_dim
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    hidden_size = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    kv_heads = model.config.num_key_value_heads
    kv_dim = model.config.head_dim
    ori_head_dim = model.config.hidden_size//model.config.num_attention_heads
    latent_dim = kv_heads * kv_dim
    ori_kv_heads = latent_dim//ori_head_dim
    print("ori_head_dim: ", ori_head_dim)
    print("ori_kv_heads: ", ori_kv_heads)
    print("latent_dim: ", latent_dim)
    assert args.num_kv_groups * args.head_dim == latent_dim
    
    if args.version == 'v1':
        print("TODO")
    elif args.version == 'v2':
        if args.rope=='extend':
            print("Apply RoPE extend, no need to change the impletation of rope, lead to minner different, support vllm.")
            init_rope_extend(model, hidden_size, n_heads, kv_heads, kv_dim, ori_kv_heads, ori_head_dim, latent_dim)
        elif args.rope=='repeat':
            print("Apply RoPE repeat, equalvlent transfer, need to change rope's code, currently not support vllm.")
            init_rope_repeat(model, hidden_size, n_heads, kv_heads, kv_dim, ori_kv_heads, ori_head_dim, latent_dim)
        elif args.rope=='pruning':
            raise NotImplementedError
        else:
            raise ValueError
        if args.absorb:
            absorb(model, hidden_size, n_heads, kv_heads, kv_dim, ori_kv_heads, ori_head_dim, latent_dim)
            model.config.absorb=True
        if args.test:
            print(model)
            test_model(model, tokenizer)
    else:
        raise NotImplementedError
    
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)