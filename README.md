# TransMLA
TransMLA: Equivalently Transforms Group Query Attention into Multi-head Latent Attention.

Modern large language models (LLMs) often encounter communication bottlenecks, rather than computational ones, on current hardware. Multi-head Latent Attention (MLA) addresses these constraints by employing low-rank matrices for KV layers, allowing for caching of compressed latent KV states. This substantially reduces the activation cache size compared to standard multi-head attention and accelerates inference. Moreover, MLA incorporates an up-projection matrix for enhanced expressiveness, effectively trading additional computation for reduced communication overhead.

Despite the proven efficiency and effectiveness of MLA in DeepseekV2/V3, major model providers still rely on GQA, with no public plans to transition to MLA. To facilitate broader adoption, we introduce TransMLA, a post-training method that converts widely used GQA-based pre-trained models into MLA models. This conversion is followed by further training to boost expressiveness without increasing the KV cache size. We also plan to develop MLA-specific inference acceleration techniques to ensure that the transformed models maintain inference latency, ultimately unlocking MLA’s full potential in large-scale LLM deployments.
# Install
```
conda create -n transmla python=3.10.14
conda activate transmla
pip install torch==2.4.0
pip install transformers==4.46.2 
pip install accelerate>=0.26.0
pip install ipykernel
pip install deepspeed==0.15.4
pip install vllm==0.6.2
pip install tensorboardX
pip install tqdm attrdict fraction
pip install human_eval==1.0.3
pip install evalplus==0.2.1
```
# Quick Start
```
from llama.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer
model = LlamaForCausalLM.from_pretrained("fxmeng/llama3.2_1b_instruct_transMLA", device_map='cuda:0')
tokenizer = AutoTokenizer.from_pretrained("fxmeng/llama3.2_1b_instruct_transMLA")
output = model.generate(**tokenizer("Give me a short introduction to large language model.",return_tensors="pt").to("cuda:0"), max_new_tokens=256)
print(tokenizer.batch_decode(output))
```
```
from qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers import AutoTokenizer
model = Qwen2ForCausalLM.from_pretrained("fxmeng/qwen2.5_0.5b_instruct_transMLA", attn_implementation="eager", device_map='cuda:0')
tokenizer = AutoTokenizer.from_pretrained("fxmeng/qwen2.5_0.5b_instruct_transMLA")
output = model.generate(**tokenizer("Give me a short introduction to large language model.",return_tensors="pt").to("cuda:0"), max_new_tokens=256)
tokenizer.batch_decode(output)
print(tokenizer.batch_decode(output))
```

# DIY
```
Please follow the implementation provided in llama_transMLA.ipynb and qwen_transMLA.ipynb.
```

# Todo
- [ ] Support more models (Mistral, Gemma2, ...)
- [ ] Finetune on SFT dataset
- [ ] Continue Pre-training, SFT, DPO for improving the capability
- [ ] Optimize inference implementation for higher speed

# Citation
```
@misc{meng2025transmla,
  author = {Fanxu meng, Zengwei Yao, Muhan Zhang},
  title = {TransMLA: Equivalently Transforms Group Query Attention into Multi-head Latent Attention},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/fxmeng/TransMLA}
}
```