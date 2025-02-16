# TransMLA: Equivalently Transforms Group Query Attention into Multi-Head Latent Attention.

Modern large language models (LLMs) often encounter communication bottlenecks, rather than computational ones, on current hardware. Multi-head Latent Attention (MLA) addresses these constraints by employing low-rank matrices for KV layers, allowing for caching of compressed latent KV states. This substantially reduces the activation cache size compared to standard multi-head attention and accelerates inference. Moreover, MLA incorporates an up-projection matrix for enhanced expressiveness, effectively trading additional computation for reduced communication overhead.

Despite the proven efficiency and effectiveness of MLA in DeepseekV2/V3, major model providers still rely on GQA, with no public plans to transition to MLA. To facilitate broader adoption, we introduce TransMLA, a post-training method that converts widely used GQA-based pre-trained models into MLA models. This conversion is followed by further training to boost expressiveness without increasing the KV cache size. We also plan to develop MLA-specific inference acceleration techniques to ensure that the transformed models maintain inference latency, ultimately unlocking MLA’s full potential in large-scale LLM deployments.

# News

- [2025.02.16] Released the second version of the TransMLA model and usage code, compatible with RoPE and supporting Absorb operation.
- [2025.02.13] The technical report of TransMLA is publicly available: [https://huggingface.co/papers/2502.07864](https://huggingface.co/papers/2502.07864)
- [2025.01.02] Released the first version of the TransMLA model code, providing usage code for converting Qwen2.5 and LLaMA-3’s GQA to MLA equivalence.

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

# DIY
```
Please follow the implementation provided in qwen_transMLA_rope.ipynb.
```

# To-Do
- [ ] Publish the technical report for the new version, detailing how TransMLA is compatible with RoPE, supports the Absorb operation, and includes experimental comparisons.
- [ ] Add support for vLLM to improve inference speed.
- [ ] Extend support to additional models (e.g., LLaMA, Mistral, Gemma2, etc.).
- [ ] Fine-tune on R1 distillation datasets.

# Citation
```
@article{meng2025transmla,
  title={TransMLA: Multi-head Latent Attention Is All You Need},
  author={Meng, Fanxu and Yao, Zengwei and Zhang, Muhan},
  journal={arXiv preprint arXiv:2502.07864},
  year={2025}
}
```