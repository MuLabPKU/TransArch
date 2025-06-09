# 🚀 TransMLA: Migrating GQA Models to MLA with Full DeepSeek Compatibility and Speedup

Modern large-language models often face communication bottlenecks on current hardware rather than computational limitations. Multi-head latent attention (MLA) addresses this by compressing the key-value cache using low-rank matrices, while the Absorb operation prevents the KV cache from reverting to its original size, significantly boosting both training and inference speed. 

Despite the success of DeepSeek V2/V3/R1, most model vendors have heavily invested in optimizing GQA-based models and therefore lack strong incentives to retrain MLA-based models from scratch. In this paper, we introduce TransMLA, a framework that seamlessly converts any GQA-based pre-trained model (e.g., LLaMA, Qwen, Mixtral) into an MLA-based model. 


# 📰 News
- [2025.05.29] A new version of technical report is released: [https://arxiv.org/abs/2502.07864](https://arxiv.org/abs/2502.07864).
- [2025.04.28] Released TransMLA v3, successfully apply PCA across RoPE and reduce KV Cache.
- [2025.02.16] Released the second version of the TransMLA model and usage code, compatible with RoPE and supporting Absorb operation.
- [2025.02.13] The technical report of TransMLA is publicly available: [https://huggingface.co/papers/2502.07864](https://huggingface.co/papers/2502.07864)
- [2025.01.02] Released the first version of the TransMLA model code, providing usage code for converting Qwen2.5 and LLaMA-3’s GQA to MLA equivalence.

# 🛠 Installation
```
git clone https://github.com/fxmeng/TransMLA.git
cd TransMLA
conda create -n transmla python=3.12.8
conda activate transmla
pip install -r requirements.txt
```

# ⚡ Quick Start

1. Convert MHA / GQA models (e.g. Qwen2.5-7B-Instruct) into DeepSeek-MLA:
    ```bash
    bash scripts/convert/qwen2.5-7B-Instruct.sh
    ```
2. Have fun playing with the converted models!
    ```python
    # using `Transformers.AutoModelForCausalLM`
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("outputs/qwen2_5-7B-Instruct-deepseek", trust_remote_code=True)
    ```

## 🔧 Advanced Usage (`converter.py`)

The converter.py script allows you to perform fine-grained control over RoPE removal and low-rank QKV projection towards DeepSeek-MLA. It supports:
- Auto-search for optimal freqfold that minimizes PPL.
- Automatic computation of collapse based on head_dim / qk_mqa_dim.
- Evaluation of original, RoPE-removed, and final MLA models.


### ✅ Example Command:
```bash
python transmla/converter.py \
    --model-path meta-llama/Llama-2-7b-hf \
    --save-path ./outputs/llama2-7b-deepseek \
    --dtype bf16 \
    --device auto \
    --cal-dataset wikitext2 \
    --cal-nsamples 128 \
    --cal-max-seqlen 256 \
    --cal-batch-size 8 \
    --ppl-eval-batch-size 4 \
    --freqfold auto \
    --collapse auto \
    --qk-mqa-dim 64 \
    --q-lora-rank 512 \
    --kv-lora-rank 512
```

### 📘 Argument Details

| Argument | Description |
|----------|-------------|
| --model-path | Path to the base model (e.g., from HuggingFace hub). |
| --save-path | Output path for the converted model and tokenizer. |
| --cal-dataset | Calibration dataset: wikitext2, ptb, c4, or alpaca. |
| --cal-nsamples, --cal-max-seqlen, --cal-batch-size | Number, max sequence length, and batch size of samples used for calibration. |
| --freqfold | RoPE frequency folding factor, or `auto` to search for the best value. |
| --collapse | Collapse factor for RoPE. Use `auto` to compute as `head_dim // qk_mqa_dim`. Collapse factor reduces the dim of RoPEd KV cache from `head_dim` to `head_dim // collapse`. |
| --qk-mqa-dim | Target dimension for decoupled RoPE. |
| --q-lora-rank | The inner dimension for query low-rank decomposition, or `None` to disable low-rank decomposition for query. |
| --kv-lora-rank | The inner dimension for key/value joint low-rank decomposition. |


### 🧠 Tips
- Set `--freqfold auto` and `--collapse auto` to simplify configuration. The script will automatically search for the best freqfold factor based on ppl results.
- We recommend setting `--qk-mqa-dim` to 64 and `--kv-lora-rank` to 512 to satisfy FlashMLA's requirements on H100.


# 🐒 Model Zoo

- [x] Llama2
- [x] Llama3
- [x] Qwen2
- [x] Gemma2
- [x] Mistral
- [x] Mixtral
- [ ] MiMo
- [ ] Dots.LLM1


# 📋 To-Do
- [x] Publish the technical report for the new version, detailing how TransMLA is compatible with RoPE, supports the Absorb operation.
- [x] Compress the dimensions of the KV cache to improve inference speed.
- [x] Add support for vLLM to improve inference speed.
- [x] Support FlashMLA.
- [x] Extend support to additional models (e.g., LLaMA, Mistral, Gemma2, etc.).
- [x] Support GTA & GLA
- [ ] Release checkpoints.
- [ ] Fine-tune on R1 distillation datasets.


# 📚 Citation
```
@article{meng2025transmla,
  title={TransMLA: Multi-head Latent Attention Is All You Need},
  author={Meng, Fanxu and Tang, Pingzhi and Yao, Zengwei and Zhang, Muhan},
  journal={arXiv preprint arXiv:2502.07864},
  year={2025}
}
```

# ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=fxmeng/TransMLA&type=Date)](https://www.star-history.com/#fxmeng/TransMLA&Date)
