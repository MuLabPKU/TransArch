model_path=mistralai/Mistral-7B-v0.3
save_path=outputs/mistral-7B-v0.3-deepseek
eval_batch_size=8


# 1. convert to deepseek-mla
python transmla/converter.py \
    --model-path $model_path \
    --save-path $save_path \
    --freqfold 8 \
    --ppl-eval-batch-size $eval_batch_size


# 2. copy `modeling_llamamla.py` and `configuration_llamamla.py`
cp transmla/transformers/llama/* $save_path/