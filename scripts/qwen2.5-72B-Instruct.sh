model_path=Qwen/Qwen2.5-72B-Instruct
save_path=outputs/qwen2_5-72B-Instruct-deepseek
eval_batch_size=4


# 1. convert to deepseek-mla
python transmla/converter.py \
    --model-path $model_path \
    --save-path $save_path \
    --freqfold 4 \
    --ppl-eval-batch-size $eval_batch_size


# 2. copy modeling and configuration files
cp transmla/transformers/llama/* $save_path/
cp transmla/transformers/mla.py $save_path/