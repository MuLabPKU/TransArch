model_path=meta-llama/Llama-2-7b-hf
save_path=outputs/llama2-7B-deepseek
eval_batch_size=4


# 1. convert to deepseek-mla
python transmla/converter.py \
    --model-path $model_path \
    --save-path $save_path \
    --freqfold 8 \
    --ppl-eval-batch-size $eval_batch_size


# 2. copy modeling and configuration files
cp transmla/transformers/llama/* $save_path/
cp transmla/transformers/mla.py $save_path/