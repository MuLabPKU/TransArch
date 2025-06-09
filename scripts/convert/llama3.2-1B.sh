model_path=meta-llama/Llama-3.2-1B
save_path=outputs/llama3_2-1B-deepseek
eval_batch_size=16


# 1. convert to deepseek-mla
python transmla/converter.py \
    --model-path $model_path \
    --save-path $save_path \
    --freqfold 4 \
    --ppl-eval-batch-size $eval_batch_size


# 2. copy modeling and configuration files
cp transmla/transformers/llama/* $save_path/
cp transmla/transformers/mla.py $save_path/