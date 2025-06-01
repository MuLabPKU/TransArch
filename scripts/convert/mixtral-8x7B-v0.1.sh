model_path=mistralai/Mixtral-8x7B-v0.1
save_path=outputs/mixtral-8x7B-deepseek
eval_batch_size=8


# 1. convert to deepseek-mla
python transmla/converter.py \
    --model-path $model_path \
    --save-path $save_path \
    --freqfold 8 \
    --ppl-eval-batch-size $eval_batch_size


# 2. copy modeling and configuration files
cp transmla/transformers/mixtral/* $save_path/
cp transmla/transformers/mla.py $save_path/