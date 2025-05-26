model_path=meta-llama/Llama-2-7b-hf
save_path=outputs/llama2-7B-deepseek
eval_batch_size=4


# 1. convert to deepseek-mla
python main.py \
    --model-path $model_path \
    --save-path $save_path \
    --freqfold 8 \
    --ppl-eval-batch-size $eval_batch_size


# 2. copy `modeling_deepseek_v3.py` and `configuration_deepseek_v3.py`
cp src/modeling_deepseek_v3.py $save_path/modeling_deepseek_v3.py
cp src/configuration_deepseek_v3.py $save_path/configuration_deepseek_v3.py