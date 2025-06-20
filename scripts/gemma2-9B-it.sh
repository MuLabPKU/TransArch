model_path=/data2/mengfanxu/huggingface/gemma-2-9b-it
save_path=outputs/gemma2-9B-it-deepseek
eval_batch_size=4


# 1. convert to deepseek-mla
python transmla/converter.py \
    --model-path $model_path \
    --save-path $save_path \
    --freqfold 8 \
    --ppl-eval-batch-size $eval_batch_size \
    --device cuda:2


# 2. copy modeling and configuration files
cp transmla/transformers/gemma2/* $save_path/
cp transmla/transformers/mla.py $save_path/

python debug/test.py --model-path $save_path --device cuda:2 --ppl-eval-batch-size $eval_batch_size