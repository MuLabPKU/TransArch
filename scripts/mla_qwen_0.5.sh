BASE_MODEL="/data2/mengfanxu/TransMLA/output/qwen2.5_3b_transMLA"
OUTPUT_PATH="output/python-MLA-Qwen2.5-3B_up_only"
DATA_PATH="pissa-dataset"
export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download --token hf_*** --resume-download $BASE_MODEL --local-dir $BASE_MODEL

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --master_port=16972 --include=localhost:4,5,6,7 train.py \
    --deepspeed ds_configs/zero2.json \
    --model_name_or_path $BASE_MODEL \
    --use_mla True \
    --target_modules "k_up_proj,v_up_proj" \
    --bf16 \
    --data_path $DATA_PATH \
    --sub_task python:100000 \
    --dataset_split "train"\
    --dataset_field instruction output \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
