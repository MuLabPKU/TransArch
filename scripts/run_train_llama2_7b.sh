BASE_MODEL="outputs/TransMLA-llama-2-7b-q4096-kv448"
OUTPUT_PATH="outputs/ft100m-TransMLA-llama-2-7b-q4096-kv448"
DATA_PATH="/data2/mengfanxu/nanotron/datasets/100m"

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --master_port=16971 --include=localhost:0,1,2,3,4,5,6,7 train.py \
    --deepspeed scripts/ds_config_zero3.json \
    --model_name_or_path $BASE_MODEL \
    --bf16 \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --seq_len 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard"
