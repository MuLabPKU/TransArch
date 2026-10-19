#!/usr/bin/env bash
# Run a model's code and/or retrieval experiments with the shared vLLM backend.
set -euo pipefail
cd "$(dirname "$0")"
stage="${1:-all}"
model_group="${2:-both}"
case "$stage" in all|code|niah) ;; *) echo 'Usage: bash run_suite.sh [all|code|niah] [v4|v41|both]'; exit 2;; esac
case "$model_group" in v4|v41|both) ;; *) echo 'Model must be v4, v41, or both'; exit 2;; esac
mkdir -p runs results
# On failure, retain and publish partial evidence without calling it complete.
trap 'code=$?; if (( code != 0 )); then "${DEEPSEEK_PYTHON:-$PWD/python.sh}" report.py --from-results results || true; fi' EXIT
exec 9>runs/deepseek_suite.lock
flock -n 9 || { echo 'A suite is already running.'; exit 1; }
python_bin="${DEEPSEEK_PYTHON:-$PWD/python.sh}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
models=(v4 v41)
if [[ "$model_group" != both ]]; then models=("$model_group"); fi
for model in "${models[@]}"; do
    if [[ "$model" == v4 ]]; then
        model_path="${DEEPSEEK_V4_MODEL:-${DEEPSEEK_MODELS:-$PWD/models}/DeepSeek-V4-Flash-0731}"
    else
        model_path="${DEEPSEEK_V41_MODEL:-${DEEPSEEK_MODELS:-$PWD/models}/DeepSeek-V4.1-Flash}"
    fi
    for task in code niah; do
        if [[ "$stage" != all && "$stage" != "$task" ]]; then continue; fi
        if [[ "$task" == code ]]; then
            script=score_code.py
            input=(--prompts prompts/all_code.jsonl)
            output="results/${model}_all_code.jsonl"
        else
            script=run_niah.py
            input=(--data "prompts/niah_${model}")
            output="results/${model}_niah.jsonl"
        fi
        log="runs/${model}_vllm_${task}.log"
        if [[ -f "$log" ]]; then mv "$log" "$log.$(date -u +%Y%m%dT%H%M%SZ).previous"; fi
        echo "Starting $model $task; log=$log"
        "$python_bin" "$script" --model "$model_path" --output "$output" \
            --tp "${TP:-8}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}" \
            --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
            "${input[@]}" > "$log" 2>&1
        "$python_bin" report.py --from-results results --plots
    done
done
