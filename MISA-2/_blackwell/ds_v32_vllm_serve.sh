#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: bash ds_v32_vllm_serve.sh MODEL_PATH [vLLM options]}"
shift
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
export ResMISA_POOL_MAX_BATCH=1

# config.json contains hf_overrides; the checkpoint config is never changed.
exec vllm serve "$MODEL" \
    --hf-overrides "$(cat "$SOURCE_DIR/config.json")" \
    --tokenizer-mode deepseek_v32 \
    --trust-remote-code \
    --served-model-name deepseek-v32 \
    --tensor-parallel-size "${TP:-8}" \
    --enable-expert-parallel \
    --host 127.0.0.1 --port "${PORT:-8000}" \
    --dtype bfloat16 --seed 17 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 135168 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --enforce-eager \
    --attention-backend FLASHMLA_SPARSE \
    --compilation-config '{"mode":0}' \
    "$@"
