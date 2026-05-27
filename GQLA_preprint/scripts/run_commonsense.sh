#!/usr/bin/env bash
# One-click commonsense evaluation: baseline MLA / GQLA (GQA path) / GQLA-absorb
# (MLA absorb path), six tasks each via lm-eval-harness + vLLM. Each mode runs
# in its own python process so vLLM/cuda state is released between runs.
#
# Both GQLA arches are auto-registered with vLLM via the `vllm.general_plugins`
# entry point installed by `pip install -e .` (see pyproject.toml), in the main
# process and every TP worker — so plain `lm_eval` is enough.
#
# Env overrides:
#   OUT_ROOT           output directory (default: outputs/lm_eval_<ts>)
#   BASELINE_PATH      MLA checkpoint
#   GQLA_PATH          GQLA checkpoint (from `python -m src.convert`)
#   GQLA_ABSORB_PATH   same weights, served via MLA absorb (default: $GQLA_PATH)
#   TP                 tensor parallel size (default: 2)
#   TASKS              comma-separated task list (default: 6 commonsense tasks)

set -euo pipefail

TASKS=${TASKS:-"hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa"}
TP=${TP:-2}
OUT_ROOT=${OUT_ROOT:-outputs/lm_eval_$(date +%Y%m%d_%H%M%S)}
BASELINE_PATH=${BASELINE_PATH:?set BASELINE_PATH to the MLA checkpoint}
GQLA_PATH=${GQLA_PATH:?set GQLA_PATH to the converted GQLA checkpoint}
GQLA_ABSORB_PATH=${GQLA_ABSORB_PATH:-$GQLA_PATH}

mkdir -p "$OUT_ROOT"
echo "writing to $OUT_ROOT (TP=$TP, tasks=$TASKS)"

declare -A PATHS=(
    [baseline]="$BASELINE_PATH"
    [gqla]="$GQLA_PATH"
    [gqla-absorb]="$GQLA_ABSORB_PATH"
)
declare -A ARCH_OVERRIDE=(
    [baseline]=""
    [gqla]=',"hf_overrides":{"architectures":["Glm4MoeLiteGQLAForCausalLM"]}'
    [gqla-absorb]=',"hf_overrides":{"architectures":["Glm4MoeLiteGQLAAbsorbForCausalLM"]}'
)

for MODE in baseline gqla gqla-absorb; do
    OUT_DIR="$OUT_ROOT/$MODE"
    LOG="$OUT_ROOT/${MODE}.log"
    echo "=== $MODE (${PATHS[$MODE]}) ==="
    MODEL_ARGS="{\"pretrained\":\"${PATHS[$MODE]}\",\"dtype\":\"bfloat16\",\"tensor_parallel_size\":${TP},\"gpu_memory_utilization\":0.85,\"max_model_len\":4096,\"enforce_eager\":false${ARCH_OVERRIDE[$MODE]}}"
    python -u -m lm_eval --model vllm \
        --model_args "$MODEL_ARGS" \
        --tasks "$TASKS" \
        --batch_size auto \
        --output_path "$OUT_DIR" 2>&1 | tee "$LOG"
done

echo "=== summary ==="
python -u -m scripts.summarize_lm_eval "$OUT_ROOT"
