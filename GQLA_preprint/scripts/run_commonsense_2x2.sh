#!/usr/bin/env bash
# Batched 4-parallel commonsense sweep for the 2x2 ablation + MLA baseline.
#
# 9 modes total:
#   - mla (Glm4MoeLite, baseline)
#   - {neigh_nohess, neigh_hess, sim_nohess, sim_hess} × {gqa, absorb}
# Each ablation config's GQA and MLA-absorb paths share the SAME on-disk weights;
# only `hf_overrides.architectures` differs (Glm4MoeLiteGQLA{,Absorb}ForCausalLM).
#
# Topology: 8 GPUs split into 4 groups of TP=2. Each batch launches 4 modes in
# parallel (one per GPU group); waits for all to finish; launches next batch.
# 9 modes / 4 workers = 3 batches; expected wall-clock ~9h at TP=2.
#
# Resumable: a mode whose output_path already contains a results*.json is skipped.
#
# Env overrides:
#   PYTHON               (default: python)
#   OUT_ROOT             (default: outputs/abl_2x2_commonsense)
#   ABL_ROOT             (default: outputs/abl_2x2)
#   BASELINE_PATH        (default: GLM-4.7-Flash local path)
#   TASKS                (default: 7 commonsense tasks, matches README full table)
#   TP                   (default: 2)
#   LIMIT                (default: unset = full eval; set to integer for smoke)

set -euo pipefail

PYTHON=${PYTHON:-python}
OUT_ROOT=${OUT_ROOT:-outputs/abl_2x2_commonsense}
ABL_ROOT=${ABL_ROOT:-outputs/abl_2x2}
BASELINE_PATH=${BASELINE_PATH:-/path/to/GLM-4.7-Flash}
TASKS=${TASKS:-"hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,boolq"}
TP=${TP:-2}
LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

mkdir -p "$OUT_ROOT"

GPU_GROUPS=( "0,1" "2,3" "4,5" "6,7" )

# (tag | model_path | arch_suffix) — arch_suffix is the comma-prefixed JSON fragment
# appended to lm_eval's --model_args (empty for baseline MLA).
gqa_arch=',"hf_overrides":{"architectures":["Glm4MoeLiteGQLAForCausalLM"]}'
abs_arch=',"hf_overrides":{"architectures":["Glm4MoeLiteGQLAAbsorbForCausalLM"]}'

modes=(
    "mla|$BASELINE_PATH|"
    "neigh_nohess_gqa|$ABL_ROOT/neigh_nohess|$gqa_arch"
    "neigh_nohess_absorb|$ABL_ROOT/neigh_nohess|$abs_arch"
    "neigh_hess_gqa|$ABL_ROOT/neigh_hess|$gqa_arch"
    "neigh_hess_absorb|$ABL_ROOT/neigh_hess|$abs_arch"
    "sim_nohess_gqa|$ABL_ROOT/sim_nohess|$gqa_arch"
    "sim_nohess_absorb|$ABL_ROOT/sim_nohess|$abs_arch"
    "sim_hess_gqa|$ABL_ROOT/sim_hess|$gqa_arch"
    "sim_hess_absorb|$ABL_ROOT/sim_hess|$abs_arch"
)

run_one() {
    local tag=$1 path=$2 arch=$3 gpus=$4
    local out_dir="$OUT_ROOT/$tag"
    local log="$OUT_ROOT/${tag}.log"

    if [ -d "$out_dir" ] && [ -n "$(find "$out_dir" -name 'results*.json' 2>/dev/null | head -1)" ]; then
        echo "[$(date +%H:%M:%S)] SKIP   $tag (results.json present)"
        return 0
    fi

    rm -rf "$out_dir"
    mkdir -p "$out_dir"
    local model_args
    model_args="{\"pretrained\":\"$path\",\"dtype\":\"bfloat16\",\"tensor_parallel_size\":$TP,\"gpu_memory_utilization\":0.85,\"max_model_len\":4096,\"enforce_eager\":false${arch}}"

    echo "[$(date +%H:%M:%S)] START  $tag  GPU($gpus)  path=$path"
    CUDA_VISIBLE_DEVICES="$gpus" "$PYTHON" -m lm_eval \
        --model vllm \
        --model_args "$model_args" \
        --tasks "$TASKS" \
        --batch_size auto \
        $LIMIT_ARG \
        --output_path "$out_dir" > "$log" 2>&1 \
        && echo "[$(date +%H:%M:%S)] DONE   $tag  (log: $log)" \
        || echo "[$(date +%H:%M:%S)] FAIL   $tag  (log: $log; non-fatal, continuing)"
}

started=$(date +%s)
N=${#modes[@]}
echo "[$(date +%H:%M:%S)] $N modes, 4-parallel, TP=$TP"
echo "  out: $OUT_ROOT"
echo "  tasks: $TASKS"
[ -n "${LIMIT:-}" ] && echo "  LIMIT=$LIMIT (smoke mode)"

i=0
while [ $i -lt $N ]; do
    pids=()
    tags=()
    echo
    echo "================= BATCH starting at $(date +%H:%M:%S) ================="
    for j in 0 1 2 3; do
        idx=$((i + j))
        if [ $idx -ge $N ]; then break; fi
        cfg="${modes[$idx]}"
        IFS='|' read -r tag path arch <<< "$cfg"
        run_one "$tag" "$path" "$arch" "${GPU_GROUPS[$j]}" &
        pids+=($!)
        tags+=("$tag")
    done
    # Wait for entire batch
    for k in "${!pids[@]}"; do
        wait "${pids[$k]}" || true
    done
    echo "[$(date +%H:%M:%S)] batch done (modes: ${tags[*]})"
    i=$((i + 4))
done

elapsed=$(( $(date +%s) - started ))
echo
echo "================================================================"
echo "[$(date +%H:%M:%S)] all 9 modes done in ${elapsed}s ($((elapsed/60)) min)"
echo "================================================================"
echo "results under: $OUT_ROOT/<tag>/"
echo "run: $PYTHON scripts/summarize_lm_eval.py $OUT_ROOT  (aggregator)"
