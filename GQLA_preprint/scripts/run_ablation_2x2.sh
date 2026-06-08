#!/usr/bin/env bash
# 2x2 ablation over (head_grouping x hessian_pca) on GLM-4.7-Flash.
#
# Reproduces the README's neighbor + no-hess baseline (PPL ≈ 12.07) and
# extends with the three new configs added in this branch:
#   (neighbor,   off    )   <- legacy / README reproduction
#   (neighbor,   nll    )
#   (similarity, off    )
#   (similarity, nll    )
#
# Calibration: 128 x 512 = 65k tokens (matches README).
# PPL: wikitext-2 concat-and-chunk, seqlen=1024, all 282 chunks.
#
# Serial / resumable: a config whose gqla_meta.json already has "ppl_gqla"
# is skipped. Re-run after a crash and only missing runs execute.
#
# Env overrides:
#   PYTHON           interpreter (default: python)
#   MODEL_PATH       MLA checkpoint (default: GLM-4.7-Flash local)
#   OUT_ROOT         output directory (default: outputs/abl_2x2)
#   CAL_NSAMPLES     calibration samples (default: 128)
#   CAL_SEQLEN       calibration seqlen (default: 512)
#   NUM_KV_HEADS     GQA groups (default: 4 — matches README's 5x KV reduction)
#   DEVICE_MAP       HF device_map (default: auto)
#   EVAL_PPL_MAX_CHUNKS  cap eval chunks (default: empty = all)

set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-/path/to/GLM-4.7-Flash}
OUT_ROOT=${OUT_ROOT:-outputs/abl_2x2}
CAL_NSAMPLES=${CAL_NSAMPLES:-128}
CAL_SEQLEN=${CAL_SEQLEN:-512}
NUM_KV_HEADS=${NUM_KV_HEADS:-4}
DEVICE_MAP=${DEVICE_MAP:-auto}

mkdir -p "$OUT_ROOT"
cd "$(dirname "$0")/.."

# (tag | head_grouping | hessian_flag) — '|'-delimited so empty hess field is unambiguous.
# Order: grouping-first sweep (matches user's "grouping-first" ablation preference).
configs=(
  "neigh_nohess|neighbor|"
  "neigh_hess|neighbor|--hessian_pca --hessian_mode nll"
  "sim_nohess|similarity|"
  "sim_hess|similarity|--hessian_pca --hessian_mode nll"
)

started_at=$(date +%s)
for cfg in "${configs[@]}"; do
    IFS='|' read -r tag grouping hess <<< "$cfg"
    SAVE="$OUT_ROOT/$tag"
    LOG="$OUT_ROOT/${tag}.log"
    META="$SAVE/gqla_meta.json"

    echo
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] CONFIG  tag=$tag  grouping=$grouping  hess=${hess:-OFF}"
    echo "================================================================"

    # Resume guard: if meta exists AND has ppl_gqla, skip.
    if [ -f "$META" ] && grep -q '"ppl_gqla":' "$META"; then
        cached_ppl=$(grep '"ppl_gqla":' "$META" | head -1 | sed 's/.*: *\([0-9.]*\).*/\1/')
        echo "  SKIP — already complete, ppl_gqla=$cached_ppl"
        continue
    fi

    # Otherwise (re)compute from scratch.
    rm -rf "$SAVE"
    "$PYTHON" -m src.convert \
        --model_path "$MODEL_PATH" \
        --save_path  "$SAVE" \
        --num_kv_heads "$NUM_KV_HEADS" --dtype bf16 --device_map "$DEVICE_MAP" \
        --cal_dataset wikitext2 --cal_nsamples "$CAL_NSAMPLES" --cal_seqlen "$CAL_SEQLEN" \
        --head_grouping "$grouping" \
        $hess \
        --eval_ppl --eval_ppl_dataset wikitext2 --eval_ppl_seqlen 1024 \
        ${EVAL_PPL_MAX_CHUNKS:+--eval_ppl_max_chunks $EVAL_PPL_MAX_CHUNKS} \
        > "$LOG" 2>&1

    ppl=$(grep '"ppl_gqla":' "$META" 2>/dev/null | head -1 | sed 's/.*: *\([0-9.]*\).*/\1/' || true)
    echo "  done. ppl_gqla=${ppl:-UNKNOWN}  (log: $LOG)"
done

echo
echo "================================================================"
echo "[$(date +%H:%M:%S)] all done in $(( $(date +%s) - started_at ))s. summary:"
echo "================================================================"
printf "%-15s | %-10s | %-10s | %s\n" "tag" "grouping" "hess" "ppl_gqla"
printf "%-15s-+-%-10s-+-%-10s-+-%s\n" "---------------" "----------" "----------" "--------"
for cfg in "${configs[@]}"; do
    IFS='|' read -r tag grouping hess <<< "$cfg"
    META="$OUT_ROOT/$tag/gqla_meta.json"
    ppl=$(grep '"ppl_gqla":' "$META" 2>/dev/null | head -1 | sed 's/.*: *\([0-9.]*\).*/\1/' || echo "FAIL")
    hflag=$([ -z "$hess" ] && echo "OFF" || echo "nll")
    printf "%-15s | %-10s | %-10s | %s\n" "$tag" "$grouping" "$hflag" "$ppl"
done
