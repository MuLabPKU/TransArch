#!/bin/bash
# run_warmup.sh — Stage 1: train the Lightning Indexer to imitate MLA attention.
# Backbone frozen. Two modes:
#   CP_MODE=0 (default): DDP (replicate model, DP over ranks, per-doc within one GPU)
#   CP_MODE=1: context-parallel (split each sequence across NPROC GPUs) — needed for
#              very long ctx (>=64k) where one GPU can't hold the sequence.
#
#   bash scripts/run_warmup.sh [extra --flags passed to opendsa.train.warmup]
#   CP_MODE=1 bash scripts/run_warmup.sh          # context-parallel
#
# Defaults: full 27 layers, 32k ctx, topk 2048, q_chunk 512, UltraData pack_32k.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f env.sh ]; then
    source env.sh
fi

DATA=${DATA:-data_cache/pack_32k}
OUT=${OUT:-runs/warmup}
STEPS=${STEPS:--1}          # -1 = full epoch; set e.g. 200 for a bounded run
NPROC=${NPROC:-8}
TOPK=${TOPK:-2048}
GRAD_ACCUM=${GRAD_ACCUM:-8}
SEQ_LEN=${SEQ_LEN:-32768}
CP_MODE=${CP_MODE:-0}
CFG=configs/accelerate_ddp.yaml

mkdir -p "$OUT" logs
python -m accelerate.commands.launch --config_file "$CFG" --num_processes "$NPROC" \
    -m opendsa.train.warmup \
    --data-dir "$DATA" \
    --output-dir "$OUT" \
    --seq-len "$SEQ_LEN" \
    --topk "$TOPK" \
    --q-chunk 512 \
    --lr 1e-3 \
    --per-device-bs 1 \
    --grad-accum "$GRAD_ACCUM" \
    --logging-steps 5 \
    --save-steps 200 \
    --log-recall-every 25 \
    --max-steps "$STEPS" \
    "$@" 2>&1 | tee "logs/warmup_$(date +%Y%m%d_%H%M%S).log"
