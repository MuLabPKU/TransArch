#!/bin/bash
# run_sparse.sh — Stage 2: sparse continued-training (backbone + indexer).
# Default path: FSDP ZeRO-3 (shard params/grads/optimizer across GPUs).
# CP_MODE=1: context+expert parallel with Zero2Adam, no FSDP, for long contexts.
# Loads warmup-trained indexer weights.
#
#   bash scripts/run_sparse.sh [extra --flags passed to opendsa.train.sparse]
#
# Defaults: full 27 layers, 32k ctx, topk 2048, q_chunk 512, UltraData pack_32k.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f env.sh ]; then
    source env.sh
fi

DATA=${DATA:-data_cache/pack_32k}
OUT=${OUT:-runs/sparse}
INDEXER=${INDEXER:-runs/warmup/indexer.pt}
STEPS=${STEPS:--1}
NPROC=${NPROC:-8}
TOPK=${TOPK:-2048}
QCHUNK=${QCHUNK:-512}
GRAD_ACCUM=${GRAD_ACCUM:-8}
SEQ_LEN=${SEQ_LEN:-32768}
LOG_RECALL_EVERY=${LOG_RECALL_EVERY:-0}

# expandable_segments cuts fragmentation from the per-chunk gather/free cycle
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CP_MODE=1 -> context+expert parallel (CP=EP=NPROC, no FSDP); else FSDP ZeRO-3.
CP_MODE=${CP_MODE:-0}
CFG=configs/accelerate_fsdp.yaml
[ "$CP_MODE" = "1" ] && CFG=configs/accelerate_ddp.yaml

mkdir -p "$OUT" logs
python -m accelerate.commands.launch --config_file "$CFG" --num_processes "$NPROC" \
    -m opendsa.train.sparse \
    --data-dir "$DATA" \
    --output-dir "$OUT" \
    --indexer-init "$INDEXER" \
    --seq-len "$SEQ_LEN" \
    --topk "$TOPK" \
    --q-chunk "$QCHUNK" \
    --lr 2e-5 \
    --per-device-bs 1 \
    --grad-accum "$GRAD_ACCUM" \
    --indexer-loss-coeff 1.0 \
    --grad-ckpt True \
    --logging-steps 5 \
    --save-steps 200 \
    --log-recall-every "$LOG_RECALL_EVERY" \
    --max-steps "$STEPS" \
    "$@" 2>&1 | tee "logs/sparse_$(date +%Y%m%d_%H%M%S).log"
