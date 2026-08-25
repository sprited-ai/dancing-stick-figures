#!/usr/bin/env bash
set -euo pipefail

# Matched gate for the current temporal ResUNet and its neighbour-aware variant.
# WAIT_PID may name an existing GPU job; this script leaves it alone and begins
# only after that process exits.
TASK_ROOT="${TASK_ROOT:-/data/dancing-stick-figure-paper}"
WAIT_PID="${WAIT_PID:-}"
GATE_BATCH="${GATE_BATCH:-16}"
cd "$TASK_ROOT"

if [[ -n "$WAIT_PID" ]]; then
  echo "waiting for GPU job $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
fi

COMMON=(
  --cache cache/mini
  --frames 8 --ar_ctx 8 --stride 1
  --batch "$GATE_BATCH" --steps 2500 --ch 64
  --cond text --sample_every 2500 --val_every 500
  --workers 4 --amp bf16 --rollout 2 --size 64 --seed 0 --fast
  --init runs/v02/full64_text_image_30k/ckpt.pt
)

echo "starting matched baseline"
python3 -m train.video_ddpm "${COMMON[@]}" \
  --out runs/neighbor_gate/base_2500_clean \
  2>&1 | tee runs/neighbor_gate/base_clean_launcher.log

echo "starting neighbour-aware challenger"
python3 -m train.video_ddpm "${COMMON[@]}" \
  --temporal_neighbors 1 --temporal_pos_bias \
  --out runs/neighbor_gate/neighbor_2500_clean \
  2>&1 | tee runs/neighbor_gate/neighbor_clean_launcher.log

echo "matched gate complete"
