#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="${TASK_ROOT:-/workspace/dsf}"
cd "$TASK_ROOT"
BASE="runs/neighbor_gate/base_2500_clean/ckpt.pt"
CHALLENGER="runs/neighbor_gate/neighbor_2500_clean/ckpt.pt"

while [[ ! -s "$CHALLENGER" ]]; do
  sleep 30
done

COMMON=(
  --cache cache/mini
  --prompts_file paper/results/neighbor_gate_prompts.txt
  --same_prompt "A person runs forward."
  --n 3 --frames 60 --steps 30 --cfg 3 --seed 1234 --fps 20
  --strip_frames 0,20,40,59 --save_rgba
)

python3 -m eval.post_eval_unet --ckpt "$BASE" \
  --out runs/neighbor_gate/eval_base "${COMMON[@]}"
python3 -m eval.post_eval_unet --ckpt "$CHALLENGER" \
  --out runs/neighbor_gate/eval_neighbor "${COMMON[@]}"

echo "neighbor gate evaluation complete"
