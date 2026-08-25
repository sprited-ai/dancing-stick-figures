#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="${TASK_ROOT:-/workspace/dsf}"
cd "$TASK_ROOT"
RUN="runs/dit120_native20fps"
CKPT="$RUN/ckpt_003000.pt"

while [[ ! -s "$CKPT" ]]; do
  sleep 30
done

python3 -m eval.post_eval_t2v \
  --ckpt "$CKPT" --out "$RUN/prompt_suite" \
  --prompts_file paper/results/neighbor_gate_prompts.txt \
  --same_prompt "A person runs forward." \
  --n 3 --steps 50 --cfg 3 --batch 1 --seed 1234 --fps 20 \
  --strip_frames 0,40,80,119 --save_rgba

python3 -m eval.run_ckpt \
  --run "$RUN" --cache cache/mini \
  --n 64 --seeds 3 --frames 120 --stride 1 \
  --sample_steps 50 --batch 4 --cfg 3

echo "DiT-120 evaluation complete"

