#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="${TASK_ROOT:-/workspace/dsf}"
cd "$TASK_ROOT"

while [[ ! -f runs/dit50_local3d_matched10k/PAIR_COMPLETE ]]; do
  sleep 30
done

MANIFEST="runs/dit50_matched10k_reference_manifest.json"
for RUN in dit50_factorized_matched10k dit50_local3d_matched10k; do
  python3 -m eval.post_eval_t2v \
    --ckpt "runs/$RUN/ckpt.pt" --cache cache/mini --out "runs/$RUN/prompt_suite" \
    --n 8 --steps 50 --cfg 3 --batch 2 --seed 20260824 --fps 10 \
    --strip_frames 0,9,19,29,39,49 --save_rgba

  python3 -m eval.run_ckpt \
    --run "runs/$RUN" --cache cache/mini --n 64 --seeds 3 \
    --frames 50 --stride 2 --sample_steps 50 --batch 4 --cfg 3 \
    --manifest "$MANIFEST"
done

touch runs/dit50_local3d_matched10k/EVAL_COMPLETE
