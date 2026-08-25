#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="${TASK_ROOT:-/workspace/dsf}"
cd "$TASK_ROOT"

python3 -m train.video_dit_fm \
  --cache cache/mini \
  --out runs/dit120_native20fps \
  --arch dit --size 64 --frames 120 --stride 1 \
  --dim 384 --depth 12 --heads 6 --patch 4 \
  --cond text --text_encoder google-t5/t5-small --text_len 32 --cfg_drop 0.1 \
  --init runs/m3_warm_10k_ema.pt \
  --batch "${DIT_BATCH:-8}" --accum "${DIT_ACCUM:-1}" \
  --steps "${DIT_STEPS:-3000}" --lr 0.0001 --lr_final 0.1 \
  --img_frac 0.1 --i2v_frac 0.2 --fg_weight 2.0 \
  --grad_ckpt --fast --workers 8 --seed 0 \
  --val_every 500 --sample_every 1000 --early_sample_steps 0,100,250,500

