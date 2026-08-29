#!/usr/bin/env bash
set -euo pipefail

# Speed-prioritised 30k pixel baselines.  The loss, optimiser-update count,
# architecture, and declared initialisation stay fixed; physical/effective
# batch is reduced per architecture to minimise wall-clock time at 300 W.

cd /data/dancing-stick-figure-paper
image_checkpoint="results/paper1_v02c_t2i30k_s0/ckpt_030000.pt"

run_fresh() {
  local output_run="$1"
  local physical_batch="$2"
  shift 2

  if [[ -f "${output_run}/ckpt_030000.pt" ]]; then
    echo "SKIP ${output_run}: ckpt_030000.pt already exists"
    return
  fi
  if [[ -e "${output_run}" ]]; then
    echo "REFUSE ${output_run}: output already exists but is incomplete" >&2
    return 1
  fi

  python3 -m train.video_dit_fm \
    --cache cache/mini_v02 \
    --out "${output_run}" \
    --arch dit \
    --frames 64 \
    --first_frames 64 \
    --stride 1 \
    --size 64 \
    --patch 4 \
    --dim 384 \
    --depth 12 \
    --heads 6 \
    --cond text \
    --text_encoder google-t5/t5-small \
    --text_len 32 \
    --cfg_drop 0.1 \
    --batch "${physical_batch}" \
    --accum 1 \
    --optimizer_steps \
    --steps 30000 \
    --lr 2e-4 \
    --lr_final 0.1 \
    --optimizer_beta2 0.999 \
    --grad_clip 0 \
    --ema_max 0.999 \
    --fg_weight 2.0 \
    --rgba_aux_loss 1.0 \
    --img_frac 0.1 \
    --sample_every 10000 \
    --val_every 2000 \
    --workers 8 \
    --seed 0 \
    --fast \
    --compile \
    "$@"
}

# Factorised attention is inexpensive enough that batch 4 gives a useful
# gradient estimate while substantially reducing each optimiser update.
run_fresh \
  results/paper1_v03c_t2v64_factorised_image30k_rgba_s0 \
  4 \
  --init "$image_checkpoint"

run_fresh \
  results/paper1_v03c_t2v64_factorised_random30k_rgba_s0 \
  4

# Full joint attention is quadratic in the 64-frame token sequence.  Batch 2
# avoids activation checkpoint recomputation and is the speed-first setting.
run_fresh \
  results/paper1_v03c_t2v64_fullst_image30k_rgba_s0 \
  2 \
  --init "$image_checkpoint" --full_st
