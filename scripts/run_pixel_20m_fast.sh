#!/usr/bin/env bash
set -euo pipefail

# Fast ~20M pixel-space reference suite.  All pixel models share width 384,
# depth 6, p4 tokenisation, the same loss, and 30k optimiser updates.  Physical
# batch is reduced only for the quadratic full-ST attention model.

cd /data/dancing-stick-figure-paper
image_checkpoint="results/paper1_v02c_t2i30k_s0/ckpt_030000.pt"

run_fresh() {
  local output_run="$1"
  local physical_batch="$2"
  shift 2

  if [[ -f "${output_run}/ckpt_030000.pt" ]]; then
    echo "SKIP ${output_run}: complete"
    return
  fi
  if [[ -e "${output_run}" ]]; then
    echo "REFUSE ${output_run}: incomplete output already exists" >&2
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
    --depth 6 \
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
    --sample_every 5000 \
    --val_every 2500 \
    --workers 8 \
    --seed 0 \
    --fast \
    --compile \
    "$@"
}

run_fresh \
  results/paper1_v04_t2v64_local3d_d6_image30k_rgba_s0 \
  4 \
  --init "$image_checkpoint" --local_3d

run_fresh \
  results/paper1_v04_t2v64_factorised_d6_image30k_rgba_s0 \
  4 \
  --init "$image_checkpoint"

run_fresh \
  results/paper1_v04_t2v64_factorised_d6_random30k_rgba_s0 \
  4

run_fresh \
  results/paper1_v04_t2v64_fullst_d6_image30k_rgba_s0 \
  2 \
  --init "$image_checkpoint" --full_st
