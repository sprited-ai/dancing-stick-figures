#!/usr/bin/env bash
set -euo pipefail

# Fresh 30k pixel-model runs matched to the Mini-Wan 40M decode-loss recipe.
# Intended experimental differences are backbone/attention and the table's
# declared initialization condition. Pixel models apply the decoded-RGBA
# auxiliary directly because they have neither latents nor a VAE.

cd /data/dancing-stick-figure-paper
image_checkpoint="results/paper1_v02c_t2i30k_s0/ckpt_030000.pt"
local_resume_checkpoint="results/aborted/paper1_v03c_t2v64_local3d_b8a2_20260827/ckpt.pt"

run_fresh() {
  local output_run="$1"
  shift

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
    --batch 8 \
    --accum 2 \
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
    --timing_breakdown \
    "$@"
}

# Strongest and fastest 10k architecture first.
run_fresh \
  results/paper1_v03c_t2v64_local3d_image30k_rgba_s0 \
  --resume "$local_resume_checkpoint" --local_3d

run_fresh \
  results/paper1_v03c_t2v64_factorised_image30k_rgba_s0 \
  --init "$image_checkpoint"

run_fresh \
  results/paper1_v03c_t2v64_factorised_random30k_rgba_s0

# Joint attention uses activation checkpointing only because its quadratic
# 64-frame attention does not fit at the common physical batch otherwise.
run_fresh \
  results/paper1_v03c_t2v64_fullst_image30k_rgba_s0 \
  --init "$image_checkpoint" --full_st --grad_ckpt
