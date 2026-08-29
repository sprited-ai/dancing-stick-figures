#!/usr/bin/env bash
set -euo pipefail

# Paper-reference queue: factorised Pixel DiTs matched to the completed
# local-mixer and Mini-Wan runs by sampled-window exposure.

cd /data/dancing-stick-figure-paper

image_checkpoint="results/paper1_v02c_t2i30k_s0/ckpt_030000.pt"
cache="cache/mini_v02"

run_one() {
  local output_run="$1"
  local init_mode="$2"

  if [[ -s "${output_run}/ckpt_030000.pt" ]]; then
    echo "SKIP ${output_run}: verified-looking final checkpoint already exists"
    return
  fi
  if [[ -e "${output_run}" ]]; then
    echo "REFUSE ${output_run}: output already exists but is incomplete" >&2
    return 1
  fi

  mkdir "${output_run}"
  sha256sum \
    "${cache}/frames.npy" \
    "${cache}/clips.json" \
    "${cache}/meta.json" \
    train/video_dit_fm.py \
    "${image_checkpoint}" \
    > "${output_run}/input_sha256.txt"

  local init_args=()
  if [[ "${init_mode}" == "image" ]]; then
    init_args=(--init "${image_checkpoint}")
  fi

  python3 -u -m train.video_dit_fm \
    --cache "${cache}" \
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
    --sample_every 1000 \
    --checkpoint_every 1000 \
    --val_every 2000 \
    --workers 8 \
    --seed 0 \
    --fast \
    --compile \
    "${init_args[@]}" \
    2>&1 | tee -a "${output_run}/launcher.log"
}

run_one \
  results/paper1_v04_t2v64_factorised_image30k_rgba_b8a2_s0 \
  image

run_one \
  results/paper1_v04_t2v64_factorised_random30k_rgba_b8a2_s0 \
  random

touch results/GIN_FACTORISED_MATCHED_EPOCHS_QUEUE_COMPLETE
