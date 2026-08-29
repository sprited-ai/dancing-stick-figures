#!/usr/bin/env bash
set -euo pipefail

# Exploratory pixel-space baselines for fin's 16 GB RTX 4090 Laptop GPU.
# These use a dataset-specific RGBA auxiliary objective.  The objective is not
# mathematically identical to Mini-Wan's decoded loss and these artifacts must
# not be promoted into the frozen paper table without a separate review.

cd /home/fin/dancing-stick-figures

python=/home/fin/venvs/ardy/bin/python
cache=cache/mini_v02
image_checkpoint=results/paper1_v02c_t2i30k_s0/ckpt_030000.pt

expected_frames_bytes=7903641728
expected_frames_sha256=3d70537fdfb43f85db9dc1227c49ed1b5174a2e3dcceda3659e64b650f813a90
expected_clips_sha256=461db6d0be28005e4d2821fa39502936db285f8538cdbb7d048d2e775b49508b
expected_meta_sha256=af56fa12e81de2bceeb714f37ebe915a34791b98c5e6fa0a2fb6e76559ad22bc
expected_image_sha256=d3865d04e2f7f0660d13c56050af89f625f1d034d437b6a9c2c07b5ae4b084e9

[[ $(stat -c %s "$cache/frames.npy") == "$expected_frames_bytes" ]]
echo "$expected_frames_sha256  $cache/frames.npy" | sha256sum --check --status
echo "$expected_clips_sha256  $cache/clips.json" | sha256sum --check --status
echo "$expected_meta_sha256  $cache/meta.json" | sha256sum --check --status
echo "$expected_image_sha256  $image_checkpoint" | sha256sum --check --status

run_one() {
  local out=$1
  local init_mode=$2

  if [[ -s "$out/ckpt_030000.pt" ]]; then
    echo "SKIP $out: final checkpoint already exists"
    return
  fi

  mkdir -p "$out"
  local state_args=()
  if [[ -s "$out/ckpt.pt" ]]; then
    state_args=(--resume "$out/ckpt.pt")
  elif [[ "$init_mode" == image ]]; then
    state_args=(--init "$image_checkpoint")
  fi

  "$python" -m train.video_dit_fm \
    --cache "$cache" \
    --out "$out" \
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
    --batch 4 \
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
    --sample_every 1000 \
    --checkpoint_every 1000 \
    --val_every 2000 \
    --workers 2 \
    --seed 0 \
    --fast \
    --compile \
    --grad_ckpt \
    "${state_args[@]}" \
    2>&1 | tee -a "$out/launcher.log"
}

run_one results/explore_factorised_rgba_aux_image30k_fin_s0 image
run_one results/explore_factorised_rgba_aux_random30k_fin_s0 random

touch results/FIN_FACTORISED_RGBA_AUX_QUEUE_COMPLETE
