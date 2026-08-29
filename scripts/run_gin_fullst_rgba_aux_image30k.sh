#!/usr/bin/env bash
set -euo pipefail

cd /data/dancing-stick-figure-paper

output_run="results/paper1_v03c_t2v64_fullst_image30k_rgba_s0"
image_checkpoint="results/paper1_v02c_t2i30k_s0/ckpt_030000.pt"

if [[ -e "$output_run" ]]; then
  echo "REFUSE: $output_run already exists" >&2
  exit 1
fi
if [[ ! -s "$image_checkpoint" ]]; then
  echo "REFUSE: missing image checkpoint $image_checkpoint" >&2
  exit 1
fi

mkdir "$output_run"
sha256sum cache/mini_v02/frames.npy cache/mini_v02/clips.json \
  cache/mini_v02/meta.json train/video_dit_fm.py "$image_checkpoint" \
  > "$output_run/input_sha256.txt"

exec python3 -u -m train.video_dit_fm \
  --cache cache/mini_v02 \
  --out "$output_run" \
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
  --init "$image_checkpoint" \
  --full_st \
  --grad_ckpt
