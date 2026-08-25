#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="${TASK_ROOT:-/data/dancing-stick-figure-paper}"
CACHE="${CACHE:-$TASK_ROOT/cache/mini_seed_split_v1}"
IMAGE_RUN="${IMAGE_RUN:-$TASK_ROOT/results/paper1_v02_seed_split_t2i30k_s0}"
VIDEO_RUN="${VIDEO_RUN:-$TASK_ROOT/results/paper1_v02_seed_split_t2v50_10k_s0}"

cd "$TASK_ROOT"
test -s "$CACHE/frames.npy"
test -s "$CACHE/clips.json"
for destination in "$IMAGE_RUN" "$VIDEO_RUN"; do
  if [ -e "$destination" ]; then
    echo "refusing to overwrite existing run: $destination" >&2
    exit 1
  fi
done

mkdir -p "$IMAGE_RUN"
sha256sum "$CACHE/clips.json" train/video_dit_fm.py > "$IMAGE_RUN/input_sha256.txt"
python3 -u -m train.video_dit_fm \
  --cache "$CACHE" --out "$IMAGE_RUN" --arch dit \
  --size 64 --frames 1 --stride 2 --patch 4 --dim 384 --depth 12 --heads 6 \
  --cond text --cfg_drop 0.1 --text_encoder google-t5/t5-small --text_len 32 \
  --batch 128 --steps 30000 --lr 0.0002 --lr_final 0.02 \
  --fg_weight 2.0 --workers 8 --fast --seed 0 \
  --val_every 500 --sample_every 5000 --early_sample_steps 0,100,250,500 \
  > "$IMAGE_RUN/launcher.log" 2>&1

test -s "$IMAGE_RUN/ckpt_030000.pt"
mkdir -p "$VIDEO_RUN"
sha256sum "$CACHE/clips.json" train/video_dit_fm.py "$IMAGE_RUN/ckpt_030000.pt" > "$VIDEO_RUN/input_sha256.txt"
python3 -u -m train.video_dit_fm \
  --cache "$CACHE" --out "$VIDEO_RUN" --arch dit \
  --size 64 --frames 50 --stride 2 --patch 4 --dim 384 --depth 12 --heads 6 \
  --cond text --cfg_drop 0.1 --text_encoder google-t5/t5-small --text_len 32 \
  --batch 8 --accum 2 --grad_ckpt --steps 10000 --lr 0.0002 --lr_final 0.1 \
  --fg_weight 2.0 --img_frac 0.1 --i2v_frac 0.2 --noise_corr 0.0 \
  --workers 8 --fast --seed 0 --val_every 500 --sample_every 1000 \
  --early_sample_steps 0,100,250,500 --init "$IMAGE_RUN/ckpt_030000.pt" \
  > "$VIDEO_RUN/launcher.log" 2>&1

test -s "$VIDEO_RUN/ckpt_010000.pt"
touch "$VIDEO_RUN/COMPLETE"
