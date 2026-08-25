#!/usr/bin/env bash
# Official v0.2 reference pipeline: 120-frame (6 s, 20 fps) seed-disjoint curated dataset.
# All training samples are restricted to the first 64 frames (3.2 s) of each clip,
# the action-dense span. Video models jointly generate that complete 64-frame,
# stride-1, native-cadence window.
# Stage 1: text-conditioned image DiT, 30k steps.
# Stage 2: factorised video DiT, image-warm-started, 64-frame window, 10k steps.
# Stage 3: matched random-init video DiT, identical protocol minus --init.
set -euo pipefail

TASK_ROOT="${TASK_ROOT:-/data/dancing-stick-figure-paper}"
CACHE="${CACHE:-$TASK_ROOT/cache/mini_seed_split_v1}"
IMAGE_RUN="${IMAGE_RUN:-$TASK_ROOT/results/paper1_v02_t2i30k_s0}"
WARM_RUN="${WARM_RUN:-$TASK_ROOT/results/paper1_v02_t2v40win_warm10k_s0}"
RAND_RUN="${RAND_RUN:-$TASK_ROOT/results/paper1_v02_t2v40win_rand10k_s0}"

cd "$TASK_ROOT"
test -s "$CACHE/frames.npy"
test -s "$CACHE/clips.json"
for destination in "$IMAGE_RUN" "$WARM_RUN" "$RAND_RUN"; do
  if [ -e "$destination" ]; then
    echo "refusing to overwrite existing run: $destination" >&2
    exit 1
  fi
done

mkdir -p "$IMAGE_RUN"
sha256sum "$CACHE/clips.json" train/video_dit_fm.py > "$IMAGE_RUN/input_sha256.txt"
python3 -u -m train.video_dit_fm \
  --cache "$CACHE" --out "$IMAGE_RUN" --arch dit \
  --size 64 --frames 1 --stride 1 --first_frames 64 --patch 4 --dim 384 --depth 12 --heads 6 \
  --cond text --cfg_drop 0.1 --text_encoder google-t5/t5-small --text_len 32 \
  --batch 128 --steps 30000 --lr 0.0002 --lr_final 0.02 \
  --fg_weight 2.0 --workers 8 --fast --seed 0 \
  --val_every 500 --sample_every 5000 --early_sample_steps 0,100,250,500 \
  > "$IMAGE_RUN/launcher.log" 2>&1
test -s "$IMAGE_RUN/ckpt_030000.pt"
touch "$IMAGE_RUN/COMPLETE"

video_stage() {
  local run="$1"; shift
  mkdir -p "$run"
  sha256sum "$CACHE/clips.json" train/video_dit_fm.py > "$run/input_sha256.txt"
  python3 -u -m train.video_dit_fm \
    --cache "$CACHE" --out "$run" --arch dit \
    --size 64 --frames 64 --stride 1 --first_frames 64 --patch 4 --dim 384 --depth 12 --heads 6 \
    --cond text --cfg_drop 0.1 --text_encoder google-t5/t5-small --text_len 32 \
    --batch 8 --accum 2 --grad_ckpt --steps 10000 --lr 0.0002 --lr_final 0.1 \
    --fg_weight 2.0 --img_frac 0.1 --i2v_frac 0.2 --noise_corr 0.0 \
    --workers 8 --fast --seed 0 --val_every 500 --sample_every 1000 \
    --early_sample_steps 0,100,250,500 "$@" \
    > "$run/launcher.log" 2>&1
  test -s "$run/ckpt_010000.pt"
  touch "$run/COMPLETE"
}

sha256sum "$IMAGE_RUN/ckpt_030000.pt" >> "$IMAGE_RUN/input_sha256.txt"
video_stage "$WARM_RUN" --init "$IMAGE_RUN/ckpt_030000.pt"
video_stage "$RAND_RUN"
touch "$TASK_ROOT/results/PAPER1_V02_REFERENCE_DONE"
