#!/usr/bin/env bash
set -euo pipefail

root=/data/dancing-stick-figure-paper
python=/home/gin/venvs/ardy/bin/python
candidate_run="$root/results/vae_final_f8t2d32_mirrored_40k"
candidate_ckpt="$candidate_run/ckpt_040000.pt"
baseline_ckpt="$root/results/vae_clean2_block_t4d32_f80_b16_50k/ckpt_042500.pt"
result_root="$root/results/codec_selection_f8t2_vs_f4t4"
lock="$root/results/codec_finalize.lock"

mkdir -p "$result_root"
exec >>"$result_root/finalizer.log" 2>&1
echo "[$(date -Is)] finalizer starting"

if ! mkdir "$lock" 2>/dev/null; then
  echo "another finalizer owns $lock; exiting"
  exit 0
fi
trap 'rmdir "$lock" 2>/dev/null || true' EXIT

while true; do
  if [[ -f "$candidate_ckpt" ]]; then
    break
  fi
  pid=$(cat "$candidate_run/pid")
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "candidate trainer exited without final checkpoint"
    touch "$result_root/FAILED"
    exit 1
  fi
  echo "[$(date -Is)] waiting: $(tail -1 "$candidate_run/log.jsonl")"
  sleep 60
done

cd "$root"
PYTHONPATH=. "$python" - "$candidate_ckpt" <<'PY'
import sys, torch
from train.video_vae_train import state_sha256
p=sys.argv[1]
c=torch.load(p,map_location='cpu',weights_only=False)
assert c['step']==40000, c['step']
assert state_sha256(c['model'])==c['model_sha256']
print('candidate checkpoint integrity verified', c['model_sha256'])
PY

mkdir -p "$result_root/f8_short" "$result_root/f4_short" \
         "$result_root/f8_long" "$result_root/f4_long"

PYTHONPATH=. "$python" -m eval.eval_video_vae \
  --cache cache/mini --ckpt "$candidate_ckpt" \
  --out "$result_root/f8_short/metrics.json" --frames 20 --repeats 4 --batch 16 --workers 4
PYTHONPATH=. "$python" -m eval.eval_video_vae \
  --cache cache/mini --ckpt "$baseline_ckpt" \
  --out "$result_root/f4_short/metrics.json" --frames 20 --repeats 4 --batch 16 --workers 4

PYTHONPATH=. "$python" -m eval.eval_video_vae_long \
  --cache cache/mini --ckpt "$candidate_ckpt" --out "$result_root/f8_long" \
  --clips 24 --frames 120 --tile 20 --commit 4 --gifs 4 --fps 20
PYTHONPATH=. "$python" -m eval.eval_video_vae_long \
  --cache cache/mini --ckpt "$baseline_ckpt" --out "$result_root/f4_long" \
  --clips 24 --frames 120 --tile 20 --commit 4 --gifs 4 --fps 20

PYTHONPATH=. "$python" -m eval.vae_latent_stats \
  --cache cache/mini --ckpt "$candidate_ckpt" --out "$result_root/f8_latent_stats.json" \
  --frames 24 --windows 1024 --repeats 4 --batch 16 --workers 4

PYTHONPATH=. "$python" -m scripts.compare_vae_codecs \
  --protocol configs/codec_selection_f8t2_vs_f4t4.json \
  --baseline-short "$result_root/f4_short/metrics.json" \
  --baseline-long "$result_root/f4_long/metrics.json" \
  --candidate-short "$result_root/f8_short/metrics.json" \
  --candidate-long "$result_root/f8_long/metrics.json" \
  --out "$result_root/comparison.json"

find "$result_root" -type f ! -name SHA256SUMS ! -name finalizer.log -print0 | \
  sort -z | xargs -0 sha256sum > "$result_root/SHA256SUMS"
touch "$result_root/READY_FOR_VISUAL_AUDIT"
echo "[$(date -Is)] quantitative codec finalization complete"
