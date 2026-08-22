#!/usr/bin/env bash
set -euo pipefail

root=/data/dancing-stick-figure-paper
python=/home/gin/venvs/ardy/bin/python
out="$root/results/m6v3_start_aligned_h8_20k_s0"
tensorboard=/home/gin/dev/stickdance/runs/paper_m6v3_start_aligned_h8_20k_s0
codec="$root/results/vae_reference_f8t4d16_40k/ckpt_040000.pt"
stats="$root/results/codec_selection_f8t4d16_vs_f8t2d32/candidate_latent_stats.json"
protocol="$root/configs/m6_protocol_v3_start_aligned_h8_20k.json"

mkdir -p "$out"
if [[ -f "$out/COMPLETE" ]]; then
  echo "[$(date -Is)] selected H8 20k run already complete"
  exit 0
fi

# Preserve the exact uncommitted experimental sources alongside the run.
if [[ ! -f "$out/source_latent_video_dit_ar.py" ]]; then
  cp "$root/train/latent_video_dit_ar.py" "$out/source_latent_video_dit_ar.py"
  cp "$root/train/video_dit_ar.py" "$out/source_video_dit_ar.py"
  cp "$protocol" "$out/source_protocol.json"
fi

resume=()
if [[ -f "$out/latest.pt" ]]; then
  resume=(--resume "$out/latest.pt")
fi

cd "$root"
PYTHONPATH=. "$python" -u -m train.latent_video_dit_ar \
  --cache cache/mini --codec "$codec" --latent-stats "$stats" --out "$out" \
  --protocol "$protocol" \
  --history-max 5 --target-latents 2 --history-choices 0,1,2,3,4,5 \
  --rollout-latents 25 --output-size 64 --fps 20 \
  --patch 1 --dim 384 --depth 12 --heads 6 \
  --attention-mode full --training-mode block_ar --start-aligned \
  --batch 16 --steps 20000 --lr 0.0002 --lr-final 0.05 --warmup 500 \
  --shift 1.0 --cfg-drop 0.1 --text-encoder google-t5/t5-small --text-len 32 \
  --sample-steps 10 --sample-cfg 2.0 --sample-every 5000 \
  --sample-milestones 0,100,250,500,1000,2000,5000,10000,15000,20000 \
  --save-every 100 --val-every 500 --workers 8 --seed 0 \
  --compile --fast --tensorboard-dir "$tensorboard" "${resume[@]}"

PYTHONPATH=. "$python" - "$out/latest.pt" <<'PY'
import pathlib, sys, torch
path = pathlib.Path(sys.argv[1])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
assert checkpoint["step"] == 20000
assert checkpoint["protocol"] == "m6_latent_block_ar_v3_start_aligned"
assert checkpoint["args"]["start_aligned"] is True
assert checkpoint["args"]["target_latents"] == 2
assert checkpoint["args"]["decoded_loss_weight"] == 0.0
assert all(key in checkpoint for key in ("model", "ema", "opt", "args", "codec"))
print("verified", path)
PY

echo "[$(date -Is)] selected H8 20k run verified" >> "$out/supervisor.log"
find "$out" -maxdepth 1 -type f ! -name SHA256SUMS ! -name COMPLETE -print0 \
  | sort -z | xargs -0 sha256sum > "$out/SHA256SUMS"
sha256sum -c "$out/SHA256SUMS" >/dev/null
touch "$out/COMPLETE"
