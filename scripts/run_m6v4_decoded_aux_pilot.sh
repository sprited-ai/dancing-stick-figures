#!/usr/bin/env bash
set -euo pipefail

# Matched H8 ablation against m6v3_start_aligned_h8_2k_s0.  Architecture,
# seed, data, horizon, optimizer budget, and sampling are unchanged; only the
# frozen-decoder RGBA auxiliary term is enabled.
root=/data/dancing-stick-figure-paper
python=/home/gin/venvs/ardy/bin/python
out="$root/results/m6v4_start_aligned_h8_decoded_aux_2k_s0"
tensorboard=/home/gin/dev/stickdance/runs/paper_m6v4_start_aligned_h8_decoded_aux_2k_s0
codec="$root/results/vae_reference_f8t4d16_40k/ckpt_040000.pt"
stats="$root/results/codec_selection_f8t4d16_vs_f8t2d32/candidate_latent_stats.json"
protocol="$root/configs/m6_protocol_v4_start_aligned_h8_decoded_aux_pilot.json"

mkdir -p "$out"
if [[ -f "$out/COMPLETE" ]]; then
  echo "[$(date -Is)] decoded auxiliary pilot already complete"
  exit 0
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
  --decoded-loss-weight 0.25 --decoded-background-alpha-weight 0.02 \
  --batch 16 --steps 2000 --lr 0.0002 --lr-final 0.05 --warmup 500 \
  --shift 1.0 --cfg-drop 0.1 --text-encoder google-t5/t5-small --text-len 32 \
  --sample-steps 10 --sample-cfg 2.0 --sample-every 9999 \
  --sample-milestones 0,100,250,500,1000,2000 \
  --save-every 100 --val-every 250 --workers 8 --seed 0 \
  --compile --fast --tensorboard-dir "$tensorboard" "${resume[@]}"

PYTHONPATH=. "$python" - "$out/latest.pt" <<'PY'
import pathlib, sys, torch
path = pathlib.Path(sys.argv[1])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
assert checkpoint["step"] == 2000
assert checkpoint["protocol"] == "m6_latent_block_ar_v4_decoded_rgba_aux"
assert checkpoint["args"]["start_aligned"] is True
assert checkpoint["args"]["target_latents"] == 2
assert checkpoint["args"]["decoded_loss_weight"] == 0.25
assert checkpoint["args"]["decoded_background_alpha_weight"] == 0.02
assert all(key in checkpoint for key in ("model", "ema", "opt", "args", "codec"))
print("verified", path)
PY

echo "[$(date -Is)] decoded auxiliary H8 pilot verified" >> "$out/supervisor.log"
find "$out" -maxdepth 1 -type f ! -name SHA256SUMS ! -name COMPLETE -print0 \
  | sort -z | xargs -0 sha256sum > "$out/SHA256SUMS"
touch "$out/COMPLETE"
