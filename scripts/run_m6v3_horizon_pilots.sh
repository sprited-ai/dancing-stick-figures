#!/usr/bin/env bash
set -euo pipefail

# Sequential, evidence-preserving M6 v3 pilots.  Only the target horizon varies;
# all arms use the corrected true-start H=0 loader and the same seed/budget.
root=/data/dancing-stick-figure-paper
python=/home/gin/venvs/ardy/bin/python
codec="$root/results/vae_reference_f8t4d16_40k/ckpt_040000.pt"
stats="$root/results/codec_selection_f8t4d16_vs_f8t2d32/candidate_latent_stats.json"

run_arm() {
  local label=$1 target_latents=$2 protocol=$3
  local out="$root/results/m6v3_start_aligned_${label}_2k_s0"
  local tensorboard="/home/gin/dev/stickdance/runs/paper_m6v3_start_aligned_${label}_2k_s0"
  mkdir -p "$out"
  if [[ -f "$out/COMPLETE" ]]; then
    echo "[$(date -Is)] $label already complete"
    return
  fi

  local resume=()
  if [[ -f "$out/latest.pt" ]]; then
    resume=(--resume "$out/latest.pt")
  fi
  echo "[$(date -Is)] starting $label target_latents=$target_latents"
  cd "$root"
  PYTHONPATH=. "$python" -u -m train.latent_video_dit_ar \
    --cache cache/mini --codec "$codec" --latent-stats "$stats" --out "$out" \
    --protocol "$root/configs/$protocol" \
    --history-max 5 --target-latents "$target_latents" --history-choices 0,1,2,3,4,5 \
    --rollout-latents 25 --output-size 64 --fps 20 \
    --patch 1 --dim 384 --depth 12 --heads 6 \
    --attention-mode full --training-mode block_ar --start-aligned \
    --batch 16 --steps 2000 --lr 0.0002 --lr-final 0.05 --warmup 500 \
    --shift 1.0 --cfg-drop 0.1 --text-encoder google-t5/t5-small --text-len 32 \
    --sample-steps 10 --sample-cfg 2.0 --sample-every 9999 \
    --sample-milestones 0,100,250,500,1000,2000 \
    --save-every 100 --val-every 250 --workers 8 --seed 0 \
    --compile --fast --tensorboard-dir "$tensorboard" "${resume[@]}"

  PYTHONPATH=. "$python" - "$out/latest.pt" "$target_latents" <<'PY'
import pathlib, sys, torch
path, target = pathlib.Path(sys.argv[1]), int(sys.argv[2])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
assert checkpoint["step"] == 2000
assert checkpoint["protocol"] == "m6_latent_block_ar_v3_start_aligned"
assert checkpoint["args"]["start_aligned"] is True
assert checkpoint["args"]["target_latents"] == target
assert all(key in checkpoint for key in ("model", "ema", "opt", "args", "codec"))
print("verified", path, "target_latents", target)
PY
  echo "[$(date -Is)] $label complete; writing final checksums" >> "$out/supervisor.log"
  find "$out" -maxdepth 1 -type f ! -name SHA256SUMS ! -name COMPLETE -print0 \
    | sort -z | xargs -0 sha256sum > "$out/SHA256SUMS"
  touch "$out/COMPLETE"
}

run_arm h4 1 m6_protocol_v3_start_aligned_h4_pilot.json
run_arm h8 2 m6_protocol_v3_start_aligned_h8_pilot.json
run_arm h40 10 m6_protocol_v3_start_aligned_h40_pilot.json

echo "[$(date -Is)] all M6 v3 horizon pilots complete"
