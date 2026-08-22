#!/usr/bin/env bash
set -euo pipefail

# Sequential, evidence-preserving f8t4d16 reference-codec experiment on Gin.
# It waits for the current f8t2 finalizer, runs a bounded 10k pilot, applies a
# predeclared viability gate, then resumes the same checkpoint to 40k. It never
# launches M6 or selects a codec; final quantitative + visual selection remains
# a separate decision.

root=/data/dancing-stick-figure-paper
python=/home/gin/venvs/ardy/bin/python
run="$root/results/vae_reference_f8t4d16_40k"
compare="$root/results/codec_selection_f8t4d16_vs_f8t2d32"
f8t2_eval="$root/results/codec_selection_f8t2_vs_f4t4"
tb=/home/gin/dev/stickdance/runs/paper_vae_reference_f8t4d16_40k
log="$run/supervisor.log"
lock="$root/results/f8t4_reference_supervisor.lock"

mkdir -p "$run" "$compare"
exec >>"$log" 2>&1
echo "[$(date -Is)] f8t4 reference supervisor starting"
if ! mkdir "$lock" 2>/dev/null; then
  echo "another supervisor owns $lock; exiting"
  exit 0
fi
trap 'rmdir "$lock" 2>/dev/null || true' EXIT

while [[ ! -f "$f8t2_eval/READY_FOR_VISUAL_AUDIT" ]]; do
  echo "[$(date -Is)] waiting for corrected f8t2 final evaluation"
  sleep 60
done

cd "$root"
common=(
  --cache cache/mini --out "$run"
  --spatial-compression 8 --temporal-compression 4 --latent-channels 16
  --base-channels 32 --blocks-per-stage 2 --frames 20 --size 64 --fps 20
  --batch 16 --lr 0.0002 --workers 4 --seed 0 --compile
  --posterior-mode mean --fixed-samples 4 --alpha-background-weight 1
  --rgb-velocity-weight 0 --alpha-velocity-weight 0
  --rgb-acceleration-weight 0 --alpha-acceleration-weight 0 --kl-max 0
  --tensorboard-dir "$tb"
)

verify_checkpoint() {
  PYTHONPATH=. "$python" - "$1" "$2" <<'PY'
import sys, torch
from train.video_vae_train import state_sha256
path, expected = sys.argv[1], int(sys.argv[2])
c = torch.load(path, map_location="cpu", weights_only=False)
assert c["step"] == expected, (c["step"], expected)
assert state_sha256(c["model"]) == c["model_sha256"]
print("verified", path, c["step"], c["model_sha256"])
PY
}

if [[ ! -f "$run/ckpt_010000.pt" ]]; then
  PYTHONPATH=. "$python" -u -m train.video_vae_train "${common[@]}" \
    --steps 10000 \
    --milestones 0,1,5,10,25,50,100,250,500,1000,2000,3000,5000,7500,10000
fi
verify_checkpoint "$run/ckpt_010000.pt" 10000

mkdir -p "$compare/pilot_short"
PYTHONPATH=. "$python" -m eval.eval_video_vae \
  --cache cache/mini --ckpt "$run/ckpt_010000.pt" \
  --out "$compare/pilot_short/metrics.json" --frames 20 --repeats 4 --batch 16 --workers 4

PYTHONPATH=. "$python" - "$run" "$compare/pilot_short/metrics.json" <<'PY'
import json, pathlib, sys
run, metrics_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
m5 = json.loads((run / "manifest_005000.json").read_text())["metrics"]
m10 = json.loads((run / "manifest_010000.json").read_text())["metrics"]
full = json.loads(metrics_path.read_text())["metrics"]
checks = {
    "fixed_objective_improved_5k_to_10k": m10["total"] < m5["total"],
    "short_rgba_l1_below_loose_pilot_gate": full["rgba_l1"] <= 0.006,
    "short_edge_l1_below_loose_pilot_gate": full["rgb_edge_l1"] <= 0.06,
    "alpha_iou_above_loose_pilot_gate": full["alpha_iou_0.5"] >= 0.85,
}
result = {"checks": checks, "pass": all(checks.values()), "fixed_5k": m5,
          "fixed_10k": m10, "full_short_10k": full}
(run / "pilot_gate.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if not result["pass"]:
    (run / "PILOT_REJECTED").touch()
    raise SystemExit(20)
PY

if [[ ! -f "$run/ckpt_040000.pt" ]]; then
  PYTHONPATH=. "$python" -u -m train.video_vae_train "${common[@]}" \
    --steps 40000 --resume "$run/ckpt_010000.pt" \
    --milestones 15000,20000,25000,30000,35000,40000
fi
verify_checkpoint "$run/ckpt_040000.pt" 40000

mkdir -p "$compare/candidate_short" "$compare/candidate_long"
PYTHONPATH=. "$python" -m eval.eval_video_vae \
  --cache cache/mini --ckpt "$run/ckpt_040000.pt" \
  --out "$compare/candidate_short/metrics.json" --frames 20 --repeats 4 --batch 16 --workers 4
PYTHONPATH=. "$python" -m eval.eval_video_vae_long \
  --cache cache/mini --ckpt "$run/ckpt_040000.pt" --out "$compare/candidate_long" \
  --clips 24 --frames 120 --tile 20 --commit 4 --gifs 4 --fps 20
PYTHONPATH=. "$python" -m eval.vae_latent_stats \
  --cache cache/mini --ckpt "$run/ckpt_040000.pt" --out "$compare/candidate_latent_stats.json" \
  --frames 24 --windows 1024 --repeats 4 --batch 16 --workers 4

PYTHONPATH=. "$python" -m scripts.compare_vae_codecs \
  --protocol configs/codec_selection_f8t4d16_reference.json \
  --baseline-short "$f8t2_eval/f8_short/metrics.json" \
  --baseline-long "$f8t2_eval/f8_long/metrics.json" \
  --candidate-short "$compare/candidate_short/metrics.json" \
  --candidate-long "$compare/candidate_long/metrics.json" \
  --out "$compare/comparison.json"

find "$compare" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$compare/SHA256SUMS"
touch "$compare/READY_FOR_VISUAL_AUDIT"
echo "[$(date -Is)] f8t4 reference convergence and quantitative comparison complete"
