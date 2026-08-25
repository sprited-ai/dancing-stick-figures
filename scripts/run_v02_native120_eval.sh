#!/usr/bin/env bash
set -euo pipefail

cd "${DSF_ROOT:-/workspace/dsf}"

run="runs/v02/full64_text_video_native20fps_10k"
checkpoint="$run/ckpt_010000.pt"
prompt_out="runs/v02/full64_native120_prompt_suite"
manifest="runs/v02/native120_manifest_n64.json"
sre_v1="output/sre_v1/ckpt_final.pt"
sre_confidence_out="runs/v02/sre_confidence_10k"

while [[ ! -s "$checkpoint" ]]; do
  sleep 20
done

# The trainer writes its final checkpoint before producing the final preview.
# Keep evaluation off the GPU until that process and its data workers exit.
while ps -eo comm=,args= | awk -v target="--out $run" '
  $1 ~ /^python/ && index($0, "train.video_ddpm") && index($0, target) { found=1 }
  END { exit found ? 0 : 1 }
'; do
  sleep 20
done

python - "$checkpoint" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if int(checkpoint.get("step", -1)) != 10_000:
    raise SystemExit(f"expected step 10000, found {checkpoint.get('step')}")
if checkpoint.get("args", {}).get("cond") != "text":
    raise SystemExit("expected a full-prompt text-conditioned checkpoint")
if int(checkpoint.get("args", {}).get("stride", -1)) != 1:
    raise SystemExit("expected stride-1 (20-fps) training")
if int(checkpoint.get("args", {}).get("ar_ctx", -1)) != 8:
    raise SystemExit("expected an 8-frame autoregressive context")
print("verified final full-prompt 20-fps checkpoint", flush=True)
PY

# Complete the learned evaluation track while the same aligned frame/rig cache is mounted.
# The coordinate regressor is already validated; this short warm-started run adds a
# calibrated per-joint uncertainty head before the generated-video evaluations.
if [[ ! -s "$sre_confidence_out/ckpt_final.pt" ]]; then
  [[ -s "$sre_v1" ]] || { echo "missing SRE v1 checkpoint: $sre_v1" >&2; exit 1; }
  python -m train.sre_confidence \
    --cache data/cache_all \
    --out "$sre_confidence_out" \
    --init-v1 "$sre_v1" \
    --steps 10000 \
    --batch 256 \
    --workers 4
fi

python -m eval.post_eval_unet \
  --ckpt "$checkpoint" \
  --out "$prompt_out" \
  --cache data/cache_all \
  --n 6 \
  --frames 120 \
  --steps 50 \
  --cfg 3 \
  --seed 1234 \
  --fps 20 \
  --save_rgba

python -m eval.run_ckpt \
  --run "$run" \
  --cache data/cache_all \
  --n 64 \
  --seeds 3 \
  --frames 120 \
  --stride 1 \
  --sample_steps 50 \
  --batch 8 \
  --cfg 3 \
  --manifest "$manifest"

python - "$prompt_out/manifest.json" "$run/eval/010000.json" <<'PY'
import json
import sys

prompt = json.load(open(sys.argv[1]))
metrics = json.load(open(sys.argv[2]))
if int(metrics.get("step", -1)) != 10_000:
    raise SystemExit("metric output is not tied to step 10000")
if metrics.get("conditioning") != "full_prompt":
    raise SystemExit("metric output does not record full-prompt conditioning")
if int(metrics.get("target_frames", -1)) != 120 or int(metrics.get("reference_stride", -1)) != 1:
    raise SystemExit("metric output does not use the 120-frame, 20-fps protocol")
if int(metrics.get("n", -1)) != 64 or len(metrics.get("sampling_seeds", [])) != 3:
    raise SystemExit("metric output does not contain the declared sample contract")
shape = prompt.get("shape", [])
if len(shape) != 4 or int(shape[1]) != 120:
    raise SystemExit("prompt suite is not 120 frames")
if int(prompt.get("sampler", {}).get("fps", -1)) != 20:
    raise SystemExit("prompt suite is not marked as 20 fps")
print("V02_NATIVE120_COMPLETE=1", flush=True)
PY
