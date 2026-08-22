#!/usr/bin/env bash
set -euo pipefail

# Evidence-preserving M6 v2 run on Gin. The selected codec, latent statistics,
# protocol, and source files are checksum-bound in the produced checkpoint.

root=/data/dancing-stick-figure-paper
python=/home/gin/venvs/ardy/bin/python
out="$root/results/m6_f8t4d16_fullst_10k_s0"
codec="$root/results/vae_reference_f8t4d16_40k/ckpt_040000.pt"
stats="$root/results/codec_selection_f8t4d16_vs_f8t2d32/candidate_latent_stats.json"
protocol="$root/configs/m6_protocol_v2_full_st.json"
selection="$root/configs/selected_codec_m6.json"
tensorboard=/home/gin/dev/stickdance/runs/paper_m6_f8t4d16_fullst_10k_s0
lock="$root/results/m6_f8t4d16_fullst_10k_s0.lock"

mkdir -p "$out"
exec >>"$out/supervisor.log" 2>&1
echo "[$(date -Is)] M6 supervisor starting"
if ! mkdir "$lock" 2>/dev/null; then
  echo "another supervisor owns $lock; exiting"
  exit 0
fi
trap 'rmdir "$lock" 2>/dev/null || true' EXIT

cd "$root"
PYTHONPATH=. "$python" - "$selection" <<'PY'
import hashlib, json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert manifest["selection"] == "f8t4d16"
for section, path_key in (("checkpoint", "path_on_gin"), ("latent_statistics", "path_on_gin")):
    item = manifest[section]
    path = pathlib.Path(item[path_key])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == item["sha256"], (section, digest, item["sha256"])
print("selected codec bindings verified")
PY

if [[ -f "$out/COMPLETE" ]]; then
  echo "verified run already complete"
  exit 0
fi

resume=()
if [[ -f "$out/latest.pt" ]]; then
  resume=(--resume "$out/latest.pt")
  echo "resuming verified declared run from $out/latest.pt"
fi

PYTHONPATH=. "$python" -u -m train.latent_video_dit_ar \
  --cache cache/mini --codec "$codec" --latent-stats "$stats" --out "$out" \
  --protocol "$protocol" \
  --history-max 5 --target-latents 1 --history-choices 0,1,2,3,4,5 \
  --rollout-latents 25 --output-size 64 --fps 20 \
  --patch 1 --dim 384 --depth 12 --heads 6 \
  --attention-mode full --training-mode block_ar \
  --batch 16 --steps 10000 --lr 0.0002 --lr-final 0.05 --warmup 500 \
  --shift 1.0 --cfg-drop 0.1 --text-encoder google-t5/t5-small --text-len 32 \
  --sample-steps 10 --sample-cfg 2.0 --sample-every 1000 \
  --sample-milestones 0,10,50,100,250,500,1000,2000,5000,10000 \
  --save-every 100 --val-every 250 --workers 8 --seed 0 \
  --compile --fast --tensorboard-dir "$tensorboard" "${resume[@]}"

PYTHONPATH=. "$python" - "$out/latest.pt" "$codec" "$stats" "$protocol" <<'PY'
import hashlib, pathlib, sys, torch
checkpoint_path, codec_path, stats_path, protocol_path = map(pathlib.Path, sys.argv[1:])
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert checkpoint["step"] == 10000
assert checkpoint["protocol"] == "m6_latent_block_ar_v2_full_st"
assert checkpoint["codec"]["checkpoint_sha256"] == sha(codec_path)
assert checkpoint["codec"]["latent_stats_sha256"] == sha(stats_path)
assert checkpoint["codec"]["experiment_protocol_sha256"] == sha(protocol_path)
assert all(key in checkpoint for key in ("model", "ema", "opt", "args"))
print("M6 final checkpoint verified")
PY

# Write the terminal log record before hashing.  Writing it afterwards makes
# the otherwise valid manifest fail on supervisor.log even though every model
# and sample artifact is intact.
echo "[$(date -Is)] M6 10k complete; writing final checksums"
find "$out" -maxdepth 1 -type f ! -name SHA256SUMS ! -name COMPLETE -print0 | sort -z | xargs -0 sha256sum > "$out/SHA256SUMS"
touch "$out/COMPLETE"
