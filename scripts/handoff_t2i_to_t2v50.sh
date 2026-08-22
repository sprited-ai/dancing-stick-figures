#!/bin/bash
# Wait for the locally verified T2I run, then reuse its live pod/cache for the
# canonical 50-frame, 10-fps image-pretrained video pilot.  Failure never
# terminates the pod; the final video watcher handles verified collection and
# termination.
set -euo pipefail

REPO=/Users/jin/dev/dancing-stick-figure
POD_ID=geh83i1bx2qnim
T2I_LOCAL="$REPO/pod_results/t2i64_color_text_p4_fg2_30k"
T2I_REMOTE=/workspace/stickdance/runs/t2i64_color_text_p4_fg2_30k
VIDEO_NAME=t2v50_s2_text_p4_fg2_img10_i2v20_from_t2i30k_10k
VIDEO_REMOTE="/workspace/stickdance/runs/$VIDEO_NAME"
VIDEO_LOCAL="$REPO/pod_results/$VIDEO_NAME"

mkdir -p "$VIDEO_LOCAL"
timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
log() { printf '%s %s\n' "$(timestamp)" "$*" >> "$VIDEO_LOCAL/handoff.log"; }

while [ ! -f "$T2I_LOCAL/COMPLETE" ]; do
  log "waiting for verified T2I completion"
  sleep 60
done

line=$(cd "$REPO" && python3 scripts/runpod.py ssh "$POD_ID")
port=$(printf '%s\n' "$line" | awk '{print $3}')
destination=$(printf '%s\n' "$line" | awk '{print $4}')
host=${destination#root@}
case "$port" in ''|*[!0-9]*) log "invalid endpoint"; exit 1 ;; esac

# Score the completed image foundation before the GPU is committed to the
# video stage.  T=1 evaluation is cheap and gives the curriculum a quantitative
# anatomy/prompt-sensitivity checkpoint of its own.
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" "root@$host" \
  'mkdir -p /workspace/stickdance/eval /workspace/stickdance/scripts /workspace/stickdance/configs'
scp -P "$port" \
  "$REPO/eval/__init__.py" "$REPO/eval/text_dit_metrics.py" \
  "$REPO/eval/post_eval_t2v.py" "$REPO/eval/oracle.py" \
  "$REPO/eval/run_ckpt.py" "$REPO/eval/fvd.py" "$REPO/eval/protocol.py" \
  "root@$host:/workspace/stickdance/eval/"
scp -P "$port" "$REPO/scripts/dataset_preflight.py" \
  "root@$host:/workspace/stickdance/scripts/"
scp -P "$port" "$REPO/configs/dataset_fingerprints.json" \
  "root@$host:/workspace/stickdance/configs/"
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" "root@$host" \
  "cd /workspace/stickdance && python scripts/dataset_preflight.py \
    --cache data/cache64 --profile colored_k1_v1 \
    --out '$T2I_REMOTE/dataset_preflight.json' \
    --grid '$T2I_REMOTE/reference_grid.png'"
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" "root@$host" \
  "cd /workspace/stickdance && python -u -m eval.text_dit_metrics \
    --ckpt '$T2I_REMOTE/ckpt_030000.pt' --cache data/cache64 \
    --out '$T2I_REMOTE/eval_text_030000.json' --split test \
    --n 64 --sensitivity_n 8 --steps 50 --cfg 3 --batch 32 --seed 1234 \
    --save_rgba '$T2I_REMOTE/eval_text_030000_rgba.npz' \
    > '$T2I_REMOTE/eval_text_030000.log' 2>&1"
rsync -a --partial -e "ssh -p $port -o StrictHostKeyChecking=no" \
  --include='/eval_text_030000.json' --include='/eval_text_030000.log' \
  --include='/eval_text_030000_rgba.npz' --exclude='*' \
  "root@$host:$T2I_REMOTE/" "$T2I_LOCAL/"
python3 - "$T2I_LOCAL/eval_text_030000.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
if m.get('checkpoint_step') != 30000 or m.get('n') != 64:
    raise SystemExit('T2I evaluation manifest mismatch')
PY
log "T2I quantitative evaluation downloaded and validated"

ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" "root@$host" \
  bash -s -- "$T2I_REMOTE" "$VIDEO_REMOTE" "$VIDEO_NAME" <<'REMOTE'
set -euo pipefail
t2i="$1"; video="$2"; video_name="$3"
cd /workspace/stickdance
test -s "$t2i/ckpt_030000.pt"
python - "$t2i/ckpt_030000.pt" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
if int(ck.get('step', -1)) != 30000 or 'ema' not in ck:
    raise SystemExit('invalid T2I warm-start checkpoint')
args = ck.get('args', {})
required = dict(size=64, frames=1, patch=4, cond='text', fg_weight=2.0)
for key, expected in required.items():
    if args.get(key) != expected:
        raise SystemExit(f'T2I argument mismatch: {key}={args.get(key)!r}')
PY
if [ -e "$video" ]; then
  echo "refusing to overwrite existing video run: $video" >&2
  exit 1
fi
mkdir -p "$video"
sha256sum train/video_dit_fm.py > "$video/code_sha256.txt"
nohup python -u -m train.video_dit_fm \
  --cache data/cache64 --out "runs/$video_name" \
  --size 64 --frames 50 --stride 2 \
  --patch 4 --dim 384 --depth 12 --heads 6 \
  --cond text --cfg_drop 0.1 \
  --text_encoder google-t5/t5-small --text_len 32 \
  --batch 16 --grad_ckpt --steps 10000 \
  --sample_every 1000 \
  --early_sample_steps 0,1,5,10,25,50,100,250,500 \
  --val_every 500 --workers 8 --fast \
  --shift 1.0 --fg_weight 2.0 --img_frac 0.1 --i2v_frac 0.2 \
  --noise_corr 0.0 --lr_final 0.1 --seed 0 \
  --init "$t2i/ckpt_030000.pt" \
  > "$video/launcher.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$video/pid"
sleep 3
kill -0 "$pid"
REMOTE

log "launched canonical stride-2 video run"

# Collect GIFs/checkpoints, verify the exact final EMA/full checkpoint pair,
# and only then terminate the reused pod.
POD_ID="$POD_ID" RUN_NAME="$VIDEO_NAME" REMOTE_RUN="$VIDEO_REMOTE" \
LOCAL_RUN="$VIDEO_LOCAL" FINAL_STEP=10000 ARTIFACT_EXT=gif \
KEEP_POD_AFTER_COMPLETE=0 POLL_SECONDS=120 \
  caffeinate -dims bash "$REPO/scripts/watch_t2i64_text_30k.sh"
