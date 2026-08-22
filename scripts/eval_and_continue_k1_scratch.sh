#!/bin/bash
# Preserve/evaluate the 3k scratch pilot, then reuse the same pod for a fresh
# matched 10k scratch run. The pod is terminated only after the generic watcher
# has downloaded and verified the 10k full checkpoint and EMA artifacts.
set -euo pipefail

REPO=/Users/jin/dev/dancing-stick-figure
POD_ID=7s41nj0nfyl1zx
PILOT_NAME=k1_canonical_s2_scratch_mix3k
PILOT_REMOTE="/workspace/stickdance/runs/$PILOT_NAME"
PILOT_LOCAL="$REPO/pod_results/$PILOT_NAME"
RUN_NAME=k1_canonical_s2_scratch_mix10k
REMOTE_RUN="/workspace/stickdance/runs/$RUN_NAME"
LOCAL_RUN="$REPO/pod_results/$RUN_NAME"
mkdir -p "$PILOT_LOCAL" "$LOCAL_RUN"

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
log() { printf '%s %s\n' "$(timestamp)" "$*" >> "$LOCAL_RUN/handoff.log"; }

while [ ! -f "$PILOT_LOCAL/COMPLETE" ]; do
  log "waiting for verified step-3000 pilot collection"
  sleep 60
done

line=$(cd "$REPO" && python3 scripts/runpod.py ssh "$POD_ID")
port=$(printf '%s\n' "$line" | awk '{print $3}')
destination=$(printf '%s\n' "$line" | awk '{print $4}')
host=${destination#root@}
case "$port" in ''|*[!0-9]*) log "invalid endpoint"; exit 1 ;; esac

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
    --out '$PILOT_REMOTE/dataset_preflight.json' \
    --grid '$PILOT_REMOTE/reference_grid.png'"

ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" "root@$host" \
  "cd /workspace/stickdance && python -u -m eval.text_dit_metrics \
    --ckpt '$PILOT_REMOTE/ckpt_003000.pt' --cache data/cache64 \
    --out '$PILOT_REMOTE/eval_text_003000.json' --split test \
    --n 64 --sensitivity_n 8 --steps 50 --cfg 3 --batch 1 --seed 1234 \
    --save_rgba '$PILOT_REMOTE/eval_text_003000_rgba.npz' \
    > '$PILOT_REMOTE/eval_text_003000.log' 2>&1"

rsync -a --partial -e "ssh -p $port -o StrictHostKeyChecking=no" \
  --include='/eval_text_003000.json' --include='/eval_text_003000.log' \
  --include='/eval_text_003000_rgba.npz' --exclude='*' \
  "root@$host:$PILOT_REMOTE/" "$PILOT_LOCAL/"

python3 - "$PILOT_LOCAL/eval_text_003000.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
if m.get('protocol') != 'text_dit_metrics_v1' or m.get('checkpoint_step') != 3000 or m.get('n') != 64:
    raise SystemExit('pilot evaluation manifest mismatch')
if not {'tvr', 'lie', 'cpe', 'motion_fraction'}.issubset(m.get('oracle', {})):
    raise SystemExit('pilot oracle metrics incomplete')
PY
touch "$PILOT_LOCAL/EVAL_COMPLETE"
log "3k pilot evaluation downloaded and validated"

ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" "root@$host" \
  bash -s -- "$REMOTE_RUN" "$RUN_NAME" <<'REMOTE'
set -euo pipefail
remote_run="$1"; run_name="$2"
cd /workspace/stickdance
if [ -e "$remote_run" ]; then
  echo "refusing to overwrite existing run: $remote_run" >&2
  exit 1
fi
mkdir -p "$remote_run"
sha256sum train/video_dit_fm.py > "$remote_run/code_sha256.txt"
nohup python -u -m train.video_dit_fm \
  --cache data/cache64 --out "runs/$run_name" \
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
  > "$remote_run/launcher.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$remote_run/pid"
sleep 3
kill -0 "$pid"
REMOTE

log "launched fresh matched 10k scratch run"
POD_ID="$POD_ID" RUN_NAME="$RUN_NAME" REMOTE_RUN="$REMOTE_RUN" \
LOCAL_RUN="$LOCAL_RUN" FINAL_STEP=10000 ARTIFACT_EXT=gif \
KEEP_POD_AFTER_COMPLETE=0 POLL_SECONDS=120 \
  caffeinate -dims bash "$REPO/scripts/watch_t2i64_text_30k.sh"
