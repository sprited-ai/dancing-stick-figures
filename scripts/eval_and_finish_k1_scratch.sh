#!/bin/bash
# Evaluate the verified canonical scratch checkpoint on the same live pod,
# download the result, then terminate only after local validation succeeds.
set -euo pipefail

REPO=/Users/jin/dev/dancing-stick-figure
POD_ID=7s41nj0nfyl1zx
RUN_NAME=k1_canonical_s2_scratch_mix3k
REMOTE_RUN="/workspace/stickdance/runs/$RUN_NAME"
LOCAL_RUN="$REPO/pod_results/$RUN_NAME"
mkdir -p "$LOCAL_RUN"
timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
log() { printf '%s %s\n' "$(timestamp)" "$*" >> "$LOCAL_RUN/eval-watch.log"; }

while [ ! -f "$LOCAL_RUN/COMPLETE" ]; do
  log "waiting for verified step-3000 collection"
  sleep 60
done

line=$(cd "$REPO" && python3 scripts/runpod.py ssh "$POD_ID")
port=$(printf '%s\n' "$line" | awk '{print $3}')
host=${line##*root@}
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
    --out '$REMOTE_RUN/dataset_preflight.json' \
    --grid '$REMOTE_RUN/reference_grid.png'"

if ! ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" "root@$host" \
  "cd /workspace/stickdance && python -u -m eval.text_dit_metrics \
    --ckpt '$REMOTE_RUN/ckpt_003000.pt' --cache data/cache64 \
    --out '$REMOTE_RUN/eval_text_003000.json' --split test \
    --n 64 --sensitivity_n 8 --steps 50 --cfg 3 --batch 1 --seed 1234 \
    --save_rgba '$REMOTE_RUN/eval_text_003000_rgba.npz' \
    > '$REMOTE_RUN/eval_text_003000.log' 2>&1"; then
  log "evaluation failed; pod intentionally left running"
  exit 1
fi

ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" "root@$host" \
  python - "$REMOTE_RUN/eval_text_003000.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
if m.get('protocol') != 'text_dit_metrics_v1' or m.get('checkpoint_step') != 3000 or m.get('n') != 64:
    raise SystemExit('remote evaluation manifest mismatch')
if not {'tvr', 'lie', 'cpe', 'motion_fraction'}.issubset(m.get('oracle', {})):
    raise SystemExit('remote oracle metrics incomplete')
PY

rsync -a --partial -e "ssh -p $port -o StrictHostKeyChecking=no" \
  --include='/eval_text_003000.json' --include='/eval_text_003000.log' \
  --include='/eval_text_003000_rgba.npz' --exclude='*' \
  "root@$host:$REMOTE_RUN/" "$LOCAL_RUN/"

python3 - "$LOCAL_RUN/eval_text_003000.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
if m.get('checkpoint_step') != 3000 or m.get('n') != 64:
    raise SystemExit('local evaluation manifest mismatch')
print(json.dumps({'oracle': m['oracle'], 'prompt_sensitivity': m['prompt_sensitivity']}, indent=2))
PY

log "evaluation downloaded and validated; terminating pod"
(cd "$REPO" && python3 scripts/runpod.py terminate "$POD_ID") >> "$LOCAL_RUN/eval-watch.log" 2>&1
touch "$LOCAL_RUN/EVAL_COMPLETE"
log "evaluation complete and pod terminated"
