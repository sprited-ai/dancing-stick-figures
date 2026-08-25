#!/usr/bin/env bash
set -euo pipefail

POD_ID="0bnxh7iq28f4mj"
POD_HOST="103.196.86.68"
POD_PORT="34667"
REMOTE_ROOT="/workspace/dsf"
LOCAL_OUT="/Users/jin/dev/dancing-stick-figure/output/runpod_neighbor_gate"
SSH=(ssh -o StrictHostKeyChecking=no -p "$POD_PORT" root@"$POD_HOST")

while ! "${SSH[@]}" "test -s $REMOTE_ROOT/runs/neighbor_gate/eval_neighbor/manifest.json"; do
  sleep 30
done

mkdir -p "$LOCAL_OUT"
"${SSH[@]}" "cd $REMOTE_ROOT/runs/neighbor_gate && tar --exclude=ckpt.pt --exclude=tb -cf - base_2500_clean neighbor_2500_clean eval_base eval_neighbor pair_driver.log eval_driver.log" \
  | tar -xf - -C "$LOCAL_OUT"

shasum -a 256 \
  "$LOCAL_OUT/base_2500_clean/ckpt_002500.pt" \
  "$LOCAL_OUT/neighbor_2500_clean/ckpt_002500.pt" \
  "$LOCAL_OUT/eval_base/manifest.json" \
  "$LOCAL_OUT/eval_neighbor/manifest.json" \
  > "$LOCAL_OUT/SHA256SUMS"

python /Users/jin/dev/dancing-stick-figure/scripts/runpod.py terminate "$POD_ID"

# The free Gin fallback should still be waiting for Claudia's active GPU job.
ssh gin 'if test -s /data/dancing-stick-figure-paper/runs/neighbor_gate/queued_driver.pid; then kill "$(cat /data/dancing-stick-figure-paper/runs/neighbor_gate/queued_driver.pid)" 2>/dev/null || true; fi'
