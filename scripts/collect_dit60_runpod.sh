#!/usr/bin/env bash
set -euo pipefail

POD_ID="0bnxh7iq28f4mj"
POD_HOST="103.196.86.68"
POD_PORT="34667"
REMOTE_ROOT="/workspace/dsf"
LOCAL_OUT="/Users/jin/dev/dancing-stick-figure/output/runpod_dit60"
SSH=(ssh -o StrictHostKeyChecking=no -p "$POD_PORT" root@"$POD_HOST")

while ! "${SSH[@]}" "test -s $REMOTE_ROOT/runs/dit60_stride2_extension/eval/006000.json && test -s $REMOTE_ROOT/runs/dit60_stride2_extension/prompt_suite/manifest.json"; do
  sleep 30
done

mkdir -p "$LOCAL_OUT"
"${SSH[@]}" "cd $REMOTE_ROOT/runs && tar --exclude=ckpt.pt --exclude=tb -cf - dit60_stride2_extension neighbor_gate/eval_base neighbor_gate/eval_neighbor" \
  | tar -xf - -C "$LOCAL_OUT"

shasum -a 256 \
  "$LOCAL_OUT/dit60_stride2_extension/ckpt_006000.pt" \
  "$LOCAL_OUT/dit60_stride2_extension/eval/006000.json" \
  "$LOCAL_OUT/dit60_stride2_extension/prompt_suite/manifest.json" \
  > "$LOCAL_OUT/SHA256SUMS"

python3 /Users/jin/dev/dancing-stick-figure/scripts/runpod.py terminate "$POD_ID"

