#!/bin/bash
set -u

POD_ID="15bybntswp1ix0"
POD_HOST="103.196.86.83"
POD_PORT="16385"
REMOTE_RUN="/workspace/dsf/runs/k1_t2v50_64px_4090_b16_3k"
LOCAL_RUN="/Users/jin/dev/dancing-stick-figure/pod_results/k1_t2v50_64px_4090_b16_3k"
REPO="/Users/jin/dev/dancing-stick-figure"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p "$POD_PORT" "root@$POD_HOST")
SCP=(scp -q -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no -P "$POD_PORT")

mkdir -p "$LOCAL_RUN"
echo "$(date -Iseconds) watching $POD_ID" >> "$LOCAL_RUN/watch.log"

while true; do
  if "${SSH[@]}" "test -f '$REMOTE_RUN/ckpt_003000.pt'" 2>/dev/null; then
    break
  fi
  "${SSH[@]}" "tail -1 '$REMOTE_RUN/log.txt' 2>/dev/null" >> "$LOCAL_RUN/watch.log" 2>/dev/null || true
  sleep 60
done

echo "$(date -Iseconds) final checkpoint ready; collecting" >> "$LOCAL_RUN/watch.log"
"${SCP[@]}" "root@$POD_HOST:$REMOTE_RUN/args.json" "$LOCAL_RUN/"
"${SCP[@]}" "root@$POD_HOST:$REMOTE_RUN/log.txt" "$LOCAL_RUN/"
"${SCP[@]}" "root@$POD_HOST:$REMOTE_RUN/launcher.log" "$LOCAL_RUN/"
"${SCP[@]}" "root@$POD_HOST:$REMOTE_RUN/ckpt_003000.pt" "$LOCAL_RUN/"
"${SCP[@]}" "root@$POD_HOST:$REMOTE_RUN/ckpt.pt" "$LOCAL_RUN/"
"${SCP[@]}" "root@$POD_HOST:$REMOTE_RUN/sample_*.gif" "$LOCAL_RUN/"

if test -s "$LOCAL_RUN/ckpt.pt" && test -s "$LOCAL_RUN/ckpt_003000.pt"; then
  shasum -a 256 "$LOCAL_RUN/ckpt.pt" "$LOCAL_RUN/ckpt_003000.pt" > "$LOCAL_RUN/SHA256SUMS"
  cd "$REPO" || exit 1
  python scripts/runpod.py terminate "$POD_ID" >> "$LOCAL_RUN/watch.log" 2>&1
  echo "$(date -Iseconds) collected and terminated $POD_ID" >> "$LOCAL_RUN/watch.log"
  touch "$LOCAL_RUN/COMPLETE"
else
  echo "$(date -Iseconds) collection incomplete; pod intentionally left running" >> "$LOCAL_RUN/watch.log"
  exit 1
fi
