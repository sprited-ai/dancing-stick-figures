#!/bin/bash
# Safely collect the correct colored-data 64px text-conditioned T2I run at
# step 30,000.  This deliberately does not point at the preserved gray-data
# mismatch run.
# Any SSH, snapshot, transfer, validation, or API failure leaves the pod alive.
set -uo pipefail

REPO=/Users/jin/dev/dancing-stick-figure
POD_ID="${POD_ID:-geh83i1bx2qnim}"
RUN_NAME="${RUN_NAME:-t2i64_color_text_p4_fg2_30k}"
REMOTE_RUN="${REMOTE_RUN:-/workspace/stickdance/runs/t2i64_color_text_p4_fg2_30k}"
LOCAL_RUN="${LOCAL_RUN:-$REPO/pod_results/t2i64_color_text_p4_fg2_30k}"
FINAL_STEP="${FINAL_STEP:-30000}"
ARTIFACT_EXT="${ARTIFACT_EXT:-png}"
POLL_SECONDS="${POLL_SECONDS:-120}"
KEEP_POD_AFTER_COMPLETE="${KEEP_POD_AFTER_COMPLETE:-0}"

mkdir -p "$LOCAL_RUN"
timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

resolve_endpoint() {
  local line port destination host
  line=$(cd "$REPO" && python3 scripts/runpod.py ssh "$POD_ID" 2>/dev/null) || return 1
  port=$(printf '%s\n' "$line" | awk '{print $3}')
  destination=$(printf '%s\n' "$line" | awk '{print $4}')
  host=${destination#root@}
  case "$port" in ''|*[!0-9]*) return 1 ;; esac
  [ -n "$host" ] && [ "$host" != None ] || return 1
  printf '%s %s\n' "$port" "$host"
}

rsync_shell() {
  printf 'ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no -p %s' "$1"
}

sync_progress() {
  local port="$1" host="$2" shell
  shell=$(rsync_shell "$port")
  mkdir -p "$LOCAL_RUN/checkpoints"
  # Lightweight mutable logs and immutable milestone artifacts.  Full ckpt.pt
  # is intentionally excluded until training has stopped at the final step.
  rsync -a --partial -e "$shell" \
    --include='/args.json' \
    --include='/log.txt' \
    --include='/launcher.log' \
    --include='/code_sha256.txt' \
    --include='/sample_*.png' \
    --include='/sample_raw_*.png' \
    --include='/sample_*.gif' \
    --include='/sample_raw_*.gif' \
    --include='/sample_manifest_*.json' \
    --include='/ckpt_[0-9]*.pt' \
    --exclude='*' \
    "root@$host:$REMOTE_RUN/" "$LOCAL_RUN/" || return 1
  if [ "$ARTIFACT_EXT" = png ]; then
    python3 "$REPO/scripts/t2i_milestones.py" "$LOCAL_RUN" "$LOCAL_RUN/milestone_progression.png" \
      >> "$LOCAL_RUN/watch.log" 2>&1 || true
  fi
}

remote_ready() {
  local port="$1" host="$2"
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no -p "$port" "root@$host" \
    bash -s -- "$REMOTE_RUN" "$RUN_NAME" "$FINAL_STEP" "$ARTIFACT_EXT" <<'REMOTE'
set -uo pipefail
run_dir="$1"; run_name="$2"; final_step="$3"; artifact_ext="$4"
cd "$run_dir" || exit 1
snapshot=$(printf 'ckpt_%06d.pt' "$final_step")
sample=$(printf 'sample_%06d.%s' "$final_step" "$artifact_ext")
manifest=$(printf 'sample_manifest_%06d.json' "$final_step")
test -s args.json -a -s log.txt -a -s launcher.log -a -s ckpt.pt || exit 1
test -s "$snapshot" -a -s "$sample" -a -s "$manifest" || exit 1
grep -q "^step ${final_step} " log.txt || exit 1
if ps -eo comm=,args= | awk -v run="$run_name" \
  '$1 ~ /python/ && $0 ~ /train\.video_dit_fm/ && index($0, "--out runs/" run) { found=1 } END { exit(found ? 0 : 1) }'
then
  exit 1
fi
python - "$final_step" "$snapshot" <<'PY'
import sys, torch
full = torch.load('ckpt.pt', map_location='cpu', weights_only=False)
ema = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
step = int(sys.argv[1])
if int(full.get('step', -1)) != step or int(ema.get('step', -1)) != step:
    raise SystemExit('checkpoint step mismatch')
if not {'model', 'ema', 'opt', 'args', 'step'}.issubset(full):
    raise SystemExit('full checkpoint keys missing')
if not {'ema', 'args', 'step'}.issubset(ema):
    raise SystemExit('EMA snapshot keys missing')
if full['ema'].keys() != ema['ema'].keys():
    raise SystemExit('EMA state keys differ')
if not all(torch.equal(full['ema'][key], ema['ema'][key]) for key in full['ema']):
    raise SystemExit('full and snapshot EMA tensors differ')
PY
REMOTE
}

prepare_final_manifest() {
  local port="$1" host="$2"
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no -p "$port" "root@$host" \
    bash -s -- "$REMOTE_RUN" "$FINAL_STEP" <<'REMOTE'
set -euo pipefail
run_dir="$1"; final_step="$2"
cd "$run_dir"
full=$(printf 'ckpt_full_%06d.pt' "$final_step")
ema=$(printf 'ckpt_%06d.pt' "$final_step")
if [ ! -s "$full" ]; then
  cp --reflink=auto ckpt.pt "${full}.tmp"
  mv "${full}.tmp" "$full"
fi
python - "$final_step" "$full" "$ema" <<'PY'
import sys, torch
step = int(sys.argv[1])
full = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
ema = torch.load(sys.argv[3], map_location='cpu', weights_only=False)
if int(full.get('step', -1)) != step or int(ema.get('step', -1)) != step:
    raise SystemExit('step mismatch')
if not {'model', 'ema', 'opt', 'args', 'step'}.issubset(full):
    raise SystemExit('full checkpoint incomplete')
if full['ema'].keys() != ema['ema'].keys():
    raise SystemExit('EMA keys differ')
if not all(torch.equal(full['ema'][k], ema['ema'][k]) for k in full['ema']):
    raise SystemExit('EMA tensors differ')
PY
files=(args.json log.txt launcher.log "$full")
[ -s code_sha256.txt ] && files+=(code_sha256.txt)
for pattern in 'ckpt_[0-9]*.pt' 'sample_*.png' 'sample_raw_*.png' 'sample_*.gif' 'sample_raw_*.gif' 'sample_manifest_*.json'; do
  for file in $pattern; do
    [ -f "$file" ] && files+=("$file")
  done
done
sha256sum "${files[@]}" > SHA256SUMS_FINAL.tmp
mv SHA256SUMS_FINAL.tmp SHA256SUMS_FINAL
REMOTE
}

download_and_verify() {
  local port="$1" host="$2" shell file
  shell=$(rsync_shell "$port")
  rsync -a --partial -e "$shell" \
    "root@$host:$REMOTE_RUN/SHA256SUMS_FINAL" "$LOCAL_RUN/" || return 1
  while read -r _hash file; do
    case "$file" in ''|*/*|.|..) echo "unsafe manifest entry: $file"; return 1 ;; esac
    rsync -a --partial -e "$shell" \
      "root@$host:$REMOTE_RUN/$file" "$LOCAL_RUN/" || return 1
  done < "$LOCAL_RUN/SHA256SUMS_FINAL"
  (cd "$LOCAL_RUN" && shasum -a 256 -c SHA256SUMS_FINAL) || return 1
  python3 - "$LOCAL_RUN" "$FINAL_STEP" <<'PY'
import sys, torch
from pathlib import Path
run, step = Path(sys.argv[1]), int(sys.argv[2])
full = torch.load(run / f'ckpt_full_{step:06d}.pt', map_location='cpu', weights_only=False)
ema = torch.load(run / f'ckpt_{step:06d}.pt', map_location='cpu', weights_only=False)
if int(full.get('step', -1)) != step or int(ema.get('step', -1)) != step:
    raise SystemExit('local checkpoint step mismatch')
if not {'model', 'ema', 'opt', 'args', 'step'}.issubset(full):
    raise SystemExit('local full checkpoint incomplete')
if full['ema'].keys() != ema['ema'].keys():
    raise SystemExit('local EMA keys differ')
if not all(torch.equal(full['ema'][k], ema['ema'][k]) for k in full['ema']):
    raise SystemExit('local full/snapshot EMA mismatch')
PY
}

printf '%s watching pod %s for final step %d\n' "$(timestamp)" "$POD_ID" "$FINAL_STEP" >> "$LOCAL_RUN/watch.log"
while [ ! -f "$LOCAL_RUN/COMPLETE" ]; do
  endpoint=$(resolve_endpoint 2>/dev/null || true)
  if [ -z "$endpoint" ]; then
    printf '%s endpoint unavailable; retrying\n' "$(timestamp)" >> "$LOCAL_RUN/watch.log"
    sleep "$POLL_SECONDS"
    continue
  fi
  read -r port host <<< "$endpoint"

  if ! sync_progress "$port" "$host" >> "$LOCAL_RUN/watch.log" 2>&1; then
    printf '%s progress sync failed; pod intentionally left running\n' "$(timestamp)" >> "$LOCAL_RUN/watch.log"
  fi
  latest=$(tail -1 "$LOCAL_RUN/log.txt" 2>/dev/null || true)
  printf '%s %s\n' "$(timestamp)" "${latest:-waiting for log}" >> "$LOCAL_RUN/watch.log"

  if remote_ready "$port" "$host" >/dev/null 2>&1; then
    printf '%s final artifacts ready; preparing immutable full snapshot\n' "$(timestamp)" >> "$LOCAL_RUN/watch.log"
    if ! prepare_final_manifest "$port" "$host" >> "$LOCAL_RUN/watch.log" 2>&1; then
      printf '%s remote final validation failed; pod intentionally left running\n' "$(timestamp)" >> "$LOCAL_RUN/watch.log"
      sleep "$POLL_SECONDS"
      continue
    fi
    if ! download_and_verify "$port" "$host" >> "$LOCAL_RUN/watch.log" 2>&1; then
      printf '%s final download/checksum failed; pod intentionally left running\n' "$(timestamp)" >> "$LOCAL_RUN/watch.log"
      sleep "$POLL_SECONDS"
      continue
    fi
    if [ "$KEEP_POD_AFTER_COMPLETE" = 1 ]; then
      touch "$LOCAL_RUN/COMPLETE"
      printf '%s all local artifacts verified; keeping pod %s for the video stage\n' \
        "$(timestamp)" "$POD_ID" >> "$LOCAL_RUN/watch.log"
      break
    fi
    printf '%s all local artifacts verified; terminating pod %s\n' "$(timestamp)" "$POD_ID" >> "$LOCAL_RUN/watch.log"
    if ! (cd "$REPO" && python3 scripts/runpod.py terminate "$POD_ID") >> "$LOCAL_RUN/watch.log" 2>&1; then
      printf '%s termination API failed; verified data preserved and pod left running\n' "$(timestamp)" >> "$LOCAL_RUN/watch.log"
      sleep "$POLL_SECONDS"
      continue
    fi
    touch "$LOCAL_RUN/COMPLETE"
    printf '%s collected, verified, and terminated %s\n' "$(timestamp)" "$POD_ID" >> "$LOCAL_RUN/watch.log"
    break
  fi
  sleep "$POLL_SECONDS"
done
