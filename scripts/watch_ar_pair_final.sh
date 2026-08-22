#!/bin/bash
# Collect the two matched AR seed-1 runs only after each has completed step 60,000.
#
# The watcher treats the pods independently: a completed run is snapshotted,
# checksummed, downloaded, verified, and only then is that pod terminated.  A
# failed transfer or checksum deliberately leaves the pod running.

set -uo pipefail

REPO="/Users/jin/dev/dancing-stick-figure"
POLL_SECONDS="${POLL_SECONDS:-120}"
FINAL_STEP=60000
DRY_RUN=0

if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
elif [ -n "${1:-}" ]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

# Output is: PORT HOST. Resolve it every poll so a pod network remap does not
# strand a day-long watcher on stale connection details.
resolve_endpoint() {
  local pod_id="$1" line port destination host
  line=$(cd "$REPO" && python3 scripts/runpod.py ssh "$pod_id" 2>/dev/null) || return 1
  port=$(printf '%s\n' "$line" | awk '{print $3}')
  destination=$(printf '%s\n' "$line" | awk '{print $4}')
  host=${destination#root@}
  case "$port" in ''|*[!0-9]*) return 1 ;; esac
  [ -n "$host" ] && [ "$host" != "None" ] || return 1
  printf '%s %s\n' "$port" "$host"
}

ssh_options() {
  local port="$1"
  printf '%s\n' \
    -o BatchMode=yes \
    -o ConnectTimeout=15 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=no \
    -p "$port"
}

# Exit 0 only after all final artifacts exist, the checkpoint itself says
# step 60000, and the trainer is no longer running.  Loading the large
# checkpoint is intentionally deferred until the cheap file/log checks pass.
remote_ready() {
  local port="$1" host="$2" remote_run="$3" run_name="$4"
  local -a opts
  while IFS= read -r value; do opts+=("$value"); done < <(ssh_options "$port")
  ssh "${opts[@]}" "root@$host" bash -s -- "$remote_run" "$run_name" "$FINAL_STEP" <<'REMOTE_READY'
set -uo pipefail
run_dir="$1"
run_name="$2"
final_step="$3"
cd "$run_dir" || exit 1
test -s args.json -a -s log.txt -a -s launcher.log -a -s ckpt.pt || exit 1
grep -q "^step ${final_step} " log.txt || exit 1
test -s "sample_${final_step}.gif" -a -s "rollout_${final_step}.gif" || exit 1
if ps -eo comm=,args= | awk -v run="$run_name" \
  '$1 ~ /python/ && $0 ~ /train\.video_ddpm/ && index($0, "--out runs/" run) { found=1 } END { exit(found ? 0 : 1) }'
then
  exit 1
fi
python - "$final_step" <<'PY'
import sys, torch
checkpoint = torch.load("ckpt.pt", map_location="cpu", weights_only=False)
raise SystemExit(0 if int(checkpoint.get("step", -1)) == int(sys.argv[1]) else 1)
PY
REMOTE_READY
}

remote_tail() {
  local port="$1" host="$2" remote_run="$3"
  local -a opts
  while IFS= read -r value; do opts+=("$value"); done < <(ssh_options "$port")
  ssh "${opts[@]}" "root@$host" "tail -1 '$remote_run/log.txt' 2>/dev/null" 2>/dev/null || true
}

# Make an immutable full checkpoint snapshot, validate it, then checksum the
# exact set of files that will be copied.  This operation is idempotent.
prepare_remote_snapshot() {
  local port="$1" host="$2" remote_run="$3"
  local -a opts
  while IFS= read -r value; do opts+=("$value"); done < <(ssh_options "$port")
  ssh "${opts[@]}" "root@$host" bash -s -- "$remote_run" "$FINAL_STEP" <<'REMOTE_SNAPSHOT'
set -euo pipefail
run_dir="$1"
final_step="$2"
snapshot=$(printf 'ckpt_%06d.pt' "$final_step")
cd "$run_dir"

if [ ! -s "$snapshot" ]; then
  cp --reflink=auto ckpt.pt "${snapshot}.tmp"
  mv "${snapshot}.tmp" "$snapshot"
fi

python - "$snapshot" "$final_step" <<'PY'
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if int(checkpoint.get("step", -1)) != int(sys.argv[2]):
    raise SystemExit("snapshot step mismatch")
required = {"model", "ema", "opt", "args", "step"}
missing = sorted(required.difference(checkpoint))
if missing:
    raise SystemExit(f"full checkpoint missing keys: {missing}")
PY

files=(args.json log.txt launcher.log "$snapshot")
for pattern in 'sample_*.gif' 'sample_raw_*.gif' 'rollout_*.gif'; do
  for file in $pattern; do
    [ -f "$file" ] && files+=("$file")
  done
done
sha256sum "${files[@]}" > SHA256SUMS_FINAL.tmp
mv SHA256SUMS_FINAL.tmp SHA256SUMS_FINAL
REMOTE_SNAPSHOT
}

download_and_verify() {
  local port="$1" host="$2" remote_run="$3" local_run="$4"
  local rsync_ssh file
  rsync_ssh="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no -p $port"
  mkdir -p "$local_run"

  rsync -a --partial --append-verify -e "$rsync_ssh" \
    "root@$host:$remote_run/SHA256SUMS_FINAL" "$local_run/" || return 1

  while read -r _hash file; do
    case "$file" in
      ''|*/*|.|..) echo "unsafe manifest entry: $file" >&2; return 1 ;;
    esac
    rsync -a --partial --append-verify -e "$rsync_ssh" \
      "root@$host:$remote_run/$file" "$local_run/" || return 1
  done < "$local_run/SHA256SUMS_FINAL"

  (cd "$local_run" && shasum -a 256 -c SHA256SUMS_FINAL) || return 1
  test -s "$local_run/ckpt_060000.pt" || return 1
}

collect_one() {
  local pod_id="$1" run_name="$2" remote_run="$3" local_run="$4" port="$5" host="$6"
  local watch_log="$local_run/watch.log"
  mkdir -p "$local_run"
  printf '%s final artifacts detected; preparing remote snapshot\n' "$(timestamp)" >> "$watch_log"

  if ! prepare_remote_snapshot "$port" "$host" "$remote_run" >> "$watch_log" 2>&1; then
    printf '%s remote snapshot/checksum failed; pod intentionally left running\n' "$(timestamp)" >> "$watch_log"
    return 1
  fi
  if ! download_and_verify "$port" "$host" "$remote_run" "$local_run" >> "$watch_log" 2>&1; then
    printf '%s download/checksum failed; pod intentionally left running\n' "$(timestamp)" >> "$watch_log"
    return 1
  fi

  printf '%s local checksum verified; terminating completed pod %s\n' "$(timestamp)" "$pod_id" >> "$watch_log"
  if ! (cd "$REPO" && python3 scripts/runpod.py terminate "$pod_id") >> "$watch_log" 2>&1; then
    printf '%s termination call failed; verification is preserved for retry\n' "$(timestamp)" >> "$watch_log"
    return 1
  fi
  printf '%s collected, verified, and terminated %s\n' "$(timestamp)" "$pod_id" >> "$watch_log"
  touch "$local_run/COMPLETE"
}

check_dry_run() {
  local pod_id="$1" run_name="$2" remote_run="$3" local_run="$4"
  local endpoint port host tail_line
  if ! endpoint=$(resolve_endpoint "$pod_id"); then
    printf '%s %-10s endpoint unavailable\n' "$(timestamp)" "$run_name"
    return
  fi
  read -r port host <<< "$endpoint"
  tail_line=$(remote_tail "$port" "$host" "$remote_run")
  if remote_ready "$port" "$host" "$remote_run" "$run_name" >/dev/null 2>&1; then
    printf '%s %-10s READY; would collect to %s and then terminate %s\n' "$(timestamp)" "$run_name" "$local_run" "$pod_id"
  else
    printf '%s %-10s not ready; latest: %s\n' "$(timestamp)" "$run_name" "${tail_line:-unavailable}"
  fi
}

SPECS=(
  "ikbee4rtdw34o0|ar_scratch_s1|/workspace/dsf/runs/ar_scratch_s1|$REPO/pod_results/ar_scratch_s1_final"
  "tixjycpzprhmsh|ar_pre_s1|/workspace/dsf/runs/ar_pre_s1|$REPO/pod_results/ar_pre_s1_final"
)

if [ "$DRY_RUN" -eq 1 ]; then
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r pod_id run_name remote_run local_run <<< "$spec"
    check_dry_run "$pod_id" "$run_name" "$remote_run" "$local_run"
  done
  exit 0
fi

for spec in "${SPECS[@]}"; do
  IFS='|' read -r _pod_id _run_name _remote_run local_run <<< "$spec"
  mkdir -p "$local_run"
  printf '%s watching for final step %d\n' "$(timestamp)" "$FINAL_STEP" >> "$local_run/watch.log"
done

while true; do
  remaining=0
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r pod_id run_name remote_run local_run <<< "$spec"
    [ -f "$local_run/COMPLETE" ] && continue
    remaining=$((remaining + 1))

    endpoint=$(resolve_endpoint "$pod_id" 2>/dev/null || true)
    if [ -z "$endpoint" ]; then
      printf '%s endpoint unavailable; will retry\n' "$(timestamp)" >> "$local_run/watch.log"
      continue
    fi
    read -r port host <<< "$endpoint"

    if remote_ready "$port" "$host" "$remote_run" "$run_name" >/dev/null 2>&1; then
      collect_one "$pod_id" "$run_name" "$remote_run" "$local_run" "$port" "$host" || true
    else
      latest=$(remote_tail "$port" "$host" "$remote_run")
      printf '%s %s\n' "$(timestamp)" "${latest:-not ready}" >> "$local_run/watch.log"
    fi
  done

  [ "$remaining" -eq 0 ] && break
  sleep "$POLL_SECONDS"
done

printf '%s all AR runs collected and verified\n' "$(timestamp)"
