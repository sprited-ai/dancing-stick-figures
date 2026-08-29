#!/usr/bin/env bash
set -euo pipefail

cd /data/dancing-stick-figure-paper
queue_pid="$(cat results/pixel30k_fresh_matched.pid)"
local_final="results/paper1_v03c_t2v64_local3d_image30k_rgba_s0/ckpt_030000.pt"
next_out="results/paper1_v03c_t2v64_factorised_image30k_rgba_s0"

while kill -0 "${queue_pid}" 2>/dev/null; do
  if [[ -f "${local_final}" ]]; then
    # The old queue may have entered its next command during this polling
    # interval. Stop that child and preserve any partial output before handing
    # control to the speed-prioritised remaining-model queue.
    next_child="$(pgrep -P "${queue_pid}" -f 'train.video_dit_fm' | head -n 1 || true)"
    if [[ -n "${next_child}" ]]; then
      kill -TERM "${next_child}" 2>/dev/null || true
    fi
    kill -TERM "${queue_pid}" 2>/dev/null || true
    sleep 2

    if [[ -d "${next_out}" && ! -f "${next_out}/ckpt_030000.pt" ]]; then
      mkdir -p results/aborted
      mv "${next_out}" "results/aborted/$(basename "${next_out}")_pre_fast_handoff"
    fi

    nohup bash scripts/run_pixel_remaining_fast.sh >> results/pixel30k_fresh_matched.log 2>&1 &
    printf '%s\n' "$!" > results/pixel30k_fresh_matched.pid
    exit 0
  fi
  sleep 5
done

echo "Original pixel queue exited before the Local mixer final checkpoint appeared" >&2
exit 1
