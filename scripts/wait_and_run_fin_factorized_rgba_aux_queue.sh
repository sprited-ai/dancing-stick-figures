#!/usr/bin/env bash
set -euo pipefail

cd /home/fin/dancing-stick-figures

expected_frames_bytes=7903641728
while [[ $(stat -c %s cache/mini_v02/frames.npy 2>/dev/null || echo 0) -ne $expected_frames_bytes ]]; do
  sleep 30
done

exec bash scripts/run_fin_factorized_rgba_aux_queue.sh
