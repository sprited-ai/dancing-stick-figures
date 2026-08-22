#!/usr/bin/env bash
set -euo pipefail

root=/data/dancing-stick-figure-paper
python=/home/gin/venvs/ardy/bin/python
pilot="$root/results/m6v4_start_aligned_h8_decoded_aux_2k_s0"

while [[ ! -f "$pilot/COMPLETE" ]]; do
  sleep 10
done

run_eval() {
  local checkpoint=$1 output=$2
  if [[ -f "$output/metrics.json" ]]; then
    echo "[$(date -Is)] already evaluated: $output"
    return
  fi
  mkdir -p "$output"
  cd "$root"
  PYTHONPATH=. "$python" -u -m eval.eval_m6 \
    --ckpt "$checkpoint" --cache cache/mini --out "$output" \
    --split test --n 4 --sensitivity-n 4 --steps 10 --cfg 2.0 \
    --seed 20260824 --comparison-block-frames 4 --device cuda \
    2>&1 | tee "$output/eval.log"
  test -s "$output/metrics.json"
  find "$output" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > "$output/SHA256SUMS"
}

# Finish the one missing horizon arm before comparing the corrective loss.
run_eval \
  "$root/results/m6v3_start_aligned_h40_2k_s0/ckpt_002000.pt" \
  "$root/results/m6v3_start_aligned_h40_2k_s0/eval_n4_v2"
run_eval \
  "$pilot/ckpt_002000.pt" \
  "$pilot/eval_n4"

echo "[$(date -Is)] horizon and decoded-aux evaluations complete"
