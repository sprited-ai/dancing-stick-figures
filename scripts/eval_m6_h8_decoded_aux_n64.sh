#!/usr/bin/env bash
set -euo pipefail

root=/data/dancing-stick-figure-paper
python=/home/gin/venvs/ardy/bin/python

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
    --split test --n 64 --sensitivity-n 8 --steps 10 --cfg 2.0 \
    --seed 20260824 --comparison-block-frames 4 --device cuda \
    2>&1 | tee "$output/eval.log"
  test -s "$output/metrics.json"
  find "$output" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > "$output/SHA256SUMS"
  sha256sum -c "$output/SHA256SUMS" >/dev/null
}

run_eval \
  "$root/results/m6v3_start_aligned_h8_2k_s0/ckpt_002000.pt" \
  "$root/results/m6v3_start_aligned_h8_2k_s0/eval_n64"
run_eval \
  "$root/results/m6v4_start_aligned_h8_decoded_aux_2k_s0/ckpt_002000.pt" \
  "$root/results/m6v4_start_aligned_h8_decoded_aux_2k_s0/eval_n64"

echo "[$(date -Is)] matched H8 n=64 evaluations complete"
