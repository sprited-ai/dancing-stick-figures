#!/usr/bin/env bash
set -euo pipefail

root=/data/dancing-stick-figure-paper
python=/home/gin/venvs/ardy/bin/python
run="$root/results/m6v3_start_aligned_h8_20k_s0"

while [[ ! -f "$run/COMPLETE" ]]; do
  sleep 15
done

run_eval() {
  local step=$1 out=$2 checkpoint="$run/ckpt_${1}.pt"
  if [[ -f "$out/COMPLETE" ]]; then
    echo "[$(date -Is)] already evaluated: $out"
    return
  fi
  mkdir -p "$out"
  cd "$root"
  PYTHONPATH=. "$python" -u -m eval.eval_m6 \
    --ckpt "$checkpoint" --cache cache/mini --out "$out" \
    --split test --n 64 --sensitivity-n 8 --steps 10 --cfg 2.0 \
    --seed 20260824 --comparison-block-frames 4 --device cuda \
    --evaluation-protocol "$root/configs/m6_evaluation_v1_h8_5k_20k.json" \
    2>&1 | tee "$out/eval.log"
  test -s "$out/metrics.json"
  find "$out" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > "$out/SHA256SUMS"
  sha256sum -c "$out/SHA256SUMS" >/dev/null
  touch "$out/COMPLETE"
}

# Evaluate every declared late milestone rather than assuming the latest is
# best; this exposes topology/motion trade-offs over training.
run_eval 005000 "$run/eval_n64_step005000"
run_eval 010000 "$run/eval_n64_step010000"
run_eval 015000 "$run/eval_n64_step015000"
run_eval 020000 "$run/eval_n64"
echo "[$(date -Is)] selected H8 5k/10k/15k/20k n=64 evaluations verified"
