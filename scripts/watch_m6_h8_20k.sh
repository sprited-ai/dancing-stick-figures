#!/usr/bin/env bash
set -euo pipefail

remote=gin:/data/dancing-stick-figure-paper/results/m6v3_start_aligned_h8_20k_s0/
remote_path=/data/dancing-stick-figure-paper/results/m6v3_start_aligned_h8_20k_s0
local_dir=/Users/jin/dev/dancing-stick-figure/pod_results/m6v3_start_aligned_h8_20k_s0
mkdir -p "$local_dir"

sync_light() {
  rsync -az --partial \
    --include='args.json' --include='log.txt' --include='run_manifest.json' \
    --include='sample_manifest_*.json' --include='*.gif' --include='*_strip.png' \
    --include='source_*.py' --include='source_protocol.json' \
    --include='COMPLETE' --exclude='*' "$remote" "$local_dir/" || true
}

while true; do
  sync_light
  if ssh gin "test -f '$remote_path/COMPLETE' && test -f '$remote_path/eval_n64/COMPLETE'"; then
    ssh gin "sha256sum -c '$remote_path/SHA256SUMS' >/dev/null && sha256sum -c '$remote_path/eval_n64/SHA256SUMS' >/dev/null"
    rsync -az --partial \
      --include='ckpt_*.pt' --include='latest.pt' --include='SHA256SUMS' \
      --include='eval_n64/***' --include='eval_n64_step*/***' --exclude='*' "$remote" "$local_dir/"
    sync_light
    test -s "$local_dir/ckpt_020000.pt"
    test -s "$local_dir/latest.pt"
    test -s "$local_dir/eval_n64/metrics.json"
    PYTHONPATH=/Users/jin/dev/dancing-stick-figure python \
      /Users/jin/dev/dancing-stick-figure/scripts/compare_video_milestones.py \
      --input "0=$local_dir/fixed_prompt_000000_labeled.gif" \
      --input "100=$local_dir/fixed_prompt_000100_labeled.gif" \
      --input "250=$local_dir/fixed_prompt_000250_labeled.gif" \
      --input "500=$local_dir/fixed_prompt_000500_labeled.gif" \
      --input "1k=$local_dir/fixed_prompt_001000_labeled.gif" \
      --input "2k=$local_dir/fixed_prompt_002000_labeled.gif" \
      --input "5k=$local_dir/fixed_prompt_005000_labeled.gif" \
      --input "10k=$local_dir/fixed_prompt_010000_labeled.gif" \
      --input "15k=$local_dir/fixed_prompt_015000_labeled.gif" \
      --input "20k=$local_dir/fixed_prompt_020000_labeled.gif" \
      --out "$local_dir/progression"
    touch "$local_dir/VERIFIED_COMPLETE"
    exit 0
  fi
  sleep 120
done
