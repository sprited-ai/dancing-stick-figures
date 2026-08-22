#!/usr/bin/env bash
set -euo pipefail

remote="gin:/data/dancing-stick-figure-paper/results/m4_m3init_signed_rope_10k_20fps/"
local_dir="/Users/jin/dev/dancing-stick-figure/pod_results/m4_m3init_signed_rope_10k_20fps"
mkdir -p "$local_dir"

while true; do
  rsync -az --partial \
    --include='args.json' --include='launcher.log' --include='log.txt' --include='pid' \
    --include='sample_manifest_*.json' --include='*.gif' --include='*_strip.png' \
    --include='ckpt_*.pt' --exclude='*' "$remote" "$local_dir/" || true

  if ssh gin 'cd /data/dancing-stick-figure-paper/results/m4_m3init_signed_rope_10k_20fps && \
      grep -q "step 10000 " log.txt && ! kill -0 "$(cat pid)" 2>/dev/null'; then
    rsync -az --partial "${remote}latest.pt" "$local_dir/latest.pt"
    ssh gin 'cd /data/dancing-stick-figure-paper/results/m4_m3init_signed_rope_10k_20fps && \
      sha256sum args.json launcher.log log.txt latest.pt ckpt_*.pt sample_manifest_*.json *.gif *_strip.png' \
      > "$local_dir/SHA256SUMS"
    (cd "$local_dir" && shasum -a 256 -c SHA256SUMS)
    touch "$local_dir/COMPLETE"
    exit 0
  fi
  sleep 120
done
