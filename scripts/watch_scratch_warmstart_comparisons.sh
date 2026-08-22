#!/bin/bash
# Build controlled side-by-side GIFs as soon as both canonical arms reach the
# same milestone.  This is read-only with respect to training runs.
set -euo pipefail

REPO=/Users/jin/dev/dancing-stick-figure
SCRATCH="$REPO/pod_results/k1_canonical_s2_scratch_mix3k"
WARM="$REPO/pod_results/t2v50_s2_text_p4_fg2_img10_i2v20_from_t2i30k_10k"
OUT="$REPO/output/comparisons/scratch_vs_t2i_warmstart"
mkdir -p "$OUT"

for step in 500 1000 3000; do
  tag=$(printf '%06d' "$step")
  target="$OUT/step_$tag"
  while [ ! -f "$target/manifest.json" ]; do
    sg="$SCRATCH/sample_$tag.gif"; wg="$WARM/sample_$tag.gif"
    sm="$SCRATCH/sample_manifest_$tag.json"; wm="$WARM/sample_manifest_$tag.json"
    if [ -s "$sg" ] && [ -s "$wg" ] && [ -s "$sm" ] && [ -s "$wm" ]; then
      python3 - "$sm" "$wm" <<'PY'
import json, sys
a, b = (json.load(open(path)) for path in sys.argv[1:])
for key in ('seed', 'prompts', 'nfe', 'cfg', 'shift'):
    if a.get(key) != b.get(key):
        raise SystemExit(f'inference manifest mismatch for {key}: {a.get(key)!r} vs {b.get(key)!r}')
PY
      python3 "$REPO/scripts/compare_video_milestones.py" \
        --input "scratch step $step=$sg" \
        --input "T2I warm-start step $step=$wg" \
        --out "$target"
      break
    fi
    sleep 60
  done
done
touch "$OUT/COMPLETE"
