#!/usr/bin/env bash
set -euo pipefail

POD_ID="${POD_ID:-0bnxh7iq28f4mj}"
SSH_HOST="${SSH_HOST:-root@103.196.86.68}"
SSH_PORT="${SSH_PORT:-34667}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/output/runpod_dit50_matched10k"
mkdir -p "$OUT"

while ! ssh -p "$SSH_PORT" "$SSH_HOST" \
  'test -f /workspace/dsf/runs/dit50_local3d_matched10k/EVAL_COMPLETE'; do
  sleep 60
done

for RUN in dit50_factorized_matched10k dit50_local3d_matched10k; do
  mkdir -p "$OUT/$RUN"
  scp -P "$SSH_PORT" "$SSH_HOST:/workspace/dsf/runs/$RUN/args.json" "$OUT/$RUN/"
  scp -P "$SSH_PORT" "$SSH_HOST:/workspace/dsf/runs/$RUN/log.txt" "$OUT/$RUN/"
  scp -P "$SSH_PORT" "$SSH_HOST:/workspace/dsf/runs/$RUN/ckpt_010000.pt" "$OUT/$RUN/"
  scp -P "$SSH_PORT" "$SSH_HOST:/workspace/dsf/runs/$RUN/sample_*.gif" "$OUT/$RUN/"
  scp -P "$SSH_PORT" -r "$SSH_HOST:/workspace/dsf/runs/$RUN/eval" "$OUT/$RUN/"
  scp -P "$SSH_PORT" -r "$SSH_HOST:/workspace/dsf/runs/$RUN/prompt_suite" "$OUT/$RUN/"
done
scp -P "$SSH_PORT" "$SSH_HOST:/workspace/dsf/runs/dit50_matched10k_reference_manifest.json" "$OUT/"
touch "$OUT/COLLECTED"
python3 "$ROOT/scripts/runpod.py" terminate "$POD_ID"
