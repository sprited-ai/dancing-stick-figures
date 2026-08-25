#!/usr/bin/env bash
set -euo pipefail

cd "${DSF_ROOT:-/workspace/dsf}"

sre="runs/v02/sre_confidence_10k/ckpt_final.pt"
suite="runs/v02/full64_native120_prompt_suite"
out="runs/v02/sre_diagnostics"

while [[ ! -s "$sre" || ! -s "$suite/manifest.json" ]]; do
  sleep 20
done
for name in fixed_noise_varied_prompt fixed_prompt_varied_noise prompt_diverse_grid; do
  while [[ ! -s "$suite/${name}_rgba.npz" ]]; do sleep 20; done
done

mkdir -p "$out"
python -m eval.sre_score \
  --cache data/cache_all --split test --n 64 --frames 120 \
  --ckpt "$sre" --out "$out/real_test_n64.json"

for name in fixed_noise_varied_prompt fixed_prompt_varied_noise prompt_diverse_grid; do
  python -m eval.sre_score \
    --input "$suite/${name}_rgba.npz" \
    --ckpt "$sre" --out "$out/${name}.json"
done

python - "$out" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
paths = [root / "real_test_n64.json"] + [
    root / f"{name}.json" for name in (
        "fixed_noise_varied_prompt", "fixed_prompt_varied_noise", "prompt_diverse_grid"
    )
]
for path in paths:
    report = json.load(open(path))
    expected = "sre_diagnostic_vector_v1"
    if report.get("protocol") != expected:
        raise SystemExit(f"{path}: expected {expected}")
    if int(report["shape"][1]) != 120:
        raise SystemExit(f"{path}: expected 120 frames")
print("V02_SRE_DIAGNOSTICS_COMPLETE=1", flush=True)
PY
