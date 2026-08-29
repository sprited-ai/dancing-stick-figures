#!/usr/bin/env bash
set -euo pipefail

# Canonical 64-frame evaluation queue for the matched RGBA-auxiliary pixel
# references.  This waits for fin's active training queue and never shares the
# GPU with training.

repo=/home/fin/dancing-stick-figures
python=/home/fin/venvs/ardy/bin/python
cache=cache/mini_v02
manifest=results/v02c_eval/win64_manifest.json
training_done=results/FIN_FACTORISED_RGBA_AUX_QUEUE_COMPLETE

cd "$repo"

while [[ ! -e "$training_done" ]]; do
  sleep 60
done

run_eval() {
  local run=$1
  local final_json="$run/eval/030000.json"
  if [[ -s "$final_json" ]]; then
    "$python" - "$final_json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["step"] == 30000
assert d["n"] == 128
assert d["sampling_seeds"] == [0, 1, 2]
assert d["target_frames"] == 64
assert d["reference_stride"] == 1
assert d["sample_steps"] == 50
PY
    echo "SKIP $run: verified canonical 30k evaluation already exists"
    return
  fi

  "$python" -m eval.run_ckpt \
    --run "$run" \
    --cache "$cache" \
    --n 128 \
    --seeds 3 \
    --frames 64 \
    --stride 1 \
    --sample_steps 50 \
    --batch 4 \
    --cfg 3 \
    --manifest "$manifest" \
    2>&1 | tee -a "$run/canonical_eval_64f_n128.log"
}

run_eval results/explore_factorised_rgba_aux_image30k_fin_s0
run_eval results/explore_factorised_rgba_aux_random30k_fin_s0
run_eval results/eval_import_local3d_image30k_rgba_s0

touch results/FIN_MATCHED_CANONICAL_EVAL_QUEUE_COMPLETE
