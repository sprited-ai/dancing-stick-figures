#!/usr/bin/env bash
# Generate the public source-motion recipe: 143 prompts x seeds 0..9 x 6 s.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dsf_root=$(cd "$script_dir/.." && pwd)
ardy_root=${ARDY_ROOT:-"$HOME/dev/ardy"}
python_bin=${ARDY_PYTHON:-python}
model_alias=${ARDY_MODEL:-core}
prompt_file="$dsf_root/prompts/v1.txt"
output_root=${ARDY_OUT:-"$dsf_root/ardy_out/v1"}
log_file="$output_root/generation.log"
manifest_file="$output_root/generation_manifest.json"

if [[ ! -f "$ardy_root/scripts/generate.py" ]]; then
  echo "ARDY_ROOT does not contain scripts/generate.py: $ardy_root" >&2
  exit 2
fi

mkdir -p "$output_root"
ardy_git_revision=$(git -C "$ardy_root" rev-parse HEAD 2>/dev/null || printf unknown)

model_record=$(
  cd "$ardy_root"
  "$python_bin" - "$model_alias" <<'PY'
import json
import sys

from ardy.model.registry import hf_repo_id, resolve_model_name
from huggingface_hub import model_info

resolved = resolve_model_name(sys.argv[1])
repo = hf_repo_id(resolved)
info = model_info(repo)
print(json.dumps({
    "alias": sys.argv[1],
    "resolved_name": resolved,
    "huggingface_repo": repo,
    "huggingface_revision": info.sha,
}))
PY
)

started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
generated=0
skipped=0

while IFS=$'\t' read -r group prompt; do
  [[ -z "$group" || "$group" == \#* ]] && continue
  group=${group#\*}
  slug=$(printf '%s' "$prompt" | tr -cd 'a-zA-Z0-9 ' | tr ' A-Z' '_a-z' | cut -c1-60)
  mkdir -p "$output_root/$group"
  for seed in {0..9}; do
    destination="$output_root/$group/${slug}_s${seed}.npz"
    if [[ -s "$destination" ]]; then
      skipped=$((skipped + 1))
      continue
    fi
    (
      cd "$ardy_root"
      "$python_bin" scripts/generate.py "$prompt" \
        --model "$model_alias" --duration 6.0 --seed "$seed" --output "$destination"
    ) >>"$log_file" 2>&1
    generated=$((generated + 1))
    printf 'generated %d/1430: %s seed %d\n' "$((generated + skipped))" "$prompt" "$seed"
  done
done < "$prompt_file"

processed=$((generated + skipped))
if [[ "$processed" -ne 1430 ]]; then
  echo "expected 1430 prompt-seed outputs, processed $processed" >&2
  exit 1
fi

finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
"$python_bin" - "$manifest_file" "$started_utc" "$finished_utc" "$ardy_git_revision" \
  "$model_record" "$generated" "$skipped" <<'PY'
import json
import sys

path, started, finished, ardy_revision, model_json, generated, skipped = sys.argv[1:]
manifest = {
    "protocol": "dancing_stick_figures_source_motion_v1",
    "started_utc": started,
    "finished_utc": finished,
    "prompts": 143,
    "seeds": list(range(10)),
    "duration_seconds": 6.0,
    "expected_motion_files": 1430,
    "generated_this_run": int(generated),
    "reused_existing": int(skipped),
    "ardy_source_revision": ardy_revision,
    "ardy_model": json.loads(model_json),
}
with open(path, "w") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")
print(json.dumps(manifest, indent=2))
PY
