#!/bin/bash
# stickdance v1 batch: every prompt x 3 seeds, 6s, core model. outputs/v1/<group>/<slug>_s<seed>.npz
source ~/venvs/ardy/bin/activate
cd ~/dev/ardy
mkdir -p outputs/v1
n=0
while IFS=$'\t' read -r group prompt; do
  [[ -z "$group" || "$group" == \#* ]] && continue
  g=${group#\*}
  slug=$(echo "$prompt" | tr -cd "a-zA-Z0-9 " | tr " A-Z" "_a-z" | cut -c1-60)
  mkdir -p outputs/v1/$g
  for seed in 0 1 2; do
    out=outputs/v1/$g/${slug}_s$seed
    [[ -f $out.npz ]] && continue
    python scripts/generate.py "$prompt" --model core --duration 6.0 --seed $seed --output $out >> /tmp/ardy_batch_v1.log 2>&1 || echo "FAIL $prompt $seed" >> /tmp/ardy_batch_v1.fail
    n=$((n+1))
  done
done < prompts_v1.txt
echo "DONE $n" >> /tmp/ardy_batch_v1.log
