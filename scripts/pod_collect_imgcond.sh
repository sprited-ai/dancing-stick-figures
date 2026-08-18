#!/bin/bash
# poll RunPod imgcond pod; pull samples/logs every 10 min into pod_results/imgcond; on ALL_DONE (or 16h cap) collect + terminate.
cd /Users/jin/dev/dancing-stick-figure; POD=k203e8o9tfm37c; H="root@69.145.85.94"; S="ssh -o ConnectTimeout=30 -o BatchMode=yes -p 14101"; D=pod_results/imgcond; mkdir -p $D
t0=$(date +%s)
pull(){ for r in ic64 id64; do mkdir -p $D/$r; rsync -aq -e "$S" --include='*.png' --include='log.txt' --include='args.json' --include='*.json' --exclude='*.pt' --exclude='tb/' $H:/root/stickdance/runs/$r/ $D/$r/ 2>/dev/null; done
  rsync -aq -e "$S" $H:/root/watchdog.log $H:/root/stickdance/runs_ic64.log $H:/root/stickdance/runs_id64.log $D/ 2>/dev/null; }
while true; do
  pull; echo "$(date '+%m-%d %H:%M') pulled: $(tail -1 $D/ia128/log.txt 2>/dev/null | cut -c1-40) | $(tail -1 $D/ib128/log.txt 2>/dev/null | cut -c1-40)" >> $D/collect.log
  if $S $H 'test -f /root/ALL_DONE' 2>/dev/null || [ $(( $(date +%s) - t0 )) -gt $((8*3600)) ]; then
    pull; for r in ic64 id64; do rsync -aq -e "$S" $H:/root/stickdance/runs/$r/ckpt.pt $D/$r/ 2>/dev/null; done   # final EMA+model ckpts (~0.5GB)
    python3 scripts/runpod.py terminate $POD >> $D/collect.log 2>&1; echo "$(date) collected+terminated" >> $D/collect.log; exit 0
  fi
  sleep 600
done
