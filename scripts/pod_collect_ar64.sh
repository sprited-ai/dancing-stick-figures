#!/bin/bash
# poll RunPod ar64 pod; pull samples/logs every 10 min into pod_results/ar64; on ALL_DONE (or 16h cap) collect + terminate.
cd /Users/jin/dev/dancing-stick-figure; POD=s7eh8ytjg7l8xd; H="root@31.24.80.34"; S="ssh -o ConnectTimeout=30 -o BatchMode=yes -p 10860"; D=pod_results/ar64; mkdir -p $D
t0=$(date +%s)
pull(){ for r in a64AR; do mkdir -p $D/$r; rsync -aq -e "$S" --include="*.png" --include="*.gif" --include='log.txt' --include='args.json' --include='*.json' --exclude='*.pt' --exclude='tb/' $H:/root/stickdance/runs/$r/ $D/$r/ 2>/dev/null; done
  rsync -aq -e "$S" $H:/root/watchdog.log $H:/root/stickdance/runs_a64AR.log $D/ 2>/dev/null; }
while true; do
  pull; echo "$(date '+%m-%d %H:%M') pulled: $(tail -1 $D/ia128/log.txt 2>/dev/null | cut -c1-40) | $(tail -1 $D/ib128/log.txt 2>/dev/null | cut -c1-40)" >> $D/collect.log
  if $S $H 'test -f /root/ALL_DONE' 2>/dev/null || [ $(( $(date +%s) - t0 )) -gt $((12*3600)) ]; then
    pull; for r in a64AR; do rsync -aq -e "$S" $H:/root/stickdance/runs/$r/ckpt.pt $D/$r/ 2>/dev/null; done   # final EMA+model ckpts (~0.5GB)
    python3 scripts/runpod.py terminate $POD >> $D/collect.log 2>&1; echo "$(date) collected+terminated" >> $D/collect.log; exit 0
  fi
  sleep 600
done
