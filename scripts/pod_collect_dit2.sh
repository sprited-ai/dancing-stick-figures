#!/bin/bash
# poll RunPod dit2 pod; pull samples/logs every 10 min into pod_results/dit2; on ALL_DONE (or 16h cap) collect + terminate.
cd /Users/jin/dev/dancing-stick-figure; POD=c7b7aplyx0v6ey; H="root@216.81.151.54"; S="ssh -o ConnectTimeout=30 -o BatchMode=yes -p 12846"; D=pod_results/dit2; mkdir -p $D
t0=$(date +%s)
pull(){ for r in b64i b64; do mkdir -p $D/$r; rsync -aq -e "$S" --include="*.png" --include="*.gif" --include='log.txt' --include='args.json' --include='*.json' --exclude='*.pt' --exclude='tb/' $H:/root/stickdance/runs/$r/ $D/$r/ 2>/dev/null; done
  rsync -aq -e "$S" $H:/root/watchdog.log $H:/root/stickdance/runs_b64i.log $H:/root/stickdance/runs_b64.log $D/ 2>/dev/null; }
while true; do
  pull; echo "$(date '+%m-%d %H:%M') pulled: $(tail -1 $D/ia128/log.txt 2>/dev/null | cut -c1-40) | $(tail -1 $D/ib128/log.txt 2>/dev/null | cut -c1-40)" >> $D/collect.log
  if $S $H 'test -f /root/ALL_DONE' 2>/dev/null || [ $(( $(date +%s) - t0 )) -gt $((20*3600)) ]; then
    pull; for r in b64i b64; do rsync -aq -e "$S" $H:/root/stickdance/runs/$r/ckpt.pt $D/$r/ 2>/dev/null; done   # final EMA+model ckpts (~0.5GB)
    python3 scripts/runpod.py terminate $POD >> $D/collect.log 2>&1; echo "$(date) collected+terminated" >> $D/collect.log; exit 0
  fi
  sleep 600
done
