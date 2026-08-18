#!/bin/bash
# RunPod H100, 2026-08-18: a64AR — chunked autoregressive video UNet 64², K=8 ctx + 8 new frames, stride 2 (10 fps), init from ia64L, 60k. Restart ≤3x.
cd /root/stickdance; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A CMD N
CMD[a64AR]="python -m train.video_ddpm --cache data/v1_final_cache --out runs/a64AR --size 64 --frames 8 --ar_ctx 8 --stride 2 --batch 16 --steps 60000 --sample_every 2000 --val_every 500 --rollout 6 --fast --compile --lr_final 0.05 --init runs/ia64L/ckpt.pt"
N[a64AR]=0
log(){ echo "$(date '+%m-%d %H:%M') $*" >> /root/watchdog.log; }
running(){ pgrep -f "out runs/$1( |$)" >/dev/null; }
finished(){ l=$(tail -1 runs/$1/log.txt 2>/dev/null | awk '{print $2}'); [ -n "$l" ] && [ "$l" -ge 60000 ]; }
settled(){ finished $1 || [ ${N[$1]} -ge 3 ]; }
ensure(){ r=$1; running $r && return; finished $r && return
  [ ${N[$r]} -ge 3 ] && { log "$r died 3x, giving up"; return; }
  N[$r]=$((N[$r]+1)); res=""; [ -f runs/$r/ckpt.pt ] && res="--resume runs/$r/ckpt.pt"
  [ -n "$res" ] && CMD[$r]="${CMD[$r]/--init runs\/ia64L\/ckpt.pt/}"
  log "$r start #${N[$r]} $res"; nohup ${CMD[$r]} $res >> runs_${r}.log 2>&1 & sleep 150; }
while true; do
  ensure a64AR
  if settled a64AR; then log "all done"; touch /root/ALL_DONE; exit 0; fi
  sleep 300
done
