#!/bin/bash
# RunPod 4090, 2026-08-18: class-conditional 64² image models (5 prompt groups, CFG drop 0.1). ic64 UNet min-SNR 30k; id64 DiT p2 30k. Restart ≤3x.
cd /root/stickdance
COMMON="--cache data/v1_cache --size 64 --fast --compile --val_every 500 --frames 1 --batch 128 --steps 30000 --sample_every 2000 --lr_final 0.02 --cond group --cfg_drop 0.1"
declare -A CMD N
CMD[ic64]="python -m train.video_ddpm  $COMMON --out runs/ic64 --min_snr 5"
CMD[id64]="python -m train.video_dit_fm $COMMON --out runs/id64 --patch 2"
N[ic64]=0; N[id64]=0
log(){ echo "$(date '+%m-%d %H:%M') $*" >> /root/watchdog.log; }
running(){ pgrep -f "out runs/$1 " >/dev/null; }
finished(){ l=$(tail -1 runs/$1/log.txt 2>/dev/null | awk '{print $2}'); [ -n "$l" ] && [ "$l" -ge 30000 ]; }
ensure(){ r=$1; running $r && return; finished $r && return
  [ ${N[$r]} -ge 3 ] && { log "$r died 3x, giving up"; return; }
  N[$r]=$((N[$r]+1)); res=""; [ -f runs/$r/ckpt.pt ] && res="--resume runs/$r/ckpt.pt"
  log "$r start #${N[$r]} $res"; nohup ${CMD[$r]} $res >> runs_${r}.log 2>&1 & sleep 120; }
while true; do
  ensure ic64; ensure id64
  if ! running ic64 && ! running id64; then log "all done"; touch /root/ALL_DONE; exit 0; fi
  sleep 300
done
