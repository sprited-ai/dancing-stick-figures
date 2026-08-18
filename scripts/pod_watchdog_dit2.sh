#!/bin/bash
# RunPod A100-80GB, 2026-08-18: DiT stage-2 pair (Seedance §4.1 recipe on 64² 8f, patch 2, shift 2, img_frac .1, i2v_frac .2):
#   b64i = --init from image model ib64L ; b64 = same recipe from scratch. 61k steps each. Restart ≤3x.
cd /root/stickdance; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--cache data/v1_cache --size 64 --fast --compile --val_every 500 --frames 8 --batch 8 --accum 2 --steps 61000 --sample_every 2000 --patch 2 --shift 2.0 --img_frac 0.1 --i2v_frac 0.2 --lr_final 0.05"
declare -A CMD N
CMD[b64i]="python -m train.video_dit_fm $COMMON --out runs/b64i --init runs/ib64L/ckpt.pt"
CMD[b64]="python -m train.video_dit_fm $COMMON --out runs/b64"
N[b64i]=0; N[b64]=0
log(){ echo "$(date '+%m-%d %H:%M') $*" >> /root/watchdog.log; }
running(){ pgrep -f "out runs/$1 " >/dev/null; }
finished(){ l=$(tail -1 runs/$1/log.txt 2>/dev/null | awk '{print $2}'); [ -n "$l" ] && [ "$l" -ge 61000 ]; }
ensure(){ r=$1; running $r && return; finished $r && return
  [ ${N[$r]} -ge 3 ] && { log "$r died 3x, giving up"; return; }
  N[$r]=$((N[$r]+1)); res=""; [ -f runs/$r/ckpt.pt ] && res="--resume runs/$r/ckpt.pt"
  log "$r start #${N[$r]} $res"; nohup ${CMD[$r]} $res >> runs_${r}.log 2>&1 & sleep 150; }
while true; do
  ensure b64i; ensure b64
  if ! running b64i && ! running b64; then log "all done"; touch /root/ALL_DONE; exit 0; fi
  sleep 300
done
