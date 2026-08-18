#!/bin/bash
# gin overnight 2026-08-17: image-first. Phases: img (ia64/ib64 30k) -> imgL (ia64L UNet min-SNR 100k, ib64L DiT p2 50k) -> vid (resume a64, then b64 fresh).
source ~/venvs/ardy/bin/activate; cd ~/dev/stickdance
COMMON="--cache data/v1_cache --size 64 --fast --compile --val_every 500"
declare -A CMD N DONE
CMD[ia64]="python -m train.video_ddpm  $COMMON --out runs/ia64 --frames 1 --batch 128 --steps 30000 --sample_every 1000"
CMD[ib64]="python -m train.video_dit_fm $COMMON --out runs/ib64 --frames 1 --batch 128 --steps 30000 --sample_every 1000 --shift 1.0"
CMD[ia64L]="python -m train.video_ddpm  $COMMON --out runs/ia64L --frames 1 --batch 128 --steps 100000 --sample_every 2000 --min_snr 5 --lr_final 0.02"
CMD[ib64L]="python -m train.video_dit_fm $COMMON --out runs/ib64L --frames 1 --batch 128 --steps 50000 --sample_every 2000 --patch 2 --lr_final 0.02"
CMD[a64]="python -m train.video_ddpm  $COMMON --out runs/a64 --frames 8 --batch 16 --steps 85000 --sample_every 2000"
CMD[b64]="python -m train.video_dit_fm $COMMON --out runs/b64 --frames 8 --batch 16 --steps 85000 --sample_every 2000 --shift 1.0 --img_frac 0.2"
DONE[ia64]=30000; DONE[ib64]=30000; DONE[ia64L]=100000; DONE[ib64L]=50000; DONE[a64]=85000; DONE[b64]=85000
for r in ia64 ib64 ia64L ib64L a64 b64; do N[$r]=0; done
log(){ echo "$(date '+%m-%d %H:%M') $*" >> /tmp/watchdog.log; }
running(){ pgrep -f "out runs/$1 " >/dev/null; }
laststep(){ tail -1 runs/$1/log.txt 2>/dev/null | awk '{print $2}'; }
finished(){ l=$(laststep $1); [ -n "$l" ] && [ "$l" -ge ${DONE[$1]} ]; }
ensure(){ r=$1                                   # start or resume $r unless finished / given up
  running $r && return; finished $r && return
  [ ${N[$r]} -ge 3 ] && { log "$r died 3x, giving up"; return; }
  N[$r]=$((N[$r]+1)); res=""; [ -f runs/$r/ckpt.pt ] && res="--resume runs/$r/ckpt.pt"
  log "$r start #${N[$r]} $res (last step $(laststep $r))"
  nohup ${CMD[$r]} $res >> runs_${r}.log 2>&1 &
  sleep 180; }
phase=img
while true; do
  if [ $phase = img ]; then
    ensure ia64; ensure ib64
    if ! running ia64 && ! running ib64; then log "image phase over -> long image phase"; phase=imgL; fi
  elif [ $phase = imgL ]; then                   # long/proper image runs: UNet min-SNR-5 100k, DiT patch-2 50k
    ensure ia64L; ensure ib64L
    if ! running ia64L && ! running ib64L; then log "long image phase over -> video phase"; phase=vid; fi
  else
    ensure a64
    if ! running a64; then ensure b64; fi     # b64 fresh once a64 is done (or gave up)
    if ! running a64 && ! running b64; then log "all done"; exit 0; fi
  fi
  sleep 300
done
