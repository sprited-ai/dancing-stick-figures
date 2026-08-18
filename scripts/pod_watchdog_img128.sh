#!/bin/bash
# RunPod A100-80GB, 2026-08-17 night: 128² image models ia128 (UNet min-SNR) + ib128 (DiT p4), 40k steps each. Restart ≤3x. Log /root/watchdog.log
cd /root/stickdance
COMMON="--cache data/v1_cache --size 128 --fast --compile --val_every 500 --frames 1 --batch 128 --steps 40000 --sample_every 2000 --lr_final 0.02"
declare -A CMD N
CMD[ia128]="python -m train.video_ddpm  $COMMON --out runs/ia128 --min_snr 5 --steps 20000"
CMD[ib128]="python -m train.video_dit_fm $COMMON --out runs/ib128 --patch 4"
N[ia128]=0; N[ib128]=0
log(){ echo "$(date '+%m-%d %H:%M') $*" >> /root/watchdog.log; }
running(){ pgrep -f "out runs/$1 " >/dev/null; }
finished(){ l=$(tail -1 runs/$1/log.txt 2>/dev/null | awk "{print \$2}"); d=40000; [ $1 = ia128 ] && d=20000; [ -n "$l" ] && [ "$l" -ge $d ]; }
ensure(){ r=$1; running $r && return; finished $r && return
  [ ${N[$r]} -ge 3 ] && { log "$r died 3x, giving up"; return; }
  N[$r]=$((N[$r]+1)); res=""; [ -f runs/$r/ckpt.pt ] && res="--resume runs/$r/ckpt.pt"
  log "$r start #${N[$r]} $res"; nohup ${CMD[$r]} $res >> runs_${r}.log 2>&1 & sleep 120; }
while true; do
  ensure ia128; ensure ib128
  if ! running ia128 && ! running ib128; then log "all done"; touch /root/ALL_DONE; exit 0; fi
  sleep 300
done
