"""Tail a run's log.txt into TensorBoard events (for runs started before TB logging existed).
    python -m train.log2tb runs/a0
"""
import re, sys, time, os
from torch.utils.tensorboard import SummaryWriter
run = sys.argv[1]; w = SummaryWriter(os.path.join(run, "tb")); seen = 0
pat = re.compile(r"step (\d+) loss ([\d.]+)(?: lr ([\d.e+-]+))? ([\d.]+)s/it(?: peak)? ([\d.]+)GB(?: ETA ([\d.]+)h)?(?: val ([\d.]+))?")
while True:
    lines = open(os.path.join(run, "log.txt")).read().splitlines()
    for ln in lines[seen:]:
        m = pat.search(ln)
        if not m: continue
        s = int(m.group(1)); w.add_scalar("loss/train_ema", float(m.group(2)), s)
        if m.group(3): w.add_scalar("lr", float(m.group(3)), s)
        w.add_scalar("perf/s_per_it", float(m.group(4)), s); w.add_scalar("perf/peak_gb", float(m.group(5)), s)
        if m.group(7): w.add_scalar("loss/val", float(m.group(7)), s)
    seen = len(lines); w.flush(); time.sleep(30)
