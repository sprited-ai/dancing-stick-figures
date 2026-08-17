"""Export ARDY clips for the web viewer: viewer/public/clips/index.json + <id>.bin (float32 [T,27,3]).

    python -m generator.export_web --npz ardy_out/v1 --out viewer/public/clips

Joints are in each clip's frame-0 figure frame (x left, y up, z fwd) with real world root motion,
feet-on-floor at frame 0 — same as whiteroom.py.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from .ardy_adapter import load
from .whiteroom import world_to_fig0
from .skeleton import NAMES, PARENT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="ardy_out/v1"); ap.add_argument("--out", default="viewer/public/clips")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    index = []
    for f in sorted(glob.glob(os.path.join(a.npz, "*", "*.npz"))):
        group = os.path.basename(os.path.dirname(f))
        stem = os.path.splitext(os.path.basename(f))[0]
        P, fps, text = load(f)
        Q = world_to_fig0(P).astype(np.float32)
        cid = f"{group}__{stem}"
        Q.tofile(os.path.join(a.out, cid + ".bin"))
        index.append(dict(id=cid, group=group, text=text, seed=int(stem.rsplit("_s", 1)[1]) if "_s" in stem else 0,
                          frames=int(Q.shape[0]), fps=fps))
    json.dump(dict(joints=NAMES, parents=[PARENT[n] for n in NAMES], clips=index), open(os.path.join(a.out, "index.json"), "w"))
    print(len(index), "clips ->", a.out)


if __name__ == "__main__":
    main()
