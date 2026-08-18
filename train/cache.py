"""Decode the parquet colour frames once into a uint8 memmap for fast video sampling.

    python -m train.cache --data <dataset>/frames --out cache          # <dataset> = `hf download sprited/dancing-stick-figures --repo-type dataset`

Writes frames.npy [N,S,S,4] (straight RGBA as stored, S inferred from the first image), clips.json
(clip_id -> start, n, split, group, text, qa). Streams one shard at a time (a few hundred MB of RAM), ~2 min for
the full 514,800-frame set on 8 threads. Premultiplication happens in the loader.
"""
from __future__ import annotations
import argparse, glob, io, json, os
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory with *.parquet shards (e.g. <hf-download>/frames)")
    ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--splits", default="", help="comma list to keep (default: all), e.g. train,val")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.data, "*.parquet")))
    if not files: raise SystemExit(f"no parquet files in {a.data}")
    keep = set(a.splits.split(",")) if a.splits else None
    # pass 1: metadata only -> global order (clip_id, frame_idx)
    meta = []                                                     # (clip, fi, split, group, text, qa, file_idx, row_idx)
    for fi_, f in enumerate(files):
        t = pq.read_table(f, columns=["clip_id", "frame_idx", "split", "group", "text", "qa_flags"])
        d = t.to_pydict()
        for r in range(t.num_rows):
            if keep and d["split"][r] not in keep: continue
            meta.append((d["clip_id"][r], d["frame_idx"][r], d["split"][r], d["group"][r], d["text"][r], d["qa_flags"][r], fi_, r))
    meta.sort(key=lambda m: (m[0], m[1]))
    N = len(meta); pos = {(m[6], m[7]): k for k, m in enumerate(meta)}
    S = Image.open(io.BytesIO(pq.read_table(files[0], columns=["color"]).slice(0, 1).column("color")[0].as_py()["bytes"])).size[0]
    print(N, "frames,", S, "px")
    mm = np.lib.format.open_memmap(os.path.join(a.out, "frames.npy"), mode="w+", dtype=np.uint8, shape=(N, S, S, 4))
    # pass 2: decode shard by shard
    for fi_, f in enumerate(files):
        col = pq.read_table(f, columns=["color"]).column("color").to_pylist()
        rows = [(r, pos[(fi_, r)]) for r in range(len(col)) if (fi_, r) in pos]
        def dec(rk):
            r, k = rk; mm[k] = np.asarray(Image.open(io.BytesIO(col[r]["bytes"])).convert("RGBA"))
        with ThreadPoolExecutor(a.threads) as ex: list(ex.map(dec, rows))
        if fi_ % 20 == 0: print(f"  {fi_ + 1}/{len(files)} shards", flush=True)
    mm.flush()
    clips = {}
    for k, m in enumerate(meta):
        c = m[0]
        if c not in clips: clips[c] = dict(start=k, n=0, split=m[2], group=m[3], text=m[4], qa=m[5])
        clips[c]["n"] += 1
    json.dump(clips, open(os.path.join(a.out, "clips.json"), "w"))
    json.dump(dict(size=S, frames=N, clips=len(clips)), open(os.path.join(a.out, "meta.json"), "w"))
    print(len(clips), "clips ->", a.out)


if __name__ == "__main__":
    main()
