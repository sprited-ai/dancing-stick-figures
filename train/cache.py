"""Decode the parquet colour frames once into a uint8 memmap for fast video sampling.

    python -m train.cache --data data/v1 --out data/v1_cache

Writes frames.u8 [N,128,128,4] (straight RGBA as stored), clips.json (clip_id -> [start, n], split,
group, text) and index.npy. Premultiplication happens in the loader.
"""
from __future__ import annotations
import argparse, glob, io, json, os
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.data, "*.parquet")))
    cols = ["sample_id", "clip_id", "frame_idx", "split", "group", "text", "qa_flags", "color"]
    tabs = [pq.read_table(f, columns=cols) for f in files]
    import pyarrow as pa
    t = pa.concat_tables(tabs)
    clip = t.column("clip_id").to_pylist(); fi = t.column("frame_idx").to_pylist()
    order = sorted(range(len(clip)), key=lambda i: (clip[i], fi[i]))
    N = len(order)
    print(N, "frames")
    mm = np.lib.format.open_memmap(os.path.join(a.out, "frames.npy"), mode="w+", dtype=np.uint8, shape=(N, 128, 128, 4))
    color = t.column("color").to_pylist()

    def dec(k):
        i = order[k]
        mm[k] = np.asarray(Image.open(io.BytesIO(color[i]["bytes"])).convert("RGBA"))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(dec, range(N)))
    mm.flush()
    split = t.column("split").to_pylist(); group = t.column("group").to_pylist(); text = t.column("text").to_pylist()
    qa = t.column("qa_flags").to_pylist()
    clips = {}
    for k, i in enumerate(order):
        c = clip[i]
        if c not in clips:
            clips[c] = dict(start=k, n=0, split=split[i], group=group[i], text=text[i], qa=qa[i])
        clips[c]["n"] += 1
    json.dump(clips, open(os.path.join(a.out, "clips.json"), "w"))
    print(len(clips), "clips ->", a.out)


if __name__ == "__main__":
    main()
