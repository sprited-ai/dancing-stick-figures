"""`mini` config: 64² colour (+seg) and all labels, no depth/normal — ~1 GB, for laptops / Colab / classrooms.

    python scripts/make_mini.py --src data/v1_final --out data/v1_final/mini [--size 64]

Colour is area-downsampled in premultiplied space (so background never bleeds into strokes) and stored back as
straight RGBA; seg is nearest-neighbour. Label columns are copied unchanged (joint_xy is normalised, so it stays valid).
"""
from __future__ import annotations
import argparse, glob, io, os, json, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

DROP = ("depth", "normal")


def shrink_color(b: bytes, S: int) -> bytes:
    x = np.asarray(Image.open(io.BytesIO(b)).convert("RGBA")).astype(np.float32) / 255.0
    f = x.shape[0] // S
    pm = np.concatenate([x[..., :3] * x[..., 3:4], x[..., 3:4]], -1)
    pm = pm.reshape(S, f, S, f, 4).mean((1, 3))
    a = pm[..., 3:4]
    rgb = np.where(a > 1e-4, pm[..., :3] / np.maximum(a, 1e-4), 0.0)
    out = (np.clip(np.concatenate([rgb, a], -1), 0, 1) * 255 + 0.5).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(out, "RGBA").save(buf, format="PNG", optimize=True); return buf.getvalue()


def shrink_seg(b: bytes, S: int) -> bytes:
    im = Image.open(io.BytesIO(b)).convert("L").resize((S, S), Image.NEAREST)
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True); return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True); ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.src, "*.parquet"))); n_rows = 0
    for i, f in enumerate(files):
        t = pq.read_table(f); t = t.drop([c for c in DROP if c in t.column_names])
        col = t.column("color").to_pylist(); seg = t.column("seg").to_pylist()
        with ThreadPoolExecutor(a.threads) as ex:
            nc = list(ex.map(lambda v: shrink_color(v["bytes"], a.size), col))
            ns = list(ex.map(lambda v: shrink_seg(v["bytes"], a.size), seg))
        img_t = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
        t = t.set_column(t.schema.get_field_index("color"), "color", pa.array([{"bytes": b, "path": None} for b in nc], img_t))
        t = t.set_column(t.schema.get_field_index("seg"), "seg", pa.array([{"bytes": b, "path": None} for b in ns], img_t))
        pq.write_table(t, os.path.join(a.out, os.path.basename(f)), compression="zstd")
        n_rows += t.num_rows
        if i % 20 == 0: print(f"  {i + 1}/{len(files)} shards, {n_rows} rows", flush=True)
    meta = json.load(open(os.path.join(a.src, "meta.json"))); meta.update(config="mini", size=a.size, columns_dropped=list(DROP),
                                                                    note="64px colour (premultiplied area downsample) + seg (nearest) + all labels")
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("done", n_rows, "rows ->", a.out)


if __name__ == "__main__":
    main()
