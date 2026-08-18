"""ARDY raw motion -> `motion` config parquet (one row per clip). Ships the generator's full output so users can
retarget (rotations, foot contacts) or learn render->3D pose; the frame config only carries figure-frame joints.

    python -m generator.export_motion --npz ~/dev/ardy/outputs/v1 --prompts prompts/v1.txt --out data/v1_final/motion

Columns: clip_id, group, held_out, split, text, seed, fps, n_frames, and float32/bool arrays as raw little-endian bytes
with their shapes in meta.json: posed_joints [T,27,3] (world, metres), local_rot_mats / global_rot_mats [T,27,3,3],
root_positions / smooth_root_pos [T,3], global_root_heading [T,2], foot_contacts [T,4] bool, frame0_basis [3,3]
(rows = figure-frame x/left, y/up, z/forward in world coords; figure_joints = (world - hips) @ basis.T).
Split/group are identical to the frame config (same hash of (group, prompt slug)).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from .build import _h, load_prompts
from .ardy_adapter import _frame0_basis
from .skeleton import NAMES

ARR = ["posed_joints", "local_rot_mats", "global_rot_mats", "root_positions", "smooth_root_pos", "global_root_heading", "foot_contacts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rows_per_shard", type=int, default=200)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    groups, held = load_prompts(a.prompts)
    rows = []; shapes = {}
    for g in sorted(os.listdir(a.npz)):
        gd = os.path.join(a.npz, g)
        if not os.path.isdir(gd): continue
        for f in sorted(os.listdir(gd)):
            if not f.endswith(".npz"): continue
            d = np.load(os.path.join(gd, f), allow_pickle=True)
            stem = f[:-4]; seed = int(stem.rsplit("_s", 1)[1]) if "_s" in stem else 0
            hh = _h(g, stem.rsplit("_s", 1)[0]) % 100
            split = "test" if g in held else ("train" if hh < 90 else "val" if hh < 95 else "test")
            P = np.asarray(d["posed_joints"], np.float32); R, _ = _frame0_basis(P.astype(np.float64))
            row = dict(clip_id=f"{g}/{stem}", group=g, held_out=g in held, split=split, text=str(d["text"]), seed=seed,
                       fps=int(d["fps"]) if "fps" in d else 20, n_frames=int(P.shape[0]),
                       frame0_basis=R.astype(np.float32).tobytes())
            for k in ARR:
                v = np.ascontiguousarray(d[k]); row[k] = v.tobytes()
                shapes[k] = dict(dtype=str(v.dtype), shape=["T"] + list(v.shape[1:]))
            rows.append(row)
    shapes["frame0_basis"] = dict(dtype="float32", shape=[3, 3])
    schema = pa.schema([("clip_id", pa.string()), ("group", pa.string()), ("held_out", pa.bool_()), ("split", pa.string()),
                        ("text", pa.string()), ("seed", pa.int32()), ("fps", pa.int32()), ("n_frames", pa.int32()),
                        ("frame0_basis", pa.binary())] + [(k, pa.binary()) for k in ARR])
    by = {}
    for r in rows: by.setdefault(r["split"], []).append(r)
    n_sh = {}
    for split, rs in by.items():
        for i in range(0, len(rs), a.rows_per_shard):
            t = pa.Table.from_pylist(rs[i:i + a.rows_per_shard], schema=schema)
            pq.write_table(t, os.path.join(a.out, f"{split}-{i // a.rows_per_shard:05d}.parquet"), compression="zstd")
        n_sh[split] = (len(rs) + a.rows_per_shard - 1) // a.rows_per_shard
    json.dump(dict(config="motion", clips=len(rows), shards=n_sh, joints=NAMES, arrays=shapes,
                   note="raw NVIDIA ARDY outputs (world frame); frame config joints = (posed_joints - Hips) @ frame0_basis.T"),
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(dict(clips=len(rows), shards=n_sh))


if __name__ == "__main__":
    main()
