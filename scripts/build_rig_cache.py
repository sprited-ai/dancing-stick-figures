"""Build rig.npy / rig_depth.npy aligned 1:1 with a frames.npy cache.

The 2D screen-space cskel27 coordinates are already stored per frame in the
parquet shards that produced the cache (`joint_xy`, normalised by the 128-px
render frame; `joint_depth` in metres toward the camera).  This script streams
the shards and writes:

  rig.npy        [N, 27, 2] float16, normalised [0,1] (pixel j spans [j,j+1);
                 index with floor, not round)
  rig_depth.npy  [N, 27]    float16, metres toward camera
  rig_meta.json  provenance: joint order/parents, units, source dir, hashes

Source-of-truth guard: only the parquet tree that byte-reproduces the cache
frames is valid (a divergent second build exists on gin); we bind rig.npy to
frames.npy by sha256 in rig_meta.json.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

NAMES = [
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandEnd", "RightHandThumb1",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandEnd", "LeftHandThumb1",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
]
PARENTS = [-1, 0, 1, 2, 3, 4, 5,
           4, 7, 8, 9, 10, 10,
           4, 13, 14, 15, 16, 16,
           0, 19, 20, 21,
           0, 23, 24, 25]


def file_sha256(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            read += len(chunk)
            if limit and read >= limit:
                break
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="directory of source shards (hf_stage/mini)")
    parser.add_argument("--cache", required=True, help="cache dir containing frames.npy + clips.json")
    parser.add_argument("--verify", type=int, default=10000,
                        help="random frames for the foreground-hit check")
    args = parser.parse_args()

    cache = Path(args.cache)
    clips = json.loads((cache / "clips.json").read_text())
    meta = json.loads((cache / "meta.json").read_text())
    total = int(meta["frames"])
    index = {cid: int(row["start"]) for cid, row in clips.items()}
    if sum(int(row["n"]) for row in clips.values()) != total:
        raise SystemExit("clips.json frame counts do not partition frames.npy")

    rig = np.lib.format.open_memmap(cache / "rig.npy", mode="w+",
                                    dtype=np.float16, shape=(total, 27, 2))
    depth = np.lib.format.open_memmap(cache / "rig_depth.npy", mode="w+",
                                      dtype=np.float16, shape=(total, 27))
    rig[:] = np.nan
    depth[:] = np.nan

    shards = sorted(Path(args.parquet).glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards under {args.parquet}")
    filled = 0
    offscreen = 0
    for shard in shards:
        table = pq.read_table(shard, columns=["clip_id", "frame_idx", "joint_xy", "joint_depth"])
        cids = table.column("clip_id").to_pylist()
        fis = table.column("frame_idx").to_pylist()
        xys = table.column("joint_xy").to_pylist()
        dps = table.column("joint_depth").to_pylist()
        for cid, fi, xy_blob, d_blob in zip(cids, fis, xys, dps):
            if cid not in index:
                raise SystemExit(f"shard row {cid} missing from clips.json")
            row = index[cid] + int(fi)
            xy = np.frombuffer(xy_blob, "<f4").reshape(27, 2)
            # Joints may legitimately leave the frame (e.g. lying poses);
            # store them as-is and only reject wildly implausible values.
            if not ((xy > -0.5).all() and (xy < 1.5).all()):
                raise SystemExit(f"implausible joint at {cid}/f{fi}: {xy.min()} {xy.max()}")
            offscreen += int(((xy < 0) | (xy > 1)).any(axis=1).sum())
            rig[row] = xy.astype(np.float16)
            depth[row] = np.frombuffer(d_blob, "<f4").astype(np.float16)
            filled += 1
        print(f"{shard.name}: cumulative {filled}/{total}", flush=True)
    if filled != total or np.isnan(rig).any() or np.isnan(depth).any():
        raise SystemExit(f"incomplete fill: {filled}/{total}")
    rig.flush(); depth.flush()

    frames = np.load(cache / "frames.npy", mmap_mode="r")
    rng = np.random.default_rng(20260823)
    picks = rng.choice(total, size=min(args.verify, total), replace=False)
    size = int(meta["size"])
    hit, checked = 0, 0
    for i in picks:
        xy = rig[i].astype(np.float32)
        onscreen = ((xy >= 0) & (xy < 1)).all(axis=1)
        px = np.floor(xy * size).astype(int).clip(0, size - 1)
        alpha = frames[i][..., 3]
        hit += int((alpha[px[onscreen, 1], px[onscreen, 0]] > 0).sum())
        checked += int(onscreen.sum())
    rate = hit / checked
    print(f"foreground-hit rate over {len(picks)} frames: {rate:.4f}")
    # Hips/Head/thumbs draw no capsule and thin bones can miss single pixels
    # at 64 px, so demand a high but not perfect rate.
    if rate < 0.80:
        raise SystemExit("foreground-hit rate too low; rig/frames misalignment suspected")

    (cache / "rig_meta.json").write_text(json.dumps({
        "joints": NAMES, "parents": PARENTS,
        "units": "normalised [0,1] over the rendered frame",
        "coordinate_convention": "continuous pixel space; pixel j spans [j, j+1); index with floor",
        "size": size, "frames": total,
        "depth_units": "metres toward camera (rig_depth.npy)",
        "source_parquet_dir": str(Path(args.parquet).resolve()),
        "source_shards": len(shards),
        "first_shard_sha256": file_sha256(shards[0]),
        "frames_npy_sha256_first_1gb": file_sha256(cache / "frames.npy", limit=1 << 30),
        "clips_json_sha256": file_sha256(cache / "clips.json"),
        "foreground_hit_rate": rate,
        "verified_frames": int(len(picks)),
        "joints_offscreen_total": int(offscreen),
    }, indent=2) + "\n")
    print("rig cache complete")


if __name__ == "__main__":
    main()
