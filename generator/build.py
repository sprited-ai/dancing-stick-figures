"""Dataset builder: ARDY .npz clips -> stickdance-128 parquet shards.

    python -m generator.build --npz ardy_out/v1 --prompts prompts/v1.txt --out data/v1 [--workers 8] [--limit N]

Per clip (deterministic from clip_id): a Body (bone_scale jitter, stroke, scale) and 1–2 Cameras.
Per frame: colour / depth / normal / seg PNGs + the label row (§3.5 of SPARK.md).
Split by PROMPT (not seed): held-out prompt groups -> test; else hash(prompt) -> train 90 / val 5 / test 5.
"""
from __future__ import annotations
import argparse, hashlib, io, json, math, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from .skeleton import Body, Camera, project, NAMES
from .render import render_all, SIZE
from .ardy_adapter import load, load_root, to_figure_frame, frame_joints, apply_bone_scale

FPS = 20
DEPTH_RANGE = (-1.5, 1.5)          # metres toward camera, Hips-relative -> uint16
CANON_YAWS = [0, 45, -45, 90, -90, 135, -135, 180]
VERSION = "0.1.0"


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def load_prompts(path):
    groups, held = {}, set()
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#"): continue
        g, p = line.split("\t", 1)
        if g.startswith("*"):
            g = g[1:]; held.add(g)
        groups[p] = g
    return groups, held


def sample_body(rng: random.Random) -> Body:
    bs = {}
    for side in ("Left", "Right"):
        for b in ("ForeArm", "Hand", "Leg", "Foot"):
            bs[f"{side}{b}"] = round(rng.uniform(0.92, 1.08), 3)
    for b in ("Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head"):
        bs[b] = round(rng.uniform(0.94, 1.06), 3)
    return Body(px_per_m=round(rng.uniform(50, 58), 2), bone_scale=bs, stroke=rng.choice([3.0, 3.5, 4.0, 4.0, 4.5, 5.0]))


def sample_cameras(rng: random.Random) -> list[Camera]:
    n = 3   # v1: three cameras per clip (seed/camera diversity before prompt diversity)
    cams = []
    for _ in range(n):
        yaw = rng.choice(CANON_YAWS) + rng.uniform(-6, 6) if rng.random() < 0.7 else rng.uniform(-180, 180)
        pitch = rng.uniform(-3, 10)
        cams.append(Camera(yaw=math.radians(yaw), pitch=math.radians(pitch),
                           center=(64.0 + rng.uniform(-3, 3), 66.0 + rng.uniform(-3, 3))))
    return cams


def png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO(); img.save(b, format="PNG", optimize=False); return b.getvalue()


def depth_png(depth: np.ndarray) -> bytes:
    lo, hi = DEPTH_RANGE
    d = np.nan_to_num(depth, nan=lo)
    u = np.clip((d - lo) / (hi - lo), 0, 1) * 65535
    u[np.isnan(depth)] = 0
    return png_bytes(Image.fromarray(u.astype(np.uint16), "I;16"))


def build_clip(args):
    npz_path, group, held, out_dir, shard_tag = args
    P, fps, text = load(npz_path)
    Q = to_figure_frame(P)
    rpos, rvel, rhead = load_root(npz_path)
    # QA flags per clip (kept in the data, not filtered — users decide): levitation = Hips ever > 1.6 m
    # above its frame-0 floor (only jumps get near 1.5); frozen = mean joint speed < 0.02 m/s.
    hips_h = rpos[:, 1] + (P[0, 0, 1] - P[0, [21, 22, 25, 26], 1].min())
    speed = float(np.linalg.norm(np.diff(Q, axis=0), axis=-1).mean() * fps)
    qa = []
    if hips_h.max() > 1.6: qa.append("levitation")
    if speed < 0.02: qa.append("frozen")
    qa_flags = ",".join(qa)
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    seed = int(stem.rsplit("_s", 1)[1]) if "_s" in stem else 0
    clip_id = f"{group}/{stem}"
    rng = random.Random(_h(clip_id))
    body = sample_body(rng)
    cams = sample_cameras(rng)
    Q = apply_bone_scale(Q, body.bone_scale)     # per-clip proportions (v1 final: applied to ARDY joints, not just recorded)
    # split by PROMPT (all seeds + cameras of one prompt share a split) so val/test = held-out prompts
    # within seen groups; the two `*` groups are held-out concepts. Never split by seed.
    hh = _h(group, stem.rsplit("_s", 1)[0]) % 100
    split = "test" if group in held else ("train" if hh < 90 else "val" if hh < 95 else "test")
    rows = []
    for ci, cam in enumerate(cams):
        for t in range(Q.shape[0]):
            j3 = frame_joints(Q, t)
            j2, dep = project(j3, cam, body.px_per_m)
            o = render_all(j2, dep, body)
            xyz = np.array([j3[n] for n in NAMES], np.float32)
            xy = np.array([j2[n] for n in NAMES], np.float32) / SIZE
            jd = np.array([dep[n] for n in NAMES], np.float32)
            vis = np.array([(o["seg"] == i + 1).any() for i in range(len(NAMES))], bool)
            rows.append(dict(
                sample_id=f"{clip_id}/c{ci}/f{t:03d}", clip_id=f"{clip_id}/c{ci}", frame_idx=t, n_frames=int(Q.shape[0]),
                fps=FPS, split=split, group=group, held_out=group in held, text=text, seed=seed, qa_flags=qa_flags,
                cam_yaw=float(cam.yaw), cam_pitch=float(cam.pitch), cam_center_x=cam.center[0], cam_center_y=cam.center[1],
                px_per_m=body.px_per_m, stroke=body.stroke, bone_scale=json.dumps(body.bone_scale),
                joint_xyz=xyz.tobytes(), joint_xy=xy.tobytes(), joint_depth=jd.tobytes(), joint_visible=vis.tobytes(),
                root_pos=rpos[t].tobytes(), root_vel=rvel[t].tobytes(), root_heading=rhead[t].tobytes(),
                color={"bytes": png_bytes(o["color"]), "path": None},
                depth={"bytes": depth_png(o["depth"]), "path": None},
                normal={"bytes": png_bytes(o["normal"]), "path": None},
                seg={"bytes": png_bytes(Image.fromarray(o["seg"], "L")), "path": None},
            ))
    return rows


IMG = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("clip_id", pa.string()), ("frame_idx", pa.int32()), ("n_frames", pa.int32()),
    ("fps", pa.int32()), ("split", pa.string()), ("group", pa.string()), ("held_out", pa.bool_()), ("text", pa.string()),
    ("seed", pa.int32()), ("qa_flags", pa.string()), ("cam_yaw", pa.float32()), ("cam_pitch", pa.float32()), ("cam_center_x", pa.float32()),
    ("cam_center_y", pa.float32()), ("px_per_m", pa.float32()), ("stroke", pa.float32()), ("bone_scale", pa.string()),
    ("joint_xyz", pa.binary()), ("joint_xy", pa.binary()), ("joint_depth", pa.binary()), ("joint_visible", pa.binary()),
    ("root_pos", pa.binary()), ("root_vel", pa.binary()), ("root_heading", pa.binary()),
    ("color", IMG), ("depth", IMG), ("normal", IMG), ("seg", IMG),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2)); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rows_per_shard", type=int, default=2000)
    a = ap.parse_args()
    groups, held = load_prompts(a.prompts)
    clips = []
    for root, _, files in os.walk(a.npz):
        for f in sorted(files):
            if f.endswith(".npz"):
                clips.append((os.path.join(root, f), os.path.basename(root)))
    clips.sort()
    if a.limit: clips = clips[:a.limit]
    print(f"{len(clips)} clips, {a.workers} workers")
    os.makedirs(a.out, exist_ok=True)
    jobs = [(p, g, held, a.out, i) for i, (p, g) in enumerate(clips)]
    buf = {"train": [], "val": [], "test": []}; shard_n = {"train": 0, "val": 0, "test": 0}; total = 0
    t0 = time.time()

    def flush(split, force=False):
        while len(buf[split]) >= a.rows_per_shard or (force and buf[split]):
            rows, buf[split] = buf[split][:a.rows_per_shard], buf[split][a.rows_per_shard:]
            tbl = pa.Table.from_pylist(rows, schema=SCHEMA)
            pq.write_table(tbl, os.path.join(a.out, f"{split}-{shard_n[split]:05d}.parquet"), compression="zstd")
            shard_n[split] += 1

    with ProcessPoolExecutor(a.workers) as ex:
        for i, rows in enumerate(ex.map(build_clip, jobs, chunksize=1)):
            for r in rows: buf[r["split"]].append(r)
            total += len(rows)
            for s in buf: flush(s)
            if i % 10 == 0:
                print(f"  {i+1}/{len(clips)} clips, {total} frames, {time.time()-t0:.0f}s", flush=True)
    for s in buf: flush(s, force=True)
    meta = dict(version=VERSION, clips=len(clips), frames=total, shards=shard_n, fps=FPS, size=SIZE, depth_range=DEPTH_RANGE,
                joints=NAMES, held_out_groups=sorted(held), seconds=round(time.time() - t0))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("done", meta)


if __name__ == "__main__":
    main()
