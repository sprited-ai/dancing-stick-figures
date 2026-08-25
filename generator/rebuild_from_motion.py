"""Rebuild the released 128px frame dataset from the public ``motion`` parquet files.

The default path reproduces the original body and camera parameters from each
``clip_id``.  An instructor can instead supply a private ``--variant_seed`` and
small JSON config to create a course-specific rendering without changing the
underlying motions or prompt split.

Examples
--------
Rebuild the public rendering::

    python -m generator.rebuild_from_motion \
      --motion data/hf/motion --out data/rebuilt_frames --workers 8

Create an instructor rendering (keep the seed private if it defines a hidden
evaluation set)::

    python -m generator.rebuild_from_motion \
      --motion data/hf/motion --out data/course_frames --workers 8 \
      --variant_seed course-2026 \
      --variant_config configs/instructor_variant.example.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .ardy_adapter import apply_bone_scale, frame_joints, root_trajectory, to_figure_frame
from .build import (
    DEPTH_RANGE,
    FPS,
    SCHEMA,
    VERSION,
    _h,
    depth_png,
    png_bytes,
    sample_body,
    sample_cameras,
)
from .render import SIZE, render_all
from .skeleton import NAMES, project


MOTION_COLUMNS = (
    "clip_id", "group", "held_out", "split", "text", "seed", "fps",
    "n_frames", "posed_joints",
)


def load_variant(path: str) -> dict:
    if not path:
        return {}
    config = json.loads(Path(path).read_text())
    allowed = {
        "px_per_m_scale", "stroke_scale", "camera_yaw_degrees",
        "camera_pitch_degrees", "camera_center_x", "camera_center_y",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown instructor-variant fields: {sorted(unknown)}")
    return config


def _body_and_cameras(clip_id: str, variant_seed: str, variant: dict):
    seed = _h(clip_id) if not variant_seed else _h(clip_id, "instructor", variant_seed)
    rng = random.Random(seed)
    body, cameras = sample_body(rng), sample_cameras(rng)
    if variant:
        body.px_per_m *= float(variant.get("px_per_m_scale", 1.0))
        body.stroke *= float(variant.get("stroke_scale", 1.0))
        yaw = math.radians(float(variant.get("camera_yaw_degrees", 0.0)))
        pitch = math.radians(float(variant.get("camera_pitch_degrees", 0.0)))
        dx = float(variant.get("camera_center_x", 0.0))
        dy = float(variant.get("camera_center_y", 0.0))
        for camera in cameras:
            camera.yaw += yaw
            camera.pitch += pitch
            camera.center = (camera.center[0] + dx, camera.center[1] + dy)
    return body, cameras


def render_motion_clip(job):
    """Convert one motion-parquet row into the original frame-parquet rows."""
    row, variant_seed, variant = job
    frames = int(row["n_frames"])
    posed = np.frombuffer(row["posed_joints"], dtype=np.float32).reshape(frames, len(NAMES), 3)
    fps = float(row.get("fps") or FPS)
    figure = to_figure_frame(posed.astype(np.float64))
    root_pos, root_vel, root_heading = root_trajectory(posed, fps)

    floor = posed[0, [21, 22, 25, 26], 1].min()
    hips_h = root_pos[:, 1] + (posed[0, 0, 1] - floor)
    speed = float(np.linalg.norm(np.diff(figure, axis=0), axis=-1).mean() * fps)
    flags = []
    if hips_h.max() > 1.6:
        flags.append("levitation")
    if speed < 0.02:
        flags.append("frozen")
    qa_flags = ",".join(flags)

    clip_id = row["clip_id"]
    body, cameras = _body_and_cameras(clip_id, variant_seed, variant)
    figure = apply_bone_scale(figure, body.bone_scale)
    rendered = []
    for camera_index, camera in enumerate(cameras):
        for frame_index in range(frames):
            joints3 = frame_joints(figure, frame_index)
            joints2, joint_depth = project(joints3, camera, body.px_per_m)
            outputs = render_all(joints2, joint_depth, body)
            xyz = np.array([joints3[name] for name in NAMES], np.float32)
            xy = np.array([joints2[name] for name in NAMES], np.float32) / SIZE
            depth = np.array([joint_depth[name] for name in NAMES], np.float32)
            visible = np.array(
                [(outputs["seg"] == index + 1).any() for index in range(len(NAMES))], bool
            )
            view_id = f"{clip_id}/c{camera_index}"
            rendered.append({
                "sample_id": f"{view_id}/f{frame_index:03d}",
                "clip_id": view_id,
                "frame_idx": frame_index,
                "n_frames": frames,
                "fps": int(round(fps)),
                "split": row["split"],
                "group": row["group"],
                "held_out": bool(row["held_out"]),
                "text": row["text"],
                "seed": int(row["seed"]),
                "qa_flags": qa_flags,
                "cam_yaw": float(camera.yaw),
                "cam_pitch": float(camera.pitch),
                "cam_center_x": camera.center[0],
                "cam_center_y": camera.center[1],
                "px_per_m": body.px_per_m,
                "stroke": body.stroke,
                "bone_scale": json.dumps(body.bone_scale),
                "joint_xyz": xyz.tobytes(),
                "joint_xy": xy.tobytes(),
                "joint_depth": depth.tobytes(),
                "joint_visible": visible.tobytes(),
                "root_pos": root_pos[frame_index].tobytes(),
                "root_vel": root_vel[frame_index].tobytes(),
                "root_heading": root_heading[frame_index].tobytes(),
                "color": {"bytes": png_bytes(outputs["color"]), "path": None},
                "depth": {"bytes": depth_png(outputs["depth"]), "path": None},
                "normal": {"bytes": png_bytes(outputs["normal"]), "path": None},
                "seg": {"bytes": png_bytes(Image.fromarray(outputs["seg"], "L")), "path": None},
            })
    return rendered


def motion_rows(path: str, limit: int = 0):
    files = sorted(Path(path).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no motion parquet files under {path}")
    rows = []
    for file in files:
        table = pq.read_table(file, columns=list(MOTION_COLUMNS))
        rows.extend(table.to_pylist())
    rows.sort(key=lambda row: row["clip_id"])
    return rows[:limit] if limit else rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True, help="directory containing released motion parquet shards")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--rows_per_shard", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0, help="motion clips; useful for verification")
    parser.add_argument("--variant_seed", default="", help="private instructor seed; empty reproduces public parameters")
    parser.add_argument("--variant_config", default="", help="small JSON renderer override")
    args = parser.parse_args()

    variant = load_variant(args.variant_config)
    rows = motion_rows(args.motion, args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    buffers = {split: [] for split in ("train", "val", "test")}
    shard_count = {split: 0 for split in buffers}
    total = 0
    started = time.time()

    def flush(split: str, force: bool = False):
        while len(buffers[split]) >= args.rows_per_shard or (force and buffers[split]):
            chunk = buffers[split][:args.rows_per_shard]
            del buffers[split][:args.rows_per_shard]
            table = pa.Table.from_pylist(chunk, schema=SCHEMA)
            destination = out / f"{split}-{shard_count[split]:05d}.parquet"
            pq.write_table(table, destination, compression="zstd")
            shard_count[split] += 1

    jobs = ((row, args.variant_seed, variant) for row in rows)
    with ProcessPoolExecutor(args.workers) as executor:
        for index, frames in enumerate(executor.map(render_motion_clip, jobs, chunksize=1), start=1):
            split = frames[0]["split"]
            buffers[split].extend(frames)
            total += len(frames)
            for name in buffers:
                flush(name)
            if index == 1 or index % 10 == 0:
                print(f"{index}/{len(rows)} motions, {total} frames", flush=True)
    for name in buffers:
        flush(name, force=True)

    metadata = {
        "version": VERSION,
        "source": "released motion parquet",
        "clips": len(rows),
        "frames": total,
        "shards": shard_count,
        "fps": FPS,
        "size": SIZE,
        "depth_range": DEPTH_RANGE,
        "joints": NAMES,
        "variant_seeded": bool(args.variant_seed),
        "variant": variant,
        "seconds": round(time.time() - started),
    }
    (out / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("done", metadata)


if __name__ == "__main__":
    main()
