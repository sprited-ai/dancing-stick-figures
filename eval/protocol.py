"""Frozen sampling protocol for reproducible long-horizon evaluation.

The statistical unit is an underlying motion, not a rendered camera view or an
individual frame.  A manifest therefore selects at most one camera view from a
motion and keeps the two real-reference halves motion-disjoint.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch


_CAMERA_SUFFIX = re.compile(r"/c\d+$")


def motion_id(clip_id: str) -> str:
    """Map ``group/prompt_seed/camera`` clip IDs to their source motion ID."""
    return _CAMERA_SUFFIX.sub("", clip_id)


def _eligible_clips(cache, frames, stride, split, drop_flags):
    clips = json.load(open(os.path.join(cache, "clips.json")))
    span = (frames - 1) * stride + 1
    splits = {split} if isinstance(split, str) else set(split)
    out = []
    for clip_id, raw in clips.items():
        if raw["split"] not in splits or raw["n"] < span:
            continue
        if any(flag in (raw.get("qa") or "") for flag in drop_flags):
            continue
        out.append({"clip_id": clip_id, **raw})
    return out, span


def build_reference_manifest(
    cache,
    frames=50,
    stride=1,
    split=("val", "test"),
    n_per_half=64,
    seed=0,
    drop_flags=("levitation",),
    first_frames=0,
):
    """Create two deterministic, source-motion-disjoint real reference sets.

    Each entry records the exact clip, camera, and frame offset.  Raising when
    there are too few independent motions is intentional: silently duplicating
    clips would make the real-real reference look more stable than it is.
    """
    clips, span = _eligible_clips(cache, frames, stride, split, drop_flags)
    by_motion = {}
    for clip in clips:
        by_motion.setdefault(motion_id(clip["clip_id"]), []).append(clip)

    rng = np.random.default_rng(seed)
    motions = np.array(sorted(by_motion), dtype=object)
    rng.shuffle(motions)
    needed = 2 * n_per_half
    if len(motions) < needed:
        raise ValueError(
            f"need {needed} independent motions for two {n_per_half}-video references; "
            f"found {len(motions)} in split={split!r}"
        )

    def choose(ids):
        entries = []
        for mid in ids:
            views = sorted(by_motion[str(mid)], key=lambda x: x["clip_id"])
            clip = views[int(rng.integers(0, len(views)))]
            max_offset = clip["n"] - span
            if first_frames:
                max_offset = max(0, min(max_offset, first_frames - span))
            offset = int(rng.integers(0, max_offset + 1)) if max_offset else 0
            entries.append(
                {
                    "motion_id": str(mid),
                    "clip_id": clip["clip_id"],
                    "frame_offset": offset,
                    "absolute_start": int(clip["start"] + offset),
                    "frames": int(frames),
                    "stride": int(stride),
                }
            )
        return entries

    return {
        "protocol_version": 1,
        "cache": str(Path(cache)),
        "split": sorted({split} if isinstance(split, str) else set(split)),
        "frames": int(frames),
        "stride": int(stride),
        "seed": int(seed),
        "first_frames": int(first_frames),
        "statistical_unit": "source_motion",
        "reference_a": choose(motions[:n_per_half]),
        "reference_b": choose(motions[n_per_half:needed]),
    }


def save_manifest(manifest, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def load_manifest_windows(cache, entries, size=None):
    """Load exact manifest windows as premultiplied tensors [N,4,T,H,W]."""
    frames = np.load(os.path.join(cache, "frames.npy"), mmap_mode="r")
    out = []
    for entry in entries:
        start, count, stride = entry["absolute_start"], entry["frames"], entry["stride"]
        x = np.asarray(frames[start:start + count * stride:stride]).astype(np.float32) / 255.0
        if len(x) != count:
            raise ValueError(f"manifest window runs past clip boundary: {entry}")
        if size is not None and size != x.shape[1]:
            if x.shape[1] % size:
                raise ValueError(f"cannot area-downsample {x.shape[1]} to {size}")
            factor = x.shape[1] // size
            x = np.concatenate([x[..., :3] * x[..., 3:4], x[..., 3:4]], -1)
            x = x.reshape(count, size, factor, size, factor, 4).mean((2, 4))
        else:
            alpha = x[..., 3:4]
            x = np.concatenate([x[..., :3] * alpha, alpha], -1)
        out.append(torch.from_numpy(x).permute(3, 0, 1, 2) * 2 - 1)
    return torch.stack(out)


def main():
    parser = argparse.ArgumentParser(description="Freeze motion-disjoint real-reference windows")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--n", type=int, default=128, help="videos in each reference half")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", default="val,test", help="comma-separated source splits")
    parser.add_argument("--first_frames", type=int, default=0,
                        help="restrict reference windows to the first N frames of each clip (0 = whole clip)")
    args = parser.parse_args()
    manifest = build_reference_manifest(
        args.cache,
        frames=args.frames,
        stride=args.stride,
        split=tuple(x for x in args.split.split(",") if x),
        n_per_half=args.n,
        seed=args.seed,
        first_frames=args.first_frames,
    )
    save_manifest(manifest, args.out)
    left = {x["motion_id"] for x in manifest["reference_a"]}
    right = {x["motion_id"] for x in manifest["reference_b"]}
    print(
        f"wrote {args.out}: {len(left)} + {len(right)} source motions, "
        f"overlap={len(left & right)}, {args.frames} frames at stride {args.stride}"
    )


if __name__ == "__main__":
    main()
