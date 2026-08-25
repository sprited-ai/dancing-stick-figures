"""Summarise label disagreements between a rebuilt corpus and a release tier.

This deliberately reads only the small columns needed for diagnosis; the full
pixel verifier remains ``scripts/verify_rebuild.py``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.skeleton import NAMES


def rows(root: str):
    files = sorted(glob.glob(str(Path(root) / "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet shards under {root}")
    return ds.dataset(files, format="parquet").to_table(
        columns=["sample_id", "clip_id", "frame_idx", "root_heading", "joint_visible"]
    ).to_pylist()


def camera(clip_id: str) -> str:
    return clip_id.rsplit("/", 1)[-1]


def numeric_array(value) -> np.ndarray:
    """Decode Arrow binary float arrays as well as ordinary list columns."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.frombuffer(value, dtype=np.float32).astype(np.float64)
    return np.asarray(value, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuilt", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args()

    reference = {row["sample_id"]: row for row in rows(args.reference)}
    heading_examples = []
    visibility_examples = []
    heading_count = heading_allclose_count = visibility_samples = visibility_bits = 0
    heading_max_abs = 0.0
    visibility_by_camera = Counter()
    visibility_by_joint = Counter()
    rebuilt_visible_only_by_joint = Counter()
    released_visible_only_by_joint = Counter()
    visibility_by_frame = Counter()
    missing = 0

    for row in rows(args.rebuilt):
        other = reference.get(row["sample_id"])
        if other is None:
            missing += 1
            continue
        if row["root_heading"] != other["root_heading"]:
            heading_count += 1
            left = numeric_array(row["root_heading"])
            right = numeric_array(other["root_heading"])
            close = bool(np.allclose(left, right, rtol=0.0, atol=1e-6, equal_nan=True))
            heading_allclose_count += int(close)
            finite = np.abs(left - right)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                heading_max_abs = max(heading_max_abs, float(finite.max()))
            if len(heading_examples) < args.examples:
                heading_examples.append({
                    "sample_id": row["sample_id"], "rebuilt": left.tolist(),
                    "released": right.tolist(), "allclose_atol_1e-6": close,
                })

        left_visible = np.frombuffer(row["joint_visible"], dtype=np.bool_)
        right_visible = np.frombuffer(other["joint_visible"], dtype=np.bool_)
        changed = np.flatnonzero(left_visible != right_visible)
        if changed.size:
            visibility_samples += 1
            visibility_bits += int(changed.size)
            visibility_by_camera[camera(row["clip_id"])] += int(changed.size)
            visibility_by_frame[int(row["frame_idx"])] += int(changed.size)
            for joint in changed:
                visibility_by_joint[int(joint)] += 1
                if left_visible[joint]:
                    rebuilt_visible_only_by_joint[int(joint)] += 1
                else:
                    released_visible_only_by_joint[int(joint)] += 1
            if len(visibility_examples) < args.examples:
                visibility_examples.append({
                    "sample_id": row["sample_id"],
                    "different_joints": [NAMES[int(index)] for index in changed],
                })

    result = {
        "missing_rows": missing,
        "root_heading_mismatched_rows": heading_count,
        "root_heading_allclose_atol_1e-6_rows": heading_allclose_count,
        "root_heading_max_abs_difference": heading_max_abs,
        "root_heading_examples": heading_examples,
        "visibility_mismatched_samples": visibility_samples,
        "visibility_mismatched_bits": visibility_bits,
        "visibility_bits_by_camera": dict(visibility_by_camera),
        "visibility_bits_by_joint": {
            NAMES[index]: count for index, count in sorted(visibility_by_joint.items())
        },
        "rebuilt_visible_only_by_joint": {
            NAMES[index]: count for index, count in sorted(rebuilt_visible_only_by_joint.items())
        },
        "released_visible_only_by_joint": {
            NAMES[index]: count for index, count in sorted(released_visible_only_by_joint.items())
        },
        "visibility_bits_top_frames": visibility_by_frame.most_common(20),
        "visibility_examples": visibility_examples,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n")


if __name__ == "__main__":
    main()
