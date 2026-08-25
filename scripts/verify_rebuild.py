"""Compare a motion-based 128px rebuild with released mini or frames data.

Discrete labels and stored projected joints are required to match exactly;
recomputed floating-point root headings use an explicit numeric tolerance.
Images are compared after applying the published mini downsampling transform when --mini is used.  --frames also
checks the native colour, segmentation, depth, and normal buffers.  Small pixel
tolerances accommodate rasterizer/library boundary differences rather than
pretending parquet or PNG files are a byte-stable archival format.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.make_mini import shrink_color, shrink_seg


LABELS = (
    "clip_id", "frame_idx", "n_frames", "fps", "split", "group", "held_out",
    "text", "seed", "qa_flags", "cam_yaw", "cam_pitch", "cam_center_x",
    "cam_center_y", "px_per_m", "stroke", "bone_scale", "joint_xyz",
    "joint_xy", "joint_depth", "root_pos", "root_vel",
)
ROOT_HEADING_ATOL = 1e-6


def pixels(value, mode):
    return np.asarray(Image.open(io.BytesIO(value["bytes"])).convert(mode))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuilt", required=True)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--mini")
    reference.add_argument("--frames")
    parser.add_argument("--out", default="")
    parser.add_argument("--max_color_channel_fraction", type=float, default=0.005)
    parser.add_argument("--max_seg_pixel_fraction", type=float, default=0.005)
    parser.add_argument(
        "--max_visible_bit_fraction", type=float, default=0.01,
        help=("maximum disagreement in joint-visible bits; visibility is the "
              "discontinuous test that a segment owns at least one raster pixel"),
    )
    parser.add_argument("--max_depth_pixel_fraction", type=float, default=0.005)
    parser.add_argument("--max_normal_channel_fraction", type=float, default=0.005)
    parser.add_argument(
        "--progress_every", type=int, default=25000,
        help="report comparison progress every N rebuilt rows; use 0 to disable",
    )
    args = parser.parse_args()

    rebuilt_files = sorted(glob.glob(str(Path(args.rebuilt) / "*.parquet")))
    reference_root = args.frames or args.mini
    reference_files = sorted(glob.glob(str(Path(reference_root) / "*.parquet")))
    if not rebuilt_files or not reference_files:
        raise SystemExit("both --rebuilt and the selected reference must contain parquet shards")
    full_resolution = bool(args.frames)
    rebuilt = pq.read_table(rebuilt_files).to_pylist()
    wanted = [row["sample_id"] for row in rebuilt]
    released_table = ds.dataset(reference_files, format="parquet").to_table(
        filter=ds.field("sample_id").isin(wanted)
    )
    released = {row["sample_id"]: row for row in released_table.to_pylist()}

    missing = sorted(set(wanted) - set(released))
    label_mismatches = {key: 0 for key in LABELS}
    color_different = color_total = seg_different = seg_total = 0
    depth_different = depth_total = normal_different = normal_total = 0
    visible_different = visible_total = 0
    color_max = depth_max = normal_max = 0
    root_heading_mismatches = 0
    root_heading_max_abs_difference = 0.0
    total_rows = len(rebuilt)
    for row_index, row in enumerate(rebuilt, 1):
        other = released.get(row["sample_id"])
        if other is None:
            continue
        for key in LABELS:
            label_mismatches[key] += int(row[key] != other[key])
        rebuilt_heading = np.frombuffer(row["root_heading"], dtype=np.float32)
        released_heading = np.frombuffer(other["root_heading"], dtype=np.float32)
        heading_delta = np.abs(rebuilt_heading.astype(np.float64) - released_heading.astype(np.float64))
        root_heading_max_abs_difference = max(
            root_heading_max_abs_difference,
            float(np.nanmax(heading_delta)) if heading_delta.size else 0.0,
        )
        root_heading_mismatches += int(
            not np.allclose(
                rebuilt_heading, released_heading, rtol=0.0,
                atol=ROOT_HEADING_ATOL, equal_nan=True,
            )
        )
        rebuilt_visible = np.frombuffer(row["joint_visible"], dtype=np.bool_)
        released_visible = np.frombuffer(other["joint_visible"], dtype=np.bool_)
        visible_different += int(np.count_nonzero(rebuilt_visible != released_visible))
        visible_total += int(rebuilt_visible.size)
        rebuilt_color = pixels(
            row["color"] if full_resolution
            else {"bytes": shrink_color(row["color"]["bytes"], 64)},
            "RGBA",
        )
        released_color = pixels(other["color"], "RGBA")
        delta = np.abs(rebuilt_color.astype(np.int16) - released_color.astype(np.int16))
        color_different += int(np.count_nonzero(delta))
        color_total += int(delta.size)
        color_max = max(color_max, int(delta.max()))
        rebuilt_seg = pixels(
            row["seg"] if full_resolution
            else {"bytes": shrink_seg(row["seg"]["bytes"], 64)},
            "L",
        )
        released_seg = pixels(other["seg"], "L")
        seg_different += int(np.count_nonzero(rebuilt_seg != released_seg))
        seg_total += int(rebuilt_seg.size)
        if full_resolution:
            rebuilt_depth = pixels(row["depth"], "I")
            released_depth = pixels(other["depth"], "I")
            depth_delta = np.abs(rebuilt_depth.astype(np.int64) - released_depth.astype(np.int64))
            depth_different += int(np.count_nonzero(depth_delta))
            depth_total += int(rebuilt_depth.size)
            depth_max = max(depth_max, int(depth_delta.max()))
            rebuilt_normal = pixels(row["normal"], "RGBA")
            released_normal = pixels(other["normal"], "RGBA")
            normal_delta = np.abs(
                rebuilt_normal.astype(np.int16) - released_normal.astype(np.int16)
            )
            normal_different += int(np.count_nonzero(normal_delta))
            normal_total += int(rebuilt_normal.size)
            normal_max = max(normal_max, int(normal_delta.max()))
        if args.progress_every and (
            row_index % args.progress_every == 0 or row_index == total_rows
        ):
            print(
                f"verified {row_index:,}/{total_rows:,} rebuilt rows "
                f"({100 * row_index / max(1, total_rows):.1f}%)",
                file=sys.stderr,
                flush=True,
            )

    label_mismatches = {key: count for key, count in label_mismatches.items() if count}
    result = {
        "rebuilt_rows": len(rebuilt),
        "matched_rows": len(released),
        "missing_sample_ids": missing,
        "label_mismatches": label_mismatches,
        "root_heading_rows_outside_tolerance": root_heading_mismatches,
        "root_heading_max_abs_difference": root_heading_max_abs_difference,
        "root_heading_atol": ROOT_HEADING_ATOL,
        "color_channel_difference_fraction": color_different / max(1, color_total),
        "color_max_channel_delta": color_max,
        "seg_pixel_difference_fraction": seg_different / max(1, seg_total),
        "joint_visibility_bit_difference_fraction": visible_different / max(1, visible_total),
        "joint_visibility_contract": (
            "joint_visible is true when a segment owns at least one raster pixel; "
            "one boundary pixel can therefore flip the stored bit"
        ),
        "depth_pixel_difference_fraction": (
            depth_different / max(1, depth_total) if full_resolution else None
        ),
        "depth_max_value_delta": depth_max if full_resolution else None,
        "normal_channel_difference_fraction": (
            normal_different / max(1, normal_total) if full_resolution else None
        ),
        "normal_max_channel_delta": normal_max if full_resolution else None,
        "reference_config": "frames" if full_resolution else "mini",
        "thresholds": {
            "max_color_channel_fraction": args.max_color_channel_fraction,
            "max_seg_pixel_fraction": args.max_seg_pixel_fraction,
            "max_visible_bit_fraction": args.max_visible_bit_fraction,
            "max_depth_pixel_fraction": args.max_depth_pixel_fraction,
            "max_normal_channel_fraction": args.max_normal_channel_fraction,
        },
    }
    result["passed"] = (
        not missing
        and not label_mismatches
        and not root_heading_mismatches
        and result["color_channel_difference_fraction"] <= args.max_color_channel_fraction
        and result["seg_pixel_difference_fraction"] <= args.max_seg_pixel_fraction
        and result["joint_visibility_bit_difference_fraction"] <= args.max_visible_bit_fraction
        and (
            not full_resolution
            or result["depth_pixel_difference_fraction"] <= args.max_depth_pixel_fraction
        )
        and (
            not full_resolution
            or result["normal_channel_difference_fraction"] <= args.max_normal_channel_fraction
        )
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
