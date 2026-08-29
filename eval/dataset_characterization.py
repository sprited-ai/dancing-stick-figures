"""Characterise the complete released dataset from the training caches.

The v0.2 cache stores rendered clips after prompt curation.  ``legacy_cache``
is the pre-curation cache whose clip ids are a strict superset; its projected
rig array is reused only for retained ids.  This avoids treating the three
camera views as independent source motions and avoids decoding Parquet images.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


JOINTS = [
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandEnd",
    "RightHandThumb1", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandEnd", "LeftHandThumb1", "RightUpLeg", "RightLeg", "RightFoot",
    "RightToeBase", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
]
JOINT_INDEX = {name: index for index, name in enumerate(JOINTS)}
PARTS = {
    "arm_L": ("LeftArm", "LeftForeArm"),
    "fore_L": ("LeftForeArm", "LeftHand"),
    "arm_R": ("RightArm", "RightForeArm"),
    "fore_R": ("RightForeArm", "RightHand"),
    "leg_L": ("LeftUpLeg", "LeftLeg"),
    "shin_L": ("LeftLeg", "LeftFoot"),
    "leg_R": ("RightUpLeg", "RightLeg"),
    "shin_R": ("RightLeg", "RightFoot"),
}
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _summary(values: list[float] | np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"n": 0, "quantiles": {str(q): None for q in QUANTILES}}
    return {
        "n": int(len(finite)),
        "mean": float(finite.mean()),
        "quantiles": {
            str(q): float(value)
            for q, value in zip(QUANTILES, np.quantile(finite, QUANTILES))
        },
    }


def _ks_distance(left: list[float], right: list[float]) -> float | None:
    a = np.sort(np.asarray(left, dtype=np.float64))
    b = np.sort(np.asarray(right, dtype=np.float64))
    if not len(a) or not len(b):
        return None
    points = np.sort(np.concatenate([a, b]))
    return float(np.max(np.abs(
        np.searchsorted(a, points, side="right") / len(a)
        - np.searchsorted(b, points, side="right") / len(b)
    )))


def _motion_id(clip_id: str) -> str:
    return clip_id.rsplit("/c", 1)[0]


def _camera_index(clip_id: str) -> int:
    return int(clip_id.rsplit("/c", 1)[1])


def _centroid_speed_and_foreground(frames: np.ndarray) -> tuple[float, float]:
    alpha = frames[..., 3] > 127
    foreground = float(alpha.mean())
    height, width = alpha.shape[1:]
    xs = np.arange(width, dtype=np.float64)[None, None, :]
    ys = np.arange(height, dtype=np.float64)[None, :, None]
    mass = alpha.sum(axis=(1, 2)).astype(np.float64)
    cx = np.divide((alpha * xs).sum(axis=(1, 2)), mass,
                   out=np.full(len(alpha), np.nan), where=mass > 0)
    cy = np.divide((alpha * ys).sum(axis=(1, 2)), mass,
                   out=np.full(len(alpha), np.nan), where=mass > 0)
    displacement = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
    finite = displacement[np.isfinite(displacement)]
    return (float(finite.mean()) if len(finite) else float("nan"), foreground)


def _rig_paths(rig: np.ndarray) -> dict[str, float]:
    paths = {}
    for name, (parent, child) in PARTS.items():
        vector = rig[:, JOINT_INDEX[child]] - rig[:, JOINT_INDEX[parent]]
        angle = np.arctan2(vector[:, 1], vector[:, 0])
        step = np.diff(angle)
        # Principal axes are undirected, matching the evaluator's pi-periodic
        # limb-mask orientation.
        step = (step + np.pi / 2) % np.pi - np.pi / 2
        paths[name] = float(np.abs(step[np.isfinite(step)]).sum())
    return paths


def _parameter_ranges(parquet_dir: str | Path, retained: set[str]) -> dict:
    import pyarrow.dataset as ds

    paths = sorted(str(path) for path in Path(parquet_dir).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no Parquet shards under {parquet_dir}")
    columns = [
        "clip_id", "frame_idx", "cam_yaw", "cam_pitch", "cam_center_x",
        "cam_center_y", "px_per_m", "stroke", "bone_scale",
    ]
    table = ds.dataset(paths, format="parquet").to_table(
        columns=columns, filter=ds.field("frame_idx") == 0,
    )
    rows = [row for row in table.to_pylist() if row["clip_id"] in retained]
    if len(rows) != len(retained):
        raise ValueError(f"expected {len(retained)} retained clips, found {len(rows)} parameter rows")
    result = {}
    for key in columns[2:-1]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = {"min": float(values.min()), "max": float(values.max())}
    bone_values = [
        float(value)
        for row in rows
        for value in json.loads(row["bone_scale"]).values()
    ]
    result["bone_scale_all_named_bones"] = {
        "min": float(min(bone_values)), "max": float(max(bone_values)),
    }
    return result


def characterise(
    release_cache: str | Path,
    legacy_cache: str | Path,
    frame_parquet: str | Path | None = None,
) -> dict:
    release_cache, legacy_cache = Path(release_cache), Path(legacy_cache)
    release = json.loads((release_cache / "clips.json").read_text())
    legacy = json.loads((legacy_cache / "clips.json").read_text())
    frames = np.load(release_cache / "frames.npy", mmap_mode="r")
    rig = np.load(legacy_cache / "rig.npy", mmap_mode="r")

    retained = set(release)
    if not retained.issubset(legacy):
        missing = sorted(retained - set(legacy))[:3]
        raise ValueError(f"release ids absent from legacy rig cache: {missing}")

    source = {}
    for clip_id, meta in release.items():
        source.setdefault(_motion_id(clip_id), meta)

    source_by_group = Counter(meta["group"] for meta in source.values())
    source_by_split = Counter(meta["split"] for meta in source.values())
    source_group_split = Counter((meta["group"], meta["split"]) for meta in source.values())
    qa_overall = Counter(meta.get("qa", "") or "none" for meta in source.values())
    qa_group = defaultdict(Counter)
    qa_split = defaultdict(Counter)
    for meta in source.values():
        flag = meta.get("qa", "") or "none"
        qa_group[meta["group"]][flag] += 1
        qa_split[meta["split"]][flag] += 1

    metrics = defaultdict(list)
    split_metrics = defaultdict(lambda: defaultdict(list))
    group_metrics = defaultdict(lambda: defaultdict(list))
    part_metrics = defaultdict(list)

    for clip_id, meta in release.items():
        start, count = int(meta["start"]), int(meta["n"])
        speed, foreground = _centroid_speed_and_foreground(frames[start:start + count])
        metrics["foreground_fraction"].append(foreground)
        group_metrics[meta["group"]]["foreground_fraction"].append(foreground)
        split_metrics[meta["split"]]["foreground_fraction"].append(foreground)

        # Motion distributions use one fixed camera per source trajectory.
        if _camera_index(clip_id) != 0:
            continue
        old = legacy[clip_id]
        old_start, old_count = int(old["start"]), int(old["n"])
        paths = _rig_paths(np.asarray(rig[old_start:old_start + old_count], dtype=np.float32))
        metrics["centroid_speed_px_per_frame_camera0"].append(speed)
        split_metrics[meta["split"]]["centroid_speed_px_per_frame_camera0"].append(speed)
        group_metrics[meta["group"]]["centroid_speed_px_per_frame_camera0"].append(speed)
        total_path = float(sum(paths.values()))
        metrics["rig_part_angular_path_rad_camera0"].append(total_path)
        split_metrics[meta["split"]]["rig_part_angular_path_rad_camera0"].append(total_path)
        group_metrics[meta["group"]]["rig_part_angular_path_rad_camera0"].append(total_path)
        for part, value in paths.items():
            part_metrics[part].append(value)

    body_path = legacy_cache / "body_params.json"
    body_ranges = {}
    if body_path.exists():
        body = json.loads(body_path.read_text())
        selected = [body[clip_id] for clip_id in release]
        for key in sorted(selected[0]):
            values = [row[key] for row in selected if isinstance(row.get(key), (int, float))]
            if values:
                body_ranges[key] = {"min": float(min(values)), "max": float(max(values))}

    split_comparisons = {}
    for key in sorted(split_metrics["train"]):
        split_comparisons[key] = {
            "train_vs_val_ks": _ks_distance(split_metrics["train"][key], split_metrics["val"][key]),
            "train_vs_test_ks": _ks_distance(split_metrics["train"][key], split_metrics["test"][key]),
        }

    return {
        "release_cache": str(release_cache),
        "legacy_rig_cache": str(legacy_cache),
        "counts": {
            "prompts": len({meta["text"] for meta in source.values()}),
            "source_motions": len(source),
            "rendered_clips": len(release),
            "frames": int(sum(int(meta["n"]) for meta in release.values())),
            "source_motions_by_group": dict(sorted(source_by_group.items())),
            "prompts_by_group": {
                group: len({meta["text"] for meta in source.values() if meta["group"] == group})
                for group in sorted(source_by_group)
            },
            "source_motions_by_split": dict(sorted(source_by_split.items())),
            "source_motions_by_group_and_split": {
                f"{group}/{split}": count
                for (group, split), count in sorted(source_group_split.items())
            },
        },
        "qa_flags_source_motion": {
            "overall": dict(sorted(qa_overall.items())),
            "by_group": {key: dict(sorted(value.items())) for key, value in sorted(qa_group.items())},
            "by_split": {key: dict(sorted(value.items())) for key, value in sorted(qa_split.items())},
        },
        "distributions": {key: _summary(value) for key, value in sorted(metrics.items())},
        "by_split": {
            split: {key: _summary(value) for key, value in sorted(rows.items())}
            for split, rows in sorted(split_metrics.items())
        },
        "split_comparisons_ks": split_comparisons,
        "by_group": {
            group: {key: _summary(value) for key, value in sorted(rows.items())}
            for group, rows in sorted(group_metrics.items())
        },
        "part_angular_path_camera0": {
            part: _summary(value) for part, value in sorted(part_metrics.items())
        },
        "body_parameter_ranges": body_ranges,
        "released_frame_parameter_ranges": (
            _parameter_ranges(frame_parquet, retained) if frame_parquet is not None else {}
        ),
        "notes": {
            "unit_of_qa_prevalence": "source motion; three camera views are deduplicated",
            "foreground_unit": "rendered clip across all frames and pixels",
            "motion_unit": "one camera-0 rendered clip per source motion",
            "rig_path_definition": "sum of absolute pi-periodic projected bone-angle changes over 120 frames",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-cache", required=True)
    parser.add_argument("--legacy-cache", required=True)
    parser.add_argument("--frame-parquet")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = characterise(args.release_cache, args.legacy_cache, args.frame_parquet)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["counts"], indent=2))
    print(json.dumps(payload["qa_flags_source_motion"]["overall"], indent=2))


if __name__ == "__main__":
    main()
