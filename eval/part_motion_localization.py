"""Validate whether limb-wise motion scores localise a targeted rig corruption.

For each held-out source motion, the more active arm (by clean rendered angular
path) is selected.  Its connected chain is progressively frozen toward the
first-frame local directions while preserving attachment and every bone length.
The released renderer then produces the corrupted videos.  We compare angular
path changes for the two affected colour parts against the six unaffected limb
parts and separately report visible-topology changes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from eval.corrupt import body_cam, render_joints
from eval.oracle import LIMBS, score_video
from generator.skeleton import IDX, NAMES, PARENT


CHAINS = {
    "L": ("LeftForeArm", "LeftHand", "LeftHandEnd", "LeftHandThumb1"),
    "R": ("RightForeArm", "RightHand", "RightHandEnd", "RightHandThumb1"),
}
TARGET_PARTS = {"L": ("arm_L", "fore_L"), "R": ("arm_R", "fore_R")}


def freeze_chain(joints: np.ndarray, side: str, severity: float) -> np.ndarray:
    """Blend one arm's local directions toward frame 0, preserving lengths."""
    if side not in CHAINS:
        raise ValueError(f"side must be one of {sorted(CHAINS)}, got {side!r}")
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must lie in [0, 1]")
    clean = np.asarray(joints, dtype=np.float64)
    result = clean.copy()
    for child_name in CHAINS[side]:
        parent_name = PARENT[child_name]
        child, parent = IDX[child_name], IDX[parent_name]
        reference = clean[0, child] - clean[0, parent]
        reference /= max(float(np.linalg.norm(reference)), 1e-12)
        for frame in range(len(clean)):
            vector = clean[frame, child] - clean[frame, parent]
            length = float(np.linalg.norm(vector))
            direction = vector / max(length, 1e-12)
            blended = (1.0 - severity) * direction + severity * reference
            norm = float(np.linalg.norm(blended))
            if norm < 1e-8:
                blended = reference
                norm = 1.0
            result[frame, child] = result[frame, parent] + length * blended / norm
    return result.astype(np.float32)


def _clip_ids(clips_path: Path, split: str, n: int, seed: int) -> list[str]:
    clips = json.loads(clips_path.read_text())
    groups = defaultdict(list)
    for clip_id, meta in clips.items():
        if not clip_id.endswith("/c0") or meta["split"] != split or meta.get("qa"):
            continue
        groups[meta["group"]].append(clip_id)
    rng = np.random.default_rng(seed)
    for values in groups.values():
        values.sort()
        rng.shuffle(values)
    selected = []
    ordered_groups = sorted(groups)
    while len(selected) < n and any(groups.values()):
        for group in ordered_groups:
            if groups[group] and len(selected) < n:
                selected.append(groups[group].pop())
    if len(selected) < n:
        raise ValueError(f"requested {n} clips but found {len(selected)} clean camera-0 clips")
    return selected


def _load_rows(parquet_dir: Path, clip_ids: list[str]) -> dict[str, list[dict]]:
    paths = sorted(parquet_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no Parquet shards under {parquet_dir}")
    columns = [
        "clip_id", "frame_idx", "joint_xyz", "cam_yaw", "cam_pitch",
        "cam_center_x", "cam_center_y", "px_per_m", "stroke", "bone_scale",
    ]
    table = ds.dataset(paths, format="parquet").to_table(
        columns=columns, filter=ds.field("clip_id").isin(clip_ids),
    )
    grouped = defaultdict(list)
    for row in table.to_pylist():
        grouped[row["clip_id"]].append(row)
    for clip_id in clip_ids:
        grouped[clip_id].sort(key=lambda row: row["frame_idx"])
        if len(grouped[clip_id]) != 120:
            raise ValueError(f"expected 120 frames for {clip_id}, found {len(grouped[clip_id])}")
    return grouped


def _render_score(joints: np.ndarray, row: dict) -> dict:
    body, camera = body_cam(row)
    frames = []
    for xyz in joints:
        pose = {name: tuple(map(float, xyz[index])) for index, name in enumerate(NAMES)}
        frames.append(render_joints(pose, body, camera))
    return score_video(np.asarray(frames))


def _interval(values: list[float], rng: np.random.Generator, draws: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    medians = np.median(array[rng.integers(0, len(array), size=(draws, len(array)))], axis=1)
    low, high = np.quantile(medians, (0.025, 0.975))
    return {"median": float(np.median(array)), "bootstrap_ci95": [float(low), float(high)]}


def run(
    parquet_dir: str | Path,
    clips_json: str | Path,
    *,
    n: int = 24,
    split: str = "test",
    seed: int = 20260827,
    severities: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    bootstrap_draws: int = 2000,
) -> dict:
    clip_ids = _clip_ids(Path(clips_json), split, n, seed)
    grouped = _load_rows(Path(parquet_dir), clip_ids)
    records = []
    for clip_id in clip_ids:
        rows = grouped[clip_id]
        joints = np.stack([
            np.frombuffer(row["joint_xyz"], np.float32).reshape(len(NAMES), 3)
            for row in rows
        ])
        clean = _render_score(joints, rows[0])
        side_totals = {
            side: sum(clean[f"ang_path_{part}"] for part in TARGET_PARTS[side])
            for side in TARGET_PARTS
        }
        side = max(side_totals, key=side_totals.get)
        target = set(TARGET_PARTS[side])
        unaffected = [part for part in LIMBS if part not in target]
        for severity in severities:
            corrupted_joints = freeze_chain(joints, side, severity)
            corrupted = _render_score(corrupted_joints, rows[0])
            target_drop = sum(
                clean[f"ang_path_{part}"] - corrupted[f"ang_path_{part}"]
                for part in target
            )
            unaffected_change = float(np.mean([
                abs(corrupted[f"ang_path_{part}"] - clean[f"ang_path_{part}"])
                for part in unaffected
            ]))
            records.append({
                "clip_id": clip_id,
                "selected_side": side,
                "severity": severity,
                "clean_target_path": side_totals[side],
                "target_path_drop": float(target_drop),
                "unaffected_part_abs_change": unaffected_change,
                "target_drop_exceeds_unaffected_change": bool(target_drop > unaffected_change),
                "tvr_abs_change": abs(float(corrupted["tvr"] - clean["tvr"])),
                "lie_abs_change": abs(float(corrupted["lie"] - clean["lie"])),
            })

    rng = np.random.default_rng(seed + 1)
    by_severity = {}
    for severity in severities:
        rows = [record for record in records if record["severity"] == severity]
        by_severity[str(severity)] = {
            "target_path_drop": _interval([row["target_path_drop"] for row in rows], rng, bootstrap_draws),
            "unaffected_part_abs_change": _interval(
                [row["unaffected_part_abs_change"] for row in rows], rng, bootstrap_draws,
            ),
            "fraction_target_drop_exceeds_unaffected_change": float(np.mean([
                row["target_drop_exceeds_unaffected_change"] for row in rows
            ])),
            "tvr_abs_change": _interval([row["tvr_abs_change"] for row in rows], rng, bootstrap_draws),
            "lie_abs_change": _interval([row["lie_abs_change"] for row in rows], rng, bootstrap_draws),
        }
    return {
        "n_clips": n,
        "split": split,
        "camera_policy": "camera 0, one rendered view per source motion",
        "clip_sampling": "seeded round-robin across groups; QA-flagged motions excluded",
        "target_policy": "more active clean rendered arm; both coloured arm parts are targeted",
        "corruption": "blend local arm-chain directions toward frame 0 while preserving attachment and bone lengths",
        "severities": list(severities),
        "bootstrap_draws": bootstrap_draws,
        "seed": seed,
        "by_severity": by_severity,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--clips-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    payload = run(
        args.parquet_dir, args.clips_json, n=args.n, split=args.split,
        seed=args.seed, bootstrap_draws=args.bootstrap_draws,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["by_severity"], indent=2))


if __name__ == "__main__":
    main()
