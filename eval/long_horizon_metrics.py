"""Model-agnostic long-horizon diagnostics for M3/M4 comparisons.

The same fixed frame indices are used for one-shot videos (virtual boundaries)
and autoregressive videos (true chunk boundaries).  This prevents choosing a
seam metric after seeing the AR model.  Repetition is reported separately from
motion amount: a frozen video can be perfectly repetitive while also having
zero motion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from eval.oracle import NAMES, label_colours, score_seams, score_video


DEFAULT_BOUNDARIES = (16, 24, 32, 40, 48)
DRIFT_KEYS = ("tvr", "lie", "cpe", "fg", "mass_drift", "centroid_speed",
              "centroid_accel", "motion_fraction", "angle_speed", "angle_jerk",
              "height_var")
BOUNDARY_SIGNALS = ("centroid_speed", "centroid_accel", "angle_speed", "angle_jerk")
MIN_PART_PIXELS = 4


def _finite_mean(values: Iterable[float]) -> float:
    values = np.asarray(list(values), np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _slope(values: Sequence[float]) -> float:
    y = np.asarray(values, np.float64)
    good = np.isfinite(y)
    if good.sum() < 2:
        return float("nan")
    x = np.linspace(0.0, 1.0, len(y))
    return float(np.polyfit(x[good], y[good], 1)[0])


def binned_drift(frames: np.ndarray, bins: int = 5) -> dict:
    """Score contiguous time bins and fit change from start to end.

    Slopes are expressed per normalized video duration.  They are descriptive,
    not one-sided errors: for example, motion can legitimately increase.
    """
    if frames.ndim != 4 or frames.shape[-1] != 4:
        raise ValueError("expected uint8 RGBA video shaped [T,H,W,4]")
    if bins < 2 or bins > frames.shape[0]:
        raise ValueError("bins must be between 2 and the number of frames")
    edges = np.linspace(0, frames.shape[0], bins + 1, dtype=int)
    rows = [score_video(frames[edges[i]:edges[i + 1]]) for i in range(bins)]
    return {
        "frame_ranges": [[int(edges[i]), int(edges[i + 1])] for i in range(bins)],
        "bins": [{key: float(row[key]) for key in DRIFT_KEYS} for row in rows],
        "slope": {key: _slope([row[key] for row in rows]) for key in DRIFT_KEYS},
    }


def _pose_features(frames: np.ndarray) -> np.ndarray:
    """Palette-derived pose descriptors for lagged self-similarity.

    Per colour we store area fraction and normalized centroid.  Missing colours
    use (-1,-1), making disappearance observable instead of silently imputed.
    """
    _, h, w, _ = frames.shape
    rows = []
    for frame in frames:
        labels, _ = label_colours(frame)
        feature = []
        for index in range(len(NAMES)):
            ys, xs = np.nonzero(labels == index)
            feature.append(float(len(xs) / (h * w)))
            if len(xs):
                feature.extend((float(xs.mean() / max(w - 1, 1)),
                                float(ys.mean() / max(h - 1, 1))))
            else:
                feature.extend((-1.0, -1.0))
        rows.append(feature)
    return np.asarray(rows, np.float32)


def _longest_false_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = 0 if value else current + 1
        best = max(best, current)
    return int(best)


def palette_part_trajectories(frames: np.ndarray) -> dict:
    """Extract visible colour-part centroids without a learned pose model.

    Absolute centroids describe screen-space/root translation.  Relative
    centroids subtract the ``ink`` centroid and divide by visible body height,
    isolating articulated motion from whole-figure translation.  Missing parts
    remain NaN and are summarized as visibility/dropout rather than imputed.
    """
    if frames.ndim != 4 or frames.shape[-1] != 4:
        raise ValueError("expected uint8 RGBA video shaped [T,H,W,4]")
    t, h, w, _ = frames.shape
    centroids = np.full((t, len(NAMES), 2), np.nan, np.float64)
    areas = np.zeros((t, len(NAMES)), np.float64)
    scales = np.full(t, np.nan, np.float64)
    for frame_index, frame in enumerate(frames):
        labels, fg = label_colours(frame)
        ys_fg, _ = np.nonzero(fg)
        if len(ys_fg) >= 2:
            scales[frame_index] = max(float(ys_fg.max() - ys_fg.min()), 1.0)
        for part_index in range(len(NAMES)):
            ys, xs = np.nonzero(labels == part_index)
            areas[frame_index, part_index] = len(xs) / (h * w)
            if len(xs) >= MIN_PART_PIXELS:
                centroids[frame_index, part_index] = (xs.mean(), ys.mean())
    ink_index = NAMES.index("ink")
    root = centroids[:, ink_index]
    relative = (centroids - root[:, None, :]) / scales[:, None, None]
    return {
        "centroids_px": centroids,
        "relative_centroids": relative,
        "area_fraction": areas,
        "body_scale_px": scales,
    }


def _valid_step_norm(series: np.ndarray, order: int = 1) -> np.ndarray:
    """Finite-difference norm with NaN whenever any required frame is missing."""
    values = np.asarray(series, np.float64)
    for _ in range(order):
        values = np.diff(values, axis=0)
    return np.linalg.norm(values, axis=-1)


def part_motion_diagnostic(frames: np.ndarray,
                           boundaries: Sequence[int] = DEFAULT_BOUNDARIES) -> dict:
    """Summarize per-part visibility, motion energy, and seam-local jumps."""
    tracks = palette_part_trajectories(frames)
    absolute = tracks["centroids_px"]
    relative = tracks["relative_centroids"]
    boundaries = {int(v) for v in boundaries if 0 < int(v) < len(frames)}
    parts = {}
    for index, name in enumerate(NAMES):
        visible = np.isfinite(absolute[:, index]).all(axis=1)
        rel_speed = _valid_step_norm(relative[:, index], order=1)
        rel_accel = _valid_step_norm(relative[:, index], order=2)
        destinations = np.arange(1, len(frames))
        seam_mask = np.asarray([value in boundaries for value in destinations])
        seam = rel_speed[seam_mask]
        within = rel_speed[~seam_mask]
        seam_mean, within_mean = _finite_mean(seam), _finite_mean(within)
        parts[name] = {
            "visibility": float(visible.mean()),
            "longest_dropout": _longest_false_run(visible),
            "relative_motion_energy": _finite_mean(rel_speed),
            "relative_acceleration": _finite_mean(rel_accel),
            "boundary_relative_speed": seam_mean,
            "within_relative_speed": within_mean,
            "boundary_relative_excess": (
                float(seam_mean - within_mean)
                if np.isfinite(seam_mean) and np.isfinite(within_mean) else float("nan")
            ),
        }
    root_speed = _valid_step_norm(absolute[:, NAMES.index("ink")], order=1)
    root_accel = _valid_step_norm(absolute[:, NAMES.index("ink")], order=2)
    limb_names = [name for name in NAMES if name != "ink"]
    return {
        "root_motion_energy_px": _finite_mean(root_speed),
        "root_acceleration_px": _finite_mean(root_accel),
        "mean_limb_relative_motion": _finite_mean(
            parts[name]["relative_motion_energy"] for name in limb_names
        ),
        "mean_limb_boundary_excess": _finite_mean(
            parts[name]["boundary_relative_excess"] for name in limb_names
        ),
        "parts": parts,
        "claim_limit": "Colour centroids represent visible limb segments, not anatomical joint locations.",
    }


def prompt_attribute_features(frames: np.ndarray) -> dict:
    """Low-dimensional, interpretable motion attributes for prompt subsets."""
    result = part_motion_diagnostic(frames, boundaries=())
    parts = result["parts"]
    energy = lambda name: parts[name]["relative_motion_energy"]
    left_arm = _finite_mean((energy("arm_L"), energy("fore_L")))
    right_arm = _finite_mean((energy("arm_R"), energy("fore_R")))
    left_leg = _finite_mean((energy("leg_L"), energy("shin_L")))
    right_leg = _finite_mean((energy("leg_R"), energy("shin_R")))
    tracks = palette_part_trajectories(frames)
    ink_y = tracks["centroids_px"][:, NAMES.index("ink"), 1]
    finite_y = ink_y[np.isfinite(ink_y)]
    vertical_extent = float(finite_y.max() - finite_y.min()) if finite_y.size else float("nan")
    return {
        "left_arm_energy": left_arm,
        "right_arm_energy": right_arm,
        "left_leg_energy": left_leg,
        "right_leg_energy": right_leg,
        "arm_laterality": float(left_arm - right_arm),
        "leg_laterality": float(left_leg - right_leg),
        "root_motion_energy_px": result["root_motion_energy_px"],
        "root_vertical_extent_px": vertical_extent,
    }


def repetition_diagnostic(frames: np.ndarray, min_lag: int = 4,
                          max_lag: int | None = None) -> dict:
    """Find non-trivial repeated poses without confusing it with motion quality.

    ``similarity`` is 1 for an exact repeated cycle at the best lag and tends
    toward 0 when that lag is no more similar than the typical candidate lag.
    Always interpret it beside ``motion_fraction`` from :func:`score_video`.
    """
    features = _pose_features(frames)
    t = len(features)
    max_lag = min(t // 2, max_lag or t // 2)
    if min_lag > max_lag:
        raise ValueError("video is too short for the requested repetition lags")
    distances = {}
    for lag in range(min_lag, max_lag + 1):
        distances[lag] = float(np.abs(features[lag:] - features[:-lag]).mean())
    best_lag = min(distances, key=distances.get)
    best = distances[best_lag]
    typical = float(np.median(list(distances.values())))
    similarity = float(np.clip(1.0 - best / max(typical, 1e-12), 0.0, 1.0))
    return {
        "best_lag": int(best_lag),
        "best_lag_distance": best,
        "median_candidate_distance": typical,
        "similarity": similarity,
        "candidate_lags": {str(k): v for k, v in distances.items()},
        "claim_limit": "High similarity also occurs for frozen video; read with motion_fraction.",
    }


def boundary_diagnostic(frames: np.ndarray,
                        boundaries: Sequence[int] = DEFAULT_BOUNDARIES) -> dict:
    """Compare fixed boundary transitions with all other transitions."""
    boundaries = [int(v) for v in boundaries if 0 < int(v) < len(frames)]
    raw = score_seams(frames, boundaries)
    ratios = {}
    for signal in BOUNDARY_SIGNALS:
        seam = raw[f"seam_{signal}"]
        within = raw[f"within_{signal}"]
        ratios[f"{signal}_ratio"] = (
            float(seam / within) if np.isfinite(seam) and np.isfinite(within) and within > 0
            else float("nan")
        )
        ratios[f"{signal}_excess"] = (
            float(seam - within) if np.isfinite(seam) and np.isfinite(within)
            else float("nan")
        )
    return {"frames": boundaries, "raw": raw, **ratios}


def score_long_horizon(frames: np.ndarray, bins: int = 5,
                       boundaries: Sequence[int] = DEFAULT_BOUNDARIES) -> dict:
    base = score_video(frames)
    return {
        "whole_video": base,
        "time_drift": binned_drift(frames, bins=bins),
        "boundary": boundary_diagnostic(frames, boundaries=boundaries),
        "repetition": repetition_diagnostic(frames),
        "part_motion": part_motion_diagnostic(frames, boundaries=boundaries),
        "prompt_attributes": prompt_attribute_features(frames),
    }


def aggregate(rows: Sequence[dict]) -> dict:
    """Aggregate per-video diagnostics without hiding the raw rows."""
    summary = {
        "whole_video": {
            key: _finite_mean(row["whole_video"][key] for row in rows)
            for key in DRIFT_KEYS
        },
        "drift_slope": {
            key: _finite_mean(row["time_drift"]["slope"][key] for row in rows)
            for key in DRIFT_KEYS
        },
        "boundary": {},
        "repetition": {
            "similarity": _finite_mean(row["repetition"]["similarity"] for row in rows),
            "best_lag_distance": _finite_mean(row["repetition"]["best_lag_distance"] for row in rows),
        },
        "part_motion": {
            "root_motion_energy_px": _finite_mean(
                row["part_motion"]["root_motion_energy_px"] for row in rows),
            "root_acceleration_px": _finite_mean(
                row["part_motion"]["root_acceleration_px"] for row in rows),
            "mean_limb_relative_motion": _finite_mean(
                row["part_motion"]["mean_limb_relative_motion"] for row in rows),
            "mean_limb_boundary_excess": _finite_mean(
                row["part_motion"]["mean_limb_boundary_excess"] for row in rows),
        },
        "prompt_attributes": {
            key: _finite_mean(row["prompt_attributes"][key] for row in rows)
            for key in ("left_arm_energy", "right_arm_energy", "left_leg_energy",
                        "right_leg_energy", "arm_laterality", "leg_laterality",
                        "root_motion_energy_px", "root_vertical_extent_px")
        },
    }
    for signal in BOUNDARY_SIGNALS:
        for suffix in ("ratio", "excess"):
            key = f"{signal}_{suffix}"
            summary["boundary"][key] = _finite_mean(row["boundary"][key] for row in rows)
    bin_count = len(rows[0]["time_drift"]["bins"])
    summary["time_bins"] = [
        {key: _finite_mean(row["time_drift"]["bins"][i][key] for row in rows)
         for key in DRIFT_KEYS}
        for i in range(bin_count)
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="archive containing uint8 rgba [N,T,H,W,4]")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--boundaries", default="16,24,32,40,48")
    args = parser.parse_args()
    archive = np.load(args.npz, allow_pickle=False)
    videos = archive["rgba"]
    boundaries = [int(v) for v in args.boundaries.split(",") if v.strip()]
    rows = [score_long_horizon(video, bins=args.bins, boundaries=boundaries) for video in videos]
    result = {
        "protocol": "long_horizon_m3_m4_v1",
        "source_npz": str(Path(args.npz).resolve()),
        "n": int(len(videos)),
        "shape": list(videos.shape[1:]),
        "boundary_policy": {
            "frames": boundaries,
            "m3": "virtual control boundaries",
            "m4": "true AR chunk boundaries when the chunk schedule matches",
        },
        "summary": aggregate(rows),
        "per_video": rows,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps({"out": str(destination), "n": len(videos), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
