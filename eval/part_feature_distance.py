"""Dataset-specific Fréchet distance over palette-part geometry and motion.

This prototype converts each frame into interpretable measurements for the
nine released colour parts.  It reports a static pose-distribution distance
and a motion-aware distance that appends first and second temporal differences.
Unlike I3D FVD, every input dimension has a direct geometric meaning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eval.fvd import frechet
from eval.oracle import COLS, NAMES, TAU


PART_FEATURES = (
    "present", "area_fraction", "relative_centroid_x", "relative_centroid_y",
    "relative_major_length", "relative_minor_length", "elongation",
    "torso_relative_axis_cos2", "torso_relative_axis_sin2",
)


def label_video(rgba: np.ndarray, chunk: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Return palette labels and alpha foreground for uint8 [T,H,W,4]."""
    labels, foreground = [], []
    for start in range(0, len(rgba), chunk):
        value = rgba[start:start + chunk]
        alpha = value[..., 3] > 127
        rgb = value[..., :3].astype(np.float32)
        distance = np.linalg.norm(rgb[..., None, :] - COLS[None, None, None, :, :], axis=-1)
        nearest = distance.argmin(axis=-1).astype(np.int16)
        nearest = np.where(alpha & (distance.min(axis=-1) < TAU), nearest, -1)
        labels.append(nearest)
        foreground.append(alpha)
    return np.concatenate(labels), np.concatenate(foreground)


def frame_part_features(rgba: np.ndarray) -> np.ndarray:
    """Extract camera-normalised part geometry as [T, 2 + 9*9].

    Part centroids are relative to the ink/torso centroid, lengths are divided
    by the video's median foreground height, and axes are relative to the
    torso's principal axis.  This removes image translation, in-plane body
    rotation, and most zoom variation without requiring rig or camera labels.
    """
    rgba = np.asarray(rgba)
    if rgba.ndim != 4 or rgba.shape[-1] != 4:
        raise ValueError("expected uint8 [T,H,W,4]")
    labels, foreground = label_video(rgba)
    time, height, width = labels.shape
    yy, xx = np.mgrid[:height, :width]
    fg_count = foreground.sum(axis=(1, 2)).astype(np.float64)
    body_heights = np.zeros(time, np.float64)
    for frame_index, mask in enumerate(foreground):
        ys = np.flatnonzero(mask.any(axis=1))
        body_heights[frame_index] = float(ys[-1] - ys[0] + 1) if len(ys) else np.nan
    valid_heights = body_heights[np.isfinite(body_heights) & (body_heights > 0)]
    body_scale = float(np.median(valid_heights)) if len(valid_heights) else float(max(height, 1))
    assigned = labels >= 0
    output = np.zeros((time, 2 + len(NAMES) * len(PART_FEATURES)), np.float64)
    output[:, 0] = fg_count / (height * width)
    output[:, 1] = np.divide(
        (foreground & ~assigned).sum(axis=(1, 2)), fg_count,
        out=np.zeros(time, np.float64), where=fg_count > 0,
    )

    counts = np.zeros((time, len(NAMES)), np.float64)
    valid = np.zeros((time, len(NAMES)), bool)
    mean_x = np.zeros((time, len(NAMES)), np.float64)
    mean_y = np.zeros((time, len(NAMES)), np.float64)
    major_var = np.zeros((time, len(NAMES)), np.float64)
    minor_var = np.zeros((time, len(NAMES)), np.float64)
    angle = np.zeros((time, len(NAMES)), np.float64)
    for part_index in range(len(NAMES)):
        mask = labels == part_index
        count = mask.sum(axis=(1, 2)).astype(np.float64)
        part_valid = count >= 4
        safe = np.maximum(count, 1.0)
        part_mean_x = (mask * xx).sum(axis=(1, 2)) / safe
        part_mean_y = (mask * yy).sum(axis=(1, 2)) / safe
        second_x = (mask * (xx * xx)).sum(axis=(1, 2)) / safe
        second_y = (mask * (yy * yy)).sum(axis=(1, 2)) / safe
        cross = (mask * (xx * yy)).sum(axis=(1, 2)) / safe
        cov_xx = np.maximum(second_x - part_mean_x * part_mean_x, 0.0)
        cov_yy = np.maximum(second_y - part_mean_y * part_mean_y, 0.0)
        cov_xy = cross - part_mean_x * part_mean_y
        trace = cov_xx + cov_yy
        discriminant = np.sqrt(np.maximum((cov_xx - cov_yy) ** 2 + 4 * cov_xy ** 2, 0.0))
        counts[:, part_index] = count
        valid[:, part_index] = part_valid
        mean_x[:, part_index] = part_mean_x
        mean_y[:, part_index] = part_mean_y
        major_var[:, part_index] = np.maximum((trace + discriminant) / 2, 0.0)
        minor_var[:, part_index] = np.maximum((trace - discriminant) / 2, 0.0)
        angle[:, part_index] = 0.5 * np.arctan2(2 * cov_xy, cov_xx - cov_yy)

    torso_valid = valid[:, 0]
    torso_x, torso_y, torso_angle = mean_x[:, 0], mean_y[:, 0], angle[:, 0]
    for part_index in range(len(NAMES)):
        part_valid = valid[:, part_index]
        joint_valid = part_valid & torso_valid
        trace = major_var[:, part_index] + minor_var[:, part_index]
        relative_angle = angle[:, part_index] - torso_angle
        values = np.stack([
            part_valid.astype(np.float64),
            np.divide(counts[:, part_index], fg_count, out=np.zeros(time), where=fg_count > 0),
            (mean_x[:, part_index] - torso_x) / body_scale,
            (mean_y[:, part_index] - torso_y) / body_scale,
            4.0 * np.sqrt(major_var[:, part_index]) / body_scale,
            4.0 * np.sqrt(minor_var[:, part_index]) / body_scale,
            np.divide(major_var[:, part_index] - minor_var[:, part_index], trace,
                      out=np.zeros(time), where=trace > 1e-8),
            np.cos(2 * relative_angle),
            np.sin(2 * relative_angle),
        ], axis=1)
        values[~joint_valid] = 0.0
        values[:, 0] = part_valid.astype(np.float64)
        values[:, 1] = np.divide(counts[:, part_index], fg_count,
                                 out=np.zeros(time), where=fg_count > 0)
        begin = 2 + part_index * len(PART_FEATURES)
        output[:, begin:begin + len(PART_FEATURES)] = values
    return output.astype(np.float32)


def temporal_part_features(static: np.ndarray) -> np.ndarray:
    """Append signed frame velocity and acceleration, preserving T rows."""
    velocity = np.zeros_like(static)
    acceleration = np.zeros_like(static)
    velocity[1:] = np.diff(static, axis=0)
    acceleration[2:] = np.diff(static, n=2, axis=0)
    return np.concatenate([static, velocity, acceleration], axis=1)


def feature_names(temporal: bool = False) -> list[str]:
    base = ["foreground_fraction", "unassigned_foreground_fraction"]
    base += [f"{part}.{feature}" for part in NAMES for feature in PART_FEATURES]
    if not temporal:
        return base
    return base + [f"delta.{name}" for name in base] + [f"delta2.{name}" for name in base]


def fit_standardizer(reference: np.ndarray, epsilon: float = 1e-6) -> dict:
    mean = reference.mean(axis=0, dtype=np.float64)
    scale = reference.std(axis=0, dtype=np.float64)
    keep = scale > epsilon
    return {"mean": mean, "scale": scale, "keep": keep}


def standardize(values: np.ndarray, fitted: dict) -> np.ndarray:
    keep = fitted["keep"]
    return ((values[:, keep] - fitted["mean"][keep]) / fitted["scale"][keep]).astype(np.float64)


def frechet_features(reference: np.ndarray, candidate: np.ndarray, fitted: dict | None = None) -> float:
    fitted = fitted or fit_standardizer(reference)
    left, right = standardize(reference, fitted), standardize(candidate, fitted)
    return frechet(left.mean(0), np.cov(left, rowvar=False), right.mean(0), np.cov(right, rowvar=False))


def load_manifest_videos(cache: Path, entries: list[dict]) -> np.ndarray:
    frames = np.load(cache / "frames.npy", mmap_mode="r")
    videos = []
    for row in entries:
        start, count, stride = int(row["absolute_start"]), int(row["frames"]), int(row["stride"])
        video = np.asarray(frames[start:start + count * stride:stride])
        if len(video) != count:
            raise ValueError(f"manifest entry crosses clip boundary: {row}")
        videos.append(video)
    return np.stack(videos)


def corruptions(videos: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    frozen = np.repeat(videos[:, :1], videos.shape[1], axis=1)
    half_freeze = videos.copy()
    half_freeze[:, videos.shape[1] // 2:] = videos[:, videos.shape[1] // 2 - 1:videos.shape[1] // 2]
    shuffled = np.stack([video[rng.permutation(len(video))] for video in videos])
    return {
        "repeat_first": frozen,
        "freeze_second_half": half_freeze,
        "shuffle_frames": shuffled,
        "reverse_time": videos[:, ::-1].copy(),
    }


def extract_sets(video_sets: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    static_sets, temporal_sets = {}, {}
    for set_name, videos in video_sets.items():
        static_clips, temporal_clips = [], []
        for index, video in enumerate(videos):
            static = frame_part_features(video)
            static_clips.append(static)
            temporal_clips.append(temporal_part_features(static))
            if (index + 1) % 16 == 0 or index + 1 == len(videos):
                print(f"{set_name}: extracted {index + 1}/{len(videos)} videos", flush=True)
        static_sets[set_name] = np.concatenate(static_clips)
        temporal_sets[set_name] = np.concatenate(temporal_clips)
    return static_sets, temporal_sets


def compare(feature_sets: dict[str, np.ndarray], reference_name: str) -> dict:
    reference = feature_sets[reference_name]
    fitted = fit_standardizer(reference)
    return {
        "input_dimensions": int(reference.shape[1]),
        "retained_dimensions": int(fitted["keep"].sum()),
        "frames_per_reference_set": int(len(reference)),
        "distance": {
            name: float(frechet_features(reference, value, fitted))
            for name, value in feature_sets.items()
        },
    }


def part_delta_quantiles(static: np.ndarray, videos: int, frames: int,
                         quantiles=(0.5, 0.9, 0.95)) -> dict:
    """Summarise within-video part changes without crossing clip boundaries."""
    values = np.asarray(static).reshape(videos, frames, -1)
    output = {}
    labels = [f"p{int(round(q * 100))}" for q in quantiles]
    for part_index, part in enumerate(NAMES):
        begin = 2 + part_index * len(PART_FEATURES)
        part_values = values[:, :, begin:begin + len(PART_FEATURES)]
        previous, current = part_values[:, :-1], part_values[:, 1:]
        present_previous = previous[..., 0] > 0.5
        present_current = current[..., 0] > 0.5
        both = present_previous & present_current
        direction_previous = np.arctan2(previous[..., 8], previous[..., 7])
        direction_current = np.arctan2(current[..., 8], current[..., 7])
        doubled_step = (direction_current - direction_previous + np.pi) % (2 * np.pi) - np.pi
        measurements = {
            "area_abs_delta": np.abs(current[..., 1] - previous[..., 1]),
            "relative_centroid_displacement": np.hypot(
                current[..., 2] - previous[..., 2], current[..., 3] - previous[..., 3]
            ),
            "relative_major_length_abs_delta": np.abs(current[..., 4] - previous[..., 4]),
            "relative_minor_length_abs_delta": np.abs(current[..., 5] - previous[..., 5]),
            "elongation_abs_delta": np.abs(current[..., 6] - previous[..., 6]),
            "torso_relative_axis_abs_delta_rad": np.abs(doubled_step) / 2.0,
        }
        summaries = {}
        for name, measurement in measurements.items():
            selected = measurement[both]
            summaries[name] = {
                label: float(value) for label, value in zip(labels, np.quantile(selected, quantiles))
            } if len(selected) else {label: None for label in labels}
        output[part] = {
            "valid_transition_count": int(both.sum()),
            "valid_transition_fraction": float(both.mean()),
            "presence_transition_rate": float(np.mean(present_previous != present_current)),
            **summaries,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--features_out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    entries_a, entries_b = manifest["reference_a"], manifest["reference_b"]
    if args.limit:
        entries_a, entries_b = entries_a[:args.limit], entries_b[:args.limit]
    reference_a = load_manifest_videos(args.cache, entries_a)
    reference_b = load_manifest_videos(args.cache, entries_b)
    sets = {"real_reference_a": reference_a, "real_reference_b": reference_b,
            **corruptions(reference_b, args.seed)}
    static, temporal = extract_sets(sets)
    if args.features_out:
        args.features_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.features_out, **{
            **{f"static__{key}": value for key, value in static.items()},
            **{f"temporal__{key}": value for key, value in temporal.items()},
        })
    result = {
        "protocol_version": 1,
        "metric_name": "Fréchet Part-Feature Distance (prototype)",
        "palette_threshold": TAU,
        "parts": NAMES,
        "part_features": PART_FEATURES,
        "static_feature_names": feature_names(False),
        "temporal_feature_names": feature_names(True),
        "videos_per_reference_set": len(entries_a),
        "frames_per_video": int(reference_a.shape[1]),
        "standardization": "mean and standard deviation fitted on real_reference_a; constant dimensions removed",
        "static": compare(static, "real_reference_a"),
        "motion_aware": compare(temporal, "real_reference_a"),
        "part_delta_quantiles": {
            name: part_delta_quantiles(value, len(sets[name]), int(sets[name].shape[1]))
            for name, value in static.items()
        },
        "interpretation_guardrail": (
            "Frames are pooled for the point estimate and are not independent statistical units. "
            "Static FPFD is invariant to frame order by construction; motion-aware FPFD adds signed first and second differences."
        ),
        "inputs": {
            "manifest": str(args.manifest),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"static": result["static"]["distance"],
                      "motion_aware": result["motion_aware"]["distance"]}, indent=2))


if __name__ == "__main__":
    main()
