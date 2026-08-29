"""Training-free 50-frame baselines for validating the evaluation protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.fvd import fvd, rgba_premult_to_rgb
from eval.oracle import score_video
from eval.protocol import build_reference_manifest, load_manifest_windows


METRICS = (
    "tvr", "lie", "cpe", "fg", "mass_drift", "centroid_speed", "centroid_accel",
    "motion_fraction", "angle_speed", "angle_jerk", "height_var",
)


def tensors_to_rgba(x):
    """Premultiplied [-1,1] [N,4,T,H,W] tensors to straight uint8 RGBA."""
    v = ((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 4, 1).numpy()
    alpha = v[..., 3:4]
    rgb = np.where(alpha > 0.05, v[..., :3] / np.maximum(alpha, 0.05), 0.0)
    return (np.clip(np.concatenate([rgb, alpha], -1), 0, 1) * 255).astype(np.uint8)


def tensors_to_premult_rgb(x):
    """Composite premultiplied model/cache tensors over white without an RGBA uint8 round trip."""
    value = ((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 4, 1).numpy()
    return rgba_premult_to_rgb(value)


def composite_rgb(rgba):
    """Composite straight uint8 RGBA; retained for visual/oracle callers only."""
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return np.clip(rgba[..., :3] * alpha + 255.0 * (1 - alpha), 0, 255).astype(np.uint8)


def degenerate_videos(real_rgba, seed=0):
    """Baselines designed to preserve frames while breaking temporal structure."""
    n, frames = real_rgba.shape[:2]
    rng = np.random.default_rng(seed)
    shuffled = np.empty_like(real_rgba)
    for i in range(n):
        shuffled[i] = real_rgba[i, rng.permutation(frames)]
    return {
        "repeat_first": np.repeat(real_rgba[:, :1], frames, axis=1),
        "shuffle_frames": shuffled,
        "reverse_time": real_rgba[:, ::-1].copy(),
        "loop_first_8": np.concatenate([real_rgba[:, :8]] * ((frames + 7) // 8), axis=1)[:, :frames],
    }


def _summary(values, seed=0, draws=2000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci95": [float(lo), float(hi)]}


def score_set(videos):
    per_video = [score_video(video) for video in videos]
    return {metric: _summary([row[metric] for row in per_video]) for metric in METRICS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--with_fvd", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bundle", default="", help="optional compressed RGB bundle for remote FVD")
    parser.add_argument("--only_bundle", action="store_true")
    args = parser.parse_args()

    manifest = json.load(open(args.manifest))
    a_tensor = load_manifest_windows(args.cache, manifest["reference_a"], size=args.size)
    b_tensor = load_manifest_windows(args.cache, manifest["reference_b"], size=args.size)
    a = tensors_to_rgba(a_tensor)
    b = tensors_to_rgba(b_tensor)
    # Corrupt reference B and always compare it with the independent reference A.
    # Transforming A and comparing it with itself would leak exact content into
    # both FVD sets and understate the effect of a temporal corruption.
    videos = {"real_reference_a": a, "real_reference_b": b, **degenerate_videos(b, seed=manifest["seed"])}
    # FVD consumes the white-composited RGB frame directly. Applying temporal
    # corruptions after compositing is exact for these frame-only transforms and
    # avoids the old un-premultiply -> uint8 -> re-composite quantisation path.
    reference_rgb = tensors_to_premult_rgb(a_tensor)
    candidate_rgb = tensors_to_premult_rgb(b_tensor)
    fvd_videos = {
        "real_reference_a": reference_rgb,
        "real_reference_b": candidate_rgb,
        **degenerate_videos(candidate_rgb, seed=manifest["seed"]),
    }

    # A train replay is a deliberately strong retrieval/memorisation baseline.
    train_manifest = build_reference_manifest(
        args.cache,
        frames=manifest["frames"],
        stride=manifest["stride"],
        split="train",
        n_per_half=len(manifest["reference_a"]),
        seed=manifest["seed"] + 1,
        first_frames=manifest.get("first_frames", 0),
    )
    train_tensor = load_manifest_windows(args.cache, train_manifest["reference_a"], size=args.size)
    videos["train_replay"] = tensors_to_rgba(train_tensor)
    fvd_videos["train_replay"] = tensors_to_premult_rgb(train_tensor)
    if args.bundle:
        bundle = Path(args.bundle)
        bundle.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(bundle, **fvd_videos)
        print(f"wrote {bundle}", flush=True)
        if args.only_bundle:
            return

    result = {
        "protocol_version": 1,
        "n": len(a),
        "frames": manifest["frames"],
        "stride": manifest["stride"],
        "statistical_unit": manifest["statistical_unit"],
        "baselines": {},
    }
    for name, baseline in videos.items():
        print(f"scoring {name} ...", flush=True)
        row = score_set(baseline)
        if args.with_fvd:
            row["fvd"] = float(fvd(reference_rgb, fvd_videos[name], device=args.device))
        result["baselines"][name] = row

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
