"""Score a video through the learned Skeleton Recovery Evaluator (SRE).

SRE first maps every RGBA frame to 27 image-space joints.  This module then
reports a diagnostic vector rather than hiding unrelated failures in one
number: foreground support, calibrated coordinate confidence, bone-length
variation, and joint speed/acceleration/jerk.  Values are meant to be read
beside the same measurements on held-out rendered clips.

The confidence model is calibrated on held-out renders.  On generated or
renderer-shifted frames its confidence is a useful model signal, not a
certified probability of anatomical correctness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from generator.skeleton import NAMES, PARENT
from train.sre import RIG_JOINTS
from train.sre_confidence import SREConfidence


PARENTS = np.asarray([
    -1 if PARENT.get(name) is None else NAMES.index(PARENT[name]) for name in NAMES
], dtype=np.int64)
CHILDREN = np.flatnonzero(PARENTS >= 0)


def load_confidence_model(path: str, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = SREConfidence(
        size=int(checkpoint["size"]), hidden=int(checkpoint.get("hidden", 512))
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    temperature = float(checkpoint.get("temperature", 1.0))
    return model, int(checkpoint["size"]), temperature


def rgba_to_model_input(rgba: np.ndarray, size: int) -> torch.Tensor:
    """Straight uint8 RGBA [B,H,W,4] -> premultiplied float [B,4,S,S]."""
    value = torch.from_numpy(np.asarray(rgba)).float().permute(0, 3, 1, 2) / 255.0
    value[:, :3] *= value[:, 3:4]
    if value.shape[-2:] != (size, size):
        value = F.interpolate(value, size=(size, size), mode="area")
    return value


@torch.no_grad()
def recover_rig(model, rgba: np.ndarray, device: str, size: int, temperature: float,
                batch: int = 256, limb_radius_px: float = 1.6):
    flat = rgba.reshape(-1, *rgba.shape[-3:])
    value = rgba_to_model_input(flat, size)
    joints, confidence = [], []
    for start in range(0, len(value), batch):
        mean, log_sigma = model(value[start:start + batch].to(device))
        sigma_px = log_sigma.exp() * size * temperature
        probability = 1.0 - torch.exp(
            -(limb_radius_px ** 2) / (2.0 * sigma_px.square().clamp_min(1e-12))
        )
        joints.append(mean.cpu()); confidence.append(probability.cpu())
    shape = rgba.shape[:2]
    return (torch.cat(joints).numpy().reshape(*shape, RIG_JOINTS, 2),
            torch.cat(confidence).numpy().reshape(*shape, RIG_JOINTS))


def _weighted_mean(value: np.ndarray, weight: np.ndarray, axis) -> np.ndarray:
    total = weight.sum(axis=axis)
    return (value * weight).sum(axis=axis) / np.maximum(total, 1e-8)


def _sequence_metric(value: np.ndarray, confidence: np.ndarray) -> list[float]:
    return [float(_weighted_mean(v, w, axis=None)) for v, w in zip(value, confidence)]


def _ece(probability: np.ndarray, outcome: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (probability >= left) & (
            probability <= right if index == bins - 1 else probability < right
        )
        if selected.any():
            value += selected.mean() * abs(probability[selected].mean() - outcome[selected].mean())
    return float(value)


def recovery_accuracy(prediction: np.ndarray, confidence: np.ndarray,
                      target: np.ndarray, size: int, radius_px: float = 1.6) -> dict:
    """Coordinate accuracy and fixed-temperature calibration on labelled clips."""
    visible = ((target >= 0.0) & (target <= 1.0)).all(axis=-1)
    distance = np.linalg.norm(prediction - target, axis=-1) * size
    distance = distance[visible]
    probability = confidence[visible]
    outcome = distance <= radius_px
    return {
        "visible_joints": int(visible.sum()),
        "mean_error_px": float(distance.mean()),
        "pck2": float((distance <= 2.0).mean()),
        "pck4": float((distance <= 4.0).mean()),
        "confidence_radius_px": radius_px,
        "mean_confidence_within_limb": float(probability.mean()),
        "empirical_within_limb": float(outcome.mean()),
        "brier_within_limb": float(np.mean((probability - outcome) ** 2)),
        "ece_within_limb": _ece(probability, outcome),
    }


def rasterize_rig(rig: np.ndarray, confidence: np.ndarray, height: int, width: int,
                  model_size: int, radius_px: float = 1.6) -> np.ndarray:
    """Draw confidence-weighted 2D bones as a soft [N,T,H,W] support mask."""
    scale = 0.5 * (height + width) / model_size
    line_width = max(1, int(round(2.0 * radius_px * scale)))
    joint_radius = max(1, line_width // 2)
    output = np.zeros((*rig.shape[:2], height, width), np.float32)
    for clip in range(rig.shape[0]):
        for frame in range(rig.shape[1]):
            canvas = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(canvas)
            xy = rig[clip, frame] * np.asarray([width - 1, height - 1], np.float32)
            for child in CHILDREN:
                parent = PARENTS[child]
                weight = float(min(confidence[clip, frame, child],
                                   confidence[clip, frame, parent]))
                draw.line((tuple(xy[parent]), tuple(xy[child])),
                          fill=int(round(255 * weight)), width=line_width)
            for joint, (x, y) in enumerate(xy):
                value = int(round(255 * float(confidence[clip, frame, joint])))
                draw.ellipse((x - joint_radius, y - joint_radius,
                              x + joint_radius, y + joint_radius), fill=value)
            output[clip, frame] = np.asarray(canvas, np.float32) / 255.0
    return output


def rig_frame_agreement(rig: np.ndarray, confidence: np.ndarray, alpha: np.ndarray,
                        model_size: int) -> dict:
    """Soft overlap between a recovered 2D rig and the observed alpha foreground."""
    support = rasterize_rig(
        rig, confidence, alpha.shape[-2], alpha.shape[-1], model_size
    )
    foreground = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    intersection = np.minimum(support, foreground).sum(axis=(-2, -1))
    precision = intersection / np.maximum(support.sum(axis=(-2, -1)), 1e-8)
    recall = intersection / np.maximum(foreground.sum(axis=(-2, -1)), 1e-8)
    iou = intersection / np.maximum(np.maximum(support, foreground).sum(axis=(-2, -1)), 1e-8)
    per_clip = {
        "rig_foreground_precision": precision.mean(axis=1).astype(float).tolist(),
        "rig_foreground_recall": recall.mean(axis=1).astype(float).tolist(),
        "rig_foreground_iou": iou.mean(axis=1).astype(float).tolist(),
    }
    return {
        "per_clip": per_clip,
        "aggregate": {
            name: {"mean": float(np.mean(values)), "median": float(np.median(values))}
            for name, values in per_clip.items()
        },
    }


def score_sequences(rig: np.ndarray, confidence: np.ndarray, size: int,
                    alpha: np.ndarray | None = None) -> dict:
    """Return per-clip and aggregate diagnostics for [N,T,27,2] coordinates."""
    if rig.ndim != 4 or rig.shape[-2:] != (RIG_JOINTS, 2):
        raise ValueError(f"expected rig [N,T,{RIG_JOINTS},2], got {rig.shape}")
    if confidence.shape != rig.shape[:-1]:
        raise ValueError("confidence must have shape [N,T,27]")

    child = CHILDREN
    parent = PARENTS[child]
    bones = np.linalg.norm(rig[..., child, :] - rig[..., parent, :], axis=-1) * size
    bone_weight = np.minimum(confidence[..., child], confidence[..., parent])
    bone_mean = _weighted_mean(bones, bone_weight, axis=1)
    bone_var = _weighted_mean(
        (bones - bone_mean[:, None, :]) ** 2, bone_weight, axis=1
    )
    bone_cv = np.sqrt(bone_var) / np.maximum(bone_mean, 1e-6)
    per_clip_bone_cv = _sequence_metric(bone_cv, bone_weight.mean(axis=1))

    velocity = np.linalg.norm(np.diff(rig, axis=1), axis=-1) * size
    velocity_weight = np.minimum(confidence[:, 1:], confidence[:, :-1])
    acceleration = np.linalg.norm(np.diff(rig, n=2, axis=1), axis=-1) * size
    acceleration_weight = np.minimum.reduce(
        [confidence[:, 2:], confidence[:, 1:-1], confidence[:, :-2]]
    )
    jerk = np.linalg.norm(np.diff(rig, n=3, axis=1), axis=-1) * size
    jerk_weight = np.minimum.reduce(
        [confidence[:, 3:], confidence[:, 2:-1], confidence[:, 1:-2], confidence[:, :-3]]
    )

    per_clip = {
        "mean_joint_confidence": confidence.mean(axis=(1, 2)).astype(float).tolist(),
        "confidence_coverage_0_5": (confidence >= 0.5).mean(axis=(1, 2)).astype(float).tolist(),
        "bone_length_temporal_cv": per_clip_bone_cv,
        "joint_speed_px_per_frame": _sequence_metric(velocity, velocity_weight),
        "joint_accel_px_per_frame2": _sequence_metric(acceleration, acceleration_weight),
        "joint_jerk_px_per_frame3": _sequence_metric(jerk, jerk_weight),
    }

    if alpha is not None:
        if alpha.shape[:2] != rig.shape[:2]:
            raise ValueError("alpha and rig sequence dimensions do not match")
        radius = max(1, round(1.6 * alpha.shape[-1] / size))
        mask = torch.from_numpy(alpha.astype(np.float32)).reshape(
            -1, 1, alpha.shape[-2], alpha.shape[-1]
        )
        near_fg = F.max_pool2d(mask, kernel_size=2 * radius + 1, stride=1, padding=radius)
        xy = np.rint(rig * np.asarray([alpha.shape[-1] - 1, alpha.shape[-2] - 1])).astype(int)
        valid = ((xy >= 0) & (xy < np.asarray([alpha.shape[-1], alpha.shape[-2]]))).all(-1)
        support = np.zeros(valid.shape, dtype=np.float32)
        flat_support = near_fg[:, 0].numpy()
        flat_xy = xy.reshape(-1, RIG_JOINTS, 2)
        flat_valid = valid.reshape(-1, RIG_JOINTS)
        flat_out = support.reshape(-1, RIG_JOINTS)
        for frame in range(len(flat_xy)):
            selected = flat_valid[frame]
            points = flat_xy[frame, selected]
            flat_out[frame, selected] = flat_support[frame, points[:, 1], points[:, 0]] >= 0.1
        support_weight = confidence * valid
        per_clip["foreground_joint_support"] = _sequence_metric(support, support_weight)

        agreement = rig_frame_agreement(rig, confidence, alpha, size)["per_clip"]
        per_clip.update(agreement)

    aggregate = {
        name: {"mean": float(np.mean(values)), "median": float(np.median(values))}
        for name, values in per_clip.items()
    }
    return {"per_clip": per_clip, "aggregate": aggregate}


def load_npz(path: str):
    archive = np.load(path, allow_pickle=False)
    rgba = archive["rgba"]
    if rgba.ndim != 5 or rgba.shape[-1] != 4:
        raise ValueError(f"expected rgba [N,T,H,W,4], got {rgba.shape}")
    prompts = archive["prompts"].astype(str).tolist() if "prompts" in archive else []
    seeds = archive["seeds"].astype(int).tolist() if "seeds" in archive else []
    return rgba, prompts, seeds


def load_reference(cache: str, split: str, n: int, frames: int):
    root = Path(cache)
    clips = json.loads((root / "clips.json").read_text())
    selected = [clip for _, clip in sorted(clips.items())
                if clip["split"] == split and int(clip["n"]) >= frames][:n]
    if len(selected) < n:
        raise ValueError(f"requested {n} {split} clips with {frames} frames; found {len(selected)}")
    pixels = np.load(root / "frames.npy", mmap_mode="r")
    rigs = np.load(root / "rig.npy", mmap_mode="r")
    rgba = np.stack([pixels[int(c["start"]):int(c["start"]) + frames] for c in selected])
    rig = np.stack([rigs[int(c["start"]):int(c["start"]) + frames] for c in selected])
    prompts = [str(c.get("text", "")) for c in selected]
    return rgba, rig, prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="lossless RGBA npz from eval.post_eval_unet")
    source.add_argument("--cache", help="aligned frame/rig cache for a real-reference score")
    parser.add_argument("--ckpt", required=True, help="confidence-aware SRE checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    options = parser.parse_args()

    model, model_size, temperature = load_confidence_model(options.ckpt, options.device)
    ground_truth = None
    if options.input:
        rgba, prompts, seeds = load_npz(options.input)
        source_info = {"kind": "generated_rgba", "path": str(Path(options.input).resolve())}
    else:
        rgba, ground_truth, prompts = load_reference(
            options.cache, options.split, options.n, options.frames
        )
        seeds = []
        source_info = {"kind": "real_reference", "cache": str(Path(options.cache).resolve()),
                       "split": options.split}

    rig, confidence = recover_rig(
        model, rgba, options.device, model_size, temperature, batch=options.batch
    )
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    report = {
        "protocol": "sre_diagnostic_vector_v1",
        "source": source_info,
        "shape": list(rgba.shape),
        "prompts": prompts,
        "seeds": seeds,
        "sre_checkpoint": str(Path(options.ckpt).resolve()),
        "confidence_scope": (
            "calibrated on held-out renders; generated-video confidence is not a certified OOD probability"
        ),
        "prediction": score_sequences(rig, confidence, model_size, alpha),
    }
    if ground_truth is not None:
        report["recovery_accuracy"] = recovery_accuracy(
            rig, confidence, ground_truth, model_size
        )
        report["ground_truth"] = score_sequences(
            ground_truth, np.ones(ground_truth.shape[:-1], np.float32), model_size, alpha
        )
    destination = Path(options.out); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["prediction"]["aggregate"], indent=2))


if __name__ == "__main__":
    main()
