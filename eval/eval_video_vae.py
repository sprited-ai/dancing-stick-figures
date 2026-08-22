"""Deterministic full-validation metrics for DSF causal video autoencoders."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train.video_ddpm import VideoWindows
from train.video_vae import DSFCausalVideoVAE


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_evaluation_frames(saved: dict, requested: int) -> int:
    frames = int(requested or saved["frames"])
    if frames <= 0:
        raise ValueError("evaluation frames must be positive")
    return frames


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    compression = int(saved["temporal_compression"])
    model = DSFCausalVideoVAE(
        temporal_compression=compression,
        spatial_compression=int(saved.get("spatial_compression", 4)),
        latent_channels=int(saved["latent_channels"]),
        base_channels=int(saved["base_channels"]),
        blocks_per_stage=int(saved["blocks_per_stage"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = choose_device(args.device)
    model.to(device).eval()

    frames = resolve_evaluation_frames(saved, args.frames)
    dataset = VideoWindows(
        args.cache,
        frames=frames,
        split="val",
        size=int(saved["size"]),
        deterministic=True,
        repeats=args.repeats,
        return_text=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    frame_abs = torch.zeros(frames, dtype=torch.float64)
    frame_count = torch.zeros(frames, dtype=torch.float64)
    totals = {name: 0.0 for name in (
        "rgba_abs", "rgba_count", "rgb_support_abs", "rgb_support_count",
        "rgb_edge_abs", "rgb_edge_count", "alpha_fg_abs", "alpha_fg_count",
        "alpha_bg_abs", "alpha_bg_count", "alpha_intersection", "alpha_union",
        "composite_sq", "composite_count",
    )}
    windows = 0
    for batch, _ in loader:
        batch = ((batch.to(device) + 1) / 2).clamp(0, 1)
        reconstruction = model(batch, sample=False).reconstruction
        error = (reconstruction - batch).abs()
        b = batch.shape[0]
        windows += b
        per_frame = error.mean(dim=(0, 1, 3, 4)).double().cpu()
        frame_abs += per_frame * b
        frame_count += b

        alpha = batch[:, 3:4]
        pred_alpha = reconstruction[:, 3:4]
        support = alpha > 0
        edge = (alpha > 0) & (alpha < 1)
        background = 1 - alpha
        rgb_error = error[:, :3]
        totals["rgba_abs"] += float(error.sum())
        totals["rgba_count"] += error.numel()
        totals["rgb_support_abs"] += float((rgb_error * support).sum())
        totals["rgb_support_count"] += int(support.sum()) * 3
        totals["rgb_edge_abs"] += float((rgb_error * edge).sum())
        totals["rgb_edge_count"] += int(edge.sum()) * 3
        totals["alpha_fg_abs"] += float(((pred_alpha - alpha).abs() * alpha).sum())
        totals["alpha_fg_count"] += float(alpha.sum())
        totals["alpha_bg_abs"] += float(((pred_alpha - alpha).abs() * background).sum())
        totals["alpha_bg_count"] += float(background.sum())
        pred_mask, target_mask = pred_alpha >= 0.5, alpha >= 0.5
        totals["alpha_intersection"] += int((pred_mask & target_mask).sum())
        totals["alpha_union"] += int((pred_mask | target_mask).sum())
        pred_composite = reconstruction[:, :3] + (1 - pred_alpha)
        target_composite = batch[:, :3] + (1 - alpha)
        totals["composite_sq"] += float((pred_composite - target_composite).square().sum())
        totals["composite_count"] += pred_composite.numel()

    def ratio(numerator: str, denominator: str) -> float:
        return totals[numerator] / max(totals[denominator], 1e-12)

    per_frame = frame_abs / frame_count.clamp_min(1)
    phase = [float(per_frame[index::compression].mean()) for index in range(compression)]
    composite_mse = ratio("composite_sq", "composite_count")
    return {
        "version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "model_sha256": checkpoint.get("model_sha256"),
        "step": int(checkpoint["step"]),
        "variant": f"f{saved.get('spatial_compression', 4)}t{compression}d{saved['latent_channels']}",
        "posterior": "mean",
        "validation_windows": windows,
        "windows_per_clip": args.repeats,
        "frames_per_window": frames,
        "checkpoint_training_frames": int(saved["frames"]),
        "metrics": {
            "rgba_l1": ratio("rgba_abs", "rgba_count"),
            "rgb_support_l1": ratio("rgb_support_abs", "rgb_support_count"),
            "rgb_edge_l1": ratio("rgb_edge_abs", "rgb_edge_count"),
            "alpha_fg_l1": ratio("alpha_fg_abs", "alpha_fg_count"),
            "alpha_bg_l1": ratio("alpha_bg_abs", "alpha_bg_count"),
            "alpha_iou_0.5": ratio("alpha_intersection", "alpha_union"),
            "white_composite_mse": composite_mse,
            "white_composite_psnr": -10 * math.log10(max(composite_mse, 1e-12)),
            "per_frame_rgba_l1": [float(value) for value in per_frame],
            "phase_rgba_l1": phase,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--frames", type=int, default=0,
                        help="common evaluation window; 0 uses the checkpoint training window")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = evaluate(args)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
