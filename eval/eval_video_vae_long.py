"""Audit long-sequence behavior of a DSF causal video autoencoder.

The training windows are short (currently 20 frames), while a causal decoder
can have a substantially larger theoretical receptive field.  This evaluator
therefore compares two reconstructions of the same validation clips:

* ``continuous``: encode/decode the complete clip in one call;
* ``tiled``: independently encode/decode non-overlapping fixed-size windows;
* ``sliding``: decode the first window, then advance by one latent block and
  commit only the newest output block. This matches bounded-memory deployment.

It reports reconstruction error, boundary transition error, continuous-vs-
tiled disagreement, and prefix consistency.  The latter verifies that adding
future frames cannot alter an already decoded causal prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from train.video_vae import DSFCausalVideoVAE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_clip(frames: np.ndarray, clip: dict, size: int, length: int) -> torch.Tensor:
    value = np.asarray(frames[clip["start"]:clip["start"] + length]).astype(np.float32) / 255.0
    if value.shape[1] != size:
        factor = value.shape[1] // size
        value = np.concatenate((value[..., :3] * value[..., 3:4], value[..., 3:4]), axis=-1)
        value = value.reshape(length, size, factor, size, factor, 4).mean((2, 4))
    else:
        value = np.concatenate((value[..., :3] * value[..., 3:4], value[..., 3:4]), axis=-1)
    return torch.from_numpy(value).permute(3, 0, 1, 2).unsqueeze(0)


def _composite(video: torch.Tensor) -> np.ndarray:
    value = video.detach().float().clamp(0, 1).cpu()[0]
    rgb = value[:3] + (1 - value[3:4])
    return (rgb.permute(1, 2, 3, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)


def _save_gif(target: torch.Tensor, continuous: torch.Tensor, tiled: torch.Tensor,
              sliding: torch.Tensor, prompt: str, path: Path, fps: int) -> None:
    rows = [("ORIGINAL", _composite(target)), ("CONTINUOUS 120F", _composite(continuous)),
            ("INDEPENDENT 20F TILES", _composite(tiled)),
            ("SLIDING 20F / COMMIT 4F", _composite(sliding))]
    height, width = rows[0][1].shape[1:3]
    banner = 20
    output = []
    for index in range(target.shape[2]):
        canvas = Image.new("RGB", (width, len(rows) * (height + banner)), "white")
        draw = ImageDraw.Draw(canvas)
        for row, (label, frames) in enumerate(rows):
            top = row * (height + banner)
            draw.text((3, top + 3), f"{label} | f={index:03d} | {prompt[:48]}", fill="black")
            canvas.paste(Image.fromarray(frames[index]), (0, top + banner))
        output.append(np.asarray(canvas))
    imageio.mimsave(path, output, duration=1000 / fps, loop=0)


def transition_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-transition L1 error of first temporal differences."""
    pred_delta = prediction[:, :, 1:] - prediction[:, :, :-1]
    target_delta = target[:, :, 1:] - target[:, :, :-1]
    return (pred_delta - target_delta).abs().mean(dim=(0, 1, 3, 4))


@torch.no_grad()
def strict_causal_prefix_error(
    model: DSFCausalVideoVAE, target: torch.Tensor, *, prefix_frames: int,
) -> float:
    """Test causal prefix invariance without TF32 shape-dependent roundoff.

    cuDNN may select different TF32 convolution kernels for a short prefix and
    a long sequence. Their mean discrepancy is tiny, but a max-only causality
    assertion can reach ~2e-3 even when the architecture is exactly causal.
    Strict convolution arithmetic isolates future leakage from that numerical
    implementation detail, while the separately timed reconstruction paths
    retain their normal fast settings.
    """
    previous = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        long_value = model(target, sample=False).reconstruction[:, :, :prefix_frames]
        short_value = model(target[:, :, :prefix_frames], sample=False).reconstruction
        return float((short_value - long_value).abs().max())
    finally:
        torch.backends.cudnn.allow_tf32 = previous


@torch.no_grad()
def sliding_reconstruct(
    model: DSFCausalVideoVAE,
    target: torch.Tensor,
    *,
    window: int,
    commit: int,
) -> torch.Tensor:
    """Bounded-context reconstruction, committing each decoded block once."""
    frames = target.shape[2]
    if frames < window or (frames - window) % commit:
        raise ValueError("sequence after the first window must divide evenly into commit blocks")
    if window % commit:
        raise ValueError("window must contain a whole number of commit blocks")
    first = model(target[:, :, :window], sample=False).reconstruction
    parts = [first]
    for end in range(window + commit, frames + 1, commit):
        reconstruction = model(target[:, :, end - window:end], sample=False).reconstruction
        parts.append(reconstruction[:, :, -commit:])
    return torch.cat(parts, dim=2)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    model = DSFCausalVideoVAE(
        temporal_compression=int(saved["temporal_compression"]),
        spatial_compression=int(saved.get("spatial_compression", 4)),
        latent_channels=int(saved["latent_channels"]),
        base_channels=int(saved["base_channels"]),
        blocks_per_stage=int(saved["blocks_per_stage"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()

    cache = Path(args.cache)
    frame_store = np.load(cache / "frames.npy", mmap_mode="r")
    clip_map = json.loads((cache / "clips.json").read_text())
    clips = [dict(clip_id=key, **value) for key, value in sorted(clip_map.items())
             if value["split"] == "val" and value["n"] >= args.frames and not value.get("qa")]
    clips = clips[:args.clips]
    if not clips:
        raise ValueError("no eligible validation clips")
    if args.frames % args.tile != 0:
        raise ValueError("--frames must be divisible by --tile")

    totals = {key: 0.0 for key in (
        "continuous_abs", "tiled_abs", "sliding_abs", "disagreement_abs", "elements",
        "continuous_sq", "tiled_sq", "sliding_sq", "composite_elements", "prefix_max",
        "boundary_transition", "boundary_count", "within_transition", "within_count",
        "sliding_boundary_transition", "sliding_boundary_count",
        "sliding_within_transition", "sliding_within_count",
    )}
    timings = {"continuous_seconds": 0.0, "tiled_seconds": 0.0}
    examples = []
    # Exclude CUDA context/kernel initialization from the first measured clip.
    warmup = _load_clip(frame_store, clips[0], int(saved["size"]), args.tile).to(device)
    model(warmup, sample=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    for clip_index, clip in enumerate(clips):
        target = _load_clip(frame_store, clip, int(saved["size"]), args.frames).to(device)
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        continuous = model(target, sample=False).reconstruction
        if device.type == "cuda": torch.cuda.synchronize()
        timings["continuous_seconds"] += time.perf_counter() - start

        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        tiled_parts = [model(target[:, :, offset:offset + args.tile], sample=False).reconstruction
                       for offset in range(0, args.frames, args.tile)]
        tiled = torch.cat(tiled_parts, dim=2)
        if device.type == "cuda": torch.cuda.synchronize()
        timings["tiled_seconds"] += time.perf_counter() - start

        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        sliding = sliding_reconstruct(model, target, window=args.tile, commit=args.commit)
        if device.type == "cuda": torch.cuda.synchronize()
        timings.setdefault("sliding_seconds", 0.0)
        timings["sliding_seconds"] += time.perf_counter() - start

        elements = target.numel()
        totals["continuous_abs"] += float((continuous - target).abs().sum())
        totals["tiled_abs"] += float((tiled - target).abs().sum())
        totals["sliding_abs"] += float((sliding - target).abs().sum())
        totals["disagreement_abs"] += float((continuous - tiled).abs().sum())
        totals["elements"] += elements

        for name, reconstruction in (("continuous", continuous), ("tiled", tiled), ("sliding", sliding)):
            pred_comp = reconstruction[:, :3] + (1 - reconstruction[:, 3:4])
            true_comp = target[:, :3] + (1 - target[:, 3:4])
            totals[f"{name}_sq"] += float((pred_comp - true_comp).square().sum())
        totals["composite_elements"] += target[:, :3].numel()

        totals["prefix_max"] = max(
            totals["prefix_max"],
            strict_causal_prefix_error(model, target, prefix_frames=args.tile),
        )

        transition = transition_error(continuous, target)
        boundary_indices = torch.tensor([index - 1 for index in range(args.tile, args.frames, args.tile)], device=device)
        mask = torch.ones_like(transition, dtype=torch.bool)
        mask[boundary_indices] = False
        totals["boundary_transition"] += float(transition[boundary_indices].sum())
        totals["boundary_count"] += int(boundary_indices.numel())
        totals["within_transition"] += float(transition[mask].sum())
        totals["within_count"] += int(mask.sum())

        sliding_transition = transition_error(sliding, target)
        sliding_boundaries = torch.tensor(
            [index - 1 for index in range(args.tile, args.frames, args.commit)], device=device
        )
        sliding_mask = torch.ones_like(sliding_transition, dtype=torch.bool)
        sliding_mask[sliding_boundaries] = False
        totals["sliding_boundary_transition"] += float(sliding_transition[sliding_boundaries].sum())
        totals["sliding_boundary_count"] += int(sliding_boundaries.numel())
        totals["sliding_within_transition"] += float(sliding_transition[sliding_mask].sum())
        totals["sliding_within_count"] += int(sliding_mask.sum())

        if clip_index < args.gifs:
            gif_name = f"long_audit_{clip_index:02d}.gif"
            _save_gif(target, continuous, tiled, sliding, clip["text"], Path(args.out) / gif_name, args.fps)
            examples.append({"clip_id": clip["clip_id"], "prompt": clip["text"], "gif": gif_name})

    count = totals["elements"]
    comp_count = totals["composite_elements"]
    continuous_mse = totals["continuous_sq"] / comp_count
    tiled_mse = totals["tiled_sq"] / comp_count
    sliding_mse = totals["sliding_sq"] / comp_count
    average_timing = {key: value / len(clips) for key, value in timings.items()}
    average_timing.update({
        "continuous_frames_per_second": args.frames / max(average_timing["continuous_seconds"], 1e-12),
        "tiled_frames_per_second": args.frames / max(average_timing["tiled_seconds"], 1e-12),
        "sliding_frames_per_second": args.frames / max(average_timing["sliding_seconds"], 1e-12),
    })
    memory = None
    if device.type == "cuda":
        memory = {
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
    result = {
        "version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "step": int(checkpoint["step"]),
        "variant": f"f{saved.get('spatial_compression', 4)}t{saved['temporal_compression']}d{saved['latent_channels']}",
        "validation_clips": len(clips),
        "frames_per_clip": args.frames,
        "independent_tile_frames": args.tile,
        "sliding_commit_frames": args.commit,
        "metrics": {
            "continuous_rgba_l1": totals["continuous_abs"] / count,
            "tiled_rgba_l1": totals["tiled_abs"] / count,
            "sliding_rgba_l1": totals["sliding_abs"] / count,
            "continuous_vs_tiled_rgba_l1": totals["disagreement_abs"] / count,
            "continuous_white_psnr": -10 * math.log10(max(continuous_mse, 1e-12)),
            "tiled_white_psnr": -10 * math.log10(max(tiled_mse, 1e-12)),
            "sliding_white_psnr": -10 * math.log10(max(sliding_mse, 1e-12)),
            "continuous_boundary_transition_l1": totals["boundary_transition"] / totals["boundary_count"],
            "continuous_within_transition_l1": totals["within_transition"] / totals["within_count"],
            "sliding_boundary_transition_l1": totals["sliding_boundary_transition"] / totals["sliding_boundary_count"],
            "sliding_within_transition_l1": totals["sliding_within_transition"] / totals["sliding_within_count"],
            "causal_prefix_max_abs": totals["prefix_max"],
        },
        "timing": average_timing,
        "memory": memory,
        "causal_prefix_numeric_mode": "cuDNN TF32 disabled for paired prefix assertion only",
        "examples": examples,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--clips", type=int, default=24)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--tile", type=int, default=20)
    parser.add_argument("--commit", type=int, default=4)
    parser.add_argument("--gifs", type=int, default=2)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    result = evaluate(args)
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
