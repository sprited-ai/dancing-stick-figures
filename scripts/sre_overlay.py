"""Overlay an SRE-predicted 2D rig on a generated GIF tile.

The training milestone GIFs are white-composited grids, so this utility
recovers an approximate premultiplied RGBA input before running SRE.  Final
evaluation should prefer the lossless ``*_rgba.npz`` artifacts produced by
``eval.post_eval_unet --save_rgba``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageSequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.sre_validate import load_model


PARENTS = (-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 10, 10, 4, 13, 14, 15, 16, 16,
           0, 19, 20, 21, 0, 23, 24, 25)


def recover_premultiplied(composite_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover approximate premultiplied RGB and alpha from a white composite.

    The released stick palette has at least one zero-valued channel per part,
    making ``1 - min(composite)`` an exact alpha inversion for renderer output.
    Generated colours only satisfy this approximately, which is recorded in
    the output manifest.
    """
    composite = composite_rgb.astype(np.float32) / 255.0
    alpha = 1.0 - composite.min(axis=-1, keepdims=True)
    premultiplied = np.clip(composite - (1.0 - alpha), 0.0, 1.0)
    return premultiplied, alpha


def load_tile(path: Path, tile: int, size: int) -> tuple[np.ndarray, int]:
    source = Image.open(path)
    cols = source.width // size
    rows = source.height // size
    if source.width % size or source.height % size or not 0 <= tile < cols * rows:
        raise ValueError(f"tile {tile} is invalid for {source.size} with {size}px cells")
    row, column = divmod(tile, cols)
    frames = []
    for frame in ImageSequence.Iterator(source):
        rgb = frame.convert("RGB").crop(
            (column * size, row * size, (column + 1) * size, (row + 1) * size)
        )
        frames.append(np.asarray(rgb, dtype=np.uint8))
    if not frames:
        raise ValueError(f"no frames in {path}")
    return np.stack(frames), int(source.info.get("duration", 100))


@torch.no_grad()
def predict(model, frames: np.ndarray, device: str, batch: int = 128):
    premultiplied, alpha = recover_premultiplied(frames)
    rgba = np.concatenate((premultiplied, alpha), axis=-1)
    tensor = torch.from_numpy(rgba).permute(0, 3, 1, 2)
    outputs = []
    for start in range(0, len(tensor), batch):
        outputs.append(model(tensor[start:start + batch].to(device)).cpu())
    return torch.cat(outputs).numpy(), alpha[..., 0]


def rig_metrics(rig: np.ndarray, alpha: np.ndarray, size: int) -> dict:
    child = np.arange(1, len(PARENTS))
    parent = np.asarray(PARENTS[1:])
    bones = np.linalg.norm(rig[:, child] - rig[:, parent], axis=-1) * size
    speed = np.linalg.norm(np.diff(rig, axis=0), axis=-1) * size
    hits = total = 0
    for frame_index, joints in enumerate(rig):
        xy = np.floor(joints * size).astype(int)
        valid = ((xy >= 0) & (xy < size)).all(axis=-1)
        for x_y in xy[valid]:
            total += 1
            hits += int(alpha[frame_index, x_y[1], x_y[0]] > 0.1)
    mean_bone = np.maximum(bones.mean(axis=0), 1e-6)
    return {
        "on_foreground_joint_fraction": hits / max(total, 1),
        "mean_joint_speed_px_per_frame": float(speed.mean()) if speed.size else 0.0,
        "bone_length_temporal_cv": float((bones.std(axis=0) / mean_bone).mean()),
        "claim_limit": (
            "SRE v1 has no calibrated confidence head. Alpha was approximately inverted "
            "from a white-composited milestone GIF; use lossless RGBA for final scores."
        ),
    }


def render(frames: np.ndarray, rig: np.ndarray, prompt: str, scale: int) -> list[Image.Image]:
    size = frames.shape[1]
    font = ImageFont.load_default()
    banner = 34
    output = []
    for rgb, joints in zip(frames, rig):
        original = Image.fromarray(rgb).resize((size * scale, size * scale), Image.Resampling.NEAREST)
        overlay = original.copy()
        draw = ImageDraw.Draw(overlay)
        xy = joints * size * scale
        for child in range(1, len(PARENTS)):
            draw.line((tuple(xy[PARENTS[child]]), tuple(xy[child])), fill=(0, 255, 80),
                      width=max(2, scale // 2))
        radius = max(2, scale // 2)
        for x, y in xy:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         fill=(255, 220, 0), outline=(20, 20, 20))
        canvas = Image.new("RGB", (size * scale * 2, banner + size * scale), "white")
        label = f"Prompt: {prompt}" if prompt else "SRE v1 overlay (coordinates only; no confidence)"
        ImageDraw.Draw(canvas).text((8, 5), label[:110], fill=(25, 25, 25), font=font)
        ImageDraw.Draw(canvas).text((8, 19), "original", fill=(70, 70, 70), font=font)
        ImageDraw.Draw(canvas).text((size * scale + 8, 19), "SRE overlay", fill=(70, 70, 70), font=font)
        canvas.paste(original, (0, banner)); canvas.paste(overlay, (size * scale, banner))
        output.append(canvas)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="white-composited GIF, optionally a sample grid")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True, help="output GIF")
    parser.add_argument("--tile", type=int, default=0, help="row-major grid cell")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, size = load_model(args.ckpt, args.device)
    frames, duration = load_tile(Path(args.input), args.tile, size)
    rig, alpha = predict(model, frames, args.device)
    images = render(frames, rig, args.prompt, args.scale)
    destination = Path(args.out); destination.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(destination, save_all=True, append_images=images[1:], duration=duration,
                   loop=0, disposal=2)
    np.savez_compressed(destination.with_suffix(".npz"), rig=rig, prompt=args.prompt)
    report = {
        "protocol": "sre_v1_milestone_overlay_v1",
        "input": str(Path(args.input).resolve()),
        "tile": args.tile,
        "prompt": args.prompt,
        "frames": len(frames),
        "fps": 1000.0 / duration,
        "sre_checkpoint": str(Path(args.ckpt).resolve()),
        "sre_checkpoint_sha256": hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(),
        "metrics": rig_metrics(rig, alpha, size),
        "output": str(destination.resolve()),
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
