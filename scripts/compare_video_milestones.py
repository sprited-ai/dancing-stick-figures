"""Build a labeled, reproducible visual comparison from fixed-inference GIFs.

Example::

  python scripts/compare_video_milestones.py \
    --input 'K1 step 500=pod_results/run/sample_000500.gif' \
    --input 'K1 step 1000=pod_results/run/sample_001000.gif' \
    --input 'K1 step 3000=pod_results/run/sample_003000.gif' \
    --out output/comparisons/k1_milestones

The same CLI accepts future ablation outputs. It does not load checkpoints or
touch training jobs; it only reads already-generated GIFs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence


HEADER = 24
LABEL_WIDTH = 120


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + HEADER), "white")
    canvas.paste(image.convert("RGB"), (0, HEADER))
    ImageDraw.Draw(canvas).text((6, 6), label, fill="black", font=ImageFont.load_default())
    return canvas


def load_gif(path: Path) -> tuple[list[Image.Image], int]:
    with Image.open(path) as image:
        duration = int(image.info.get("duration", 100))
        frames = [frame.convert("RGB").copy() for frame in ImageSequence.Iterator(image)]
    if not frames:
        raise ValueError(f"GIF has no frames: {path}")
    return frames, duration


def build_comparison(inputs: list[tuple[str, Path]], out_dir: Path,
                     strip_indices: list[int] | None = None) -> dict[str, object]:
    if len(inputs) < 2:
        raise ValueError("comparison requires at least two labeled GIFs")
    loaded = [(label, path.resolve(), *load_gif(path)) for label, path in inputs]
    counts = {len(frames) for _, _, frames, _ in loaded}
    if len(counts) != 1:
        raise ValueError(f"all GIFs must have the same frame count, got {sorted(counts)}")
    frame_count = counts.pop()
    duration = loaded[0][3]
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_frames = []
    for frame_index in range(frame_count):
        panels = [_label(frames[frame_index], label) for label, _, frames, _ in loaded]
        canvas = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "#eeeeee")
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width
        comparison_frames.append(canvas)
    gif_path = out_dir / "comparison.gif"
    comparison_frames[0].save(gif_path, save_all=True, append_images=comparison_frames[1:],
                              duration=duration, loop=0)

    indices = strip_indices or [0, frame_count // 4, frame_count // 2,
                                3 * frame_count // 4, frame_count - 1]
    indices = sorted(set(max(0, min(frame_count - 1, i)) for i in indices))
    cell_w = max(frames[0].width for _, _, frames, _ in loaded)
    cell_h = max(frames[0].height for _, _, frames, _ in loaded)
    strip = Image.new("RGB", (LABEL_WIDTH + cell_w * len(indices), cell_h * len(loaded)), "white")
    draw = ImageDraw.Draw(strip)
    font = ImageFont.load_default()
    for row, (label, _, frames, _) in enumerate(loaded):
        y = row * cell_h
        draw.text((6, y + 6), label, fill="black", font=font)
        for column, index in enumerate(indices):
            frame = frames[index]
            strip.paste(frame, (LABEL_WIDTH + column * cell_w, y))
    strip_path = out_dir / "strip.png"
    strip.save(strip_path)

    manifest = {
        "version": 1,
        "comparison_gif": "comparison.gif",
        "strip": "strip.png",
        "frame_count": frame_count,
        "frame_duration_ms": duration,
        "strip_frame_indices": indices,
        "inputs": [{"label": label, "path": str(path), "sha256": _sha256(path),
                    "size": list(frames[0].size), "frames": len(frames)}
                   for label, path, frames, _ in loaded],
        "warning": "Comparable only when source GIFs used the same prompts, noise, sampler and layout; this tool does not infer that provenance."
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**manifest, "gif": str(gif_path.resolve()), "strip": str(strip_path.resolve()),
            "manifest": str(manifest_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=GIF")
    parser.add_argument("--out", required=True)
    parser.add_argument("--strip-frames", default="", help="comma-separated zero-based frame indices")
    args = parser.parse_args()
    inputs = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(f"--input must be LABEL=GIF, got {spec!r}")
        label, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if not path.is_file():
            raise SystemExit(f"missing GIF: {path}")
        inputs.append((label, path))
    strip_indices = [int(value) for value in args.strip_frames.split(",") if value.strip()] or None
    result = build_comparison(inputs, Path(args.out), strip_indices)
    print(json.dumps({name: result[name] for name in ("gif", "strip", "manifest")}, indent=2))


if __name__ == "__main__":
    main()
