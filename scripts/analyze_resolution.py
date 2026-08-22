"""Measure and visualise structural information retained at small resolutions.

Example:
  python scripts/analyze_resolution.py --data data/v1 --out-dir /tmp/resolution-study
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.oracle import LIMBS, label_colours, score_frame


def resize_rgba(image: np.ndarray, size: int) -> np.ndarray:
    """Area-resize straight RGBA through premultiplied colour."""
    x = image.astype(np.float32) / 255.0
    alpha = x[..., 3:4]
    premult = np.concatenate([x[..., :3] * alpha, alpha], axis=-1)
    channels = []
    for channel in range(4):
        plane = Image.fromarray(premult[..., channel].astype(np.float32), mode="F")
        channels.append(np.asarray(plane.resize((size, size), Image.Resampling.BOX), np.float32))
    out = np.stack(channels, axis=-1)
    a = out[..., 3:4]
    rgb = np.divide(out[..., :3], a, out=np.zeros_like(out[..., :3]), where=a > 1e-4)
    return (np.clip(np.concatenate([rgb, a], axis=-1), 0, 1) * 255).round().astype(np.uint8)


def composite_white(rgba: np.ndarray) -> np.ndarray:
    x = rgba.astype(np.float32) / 255.0
    a = x[..., 3:4]
    return (np.clip(x[..., :3] * a + 1 - a, 0, 1) * 255).astype(np.uint8)


def font(size: int):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--sizes", default="64,48,32")
    args = parser.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    files = sorted(Path(args.data).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {args.data}")

    rows = []
    per_file = max(1, int(np.ceil(args.n / len(files))))
    for path in files:
        table = pq.read_table(path, columns=["color", "group", "text"])
        indices = np.linspace(0, table.num_rows - 1, per_file, dtype=int)
        data = table.take(indices).to_pydict()
        for payload, group, text in zip(data["color"], data["group"], data["text"]):
            rows.append((np.asarray(Image.open(io.BytesIO(payload["bytes"])).convert("RGBA")), group, text))
            if len(rows) >= args.n:
                break
        if len(rows) >= args.n:
            break

    results = {"source": str(args.data), "n": len(rows), "resolutions": {}}
    resized = {size: [] for size in sizes}
    for size in sizes:
        metrics = []
        fg_pixels, min_limb_pixels = [], []
        for image, _, _ in rows:
            small = resize_rgba(image, size)
            resized[size].append(small)
            score = score_frame(small)
            lab, fg = label_colours(small)
            metrics.append(score)
            fg_pixels.append(int(fg.sum()))
            counts = [(lab == i + 1).sum() for i in range(len(LIMBS))]
            min_limb_pixels.append(int(min(counts)))
        results["resolutions"][str(size)] = {
            "pixel_ratio_vs_64": (size / 64) ** 2,
            "tvr": float(np.mean([m["tvr"] for m in metrics])),
            "lie": float(np.mean([m["lie"] for m in metrics])),
            "cpe": float(np.mean([m["cpe"] for m in metrics])),
            "frames_with_missing_colour": float(np.mean([m["n_missing_colours"] > 0 for m in metrics])),
            "median_foreground_pixels": float(np.median(fg_pixels)),
            "median_smallest_limb_pixels": float(np.median(min_limb_pixels)),
        }

    chosen, seen = [], set()
    for i, (_, group, _) in enumerate(rows):
        if group not in seen:
            chosen.append(i); seen.add(group)
        if len(chosen) == 6:
            break
    chosen += [i for i in range(len(rows)) if i not in chosen][: 6 - len(chosen)]

    cell, label_h, gap = 192, 52, 12
    canvas = Image.new("RGB", (gap + len(sizes) * (cell + gap), label_h + len(chosen) * (cell + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, body_font = font(25), font(17)
    for col, size in enumerate(sizes):
        x = gap + col * (cell + gap)
        draw.text((x + cell / 2, 10), f"{size} × {size}", fill=(25, 25, 25), font=title_font, anchor="ma")
    for row, index in enumerate(chosen):
        _, group, text = rows[index]
        y = label_h + row * (cell + label_h)
        for col, size in enumerate(sizes):
            x = gap + col * (cell + gap)
            image = Image.fromarray(composite_white(resized[size][index])).resize((cell, cell), Image.Resampling.NEAREST)
            canvas.paste(image, (x, y))
        caption = f"{group}: {text}"
        draw.text((gap, y + cell + 8), caption[:92], fill=(55, 55, 55), font=body_font)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolution-metrics.json").write_text(json.dumps(results, indent=2))
    canvas.save(out / "resolution-comparison.png", optimize=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
