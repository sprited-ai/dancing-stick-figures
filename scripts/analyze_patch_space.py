"""Measure sparsity and diversity of raw RGBA patches in the released dataset.

The analysis matches the pixel-video loader: source RGBA is premultiplied and
area-downsampled to 64x64 before extracting non-overlapping patches.

Example:
  python scripts/analyze_patch_space.py --data data/v1 \
      --out-dir output/patch_space --n 1024 --patches 2,4
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator.skeleton import NAMES
from scripts.analyze_resolution import resize_rgba


def resize_seg_majority(seg: np.ndarray, size: int) -> np.ndarray:
    """Integer-factor majority resize, including background as a real vote."""
    if seg.shape == (size, size):
        return seg.astype(np.uint8, copy=True)
    factor = seg.shape[0] // size
    if seg.shape[0] != seg.shape[1] or factor * size != seg.shape[0]:
        raise ValueError(f"seg shape {seg.shape} is not integer-resizable to {size}")
    blocks = seg.reshape(size, factor, size, factor).transpose(0, 2, 1, 3).reshape(size, size, factor * factor)
    counts = np.stack([(blocks == label).sum(-1) for label in range(len(NAMES) + 1)], axis=-1)
    return counts.argmax(-1).astype(np.uint8)


def patchify(array: np.ndarray, patch: int) -> np.ndarray:
    h, w = array.shape[:2]
    if h % patch or w % patch:
        raise ValueError(f"{patch} does not divide {h}x{w}")
    trailing = array.shape[2:]
    x = array.reshape(h // patch, patch, w // patch, patch, *trailing)
    axes = (0, 2, 1, 3) + tuple(range(4, x.ndim))
    return x.transpose(axes).reshape((h // patch) * (w // patch), patch * patch, *trailing)


def _entropy(counts: np.ndarray) -> float:
    probability = counts.astype(np.float64) / counts.sum()
    return float(-(probability * np.log2(probability)).sum())


def summarize(features: np.ndarray, alpha_occupancy: np.ndarray, seg_patches: np.ndarray,
              patch: int, max_analysis_patches: int = 50_000) -> dict[str, object]:
    nonblank = alpha_occupancy > 0
    foreground = features[nonblank].reshape(nonblank.sum(), -1)
    occupancy = alpha_occupancy[nonblank]
    segments = seg_patches[nonblank]
    if len(foreground) > max_analysis_patches:
        selection = np.linspace(0, len(foreground) - 1, max_analysis_patches, dtype=np.int64)
        sample = foreground[selection]
    else:
        sample = foreground

    centered = sample.astype(np.float64) - sample.mean(axis=0, dtype=np.float64)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0)
    total_variance = eigenvalues.sum()
    cumulative = np.cumsum(eigenvalues) / max(total_variance, 1e-12)
    effective_dimension = float(total_variance**2 / max(np.square(eigenvalues).sum(), 1e-12))

    quantized = np.rint(np.clip(sample, 0, 1) * 15).astype(np.uint8)
    _, counts = np.unique(quantized, axis=0, return_counts=True)
    entropy = _entropy(counts)
    segment_stats = []
    for label, name in enumerate(NAMES, start=1):
        present = np.any(segments == label, axis=1)
        pixels = int((segments == label).sum())
        segment_stats.append({"joint": name, "patches": int(present.sum()),
                              "fraction_of_nonblank_patches": float(present.mean()),
                              "pixels_in_nonblank_patches": pixels})
    rarest = sorted((row for row in segment_stats if row["patches"]), key=lambda row: row["patches"])[:5]
    fg_pixel_count = (segments > 0).sum(axis=1)
    return {
        "patch": patch,
        "raw_dimensions": int(patch * patch * 4),
        "total_patches": int(len(features)),
        "blank_patch_fraction": float(1 - nonblank.mean()),
        "nonblank_patches": int(nonblank.sum()),
        "foreground_pixel_occupancy_nonblank": {
            "mean": float(occupancy.mean()), "p10": float(np.quantile(occupancy, 0.1)),
            "median": float(np.median(occupancy)), "p90": float(np.quantile(occupancy, 0.9))},
        "thin_nonblank_fraction_le_25pct_alpha_pixels": float((occupancy <= 0.25).mean()),
        "single_seg_pixel_fraction_nonblank": float((fg_pixel_count == 1).mean()),
        "multi_segment_fraction_nonblank": float((np.asarray([(np.unique(row[row > 0]).size) for row in segments]) >= 2).mean()),
        "pca": {"analysis_patches": int(len(sample)), "total_variance": float(total_variance),
                "effective_dimension_participation_ratio": effective_dimension,
                "components_for_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
                "components_for_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
                "top_component_variance_fraction": float(cumulative[0])},
        "quantized_4bit_rgba": {"unique_codes": int(len(counts)),
                "unique_code_fraction": float(len(counts) / len(sample)),
                "singleton_patch_fraction": float(counts[counts == 1].sum() / len(sample)),
                "shannon_entropy_bits": entropy,
                "normalized_entropy_vs_sample_count": float(entropy / math.log2(max(2, len(sample))))},
        "rarest_segment_patch_presence": rarest,
        "per_segment": segment_stats,
    }


def font(size: int):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_summary(results: dict[str, object], out: Path) -> None:
    patches = results["patches"]
    width, height = 1120, 680
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title, body, small = font(30), font(20), font(16)
    draw.text((40, 28), "Raw premultiplied RGBA patch space at 64 x 64", fill="#171717", font=title)
    draw.text((40, 70), f"{results['frames']} released train frames, deterministic sampling", fill="#555555", font=body)

    metrics = [
        ("Blank patches", lambda r: r["blank_patch_fraction"], 1.0, "%"),
        ("Thin among nonblank", lambda r: r["thin_nonblank_fraction_le_25pct_alpha_pixels"], 1.0, "%"),
        ("PCA effective / raw dim", lambda r: r["pca"]["effective_dimension_participation_ratio"] / r["raw_dimensions"], 1.0, "%"),
        ("Quantized singleton patches", lambda r: r["quantized_4bit_rgba"]["singleton_patch_fraction"], 1.0, "%"),
        ("Normalized patch entropy", lambda r: r["quantized_4bit_rgba"]["normalized_entropy_vs_sample_count"], 1.0, "%"),
    ]
    colors = ["#4776E6", "#E45756"]
    start_y = 125
    for row, (label, getter, maximum, unit) in enumerate(metrics):
        y = start_y + row * 92
        draw.text((40, y), label, fill="#202020", font=body)
        for column, (patch_name, result) in enumerate(patches.items()):
            value = getter(result)
            x = 340 + column * 360
            draw.rounded_rectangle((x, y, x + 280, y + 26), radius=8, fill="#eeeeee")
            draw.rounded_rectangle((x, y, x + 280 * min(value / maximum, 1), y + 26), radius=8,
                                   fill=colors[column])
            formatted = f"{100 * value:.1f}{unit}" if unit == "%" else f"{value:.2f}"
            draw.text((x, y + 34), f"{patch_name} patch: {formatted}", fill="#404040", font=small)

    y = 600
    p2, p4 = patches["2x2"], patches["4x4"]
    conclusion = (f"Measured result: blank background dominates both ({p2['blank_patch_fraction']:.1%} / "
                  f"{p4['blank_patch_fraction']:.1%}); among nonblank patches, 4-bit code singleton rates are "
                  f"{p2['quantized_4bit_rgba']['singleton_patch_fraction']:.1%} / "
                  f"{p4['quantized_4bit_rgba']['singleton_patch_fraction']:.1%}.")
    draw.text((40, y), conclusion, fill="#202020", font=small)
    image.save(out, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--patches", default="2,4")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-analysis-patches", type=int, default=50_000)
    args = parser.parse_args()
    patch_sizes = [int(value) for value in args.patches.split(",")]
    files = sorted(Path(args.data).glob(f"{args.split}-*.parquet"))
    if not files:
        raise SystemExit(f"no {args.split} parquet shards under {args.data}")

    per_file = max(1, math.ceil(args.n / len(files)))
    rows = []
    for path in files:
        table = pq.read_table(path, columns=["color", "seg"])
        indices = np.linspace(0, table.num_rows - 1, min(per_file, table.num_rows), dtype=np.int64)
        for row in table.take(indices).to_pylist():
            rgba = np.asarray(Image.open(io.BytesIO(row["color"]["bytes"])).convert("RGBA"))
            seg = np.asarray(Image.open(io.BytesIO(row["seg"]["bytes"])).convert("L"))
            rows.append((rgba, seg))
    rows = rows[:args.n]

    accumulators = {patch: {"features": [], "occupancy": [], "seg": []} for patch in patch_sizes}
    for rgba, seg in rows:
        small = resize_rgba(rgba, args.size).astype(np.float32) / 255.0
        alpha = small[..., 3:4]
        premult = np.concatenate([small[..., :3] * alpha, alpha], axis=-1)
        small_seg = resize_seg_majority(seg, args.size)
        for patch in patch_sizes:
            rgba_patches = patchify(premult, patch)
            alpha_pixels = rgba_patches[..., 3] > (1 / 255)
            accumulators[patch]["features"].append(rgba_patches.reshape(len(rgba_patches), -1))
            accumulators[patch]["occupancy"].append(alpha_pixels.mean(axis=1))
            accumulators[patch]["seg"].append(patchify(small_seg[..., None], patch)[..., 0])

    results = {"source": str(Path(args.data).resolve()), "split": args.split,
               "frames": len(rows), "resolution": args.size,
               "representation": "premultiplied straight-RGBA area-downsampled to 64, values [0,1]",
               "blank_definition": "no patch alpha pixel greater than 1/255",
               "sampling": "evenly spaced rows per parquet shard, then deterministic truncation",
               "patches": {}}
    for patch in patch_sizes:
        values = accumulators[patch]
        summary = summarize(np.concatenate(values["features"]), np.concatenate(values["occupancy"]),
                            np.concatenate(values["seg"]), patch, args.max_analysis_patches)
        results["patches"][f"{patch}x{patch}"] = summary

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "patch-space-metrics.json"
    png_path = out / "patch-space-summary.png"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    draw_summary(results, png_path)
    print(json.dumps({"json": str(json_path.resolve()), "png": str(png_path.resolve()),
                      "frames": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
