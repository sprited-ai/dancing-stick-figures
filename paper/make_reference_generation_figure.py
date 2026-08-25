"""Pair released clips with fixed-noise DiT samples for the same prompts."""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as ds
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/v1"
GENERATIONS = (
    ROOT
    / "output/inference/m3_diverse_prompts_10k_rgba/prompt_diverse_grid_rgba.npz"
)
OUT = ROOT / "paper/figs/reference_generation_pairs"

# Prompt index, released clip, and matched (source-frame, model-frame) keyframes.
# The model uses every second source frame over a five-second training window.
PAIRS = (
    (
        "wave left",
        2,
        "gesture/a_person_waves_hello_with_the_left_hand_s1/c0",
        ((0, 0), (18, 9), (38, 19), (58, 29)),
    ),
    (
        "wave right",
        3,
        "gesture/a_person_waves_hello_with_the_right_hand_s1/c0",
        ((0, 0), (18, 9), (38, 19), (58, 29)),
    ),
    (
        "run forward",
        4,
        "locomotion/a_person_runs_forward_s0/c0",
        ((0, 0), (18, 9), (38, 19), (58, 29)),
    ),
    (
        "walk backward",
        5,
        "locomotion/a_person_walks_backwards_s1/c0",
        ((0, 0), (18, 9), (38, 19), (58, 29)),
    ),
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIX Two Text"],
        "font.size": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def decode(cell):
    rgba = Image.open(io.BytesIO(cell["bytes"])).convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    white.alpha_composite(rgba)
    return np.asarray(white.convert("RGB"))


def on_white(rgba):
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return np.clip(rgb * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)


def crop_display(image, border_fraction=0.125):
    """Remove the shared outer canvas while preserving matched coordinates."""
    height, width = image.shape[:2]
    border_y = round(height * border_fraction)
    border_x = round(width * border_fraction)
    return image[border_y : height - border_y, border_x : width - border_x]


def style_frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_edgecolor("#999999")


def main():
    shards = [
        path
        for split in ("train", "val", "test")
        for path in sorted(DATA.glob(f"{split}-*.parquet"))
    ]
    rows = ds.dataset(shards, format="parquet").to_table(
        columns=["clip_id", "frame_idx", "color"]
    ).to_pylist()
    by_clip = {}
    wanted = {clip_id for _, _, clip_id, _ in PAIRS}
    for row in rows:
        if row["clip_id"] in wanted:
            by_clip.setdefault(row["clip_id"], {})[row["frame_idx"]] = row

    generated = np.load(GENERATIONS)["rgba"]
    if generated.shape != (8, 50, 64, 64, 4):
        raise RuntimeError(f"unexpected generated-video shape: {generated.shape}")

    fig = plt.figure(figsize=(12.6, 1.7), facecolor="white")
    outer = fig.add_gridspec(
        1, len(PAIRS), left=0.008, right=0.997, top=0.985, bottom=0.008, wspace=0.02
    )
    for block, (label, sample_index, clip_id, frames) in enumerate(PAIRS):
        if len(by_clip.get(clip_id, {})) != 120:
            raise RuntimeError(f"expected a complete released clip for {clip_id}")
        block_grid = outer[0, block].subgridspec(
            3, 5, height_ratios=(0.13, 1, 1), width_ratios=(0.34, 1, 1, 1, 1),
            hspace=0.012, wspace=0.012,
        )
        title = fig.add_subplot(block_grid[0, :])
        title.axis("off")
        title.text(0, 0.55, label, fontsize=8.2, fontweight="bold", va="center")

        for row_index, row_label in enumerate(("dataset", "DiT"), start=1):
            label_ax = fig.add_subplot(block_grid[row_index, 0])
            label_ax.axis("off")
            label_ax.text(0.96, 0.5, row_label, ha="right", va="center", color="#333333")
            for column, (source_frame, model_frame) in enumerate(frames):
                ax = fig.add_subplot(block_grid[row_index, column + 1])
                image = (
                    decode(by_clip[clip_id][source_frame]["color"])
                    if row_label == "dataset"
                    else on_white(generated[sample_index, model_frame])
                )
                ax.imshow(crop_display(image), interpolation="nearest")
                style_frame(ax)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.01)
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
