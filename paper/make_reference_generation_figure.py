"""Pair released clips with representative 30k-step Pixel DiT samples."""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as ds
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/v1"
GENERATION_STRIP = (
    ROOT / "paper/results/figure1_factorised_image30k_preview/prompt_diverse_grid_strip.png"
)
OUT = ROOT / "paper/figs/reference_generation_pairs"

# The saved strip contains four prompt rows and four 64x64 frame columns after
# a 192-pixel text margin. Its manifest records frames 0/21/42/63.
STRIP_CELL = 64
STRIP_MARGIN = 192
FRAMES = (0, 21, 42, 63)
BLOCKS = (
    ("run forward", 0, "locomotion/a_person_runs_forward_s0/c0"),
    ("wave left", 1, "gesture/a_person_waves_hello_with_the_left_hand_s0/c0"),
    ("sit down", 2, "transitions/a_person_sits_down_on_a_chair_s0/c0"),
    ("dance", 3, "dance/a_person_dances_energetically_waving_arms_and_stepping_side__s0/c0"),
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


def load_strip(path):
    image = np.asarray(Image.open(path).convert("RGB"))
    expected = (4 * STRIP_CELL, STRIP_MARGIN + len(FRAMES) * STRIP_CELL, 3)
    if image.shape != expected:
        raise RuntimeError(f"unexpected strip shape for {path}: {image.shape} != {expected}")
    return image


def strip_cell(strip, row, column):
    y0 = row * STRIP_CELL
    x0 = STRIP_MARGIN + column * STRIP_CELL
    return strip[y0 : y0 + STRIP_CELL, x0 : x0 + STRIP_CELL]


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-strip", type=Path, default=GENERATION_STRIP)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--row-label", default="DiT")
    options = parser.parse_args()
    shards = [
        path
        for split in ("train", "val", "test")
        for path in sorted(DATA.glob(f"{split}-*.parquet"))
    ]
    rows = ds.dataset(shards, format="parquet").to_table(
        columns=["clip_id", "frame_idx", "color"]
    ).to_pylist()
    by_clip = {}
    wanted = {clip_id for _, _, clip_id in BLOCKS}
    for row in rows:
        if row["clip_id"] in wanted:
            by_clip.setdefault(row["clip_id"], {})[row["frame_idx"]] = row

    generated = load_strip(options.generation_strip)

    fig = plt.figure(figsize=(12.6, 1.7), facecolor="white")
    outer = fig.add_gridspec(
        1, len(BLOCKS), left=0.008, right=0.997, top=0.985, bottom=0.008, wspace=0.02
    )
    for block, (label, strip_row, clip_id) in enumerate(BLOCKS):
        if len(by_clip.get(clip_id, {})) != 120:
            raise RuntimeError(f"expected a complete released clip for {clip_id}")
        block_grid = outer[0, block].subgridspec(
            3, 5, height_ratios=(0.13, 1, 1), width_ratios=(0.34, 1, 1, 1, 1),
            hspace=0.012, wspace=0.012,
        )
        title = fig.add_subplot(block_grid[0, :])
        title.axis("off")
        title.text(0, 0.55, label, fontsize=8.2, fontweight="bold", va="center")

        for row_index, row_label in enumerate(("dataset", options.row_label), start=1):
            label_ax = fig.add_subplot(block_grid[row_index, 0])
            label_ax.axis("off")
            label_ax.text(0.96, 0.5, row_label, ha="right", va="center", color="#333333")
            for column, source_frame in enumerate(FRAMES):
                ax = fig.add_subplot(block_grid[row_index, column + 1])
                image = (
                    decode(by_clip[clip_id][source_frame]["color"])
                    if row_index == 1
                    else strip_cell(generated, strip_row, column)
                )
                ax.imshow(crop_display(image), interpolation="nearest")
                style_frame(ax)

    options.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(options.out.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(options.out.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.01)
    print(options.out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
