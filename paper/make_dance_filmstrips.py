"""Build a film-strip survey of easily readable motions in the dataset."""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import pyarrow.dataset as ds
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/v1"
OUT = ROOT / "paper/figs/dance_filmstrips"
MOVES = (
    ("walking forward", "locomotion/a_person_walks_forward_s2/c1", (20, 30, 40, 50)),
    ("running forward", "locomotion/a_person_runs_forward_s0/c0", (0, 5, 10, 15)),
    ("sitting cross-legged", "transitions/a_person_sits_down_crosslegged_s1/c0", (0, 15, 30, 50)),
    ("jumping jacks", "sport/a_person_does_jumping_jacks_s0/c0", (0, 8, 18, 30)),
    ("ballet pirouette", "dance/a_person_does_a_ballet_pirouette_s1/c0", (0, 15, 30, 45)),
    ("YMCA dance", "dance/a_person_does_the_ymca_dance_s0/c0", (0, 4, 8, 12)),
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text"],
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def decode(cell):
    rgba = Image.open(io.BytesIO(cell["bytes"])).convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    return background.convert("RGB")


def main():
    shards = [path for split in ("train", "val", "test")
              for path in sorted(SOURCE.glob(f"{split}-*.parquet"))]
    dataset = ds.dataset(shards, format="parquet")
    table = dataset.to_table(columns=["clip_id", "frame_idx", "color"])
    rows = table.to_pylist()

    fig = plt.figure(figsize=(12.6, 4.25), facecolor="white")
    grid = fig.add_gridspec(2, 3, left=0.02, right=0.99, top=0.93, bottom=0.04,
                            wspace=0.10, hspace=0.25)
    for index, (label, clip_id, frames) in enumerate(MOVES):
        by_frame = {row["frame_idx"]: row for row in rows if row["clip_id"] == clip_id}
        if len(by_frame) != 120:
            raise RuntimeError(f"expected a complete clip for {clip_id}")
        strip = grid[index // 3, index % 3].subgridspec(1, 4, wspace=0.025)
        for column, frame in enumerate(frames):
            ax = fig.add_subplot(strip[0, column])
            ax.imshow(decode(by_frame[frame]["color"]), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.55); spine.set_edgecolor("#8a8a8a")
        x = 0.02 + (index % 3) * 0.333
        y = 0.965 if index < 3 else 0.49
        fig.text(x, y, f"({chr(97 + index)})", fontsize=9, fontweight="bold", va="top")
        fig.text(x + 0.028, y, label, fontsize=8, va="top")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
