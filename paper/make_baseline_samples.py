"""Build the paper figure for the fixed-noise 120-frame UNet prompt suite."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/runpod/full64_native120_prompt_suite/fixed_noise_varied_prompt_rgba.npz"
OUT = ROOT / "paper/figs/unet_prompt_samples"
FRAMES = (0, 23, 47, 71, 95, 119)
LABELS = (
    "energetic side steps",
    "slow, graceful dance",
    "cha-cha",
    "arms in the air",
    "ballet pirouette",
    "ballet plié",
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text"],
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def on_white(rgba):
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return np.clip(rgb * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)


def main():
    data = np.load(SOURCE)
    videos = data["rgba"]
    seeds = data["seeds"]
    if videos.shape != (6, 120, 64, 64, 4) or not np.all(seeds == 1234):
        raise RuntimeError(f"unexpected prompt-suite contract: {videos.shape}, seeds={seeds}")

    fig = plt.figure(figsize=(12.6, 4.25), facecolor="white")
    grid = fig.add_gridspec(2, 3, left=0.02, right=0.99, top=0.93, bottom=0.04,
                            wspace=0.10, hspace=0.25)
    for index, label in enumerate(LABELS):
        strip = grid[index // 3, index % 3].subgridspec(1, len(FRAMES), wspace=0.025)
        for column, frame in enumerate(FRAMES):
            ax = fig.add_subplot(strip[0, column])
            ax.imshow(on_white(videos[index, frame]), interpolation="nearest")
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
