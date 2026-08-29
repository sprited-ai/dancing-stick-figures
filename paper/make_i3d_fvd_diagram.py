"""Make an onion-skin diagram that distinguishes clip distance from FVD.

The examples use fixed clips from the final mini release.  A per-clip I3D
distance is shown for each illustrated pair; the pooled medians come from the
pre-registered 256-video embedding validation.  The final panel makes explicit
that FVD is computed between feature *distributions*, not between two clips.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pyarrow.dataset as ds
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.fvd import features


CLIPS = {
    "run_seed_0": "locomotion/a_person_runs_forward_s0/c0",
    "run_seed_1": "locomotion/a_person_runs_forward_s1/c0",
    "wave_seed_0": "gesture/a_person_waves_hello_with_the_right_hand_s0/c0",
}
FRAME_INDICES = (10, 20, 30, 40)
INK = "#202124"
TEAL = "#136f78"
ORANGE = "#c55a11"
BLUE = "#3568a8"


def load_clips(data: Path) -> dict[str, np.ndarray]:
    paths = sorted(data.glob("*.parquet"))
    if not paths:
        raise ValueError(f"no parquet files under {data}")
    dataset = ds.dataset([str(path) for path in paths], format="parquet")
    wanted = list(CLIPS.values())
    table = dataset.to_table(
        columns=["clip_id", "frame_idx", "color"],
        filter=ds.field("clip_id").isin(wanted),
    )
    rows: dict[str, list[tuple[int, np.ndarray]]] = {clip_id: [] for clip_id in wanted}
    for row in table.to_pylist():
        image = np.asarray(Image.open(io.BytesIO(row["color"]["bytes"])).convert("RGBA"))
        rows[row["clip_id"]].append((int(row["frame_idx"]), image))
    result = {}
    for name, clip_id in CLIPS.items():
        ordered = sorted(rows[clip_id], key=lambda item: item[0])
        if [index for index, _ in ordered] != list(range(120)):
            raise ValueError(f"{clip_id} does not contain exactly frames 0..119")
        result[name] = np.stack([image for _, image in ordered])
    return result


def composite_white(video: np.ndarray) -> np.ndarray:
    alpha = video[..., 3:4].astype(np.float32) / 255.0
    return np.clip(video[..., :3] * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)


def onion_skin(video: np.ndarray) -> np.ndarray:
    """Composite five temporal samples over white, oldest to newest."""
    canvas = np.full(video.shape[1:3] + (3,), 255.0, np.float32)
    opacities = (0.22, 0.32, 0.48, 0.88)
    for frame_index, opacity in zip(FRAME_INDICES, opacities):
        rgba = video[frame_index].astype(np.float32)
        alpha = rgba[..., 3:4] / 255.0 * opacity
        canvas = rgba[..., :3] * alpha + canvas * (1.0 - alpha)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def arrow(axis, left: tuple[float, float], right: tuple[float, float], colour: str) -> None:
    axis.add_patch(FancyArrowPatch(left, right, arrowstyle="<->", mutation_scale=11,
                                   linewidth=1.35, color=colour))


def metric_box(axis, xy, width, height, title, detail, colour):
    axis.add_patch(FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.018,rounding_size=0.018",
        facecolor="#f7fafb", edgecolor=colour, linewidth=1.15,
    ))
    axis.text(xy[0] + width / 2, xy[1] + height * 0.66, title,
              ha="center", va="center", fontsize=9.3, fontweight="bold", color=INK)
    axis.text(xy[0] + width / 2, xy[1] + height * 0.31, detail,
              ha="center", va="center", fontsize=7.7, color="#4b5358")


def make_figure(videos: dict[str, np.ndarray], distances: dict, validation: dict,
                out_stem: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(10.8, 4.15), facecolor="white")
    grid = fig.add_gridspec(2, 6, width_ratios=(1.0, .34, 1.0, .32, 1.16, 1.16),
                            wspace=.12, hspace=.44)

    pairs = [
        ("run_seed_0", "run_seed_1", "Same prompt, different motion seeds",
         "run, seed 0", "run, seed 1", TEAL, "same_prompt"),
        ("run_seed_0", "wave_seed_0", "Different prompts",
         "run, seed 0", "wave right hand, seed 0", ORANGE, "different_prompt"),
    ]
    for row, (left, right, heading, left_label, right_label, colour, key) in enumerate(pairs):
        ax_left = fig.add_subplot(grid[row, 0])
        ax_mid = fig.add_subplot(grid[row, 1])
        ax_right = fig.add_subplot(grid[row, 2])
        for axis, name, label in ((ax_left, left, left_label), (ax_right, right, right_label)):
            axis.imshow(onion_skin(videos[name]), interpolation="nearest")
            axis.set_xticks([]); axis.set_yticks([])
            axis.set_title(label, fontsize=8.8, pad=3)
            for spine in axis.spines.values():
                spine.set_color("#8a8f93"); spine.set_linewidth(.65)
        ax_left.text(-.08, 1.12, f"({'a' if row == 0 else 'b'}) {heading}",
                     transform=ax_left.transAxes, ha="left", va="bottom",
                     fontsize=10.0, fontweight="bold", color=INK)
        ax_mid.set_xlim(0, 1); ax_mid.set_ylim(0, 1); ax_mid.axis("off")
        arrow(ax_mid, (.06, .61), (.94, .61), colour)
        ax_mid.text(.5, .41, f"I3D L2\n{distances[key]:.2f}", ha="center", va="top",
                    fontsize=8.4, color=colour, fontweight="bold")

    ax_flow = fig.add_subplot(grid[:, 4:])
    ax_flow.set_xlim(0, 1); ax_flow.set_ylim(0, 1); ax_flow.axis("off")
    ax_flow.text(.5, .97, "From clip embeddings to FVD", ha="center", va="top",
                 fontsize=10.0, fontweight="bold", color=INK)
    metric_box(ax_flow, (.07, .70), .35, .16, "Video set A", "many clips → I3D features", BLUE)
    metric_box(ax_flow, (.58, .70), .35, .16, "Video set B", "many clips → I3D features", ORANGE)
    arrow(ax_flow, (.245, .68), (.245, .54), BLUE)
    arrow(ax_flow, (.755, .68), (.755, .54), ORANGE)
    metric_box(ax_flow, (.07, .36), .35, .16, "Distribution A", "feature mean + covariance", BLUE)
    metric_box(ax_flow, (.58, .36), .35, .16, "Distribution B", "feature mean + covariance", ORANGE)
    arrow(ax_flow, (.42, .44), (.58, .44), TEAL)
    ax_flow.text(.5, .30, "Fréchet distance between the two distributions = FVD",
                 ha="center", va="center", fontsize=8.6, color=TEAL, fontweight="bold")
    ax_flow.text(.5, .20, "The illustrated L2 values describe individual clip pairs;\n"
                 "FVD itself requires two sets of videos.", ha="center", va="top",
                 fontsize=7.8, color="#4b5358")

    pooled = validation["clean_pair_distance"]
    frame_text = ", ".join(str(index) for index in FRAME_INDICES[:-1]) + f", and {FRAME_INDICES[-1]}"
    fig.text(.025, .015,
             f"Onion skins overlay frames {frame_text}.  In the fixed 256-clip validation, "
             f"same-prompt/different-seed pairs had median I3D L2 {pooled['same_prompt_different_seed']['median']:.2f}; "
             f"hard same-group/different-prompt pairs had median {pooled['different_prompt_same_group']['median']:.2f}. "
             "The run–wave row is a fixed cross-category illustration, not the pooled negative protocol.",
             ha="left", va="bottom", fontsize=7.6, color="#444444")
    fig.subplots_adjust(left=.035, right=.99, top=.91, bottom=.145)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation", type=Path,
                        default=ROOT / "paper/results/i3d_embedding_validation_120f_v02c.json")
    parser.add_argument("--out", type=Path, default=ROOT / "paper/figs/i3d_fvd_onion_skin")
    parser.add_argument("--results", type=Path,
                        default=ROOT / "paper/results/i3d_fvd_onion_skin.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    videos = load_clips(args.data)
    names = list(CLIPS)
    embeddings = features(np.stack([composite_white(videos[name]) for name in names]),
                          device=args.device, bs=3)
    by_name = dict(zip(names, embeddings))
    distances = {
        "same_prompt": float(np.linalg.norm(by_name["run_seed_0"] - by_name["run_seed_1"])),
        "different_prompt": float(np.linalg.norm(by_name["run_seed_0"] - by_name["wave_seed_0"])),
    }
    validation = json.loads(args.validation.read_text())
    make_figure(videos, distances, validation, args.out)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps({
        "clips": CLIPS,
        "frame_indices": FRAME_INDICES,
        "pairwise_i3d_l2": distances,
        "guardrail": "Pairwise I3D L2 is illustrative; FVD is distribution-level.",
        "pooled_validation": {
            "same_prompt_different_seed_median": validation["clean_pair_distance"]["same_prompt_different_seed"]["median"],
            "different_prompt_same_group_median": validation["clean_pair_distance"]["different_prompt_same_group"]["median"],
        },
    }, indent=2) + "\n")
    print(args.out.with_suffix(".pdf"))
    print(json.dumps(distances, indent=2))


if __name__ == "__main__":
    main()
