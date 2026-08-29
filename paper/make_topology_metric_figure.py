"""Build a compact visual explanation of the three palette-structure metrics.

The four panels share one deterministic 64-pixel render.  Each corrupted panel
changes only the image pixels needed to trigger one metric, and every displayed
number is recomputed with :mod:`eval.oracle`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.oracle import NAMES as ORACLE_NAMES, PAL, label_colours, score_frame
from generator.render import render_all
from generator.skeleton import Body, Camera, Pose, fk3d, project
from scripts.analyze_resolution import resize_rgba


OUT = ROOT / "paper/figs/topology_metric_examples"
RESULTS = ROOT / "paper/results/topology_metric_examples.json"
METRICS = ("tvr", "lie", "cpe")


def clean_frame() -> np.ndarray:
    """Return the shared clean 64-pixel RGBA frame."""
    pose = Pose(abd={
        "arm_L": 1.15, "fore_L": 1.05, "hand_L": 0.90,
        "arm_R": 1.15, "fore_R": 1.05, "hand_R": 0.90,
        "leg_L": 0.18, "shin_L": 0.12, "foot_L": 0.10,
        "leg_R": 0.18, "shin_R": 0.12, "foot_R": 0.10,
    })
    body = Body(px_per_m=55.0, stroke=4.0)
    joints = fk3d(pose, body)
    projected, depth = project(joints, Camera(yaw=0.0, pitch=0.0, center=(64.0, 66.0)), body.px_per_m)
    native = np.asarray(render_all(projected, depth, body)["color"])
    return resize_rgba(native, 64)


def add_detached_component(image: np.ndarray) -> np.ndarray:
    """Add one detached left-upper-arm-colour component."""
    out = image.copy()
    out[7:11, 5:8, :3] = PAL["arm_L"]
    out[7:11, 5:8, 3] = 255
    return out


def break_distal_adjacency(image: np.ndarray) -> np.ndarray:
    """Move one distal color mask beyond the evaluator's two-pixel tolerance."""
    out = image.copy()
    labels, _ = label_colours(image)
    mask = labels == ORACLE_NAMES.index("fore_L")
    ys, xs = np.nonzero(mask)
    dx = -10 if float(xs.mean()) < image.shape[1] / 2 else 10
    out[mask] = 0
    moved_x = xs + dx
    keep = (moved_x >= 0) & (moved_x < image.shape[1])
    out[ys[keep], moved_x[keep]] = image[ys[keep], xs[keep]]
    return out


def add_off_palette_pixels(image: np.ndarray) -> np.ndarray:
    """Recolour the top of the ink mask with an off-palette grey."""
    out = image.copy()
    labels, _ = label_colours(image)
    ink = labels == ORACLE_NAMES.index("ink")
    ys, _ = np.nonzero(ink)
    row = np.indices(ink.shape)[0]
    patch = ink & (row <= int(ys.min()) + 4)
    out[patch, :3] = (150, 150, 150)
    return out


def examples() -> dict[str, np.ndarray]:
    clean = clean_frame()
    return {
        "clean": clean,
        "extra_component": add_detached_component(clean),
        "broken_adjacency": break_distal_adjacency(clean),
        "off_palette": add_off_palette_pixels(clean),
    }


def scores(images: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        name: {metric: float(score_frame(image)[metric]) for metric in METRICS}
        for name, image in images.items()
    }


def composite_white(image: np.ndarray) -> np.ndarray:
    alpha = image[..., 3:4].astype(np.float32) / 255.0
    return np.clip(image[..., :3] * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)


def main() -> None:
    images = examples()
    measured = scores(images)
    panels = (
        ("clean", "(a) Clean render", "all expected parts connected"),
        ("extra_component", "(b) Extra component", "TVR detects a split color mask"),
        ("broken_adjacency", "(c) Broken adjacency", "LIE detects a missing color edge"),
        ("off_palette", "(d) Off-palette pixels", "CPE counts unassigned foreground"),
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.35), facecolor="white")
    for axis, (key, title, explanation) in zip(axes, panels):
        axis.imshow(composite_white(images[key]), interpolation="nearest")
        axis.set_title(title, fontsize=9.5, fontweight="bold", pad=4)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.65)
        value = measured[key]
        axis.text(
            0.5, -0.075,
            f"TVR {value['tvr']:.3f}   LIE {value['lie']:.3f}   CPE {value['cpe']:.3f}",
            transform=axis.transAxes, ha="center", va="top", fontsize=8.4,
        )
        axis.text(0.5, -0.17, explanation, transform=axis.transAxes,
                  ha="center", va="top", fontsize=7.8, color="#444444")

    annotation_colour = "#136f78"
    axes[1].add_patch(Rectangle((4.5, 6.5), 4.0, 5.0, fill=False, edgecolor=annotation_colour,
                                linewidth=1.2, linestyle="--"))
    axes[2].add_patch(Rectangle((37.0, 16.0), 18.0, 13.0, fill=False, edgecolor=annotation_colour,
                                linewidth=1.2, linestyle="--"))
    axes[3].add_patch(Circle((31.5, 9.5), 6.0, fill=False, edgecolor=annotation_colour,
                             linewidth=1.2, linestyle="--"))
    fig.subplots_adjust(left=0.018, right=0.995, top=0.88, bottom=0.25, wspace=0.12)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    RESULTS.write_text(json.dumps({
        "resolution": 64,
        "source": "deterministic renderer frame shared by all four panels",
        "corruptions": {
            "extra_component": "add one detached arm_L-color component",
            "broken_adjacency": "translate fore_L mask 10 pixels beyond the two-pixel adjacency tolerance",
            "off_palette": "replace an ink-region patch with RGB (150,150,150)",
        },
        "scores": measured,
    }, indent=2) + "\n")
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
