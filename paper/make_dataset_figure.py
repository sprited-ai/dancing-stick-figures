"""Build the dataset-anatomy figure directly from a released parquet shard."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as ds
from matplotlib import colormaps
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.keyframes import select_representative_frames
from generator.rebuild_from_motion import _body_and_cameras
from generator.render import render_all
from generator.skeleton import NAMES, PARENT, project


SOURCE = ROOT / "data/v1"
OUT = ROOT / "paper/figs/dataset_anatomy"
EXAMPLE_MOTION = "dance/a_person_does_the_running_man_dance_s0"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def decode(cell, mode=None):
    image = Image.open(io.BytesIO(cell["bytes"]))
    return image.convert(mode) if mode else image.copy()


def on_white(image):
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    return np.asarray(background.convert("RGB"))


def segmentation(cell):
    ids = np.asarray(decode(cell)).astype(np.int32)
    palette = colormaps["turbo"](np.linspace(0.05, 0.95, 28))[..., :3]
    rgb = (palette[np.clip(ids, 0, 27)] * 255).astype(np.uint8)
    rgb[ids == 0] = 255
    return rgb


def depth_image(cell, alpha):
    depth = np.asarray(decode(cell)).astype(np.float32)
    mask = alpha > 8
    rgb = np.full((*depth.shape, 3), 255, np.uint8)
    if mask.any():
        low, high = np.percentile(depth[mask], [2, 98])
        value = np.clip((depth - low) / max(high - low, 1.0), 0.0, 1.0)
        mapped = (colormaps["viridis"](1.0 - value)[..., :3] * 255).astype(np.uint8)
        rgb[mask] = mapped[mask]
    return rgb


def joint_overlay(row):
    base = decode(row["color"], "RGBA")
    canvas = Image.new("RGBA", base.size, "white"); canvas.alpha_composite(base)
    draw = ImageDraw.Draw(canvas)
    xy = np.frombuffer(row["joint_xy"], np.float32).reshape(27, 2) * base.width
    visible = np.frombuffer(row["joint_visible"], np.uint8).astype(bool)
    for child, name in enumerate(NAMES):
        parent_name = PARENT.get(name)
        if parent_name is None:
            continue
        parent = NAMES.index(parent_name)
        draw.line((tuple(xy[parent]), tuple(xy[child])), fill=(0, 85, 0, 230), width=2)
    for index, (x, y) in enumerate(xy):
        colour = (0, 190, 70, 255) if visible[index] else (220, 50, 40, 255)
        draw.ellipse((x - 2.2, y - 2.2, x + 2.2, y + 2.2), fill=colour,
                     outline=(20, 20, 20, 255), width=1)
    return np.asarray(canvas.convert("RGB"))


def show(ax, image, title=""):
    ax.imshow(image, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=7.5, fontweight="normal", pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6); spine.set_edgecolor("#777777")


def rerender_view(row, camera, body):
    xyz = np.frombuffer(row["joint_xyz"], np.float32).reshape(len(NAMES), 3)
    joints3 = {name: xyz[index] for index, name in enumerate(NAMES)}
    joints2, joint_depth = project(joints3, camera, body.px_per_m)
    return on_white(render_all(joints2, joint_depth, body)["color"])


def main():
    dataset = ds.dataset(sorted(SOURCE.glob("*.parquet")), format="parquet")
    table = dataset.to_table(filter=ds.field("clip_id") == EXAMPLE_MOTION + "/c0")
    rows = table.to_pylist()
    if len(rows) != 120:
        raise RuntimeError(f"expected 120 example frames, found {len(rows)}")
    motion_id = EXAMPLE_MOTION
    by_frame = {row["frame_idx"]: row for row in rows}
    joints = np.stack([
        np.frombuffer(by_frame[f]["joint_xyz"], np.float32).reshape(27, 3) for f in range(120)
    ])
    strip_frames = select_representative_frames(joints, count=5)
    focus_frame = strip_frames[len(strip_frames) // 2]
    focus = by_frame[focus_frame]
    body, cameras = _body_and_cameras(motion_id, "", {})

    fig = plt.figure(figsize=(12.6, 4.1), facecolor="white")
    outer = fig.add_gridspec(2, 2, width_ratios=(0.42, 0.58), height_ratios=(1, 1),
                             left=0.025, right=0.985, top=0.94, bottom=0.04,
                             wspace=0.08, hspace=0.23)
    cams = outer[0, 0].subgridspec(1, 3, wspace=0.08)
    for camera in range(3):
        show(fig.add_subplot(cams[0, camera]), rerender_view(focus, cameras[camera], body),
             f"view {camera + 1}")

    timeline = outer[0, 1].subgridspec(1, 5, wspace=0.06)
    for column, frame in enumerate(strip_frames):
        show(fig.add_subplot(timeline[0, column]), on_white(decode(by_frame[frame]["color"])),
             rf"$t={frame / 20:.1f}\,$s")

    labels = outer[1, :].subgridspec(1, 5, wspace=0.06)
    rgba = decode(focus["color"], "RGBA")
    label_images = (
        (on_white(rgba), "RGBA"),
        (segmentation(focus["seg"]), "27 parts"),
        (depth_image(focus["depth"], np.asarray(rgba)[..., 3]), "depth"),
        (on_white(decode(focus["normal"])), "normals"),
        (joint_overlay(focus), "projected joints"),
    )
    for column, (image, title) in enumerate(label_images):
        show(fig.add_subplot(labels[0, column]), image, title)

    fig.text(0.025, 0.965, "(a)", fontsize=9, fontweight="bold")
    fig.text(0.455, 0.965, "(b)", fontsize=9, fontweight="bold")
    fig.text(0.025, 0.49, "(c)", fontsize=9, fontweight="bold")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
