"""Render released clips beside the matching fixed-noise UNet samples."""
from __future__ import annotations

import argparse
import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ROWS = (
    ("A person walks forward.", "locomotion/a_person_walks_forward_s2/c1"),
    ("A person runs forward.", "locomotion/a_person_runs_forward_s0/c0"),
    ("A person sits down cross-legged.", "transitions/a_person_sits_down_crosslegged_s1/c0"),
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def over_white_rgba(array: np.ndarray) -> Image.Image:
    rgba = Image.fromarray(array, "RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    return background.convert("RGB")


def decode_color(cell: dict) -> Image.Image:
    rgba = Image.open(io.BytesIO(cell["bytes"])).convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    return background.convert("RGB")


def load_released_clips(data_root: Path) -> list[list[Image.Image]]:
    shards = [
        path
        for split in ("train", "val", "test")
        for path in sorted(data_root.glob(f"{split}-*.parquet"))
    ]
    wanted = [clip_id for _, clip_id in ROWS]
    table = ds.dataset(shards, format="parquet").to_table(
        columns=["clip_id", "frame_idx", "color"],
        filter=ds.field("clip_id").isin(wanted),
    )
    grouped: dict[str, dict[int, Image.Image]] = {clip_id: {} for clip_id in wanted}
    for row in table.to_pylist():
        grouped[row["clip_id"]][row["frame_idx"]] = decode_color(row["color"])
    clips = []
    for clip_id in wanted:
        if sorted(grouped[clip_id]) != list(range(120)):
            raise RuntimeError(f"expected 120 released frames for {clip_id}")
        clips.append([grouped[clip_id][frame] for frame in range(120)])
    return clips


def render_frame(
    released: list[list[Image.Image]], generated: np.ndarray, frame_index: int
) -> Image.Image:
    canvas = Image.new("RGB", (1280, 960), "#f7f7f7")
    draw = ImageDraw.Draw(canvas)
    title_font, header_font, prompt_font, note_font = font(32, True), font(23, True), font(23), font(18)
    draw.text((52, 28), "Dancing Stick Figures — released clips and reference-model samples", fill="#202020", font=title_font)
    draw.text((315, 84), "released data", fill="#333333", font=header_font, anchor="mm")
    draw.text((965, 84), "factorised 3D UNet · 10k steps", fill="#333333", font=header_font, anchor="mm")

    panel = 224
    for row_index, (prompt, _) in enumerate(ROWS):
        y = 128 + row_index * 258
        draw.text((640, y - 10), prompt, fill="#202020", font=prompt_font, anchor="ms")
        left_x, right_x = 203, 853
        for x, image in (
            (left_x, released[row_index][frame_index]),
            (right_x, over_white_rgba(generated[row_index, frame_index])),
        ):
            draw.rounded_rectangle((x - 3, y + 8 - 3, x + panel + 3, y + 8 + panel + 3), radius=4, fill="#b5b5b5")
            canvas.paste(image.resize((panel, panel), Image.Resampling.NEAREST), (x, y + 8))

    seconds = frame_index / 20
    draw.text((52, 916), f"t = {seconds:0.2f} s  ·  frame {frame_index:03d}/119", fill="#444444", font=note_font)
    draw.text(
        (1228, 916),
        "fixed initial noise seed 1234 · 50 DDIM steps · guidance 3",
        fill="#444444",
        font=note_font,
        anchor="ra",
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/v1")
    parser.add_argument(
        "--samples",
        default="output/runpod/full64_native120_legible_suite/fixed_noise_varied_prompt_rgba.npz",
    )
    parser.add_argument("--out", default="output/video/unet_dataset_prompt_comparison.mp4")
    parser.add_argument("--gif", default="output/video/unet_dataset_prompt_comparison.gif")
    args = parser.parse_args()

    released = load_released_clips(ROOT / args.data)
    generated = np.load(ROOT / args.samples)["rgba"]
    if generated.shape != (len(ROWS), 120, 64, 64, 4):
        raise RuntimeError(f"unexpected generated shape: {generated.shape}")

    output, gif = ROOT / args.out, ROOT / args.gif
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dsf-unet-compare-") as temp:
        frames = Path(temp)
        for frame_index in range(120):
            render_frame(released, generated, frame_index).save(frames / f"{frame_index:04d}.png")
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "20",
                "-i", str(frames / "%04d.png"), "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "20",
                "-i", str(frames / "%04d.png"), "-vf", "fps=10,scale=960:-1:flags=lanczos",
                "-loop", "0", str(gif),
            ],
            check=True,
        )
    print(output)
    print(gif)


if __name__ == "__main__":
    main()
