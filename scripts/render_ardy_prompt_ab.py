#!/usr/bin/env python3
"""Render matched ARDY description-vs-command motions as a synchronized grid."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.ardy_adapter import render_clip


PAIRS = (
    ("walk", "Walk forward"),
    ("sit", "Sit down"),
    ("jacks", "Jumping jacks"),
    ("jump", "Jump in place"),
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, face, fill=(30, 30, 30)) -> None:
    box = draw.textbbox((0, 0), text, font=face)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), text, font=face, fill=fill)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    clips = {}
    fps = None
    for key, _ in PAIRS:
        for form in ("desc", "cmd"):
            frames, clip_fps, prompt = render_clip(str(args.input_dir / f"{form}_{key}.npz"))
            clips[(key, form)] = ([frame.convert("RGB") for frame in frames], prompt)
            fps = clip_fps if fps is None else fps

    width, height = 1920, 1080
    cell = 360
    group_x = (60, 990)
    group_y = (125, 590)
    title_font, action_font, prompt_font = font(34, True), font(25, True), font(20)
    n_frames = min(len(item[0]) for item in clips.values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None

    for frame_index in range(n_frames):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        centered(draw, (width // 2, 24), "Same ARDY model and seed — prompt form is the only change", title_font)

        for pair_index, (key, action) in enumerate(PAIRS):
            column, row = pair_index % 2, pair_index // 2
            x0, y0 = group_x[column], group_y[row]
            centered(draw, (x0 + 405, y0 - 46), action, action_font)

            for form_index, form in enumerate(("desc", "cmd")):
                frames, prompt = clips[(key, form)]
                x = x0 + form_index * 450
                label = "DESCRIPTION" if form == "desc" else "COMMAND"
                color = (80, 80, 80) if form == "desc" else (35, 95, 185)
                centered(draw, (x + cell // 2, y0 - 16), label, prompt_font, color)
                canvas.paste(frames[frame_index].resize((cell, cell), Image.Resampling.NEAREST), (x, y0))
                draw.rectangle((x, y0, x + cell, y0 + cell), outline=(195, 195, 195), width=2)
                centered(draw, (x + cell // 2, y0 + cell + 10), prompt, prompt_font, color)

        draw.text((width - 185, height - 38), f"frame {frame_index:03d}", font=prompt_font, fill=(100, 100, 100))
        process.stdin.write(canvas.tobytes())

    process.stdin.close()
    if process.wait() != 0:
        raise SystemExit("ffmpeg failed")


if __name__ == "__main__":
    main()
