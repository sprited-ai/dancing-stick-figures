#!/usr/bin/env python3
"""Render two ARDY motions side by side with synchronized frames."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.ardy_adapter import load, render_clip, root_trajectory


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def centered(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, face, fill=(30, 30, 30)) -> None:
    box = draw.textbbox((0, 0), text, font=face)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=face, fill=fill)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="Matched ARDY sampling")
    args = parser.parse_args()

    items = []
    for path in (args.left, args.right):
        frames, fps, prompt = render_clip(str(path))
        posed, _, _ = load(str(path))
        root_pos, root_vel, _ = root_trajectory(posed, fps)
        root_speed = np.linalg.norm(root_vel[:, [0, 2]], axis=1)
        items.append(([frame.convert("RGB") for frame in frames], fps, prompt, root_pos, root_speed))

    fps = items[0][1]
    n_frames = min(len(item[0]) for item in items)
    width, height, cell = 1920, 900, 620
    cell_x = (250, 1050)
    cell_y = 145
    title_font, prompt_font, metric_font = font(38, True), font(34, True), font(24)

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
        centered(draw, width // 2, 32, args.title, title_font)
        for x, (frames, _, prompt, root_pos, root_speed) in zip(cell_x, items):
            centered(draw, x + cell // 2, 96, f'prompt: "{prompt}"', prompt_font, (35, 95, 185))
            canvas.paste(frames[frame_index].resize((cell, cell), Image.Resampling.NEAREST), (x, cell_y))
            draw.rectangle((x, cell_y, x + cell, cell_y + cell), outline=(190, 190, 190), width=2)
            distance = np.linalg.norm(root_pos[frame_index, [0, 2]] - root_pos[0, [0, 2]])
            metric = f"root speed {root_speed[frame_index]:.2f} m/s    displacement {distance:.2f} m"
            centered(draw, x + cell // 2, cell_y + cell + 18, metric, metric_font, (70, 70, 70))
        draw.text((width - 180, height - 40), f"frame {frame_index:03d}", font=metric_font, fill=(110, 110, 110))
        process.stdin.write(canvas.tobytes())

    process.stdin.close()
    if process.wait() != 0:
        raise SystemExit("ffmpeg failed")


if __name__ == "__main__":
    main()
