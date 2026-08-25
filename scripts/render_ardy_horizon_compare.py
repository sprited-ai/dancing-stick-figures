#!/usr/bin/env python3
"""Compare the first ARDY generation window with its full rollout."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.ardy_adapter import load, render_clip


def font(size: int, bold: bool = False):
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def centered(draw, x, y, text, face, fill=(35, 35, 35)):
    box = draw.textbbox((0, 0), text, font=face)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=face, fill=fill)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rendered, fps, prompt = render_clip(str(args.input))
    frames = [frame.convert("RGB") for frame in rendered]
    posed, _, _ = load(str(args.input))
    relative = posed - posed[:, :1]
    pose_speed = np.linalg.norm(np.diff(relative, axis=0), axis=-1).mean(axis=1) * fps
    root_speed = np.linalg.norm(np.diff(posed[:, 0, [0, 2]], axis=0), axis=-1) * fps

    window = int(round(2 * fps))
    width, height, cell = 1600, 860, 620
    xs, y = (120, 860), 145
    title_face, panel_face, text_face = font(36, True), font(29, True), font(22)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None

    for index, full_frame in enumerate(frames):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        centered(draw, width // 2, 24, f'ARDY: "{prompt}"', title_face)
        centered(draw, xs[0] + cell // 2, 94, "first generation window (0-2 s)", panel_face, (45, 105, 190))
        centered(draw, xs[1] + cell // 2, 94, "complete requested rollout (0-6 s)", panel_face, (45, 105, 190))

        short_index = min(index, window - 1)
        short_frame = frames[short_index]
        if index >= window:
            short_frame = Image.blend(short_frame, Image.new("RGB", short_frame.size, "white"), 0.35)
        for x, frame in zip(xs, (short_frame, full_frame)):
            canvas.paste(frame.resize((cell, cell), Image.Resampling.NEAREST), (x, y))
            draw.rectangle((x, y, x + cell, y + cell), outline=(190, 190, 190), width=2)

        if index >= window:
            centered(draw, xs[0] + cell // 2, y + cell // 2 - 20, "2-second clip ended", panel_face, (130, 45, 45))
        speed_index = min(index, len(pose_speed) - 1)
        centered(
            draw, xs[1] + cell // 2, y + cell + 18,
            f"pose speed {pose_speed[speed_index]:.2f} m/s   root speed {root_speed[speed_index]:.2f} m/s",
            text_face, (70, 70, 70),
        )
        elapsed = index / fps
        draw.rectangle((120, 815, 1480, 826), fill=(225, 225, 225))
        draw.rectangle((120, 815, 120 + int(1360 * elapsed / 6), 826), fill=(45, 105, 190))
        for second in (0, 2, 4, 6):
            x = 120 + int(1360 * second / 6)
            draw.line((x, 807, x, 834), fill=(80, 80, 80), width=2)
            centered(draw, x, 836, f"{second}s", text_face)
        process.stdin.write(canvas.tobytes())

    process.stdin.close()
    if process.wait() != 0:
        raise SystemExit("ffmpeg failed")


if __name__ == "__main__":
    main()
