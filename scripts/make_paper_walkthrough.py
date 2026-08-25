"""Build a Korean narrated walkthrough from the final paper PDF.

The narration lives in paper/WALKTHROUGH_KO.md. Each scene first shows the
whole page with a gold rectangle, then fades to a readable crop of that region.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


SCENES = [
    # page, normalized x/y/w/h
    # The title remains visible in the overview; the detail crop isolates the abstract.
    (1, (0.04, 0.15, 0.92, 0.49)),
    (1, (0.04, 0.39, 0.92, 0.56)),
    (2, (0.04, 0.03, 0.92, 0.92)),
    (3, (0.04, 0.03, 0.92, 0.62)),
    (4, (0.04, 0.03, 0.92, 0.39)),
    (4, (0.04, 0.36, 0.92, 0.59)),
    (5, (0.04, 0.03, 0.92, 0.43)),
    (5, (0.04, 0.48, 0.92, 0.48)),
    (6, (0.04, 0.03, 0.92, 0.24)),
    (6, (0.04, 0.25, 0.92, 0.50)),
    (6, (0.50, 0.40, 0.46, 0.38)),
    (6, (0.50, 0.77, 0.46, 0.20)),
]

CANVAS = (1920, 1080)
GOLD = (255, 184, 55, 255)
BACKGROUND = (22, 24, 29, 255)


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def parse_narration(path: Path) -> list[tuple[str, str]]:
    text = path.read_text()
    blocks = re.split(r"(?m)^##\s+", text)[1:]
    scenes = []
    for block in blocks:
        title, _, body = block.partition("\n")
        body = re.sub(r"(?m)^#.*$", "", body)
        body = body.replace("**", "").replace(chr(96), "")
        body = re.sub(r"\s+", " ", body).strip()
        scenes.append((title.strip(), body))
    if len(scenes) != len(SCENES):
        raise ValueError(f"expected {len(SCENES)} narration scenes, found {len(scenes)}")
    if any("[" in body or "]" in body for _, body in scenes):
        raise ValueError("replace bracketed placeholders in the narration before rendering")
    return scenes


def page_box(size: tuple[int, int], canvas: tuple[int, int], margin: int = 38) -> tuple[int, int, int, int]:
    w, h = size
    scale = min((canvas[0] - 2 * margin) / w, (canvas[1] - 2 * margin) / h)
    nw, nh = round(w * scale), round(h * scale)
    return (canvas[0] - nw) // 2, (canvas[1] - nh) // 2, nw, nh


def normalized_rect(image: Image.Image, rect: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    return round(x * image.width), round(y * image.height), round((x + w) * image.width), round((y + h) * image.height)


def make_overview(page: Image.Image, rect: tuple[float, float, float, float], out: Path) -> None:
    canvas = Image.new("RGBA", CANVAS, BACKGROUND)
    x, y, w, h = page_box(page.size, CANVAS)
    resized = page.resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")
    canvas.alpha_composite(resized, (x, y))
    rx0, ry0, rx1, ry1 = normalized_rect(page, rect)
    sx, sy = w / page.width, h / page.height
    box = (x + round(rx0 * sx), y + round(ry0 * sy), x + round(rx1 * sx), y + round(ry1 * sy))
    ImageDraw.Draw(canvas).rounded_rectangle(box, radius=10, outline=GOLD, width=7)
    canvas.convert("RGB").save(out, quality=95)


def expand_to_aspect(
    box: tuple[int, int, int, int], image_size: tuple[int, int], aspect: float = 16 / 9, pad: float = 0.10
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    w, h = min(w, image_size[0]), min(h, image_size[1])
    x0 = min(max(cx - w / 2, 0), image_size[0] - w)
    y0 = min(max(cy - h / 2, 0), image_size[1] - h)
    return round(x0), round(y0), round(x0 + w), round(y0 + h)


def make_detail(page: Image.Image, rect: tuple[float, float, float, float], out: Path) -> None:
    raw = normalized_rect(page, rect)
    rx0, ry0, rx1, ry1 = raw
    pad_x = round((rx1 - rx0) * 0.03)
    pad_y = round((ry1 - ry0) * 0.04)
    crop_box = (
        max(0, rx0 - pad_x),
        max(0, ry0 - pad_y),
        min(page.width, rx1 + pad_x),
        min(page.height, ry1 + pad_y),
    )
    crop = page.crop(crop_box).convert("RGBA")
    x0, y0, _, _ = crop_box
    box = (rx0 - x0, ry0 - y0, rx1 - x0, ry1 - y0)
    ImageDraw.Draw(crop).rounded_rectangle(box, radius=12, outline=GOLD, width=7)

    margin = 38
    scale = min((CANVAS[0] - 2 * margin) / crop.width, (CANVAS[1] - 2 * margin) / crop.height)
    fitted = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", CANVAS, BACKGROUND)
    x = (CANVAS[0] - fitted.width) // 2
    y = (CANVAS[1] - fitted.height) // 2
    canvas.alpha_composite(fitted, (x, y))
    canvas.convert("RGB").save(out, quality=95)


def audio_duration(path: Path) -> float:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        check=True,
        text=True,
        capture_output=True,
    )
    return float(p.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="output/pdf/paper.pdf")
    ap.add_argument("--script", default="paper/WALKTHROUGH_KO.md")
    ap.add_argument("--out", default="output/video/dsf_paper_walkthrough_ko.mp4")
    ap.add_argument("--voice", default="Yuna")
    ap.add_argument("--rate", default="178")
    ap.add_argument("--limit", type=int, default=0, help="render only the first N scenes for QA")
    ap.add_argument("--stills-only", action="store_true", help="render overview/detail frames without audio or video")
    args = ap.parse_args()

    pdf, narration_path, output = Path(args.pdf), Path(args.script), Path(args.out)
    scenes = parse_narration(narration_path)
    scene_specs = SCENES
    if args.limit:
        scenes = scenes[:args.limit]
        scene_specs = scene_specs[:args.limit]
    work = output.parent / "walkthrough_work"
    pages = work / "pages"
    work.mkdir(parents=True, exist_ok=True)
    pages.mkdir(exist_ok=True)

    run("pdftoppm", "-png", "-r", "160", str(pdf), str(pages / "page"))

    videos = []
    for i, ((page_number, rect), (title, narration)) in enumerate(zip(scene_specs, scenes), 1):
        page_path = pages / f"page-{page_number:02d}.png"
        if not page_path.exists():
            page_path = pages / f"page-{page_number}.png"
        page = Image.open(page_path).convert("RGB")
        overview, detail = work / f"{i:02d}_overview.jpg", work / f"{i:02d}_detail.jpg"
        audio, video = work / f"{i:02d}.aiff", work / f"{i:02d}.mp4"
        make_overview(page, rect, overview)
        make_detail(page, rect, detail)
        spoken = f"{title}. {narration}"
        if args.stills_only:
            continue
        run("say", "-v", args.voice, "-r", args.rate, "-o", str(audio), spoken)
        duration = max(audio_duration(audio) + 0.35, 4.0)
        run(
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-loop", "1", "-i", str(overview),
            "-loop", "1", "-i", str(detail),
            "-i", str(audio),
            "-filter_complex",
            "[0:v]fps=30,format=yuv420p[a];[1:v]fps=30,format=yuv420p[b];"
            "[a][b]xfade=transition=fade:duration=1:offset=2[v]",
            "-map", "[v]", "-map", "2:a",
            "-t", f"{duration:.3f}", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(video),
        )
        videos.append(video)

    if args.stills_only:
        print(work)
        return

    concat = work / "concat.txt"
    concat.write_text("".join(f"file '{v.resolve()}'\n" for v in videos))
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(output),
    )
    print(output)


if __name__ == "__main__":
    main()
