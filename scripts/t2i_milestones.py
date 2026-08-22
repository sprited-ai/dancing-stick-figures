"""Build a labelled first-four-sample progression sheet from T2I PNGs."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("out")
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    run = Path(args.run)
    cells = []
    for image_path in sorted(run.glob("sample_*.png")):
        if image_path.name.startswith("sample_raw_"):
            continue
        match = re.search(r"(\d+)", image_path.stem)
        if not match:
            continue
        step = int(match.group(1))
        manifest_path = run / f"sample_manifest_{step:06d}.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        prompts = manifest.get("prompts") or []
        count = max(1, len(prompts))
        grid_cols = int(math.ceil(math.sqrt(count)))
        image = Image.open(image_path).convert("RGB")
        cell = image.width // grid_cols
        tiles = []
        for index in range(min(args.samples, count)):
            x, y = (index % grid_cols) * cell, (index // grid_cols) * cell
            tiles.append(image.crop((x, y, x + cell, y + cell)))
        cells.append((step, prompts[: len(tiles)], tiles))
    if not cells:
        raise SystemExit("no matched sample PNG + manifest pairs")

    rows = max(len(row[2]) for row in cells)
    cell = cells[0][2][0].width
    label_width, header = 180, 24
    sheet = Image.new("RGB", (label_width + len(cells) * cell, header + rows * cell), "white")
    draw = ImageDraw.Draw(sheet)
    for column, (step, prompts, tiles) in enumerate(cells):
        draw.text((label_width + column * cell + 4, 5), f"step {step}", fill=(60, 60, 60))
        for row, tile in enumerate(tiles):
            sheet.paste(tile, (label_width + column * cell, header + row * cell))
            if column == 0:
                prompt = prompts[row] if row < len(prompts) else f"sample {row}"
                if len(prompt) > 27:
                    prompt = prompt[:24] + "..."
                draw.text((4, header + row * cell + 5), prompt, fill=(20, 20, 20))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(f"wrote {output}: {len(cells)} milestones x {rows} samples")


if __name__ == "__main__":
    main()

