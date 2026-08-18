"""Evolution strip: for each runs/<run>/sample_*.gif take the middle frame of samples 0..K-1 -> rows=samples, cols=steps.
    python scripts/evolution.py runs/a0 out/a0_evolution.png [K=4] [COLS=4]
"""
import sys, glob, os, re
from PIL import Image, ImageDraw
run, out = sys.argv[1], sys.argv[2]; K = int(sys.argv[3]) if len(sys.argv) > 3 else 4
COLS = int(sys.argv[4]) if len(sys.argv) > 4 else 4   # grid columns in the sample gif (16 samples -> 4)
gifs = sorted(glob.glob(os.path.join(run, "sample_*.gif")), key=lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[0]))
cells = []
for g in gifs:
    im = Image.open(g); T = im.n_frames; im.seek(T // 2); fr = im.convert("RGB")
    cols = COLS; cell = fr.width // cols
    tiles = [fr.crop(((i % cols) * cell, (i // cols) * cell, (i % cols + 1) * cell, (i // cols + 1) * cell)) for i in range(K)]
    cells.append((int(re.findall(r"(\d+)", os.path.basename(g))[0]), tiles))
if not cells: sys.exit("no gifs")
cell = cells[0][1][0].width
W = cell * len(cells); H = cell * K + 24
sheet = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(sheet)
for x, (step, tiles) in enumerate(cells):
    d.text((x * cell + 6, 4), f"step {step}", fill=(80, 80, 80))
    for y, t in enumerate(tiles): sheet.paste(t, (x * cell, 24 + y * cell))
sheet.save(out); print("wrote", out, len(cells), "steps")
