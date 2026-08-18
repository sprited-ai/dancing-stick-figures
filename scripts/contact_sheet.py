"""Dataset contact sheet + label row from data/v1 parquet.  python scripts/contact_sheet.py data/v1 out/"""
import pyarrow.parquet as pq, glob, io, random, json, sys, os, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "."); from generator.skeleton import PARENT, NAMES
data, out = sys.argv[1], sys.argv[2]; random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 5)
files = sorted(glob.glob(f"{data}/train-*.parquet")); picks = random.sample(files, 16)
W = 128; pad = 6; lab = 58; cols = 8; items = []
import textwrap
for f in picks:
    t = pq.read_table(f)
    for i in random.sample(range(t.num_rows), 2): items.append((t, i))
def dec(t, col, i):
    v = t.column(col)[i].as_py(); b = v["bytes"] if isinstance(v, dict) else v
    return Image.open(io.BytesIO(b))
def over_white(im):
    bg = Image.new("RGBA", im.size, "white"); bg.alpha_composite(im.convert("RGBA")); return bg.convert("RGB")
sheet = Image.new("RGB", (cols * (W + pad) + pad, 4 * (W + lab + pad) + pad), "white"); d = ImageDraw.Draw(sheet)
for k, (t, i) in enumerate(items):
    r, c = divmod(k, cols); x = pad + c * (W + pad); y = pad + r * (W + lab + pad)
    sheet.paste(over_white(dec(t, "color", i)).resize((W, W)), (x, y))
    g = t.column("group")[i].as_py(); yaw = t.column("cam_yaw")[i].as_py(); txt = t.column("text")[i].as_py()
    d.text((x, y + W + 2), f"{g}  ·  yaw {np.degrees(yaw):.0f}°", fill="black")
    for li, line in enumerate(textwrap.wrap(txt, 24)[:4]):          # full prompt, wrapped, no truncation
        d.text((x, y + W + 14 + 11 * li), line, fill=(90, 90, 90))
sheet.save(os.path.join(out, "dataset_contact_sheet.png"))
S = 3; t, i = items[3]
col = over_white(dec(t, "color", i)).resize((W * S, W * S), Image.NEAREST)
xy = np.frombuffer(t.column("joint_xy")[i].as_py(), np.float32).reshape(27, 2) * 128   # normalised [0,1] -> px
vis = np.frombuffer(t.column("joint_visible")[i].as_py(), np.uint8)
ov = col.copy(); dd = ImageDraw.Draw(ov)
for j, n in enumerate(NAMES):
    p = NAMES.index(PARENT[n]) if PARENT.get(n) else -1
    if p >= 0:
        dd.line([tuple(xy[j] * S), tuple(xy[p] * S)], fill=(255, 255, 255), width=3); dd.line([tuple(xy[j] * S), tuple(xy[p] * S)], fill=(0, 0, 0), width=1)
for j in range(27):
    x_, y_ = xy[j] * S; c = (0, 200, 0) if vis[j] else (220, 0, 0); dd.ellipse([x_ - 3, y_ - 3, x_ + 3, y_ + 3], fill=c)
imgs = [col] + [dec(t, k, i).convert("RGB").resize((W * S, W * S), Image.NEAREST) for k in ("seg", "depth", "normal")] + [ov]
names = ["color (RGBA over white)", "seg (joint id)", "depth16", "normal (camera space)", "joint_xy overlay  green=visible  red=occluded"]
row = Image.new("RGB", (5 * (W * S + pad) + pad, W * S + pad * 2 + 16), "white"); dr = ImageDraw.Draw(row)
for k, (im_, n) in enumerate(zip(imgs, names)): row.paste(im_, (pad + k * (W * S + pad), 16)); dr.text((pad + k * (W * S + pad), 2), n, fill="black")
row.save(os.path.join(out, "dataset_labels_row.png"))
rec = {k: t.column(k)[i].as_py() for k in ("clip_id", "frame_idx", "group", "text", "cam_yaw", "cam_pitch", "px_per_m", "bone_scale", "root_pos", "root_heading", "qa_flags")}
print(json.dumps(rec, default=str)); print("joint_xyz[:3]", np.frombuffer(t.column("joint_xyz")[i].as_py(), np.float32).reshape(27, 3)[:3].round(3).tolist(), "visible", int(vis.sum()), "/27")
