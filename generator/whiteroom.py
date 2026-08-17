"""White-room scene test: many ARDY clips placed on one floor with their real world trajectories.

    python -m generator.whiteroom --npz ardy_out/v1 --n 24 --out out/whiteroom.gif

Not a dataset output — a sanity/visual test for root tracking and motion quality. Orthographic
3/4 camera over a grid floor, ground shadows, figures depth-sorted. PIL polylines (fast).
"""
from __future__ import annotations
import argparse, glob, math, os, random
import numpy as np
from PIL import Image, ImageDraw
from .skeleton import NAMES, IDX, PARENT
from .render import PALETTE, INK
from .ardy_adapter import load, _frame0_basis

W, H = 960, 540
SS = 2


def world_to_fig0(P):
    """all clips into a common frame: each clip's own frame-0 basis (x left, y up, z fwd), Hips0 at origin."""
    R, up = _frame0_basis(P)
    hips = IDX["Hips"]
    Q = (P - P[0:1, hips:hips + 1, :]) @ R.T
    Q[:, :, 1] += (P[0, hips] @ up) - (P[0, hips] @ up)   # keep absolute height (feet on floor)
    floor = Q[0, [IDX["LeftToeBase"], IDX["RightToeBase"], IDX["LeftFoot"], IDX["RightFoot"]], 1].min()
    Q[:, :, 1] -= floor
    return Q


class Cam:
    def __init__(self, yaw=35.0, pitch=22.0, px_per_m=70.0, cx=W / 2, cy=H * 0.62):
        self.cy, self.sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        self.cp, self.sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        self.k, self.cx, self.cyy = px_per_m, cx, cy

    def proj(self, x, y, z):
        sx = z * self.sy + x * self.cy
        d = z * self.cy - x * self.sy
        y2 = y * self.cp + d * self.sp
        d2 = d * self.cp - y * self.sp
        return (self.cx + sx * self.k) * SS, (self.cyy - y2 * self.k) * SS, d2


def draw_floor(d, cam, ext=6, step=1.0):
    n = int(ext / step)
    for i in range(-n, n + 1):
        a = cam.proj(i * step, 0, -ext); b = cam.proj(i * step, 0, ext)
        d.line([a[:2], b[:2]], fill=(225, 225, 225), width=SS)
        a = cam.proj(-ext, 0, i * step); b = cam.proj(ext, 0, i * step)
        d.line([a[:2], b[:2]], fill=(225, 225, 225), width=SS)


def draw_figure(d, cam, J, stroke=5, head_r=0.125):
    bones = [(PARENT[n], n) for n in NAMES if PARENT[n] and n not in ("LeftHandThumb1", "RightHandThumb1")]
    items = []
    for p, c in bones:
        a = cam.proj(*J[IDX[p]]); b = cam.proj(*J[IDX[c]])
        items.append(((a[2] + b[2]) / 2, a[:2], b[:2], PALETTE.get(c, INK)))
    hx, hy, hd = cam.proj(*J[IDX["Head"]])
    # shadow
    sx, sy, _ = cam.proj(J[IDX["Hips"]][0], 0, J[IDX["Hips"]][2])
    r = 0.28 * cam.k * SS
    d.ellipse([sx - r, sy - r * cam.sp, sx + r, sy + r * cam.sp], fill=(210, 210, 210))
    for _, a, b, col in sorted(items, key=lambda t: t[0]):
        d.line([a, b], fill=col, width=stroke * SS)
        d.ellipse([a[0] - stroke * SS / 2, a[1] - stroke * SS / 2, a[0] + stroke * SS / 2, a[1] + stroke * SS / 2], fill=col)
        d.ellipse([b[0] - stroke * SS / 2, b[1] - stroke * SS / 2, b[0] + stroke * SS / 2, b[1] + stroke * SS / 2], fill=col)
    rr = head_r * cam.k * SS
    d.ellipse([hx - rr, hy - rr, hx + rr, hy + rr], fill=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="ardy_out/v1"); ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="out/whiteroom.gif"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", default="locomotion,dance,gesture,sport,transitions,idle")
    a = ap.parse_args()
    files = [f for f in glob.glob(os.path.join(a.npz, "*", "*.npz")) if os.path.basename(os.path.dirname(f)) in a.groups.split(",")]
    rng = random.Random(a.seed); rng.shuffle(files); files = files[:a.n]
    clips = []
    cols = int(math.ceil(math.sqrt(a.n)))
    for i, f in enumerate(files):
        P, fps, text = load(f)
        Q = world_to_fig0(P)
        # place on a grid, each figure yawed randomly so they don't all walk the same way
        gx = (i % cols - (cols - 1) / 2) * 2.2; gz = (i // cols - (cols - 1) / 2) * 2.2
        yaw = rng.uniform(-math.pi, math.pi)
        c, s = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        Q = Q @ Rz.T + np.array([gx, 0, gz])
        clips.append((Q, os.path.basename(f)[:40]))
    T = min(q.shape[0] for q, _ in clips)
    cam = Cam()
    frames = []
    for t in range(0, T, 2):
        im = Image.new("RGB", (W * SS, H * SS), (255, 255, 255))
        d = ImageDraw.Draw(im)
        draw_floor(d, cam)
        order = sorted(range(len(clips)), key=lambda i: cam.proj(*clips[i][0][t, IDX["Hips"]])[2])
        for i in order:
            draw_figure(d, cam, clips[i][0][t])
        frames.append(im.resize((W, H), Image.LANCZOS))
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=100, loop=0)
    frames[len(frames) // 2].save(a.out.replace(".gif", "_mid.png"))
    print("wrote", a.out, len(frames), "frames,", len(clips), "figures")
    for _, n in clips: print("  ", n)


if __name__ == "__main__":
    main()
