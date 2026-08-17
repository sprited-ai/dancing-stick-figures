"""Preview tool: GIFs + contact sheets. python -m generator.preview"""
from __future__ import annotations
import math, os, sys
from PIL import Image
from .skeleton import Body, Camera, fk3d, project
from .render import render, SIZE
from .motion import pose, MotionParams, STYLES

OUT = "out"


def frame(style, t, cam, body=None, mp=None, colored=True, bg=(255, 255, 255, 255)):
    body = body or Body()
    j3 = fk3d(pose(style, t, mp), body)
    j2, depth = project(j3, cam)
    return render(j2, depth, body, colored=colored, bg=bg)


def gif(frames, path, ms=50, scale=3):
    fr = [f.resize((f.width * scale, f.height * scale), Image.NEAREST).convert("P", palette=Image.ADAPTIVE) for f in frames]
    fr[0].save(path, save_all=True, append_images=fr[1:], duration=ms, loop=0, disposal=2)


def strip(frames, cols=None):
    cols = cols or len(frames)
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGBA", (SIZE * cols, SIZE * rows), (255, 255, 255, 255))
    for i, f in enumerate(frames):
        sheet.paste(f, ((i % cols) * SIZE, (i // cols) * SIZE))
    return sheet


def main():
    os.makedirs(OUT, exist_ok=True)
    N = 32
    views = {"side": 90, "three_q": 50, "front": 0}
    # 1) each style, 3 views side by side, one GIF per style
    for style in STYLES:
        frames = []
        for i in range(N):
            t = i / N
            row = [frame(style, t, Camera(yaw=math.radians(y))) for y in views.values()]
            frames.append(strip(row))
        gif(frames, f"{OUT}/{style}.gif", ms=60)
        strip([frame(style, i / 8, Camera(yaw=math.radians(90))) for i in range(8)]).save(f"{OUT}/{style}_sheet.png")
    # 2) hero: walk while the camera orbits 360 over 4 cycles
    frames = []
    for i in range(N * 4):
        t = (i / N) % 1.0
        yaw = math.radians(90 + 360 * i / (N * 4))
        frames.append(frame("walk", t, Camera(yaw=yaw)))
    gif(frames, f"{OUT}/walk_orbit.gif", ms=50)
    # 3) all styles grid at 3/4 view
    frames = []
    for i in range(N):
        frames.append(strip([frame(s, i / N, Camera(yaw=math.radians(50))) for s in STYLES]))
    gif(frames, f"{OUT}/all_styles.gif", ms=60)
    print("wrote", os.listdir(OUT))


if __name__ == "__main__":
    main()
