"""Renderer: projected joints + depth -> 128x128 RGBA. Vector at 4x supersample, box down.

Chains are depth-sorted (far first). Each chain gets a whole-chain halo pass, then a stroke
pass, so shared joints inside a chain are never nicked. Hands = mitten (palm + 3 fingers).
"""
from __future__ import annotations
import math
from PIL import Image, ImageDraw
from .skeleton import Body, CHAINS, SEGMENTS

SS = 4
SIZE = 128
W = SIZE * SS

PALETTE = {
    "torso": (40, 40, 40), "head": (40, 40, 40),
    "arm_L": (232, 64, 48), "fore_L": (255, 150, 40), "hand_L": (255, 150, 40),
    "arm_R": (40, 110, 230), "fore_R": (80, 200, 240), "hand_R": (80, 200, 240),
    "leg_L": (200, 50, 160), "shin_L": (255, 120, 200), "foot_L": (255, 120, 200),
    "leg_R": (30, 150, 90), "shin_R": (120, 220, 90), "foot_R": (120, 220, 90),
}
INK = (40, 40, 40)


def _circ(d, p, r, col):
    d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=col)


def _seg(d, a, b, w, col):
    d.line([a, b], fill=col, width=max(1, int(round(w))))
    _circ(d, a, w / 2, col)
    _circ(d, b, w / 2, col)


def _mitten(d, wrist, tip, w, col):
    ux, uy = tip[0] - wrist[0], tip[1] - wrist[1]
    L = math.hypot(ux, uy)
    if L < 1e-3:                       # foreshortened to a point: just a palm blob
        _circ(d, wrist, w * 0.7, col)
        return
    ux, uy = ux / L, uy / L
    px, py = -uy, ux
    palm = (wrist[0] + ux * L * 0.35, wrist[1] + uy * L * 0.35)
    _circ(d, palm, w * 0.65, col)
    fw = max(2 * SS, w * 0.5)
    for f in (-1, 0, 1):
        base = (palm[0] + px * f * w * 0.45, palm[1] + py * f * w * 0.45)
        end = (tip[0] + px * f * w * 0.75, tip[1] + py * f * w * 0.75)
        d.line([base, end], fill=col, width=int(fw))
        _circ(d, end, fw / 2, col)


def render(j2: dict, depth: dict, body: Body, colored: bool = True, halo: bool = True,
           bg=(0, 0, 0, 0), halo_rgb=(255, 255, 255)) -> Image.Image:
    im = Image.new("RGBA", (W, W), bg)
    d = ImageDraw.Draw(im)
    j = {k: (x * SS, y * SS) for k, (x, y) in j2.items()}
    w = body.stroke * SS
    hw = w + 2 * SS
    halo_col = (*halo_rgb, 255)

    def chain_depth(chain):
        ns = {n for s in chain for n in SEGMENTS[s]}
        return sum(depth[n] for n in ns) / len(ns)

    items = [(chain_depth(c), c) for c in CHAINS]
    items.append((sum(depth[n] for n in ("neck", "head")) / 2, ["head"]))
    items.sort(key=lambda t: t[0])           # far first
    for _, chain in items:
        for pass_halo in ((True, False) if halo else (False,)):
            for seg in chain:
                if seg == "head":
                    r = body.head_r * SS
                    if pass_halo:
                        _circ(d, j["head"], r + 2 * SS, halo_col)
                    else:
                        _circ(d, j["head"], r, (*INK, 255))
                    continue
                a, b = (j[n] for n in SEGMENTS[seg])
                col = halo_col if pass_halo else (*(PALETTE[seg] if colored else INK), 255)
                ww = hw if pass_halo else w
                if seg.startswith("hand"):
                    _mitten(d, a, b, ww, col)
                else:
                    _seg(d, a, b, ww, col)
    return im.resize((SIZE, SIZE), Image.BOX)
