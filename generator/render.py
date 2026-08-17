"""Renderer: projected joints + depth -> 128x128 RGBA. Vector at 4x supersample, box down.

Chains are depth-sorted (far first). Each chain gets a whole-chain halo pass, then a stroke
pass, so shared joints inside a chain are never nicked. Hands = mitten (palm + 3 fingers).
"""
from __future__ import annotations
import math
from PIL import Image, ImageDraw
from .skeleton import Body, PARENT, NAMES, LIMB

SS = 4
SIZE = 128
W = SIZE * SS

INK = (40, 40, 40)
# colour per bone, keyed by CHILD joint (cskel convention). Unlisted bones = ink.
PALETTE = {
    "LeftForeArm": (232, 64, 48), "LeftHand": (255, 150, 40), "LeftHandEnd": (255, 150, 40),
    "RightForeArm": (40, 110, 230), "RightHand": (80, 200, 240), "RightHandEnd": (80, 200, 240),
    "LeftLeg": (200, 50, 160), "LeftFoot": (255, 120, 200), "LeftToeBase": (255, 120, 200),
    "RightLeg": (30, 150, 90), "RightFoot": (120, 220, 90), "RightToeBase": (120, 220, 90),
}
# draw chains (depth-sorted per frame, far first). Bones named by child joint.
CHAINS = [
    ["Spine", "Spine1", "Spine2", "Spine3", "Neck", "LeftShoulder", "LeftArm", "RightShoulder", "RightArm",
     "LeftUpLeg", "RightUpLeg"],                                   # torso + clavicles + pelvis
    ["LeftForeArm", "LeftHand", "LeftHandEnd"], ["RightForeArm", "RightHand", "RightHandEnd"],
    ["LeftLeg", "LeftFoot", "LeftToeBase"], ["RightLeg", "RightFoot", "RightToeBase"],
]
HEAD_DEPTH_JOINTS = ("Neck", "Head")


def _circ(d, p, r, col):
    d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=col)


def _seg(d, a, b, w, col):
    d.line([a, b], fill=col, width=max(1, int(round(w))))
    _circ(d, a, w / 2, col)
    _circ(d, b, w / 2, col)


from .skeleton import THUMB_LEN


def _mitten(d, wrist, tip, w, col, lat=None):
    """palm blob + 3 fingers. `lat` = projected palm-lateral vector (screen px, already
    normalised so |lat| in [0,1]): finger spread axis. Palm facing camera -> full fan;
    edge-on -> fingers stack. None -> spread perpendicular in screen space (2D fallback)."""
    ux, uy = tip[0] - wrist[0], tip[1] - wrist[1]
    L = math.hypot(ux, uy)
    if L < 1e-3:
        _circ(d, wrist, w * 0.7, col)
        return
    ux, uy = ux / L, uy / L
    if lat is None:
        px, py = -uy, ux
    else:
        px, py = lat
    palm = (wrist[0] + ux * L * 0.35, wrist[1] + uy * L * 0.35)
    _circ(d, palm, w * 0.65, col)
    fw = max(2 * SS, w * 0.5)
    for f in (-1, 0, 1):
        base = (palm[0] + px * f * w * 0.45, palm[1] + py * f * w * 0.45)
        end = (tip[0] + px * f * w * 0.75, tip[1] + py * f * w * 0.75)
        d.line([base, end], fill=col, width=int(fw))
        _circ(d, end, fw / 2, col)


def _mitten_shape(d, wrist, tip, w, col, lat=None):
    """filled mitten: capsule palm along wrist->tip whose width follows palm facing (|lat|),
    plus a thumb nub on the lateral side. Reads cleanest at 128px."""
    ux, uy = tip[0] - wrist[0], tip[1] - wrist[1]
    L = math.hypot(ux, uy)
    if L < 1e-3:
        _circ(d, wrist, w * 0.8, col)
        return
    ux, uy = ux / L, uy / L
    if lat is None:
        px, py, m = -uy, ux, 1.0
    else:
        m = math.hypot(lat[0], lat[1])
        px, py = ((lat[0] / m, lat[1] / m) if m > 1e-3 else (-uy, ux))
    pw = w * (0.55 + 0.55 * m)
    a = (wrist[0] + ux * L * 0.15, wrist[1] + uy * L * 0.15)
    b = (wrist[0] + ux * L * 0.9, wrist[1] + uy * L * 0.9)
    d.line([a, b], fill=col, width=max(1, int(pw)))
    _circ(d, a, pw / 2, col)
    _circ(d, b, pw / 2, col)
    tb = (wrist[0] + ux * L * 0.45 + px * pw * 0.55 * m, wrist[1] + uy * L * 0.45 + py * pw * 0.55 * m)
    _circ(d, tb, w * 0.42, col)


HANDS = {"mitten": _mitten_shape, "fingers": _mitten}


def _foot_segment(d, ankle, toe, w, col):
    _seg(d, ankle, toe, w, col)


def _foot_shoe(d, ankle, toe, w, col):
    """capsule from a heel point behind the ankle to the toe, 1.25x the stroke."""
    ux, uy = toe[0] - ankle[0], toe[1] - ankle[1]
    L = math.hypot(ux, uy)
    if L < 1e-3:
        _circ(d, ankle, w * 0.7, col)
        return
    ux, uy = ux / L, uy / L
    heel = (ankle[0] - ux * w * 0.35, ankle[1] - uy * w * 0.35)
    sw = w * 1.25
    d.line([heel, toe], fill=col, width=max(1, int(sw)))
    _circ(d, heel, sw / 2, col)
    _circ(d, toe, sw / 2, col)


FEET = {"shoe": _foot_shoe, "segment": _foot_segment}


def render(j2: dict, depth: dict, body: Body, colored: bool = True, halo: bool = True,
           bg=(0, 0, 0, 0), halo_rgb=(255, 255, 255)) -> Image.Image:
    im = Image.new("RGBA", (W, W), bg)
    d = ImageDraw.Draw(im)
    j = {k: (x * SS, y * SS) for k, (x, y) in j2.items()}
    w = body.stroke * SS
    hw = w + 2 * SS
    halo_col = (*halo_rgb, 255)

    def chain_depth(chain):
        ns = {n for c in chain for n in (c, PARENT[c])}
        return sum(depth[n] for n in ns) / len(ns)

    items = [(chain_depth(c), c) for c in CHAINS]
    items.append((sum(depth[n] for n in HEAD_DEPTH_JOINTS) / 2, ["Head"]))
    items.sort(key=lambda t: t[0])           # far first
    for _, chain in items:
        for pass_halo in ((True, False) if halo else (False,)):
            for child in chain:
                if child == "Head":
                    r = body.head_r * SS
                    if pass_halo:
                        _circ(d, j["Head"], r + 2 * SS, halo_col)
                    else:
                        _circ(d, j["Head"], r, (*INK, 255))
                    continue
                a, b = j[PARENT[child]], j[child]
                col = halo_col if pass_halo else (*(PALETTE.get(child, INK) if colored else INK), 255)
                ww = hw if pass_halo else w
                if child.endswith("HandEnd"):
                    thumb = child.replace("HandEnd", "HandThumb1")
                    lat = None
                    if thumb in j:   # projected thumb bone, normalised: |lat|=1 when palm faces camera
                        tl = THUMB_LEN * body.px_per_m * SS
                        lat = ((j[thumb][0] - a[0]) / tl, (j[thumb][1] - a[1]) / tl)
                    HANDS[body.hand_style](d, a, b, ww, col, lat)
                elif child.endswith("ToeBase"):
                    FEET[body.foot_style](d, a, b, ww, col)
                else:
                    _seg(d, a, b, ww, col)
    return im.resize((SIZE, SIZE), Image.BOX)
