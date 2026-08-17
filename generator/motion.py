"""Motion library: hand-authored cyclic keyframes -> Pose(t).

pose(style, t, params) is THE seam between motion source and renderer. Week 2 swaps in
ARDY behind this same signature.

Keys are dicts of channel -> value (degrees / px). Channels:
  swing.<seg>, abd.<seg>            absolute segment angles (deg)
  root.x root.y root.z              pelvis offset (px)
  lean twist head_tilt (deg), squash (unitless)
Interpolation is cyclic Catmull-Rom per channel. Overlapping action: each limb segment's
channels are sampled at t - lag * chain_depth.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from .skeleton import Pose, CHAIN_DEPTH, LIMB_SEGS


@dataclass
class MotionParams:
    lag: float = 0.03           # cycle fraction per chain depth
    amp: float = 1.0            # global amplitude scale on swing/abd
    bounce: float = 1.0         # root.y multiplier
    squash_gain: float = 1.0
    stepped: bool = False       # robot: hold keys, no interpolation


# ----------------------------------------------------------------- interpolation
def _catmull_rom_cyclic(ts, vs, t):
    n = len(ts)
    t = t % 1.0
    # find segment i with ts[i] <= t < ts[i+1] (cyclic)
    i = max(k for k in range(n) if ts[k] <= t) if any(ts[k] <= t for k in range(n)) else n - 1
    i0, i1, i2, i3 = (i - 1) % n, i, (i + 1) % n, (i + 2) % n
    t1, t2 = ts[i1], ts[i2] if i2 > i1 else ts[i2] + 1.0
    if t < t1: t += 1.0
    u = (t - t1) / (t2 - t1) if t2 > t1 else 0.0
    p0, p1, p2, p3 = vs[i0], vs[i1], vs[i2], vs[i3]
    u2, u3 = u * u, u * u * u
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)


def _step(ts, vs, t):
    t = t % 1.0
    i = max(k for k in range(len(ts)) if ts[k] <= t)
    return vs[i]


class Clip:
    def __init__(self, keys: list[tuple[float, dict]]):
        keys = sorted(keys, key=lambda k: k[0])
        self.ts = [k[0] for k in keys]
        chans = set()
        for _, d in keys: chans |= set(d)
        self.chans = {c: [d.get(c, 0.0) for _, d in keys] for c in chans}

    def sample(self, chan: str, t: float, stepped=False) -> float:
        if chan not in self.chans: return 0.0
        f = _step if stepped else _catmull_rom_cyclic
        return f(self.ts, self.chans[chan], t)


# ----------------------------------------------------------------- style authoring helpers
def _leg(side, thigh, knee, dorsi=0.0, abd=3.0):
    """thigh: swing deg. knee: relative bend (negative = shin swings back). dorsi: toe up."""
    shin = thigh + knee
    return {f"swing.leg_{side}": thigh, f"swing.shin_{side}": shin, f"swing.foot_{side}": shin + 90 + dorsi,
            f"abd.leg_{side}": abd, f"abd.shin_{side}": abd, f"abd.foot_{side}": abd}


def _arm(side, upper, elbow, wrist=5.0, abd=8.0, abd_fore=None, abd_hand=None):
    fore = upper + elbow
    af = abd if abd_fore is None else abd_fore
    ah = af if abd_hand is None else abd_hand
    return {f"swing.arm_{side}": upper, f"swing.fore_{side}": fore, f"swing.hand_{side}": fore + wrist,
            f"abd.arm_{side}": abd, f"abd.fore_{side}": af, f"abd.hand_{side}": ah}


def _shift(keys, dt):
    return [((t + dt) % 1.0, d) for t, d in keys]


def _merge(*keysets):
    """merge keysets that share the same t grid."""
    out = {}
    for ks in keysets:
        for t, d in ks:
            out.setdefault(round(t, 6), {}).update(d)
    return sorted(out.items())


# ----------------------------------------------------------------- styles
def _walk():
    T = [i / 8 for i in range(8)]
    thigh = [25, 15, 0, -15, -25, -20, 0, 20]
    knee = [0, -10, -5, 0, -5, -55, -60, -25]
    dorsi = [15, 0, 0, -10, -30, -15, 0, 10]
    upper = [-20, -12, 0, 12, 20, 15, 0, -15]
    elbow = [30, 25, 22, 28, 38, 42, 40, 35]
    legL = [(t, _leg("L", th, kn, do)) for t, th, kn, do in zip(T, thigh, knee, dorsi)]
    armL = [(t, _arm("L", up, el)) for t, up, el in zip(T, upper, elbow)]
    body = [(t, {"root.y": [-1, 0.5, 2, 0.5, -1, 0.5, 2, 0.5][i], "lean": 4,
                 "twist": -8 * math.cos(2 * math.pi * t), "head_tilt": 2}) for i, t in enumerate(T)]
    return _merge(legL, _mirror_LR(_shift(legL, 0.5)), armL, _mirror_LR(_shift(armL, 0.5)), body)


def _mirror_LR(keys):
    return [(t, {k.replace("_L", "_R"): v for k, v in d.items()}) for t, d in keys]


def _jumping_jack():
    # feet together / arms down -> airborne spreading -> feet apart / arms up -> airborne closing
    T = [0, .25, .5, .75]
    ks = []
    for i, t in enumerate(T):
        leg_abd = [2, 14, 26, 14][i]
        arm_abd = [10, 90, 168, 90][i]
        knee = [-4, 0, -6, 0][i]
        rooty = [-1.5, 6, -1.5, 6][i]
        squash = [0.10, 0, 0.10, 0][i]
        d = {}
        for s in "LR":
            d.update(_leg(s, 0, knee, 0, abd=leg_abd))
            d.update(_arm(s, 0, 8, wrist=10, abd=arm_abd, abd_fore=arm_abd + 6, abd_hand=arm_abd + 10))
        d.update({"root.y": rooty, "squash": squash, "lean": 0, "head_tilt": 0})
        ks.append((t, d))
    return ks


def _wave():
    T = [i / 8 for i in range(8)]
    ks = []
    for i, t in enumerate(T):
        d = {}
        d.update(_leg("L", 3, -4, 0)); d.update(_leg("R", -3, -4, 0))
        d.update(_arm("L", -6, 12))
        # R arm raised: upper out+forward, forearm up, hand waves
        s = math.sin(2 * math.pi * 2 * t)  # two waves per cycle
        d.update({"swing.arm_R": 30, "abd.arm_R": 55, "swing.fore_R": 175 + 25 * s, "abd.fore_R": 25 - 12 * s,
                  "swing.hand_R": 175 + 25 * s, "abd.hand_R": 25 - 12 * s + 20 * math.cos(2 * math.pi * 2 * t)})
        d.update({"root.y": 0.6 * math.sin(2 * math.pi * 2 * t), "lean": -2, "twist": 6, "head_tilt": -3 + 3 * s})
        ks.append((t, d))
    return ks


def _idle_bob():
    T = [i / 4 for i in range(4)]
    ks = []
    for i, t in enumerate(T):
        s = math.sin(2 * math.pi * t)
        d = {}
        d.update(_leg("L", 2, -3 - 3 * (i % 2), 0)); d.update(_leg("R", -2, -3 - 3 * (i % 2), 0))
        d.update(_arm("L", -4 + 3 * s, 10)); d.update(_arm("R", -4 - 3 * s, 10))
        d.update({"root.y": -1.2 * (i % 2), "lean": 2, "twist": 3 * s, "head_tilt": 1 * s})
        ks.append((t, d))
    return ks


STYLES = {
    "idle_bob": _idle_bob,
    "walk": _walk,
    "jumping_jack": _jumping_jack,
    "wave": _wave,
}
_CACHE: dict[str, Clip] = {}


def clip(style: str) -> Clip:
    if style not in _CACHE:
        _CACHE[style] = Clip(STYLES[style]())
    return _CACHE[style]


def pose(style: str, t: float, mp: MotionParams | None = None) -> Pose:
    """t in [0,1) cyclic -> Pose. THE seam."""
    mp = mp or MotionParams()
    c = clip(style)
    p = Pose()
    for seg in LIMB_SEGS:
        tt = t - mp.lag * CHAIN_DEPTH[seg]
        p.swing[seg] = math.radians(c.sample(f"swing.{seg}", tt, mp.stepped) * mp.amp)
        p.abd[seg] = math.radians(c.sample(f"abd.{seg}", tt, mp.stepped) * mp.amp)
    p.root = (c.sample("root.x", t, mp.stepped), c.sample("root.y", t, mp.stepped) * mp.bounce, c.sample("root.z", t, mp.stepped))
    p.lean = math.radians(c.sample("lean", t, mp.stepped))
    p.bend = math.radians(c.sample("bend", t, mp.stepped))
    p.twist = math.radians(c.sample("twist", t, mp.stepped))
    p.head_tilt = math.radians(c.sample("head_tilt", t, mp.stepped))
    p.squash = c.sample("squash", t, mp.stepped) * mp.squash_gain
    return p
