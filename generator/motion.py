"""Motion library: hand-authored cyclic keyframes -> Pose(t).

pose(style, t, params) is THE seam between motion source and renderer. Week 2 swaps in
ARDY behind this same signature.

Keys are dicts of channel -> value (degrees / px). Channels:
  swing.<seg>, abd.<seg>            absolute segment angles (deg)
  root.x root.y root.z              Hips offset (cm; converted to metres)
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


def _run():
    # higher amplitude, both feet airborne twice per cycle, strong arm pump, forward lean
    T = [i / 8 for i in range(8)]
    thigh = [40, 20, -5, -25, -40, -30, 0, 30]
    knee = [-15, -20, -10, -5, -20, -90, -100, -60]
    dorsi = [10, -10, -20, -30, -35, -10, 5, 10]
    upper = [-35, -20, 0, 20, 35, 25, 0, -20]
    elbow = [80, 85, 90, 95, 100, 95, 90, 85]
    legL = [(t, _leg("L", th, kn, do)) for t, th, kn, do in zip(T, thigh, knee, dorsi)]
    armL = [(t, _arm("L", up, el, wrist=15, abd=12)) for t, up, el in zip(T, upper, elbow)]
    body = [(t, {"root.y": [-2, 3, 5, 3, -2, 3, 5, 3][i], "lean": 12, "bend": 6,
                 "twist": -14 * math.cos(2 * math.pi * t), "head_tilt": 4}) for i, t in enumerate(T)]
    return _merge(legL, _mirror_LR(_shift(legL, 0.5)), armL, _mirror_LR(_shift(armL, 0.5)), body)


def _robot():
    # stepped: 8 held poses, elbows/knees at right angles, torso rigid, alternating twist
    T = [i / 8 for i in range(8)]
    ks = []
    for i, t in enumerate(T):
        d = {}
        aL = [0, 0, -30, -30, 0, 0, 30, 30][i]; aR = [30, 30, 0, 0, -30, -30, 0, 0][i]
        eL = [90, 90, 90, 90, 90, 90, 90, 90][i]
        fL_abd = [20, 60, 60, 20, 20, 60, 60, 20][i]; fR_abd = [60, 20, 20, 60, 60, 20, 20, 60][i]
        d.update(_arm("L", aL, eL, wrist=0, abd=10, abd_fore=fL_abd, abd_hand=fL_abd))
        d.update(_arm("R", aR, eL, wrist=0, abd=10, abd_fore=fR_abd, abd_hand=fR_abd))
        kn = [-10, -10, -30, -30, -10, -10, -30, -30][i]
        d.update(_leg("L", [5, 5, 0, 0, -5, -5, 0, 0][i], kn, 0, abd=6)); d.update(_leg("R", [-5, -5, 0, 0, 5, 5, 0, 0][i], kn, 0, abd=6))
        d.update({"root.y": [0, -1, 0, -1, 0, -1, 0, -1][i], "lean": 0, "twist": [15, 15, -15, -15, 15, 15, -15, -15][i],
                  "head_tilt": [0, 0, 8, 8, 0, 0, -8, -8][i]})
        ks.append((t, d))
    return ks


def _twist():
    # hips vs shoulders counter-rotate, arms bent at chest height swinging opposite to hips, knees soft
    T = [i / 8 for i in range(8)]
    ks = []
    for i, t in enumerate(T):
        s = math.sin(2 * math.pi * t)
        d = {}
        d.update(_arm("L", -10 + 25 * s, 95, wrist=5, abd=25, abd_fore=10, abd_hand=10))
        d.update(_arm("R", -10 - 25 * s, 95, wrist=5, abd=25, abd_fore=10, abd_hand=10))
        d.update(_leg("L", 5 * s, -25 - 10 * abs(s), 0, abd=8)); d.update(_leg("R", -5 * s, -25 - 10 * abs(s), 0, abd=8))
        d.update({"root.y": -1.5 * abs(s), "lean": 3, "twist": 40 * s, "head_tilt": 0})
        ks.append((t, d))
    return ks


def _disco_point():
    # alternating point up-and-across (Saturday Night Fever), hip pop on the beat
    T = [i / 8 for i in range(8)]
    ks = []
    for i, t in enumerate(T):
        # beat 0: R arm up-right, L hand on hip; beat 4: mirror
        up = math.cos(2 * math.pi * t)          # +1 -> R up, -1 -> L up
        d = {}
        rup = max(0, up); lup = max(0, -up)
        d.update({"swing.arm_R": 20 * rup, "abd.arm_R": 30 + 130 * rup, "swing.fore_R": 20 * rup, "abd.fore_R": 30 + 140 * rup,
                  "swing.hand_R": 20 * rup, "abd.hand_R": 30 + 150 * rup})
        d.update({"swing.arm_L": 20 * lup, "abd.arm_L": 30 + 130 * lup, "swing.fore_L": 20 * lup, "abd.fore_L": 30 + 140 * lup,
                  "swing.hand_L": 20 * lup, "abd.hand_L": 30 + 150 * lup})
        # the down arm rests on the hip: upper arm slightly out, forearm across
        if rup < lup:
            d.update({"swing.arm_R": -5, "abd.arm_R": 25, "swing.fore_R": 60, "abd.fore_R": -30, "swing.hand_R": 60, "abd.hand_R": -30})
        else:
            d.update({"swing.arm_L": -5, "abd.arm_L": 25, "swing.fore_L": 60, "abd.fore_L": -30, "swing.hand_L": 60, "abd.hand_L": -30})
        d.update(_leg("L", 5, -15 - 10 * lup, 0, abd=12)); d.update(_leg("R", 5, -15 - 10 * rup, 0, abd=12))
        d.update({"root.y": -1.5 + 1.5 * abs(up), "root.x": 3 * (rup - lup) if False else 0, "lean": -4, "twist": 15 * up,
                  "head_tilt": -3})
        ks.append((t, d))
    return ks


def _jump():
    # anticipation crouch -> launch -> apex -> land squash -> recover. one jump per cycle.
    T = [0, .15, .3, .45, .6, .75, .9]
    kn = [-25, -60, -10, 0, 0, -50, -35]         # knee bend
    ry = [0, -8, 8, 22, 8, -6, -3]               # root height (cm)
    sq = [0, .06, 0, 0, 0, .10, .04]
    up = [-10, -40, 40, 60, 40, -30, -20]        # arm swing (back on crouch, up on launch)
    bend = [5, 25, -5, -10, -5, 20, 10]
    ks = []
    for i, t in enumerate(T):
        d = {}
        for s in "LR":
            d.update(_leg(s, -kn[i] * 0.45, kn[i], 0 if i not in (2, 3) else -30, abd=5))
            d.update(_arm(s, up[i], 15, wrist=10, abd=15))
        d.update({"root.y": ry[i], "squash": sq[i], "lean": bend[i] * 0.4, "bend": bend[i], "head_tilt": -bend[i] * 0.5})
        ks.append((t, d))
    return ks


def _floss():
    # arms straight, swing side to side crossing in front/behind the hips; hips swing opposite
    T = [i / 8 for i in range(8)]
    ks = []
    for i, t in enumerate(T):
        s = math.sin(2 * math.pi * t); c = math.cos(2 * math.pi * t)
        # both arms move together laterally: abd of one grows as the other shrinks (crossing the body)
        # arms in front of body (swing +) on one half, behind (swing -) on the other
        front = 25 * c
        d = {}
        d.update({"swing.arm_L": front, "abd.arm_L": 20 - 45 * s, "swing.fore_L": front, "abd.fore_L": 20 - 45 * s,
                  "swing.hand_L": front, "abd.hand_L": 20 - 45 * s})
        d.update({"swing.arm_R": front, "abd.arm_R": 20 + 45 * s, "swing.fore_R": front, "abd.fore_R": 20 + 45 * s,
                  "swing.hand_R": front, "abd.hand_R": 20 + 45 * s})
        d.update(_leg("L", 0, -12, 0, abd=10)); d.update(_leg("R", 0, -12, 0, abd=10))
        d.update({"root.x": 4 * s, "root.y": -1, "lean": 2, "twist": -10 * s, "head_tilt": 0})
        ks.append((t, d))
    return ks


def _moonwalk():
    # (held-out) walk-cycle legs played "backwards" with the sliding foot flat and the other on toe;
    # arms hang loose, slight backward lean, head down.
    T = [i / 8 for i in range(8)]
    thigh = [-20, -10, 0, 10, 20, 12, 0, -12]
    knee = [-5, -15, -35, -25, -5, -8, -10, -6]
    dorsi = [-25, -30, -20, 0, 0, 0, 0, -10]
    legL = [(t, _leg("L", th, kn, do)) for t, th, kn, do in zip(T, thigh, knee, dorsi)]
    arms = [(t, {**_arm("L", -6 + 4 * math.sin(2 * math.pi * t), 15, wrist=8, abd=6),
                 **_arm("R", -6 - 4 * math.sin(2 * math.pi * t), 15, wrist=8, abd=6)}) for t in T]
    body = [(t, {"root.y": -1.5, "lean": -6, "bend": 8, "twist": 6 * math.cos(2 * math.pi * t), "head_tilt": 15}) for t in T]
    return _merge(legL, _mirror_LR(_shift(legL, 0.5)), arms, body)


def _helicopter_kick():
    # (held-out) plant on L leg, R leg sweeps a big horizontal circle; arms out for balance, torso leans away
    T = [i / 8 for i in range(8)]
    ks = []
    for i, t in enumerate(T):
        a = 2 * math.pi * t
        d = {}
        # R leg: swing forward/back = cos, abduction = sin (circle), knee nearly straight
        d.update({"swing.leg_R": 45 * math.cos(a), "abd.leg_R": 40 + 35 * math.sin(a), "swing.shin_R": 45 * math.cos(a) - 8,
                  "abd.shin_R": 40 + 35 * math.sin(a), "swing.foot_R": 45 * math.cos(a) + 70, "abd.foot_R": 40 + 35 * math.sin(a)})
        d.update(_leg("L", -8 * math.cos(a), -20, 0, abd=4))
        d.update(_arm("L", 0, 10, wrist=0, abd=85 - 10 * math.sin(a))); d.update(_arm("R", 0, 10, wrist=0, abd=85 + 10 * math.sin(a)))
        d.update({"root.x": -3 * math.sin(a), "root.y": -3, "lean": -10 * math.cos(a), "twist": 25 * math.sin(a), "head_tilt": 0})
        ks.append((t, d))
    return ks


STYLES = {
    "idle_bob": _idle_bob,
    "walk": _walk,
    "run": _run,
    "jumping_jack": _jumping_jack,
    "wave": _wave,
    "robot": _robot,
    "twist": _twist,
    "disco_point": _disco_point,
    "jump": _jump,
    "floss": _floss,
    "moonwalk": _moonwalk,          # held out (test only)
    "helicopter_kick": _helicopter_kick,   # held out (test only)
}
STEPPED = {"robot"}
HELD_OUT = {"moonwalk", "helicopter_kick"}
_CACHE: dict[str, Clip] = {}


def clip(style: str) -> Clip:
    if style not in _CACHE:
        _CACHE[style] = Clip(STYLES[style]())
    return _CACHE[style]


def pose(style: str, t: float, mp: MotionParams | None = None) -> Pose:
    """t in [0,1) cyclic -> Pose. THE seam."""
    mp = mp or MotionParams(stepped=(style in STEPPED))
    c = clip(style)
    p = Pose()
    for seg in LIMB_SEGS:
        tt = t - mp.lag * CHAIN_DEPTH[seg]
        p.swing[seg] = math.radians(c.sample(f"swing.{seg}", tt, mp.stepped) * mp.amp)
        p.abd[seg] = math.radians(c.sample(f"abd.{seg}", tt, mp.stepped) * mp.amp)
    p.root = (c.sample("root.x", t, mp.stepped) / 100.0, c.sample("root.y", t, mp.stepped) * mp.bounce / 100.0,
              c.sample("root.z", t, mp.stepped) / 100.0)
    p.lean = math.radians(c.sample("lean", t, mp.stepped))
    p.bend = math.radians(c.sample("bend", t, mp.stepped))
    p.twist = math.radians(c.sample("twist", t, mp.stepped))
    p.head_tilt = math.radians(c.sample("head_tilt", t, mp.stepped))
    p.squash = c.sample("squash", t, mp.stepped) * mp.squash_gain
    return p
