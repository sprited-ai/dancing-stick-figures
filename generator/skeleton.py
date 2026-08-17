"""Canonical skeleton = NVIDIA ARDY `cskel27` (27 joints, same names / order / parents /
rest proportions), so hand-keyed clips and ARDY clips share one label space.

Frame (cskel27 convention): **x = figure's LEFT, y = UP, z = FORWARD**. Metres in labels;
the renderer works in px via Body.px_per_m.

Hand-keyed limbs are parameterised by two angles per bone:
  swing (theta): in the sagittal (y-z) plane, 0 = straight down, +ve = forward (+z)
  abd   (phi):   out of the sagittal plane toward the figure's own side (L: +x, R: -x)
direction(theta, phi, side) = (side*sin(phi), -cos(theta)cos(phi), sin(theta)cos(phi))

Camera: orthographic, yaw psi about y. psi=0 -> front view (figure faces camera),
psi=90deg -> side view facing screen-right. Small pitch for 3/4-from-above.
"""
from __future__ import annotations
import json, math, os
from dataclasses import dataclass, field

_NEUTRAL_PATH = os.path.join(os.path.dirname(__file__), "cskel27_neutral.json")
_N = json.load(open(_NEUTRAL_PATH))
NAMES = list(_N.keys())                                   # ARDY bone order
PARENT = {n: _N[n]["parent"] for n in NAMES}
NEUTRAL = {n: tuple(_N[n]["pos"]) for n in NAMES}        # metres, T-pose, Hips at origin
IDX = {n: i for i, n in enumerate(NAMES)}
CHILDREN = {n: [c for c in NAMES if PARENT[c] == n] for n in NAMES}


def rest_offset(child: str) -> tuple[float, float, float]:
    p = PARENT[child]
    if p is None:
        return (0.0, 0.0, 0.0)
    a, b = NEUTRAL[p], NEUTRAL[child]
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def rest_len(child: str) -> float:
    return math.dist((0, 0, 0), rest_offset(child))


# Semantic limb-bone names used by the motion library -> (parent joint, child joint).
# A cskel "bone" is named by its child joint; these aliases keep keyframes readable.
LIMB = {
    "arm_L": ("LeftArm", "LeftForeArm"),   "fore_L": ("LeftForeArm", "LeftHand"),  "hand_L": ("LeftHand", "LeftHandEnd"),
    "arm_R": ("RightArm", "RightForeArm"), "fore_R": ("RightForeArm", "RightHand"), "hand_R": ("RightHand", "RightHandEnd"),
    "leg_L": ("LeftUpLeg", "LeftLeg"),     "shin_L": ("LeftLeg", "LeftFoot"),      "foot_L": ("LeftFoot", "LeftToeBase"),
    "leg_R": ("RightUpLeg", "RightLeg"),   "shin_R": ("RightLeg", "RightFoot"),    "foot_R": ("RightFoot", "RightToeBase"),
}
LIMB_SEGS = list(LIMB)
CHAIN_DEPTH = {s: {"arm": 1, "leg": 1, "fore": 2, "shin": 2, "hand": 3, "foot": 3}[s.split("_")[0]] for s in LIMB_SEGS}
SPINE = ["Spine", "Spine1", "Spine2", "Spine3", "Neck"]   # Hips -> ... -> Neck
THUMB_LEN = rest_len("LeftHandThumb1")


@dataclass
class Body:
    """Per-figure constants. Bone lengths come from cskel27 neutral × bone_scale."""
    px_per_m: float = 55.0
    bone_scale: dict = field(default_factory=dict)      # child joint -> multiplier
    head_r_m: float = 0.125                             # drawn head disc radius (m)
    stroke: float = 4.0                                 # px at 128
    hand_style: str = "mitten"    # "mitten" (filled, default) | "fingers"
    foot_style: str = "shoe"      # "shoe" (default) | "segment"

    def length_m(self, child: str) -> float:
        return rest_len(child) * self.bone_scale.get(child, 1.0)

    def offset_m(self, child: str) -> tuple[float, float, float]:
        s = self.bone_scale.get(child, 1.0)
        ox, oy, oz = rest_offset(child)
        return (ox * s, oy * s, oz * s)

    @property
    def head_r(self) -> float:
        return self.head_r_m * self.px_per_m


@dataclass
class Pose:
    """Everything that varies per frame (hand-keyed path)."""
    swing: dict[str, float] = field(default_factory=dict)   # limb seg -> theta (rad)
    abd: dict[str, float] = field(default_factory=dict)     # limb seg -> phi (rad)
    root: tuple[float, float, float] = (0.0, 0.0, 0.0)      # Hips offset (m), cskel frame
    lean: float = 0.0        # forward pitch of the whole spine at its base (rad)
    bend: float = 0.0        # extra forward curl distributed along the spine (rad, total)
    twist: float = 0.0       # shoulder yaw vs hips (rad), +ve = left shoulder forward
    squash: float = 0.0      # spine length multiplier delta
    head_tilt: float = 0.0   # forward (rad)
    palm_fwd: bool = True    # keyed default: thumbs point forward (palms face the body)


@dataclass
class Camera:
    yaw: float = math.radians(90.0)
    pitch: float = 0.0
    center: tuple[float, float] = (64.0, 66.0)   # where Hips lands on screen (px)
    scale: float = 1.0


def _dir(theta: float, phi: float, side: float) -> tuple[float, float, float]:
    return side * math.sin(phi), -math.cos(theta) * math.cos(phi), math.sin(theta) * math.cos(phi)


def _rot_x(v, a):   # pitch about x (left axis): +a tips +y toward +z (forward)
    x, y, z = v
    c, s = math.cos(a), math.sin(a)
    return (x, y * c - z * s, y * s + z * c)


def _rot_y(v, a):   # yaw about y: +a turns +x (left) toward +z (forward)
    x, y, z = v
    c, s = math.cos(a), math.sin(a)
    return (x * c + z * s, y, -x * s + z * c)


def _add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def _mul(a, k): return (a[0] * k, a[1] * k, a[2] * k)


def _unit(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / n, v[1] / n, v[2] / n)


def fk3d(pose: Pose, body: Body) -> dict[str, tuple[float, float, float]]:
    """Hand-keyed forward kinematics -> all 27 cskel joints, metres, cskel frame."""
    j: dict[str, tuple[float, float, float]] = {}
    j["Hips"] = pose.root
    # spine: rest offsets pitched progressively (lean at base, + bend/len(SPINE) per joint)
    ang = pose.lean
    prev = "Hips"
    for name in SPINE:
        ang += pose.bend / len(SPINE)
        off = _rot_x(_mul(body.offset_m(name), 1.0 - pose.squash), ang)
        j[name] = _add(j[prev], off)
        prev = name
    j["Head"] = _add(j["Neck"], _rot_x(body.offset_m("Head"), ang + pose.head_tilt))
    # clavicles + shoulder joints hang off Spine3, yawed by twist, pitched with the spine
    for side in ("Left", "Right"):
        for name in (f"{side}Shoulder", f"{side}Arm"):
            off = _rot_x(_rot_y(body.offset_m(name), pose.twist), ang)
            j[name] = _add(j[PARENT[name]], off)
    # hips
    for name in ("LeftUpLeg", "RightUpLeg"):
        j[name] = _add(j["Hips"], body.offset_m(name))
    # limbs from swing/abd
    for seg, (parent, child) in LIMB.items():
        side = 1.0 if seg.endswith("L") else -1.0
        d = _dir(pose.swing.get(seg, 0.0), pose.abd.get(seg, 0.0), side)
        j[child] = _add(j[parent], _mul(d, body.length_m(child)))
    # thumbs: perpendicular to the hand direction, toward +z (forward) by default
    for side in ("Left", "Right"):
        w, e = j[f"{side}Hand"], j[f"{side}HandEnd"]
        u = _unit((e[0] - w[0], e[1] - w[1], e[2] - w[2]))
        ref = (0.0, 0.0, 1.0 if pose.palm_fwd else -1.0)
        lat = _unit(_add(ref, _mul(u, -(ref[0] * u[0] + ref[1] * u[1] + ref[2] * u[2]))))
        j[f"{side}HandThumb1"] = _add(w, _mul(lat, THUMB_LEN * body.bone_scale.get(f"{side}HandThumb1", 1.0)))
    return j


def project(j3: dict[str, tuple[float, float, float]], cam: Camera, px_per_m: float = 55.0) -> tuple[dict, dict]:
    """Orthographic, cskel frame (x left, y up, z fwd) -> screen px + per-joint depth (nearer = larger)."""
    cy, sy = math.cos(cam.yaw), math.sin(cam.yaw)
    cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
    k = px_per_m * cam.scale
    j2, depth = {}, {}
    for name, (x, y, z) in j3.items():
        sx = z * sy + x * cy          # screen-x
        d = z * cy - x * sy           # toward camera
        y2 = y * cp + d * sp
        d = d * cp - y * sp
        j2[name] = (cam.center[0] + sx * k, cam.center[1] - y2 * k)
        depth[name] = d
    return j2, depth
