"""3D skeleton + orthographic camera.

Figure-local frame: y up, x = facing direction ("forward"), z = figure's LEFT.
Limb segments are parameterised by two angles:
  swing (theta): in the sagittal (x-y) plane, 0 = straight down, +ve = forward
  abd   (phi):   out of the sagittal plane toward the figure's own side (L: +z, R: -z)
direction(theta, phi, side) = (sin(theta)cos(phi), -cos(theta)cos(phi), side*sin(phi))

Camera: orthographic, yaw psi about the vertical axis. psi=0 -> front view (figure faces
camera), psi=90deg -> side view facing screen-right. Also a small pitch for 3/4-from-above.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

# segment -> (parent_joint, child_joint)
SEGMENTS = {
    "torso":  ("pelvis", "neck"),
    "arm_L":  ("shoulder_L", "elbow_L"), "fore_L": ("elbow_L", "wrist_L"), "hand_L": ("wrist_L", "hand_L"),
    "arm_R":  ("shoulder_R", "elbow_R"), "fore_R": ("elbow_R", "wrist_R"), "hand_R": ("wrist_R", "hand_R"),
    "leg_L":  ("hip_L", "knee_L"),  "shin_L": ("knee_L", "ankle_L"),  "foot_L": ("ankle_L", "toe_L"),
    "leg_R":  ("hip_R", "knee_R"),  "shin_R": ("knee_R", "ankle_R"),  "foot_R": ("ankle_R", "toe_R"),
}
JOINTS = ["head", "neck", "pelvis",
          "shoulder_L", "elbow_L", "wrist_L", "hand_L",
          "shoulder_R", "elbow_R", "wrist_R", "hand_R",
          "hip_L", "knee_L", "ankle_L", "toe_L",
          "hip_R", "knee_R", "ankle_R", "toe_R"]
LIMB_SEGS = [s for s in SEGMENTS if s != "torso"]
# depth from root along the chain, for overlapping-action lag
CHAIN_DEPTH = {"torso": 0, **{s: {"arm": 1, "leg": 1, "fore": 2, "shin": 2, "hand": 3, "foot": 3}[s.split("_")[0]]
                              for s in LIMB_SEGS}}
CHAINS = [["torso"], ["arm_L", "fore_L", "hand_L"], ["arm_R", "fore_R", "hand_R"],
          ["leg_L", "shin_L", "foot_L"], ["leg_R", "shin_R", "foot_R"]]


@dataclass
class Body:
    """Per-figure constants (units = px at 128 canvas). Fixed for a whole clip."""
    torso: float = 34.0
    neck_to_head: float = 13.0
    head_r: float = 9.0
    upper_arm: float = 17.0
    forearm: float = 15.0
    hand: float = 7.0
    thigh: float = 20.0
    shin: float = 19.0
    foot: float = 7.0
    hip_w: float = 4.5
    shoulder_w: float = 7.0
    stroke: float = 4.0
    hand_style: str = "mitten"    # "mitten" (filled, default) | "fingers" (palm + 3 fingers)
    foot_style: str = "shoe"      # "shoe" (default) | "segment"

    def length(self, seg: str) -> float:
        base = seg.split("_")[0]
        return {"torso": self.torso, "arm": self.upper_arm, "fore": self.forearm, "hand": self.hand,
                "leg": self.thigh, "shin": self.shin, "foot": self.foot}[base]


@dataclass
class Pose:
    """Everything that varies per frame."""
    swing: dict[str, float] = field(default_factory=dict)   # seg -> theta (rad)
    abd: dict[str, float] = field(default_factory=dict)     # seg -> phi (rad)
    root: tuple[float, float, float] = (0.0, 0.0, 0.0)      # pelvis xyz, figure-local, px
    lean: float = 0.0        # torso forward lean (rad)
    twist: float = 0.0       # shoulder yaw vs hips (rad), +ve = left shoulder forward
    squash: float = 0.0      # torso length delta multiplier
    head_tilt: float = 0.0   # forward (rad)


@dataclass
class Camera:
    yaw: float = math.radians(90.0)   # side view facing screen-right
    pitch: float = 0.0                # +ve = camera slightly above
    center: tuple[float, float] = (64.0, 70.0)   # where pelvis lands on screen (px)
    scale: float = 1.0


def _dir(theta: float, phi: float, side: float) -> tuple[float, float, float]:
    return math.sin(theta) * math.cos(phi), -math.cos(theta) * math.cos(phi), side * math.sin(phi)


def fk3d(pose: Pose, body: Body) -> dict[str, tuple[float, float, float]]:
    """Forward kinematics in figure-local 3D."""
    j: dict[str, tuple[float, float, float]] = {}
    px, py, pz = pose.root
    j["pelvis"] = (px, py, pz)
    tl = body.torso * (1.0 - pose.squash)
    nx, ny, nz = px + tl * math.sin(pose.lean), py + tl * math.cos(pose.lean), pz
    j["neck"] = (nx, ny, nz)
    ht = pose.lean + pose.head_tilt
    j["head"] = (nx + body.neck_to_head * math.sin(ht), ny + body.neck_to_head * math.cos(ht), nz)
    # hips along z; shoulders along z rotated by twist about y
    j["hip_L"] = (px, py, pz + body.hip_w)
    j["hip_R"] = (px, py, pz - body.hip_w)
    sx, sz = body.shoulder_w * math.sin(pose.twist), body.shoulder_w * math.cos(pose.twist)
    j["shoulder_L"] = (nx + sx, ny - 2.0, nz + sz)
    j["shoulder_R"] = (nx - sx, ny - 2.0, nz - sz)

    for chain in CHAINS[1:]:
        side = 1.0 if chain[0].endswith("L") else -1.0
        for seg in chain:
            parent, child = SEGMENTS[seg]
            x, y, z = j[parent]
            dx, dy, dz = _dir(pose.swing.get(seg, 0.0), pose.abd.get(seg, 0.0), side)
            L = body.length(seg)
            j[child] = (x + dx * L, y + dy * L, z + dz * L)
    # palm lateral probe for keyed styles: palms face the body (thumb forward), so the finger
    # fan spreads along the figure's forward axis, orthogonalised to the hand direction.
    for s in "LR":
        wx, wy, wz = j[f"wrist_{s}"]; hx, hy, hz = j[f"hand_{s}"]
        ux, uy, uz = hx - wx, hy - wy, hz - wz
        n = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
        ux, uy, uz = ux / n, uy / n, uz / n
        lx, ly, lz = 1.0 - ux * ux, -ux * uy, -ux * uz          # (1,0,0) minus its u-component
        n = math.sqrt(lx * lx + ly * ly + lz * lz) or 1.0
        j[f"wrist_{s}_lat"] = (wx + 10.0 * lx / n, wy + 10.0 * ly / n, wz + 10.0 * lz / n)
    return j


def project(j3: dict[str, tuple[float, float, float]], cam: Camera) -> tuple[dict, dict]:
    """Orthographic. Returns (joints2d screen px, depth per joint; larger = nearer)."""
    cy, sy = math.cos(cam.yaw), math.sin(cam.yaw)
    cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
    j2, depth = {}, {}
    for k, (x, y, z) in j3.items():
        # yaw about y: camera at azimuth psi. screen_x = x*sin + z*cos ; cam_depth = x*cos - z*sin
        sx = x * sy + z * cy
        d = x * cy - z * sy
        # pitch about screen-x axis
        sy_ = y * cp + d * sp
        d = d * cp - y * sp
        j2[k] = (cam.center[0] + sx * cam.scale, cam.center[1] - sy_ * cam.scale)
        depth[k] = d
    return j2, depth
