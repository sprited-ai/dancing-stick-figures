"""ARDY (.npz, cskel27) -> stickdance frames.

Bypasses our FK: takes world-space posed_joints [T, 27, 3], maps 27 -> our 19 joints,
re-expresses them in the figure frame (y up, x forward, z left) using the first frame's
facing, scales metres -> px, then goes through the SAME project() + render() as the
hand-keyed styles. This is the week-2 seam in action.
"""
from __future__ import annotations
import math
import numpy as np
from .skeleton import Body, Camera, project
from .render import render, LAT_LEN

CSKEL27 = ["Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
           "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandEnd", "RightHandThumb1",
           "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandEnd", "LeftHandThumb1",
           "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
           "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"]
IDX = {n: i for i, n in enumerate(CSKEL27)}

MAP = {  # ours -> theirs
    "pelvis": "Hips", "spine_lo": "Spine1", "spine_hi": "Spine3", "neck": "Neck", "head": "Head",
    "shoulder_L": "LeftArm", "elbow_L": "LeftForeArm", "wrist_L": "LeftHand", "hand_L": "LeftHandEnd",
    "shoulder_R": "RightArm", "elbow_R": "RightForeArm", "wrist_R": "RightHand", "hand_R": "RightHandEnd",
    "hip_L": "LeftUpLeg", "knee_L": "LeftLeg", "ankle_L": "LeftFoot", "toe_L": "LeftToeBase",
    "hip_R": "RightUpLeg", "knee_R": "RightLeg", "ankle_R": "RightFoot", "toe_R": "RightToeBase",
}


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def load(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    P = np.asarray(d["posed_joints"], dtype=np.float64)  # [T, 27, 3]
    fps = float(d["fps"]) if "fps" in d else 20.0
    text = str(d["text"]) if "text" in d else ""
    return P, fps, text


def to_figure_frame(P: np.ndarray, up_axis: int | None = None, px_per_m: float = 58.0,
                    keep_root_xz: bool = False) -> np.ndarray:
    """World joints -> our frame. Returns [T, 27, 3] in px, (x fwd, y up, z left)."""
    T = P.shape[0]
    hips, hipL, hipR, head = IDX["Hips"], IDX["LeftUpLeg"], IDX["RightUpLeg"], IDX["Head"]
    if up_axis is None:  # guess: axis with largest head-minus-hips component on frame 0
        up_axis = int(np.argmax(np.abs(P[0, head] - P[0, hips])))
    up = np.zeros(3); up[up_axis] = 1.0
    if (P[0, head] - P[0, hips])[up_axis] < 0: up = -up
    left = P[0, hipL] - P[0, hipR]
    left = _unit(left - up * (left @ up))
    fwd = _unit(np.cross(up, left))          # x = y × z
    R = np.stack([fwd, up, left])            # rows: new axes
    Q = (P - P[:, hips:hips + 1, :]) if not keep_root_xz else (P - P[0:1, hips:hips + 1, :])
    Q = Q @ R.T
    if not keep_root_xz:                     # keep vertical root motion, kill horizontal drift
        rooty = (P[:, hips] - P[0, hips]) @ up
        Q[:, :, 1] += rooty[:, None]
    return Q * px_per_m


def frame_joints(Q: np.ndarray, t: int) -> dict:
    j = Q[t]
    out = {k: tuple(j[IDX[v]]) for k, v in MAP.items()}
    # our "head" is the disc centre; cskel Head is the top of neck. push up a bit.
    hx, hy, hz = out["head"]; nx, ny, nz = out["neck"]
    d = _unit(np.array([hx - nx, hy - ny, hz - nz]))
    out["head"] = (hx + d[0] * 6, hy + d[1] * 6, hz + d[2] * 6)
    # palm roll from the thumb: lateral = thumb direction made perpendicular to hand direction.
    # Fingers fan along this axis; when it foreshortens (palm edge-on) the fan collapses.
    for s, side in (("L", "Left"), ("R", "Right")):
        w = j[IDX[f"{side}Hand"]]; e = j[IDX[f"{side}HandEnd"]]; th = j[IDX[f"{side}HandThumb1"]]
        u = _unit(e - w)
        lat = th - w
        lat = _unit(lat - u * (lat @ u))
        out[f"wrist_{s}_lat"] = tuple(w + lat * LAT_LEN)
    return out


def render_clip(npz_path: str, cam: Camera | None = None, body: Body | None = None,
                stride: int = 1, colored=True, bg=(255, 255, 255, 255)):
    P, fps, text = load(npz_path)
    Q = to_figure_frame(P)
    cam = cam or Camera(yaw=math.radians(50), center=(64.0, 74.0))
    body = body or Body()
    frames = []
    for t in range(0, Q.shape[0], stride):
        j2, depth = project(frame_joints(Q, t), cam)
        frames.append(render(j2, depth, body, colored=colored, bg=bg))
    return frames, fps / stride, text
