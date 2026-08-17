"""ARDY (.npz, cskel27) -> stickdance frames.

ARDY already speaks our canonical skeleton, so this is just: world -> figure frame
(x left, y up, z forward, Hips at origin, facing fixed from frame 0), metres, then the SAME
project() + render() the hand-keyed styles use.
"""
from __future__ import annotations
import math
import numpy as np
from .skeleton import Body, Camera, project, NAMES, IDX
from .render import render


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def load(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    P = np.asarray(d["posed_joints"], dtype=np.float64)  # [T, 27, 3] world, metres
    fps = float(d["fps"]) if "fps" in d else 20.0
    text = str(d["text"]) if "text" in d else ""
    return P, fps, text


def to_figure_frame(P: np.ndarray, keep_root_xz: bool = False) -> np.ndarray:
    """World -> cskel figure frame (x left, y up, z fwd), Hips-centred, metres. [T,27,3]"""
    hips, hipL, hipR, head = IDX["Hips"], IDX["LeftUpLeg"], IDX["RightUpLeg"], IDX["Head"]
    up = np.zeros(3); up[int(np.argmax(np.abs(P[0, head] - P[0, hips])))] = 1.0
    if (P[0, head] - P[0, hips]) @ up < 0: up = -up
    left = P[0, hipL] - P[0, hipR]
    left = _unit(left - up * (left @ up))
    fwd = _unit(np.cross(left, up))          # z = x × y  (right-handed: x left, y up, z fwd)
    R = np.stack([left, up, fwd])            # rows = new axes
    Q = (P - P[:, hips:hips + 1, :]) if not keep_root_xz else (P - P[0:1, hips:hips + 1, :])
    Q = Q @ R.T
    if not keep_root_xz:                     # keep vertical root motion, kill horizontal drift
        Q[:, :, 1] += ((P[:, hips] - P[0, hips]) @ up)[:, None]
    return Q


def frame_joints(Q: np.ndarray, t: int) -> dict:
    return {n: tuple(Q[t, IDX[n]]) for n in NAMES}


def render_clip(npz_path: str, cam: Camera | None = None, body: Body | None = None,
                stride: int = 1, colored=True, bg=(255, 255, 255, 255)):
    P, fps, text = load(npz_path)
    Q = to_figure_frame(P)
    cam = cam or Camera(yaw=math.radians(50))
    body = body or Body()
    frames = []
    for t in range(0, Q.shape[0], stride):
        j2, depth = project(frame_joints(Q, t), cam, body.px_per_m)
        frames.append(render(j2, depth, body, colored=colored, bg=bg))
    return frames, fps / stride, text
