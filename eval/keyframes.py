"""Rig-derived representative frames for dataset figures and manifests.

These are derived summary frames, not keyframes supplied by ARDY.
"""
from __future__ import annotations

import numpy as np


def select_representative_frames(joints: np.ndarray, count: int = 4, active_quantile: float = 0.98) -> list[int]:
    """Select ordered frames at equal increments of pose-space motion.

    Global root translation is removed. The last representative frame is the
    point by which ``active_quantile`` of the clip's pose motion has occurred,
    so a long idle tail does not consume most of a film strip.
    """
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[0] < 1 or joints.shape[1] < 1 or joints.shape[2] != 3:
        raise ValueError("joints must have shape [T, J, 3]")
    if count < 1 or count > len(joints):
        raise ValueError("count must be between 1 and the number of frames")
    relative = joints - joints[:, :1]
    increments = np.linalg.norm(np.diff(relative, axis=0), axis=-1).mean(axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(increments)])
    total = float(cumulative[-1])
    if total < 1e-8:
        return np.rint(np.linspace(0, len(joints) - 1, count)).astype(int).tolist()

    active_end = int(np.searchsorted(cumulative, active_quantile * total, side="left"))
    active_end = min(len(joints) - 1, max(count - 1, active_end))
    active_motion = float(cumulative[active_end])
    targets = np.linspace(0.0, active_motion, count)
    selected = [int(np.searchsorted(cumulative, target, side="left")) for target in targets]
    if len(set(selected)) != count:
        selected = np.rint(np.linspace(0, active_end, count)).astype(int).tolist()
    return selected

