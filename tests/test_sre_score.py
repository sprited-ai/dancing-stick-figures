import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.sre_score import (PARENTS, rasterize_rig, recovery_accuracy,
                            rig_frame_agreement, score_sequences)
from train.sre import RIG_JOINTS


def simple_rig(n=2, frames=8):
    rig = np.zeros((n, frames, RIG_JOINTS, 2), np.float32)
    for joint in range(RIG_JOINTS):
        rig[:, :, joint, 0] = 0.2 + joint * 0.01
        rig[:, :, joint, 1] = 0.3 + joint * 0.005
    return rig


def test_static_rig_has_zero_temporal_diagnostics():
    rig = simple_rig()
    confidence = np.ones(rig.shape[:-1], np.float32)
    result = score_sequences(rig, confidence, 64)
    for name in ("bone_length_temporal_cv", "joint_speed_px_per_frame",
                 "joint_accel_px_per_frame2", "joint_jerk_px_per_frame3"):
        assert np.allclose(result["per_clip"][name], 0.0, atol=1e-6)


def test_length_change_and_motion_are_detected():
    rig = simple_rig(n=1, frames=8)
    child = int(np.flatnonzero(PARENTS >= 0)[-1])
    rig[0, :, child, 0] += np.linspace(0.0, 0.2, rig.shape[1])
    rig[0, 4, child, 1] += 0.2
    confidence = np.ones(rig.shape[:-1], np.float32)
    result = score_sequences(rig, confidence, 64)
    assert result["aggregate"]["bone_length_temporal_cv"]["mean"] > 0
    assert result["aggregate"]["joint_accel_px_per_frame2"]["mean"] > 0
    assert result["aggregate"]["joint_jerk_px_per_frame3"]["mean"] > 0


def test_foreground_support_uses_nearby_alpha():
    rig = simple_rig(n=1, frames=4)
    confidence = np.ones(rig.shape[:-1], np.float32)
    alpha = np.zeros((1, 4, 64, 64), np.float32)
    xy = np.rint(rig[0, 0] * 63).astype(int)
    for x, y in xy:
        alpha[:, :, y, x] = 1.0
    result = score_sequences(rig, confidence, 64, alpha)
    assert result["aggregate"]["foreground_joint_support"]["mean"] == 1.0


def test_recovery_accuracy_uses_fixed_probabilities():
    target = simple_rig(n=1, frames=4)
    prediction = target.copy()
    confidence = np.full(target.shape[:-1], 0.9, np.float32)
    result = recovery_accuracy(prediction, confidence, target, 64)
    assert result["mean_error_px"] == 0.0
    assert result["pck2"] == 1.0
    assert result["empirical_within_limb"] == 1.0
    assert np.isclose(result["brier_within_limb"], 0.01, atol=1e-6)


def test_rig_frame_agreement_is_exact_for_its_own_rasterization():
    rig = simple_rig(n=1, frames=4)
    confidence = np.ones(rig.shape[:-1], np.float32)
    alpha = rasterize_rig(rig, confidence, 64, 64, 64)
    result = rig_frame_agreement(rig, confidence, alpha, 64)["aggregate"]
    assert np.isclose(result["rig_foreground_precision"]["mean"], 1.0)
    assert np.isclose(result["rig_foreground_recall"]["mean"], 1.0)
    assert np.isclose(result["rig_foreground_iou"]["mean"], 1.0)


def test_unexplained_foreground_reduces_recall():
    rig = simple_rig(n=1, frames=4)
    confidence = np.ones(rig.shape[:-1], np.float32)
    alpha = rasterize_rig(rig, confidence, 64, 64, 64)
    baseline = rig_frame_agreement(rig, confidence, alpha, 64)["aggregate"]
    alpha[:, :, :8, :8] = 1.0
    changed = rig_frame_agreement(rig, confidence, alpha, 64)["aggregate"]
    assert changed["rig_foreground_recall"]["mean"] < baseline["rig_foreground_recall"]["mean"]
