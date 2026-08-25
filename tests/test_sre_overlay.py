import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.sre_overlay import recover_premultiplied, rig_metrics


def test_white_composite_inversion_for_palette_colour():
    alpha = 0.4
    straight = np.array([1.0, 0.0, 0.5], np.float32)
    composite = ((straight * alpha + 1.0 - alpha) * 255).round().astype(np.uint8)[None, None]
    premultiplied, recovered_alpha = recover_premultiplied(composite)
    np.testing.assert_allclose(recovered_alpha, alpha, atol=1 / 255)
    np.testing.assert_allclose(premultiplied[0, 0], straight * alpha, atol=2 / 255)


def test_static_rig_has_zero_motion_and_jitter():
    rig = np.zeros((4, 27, 2), np.float32)
    rig[:, :, 0] = np.arange(27) / 64
    alpha = np.ones((4, 64, 64), np.float32)
    result = rig_metrics(rig, alpha, 64)
    assert result["mean_joint_speed_px_per_frame"] == 0.0
    assert result["bone_length_temporal_cv"] == 0.0
