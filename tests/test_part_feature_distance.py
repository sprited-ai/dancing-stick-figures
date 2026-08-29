import numpy as np

from eval.oracle import PAL
from eval.part_feature_distance import (
    frame_part_features, temporal_part_features, feature_names, part_delta_quantiles,
)


def _frame(horizontal=True):
    image = np.zeros((32, 32, 4), np.uint8)
    image[8:25, 15:18, :3] = PAL["ink"]
    image[8:25, 15:18, 3] = 255
    if horizontal:
        image[13:16, 5:14, :3] = PAL["arm_L"]
        image[13:16, 5:14, 3] = 255
    else:
        image[4:13, 10:13, :3] = PAL["arm_L"]
        image[4:13, 10:13, 3] = 255
    return image


def _column(name):
    return feature_names(False).index(f"arm_L.{name}")


def test_part_features_recover_axis_and_geometry():
    value = frame_part_features(np.stack([_frame(True), _frame(False)]))
    assert value.shape == (2, len(feature_names(False)))
    assert value[:, _column("present")].tolist() == [1.0, 1.0]
    assert value[0, _column("torso_relative_axis_cos2")] < -0.9
    assert value[1, _column("torso_relative_axis_cos2")] > 0.9
    assert abs(value[:, _column("torso_relative_axis_sin2")]).max() < 1e-5


def test_temporal_features_expose_order_while_static_pool_does_not():
    video = np.stack([_frame(True), _frame(False), _frame(True)])
    shuffled = video[[0, 2, 1]]
    static_a, static_b = frame_part_features(video), frame_part_features(shuffled)
    np.testing.assert_allclose(np.sort(static_a, axis=0), np.sort(static_b, axis=0))
    temporal_a = temporal_part_features(static_a)
    temporal_b = temporal_part_features(static_b)
    assert not np.allclose(np.sort(temporal_a, axis=0), np.sort(temporal_b, axis=0))


def test_part_delta_profile_respects_clip_boundaries():
    first = frame_part_features(np.stack([_frame(True), _frame(False), _frame(True)]))
    second = frame_part_features(np.stack([_frame(False), _frame(False), _frame(False)]))
    profile = part_delta_quantiles(np.concatenate([first, second]), videos=2, frames=3)
    arm = profile["arm_L"]
    assert arm["valid_transition_count"] == 4
    assert arm["torso_relative_axis_abs_delta_rad"]["p95"] > 1.0
    assert arm["presence_transition_rate"] == 0.0
