import numpy as np

from eval.long_horizon_metrics import (
    binned_drift,
    boundary_diagnostic,
    palette_part_trajectories,
    part_motion_diagnostic,
    prompt_attribute_features,
    repetition_diagnostic,
)
from eval.oracle import PAL


def _frame(x=8, y=8, size=32):
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[y:y + 4, x:x + 4, :3] = PAL["ink"]
    rgba[y:y + 4, x:x + 4, 3] = 255
    return rgba


def test_repetition_detects_an_exact_nontrivial_cycle():
    cycle = np.stack([_frame(4 + i, 8) for i in range(8)])
    video = np.concatenate([cycle, cycle, cycle, cycle], axis=0)
    result = repetition_diagnostic(video, min_lag=4, max_lag=12)
    assert result["best_lag"] == 8
    assert result["best_lag_distance"] == 0
    assert result["similarity"] == 1


def test_boundary_diagnostic_isolates_a_fixed_jump():
    video = np.stack([_frame(4 if t < 16 else 20, 8) for t in range(32)])
    result = boundary_diagnostic(video, boundaries=[16])
    assert result["raw"]["seam_centroid_speed"] > 10
    assert result["raw"]["within_centroid_speed"] == 0
    assert np.isnan(result["centroid_speed_ratio"])
    assert result["centroid_speed_excess"] > 10


def test_binned_drift_reports_foreground_growth():
    frames = []
    for t in range(20):
        f = _frame()
        f[12:12 + t // 5, 12:16, :3] = PAL["ink"]
        f[12:12 + t // 5, 12:16, 3] = 255
        frames.append(f)
    result = binned_drift(np.stack(frames), bins=4)
    assert result["slope"]["fg"] > 0
    assert len(result["bins"]) == 4


def _articulated_frame(root_x=8, left_fore_x=3, right_fore_x=21, size=32):
    frame = np.zeros((size, size, 4), np.uint8)
    parts = {
        "ink": (root_x, 12), "arm_L": (root_x - 4, 11), "fore_L": (left_fore_x, 10),
        "arm_R": (root_x + 4, 11), "fore_R": (right_fore_x, 10),
        "leg_L": (root_x - 2, 18), "shin_L": (root_x - 2, 23),
        "leg_R": (root_x + 2, 18), "shin_R": (root_x + 2, 23),
    }
    for name, (x, y) in parts.items():
        frame[y:y + 2, x:x + 2, :3] = PAL[name]
        frame[y:y + 2, x:x + 2, 3] = 255
    return frame


def test_torso_relative_tracks_remove_global_translation():
    video = np.stack([_articulated_frame(root_x=8 + t, left_fore_x=3 + t,
                                         right_fore_x=13 + t) for t in range(6)])
    tracks = palette_part_trajectories(video)
    relative = tracks["relative_centroids"]
    assert np.nanmax(np.abs(np.diff(relative, axis=0))) < 1e-9
    metric = part_motion_diagnostic(video, boundaries=())
    assert metric["root_motion_energy_px"] == 1.0
    assert metric["mean_limb_relative_motion"] < 1e-9


def test_prompt_attributes_detect_left_arm_motion():
    video = np.stack([_articulated_frame(left_fore_x=3 + (t % 4), right_fore_x=21)
                      for t in range(12)])
    attributes = prompt_attribute_features(video)
    assert attributes["left_arm_energy"] > attributes["right_arm_energy"]
    assert attributes["arm_laterality"] > 0


def test_missing_part_is_visibility_not_fake_motion():
    video = np.stack([_articulated_frame() for _ in range(8)])
    video[2:5, :, :, :][..., :] = video[2:5]
    # Remove the left-forearm colour for three consecutive frames.
    colour = np.asarray(PAL["fore_L"], np.uint8)
    for frame in video[2:5]:
        mask = np.all(frame[..., :3] == colour, axis=-1)
        frame[mask] = 0
    result = part_motion_diagnostic(video, boundaries=())
    assert result["parts"]["fore_L"]["visibility"] == 5 / 8
    assert result["parts"]["fore_L"]["longest_dropout"] == 3
    assert result["parts"]["fore_L"]["relative_motion_energy"] == 0
