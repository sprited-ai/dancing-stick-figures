import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from generator.build import _h, sample_body, sample_cameras
from generator.rebuild_from_motion import _body_and_cameras, render_motion_clip
from generator.skeleton import NAMES, NEUTRAL


def motion_row(frames=2):
    neutral = np.asarray([NEUTRAL[name] for name in NAMES], dtype=np.float32)
    posed = np.repeat(neutral[None], frames, axis=0)
    posed[:, :, 2] += np.linspace(0, 0.03, frames, dtype=np.float32)[:, None]
    return {
        "clip_id": "dance/example_s0",
        "group": "dance",
        "held_out": False,
        "split": "train",
        "text": "a person performs a small dance",
        "seed": 0,
        "fps": 20,
        "n_frames": frames,
        "posed_joints": posed.tobytes(),
    }


def test_default_body_and_cameras_match_original_clip_id_sampling():
    clip_id = "dance/example_s0"
    expected_rng = random.Random(_h(clip_id))
    expected_body = sample_body(expected_rng)
    expected_cameras = sample_cameras(expected_rng)
    body, cameras = _body_and_cameras(clip_id, "", {})
    assert body == expected_body
    assert cameras == expected_cameras


def test_rebuild_is_deterministic_and_private_variant_changes_rendering():
    row = motion_row()
    public_a = render_motion_clip((row, "", {}))
    public_b = render_motion_clip((row, "", {}))
    private = render_motion_clip((row, "course-secret", {"stroke_scale": 1.2}))

    assert len(public_a) == 3 * row["n_frames"]
    assert public_a[0]["sample_id"] == "dance/example_s0/c0/f000"
    assert public_a[0]["color"]["bytes"] == public_b[0]["color"]["bytes"]
    assert public_a[0]["joint_xyz"] == public_b[0]["joint_xyz"]
    assert public_a[0]["color"]["bytes"] != private[0]["color"]["bytes"]
    assert public_a[0]["bone_scale"] != private[0]["bone_scale"]
