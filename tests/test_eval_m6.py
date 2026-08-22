import json

import numpy as np
import pytest

from eval.eval_m6 import (
    block_boundaries,
    load_real_references,
    prompt_motion_alignment,
    real_tensor_for_oracle,
)


def test_m6_boundaries_cover_every_four_frame_generation_block():
    assert block_boundaries(100, 4) == list(range(4, 100, 4))
    assert block_boundaries(100, 100) == []


def test_real_references_use_one_distinct_camera_zero_motion(tmp_path):
    frames = np.zeros((12, 4, 4, 4), np.uint8)
    frames[..., 3] = 255
    np.save(tmp_path / "frames.npy", frames)
    clips = {
        "motion_a/c0": {"start": 0, "n": 4, "split": "test", "text": "walk", "qa": ""},
        "motion_a/c1": {"start": 4, "n": 4, "split": "test", "text": "walk", "qa": ""},
        "motion_b/c0": {"start": 8, "n": 4, "split": "test", "text": "run", "qa": ""},
    }
    (tmp_path / "clips.json").write_text(json.dumps(clips))
    videos, prompts, clip_ids = load_real_references(str(tmp_path), "test", 2, 4, 4)
    assert videos.shape == (2, 4, 4, 4, 4)
    assert prompts == ["run", "walk"]
    assert clip_ids == ["motion_b/c0", "motion_a/c0"]


def test_real_reference_conversion_premultiplies_rgb():
    rgba = np.asarray([[[[[200, 100, 50, 128]]]]], dtype=np.uint8)
    converted = (real_tensor_for_oracle(rgba) + 1) / 2
    expected_alpha = 128 / 255
    assert abs(float(converted[0, 3, 0, 0, 0]) - expected_alpha) < 1e-6
    assert abs(float(converted[0, 0, 0, 0, 0]) - (200 / 255) * expected_alpha) < 1e-6


def _motion_row(speed, fraction, decay, limb):
    return {
        "whole_video": {"centroid_speed": speed, "motion_fraction": fraction},
        "time_drift": {"slope": {"motion_fraction": decay}},
        "part_motion": {"mean_limb_relative_motion": limb},
    }


def test_prompt_motion_alignment_groups_repeats_and_recovers_perfect_ordering():
    prompts = ["run", "run", "sit", "sit"]
    real = [
        _motion_row(4, .8, -.1, .4), _motion_row(6, 1.0, -.2, .6),
        _motion_row(1, .2, -.8, .1), _motion_row(1, .2, -.6, .1),
    ]
    generated = [
        _motion_row(2, .4, -.05, .2), _motion_row(3, .5, -.1, .3),
        _motion_row(.5, .1, -.4, .05), _motion_row(.5, .1, -.3, .05),
    ]
    result = prompt_motion_alignment(prompts, generated, real)
    assert result["n_videos"] == 4
    assert result["n_prompts"] == 2
    assert result["per_prompt"]["run"]["n"] == 2
    assert result["per_prompt"]["run"]["centroid_speed"] == {"generated": 2.5, "real": 5.0}
    assert result["metrics"]["centroid_speed"]["pearson_across_prompt_means"] == pytest.approx(1.0)
    assert result["metrics"]["centroid_speed"]["normalized_mae"] == pytest.approx(0.5)
