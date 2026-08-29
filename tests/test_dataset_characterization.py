import json

import numpy as np

from eval.dataset_characterization import characterise


def test_characterisation_deduplicates_cameras_and_uses_retained_rig(tmp_path):
    release = tmp_path / "release"
    legacy = tmp_path / "legacy"
    release.mkdir()
    legacy.mkdir()
    clips = {}
    old_clips = {}
    frames = []
    rigs = []
    for motion, split, qa in (("a", "train", ""), ("b", "test", "frozen")):
        for camera in range(3):
            clip_id = f"dance/{motion}_s0/c{camera}"
            start = len(frames)
            rgba = np.zeros((4, 4, 4, 4), np.uint8)
            rgba[:, 1:3, 1:3, 3] = 255
            rgba[:, 1:3, 1:3, 0] = 40
            frames.extend(rgba)
            clips[clip_id] = {"start": start, "n": 4, "split": split, "group": "dance", "text": motion, "qa": qa}
            old_start = len(rigs)
            rig = np.zeros((4, 27, 2), np.float16)
            rig[:, :, 0] = np.arange(27)[None, :] / 27
            rig[:, :, 1] = np.arange(27)[None, :] / 27
            rig[:, 15, 0] += np.arange(4) * 0.01
            rigs.extend(rig)
            old_clips[clip_id] = {**clips[clip_id], "start": old_start}
    (release / "clips.json").write_text(json.dumps(clips))
    (legacy / "clips.json").write_text(json.dumps(old_clips))
    (legacy / "body_params.json").write_text(json.dumps({key: {"px_per_m": 50 + i} for i, key in enumerate(clips)}))
    np.save(release / "frames.npy", np.asarray(frames))
    np.save(legacy / "rig.npy", np.asarray(rigs))

    result = characterise(release, legacy)

    assert result["counts"]["source_motions"] == 2
    assert result["counts"]["rendered_clips"] == 6
    assert result["qa_flags_source_motion"]["overall"] == {"frozen": 1, "none": 1}
    assert result["distributions"]["centroid_speed_px_per_frame_camera0"]["n"] == 2
    assert result["distributions"]["foreground_fraction"]["n"] == 6
    assert result["body_parameter_ranges"]["px_per_m"] == {"min": 50.0, "max": 55.0}
    assert result["counts"]["prompts_by_group"] == {"dance": 2}
    assert set(result["split_comparisons_ks"]) == {
        "centroid_speed_px_per_frame_camera0", "foreground_fraction",
        "rig_part_angular_path_rad_camera0",
    }
