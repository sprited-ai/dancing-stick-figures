import json

import numpy as np

from generator.skeleton import IDX, NAMES, NEUTRAL
from train.motion_data import (MotionCache, MotionClip, align_clip, canonical_components,
                               frame_metadata_index, write_cache)


def _world_motion(frames=120, fps=20.0):
    base = np.asarray([NEUTRAL[name] for name in NAMES], dtype=np.float32)
    p = np.repeat(base[None], frames, axis=0)
    time = np.arange(frames, dtype=np.float32) / fps
    p[..., 0] += 0.2 * time[:, None]
    p[..., 1] += 0.04 * np.sin(time * np.pi)[:, None]
    return p, time


def _clip(clip_id="a", split="train", contacts=True):
    world, _ = _world_motion()
    joints, root, heading = canonical_components(world)
    foot = np.zeros((len(world), 4), dtype=np.bool_)
    foot[10:20, 0] = True
    return MotionClip(clip_id, split, "a person walks", joints, root, heading,
                      foot if contacts else None, "locomotion", 0, 20.0)


def test_canonical_components_separate_pose_and_root():
    world, time = _world_motion()
    joints, root, heading = canonical_components(world)
    np.testing.assert_allclose(joints[:, IDX["Hips"]], 0, atol=1e-6)
    np.testing.assert_allclose(joints[0], joints[-1], atol=1e-6)
    np.testing.assert_allclose(root[:, 0], 0.2 * time, atol=1e-5)
    np.testing.assert_allclose(root[0], 0, atol=1e-7)
    np.testing.assert_allclose(heading, np.tile([1.0, 0.0], (len(world), 1)), atol=1e-6)


def test_align_clip_is_exact_50_at_10fps_and_contacts_are_discrete():
    aligned = align_clip(_clip(), frames=50, fps=10)
    assert aligned.joints.shape == (50, 27, 3)
    assert aligned.root.shape == (50, 3)
    assert aligned.heading.shape == (50, 2)
    assert aligned.contacts.shape == (50, 4)
    assert aligned.contacts.dtype == np.bool_
    np.testing.assert_allclose(aligned.root[:, 0], np.arange(50) * 0.02, atol=1e-6)
    assert aligned.contacts[5:10, 0].all()
    assert not aligned.contacts[:5, 0].any()


def test_cache_uses_train_stats_and_preserves_contact_availability(tmp_path):
    train = _clip("train/a", "train", contacts=True)
    val = _clip("val/b", "val", contacts=False)
    val = MotionClip(val.clip_id, val.split, val.text, val.joints + 10, val.root + 20,
                     val.heading, val.contacts, val.group, val.seed, val.source_fps)
    write_cache([val, train], tmp_path)

    train_ds = MotionCache(tmp_path, "train")
    val_ds = MotionCache(tmp_path, "val")
    assert len(train_ds) == len(val_ds) == 1
    np.testing.assert_allclose(train_ds[0]["joints"].mean(axis=0), 0, atol=1e-4)
    assert val_ds[0]["joints"].mean() > 1e4  # constant train channels use a stable 0.1 mm floor
    assert train_ds[0]["contacts_available"] is True
    assert val_ds[0]["contacts_available"] is False
    assert not val_ds[0]["contacts"].any()
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["frames"] == 50 and meta["fps"] == 10
    assert meta["normalization"] == "mean/std fitted on split=train only"
