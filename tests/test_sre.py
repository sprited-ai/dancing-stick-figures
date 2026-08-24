import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train.sre import SRE, RigFrames, masked_joint_l2, RIG_JOINTS


@pytest.fixture()
def tiny_cache(tmp_path):
    """Two train clips + one val clip of 16x16 frames with aligned rig labels."""
    n, size = 12, 16
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (n, size, size, 4), dtype=np.uint8)
    rig = rng.uniform(0.1, 0.9, (n, RIG_JOINTS, 2)).astype(np.float16)
    rig[0, 0] = (1.2, 0.5)          # off-screen joint in the first frame
    np.save(tmp_path / "frames.npy", frames)
    np.save(tmp_path / "rig.npy", rig)
    clips = {
        "a/c0": {"start": 0, "n": 4, "split": "train", "group": "g", "text": "t", "qa": ""},
        "b/c0": {"start": 4, "n": 4, "split": "train", "group": "g", "text": "t", "qa": "levitation"},
        "c/c0": {"start": 8, "n": 4, "split": "val", "group": "g", "text": "t", "qa": ""},
    }
    (tmp_path / "clips.json").write_text(json.dumps(clips))
    return tmp_path


def test_rig_frames_split_and_shapes(tiny_cache):
    train = RigFrames(str(tiny_cache), "train")
    val = RigFrames(str(tiny_cache), "val")
    assert len(train) == 4          # levitation clip dropped
    assert len(val) == 4
    x, rig, visible = train[0]
    assert x.shape == (4, 16, 16) and x.dtype == torch.float32
    assert 0.0 <= x.min() and x.max() <= 1.0          # premultiplied [0,1], not [-1,1]
    assert rig.shape == (RIG_JOINTS, 2) and visible.shape == (RIG_JOINTS,)
    assert not visible[0]           # the 1.2 coord is off-frame
    assert visible[1:].all()


def test_premultiply(tiny_cache):
    ds = RigFrames(str(tiny_cache), "train")
    x, _, _ = ds[0]
    raw = np.load(tiny_cache / "frames.npy")[0].astype(np.float32) / 255.0
    expected = raw[..., :3] * raw[..., 3:4]
    np.testing.assert_allclose(x[:3].permute(1, 2, 0).numpy(), expected, atol=1e-6)


def test_masked_loss_ignores_hidden_joints():
    pred = torch.zeros(2, RIG_JOINTS, 2)
    target = torch.zeros(2, RIG_JOINTS, 2)
    target[:, 0] = 10.0             # huge error on joint 0
    visible = torch.ones(2, RIG_JOINTS, dtype=torch.bool)
    visible[:, 0] = False
    assert masked_joint_l2(pred, target, visible).item() == 0.0
    visible[:, 0] = True
    assert masked_joint_l2(pred, target, visible).item() > 0.0


def test_model_output_range_and_shape():
    model = SRE(size=16, widths=(8, 16), hidden=32)
    out = model(torch.rand(3, 4, 16, 16))
    assert out.shape == (3, RIG_JOINTS, 2)
    assert (out >= 0).all() and (out <= 1).all()


def test_tiny_overfit(tiny_cache):
    torch.manual_seed(0)
    ds = RigFrames(str(tiny_cache), "train")
    x = torch.stack([ds[i][0] for i in range(4)])
    rig = torch.stack([ds[i][1] for i in range(4)])
    visible = torch.stack([ds[i][2] for i in range(4)])
    model = SRE(size=16, widths=(8, 16), hidden=32)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(80):
        loss = masked_joint_l2(model(x), rig, visible)
        if first is None:
            first = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.2


def test_to_model_input_downsamples():
    from eval.sre_validate import to_model_input
    img = np.full((32, 32, 4), 255, dtype=np.uint8)
    x = to_model_input(img, 16, "cpu")
    assert x.shape == (1, 4, 16, 16)
    assert torch.allclose(x, torch.ones_like(x))
