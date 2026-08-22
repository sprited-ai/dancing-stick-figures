import torch

from train.motion_dit_fm import (MOTION_DIM, MotionDiT, euler_sample, flow_loss,
                                 unpack_motion, write_inference_artifacts)


def _tiny_model(contacts=False):
    torch.manual_seed(1)
    return MotionDiT(frames=50, motion_dim=MOTION_DIM, text_dim=24,
                     dim=32, depth=2, heads=4, predict_contacts=contacts)


def test_motion_dit_shape_and_backward():
    model = _tiny_model(contacts=True)
    clean = torch.randn(2, 50, MOTION_DIM)
    text = torch.randn(2, 7, 24)
    mask = torch.tensor([[1] * 7, [1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    contacts = torch.randint(0, 2, (2, 50, 4)).float()
    loss, parts = flow_loss(model, clean, text, mask, contacts,
                            torch.tensor([True, False]), contact_weight=0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["flow"] > 0 and parts["contact"] > 0
    assert model.input.weight.grad is not None


def test_euler_sample_is_finite_and_deterministic_for_fixed_noise():
    model = _tiny_model().eval()
    text = torch.randn(1, 5, 24)
    mask = torch.ones(1, 5, dtype=torch.bool)
    noise = torch.randn(1, 50, MOTION_DIM)
    a = euler_sample(model, noise.shape, text, mask, steps=3, noise=noise)
    b = euler_sample(model, noise.shape, text, mask, steps=3, noise=noise)
    assert a.shape == noise.shape
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, b)


def test_unpack_motion_denormalizes_and_normalizes_heading():
    import numpy as np
    packed = np.zeros((1, 50, MOTION_DIM), dtype=np.float32)
    packed[..., 84] = 3
    packed[..., 85] = 4
    stats = {"joints_mean": np.ones((27, 3), np.float32),
             "joints_std": np.full((27, 3), 2, np.float32),
             "root_mean": np.ones(3, np.float32), "root_std": np.full(3, 3, np.float32)}
    joints, root, heading = unpack_motion(packed, stats)
    np.testing.assert_allclose(joints, 1)
    np.testing.assert_allclose(root, 1)
    np.testing.assert_allclose(heading, np.tile([0.6, 0.8], (1, 50, 1)), atol=1e-6)


def test_fixed_inference_artifacts_include_gif_strip_and_manifest(tmp_path):
    import json
    import numpy as np
    from generator.skeleton import NAMES, NEUTRAL
    pose = np.asarray([NEUTRAL[name] for name in NAMES], dtype=np.float32)
    joints = np.tile(pose, (1, 50, 1, 1))
    root = np.zeros((1, 50, 3), dtype=np.float32)
    paths = write_inference_artifacts(joints, root, ["a person stands"], str(tmp_path), 10, 123)
    assert all((tmp_path / "inference_000010" / name).exists()
               for name in ("sample.gif", "strip.png", "manifest.json"))
    manifest = json.loads(open(paths["manifest"]).read())
    assert manifest["noise_seed"] == 123 and manifest["frames"] == 50
