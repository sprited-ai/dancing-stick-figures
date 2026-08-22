import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train.soft_skeleton_renderer import SoftSkeletonRenderer


def _renderer(size=32):
    # Two independent horizontal bones, useful for controlled overlap tests.
    return SoftSkeletonRenderer(
        parents=[-1, 0, -1, 2],
        bone_colors=[[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]],
        body_radius=1.75,
        image_size=size,
        edge_softness=0.45,
        depth_temperature=0.2,
    )


def _joints(y_first=12.0):
    return torch.tensor(
        [[[[6.0, y_first], [25.0, y_first], [6.0, 12.0], [25.0, 12.0]]]],
        requires_grad=True,
    )


def test_output_shapes_ranges_and_finite_values():
    renderer = _renderer(32)
    joints = _joints()
    depth = torch.tensor([[[1.0, 1.0, 3.0, 3.0]]])
    out = renderer(joints, depth)

    assert out["rgba"].shape == (1, 1, 4, 32, 32)
    assert out["parts"].shape == (1, 1, 2, 32, 32)
    assert out["coverage"].shape == (1, 1, 2, 32, 32)
    assert torch.isfinite(out["rgba"]).all()
    assert torch.all((out["rgba"] >= 0) & (out["rgba"] <= 1))
    assert torch.allclose(out["parts"].sum(2, keepdim=True), out["alpha"], atol=1e-6)


def test_gradients_reach_projected_joint_coordinates():
    renderer = _renderer(32)
    joints = _joints(y_first=10.0)
    depth = torch.tensor([[[1.0, 1.0, 3.0, 3.0]]])
    yy = torch.arange(32.0)[None, None, None, :, None]
    loss = (renderer(joints, depth)["alpha"] * yy).sum()
    loss.backward()

    assert joints.grad is not None
    assert torch.isfinite(joints.grad).all()
    assert joints.grad.abs().sum() > 0


def test_moving_a_limb_moves_its_coverage():
    renderer = _renderer(32)
    depth = torch.ones(1, 1, 4)
    low = renderer(_joints(y_first=8.0), depth)["coverage"][0, 0, 0]
    high = renderer(_joints(y_first=20.0), depth)["coverage"][0, 0, 0]

    rows = torch.arange(32.0)[:, None]
    low_center = (low * rows).sum() / low.sum()
    high_center = (high * rows).sum() / high.sum()
    assert high_center > low_center + 10.0


def test_far_occluded_bone_has_less_visible_ownership():
    renderer = _renderer(32)
    joints = _joints(y_first=12.0)
    depth = torch.tensor([[[1.0, 1.0, 3.0, 3.0]]])
    out = renderer(joints, depth)

    # The bones overlap exactly at this pixel, so raw coverage is equal while
    # the soft z-buffer assigns visibility to the nearer red bone.
    near = out["parts"][0, 0, 0, 12, 16]
    far = out["parts"][0, 0, 1, 12, 16]
    cov_near = out["coverage"][0, 0, 0, 12, 16]
    cov_far = out["coverage"][0, 0, 1, 12, 16]
    assert torch.allclose(cov_near, cov_far, atol=1e-6)
    assert near > 20 * far


def test_supports_64_and_normalized_coordinates():
    renderer = SoftSkeletonRenderer(
        parents=[-1, 0],
        image_size=64,
        normalized_coordinates=True,
    )
    joints = torch.tensor([[[[-0.5, 0.0], [0.5, 0.0]]]])
    depth = torch.ones(1, 1, 2)
    assert renderer(joints, depth)["rgba"].shape == (1, 1, 4, 64, 64)
