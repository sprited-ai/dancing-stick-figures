import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train.sre import RIG_JOINTS, SRE
from train.sre_confidence import SREConfidence, gaussian_joint_nll, warm_start_from_v1


def test_confidence_model_shapes_and_positive_sigma():
    model = SREConfidence(size=16, widths=(8, 16), hidden=32)
    joints, log_sigma = model(torch.rand(3, 4, 16, 16))
    assert joints.shape == (3, RIG_JOINTS, 2)
    assert log_sigma.shape == (3, RIG_JOINTS)
    assert (joints >= 0).all() and (joints <= 1).all()
    assert torch.isfinite(log_sigma).all()


def test_gaussian_nll_penalizes_larger_error_at_fixed_sigma():
    target = torch.zeros(1, RIG_JOINTS, 2)
    visible = torch.ones(1, RIG_JOINTS, dtype=torch.bool)
    log_sigma = torch.zeros(1, RIG_JOINTS)
    near = gaussian_joint_nll(torch.zeros_like(target), log_sigma, target, visible)
    far = gaussian_joint_nll(torch.ones_like(target), log_sigma, target, visible)
    assert far > near


def test_v1_warm_start_preserves_coordinate_predictions():
    torch.manual_seed(0)
    source = SRE(size=16, widths=(8, 16), hidden=32)
    target = SREConfidence(size=16, widths=(8, 16), hidden=32)
    checkpoint = {"model": source.state_dict()}
    warm_start_from_v1(target, checkpoint)
    value = torch.rand(2, 4, 16, 16)
    expected = source(value)
    actual, _ = target(value)
    torch.testing.assert_close(actual, expected)
