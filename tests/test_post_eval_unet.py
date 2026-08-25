import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.post_eval_unet import mean_pairwise_l1


def test_mean_pairwise_l1_is_zero_for_identical_and_one_for_opposites():
    identical = torch.zeros(3, 4, 2, 2, 2)
    opposites = torch.stack((torch.zeros(4, 2, 2, 2), torch.ones(4, 2, 2, 2)))
    assert mean_pairwise_l1(identical) == 0.0
    assert mean_pairwise_l1(opposites) == 1.0
