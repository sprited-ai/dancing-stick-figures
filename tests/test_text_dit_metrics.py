import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.text_dit_metrics import load_split_prompts, mean_pairwise_l1, oracle_summary


def test_load_split_prompts_filters_and_deduplicates(tmp_path):
    rows = {
        "a": {"split": "train", "text": "train prompt"},
        "b": {"split": "test", "text": "test prompt"},
        "c": {"split": "test", "text": "test prompt"},
        "d": {"split": "test", "text": "second test prompt"},
    }
    (tmp_path / "clips.json").write_text(json.dumps(rows))
    assert load_split_prompts(str(tmp_path), "test") == ["test prompt", "second test prompt"]
    with pytest.raises(ValueError, match="no prompts"):
        load_split_prompts(str(tmp_path), "val")


def test_mean_pairwise_l1_has_known_scale():
    videos = torch.tensor([0.0, 1.0, 2.0]).reshape(3, 1, 1, 1, 1)
    # unordered distances are 1, 2, 1
    assert mean_pairwise_l1(videos) == pytest.approx(4 / 3)
    with pytest.raises(ValueError, match="at least two"):
        mean_pairwise_l1(videos[:1])


def test_oracle_summary_returns_means_and_intervals():
    # Empty generated frames exercise the oracle's explicit collapse path.
    videos = -torch.ones(2, 4, 3, 16, 16)
    result = oracle_summary(videos, bootstrap=10)
    assert result["tvr"] == 1.0
    assert result["lie"] == 1.0
    assert result["motion_fraction"] == 0.0
    assert len(result["tvr_ci95"]) == 2
