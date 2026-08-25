import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train.video_dit_fm import (
    diverse_text_prompts,
    foreground_weighted_mse,
    parse_step_set,
    prepare_warmstart_state,
)


def test_diverse_text_prompts_skips_camera_and_seed_duplicates():
    ds = SimpleNamespace(clips=[
        {"text": "running man"}, {"text": "running man"}, {"text": "running man"},
        {"text": "wave"}, {"text": "wave"}, {"text": "jump"},
    ])
    assert diverse_text_prompts(ds, 3) == ["running man", "wave", "jump"]


def test_diverse_text_prompts_repeats_only_when_needed():
    ds = SimpleNamespace(clips=[{"text": "wave"}, {"text": "jump"}])
    assert diverse_text_prompts(ds, 4) == ["wave", "jump", "wave", "jump"]


def test_foreground_weight_one_is_exact_baseline_mean():
    err = torch.tensor([[[[[1.0, 3.0]]]]])
    clean = torch.zeros(1, 4, 1, 1, 2)
    assert foreground_weighted_mse(err, clean, 1.0) == err.mean()


def test_foreground_weight_emphasizes_visible_pixel_and_normalizes_scale():
    # Background error is 1 and foreground error is 3 in every channel.
    err = torch.tensor([1.0, 3.0]).reshape(1, 1, 1, 1, 2).expand(1, 4, 1, 1, 2)
    clean = torch.zeros(1, 4, 1, 1, 2)
    clean[:, 3, :, :, 0] = -1.0
    clean[:, 3, :, :, 1] = 1.0
    weighted = foreground_weighted_mse(err, clean, 3.0)
    assert weighted.item() == pytest.approx((1.0 + 3.0 * 3.0) / 4.0)
    assert weighted > err.mean()


def test_foreground_weight_rejects_downweighting():
    with pytest.raises(ValueError):
        foreground_weighted_mse(torch.ones(1), torch.zeros(1, 4, 1, 1, 1), 0.5)


def test_foreground_weight_uses_soft_alpha_edges():
    err = torch.tensor([1.0, 3.0]).reshape(1, 1, 1, 1, 2).expand(1, 4, 1, 1, 2)
    clean = torch.zeros(1, 4, 1, 1, 2)
    clean[:, 3, :, :, 0] = -1.0
    clean[:, 3, :, :, 1] = 0.0  # 50% alpha -> weight 2 when foreground_weight=3
    assert foreground_weighted_mse(err, clean, 3.0).item() == pytest.approx((1.0 + 2.0 * 3.0) / 3.0)


def test_parse_step_set_is_deduplicated_and_validated():
    assert parse_step_set("0,1, 5,5,10") == {0, 1, 5, 10}
    assert parse_step_set("") == set()
    with pytest.raises(ValueError):
        parse_step_set("1,nope")
    with pytest.raises(ValueError):
        parse_step_set("-1")


def test_image_warmstart_keeps_fresh_multi_frame_positions():
    source = {"pos_s": torch.ones(1, 1, 4, 8), "pos_t": torch.ones(1, 1, 1, 8)}
    target = {"pos_s": torch.zeros(1, 1, 4, 8), "pos_t": torch.randn(1, 50, 1, 8)}
    selected = prepare_warmstart_state(source, target)
    assert "pos_s" in selected
    assert "pos_t" not in selected


def test_video_warmstart_interpolates_temporal_positions():
    source_pos = torch.arange(50.0).reshape(1, 50, 1, 1)
    source = {"pos_t": source_pos}
    target = {"pos_t": torch.zeros(1, 60, 1, 1)}
    selected = prepare_warmstart_state(source, target)
    assert selected["pos_t"].shape == target["pos_t"].shape
    assert selected["pos_t"][0, 0, 0, 0].item() == pytest.approx(0.0)
    assert selected["pos_t"][0, -1, 0, 0].item() == pytest.approx(49.0)


def test_i2v_warmstart_zero_extends_extra_input_channels():
    source = {"embed.weight": torch.ones(8, 16)}
    target = {"embed.weight": torch.randn(8, 36)}
    selected = prepare_warmstart_state(source, target)
    assert torch.equal(selected["embed.weight"][:, :16], source["embed.weight"])
    assert torch.count_nonzero(selected["embed.weight"][:, 16:]) == 0
