import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.post_eval_t2v import (
    _straight_rgba_uint8,
    experiment_specs,
    load_prompts,
    make_noise_batch,
    save_labeled_gif,
    save_strip,
    sample_in_chunks,
)
from train.video_dit_fm import VideoDiT


def test_experiment_controls_are_explicit_and_reproducible():
    specs = experiment_specs(["wave", "jump", "turn"], n=3, base_seed=17)
    fixed_noise = specs["fixed_noise_varied_prompt"]
    fixed_prompt = specs["fixed_prompt_varied_noise"]
    assert fixed_noise["prompts"] == ["wave", "jump", "turn"]
    assert fixed_noise["noise_seeds"] == [17, 17, 17]
    assert fixed_prompt["prompts"] == ["wave"] * 3
    assert fixed_prompt["noise_seeds"] == [17, 18, 19]


def test_prompt_loader_deduplicates_json_and_cache(tmp_path):
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({"prompts": [" wave ", "jump", "wave"]}))
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "clips.json").write_text(json.dumps({"a": {"text": "jump"}, "b": {"text": "turn"}}))
    assert load_prompts(str(cache), str(prompts)) == ["wave", "jump", "turn"]


def test_noise_batch_repeats_exactly_for_equal_seeds():
    noise = make_noise_batch((3, 4, 2, 8, 8), "cpu", [5, 5, 6])
    assert torch.equal(noise[0], noise[1])
    assert not torch.equal(noise[0], noise[2])


def test_straight_rgba_export_unpremultiplies_colour():
    # Straight red=.8, alpha=.5 means premultiplied red=.4.
    value = torch.tensor([.4, 0.0, 0.0, .5]).view(1, 4, 1, 1, 1)
    videos = value * 2 - 1
    rgba = _straight_rgba_uint8(videos)
    assert rgba.shape == (1, 1, 1, 1, 4)
    assert abs(int(rgba[0, 0, 0, 0, 0]) - 204) <= 1
    assert abs(int(rgba[0, 0, 0, 0, 3]) - 127) <= 1


def test_labeled_gif_keeps_prompts_visible_on_every_frame(tmp_path):
    videos = torch.full((2, 4, 3, 8, 8), -1.0)
    videos[:, 3] = 1.0
    videos[:, 0, 1] = 0.0
    videos[:, 0, 2] = 1.0
    path = tmp_path / "labeled.gif"
    save_labeled_gif(videos, str(path), ["wave left", "run forward"], fps=10)
    image = Image.open(path)
    assert image.n_frames == 3
    assert image.size == (228, 16)


def test_tiny_cpu_sampling_and_strip_smoke(tmp_path):
    torch.manual_seed(0)
    model = VideoDiT(size=8, frames=2, patch=4, dim=16, depth=2, heads=2, text_dim=8).eval()
    noise = make_noise_batch((2, 4, 2, 8, 8), "cpu", [1, 2])
    text = torch.randn(2, 3, 8)
    mask = torch.ones(2, 3, dtype=torch.long)
    null = torch.zeros_like(text)
    videos = sample_in_chunks(model, noise, text, mask, null, mask, steps=1, cfg=1.0, shift=1.0, batch=1)
    assert videos.shape == (2, 4, 2, 8, 8)
    assert torch.isfinite(videos).all()
    path = tmp_path / "strip.png"
    assert save_strip(videos, str(path), ["one", "two"], [0, 1]) == [0, 1]
    assert path.exists() and path.stat().st_size > 0
