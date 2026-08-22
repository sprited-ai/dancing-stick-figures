import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.protocol import build_reference_manifest, load_manifest_windows, motion_id
from eval.run_ckpt import ar_seam_frames, rollout_chunks
from eval.baselines import degenerate_videos
from eval.oracle import INK, score_seams, score_video
from train.video_ddpm import (
    UNet3D,
    VideoWindows,
    adapt_warm_start_state,
    initialize_video_input,
    val_loss,
    val_losses,
)


def _cache(tmp_path, n_clips=8, frames_per_clip=60, size=4):
    frames = np.zeros((n_clips * frames_per_clip, size, size, 4), dtype=np.uint8)
    clips = {}
    for i in range(n_clips):
        start = i * frames_per_clip
        # Camera pairs share an underlying motion. The first pixel identifies the
        # absolute frame so tests can observe which window was selected.
        frames[start:start + frames_per_clip, ..., 0] = np.arange(frames_per_clip)[:, None, None]
        frames[start:start + frames_per_clip, ..., 3] = 255
        clips[f"dance/motion_{i // 2}/c{i % 2}"] = {
            "start": start,
            "n": frames_per_clip,
            "split": "val",
            "group": "dance",
            "text": "fixture",
            "qa": "",
        }
    np.save(tmp_path / "frames.npy", frames)
    (tmp_path / "clips.json").write_text(json.dumps(clips))
    return tmp_path


def test_deterministic_windows_repeat_exactly(tmp_path):
    cache = _cache(tmp_path)
    ds = VideoWindows(cache, frames=16, split="val", stride=2, size=4, deterministic=True)
    first = ds[0][0].clone()
    for _ in range(5):
        assert torch.equal(first, ds[0][0])


def test_reference_halves_are_motion_disjoint_and_reproducible(tmp_path):
    cache = _cache(tmp_path, n_clips=12)
    one = build_reference_manifest(cache, frames=16, stride=2, split="val", n_per_half=3, seed=7)
    two = build_reference_manifest(cache, frames=16, stride=2, split="val", n_per_half=3, seed=7)
    assert one == two
    left = {motion_id(x["clip_id"]) for x in one["reference_a"]}
    right = {motion_id(x["clip_id"]) for x in one["reference_b"]}
    assert left.isdisjoint(right)
    assert len(one["reference_a"]) == len(one["reference_b"]) == 3
    windows = load_manifest_windows(cache, one["reference_a"], size=4)
    assert windows.shape == (3, 4, 16, 4, 4)


def test_six_ar_chunks_cover_canonical_fifty_frames():
    assert rollout_chunks(context_frames=8, new_frames=8, target_frames=50) == 6
    assert ar_seam_frames(context_frames=8, new_frames=8, target_frames=50) == [16, 24, 32, 40, 48]


def test_degenerate_baselines_preserve_length_and_are_deterministic():
    real = np.arange(2 * 50 * 2 * 2 * 4, dtype=np.uint8).reshape(2, 50, 2, 2, 4)
    one = degenerate_videos(real, seed=9)
    two = degenerate_videos(real, seed=9)
    assert set(one) == {"repeat_first", "shuffle_frames", "reverse_time", "loop_first_8"}
    for name in one:
        assert one[name].shape == real.shape
        assert np.array_equal(one[name], two[name])
    assert np.all(one["repeat_first"] == real[:, :1])


def test_motion_amount_distinguishes_static_from_moving_video():
    moving = np.zeros((10, 16, 16, 4), dtype=np.uint8)
    for t in range(10):
        moving[t, 4:10, 2 + t // 2:8 + t // 2, :3] = INK
        moving[t, 4:10, 2 + t // 2:8 + t // 2, 3] = 255
    static = np.repeat(moving[:1], 10, axis=0)
    moving_score, static_score = score_video(moving), score_video(static)
    assert moving_score["centroid_speed"] > static_score["centroid_speed"]
    assert moving_score["motion_fraction"] > static_score["motion_fraction"]
    assert static_score["centroid_speed"] == 0.0
    assert static_score["motion_fraction"] == 0.0


def test_seam_metrics_isolate_a_boundary_jump():
    video = np.zeros((32, 24, 24, 4), dtype=np.uint8)
    for t in range(32):
        x = 3 if t < 16 else 13
        video[t, 8:14, x:x + 6, :3] = INK
        video[t, 8:14, x:x + 6, 3] = 255
    scores = score_seams(video, seam_frames=[16, 24])
    assert scores["seam_centroid_speed"] > 4.0
    assert scores["within_centroid_speed"] == 0.0
    assert scores["seam_centroid_accel"] > scores["within_centroid_accel"]


class _ZeroModel(torch.nn.Module):
    cls = None

    def forward(self, x, t, y=None, cond=None):
        return torch.zeros_like(x)


def test_validation_loss_uses_fixed_timestep_and_noise(tmp_path):
    cache = _cache(tmp_path)
    ds = VideoWindows(cache, frames=8, split="val", size=4, deterministic=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
    ac = torch.linspace(0.01, 0.99, 100)
    model = _ZeroModel()
    a = val_loss(model, dl, ac, "cpu", n_batches=2, seed=123)
    # Advance both global RNGs: validation must remain a pure function of its seed.
    torch.randn(100)
    np.random.default_rng().normal(size=100)
    b = val_loss(model, dl, ac, "cpu", n_batches=2, seed=123)
    assert a == b


class _ConditionAwareModel(torch.nn.Module):
    cls = None

    def forward(self, x, t, y=None, cond=None):
        # Make the two validation paths observably different without depending
        # on a learned model: a supplied continuation context changes output.
        if cond is None or not torch.any(cond[:, -1:]):
            return torch.zeros_like(x)
        return torch.ones_like(x)


def test_ar_validation_separates_first_chunk_and_continuation_deterministically(tmp_path):
    cache = _cache(tmp_path)
    ds = VideoWindows(cache, frames=8, split="val", size=4, deterministic=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
    ac = torch.linspace(0.01, 0.99, 100)
    model = _ConditionAwareModel()
    one = val_losses(model, dl, ac, "cpu", ar_ctx=4, ctx_noise=0.1, n_batches=2, seed=321)
    torch.randn(100)
    two = val_losses(model, dl, ac, "cpu", ar_ctx=4, ctx_noise=0.1, n_batches=2, seed=321)
    assert one == two
    assert set(one) == {"first_chunk", "continuation"}
    assert one["first_chunk"] != one["continuation"]


def test_image_warm_start_is_framewise_equivalent_for_repeated_video():
    torch.manual_seed(7)
    image = UNet3D(ch=32, mult=(1,), attn_res=(), tattn_res=(), n_res=1, size=4, cond_ch=0)
    # Mimic image training: only temporal centre slices can receive gradients.
    with torch.no_grad():
        for module in image.modules():
            if isinstance(module, torch.nn.Conv3d) and module.kernel_size[0] == 3 and module is not image.inp:
                module.weight.zero_()
                module.weight[:, :, 1].normal_(std=0.02)
        image.out[-1].bias.normal_(std=0.02)
    video = UNet3D(ch=32, mult=(1,), attn_res=(), tattn_res=(), n_res=1, size=4, cond_ch=5)
    adapted = adapt_warm_start_state(image.state_dict(), video.state_dict(), image_source=True)
    video.load_state_dict(adapted, strict=False)
    assert torch.count_nonzero(video.inp.weight[:, :, 0]) == 0
    assert torch.count_nonzero(video.inp.weight[:, :, 2]) == 0

    x = torch.randn(2, 4, 1, 4, 4)
    t = torch.tensor([10, 20])
    with torch.no_grad():
        expected = image(x, t)
        actual = video(x.repeat(1, 1, 3, 1, 1), t)
    torch.testing.assert_close(actual, expected.repeat(1, 1, 3, 1, 1), rtol=1e-5, atol=1e-6)


def test_scratch_ar_uses_matched_structural_zero_initialization():
    torch.manual_seed(11)
    model = UNet3D(ch=32, mult=(1,), attn_res=(), tattn_res=(), n_res=1, size=4, cond_ch=5)
    initialize_video_input(model)
    assert torch.count_nonzero(model.inp.weight[:, :, 0]) == 0
    assert torch.count_nonzero(model.inp.weight[:, :, 2]) == 0
    assert torch.count_nonzero(model.inp.weight[:, 4:, 1]) == 0
    assert torch.count_nonzero(model.inp.weight[:, :4, 1]) > 0
