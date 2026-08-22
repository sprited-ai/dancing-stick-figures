import torch

from train.video_dit_ar import (
    FullSTARVideoDiT,
    full_prefix_attention_mask,
    ARVideoDiT,
    TemporalRoPEAttention,
    euler_sample_block,
    history_condition,
    prefix_attention_mask,
    prepare_m3_warmstart,
    rollout_blocks,
    signed_window_positions,
)


def test_full_st_prefix_mask_and_dependency_invariant():
    mask = full_prefix_attention_mask(history=2, target=1, patches=4)
    assert mask.shape == (12, 12)
    assert not mask[:8, 8:].any()
    assert mask[8:].all()

    torch.manual_seed(4)
    model = FullSTARVideoDiT(
        size=4, patch=1, in_ch=2, cond_ch=3, dim=48, depth=2, heads=3, text_dim=16,
    ).eval()
    # Defeat zero-output initialization so the prefix-independence assertion
    # exercises the attention graph rather than comparing constant zeros.
    torch.nn.init.normal_(model.out.weight, std=0.02)
    for block in model.blocks:
        torch.nn.init.normal_(block.ada[-1].weight, std=0.02)
    first = torch.randn(1, 2, 3, 4, 4)
    second = first.clone(); second[:, :, 2:] = torch.randn_like(second[:, :, 2:])
    cond1 = history_condition(first, 2); cond2 = history_condition(second, 2)
    text = torch.randn(1, 5, 16); text_mask = torch.ones(1, 5, dtype=torch.long)
    timestep = torch.full((1,), 0.5)
    with torch.no_grad():
        out1 = model(first, timestep, cond=cond1, text=text, text_mask=text_mask, history_frames=2)
        out2 = model(second, timestep, cond=cond2, text=text, text_mask=text_mask, history_frames=2)
    torch.testing.assert_close(out1[:, :, :2], out2[:, :, :2], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(out1[:, :, 2:], out2[:, :, 2:])
from train.video_dit_fm import VideoDiT


def test_signed_positions_anchor_target_at_zero():
    assert signed_window_positions(3, 4).tolist() == [-3, -2, -1, 0, 1, 2, 3]
    assert signed_window_positions(0, 4).tolist() == [0, 1, 2, 3]


def test_prefix_mask_blocks_target_to_history_feedback_path():
    mask = prefix_attention_mask(2, 3)
    assert mask.shape == (5, 5)
    assert not mask[:2, 2:].any()
    assert mask[:2, :2].all()
    assert mask[2:].all()


def test_negative_rope_positions_are_finite_and_shape_preserving():
    attention = TemporalRoPEAttention(dim=16, heads=2)
    x = torch.randn(2, 5, 16)
    positions = torch.tensor([[-3, -2, -1, 0, 1], [-3, -2, -1, 0, 1]])
    out = attention(x, positions, prefix_attention_mask(3, 2).expand(2, -1, -1))
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_history_condition_contains_only_clean_prefix():
    video = torch.randn(2, 4, 7, 8, 8)
    cond = history_condition(video, 3)
    assert cond.shape == (2, 5, 7, 8, 8)
    torch.testing.assert_close(cond[:, :4, :3], video[:, :, :3])
    assert torch.count_nonzero(cond[:, :4, 3:]) == 0
    assert torch.all(cond[:, 4, :3] == 1)
    assert torch.all(cond[:, 4, 3:] == 0)


def test_m3_warmstart_reuses_common_weights_but_not_absolute_time_positions():
    m3 = VideoDiT(size=8, frames=5, patch=4, dim=16, depth=2, heads=2, cond_ch=5, text_dim=8)
    m4 = ARVideoDiT(size=8, patch=4, dim=16, depth=2, heads=2, cond_ch=5, text_dim=8)
    selected = prepare_m3_warmstart(m3.state_dict(), m4.state_dict())
    assert "pos_t" not in selected
    assert "pos_s" in selected
    assert "blocks.1.attn.qkv.weight" in selected
    result = m4.load_state_dict(selected, strict=False)
    assert not result.unexpected_keys


def test_model_handles_zero_and_nonzero_history():
    model = ARVideoDiT(size=8, patch=4, dim=16, depth=2, heads=2, cond_ch=5, text_dim=8)
    text = torch.randn(2, 3, 8)
    text_mask = torch.ones(2, 3, dtype=torch.bool)
    for history in (0, 3):
        video = torch.randn(2, 4, 7, 8, 8)
        out = model(video, torch.full((2,), 0.5), cond=history_condition(video, history),
                    text=text, text_mask=text_mask, history_frames=history)
        assert out.shape == video.shape
        assert torch.isfinite(out).all()


def test_block_sampler_returns_only_new_frames():
    model = ARVideoDiT(size=8, patch=4, dim=16, depth=2, heads=2, cond_ch=5, text_dim=8).eval()
    text = torch.randn(1, 3, 8)
    mask = torch.ones(1, 3, dtype=torch.bool)
    history = torch.randn(1, 4, 3, 8, 8)
    out = euler_sample_block(model, history, 2, steps=2, size=8, text=text, text_mask=mask, cfg=0)
    assert out.shape == (1, 4, 2, 8, 8)
    assert torch.isfinite(out).all()


def test_block_sampler_is_channel_agnostic_for_latent_m6():
    model = ARVideoDiT(
        size=4, patch=2, in_ch=6, dim=16, depth=2, heads=2,
        cond_ch=7, text_dim=8,
    ).eval()
    text = torch.randn(1, 3, 8)
    mask = torch.ones(1, 3, dtype=torch.bool)
    history = torch.randn(1, 6, 3, 4, 4)
    cond = history_condition(history, 2)
    assert cond.shape == (1, 7, 3, 4, 4)
    out = euler_sample_block(
        model, history, 2, steps=2, size=4, text=text, text_mask=mask, cfg=0, clamp=None,
    )
    assert out.shape == (1, 6, 2, 4, 4)


def test_rollout_can_resume_and_append_without_global_position_limit():
    model = ARVideoDiT(size=8, patch=4, dim=16, depth=2, heads=2, cond_ch=5, text_dim=8).eval()
    text = torch.randn(1, 3, 8)
    mask = torch.ones(1, 3, dtype=torch.bool)
    existing = torch.randn(1, 4, 7, 8, 8).clamp(-1, 1)
    out = rollout_blocks(
        model,
        [(text, mask)],
        initial_video=existing,
        total_frames=5,
        target_frames=3,
        history_max=4,
        steps=1,
        cfg=0,
    )
    assert out.shape == (1, 4, 12, 8, 8)
    torch.testing.assert_close(out[:, :, :7], existing)


def test_full_clip_and_block_rollouts_can_share_the_exact_noise_tensor():
    class ZeroFlow(torch.nn.Module):
        C = 2
        S = 2

        def forward(self, x, timestep, **kwargs):
            return torch.zeros_like(x)

    model = ZeroFlow().eval()
    text = torch.zeros(1, 1, 4)
    mask = torch.ones(1, 1, dtype=torch.bool)
    paired_noise = torch.randn(1, 2, 4, 2, 2)
    common = dict(
        model=model, prompts=[(text, mask)], total_frames=4, history_max=0,
        steps=1, cfg=0, initial_noise=paired_noise, sample_clamp=None,
    )
    full_clip = rollout_blocks(target_frames=4, **common)
    four_blocks = rollout_blocks(target_frames=1, **common)
    torch.testing.assert_close(full_clip, paired_noise)
    torch.testing.assert_close(four_blocks, paired_noise)
