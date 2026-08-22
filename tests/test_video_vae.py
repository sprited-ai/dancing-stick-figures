import torch

from train.video_vae import (
    CausalConv3d,
    DSFCausalVideoVAE,
    matched_variant_state,
    video_vae_loss,
)


def _tiny(temporal_compression=4, spatial_compression=4):
    return DSFCausalVideoVAE(
        temporal_compression=temporal_compression,
        spatial_compression=spatial_compression,
        latent_channels=4,
        base_channels=4,
        blocks_per_stage=1,
    )


def _rgba(batch=1, frames=9, size=8):
    alpha = torch.zeros(batch, 1, frames, size, size)
    alpha[:, :, :, 2:6, 2:6] = 1
    colour = torch.rand(batch, 3, frames, size, size)
    return torch.cat((colour * alpha, alpha), dim=1)


def test_causal_conv_future_perturbation_cannot_change_past():
    conv = CausalConv3d(2, 3, 3)
    x = torch.randn(1, 2, 8, 7, 7)
    changed = x.clone()
    changed[:, :, 5:] += torch.randn_like(changed[:, :, 5:]) * 10
    torch.testing.assert_close(conv(x)[:, :, :5], conv(changed)[:, :, :5])


def test_t1_t2_and_t4_have_expected_shapes_and_exact_output_crop():
    x = _rgba(frames=9)
    for compression, expected_t in ((1, 9), (2, 5), (4, 3)):
        model = _tiny(compression)
        mean, _ = model.encode(x)
        assert mean.shape == (1, 4, expected_t, 2, 2)
        out = model(x, sample=False)
        assert out.reconstruction.shape == x.shape
        assert torch.isfinite(out.reconstruction).all()
        assert torch.all((out.reconstruction >= 0) & (out.reconstruction <= 1))


def test_f8t2_has_expected_latent_shape_and_exact_reconstruction_shape():
    x = _rgba(frames=9, size=16)
    model = _tiny(2, spatial_compression=8)
    mean, _ = model.encode(x)
    assert mean.shape == (1, 4, 5, 2, 2)
    assert model(x, sample=False).reconstruction.shape == x.shape


def test_f8_decoder_refines_at_f8_then_f4_resolution():
    model = _tiny(2, spatial_compression=8).eval()
    observed = {}

    def record(name):
        def hook(_module, args):
            observed[name] = args[0].shape[-2:]
        return hook

    hooks = [
        model.dec_stage3.register_forward_pre_hook(record("f8")),
        model.dec_stage2.register_forward_pre_hook(record("f4")),
    ]
    try:
        model.decode(torch.randn(1, 4, 3, 2, 2), output_size=(16, 16))
    finally:
        for hook in hooks:
            hook.remove()
    assert observed == {"f8": (2, 2), "f4": (4, 4)}


def test_encoder_is_causal_across_full_stack():
    model = _tiny(4).eval()
    x = _rgba(frames=12)
    changed = x.clone()
    changed[:, :, 8:] = torch.rand_like(changed[:, :, 8:])
    mean, _ = model.encode(x)
    changed_mean, _ = model.encode(changed)
    # f4 temporal positions before frame 8 are latent indices 0 and 1.
    torch.testing.assert_close(mean[:, :, :2], changed_mean[:, :, :2], atol=1e-6, rtol=1e-5)


def test_compressed_latent_sees_its_complete_block_but_not_the_next_block():
    for compression in (2, 4):
        model = _tiny(compression).eval()
        x = _rgba(frames=8)
        base, _ = model.encode(x)

        changed_last = x.clone()
        changed_last[:, :, compression - 1] = torch.rand_like(changed_last[:, :, compression - 1])
        sees_block_end, _ = model.encode(changed_last)
        assert not torch.allclose(base[:, :, 0], sees_block_end[:, :, 0])

        changed_next = x.clone()
        changed_next[:, :, compression] = torch.rand_like(changed_next[:, :, compression])
        ignores_next_block, _ = model.encode(changed_next)
        torch.testing.assert_close(base[:, :, 0], ignores_next_block[:, :, 0], atol=1e-6, rtol=1e-5)


def test_decoder_is_causal_across_full_stack():
    model = _tiny(4).eval()
    latent = torch.randn(1, 4, 4, 2, 2)
    changed = latent.clone()
    changed[:, :, 2:] += torch.randn_like(changed[:, :, 2:]) * 10
    decoded = model.decode(latent)
    changed_decoded = model.decode(changed)
    # Two untouched t4 latent positions decode to the first eight frames.
    torch.testing.assert_close(decoded[:, :, :8], changed_decoded[:, :, :8], atol=1e-6, rtol=1e-5)


def test_decoder_enforces_premultiplied_rgba():
    model = _tiny(4).eval()
    rgba = model.decode(torch.randn(1, 4, 2, 2, 2))
    assert torch.all(rgba[:, :3] <= rgba[:, 3:4] + 1e-7)


def test_matched_t1_t2_t4_variants_have_identical_parameter_shapes_and_state():
    t1, t2, t4 = _tiny(1), _tiny(2), _tiny(4)
    shapes = [{k: v.shape for k, v in model.state_dict().items()} for model in (t1, t2, t4)]
    assert shapes[0] == shapes[1] == shapes[2]
    matched_variant_state(t1, t2)
    matched_variant_state(t1, t4)
    for key, value in t1.state_dict().items():
        torch.testing.assert_close(value, t2.state_dict()[key])
        torch.testing.assert_close(value, t4.state_dict()[key])


def test_background_area_does_not_change_foreground_rgb_loss_scale():
    model = _tiny(4)
    small = _rgba(frames=5, size=8)
    large = torch.zeros(1, 4, 5, 16, 16)
    large[:, :, :, 4:12, 4:12] = small

    def fake_output(target):
        output = model(target, sample=False)
        # Controlled equal foreground error, independent of background area.
        reconstruction = target.clone()
        reconstruction[:, :3] = (target[:, :3] + 0.1 * target[:, 3:4]).clamp(0, 1)
        output.reconstruction = reconstruction
        return output

    motion_off = dict(
        rgb_velocity_weight=0,
        alpha_velocity_weight=0,
        rgb_acceleration_weight=0,
        alpha_acceleration_weight=0,
    )
    small_loss = video_vae_loss(fake_output(small), small, **motion_off)["rgb_fg"]
    large_loss = video_vae_loss(fake_output(large), large, **motion_off)["rgb_fg"]
    torch.testing.assert_close(small_loss, large_loss)


def test_antialiased_rgb_support_is_not_downweighted_by_alpha_twice():
    model = _tiny(4)
    motion_off = dict(
        rgb_velocity_weight=0,
        alpha_velocity_weight=0,
        rgb_acceleration_weight=0,
        alpha_acceleration_weight=0,
    )

    def controlled(alpha_value):
        target = torch.zeros(1, 4, 1, 8, 8)
        target[:, 3:4, :, 2:6, 2:6] = alpha_value
        target[:, :3, :, 2:6, 2:6] = 0.5 * alpha_value
        output = model(target, sample=False)
        output.reconstruction = target.clone()
        output.reconstruction[:, :3, :, 2:6, 2:6] += 0.05
        return video_vae_loss(output, target, **motion_off)["rgb_fg"]

    torch.testing.assert_close(controlled(0.25), controlled(1.0))


def test_rgb_and_alpha_reconstruction_losses_are_independent():
    model = _tiny(4)
    target = _rgba(frames=1)
    base = model(target, sample=False)
    motion_off = dict(
        rgb_velocity_weight=0,
        alpha_velocity_weight=0,
        rgb_acceleration_weight=0,
        alpha_acceleration_weight=0,
    )

    rgb_only = target.clone()
    rgb_only[:, :3] = (rgb_only[:, :3] + 0.1 * target[:, 3:4]).clamp(0, 1)
    base.reconstruction = rgb_only
    rgb_losses = video_vae_loss(base, target, **motion_off)
    assert rgb_losses["rgb_fg"] > 0
    assert rgb_losses["alpha_fg"] == 0
    assert rgb_losses["alpha_bg"] == 0

    alpha_only = target.clone()
    alpha_only[:, 3:4] = (alpha_only[:, 3:4] * 0.8 + 0.05).clamp(0, 1)
    base.reconstruction = alpha_only
    alpha_losses = video_vae_loss(base, target, **motion_off)
    assert alpha_losses["rgb_fg"] == 0
    assert alpha_losses["alpha_fg"] > 0
    assert alpha_losses["alpha_bg"] > 0


def test_loss_is_finite_and_backpropagates():
    model = _tiny(4)
    x = _rgba(frames=9)
    losses = video_vae_loss(model(x), x, kl_weight=3e-6)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
