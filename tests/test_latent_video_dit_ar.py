import torch
from types import SimpleNamespace

from train.latent_video_dit_ar import (
    LatentStandardizer,
    decode_full,
    decode_sliding,
    decoded_rgba_auxiliary_loss,
    flow_prediction_to_clean,
    m6_switch_prompt_grid,
    select_video_history_window,
    validation_losses,
    validate_experiment_protocol,
)
from train.video_vae import DSFCausalVideoVAE


def test_latent_standardizer_round_trip():
    standardizer = LatentStandardizer([1, 2], [2, 4], "cpu")
    value = torch.randn(1, 2, 3, 2, 2)
    torch.testing.assert_close(standardizer.decode(standardizer.encode(value)), value)


def test_validation_noise_and_timesteps_are_fixed_across_calls():
    class IdentityCodec:
        temporal_compression = 1

        @staticmethod
        def encode(video):
            return video, torch.zeros_like(video)

    class ZeroModel(torch.nn.Module):
        def forward(self, x, timestep, **kwargs):
            return torch.zeros_like(x)

    model = ZeroModel().train()
    standardizer = LatentStandardizer([0] * 4, [1] * 4, "cpu")
    video = torch.linspace(-1, 1, 4 * 3 * 2 * 2).view(1, 4, 3, 2, 2)
    loader = [(video, ["walk"])]
    args = SimpleNamespace(device="cpu", history_max=2, target_latents=1)

    def text_batch(labels):
        return torch.zeros(len(labels), 1, 2), torch.ones(len(labels), 1, dtype=torch.long)

    first = validation_losses(model, IdentityCodec(), standardizer, loader, text_batch, args)
    torch.randn(100)  # Advancing global RNG must not alter the fixed validation draw.
    second = validation_losses(model, IdentityCodec(), standardizer, loader, text_batch, args)
    assert first == second
    assert model.training


def test_validation_first_case_uses_dedicated_start_loader():
    class IdentityCodec:
        temporal_compression = 1

        @staticmethod
        def encode(video):
            return video, torch.zeros_like(video)

    class ZeroModel(torch.nn.Module):
        def forward(self, x, timestep, **kwargs):
            return torch.zeros_like(x)

    seen = []

    def text_batch(labels):
        seen.extend(labels)
        return torch.zeros(len(labels), 1, 2), torch.ones(len(labels), 1, dtype=torch.long)

    regular = [(torch.ones(1, 4, 3, 2, 2), ["continuation"])]
    initial = [(torch.zeros(1, 4, 3, 2, 2), ["initial"])]
    args = SimpleNamespace(device="cpu", history_max=2, target_latents=1)
    validation_losses(
        ZeroModel(), IdentityCodec(), LatentStandardizer([0] * 4, [1] * 4, "cpu"),
        regular, text_batch, args, batches=1, initial_loader=initial,
    )
    assert seen == ["initial", "continuation"]


def test_sliding_latent_decode_has_expected_video_length():
    codec = DSFCausalVideoVAE(temporal_compression=2, spatial_compression=8,
                              latent_channels=4, base_channels=4, blocks_per_stage=1).eval()
    standardizer = LatentStandardizer([0]*4, [1]*4, "cpu")
    latent = torch.randn(1, 4, 8, 2, 2)
    decoded = decode_sliding(codec, standardizer, latent, context_latents=4,
                             commit_latents=2, output_size=16)
    assert decoded.shape == (1, 4, 16, 16, 16)


def test_sliding_decode_allows_context_not_divisible_by_commit():
    codec = DSFCausalVideoVAE(temporal_compression=2, spatial_compression=8,
                              latent_channels=4, base_channels=4, blocks_per_stage=1).eval()
    standardizer = LatentStandardizer([0]*4, [1]*4, "cpu")
    # M6 horizon arms use H=5 with F=2 or F=10.  The initial context need
    # not be an integer number of commit blocks; only the remaining suffix
    # must partition into whole commits.
    latent = torch.randn(1, 4, 9, 2, 2)
    decoded = decode_sliding(codec, standardizer, latent, context_latents=5,
                             commit_latents=2, output_size=16)
    assert decoded.shape == (1, 4, 18, 16, 16)


def test_sliding_decode_allows_commit_larger_than_context():
    codec = DSFCausalVideoVAE(temporal_compression=4, spatial_compression=8,
                              latent_channels=4, base_channels=4, blocks_per_stage=1).eval()
    standardizer = LatentStandardizer([0]*4, [1]*4, "cpu")
    # H40 uses five past latents and generates ten new latents per call.  The
    # bounded decoder must emit every one of the 25 latents (100 video frames),
    # not silently truncate each ten-latent commit to its five-latent context.
    latent = torch.randn(1, 4, 25, 2, 2)
    decoded = decode_sliding(codec, standardizer, latent, context_latents=5,
                             commit_latents=10, output_size=16)
    assert decoded.shape == (1, 4, 100, 16, 16)


def test_full_latent_decode_preserves_complete_causal_sequence_length():
    codec = DSFCausalVideoVAE(temporal_compression=4, spatial_compression=8,
                              latent_channels=4, base_channels=4, blocks_per_stage=1).eval()
    standardizer = LatentStandardizer([0]*4, [1]*4, "cpu")
    latent = torch.randn(1, 4, 6, 2, 2)
    decoded = decode_full(codec, standardizer, latent, output_size=16)
    assert decoded.shape == (1, 4, 24, 16, 16)


def test_frozen_decoder_preserves_gradient_to_latent_input():
    codec = DSFCausalVideoVAE(temporal_compression=2, spatial_compression=8,
                              latent_channels=4, base_channels=4, blocks_per_stage=1).eval()
    # The production codec is trained, whereas a newly constructed test codec
    # deliberately zero-initializes its output projection.  Give the fixture a
    # nonzero output path so this test measures autograd through a frozen
    # decoder rather than the initialization policy.
    with torch.no_grad():
        codec.dec_out.conv.weight.normal_(mean=0.0, std=0.01)
    codec.requires_grad_(False)
    standardizer = LatentStandardizer([0]*4, [1]*4, "cpu")
    latent = torch.randn(1, 4, 2, 2, 2, requires_grad=True)
    decode_full(codec, standardizer, latent, output_size=16).sum().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert latent.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in codec.parameters())


def test_decoded_rgba_loss_weakens_but_retains_background_alpha_error():
    target = torch.zeros(1, 4, 1, 2, 2)
    target[:, :, :, 0, 0] = 1
    reconstruction = target.clone().requires_grad_(True)
    reconstruction.data[:, 3, :, 1, 1] = 1  # spurious opaque background
    weak = decoded_rgba_auxiliary_loss(
        reconstruction, target, background_alpha_weight=0.02,
    )
    full = decoded_rgba_auxiliary_loss(
        reconstruction, target, background_alpha_weight=1.0,
    )
    assert 0 < weak["alpha"] < full["alpha"]
    assert weak["rgb_foreground"] == 0
    weak["total"].backward()
    assert reconstruction.grad[:, 3, :, 1, 1].abs().sum() > 0


def test_flow_prediction_recovers_clean_target_and_preserves_teacher_forced_history():
    clean = torch.randn(2, 3, 4, 2, 2)
    noise = torch.randn_like(clean)
    timestep = torch.tensor([0.2, 0.8])
    amount = timestep[:, None, None, None, None]
    model_input = (1 - amount) * clean + amount * noise
    model_input[:, :, :2] = clean[:, :, :2]
    prediction = noise - clean
    # Deliberately corrupt history predictions: the exact teacher-forced
    # prefix, not a reconstructed approximation, must reach the decoder.
    prediction[:, :, :2] = 99
    recovered = flow_prediction_to_clean(
        model_input, prediction, timestep, clean_history=clean[:, :, :2],
    )
    torch.testing.assert_close(recovered, clean)


def test_zero_history_uses_true_clip_start_before_causal_encoding():
    video = torch.arange(24).view(1, 1, 24, 1, 1)
    first = select_video_history_window(
        video, history_latents=0, target_latents=2, history_max=10,
        temporal_compression=2, initial_block=True,
    )
    continuation = select_video_history_window(
        video, history_latents=10, target_latents=2, history_max=10,
        temporal_compression=2,
    )
    # An H=0 rollout is the start of a command/clip, not a random late crop.
    # Training it on the right edge makes pre-action and post-action idle
    # indistinguishable and caused the M6 baseline to freeze at inference.
    assert first.flatten().tolist() == [0, 1, 2, 3]
    torch.testing.assert_close(continuation, video)


def test_prompt_switch_schedule_uses_declared_block_boundary():
    rows = m6_switch_prompt_grid(blocks=25, switch_block=10)
    assert all(len(row) == 25 for row in rows)
    assert rows[0][9] == "A person walks forward."
    assert rows[0][10] == "A person runs forward."


def test_predeclared_protocol_rejects_launch_drift():
    codec = DSFCausalVideoVAE(temporal_compression=2, spatial_compression=8,
                              latent_channels=4, base_channels=4, blocks_per_stage=1)
    protocol = {
        "task": {"resolution": 64, "fps": 20, "generated_video_frames": 100},
        "frozen_model": {"target_video_frames_per_block": 4, "history_seconds": 1,
                         "primary_sampling_steps_per_block": 10,
                         "planned_optimizer_steps": 10000, "training_seed": 0},
    }
    args = SimpleNamespace(output_size=64, fps=20, rollout_latents=50,
                           target_latents=2, history_max=10, sample_steps=10,
                           steps=10000, seed=0)
    validate_experiment_protocol(protocol, args, codec)
    args.sample_steps = 20
    try:
        validate_experiment_protocol(protocol, args, codec)
    except ValueError as error:
        assert "sample_steps" in str(error)
    else:
        raise AssertionError("protocol drift must fail before training")


def test_protocol_allows_only_bounded_no_preview_smoke():
    codec = DSFCausalVideoVAE(temporal_compression=2, spatial_compression=8,
                              latent_channels=4, base_channels=4, blocks_per_stage=1)
    protocol = {
        "task": {"resolution": 64, "fps": 20, "generated_video_frames": 100},
        "frozen_model": {"target_video_frames_per_block": 4, "history_seconds": 1,
                         "primary_sampling_steps_per_block": 10,
                         "planned_optimizer_steps": 10000, "training_seed": 0},
    }
    args = SimpleNamespace(output_size=64, fps=20, rollout_latents=50,
                           target_latents=2, history_max=10, sample_steps=10,
                           steps=2, seed=0, no_previews=True)
    validate_experiment_protocol(protocol, args, codec, smoke=True)
    args.no_previews = False
    try:
        validate_experiment_protocol(protocol, args, codec, smoke=True)
    except ValueError as error:
        assert "smoke" in str(error)
    else:
        raise AssertionError("smoke previews should be rejected")


def test_v2_protocol_requires_full_spatiotemporal_attention():
    codec = DSFCausalVideoVAE(temporal_compression=4, spatial_compression=8,
                              latent_channels=4, base_channels=4, blocks_per_stage=1)
    protocol = {
        "task": {"resolution": 64, "fps": 20, "generated_video_frames": 100},
        "frozen_model": {"target_video_frames_per_block": 4, "history_seconds": 1,
                         "primary_sampling_steps_per_block": 10,
                         "planned_optimizer_steps": 10000, "training_seed": 0,
                         "attention_mode": "full"},
    }
    args = SimpleNamespace(output_size=64, fps=20, rollout_latents=25,
                           target_latents=1, history_max=5, sample_steps=10,
                           steps=10000, seed=0, attention_mode="full")
    validate_experiment_protocol(protocol, args, codec)
    args.attention_mode = "factorized"
    try:
        validate_experiment_protocol(protocol, args, codec)
    except ValueError as error:
        assert "attention_mode" in str(error)
    else:
        raise AssertionError("v2 launch must not silently revert to factorized attention")


def test_start_aligned_protocol_requires_explicit_training_flag():
    codec = DSFCausalVideoVAE(temporal_compression=4, spatial_compression=8,
                              latent_channels=4, base_channels=4, blocks_per_stage=1)
    protocol = {
        "task": {"resolution": 64, "fps": 20, "generated_video_frames": 100},
        "frozen_model": {"target_video_frames_per_block": 4, "history_seconds": 1,
                         "primary_sampling_steps_per_block": 10,
                         "planned_optimizer_steps": 2000, "training_seed": 0,
                         "attention_mode": "full", "training_mode": "block_ar",
                         "start_aligned": True},
    }
    args = SimpleNamespace(output_size=64, fps=20, rollout_latents=25,
                           target_latents=1, history_max=5, sample_steps=10,
                           steps=2000, seed=0, attention_mode="full",
                           training_mode="block_ar", start_aligned=True)
    validate_experiment_protocol(protocol, args, codec)
    args.start_aligned = False
    try:
        validate_experiment_protocol(protocol, args, codec)
    except ValueError as error:
        assert "start_aligned" in str(error)
    else:
        raise AssertionError("start-aligned protocol must not silently use late H=0 crops")


def test_r0_protocol_is_same_full_st_model_without_ar_history():
    codec = DSFCausalVideoVAE(temporal_compression=4, spatial_compression=8,
                              latent_channels=4, base_channels=4, blocks_per_stage=1)
    protocol = {
        "task": {"resolution": 64, "fps": 20, "generated_video_frames": 100},
        "frozen_model": {"target_video_frames_per_block": 100, "history_seconds": 0,
                         "primary_sampling_steps_per_block": 10,
                         "planned_optimizer_steps": 10000, "training_seed": 0,
                         "attention_mode": "full", "training_mode": "full_clip"},
    }
    args = SimpleNamespace(output_size=64, fps=20, rollout_latents=25,
                           target_latents=25, history_max=0, sample_steps=10,
                           steps=10000, seed=0, attention_mode="full",
                           training_mode="full_clip")
    validate_experiment_protocol(protocol, args, codec)
    args.training_mode = "block_ar"
    try:
        validate_experiment_protocol(protocol, args, codec)
    except ValueError as error:
        assert "training_mode" in str(error)
    else:
        raise AssertionError("R0 control must remain a full-clip training run")


def test_foreground_latent_weight_upweights_figure_cells_only():
    from train.latent_video_dit_ar import foreground_latent_weight

    video = torch.full((2, 4, 8, 16, 16), -1.0)
    # one thin 1-px limb inside the first spatial cell of the target segment
    video[:, 3, 4:, 3, 0:2] = 1.0
    weight = foreground_latent_weight(video, history=1, temporal=4, spatial=8, fg_weight=4.0)
    assert weight.shape == (2, 1, 1, 2, 2)
    # figure cell strictly above background cells despite covering 2/256 pixels
    assert (weight[:, :, :, 0, 0] > weight[:, :, :, 1, 1]).all()
    # per-sample mean 1: pure redistribution, no scale change
    assert torch.allclose(weight.mean(dim=(2, 3, 4)), torch.ones(2, 1), atol=1e-5)
    # empty background video degrades to uniform weight 1
    uniform = foreground_latent_weight(
        torch.full((1, 4, 8, 16, 16), -1.0), history=0, temporal=4, spatial=8, fg_weight=4.0,
    )
    assert torch.allclose(uniform, torch.ones_like(uniform), atol=1e-5)


def test_fg_weighted_protocol_id_is_single_variable_and_start_aligned():
    from train.latent_video_dit_ar import _protocol_id

    assert _protocol_id("full", "block_ar", True, fg_latent_weight=4.0) == \
        "m6_latent_block_ar_v7_fg_weighted"
    assert _protocol_id("full", "block_ar", True, fg_latent_weight=1.0) == \
        "m6_latent_block_ar_v3_start_aligned"
    for kwargs in (
        dict(start_aligned=False),
        dict(start_aligned=True, motion_weight_alpha=1.0),
        dict(start_aligned=True, history_noise_max=0.2),
        dict(start_aligned=True, decoded_loss_weight=0.1),
    ):
        try:
            _protocol_id("full", "block_ar", fg_latent_weight=4.0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("fg-weighted treatment must stay single-variable and start-aligned")
