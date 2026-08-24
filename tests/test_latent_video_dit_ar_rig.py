import torch

from train.latent_video_dit_ar_rig import (
    RigFullSTARVideoDiT,
    make_joint_flow_batch,
    rig_condition,
    rig_tokens_from_frames,
    select_rig_history_window,
)


def test_rig_window_and_tokens_align_with_video_selection():
    rig = torch.rand(2, 36, 27, 2)  # (history_max 5 + target 4) * temporal 4 frames
    window = select_rig_history_window(rig, history_latents=3, target_latents=4,
                                       history_max=5, temporal_compression=4)
    assert window.shape == (2, 28, 27, 2)
    assert torch.equal(window, rig[:, -28:])
    tokens = rig_tokens_from_frames(window, 4)
    assert tokens.shape == (2, 7, 216)
    first = select_rig_history_window(rig, history_latents=0, target_latents=4,
                                      history_max=5, temporal_compression=4, initial_block=True)
    assert torch.equal(first, rig[:, :16])


def test_joint_flow_shares_timestep_and_freezes_history():
    clean = torch.randn(3, 16, 6, 8, 8)
    rig_clean = torch.randn(3, 6, 216)
    mi, ft, ri, rt, t = make_joint_flow_batch(clean, rig_clean, history=2, shift=1.0)
    assert t.shape == (3,)
    assert torch.equal(mi[:, :, :2], clean[:, :, :2])
    assert torch.equal(ri[:, :2], rig_clean[:, :2])
    assert ft.shape == clean.shape and rt.shape == rig_clean.shape
    cond = rig_condition(rig_clean, 2)
    assert cond.shape == (3, 6, 217)
    assert torch.equal(cond[:, :2, :216], rig_clean[:, :2])
    assert cond[:, 2:, 216].eq(0).all() and cond[:, :2, 216].eq(1).all()


def test_rig_model_forward_shapes_and_history_immutability():
    torch.manual_seed(0)
    model = RigFullSTARVideoDiT(temporal_compression=4, size=8, patch=1, in_ch=16,
                                dim=48, depth=2, heads=4, cond_ch=17, text_dim=32).eval()
    B, F = 2, 5
    x = torch.randn(B, 16, F, 8, 8)
    rig = torch.randn(B, F, model.rig_dim)
    text = torch.randn(B, 4, 32)
    mask = torch.ones(B, 4)
    cond = torch.zeros(B, 17, F, 8, 8)
    rcond = torch.zeros(B, F, model.rig_dim + 1)
    with torch.no_grad():
        vel, rvel = model(x, torch.rand(B), cond=cond, text=text, text_mask=mask,
                          history_frames=2, rig_input=rig, rig_cond=rcond)
        assert vel.shape == x.shape and rvel.shape == rig.shape
        # history queries must not see target tokens: perturbing the target
        # (pixels AND rig) must leave history-token outputs unchanged
        x2 = x.clone(); x2[:, :, 2:] += 1.0
        rig2 = rig.clone(); rig2[:, 2:] += 1.0
        vel2, rvel2 = model(x2, torch.rand(B) * 0 + 0.5, cond=cond, text=text, text_mask=mask,
                            history_frames=2, rig_input=rig2, rig_cond=rcond)
        vel1, rvel1 = model(x, torch.rand(B) * 0 + 0.5, cond=cond, text=text, text_mask=mask,
                            history_frames=2, rig_input=rig, rig_cond=rcond)
        assert torch.allclose(vel1[:, :, :2], vel2[:, :, :2], atol=1e-5)
        assert torch.allclose(rvel1[:, :2], rvel2[:, :2], atol=1e-5)


def test_bone_lengths_from_rig_tokens_match_direct_computation():
    from train.latent_video_dit_ar_rig import RIG_PARENTS, rig_bone_lengths

    rig = torch.rand(2, 3, 4 * 27 * 2) * 2 - 1
    bones = rig_bone_lengths(rig, 4)
    assert bones.shape == (2, 12, 26)
    xy = rig.reshape(2, 12, 27, 2)
    manual = (xy[:, :, 5] - xy[:, :, RIG_PARENTS[5]]).norm(dim=-1)
    assert torch.allclose(bones[:, :, 4], manual, atol=1e-6)


def test_fg_weighted_alpha_mse_ignores_background_agreement():
    from train.latent_video_dit_ar_rig import fg_weighted_alpha_mse

    # identical maps -> zero loss regardless of foreground size
    a = torch.zeros(2, 4, 16, 16); a[:, :, 5, 5] = 1.0
    assert float(fg_weighted_alpha_mse(a, a.clone())) == 0.0
    # a misplaced small figure must not be washed out by empty background:
    # plain MSE would be ~2/256, the fg-weighted loss stays near 1
    b = torch.zeros_like(a); b[:, :, 10, 10] = 1.0
    plain = float((a - b).square().mean())
    weighted = float(fg_weighted_alpha_mse(a, b))
    assert weighted > 0.2 and weighted > plain * 10
