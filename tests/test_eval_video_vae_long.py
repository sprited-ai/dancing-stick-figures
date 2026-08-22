import torch

from eval.eval_video_vae_long import sliding_reconstruct, strict_causal_prefix_error, transition_error


def test_transition_error_is_zero_for_matching_motion():
    video = torch.rand(2, 4, 8, 5, 5)
    assert torch.equal(transition_error(video, video), torch.zeros(7))


def test_transition_error_detects_a_single_bad_transition():
    target = torch.zeros(1, 4, 6, 2, 2)
    prediction = target.clone()
    prediction[:, :, 3:] = 1
    error = transition_error(prediction, target)
    assert error.argmax().item() == 2
    assert error[2].item() == 1


class _IdentityModel:
    def __call__(self, value, sample=False):
        class Output:
            reconstruction = value
        return Output()


def test_sliding_reconstruct_commits_each_frame_once():
    video = torch.arange(24.0).reshape(1, 1, 24, 1, 1)
    result = sliding_reconstruct(_IdentityModel(), video, window=8, commit=4)
    assert torch.equal(result, video)


def test_sliding_reconstruct_rejects_partial_commit():
    video = torch.zeros(1, 1, 23, 1, 1)
    try:
        sliding_reconstruct(_IdentityModel(), video, window=8, commit=4)
    except ValueError as error:
        assert "divide evenly" in str(error)
    else:
        raise AssertionError("expected invalid sequence length to fail")


def test_strict_causal_prefix_check_disables_tf32_and_restores_setting():
    seen = []

    class CausalIdentity:
        def __call__(self, value, sample=False):
            seen.append(torch.backends.cudnn.allow_tf32)
            class Output:
                reconstruction = value
            return Output()

    previous = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = True
    try:
        video = torch.rand(1, 4, 12, 2, 2)
        assert strict_causal_prefix_error(CausalIdentity(), video, prefix_frames=4) == 0.0
        assert seen == [False, False]
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cudnn.allow_tf32 = previous
