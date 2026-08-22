import torch

from eval.vae_latent_stats import ChannelMoments


def test_channel_moments_matches_direct_statistics():
    value = torch.arange(2 * 3 * 2 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2, 2)
    moments = ChannelMoments(3)
    moments.update(value[:1])
    moments.update(value[1:])
    result = moments.result()
    flat = value.double().permute(1, 0, 2, 3, 4).flatten(1)
    torch.testing.assert_close(torch.tensor(result["mean"], dtype=torch.float64), flat.mean(1))
    torch.testing.assert_close(torch.tensor(result["std"], dtype=torch.float64), flat.std(1, correction=0))
    assert result["values_per_channel"] == flat.shape[1]
