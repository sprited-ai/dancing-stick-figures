"""Small causal Video-VAE for the Dancing Stick Figures latent track.

The model is deliberately domain-specific and easy to audit:

* premultiplied RGBA input/output in ``[0, 1]``;
* two or three spatial 2x downsamples (``f4`` or ``f8``);
* matched ``t1``/``t2``/``t4`` variants which differ only in the stride/upsample
  factor of one temporal stage, keeping parameter names and shapes identical;
* causal 3D convolutions and channel-only RMS normalization throughout; and
* a background-safe reconstruction objective which never trains against the
  white display composite.

The default configuration is DSF-CVAE-f4t4d32.  The higher-compression
``f8t2d32`` variant adds one spatial stage while retaining two-frame temporal
blocks.  This module contains the model
and loss only; experiment orchestration is kept separate so reconstruction can
be validated before any latent diffusion training begins.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelRMSNorm3D(nn.Module):
    """RMS-normalize channels independently at every ``(t, y, x)`` location.

    ``GroupNorm`` over a 5D video tensor aggregates temporal/spatial positions
    and therefore leaks future statistics.  This normalization only reduces
    over the channel dimension and preserves strict temporal causality.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=1, keepdim=True) + self.eps).to(x.dtype)
        return x * scale * self.weight[None, :, None, None, None].to(x.dtype)


class CausalConv3d(nn.Module):
    """Conv3d with left-only temporal padding and symmetric spatial padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        kt, kh, kw = kernel_size
        if kh % 2 != 1 or kw % 2 != 1:
            raise ValueError("spatial kernels must be odd for symmetric padding")
        self.temporal_pad = kt - 1
        self.spatial_pad = (kw // 2, kh // 2)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pw, ph = self.spatial_pad
        # F.pad order for [B,C,T,H,W]: W, H, T.
        x = F.pad(x, (pw, pw, ph, ph, self.temporal_pad, 0))
        return self.conv(x)


class CausalResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = ChannelRMSNorm3D(in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3)
        self.norm2 = ChannelRMSNorm3D(out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3)
        self.skip = CausalConv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class CausalUpsample3D(nn.Module):
    """Nearest-neighbour upsampling followed by a causal smoothing convolution."""

    def __init__(self, in_channels: int, out_channels: int, scale: tuple[int, int, int]):
        super().__init__()
        self.scale = scale
        self.conv = CausalConv3d(in_channels, out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        st, sh, sw = self.scale
        if st != 1:
            x = x.repeat_interleave(st, dim=2)
        if sh != 1 or sw != 1:
            x = F.interpolate(x, scale_factor=(1, sh, sw), mode="nearest")
        return self.conv(x)


@dataclass
class VideoVAEOutput:
    reconstruction: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor
    latent: torch.Tensor


class DSFCausalVideoVAE(nn.Module):
    """Small f{4,8}-t{1,2,4}-dN causal Video-VAE.

    Inputs and reconstructions are premultiplied RGBA tensors in ``[0, 1]``
    with shape ``[B,4,T,H,W]``.  Decoder RGB is parameterized as
    ``sigmoid(colour) * sigmoid(alpha)`` so transparent RGB is zero by
    construction.
    """

    def __init__(
        self,
        *,
        temporal_compression: int = 4,
        spatial_compression: int = 4,
        latent_channels: int = 32,
        base_channels: int = 32,
        blocks_per_stage: int = 2,
    ):
        super().__init__()
        if temporal_compression not in (1, 2, 4):
            raise ValueError("temporal_compression must be 1, 2, or 4")
        if spatial_compression not in (2, 4, 8):
            raise ValueError("spatial_compression must be 2, 4, or 8")
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be positive")
        self.temporal_compression = temporal_compression
        self.spatial_compression = spatial_compression
        self.latent_channels = latent_channels
        c0, c1, c2 = base_channels, 2 * base_channels, 4 * base_channels

        def blocks(channels: int) -> nn.Sequential:
            return nn.Sequential(*(CausalResBlock(channels, channels) for _ in range(blocks_per_stage)))

        # Encoder: f1t1 -> f2t1 -> f4t{1,2} -> f4t{1,2,4}.
        self.enc_stem = CausalConv3d(4, c0, 3)
        self.enc_stage0 = blocks(c0)
        # f2 keeps full resolution through the first transition (stride 1);
        # the spatiotemporal stage below then provides the single 2x reduction.
        first_spatial = 1 if spatial_compression == 2 else 2
        self.enc_down_spatial = CausalConv3d(c0, c1, 3, stride=(1, first_spatial, first_spatial))
        self.enc_stage1 = blocks(c1)
        first_temporal_factor = 1 if temporal_compression == 1 else 2
        self.enc_down_spatiotemporal = CausalConv3d(
            c1, c2, 3, stride=(1, 2, 2)
        )
        self.enc_stage2 = blocks(c2)
        # f8 adds a third 2x spatial reduction without increasing the channel
        # width.  This keeps d32 and the temporal path directly interpretable
        # while reducing M6's spatial token grid from 16x16 to 8x8.
        self.enc_down_spatial_8 = (
            CausalConv3d(c2, c2, (1, 3, 3), stride=(1, 2, 2))
            if spatial_compression == 8 else nn.Identity()
        )
        self.enc_stage3 = blocks(c2) if spatial_compression == 8 else nn.Identity()
        # Temporal reduction is applied explicitly at block ends in encode().
        # A strided causal convolution starts at frame zero and would force the
        # decoder to predict the unseen remainder of every block.
        self.enc_down_temporal = CausalConv3d(c2, c2, 3, stride=1)
        self.enc_mid = blocks(c2)
        self.enc_norm = ChannelRMSNorm3D(c2)
        self.to_mean = CausalConv3d(c2, latent_channels, 1)
        self.to_logvar = CausalConv3d(c2, latent_channels, 1)

        # Decoder mirrors the encoder.  t1/t2/t4 have identical learned tensors.
        self.dec_in = CausalConv3d(latent_channels, c2, 1)
        self.dec_mid = blocks(c2)
        extra_temporal_scale = 2 if temporal_compression == 4 else 1
        self.dec_up_temporal = CausalUpsample3D(c2, c2, (extra_temporal_scale, 1, 1))
        self.dec_stage2 = blocks(c2)
        self.dec_stage3 = blocks(c2) if spatial_compression == 8 else nn.Identity()
        self.dec_up_spatial_8 = (
            CausalUpsample3D(c2, c2, (1, 2, 2))
            if spatial_compression == 8 else nn.Identity()
        )
        self.dec_up_spatiotemporal = CausalUpsample3D(
            c2, c1, (first_temporal_factor, 2, 2)
        )
        self.dec_stage1 = blocks(c1)
        self.dec_up_spatial = CausalUpsample3D(c1, c0, (1, first_spatial, first_spatial))
        self.dec_stage0 = blocks(c0)
        self.dec_norm = ChannelRMSNorm3D(c0)
        self.dec_out = CausalConv3d(c0, 4, 3)
        # A transparent prior avoids beginning reconstruction at 50% opacity.
        # RGB is still unconstrained inside visible regions, while premultiplication
        # guarantees zero RGB wherever the predicted alpha is zero.
        nn.init.zeros_(self.dec_out.conv.weight)
        nn.init.zeros_(self.dec_out.conv.bias)
        self.dec_out.conv.bias.data[3] = math.log(0.05 / 0.95)

    @staticmethod
    def _take_block_ends(x: torch.Tensor, factor: int) -> torch.Tensor:
        """Keep block-end activations, padding only a final incomplete block.

        For factor four, latent ``k`` has observed input frames
        ``4k..4k+3`` before it is emitted. This is block-causal with a bounded
        three-frame encoder latency, rather than frame-causal prediction of
        unseen frames.
        """
        if factor == 1:
            return x
        pad = (-x.shape[2]) % factor
        if pad:
            x = torch.cat((x, x[:, :, -1:].expand(-1, -1, pad, -1, -1)), dim=2)
        return x[:, :, factor - 1 :: factor]

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5 or x.shape[1] != 4:
            raise ValueError("expected premultiplied RGBA [B,4,T,H,W]")
        h = self.enc_stage0(self.enc_stem(x))
        h = self.enc_stage1(self.enc_down_spatial(h))
        h = self.enc_down_spatiotemporal(h)
        h = self._take_block_ends(h, 1 if self.temporal_compression == 1 else 2)
        h = self.enc_stage2(h)
        h = self.enc_stage3(self.enc_down_spatial_8(h))
        h = self.enc_down_temporal(h)
        h = self._take_block_ends(h, 2 if self.temporal_compression == 4 else 1)
        h = self.enc_mid(h)
        h = F.silu(self.enc_norm(h))
        return self.to_mean(h), self.to_logvar(h).clamp(-20.0, 10.0)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor, *, sample: bool) -> torch.Tensor:
        if not sample:
            return mean
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def decode(
        self,
        latent: torch.Tensor,
        *,
        output_frames: int | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        h = self.dec_mid(self.dec_in(latent))
        # Mirror the encoder: refine at f8, upsample to f4, then refine at f4.
        # Keeping both stages before the f8->f4 upsample would leave the decoder
        # without learned processing at the intermediate spatial resolution.
        h = self.dec_stage3(self.dec_up_temporal(h))
        h = self.dec_stage2(self.dec_up_spatial_8(h))
        h = self.dec_stage1(self.dec_up_spatiotemporal(h))
        h = self.dec_stage0(self.dec_up_spatial(h))
        logits = self.dec_out(F.silu(self.dec_norm(h)))
        straight_rgb = torch.sigmoid(logits[:, :3])
        alpha = torch.sigmoid(logits[:, 3:4])
        rgba = torch.cat((straight_rgb * alpha, alpha), dim=1)
        if output_frames is not None:
            rgba = rgba[:, :, :output_frames]
        if output_size is not None:
            rgba = rgba[:, :, :, : output_size[0], : output_size[1]]
        return rgba

    def forward(self, x: torch.Tensor, *, sample: bool = True) -> VideoVAEOutput:
        mean, logvar = self.encode(x)
        latent = self.reparameterize(mean, logvar, sample=sample)
        reconstruction = self.decode(latent, output_frames=x.shape[2], output_size=x.shape[-2:])
        return VideoVAEOutput(reconstruction, mean, logvar, latent)


def _masked_l1(error: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per-region mean; region area cannot change the scale of the loss."""
    if weight.shape[1] != 1:
        raise ValueError("mask must have one channel")
    channels = error.shape[1]
    return (error * weight).sum() / (weight.sum() * channels).clamp_min(1e-8)


def _temporal_region_losses(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if order not in (1, 2):
        raise ValueError("temporal difference order must be 1 or 2")
    if target.shape[2] <= order:
        zero = target.new_zeros(())
        return zero, zero
    if order == 1:
        pred_diff = reconstruction[:, :, 1:] - reconstruction[:, :, :-1]
        target_diff = target[:, :, 1:] - target[:, :, :-1]
        mask = torch.maximum(target[:, 3:4, 1:], target[:, 3:4, :-1])
    else:
        pred_diff = reconstruction[:, :, 2:] - 2 * reconstruction[:, :, 1:-1] + reconstruction[:, :, :-2]
        target_diff = target[:, :, 2:] - 2 * target[:, :, 1:-1] + target[:, :, :-2]
        mask = torch.maximum(
            torch.maximum(target[:, 3:4, 2:], target[:, 3:4, 1:-1]),
            target[:, 3:4, :-2],
        )
    error = (pred_diff - target_diff).abs()
    return _masked_l1(error[:, :3], mask), _masked_l1(error[:, 3:4], mask)


def video_vae_loss(
    output: VideoVAEOutput,
    target: torch.Tensor,
    *,
    alpha_background_weight: float = 1.0,
    rgb_velocity_weight: float = 0.0,
    alpha_velocity_weight: float = 0.0,
    rgb_acceleration_weight: float = 0.0,
    alpha_acceleration_weight: float = 0.0,
    kl_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Background-safe VAE objective for premultiplied RGBA videos.

    Transparent RGB receives no reconstruction loss.  Foreground and
    background alpha are normalized separately, so the large transparent area
    cannot dominate merely by containing more pixels.
    """
    reconstruction = output.reconstruction
    if reconstruction.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(reconstruction.shape)} vs {tuple(target.shape)}")
    weights = (
        alpha_background_weight,
        rgb_velocity_weight,
        alpha_velocity_weight,
        rgb_acceleration_weight,
        alpha_acceleration_weight,
        kl_weight,
    )
    if any(weight < 0 for weight in weights):
        raise ValueError("loss weights must be non-negative")

    alpha = target[:, 3:4].clamp(0, 1)
    inv_alpha = 1 - alpha
    # RGB is already premultiplied. Multiplying its error by alpha again would
    # suppress anti-aliased edges as alpha squared. Use a binary target-support
    # mask instead: every visible edge pixel counts once, while the much larger
    # transparent background remains the responsibility of alpha_bg.
    rgb_error = (reconstruction[:, :3] - target[:, :3]).abs()
    rgb_support = (alpha > 0).to(rgb_error.dtype)
    rgb_fg = _masked_l1(rgb_error, rgb_support)
    alpha_error = (reconstruction[:, 3:4] - alpha).abs()
    alpha_fg = _masked_l1(alpha_error, alpha)
    alpha_bg = _masked_l1(alpha_error, inv_alpha)
    rgb_velocity, alpha_velocity = _temporal_region_losses(reconstruction, target, 1)
    rgb_acceleration, alpha_acceleration = _temporal_region_losses(reconstruction, target, 2)
    kl = 0.5 * (output.mean.square() + output.logvar.exp() - 1 - output.logvar).mean()
    alpha_pred_mean = reconstruction[:, 3:4].mean()
    alpha_target_mean = alpha.mean()
    alpha_pred_fg = (reconstruction[:, 3:4] * alpha).sum() / alpha.sum().clamp_min(1e-8)
    alpha_pred_bg = (reconstruction[:, 3:4] * inv_alpha).sum() / inv_alpha.sum().clamp_min(1e-8)
    total = (
        rgb_fg
        + 0.5 * (alpha_fg + alpha_background_weight * alpha_bg)
        + rgb_velocity_weight * rgb_velocity
        + alpha_velocity_weight * alpha_velocity
        + rgb_acceleration_weight * rgb_acceleration
        + alpha_acceleration_weight * alpha_acceleration
        + kl_weight * kl
    )
    return {
        "total": total,
        "rgb_fg": rgb_fg,
        "alpha_fg": alpha_fg,
        "alpha_bg": alpha_bg,
        "rgb_velocity": rgb_velocity,
        "alpha_velocity": alpha_velocity,
        "rgb_acceleration": rgb_acceleration,
        "alpha_acceleration": alpha_acceleration,
        "kl": kl,
        "alpha_pred_mean": alpha_pred_mean,
        "alpha_target_mean": alpha_target_mean,
        "alpha_pred_fg": alpha_pred_fg,
        "alpha_pred_bg": alpha_pred_bg,
    }


def matched_variant_state(source: DSFCausalVideoVAE, target: DSFCausalVideoVAE) -> None:
    """Copy an exact initialization/checkpoint between matched t2/t4 models."""
    if source.latent_channels != target.latent_channels:
        raise ValueError("matched variants require identical latent channel counts")
    if source.spatial_compression != target.spatial_compression:
        raise ValueError("matched variants require identical spatial compression")
    target.load_state_dict(source.state_dict(), strict=True)
