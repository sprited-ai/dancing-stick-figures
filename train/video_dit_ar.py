"""M4: M3-derived block-autoregressive text-to-video rectified-flow DiT.

The architectural delta from :mod:`train.video_dit_fm` is deliberately small:

* generated video is split into a clean history prefix and a noisy target block;
* only the target block is denoised and receives a training loss;
* temporal attention uses signed RoPE positions (history ``-H..-1``, target
  ``0..F-1``) and a prefix mask; and
* generation may replace the text condition at every block boundary.

Spatial patches, spatial attention, text cross-attention, adaLN, MLPs, and the
output head retain M3-compatible parameter names.  An M3 checkpoint therefore
warm-starts every compatible tensor except its learned absolute ``pos_t``.
No VAE or motion/pose representation is introduced in M4.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from train.video_ddpm import VideoWindows, worker_init
from train.video_dit_fm import (
    Attention,
    TextCrossAttention,
    foreground_weighted_mse,
    modulate,
    sample_t,
    timestep_embedding,
)


def signed_window_positions(history: int | torch.Tensor, target: int, *, device=None) -> torch.Tensor:
    """Return signed local positions for a history-prefix/target window.

    ``history`` may be an integer shared by the batch or a ``[B]`` tensor.  A
    ten-frame history and four-frame target maps to ``[-10..-1, 0..3]``.  The
    tensor form is useful for tests and padded evaluation; training intentionally
    uses one history length per batch so no padded temporal compute is required.
    """
    if target <= 0:
        raise ValueError("target must be positive")
    if isinstance(history, int):
        if history < 0:
            raise ValueError("history must be non-negative")
        return torch.arange(-history, target, device=device)
    if history.ndim != 1 or (history < 0).any():
        raise ValueError("history tensor must be non-negative with shape [B]")
    max_h = int(history.max().item()) if history.numel() else 0
    base = torch.arange(-max_h, target, device=history.device)
    return base.unsqueeze(0).expand(history.shape[0], -1)


def prefix_attention_mask(history: int, target: int, *, device=None) -> torch.Tensor:
    """Boolean SDPA mask for immutable-prefix block denoising.

    History queries may only read history. Target queries may read the complete
    history and the complete (bidirectional) target block.  This prevents target
    information from flowing backward into history and returning through a later
    transformer layer.
    """
    if history < 0 or target <= 0:
        raise ValueError("history must be >= 0 and target must be > 0")
    total = history + target
    mask = torch.ones((total, total), dtype=torch.bool, device=device)
    if history:
        mask[:history, history:] = False
    return mask


def _apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float = 10_000.0) -> torch.Tensor:
    """Apply RoPE to ``x [B,H,T,D]`` using signed integer positions."""
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("RoPE head dimension must be even")
    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(x.shape[0], -1)
    if positions.shape != (x.shape[0], x.shape[2]):
        raise ValueError(f"positions must be [B,T], got {tuple(positions.shape)} for {tuple(x.shape)}")
    inv = torch.exp(-math.log(theta) * torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    angles = positions.to(torch.float32).unsqueeze(-1) * inv
    cos, sin = angles.cos().to(x.dtype).unsqueeze(1), angles.sin().to(x.dtype).unsqueeze(1)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class TemporalRoPEAttention(nn.Module):
    """M3 state-dict-compatible attention with signed temporal RoPE."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if (dim // heads) % 2:
            raise ValueError("temporal RoPE requires an even head dimension")
        self.h = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o = nn.Linear(dim, dim)
        self.qn = nn.LayerNorm(dim // heads)
        self.kn = nn.LayerNorm(dim // heads)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.h, dim // self.h).permute(2, 0, 3, 1, 4)
        q, k = _apply_rope(self.qn(q), positions), _apply_rope(self.kn(k), positions)
        attn_mask = None
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).expand(batch, -1, -1)
            if mask.shape != (batch, length, length):
                raise ValueError(f"mask must be [B,T,T], got {tuple(mask.shape)}")
            attn_mask = mask[:, None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o(out.transpose(1, 2).reshape(batch, length, dim))


def full_prefix_attention_mask(history: int, target: int, patches: int, *, device=None) -> torch.Tensor:
    """Token-level immutable-prefix mask for flattened ``[T,H,W]`` tokens."""
    if history < 0 or target <= 0 or patches <= 0:
        raise ValueError("history must be >= 0 and target/patches must be > 0")
    times = torch.arange(history + target, device=device).repeat_interleave(patches)
    history_query = times[:, None] < history
    target_key = times[None, :] >= history
    return ~(history_query & target_key)


def _apply_rope_3d(x: torch.Tensor, positions: torch.Tensor, theta: float = 10_000.0) -> torch.Tensor:
    """Wan-style axis-split RoPE for ``x [B,H,L,D]`` and signed ``[B,L,3]`` positions."""
    dim = x.shape[-1]
    if dim % 2 or positions.shape != (x.shape[0], x.shape[2], 3):
        raise ValueError("3D RoPE requires even head dim and positions [B,L,3]")
    spatial = 2 * (dim // 6)
    temporal = dim - 2 * spatial
    splits = (temporal, spatial, spatial)
    if min(splits) <= 0 or any(part % 2 for part in splits):
        raise ValueError(f"head dim {dim} cannot be split into even 3D RoPE axes")
    outputs = []
    start = 0
    for axis, width in enumerate(splits):
        outputs.append(_apply_rope(x[..., start:start + width], positions[..., axis], theta))
        start += width
    return torch.cat(outputs, dim=-1)


class FullSTAttention(nn.Module):
    """Global spatiotemporal self-attention with signed 3D RoPE."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o = nn.Linear(dim, dim)
        self.qn = nn.LayerNorm(dim // heads)
        self.kn = nn.LayerNorm(dim // heads)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.h, dim // self.h).permute(2, 0, 3, 1, 4)
        q, k = _apply_rope_3d(self.qn(q), positions), _apply_rope_3d(self.kn(k), positions)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(batch, -1, -1)
        if mask.shape != (batch, length, length):
            raise ValueError(f"mask must be [B,L,L], got {tuple(mask.shape)}")
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        return self.o(out.transpose(1, 2).reshape(batch, length, dim))


class FullSTARBlock(nn.Module):
    """Reference-family full-ST DiT block with text cross-attention and adaLN."""

    def __init__(self, dim: int, heads: int, mlp: int = 4, text_cond: bool = True):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = FullSTAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp * dim), nn.GELU(approximate="tanh"), nn.Linear(mlp * dim, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.text_attn = TextCrossAttention(dim, heads) if text_cond else None
        self.text_norm = nn.LayerNorm(dim, elementwise_affine=False) if text_cond else None
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c, text, text_mask, positions, mask):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c).chunk(6, -1)
        x = x + gate1[:, None] * self.attn(modulate(self.n1(x), shift1, scale1), positions, mask)
        x = x + gate2[:, None] * self.mlp(modulate(self.n2(x), shift2, scale2))
        if self.text_attn is not None and text is not None:
            x = x + self.text_attn(self.text_norm(x), text, text_mask)
        return x


class ARBlock(nn.Module):
    """M3-compatible DiT block; temporal blocks accept signed positions/masks."""

    def __init__(self, dim: int, heads: int, axis: str, mlp: int = 4, text_cond: bool = False):
        super().__init__()
        self.axis = axis
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads) if axis == "spatial" else TemporalRoPEAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp * dim), nn.GELU(approximate="tanh"), nn.Linear(mlp * dim, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.text_attn = TextCrossAttention(dim, heads) if text_cond else None
        self.text_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) if text_cond else None
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        text: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        temporal_positions: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, frames, patches, dim = x.shape
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, -1)
        if self.axis == "spatial":
            h = x.reshape(batch * frames, patches, dim)
            rep = lambda z: z.repeat_interleave(frames, 0)
            attn_out = self.attn(modulate(self.n1(h), rep(s1), rep(sc1)))
        else:
            if temporal_positions is None:
                raise ValueError("temporal_positions are required for temporal attention")
            h = x.permute(0, 2, 1, 3).reshape(batch * patches, frames, dim)
            rep = lambda z: z.repeat_interleave(patches, 0)
            pos = temporal_positions.repeat_interleave(patches, 0)
            mask = temporal_mask.repeat_interleave(patches, 0) if temporal_mask is not None else None
            attn_out = self.attn(modulate(self.n1(h), rep(s1), rep(sc1)), pos, mask)
        h = h + rep(g1).unsqueeze(1) * attn_out
        h = h + rep(g2).unsqueeze(1) * self.mlp(modulate(self.n2(h), rep(s2), rep(sc2)))
        if self.axis == "spatial":
            h = h.reshape(batch, frames, patches, dim)
        else:
            h = h.reshape(batch, patches, frames, dim).permute(0, 2, 1, 3)
        if self.text_attn is not None and text is not None:
            flat = h.reshape(batch, frames * patches, dim)
            flat = flat + self.text_attn(self.text_norm(flat), text, text_mask)
            h = flat.reshape(batch, frames, patches, dim)
        return h


class ARVideoDiT(nn.Module):
    """M4 block-AR VideoDiT with an immutable history prefix."""

    def __init__(
        self,
        size: int = 64,
        patch: int = 4,
        in_ch: int = 4,
        dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        cond_ch: int = 5,
        text_dim: int = 512,
    ):
        super().__init__()
        self.p, self.S, self.C, self.dim = patch, size, in_ch, dim
        self.cond_ch = cond_ch
        self.N = (size // patch) ** 2
        self.embed = nn.Linear((in_ch + cond_ch) * patch * patch, dim)
        self.pos_s = nn.Parameter(torch.zeros(1, 1, self.N, dim))
        nn.init.normal_(self.pos_s, std=0.02)
        self.temb = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.cls = None
        self.text_proj = nn.Linear(text_dim, dim) if text_dim else None
        self.blocks = nn.ModuleList(
            ARBlock(dim, heads, "spatial" if i % 2 == 0 else "temporal", text_cond=bool(text_dim))
            for i in range(depth)
        )
        self.nf = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, in_ch * patch * patch)
        nn.init.zeros_(self.ada_f[-1].weight)
        nn.init.zeros_(self.ada_f[-1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.grad_ckpt = False

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = x.shape
        p = self.p
        x = x.reshape(batch, channels, frames, height // p, p, width // p, p).permute(0, 2, 3, 5, 1, 4, 6)
        return x.reshape(batch, frames, (height // p) * (width // p), channels * p * p)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, patches, _ = x.shape
        p = self.p
        height = width = int(math.sqrt(patches))
        x = x.reshape(batch, frames, height, width, self.C, p, p).permute(0, 4, 1, 2, 5, 3, 6)
        return x.reshape(batch, self.C, frames, height * p, width * p)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y=None,
        cond: torch.Tensor | None = None,
        text: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        history_frames: int = 0,
    ) -> torch.Tensor:
        batch, _, frames, _, _ = x.shape
        if not 0 <= history_frames < frames:
            raise ValueError(f"history_frames must be in [0,{frames}), got {history_frames}")
        if self.cond_ch:
            if cond is None:
                cond = torch.zeros(batch, self.cond_ch, *x.shape[2:], device=x.device, dtype=x.dtype)
            x = torch.cat((x, cond.to(x.dtype)), dim=1)
        h = self.embed(self.patchify(x)) + self.pos_s
        c = self.temb(timestep_embedding(t, 256))
        if self.text_proj is not None:
            if text is None:
                raise ValueError("text-conditioned ARVideoDiT requires text embeddings")
            text = self.text_proj(text.to(h.dtype))
            weights = text_mask.to(text.dtype).unsqueeze(-1) if text_mask is not None else torch.ones_like(text[..., :1])
            c = c + (text * weights).sum(1) / weights.sum(1).clamp_min(1)
        positions = signed_window_positions(history_frames, frames - history_frames, device=x.device)
        positions = positions.unsqueeze(0).expand(batch, -1)
        attn_mask = prefix_attention_mask(history_frames, frames - history_frames, device=x.device)
        attn_mask = attn_mask.unsqueeze(0).expand(batch, -1, -1)
        for block in self.blocks:
            args = (h, c, text, text_mask, positions, attn_mask)
            h = checkpoint(block, *args, use_reentrant=False) if (self.grad_ckpt and self.training) else block(*args)
        shift, scale = self.ada_f(c).chunk(2, -1)
        h = self.nf(h) * (1 + scale[:, None, None]) + shift[:, None, None]
        return self.unpatchify(self.out(h))


class FullSTARVideoDiT(nn.Module):
    """Block-AR latent DiT using Wan/CogVideoX-style full video-token attention.

    Unlike :class:`ARVideoDiT`, every self-attention layer operates on the
    flattened spatiotemporal grid. Signed temporal coordinates keep a sliding
    history local (``-H..-1``) while spatial coordinates use ordinary grid
    indices. The prefix mask prevents target tokens from changing the
    representations used by immutable history queries.
    """

    def __init__(
        self, size: int = 8, patch: int = 1, in_ch: int = 16, dim: int = 384,
        depth: int = 12, heads: int = 6, cond_ch: int = 17, text_dim: int = 512,
    ):
        super().__init__()
        if size % patch:
            raise ValueError("latent size must be divisible by patch size")
        self.p, self.S, self.C, self.dim = patch, size, in_ch, dim
        self.cond_ch = cond_ch
        self.grid = size // patch
        self.N = self.grid**2
        self.embed = nn.Linear((in_ch + cond_ch) * patch * patch, dim)
        self.temb = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.text_proj = nn.Linear(text_dim, dim) if text_dim else None
        self.blocks = nn.ModuleList(FullSTARBlock(dim, heads, text_cond=bool(text_dim)) for _ in range(depth))
        self.nf = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, in_ch * patch * patch)
        nn.init.zeros_(self.ada_f[-1].weight); nn.init.zeros_(self.ada_f[-1].bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.grad_ckpt = False

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = x.shape
        p = self.p
        x = x.reshape(batch, channels, frames, height // p, p, width // p, p).permute(0, 2, 3, 5, 1, 4, 6)
        return x.reshape(batch, frames, self.N, channels * p * p)

    def unpatchify(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        batch, _, _ = x.shape
        p, grid = self.p, self.grid
        x = x.reshape(batch, frames, grid, grid, self.C, p, p).permute(0, 4, 1, 2, 5, 3, 6)
        return x.reshape(batch, self.C, frames, grid * p, grid * p)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y=None, cond: torch.Tensor | None = None,
        text: torch.Tensor | None = None, text_mask: torch.Tensor | None = None,
        history_frames: int = 0,
    ) -> torch.Tensor:
        batch, _, frames, _, _ = x.shape
        if not 0 <= history_frames < frames:
            raise ValueError(f"history_frames must be in [0,{frames}), got {history_frames}")
        if self.cond_ch:
            if cond is None:
                cond = torch.zeros(batch, self.cond_ch, *x.shape[2:], device=x.device, dtype=x.dtype)
            x = torch.cat((x, cond.to(x.dtype)), dim=1)
        h = self.embed(self.patchify(x)).reshape(batch, frames * self.N, self.dim)
        c = self.temb(timestep_embedding(t, 256))
        if self.text_proj is not None:
            if text is None:
                raise ValueError("text-conditioned FullSTARVideoDiT requires text embeddings")
            text = self.text_proj(text.to(h.dtype))
            weights = text_mask.to(text.dtype).unsqueeze(-1) if text_mask is not None else torch.ones_like(text[..., :1])
            c = c + (text * weights).sum(1) / weights.sum(1).clamp_min(1)

        temporal = signed_window_positions(history_frames, frames - history_frames, device=x.device)
        temporal = temporal.repeat_interleave(self.N)
        yy, xx = torch.meshgrid(
            torch.arange(self.grid, device=x.device), torch.arange(self.grid, device=x.device), indexing="ij",
        )
        yy = yy.reshape(-1).repeat(frames); xx = xx.reshape(-1).repeat(frames)
        positions = torch.stack((temporal, yy, xx), dim=-1).unsqueeze(0).expand(batch, -1, -1)
        mask = full_prefix_attention_mask(
            history_frames, frames - history_frames, self.N, device=x.device,
        ).unsqueeze(0).expand(batch, -1, -1)
        for block in self.blocks:
            args = (h, c, text, text_mask, positions, mask)
            h = checkpoint(block, *args, use_reentrant=False) if (self.grad_ckpt and self.training) else block(*args)
        shift, scale = self.ada_f(c).chunk(2, -1)
        h = self.nf(h) * (1 + scale[:, None]) + shift[:, None]
        return self.unpatchify(self.out(h), frames)


def history_condition(x0: torch.Tensor, history_frames: int) -> torch.Tensor:
    """Clean history channels plus a binary mask.

    The operation is channel-agnostic: M4 uses four RGBA channels, while M6
    uses the frozen codec's continuous latent width.
    """
    if not 0 <= history_frames < x0.shape[2]:
        raise ValueError("history_frames must leave at least one target frame")
    clean = torch.zeros_like(x0)
    mask = torch.zeros_like(x0[:, :1])
    if history_frames:
        clean[:, :, :history_frames] = x0[:, :, :history_frames]
        mask[:, :, :history_frames] = 1
    return torch.cat((clean, mask), dim=1)


def prepare_m3_warmstart(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Load all shape-compatible M3 tensors; intentionally drop absolute ``pos_t``."""
    return {key: value for key, value in source.items() if key in target and value.shape == target[key].shape}


@torch.no_grad()
def euler_sample_block(
    model: ARVideoDiT,
    history: torch.Tensor | None,
    target_frames: int,
    *,
    steps: int = 20,
    size: int = 64,
    text: torch.Tensor,
    text_mask: torch.Tensor | None = None,
    null_text: torch.Tensor | None = None,
    null_mask: torch.Tensor | None = None,
    cfg: float = 2.0,
    shift: float = 1.0,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
    clamp: tuple[float, float] | None = (-1.0, 1.0),
) -> torch.Tensor:
    """Generate one target block while keeping ``history`` bit-exactly fixed."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    batch = text.shape[0]
    history_frames = 0 if history is None else history.shape[2]
    channels = model.C
    if history is None:
        history = torch.empty((batch, channels, 0, size, size), device=text.device)
    elif history.shape[1] != channels:
        raise ValueError(f"history has {history.shape[1]} channels, model expects {channels}")
    expected_noise_shape = (batch, channels, target_frames, size, size)
    if initial_noise is None:
        noise = torch.randn(expected_noise_shape, device=text.device, generator=generator)
    else:
        if tuple(initial_noise.shape) != expected_noise_shape:
            raise ValueError(
                f"initial noise has shape {tuple(initial_noise.shape)}, expected {expected_noise_shape}"
            )
        noise = initial_noise.to(device=text.device)
    target = noise
    schedule = torch.linspace(1, 0, steps + 1, device=text.device)
    schedule = shift * schedule / (1 + (shift - 1) * schedule)
    for now, nxt in zip(schedule[:-1], schedule[1:]):
        full = torch.cat((history, target), dim=2)
        cond = history_condition(full, history_frames)
        timestep = torch.full((batch,), float(now), device=text.device)
        velocity = model(full, timestep, cond=cond, text=text, text_mask=text_mask, history_frames=history_frames)
        velocity = velocity[:, :, history_frames:]
        if cfg > 0 and null_text is not None:
            unconditional = model(full, timestep, cond=cond, text=null_text, text_mask=null_mask,
                                  history_frames=history_frames)[:, :, history_frames:]
            velocity = unconditional + cfg * (velocity - unconditional)
        target = target + (nxt - now) * velocity
    return target if clamp is None else target.clamp(*clamp)


@torch.no_grad()
def rollout_blocks(
    model: ARVideoDiT,
    prompts: Sequence[tuple[torch.Tensor, torch.Tensor | None]],
    *,
    total_frames: int = 50,
    target_frames: int = 10,
    history_max: int = 40,
    steps: int = 20,
    null_text: torch.Tensor | None = None,
    null_mask: torch.Tensor | None = None,
    cfg: float = 2.0,
    shift: float = 1.0,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
    initial_video: torch.Tensor | None = None,
    sample_clamp: tuple[float, float] | None = (-1.0, 1.0),
) -> torch.Tensor:
    """Autoregressively append blocks; ``prompts`` may change per block.

    ``total_frames`` is the number of *new* frames when ``initial_video`` is
    supplied.  The model always generates its trained full target-block length
    and trims only the returned view, rather than asking it to denoise an unseen
    shorter block.  This makes pause/save/resume and indefinite continuation a
    first-class inference path.
    """
    if total_frames <= 0 or target_frames <= 0 or history_max < 0:
        raise ValueError("invalid rollout lengths")
    if not prompts:
        raise ValueError("at least one prompt embedding is required")
    output = initial_video
    initial_length = 0 if initial_video is None else initial_video.shape[2]
    desired_length = initial_length + total_frames
    required_noise_frames = math.ceil(total_frames / target_frames) * target_frames
    if initial_noise is not None:
        expected = (prompts[0][0].shape[0], model.C, required_noise_frames, model.S, model.S)
        if tuple(initial_noise.shape) != expected:
            raise ValueError(f"initial noise has shape {tuple(initial_noise.shape)}, expected {expected}")
    block = 0
    while output is None or output.shape[2] < desired_length:
        text, text_mask = prompts[min(block, len(prompts) - 1)]
        history = None if output is None else output[:, :, -history_max:] if history_max else None
        generated = euler_sample_block(
            model, history, target_frames, steps=steps, size=model.S, text=text, text_mask=text_mask,
            null_text=null_text, null_mask=null_mask, cfg=cfg, shift=shift, generator=generator,
            initial_noise=None if initial_noise is None else
                initial_noise[:, :, block*target_frames:(block+1)*target_frames],
            clamp=sample_clamp,
        )
        output = generated if output is None else torch.cat((output, generated), dim=2)
        block += 1
    return output[:, :, :desired_length]


def _parse_steps(spec: str) -> set[int]:
    if not spec:
        return set()
    try:
        values = {int(value.strip()) for value in spec.split(",") if value.strip()}
    except ValueError as exc:
        raise ValueError(f"invalid step list: {spec!r}") from exc
    if any(value < 0 for value in values):
        raise ValueError("milestone steps must be non-negative")
    return values


def _parse_histories(spec: str, history_max: int) -> tuple[int, ...]:
    histories = tuple(sorted(_parse_steps(spec)))
    if not histories:
        raise ValueError("history choices cannot be empty")
    if histories[0] != 0 or histories[-1] > history_max:
        raise ValueError("history choices must include 0 and stay within history_max")
    return histories


def _select_history_window(video: torch.Tensor, history: int, target: int, history_max: int) -> torch.Tensor:
    """Right-align a variable history against a fixed target inside a max-size data window."""
    expected = history_max + target
    if video.shape[2] != expected:
        raise ValueError(f"expected {expected} cached frames, got {video.shape[2]}")
    return video[:, :, history_max - history:]


def _make_flow_batch(clean: torch.Tensor, history: int, shift: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Noise only the target; return model input, flow target, and timesteps."""
    timestep = sample_t(clean.shape[0], clean.device, shift)
    amount = timestep[:, None, None, None, None]
    noise = torch.randn_like(clean)
    model_input = (1 - amount) * clean + amount * noise
    if history:
        model_input[:, :, :history] = clean[:, :, :history]
    return model_input, noise - clean, timestep


@torch.no_grad()
def validation_losses(
    model: ARVideoDiT,
    loader,
    text_batch,
    *,
    history_max: int,
    target_frames: int,
    shift: float,
    device: str,
    n_batches: int = 8,
    seed: int = 20260821,
) -> dict[str, float]:
    """Deterministic first-block and teacher-forced continuation losses."""
    model.eval()
    results = {}
    for name, history in (("first", 0), ("continuation", history_max)):
        total = count = 0
        generator = torch.Generator(device=device).manual_seed(seed)
        for batch_index, (full, labels) in enumerate(loader):
            if batch_index >= n_batches:
                break
            clean = _select_history_window(full.to(device), history, target_frames, history_max)
            batch = clean.shape[0]
            timestep = torch.sigmoid(torch.randn(batch, device=device, generator=generator))
            timestep = shift * timestep / (1 + (shift - 1) * timestep)
            amount = timestep[:, None, None, None, None]
            noise = torch.randn(clean.shape, device=device, generator=generator)
            model_input = (1 - amount) * clean + amount * noise
            if history:
                model_input[:, :, :history] = clean[:, :, :history]
            text, text_mask = text_batch(labels)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                prediction = model(
                    model_input, timestep, cond=history_condition(clean, history), text=text,
                    text_mask=text_mask, history_frames=history,
                )
            error = (prediction[:, :, history:].float() - (noise - clean)[:, :, history:]) ** 2
            total += float(error.mean())
            count += 1
        results[name] = total / max(count, 1)
    return results


def _default_preview_prompts() -> list[str]:
    return [
        "A person walks forward.",
        "A person runs forward.",
        "A person sits down on a chair.",
        "A person waves hello with the left hand.",
    ]


def _switch_prompt_grid() -> list[list[str]]:
    """Four rows, five one-second blocks each."""
    return [
        ["A person walks forward."] * 2 + ["A person runs forward."] * 3,
        ["A person sits down on a chair."] * 2 + ["A person remains seated and still."] * 3,
        ["A person waves hello with the left hand."] * 2 + ["A person stands still and breathes."] * 3,
        ["A person jumps up and down."] * 2 + ["A person stands still and breathes."] * 3,
    ]


def _save_checkpoint(path: Path, model, ema, optimizer, step: int, args: dict, *, full: bool) -> None:
    payload = {
        "ema": ema.state_dict(),
        "step": step,
        "args": args,
        "arch": "dit_fm_t2v_ar",
        "protocol": "m4_signed_rope_block_ar_v1",
    }
    if full:
        payload.update(
            model=model.state_dict(),
            opt=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
        )
        if torch.cuda.is_available():
            payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


@torch.no_grad()
def save_previews(
    model: ARVideoDiT,
    raw_model: ARVideoDiT,
    text_batch,
    out: Path,
    step: int,
    args,
) -> None:
    """Write fixed-seed labeled single-prompt and prompt-switch rollouts."""
    from eval.post_eval_t2v import save_labeled_gif, save_strip

    model.eval()
    rows = _default_preview_prompts()
    text, text_mask = text_batch(rows)
    null_text, null_mask = text_batch([""] * len(rows))
    generator = torch.Generator(device=text.device).manual_seed(20260821)
    fixed = rollout_blocks(
        model, [(text, text_mask)], total_frames=args.rollout_frames, target_frames=args.target_frames,
        history_max=args.history_max, steps=args.sample_steps, null_text=null_text, null_mask=null_mask,
        cfg=args.sample_cfg, shift=args.shift, generator=generator,
    ).cpu()
    fixed_name = f"fixed_prompt_{step:06d}_labeled.gif"
    save_labeled_gif(fixed, str(out / fixed_name), rows, fps=args.fps)
    save_strip(fixed, str(out / f"fixed_prompt_{step:06d}_strip.png"), rows)

    schedules = _switch_prompt_grid()
    block_conditions = []
    for block in range(math.ceil(args.rollout_frames / args.target_frames)):
        block_prompts = [schedule[min(block, len(schedule) - 1)] for schedule in schedules]
        block_conditions.append(text_batch(block_prompts))
    generator = torch.Generator(device=text.device).manual_seed(20260822)
    switched = rollout_blocks(
        model, block_conditions, total_frames=args.rollout_frames, target_frames=args.target_frames,
        history_max=args.history_max, steps=args.sample_steps, null_text=null_text, null_mask=null_mask,
        cfg=args.sample_cfg, shift=args.shift, generator=generator,
    ).cpu()
    switch_labels = [f"{schedule[0]} -> {schedule[-1]}" for schedule in schedules]
    switch_name = f"prompt_switch_{step:06d}_labeled.gif"
    save_labeled_gif(switched, str(out / switch_name), switch_labels, fps=args.fps)
    save_strip(switched, str(out / f"prompt_switch_{step:06d}_strip.png"), switch_labels)

    manifest = {
        "step": step,
        "seed_fixed": 20260821,
        "seed_switch": 20260822,
        "target_frames": args.target_frames,
        "history_max": args.history_max,
        "rollout_frames": args.rollout_frames,
        "sampling_steps_per_block": args.sample_steps,
        "cfg": args.sample_cfg,
        "fps": args.fps,
        "fixed_prompts": rows,
        "switch_schedules": schedules,
        "outputs": [fixed_name, switch_name],
    }
    (out / f"sample_manifest_{step:06d}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    raw_model.train()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init", required=True, help="M3 full or EMA checkpoint")
    parser.add_argument("--resume", default="")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--target-frames", type=int, default=10)
    parser.add_argument("--history-max", type=int, default=40)
    parser.add_argument("--history-choices", default="0,10,20,30,40")
    parser.add_argument("--rollout-frames", type=int, default=100,
                        help="canonical five-second rollout at the default 20 fps")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-final", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--fg-weight", type=float, default=2.0)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--cfg-drop", type=float, default=0.1)
    parser.add_argument("--text-encoder", default="google-t5/t5-small")
    parser.add_argument("--text-len", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--sample-cfg", type=float, default=2.0)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--sample-milestones", default="0,10,50,100,250,500,1000,2000,3000")
    parser.add_argument("--no-previews", action="store_true", help="capacity/debug smoke only; long runs must keep previews")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.history_max + args.target_frames != 50:
        raise ValueError("M4 v1 deliberately uses a 50-frame data window (history_max + target_frames)")
    histories = _parse_histories(args.history_choices, args.history_max)
    milestones = _parse_steps(args.sample_milestones)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.fast and args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_set = VideoWindows(args.cache, 50, "train", args.stride, size=args.size, return_text=True)
    val_set = VideoWindows(args.cache, 50, "val", args.stride, size=args.size, return_text=True,
                           deterministic=True, repeats=1)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
        pin_memory=args.device.startswith("cuda"), persistent_workers=args.workers > 0, worker_init_fn=worker_init,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch, shuffle=False, num_workers=min(2, args.workers), drop_last=True,
        worker_init_fn=worker_init,
    )

    from transformers import AutoTokenizer, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    encoder = T5EncoderModel.from_pretrained(args.text_encoder).to(args.device).eval().requires_grad_(False)
    prompt_set = sorted({clip["text"] for clip in train_set.clips + val_set.clips} | {""} |
                        set(_default_preview_prompts()) |
                        {prompt for schedule in _switch_prompt_grid() for prompt in schedule})
    text_cache = {}
    for start in range(0, len(prompt_set), 32):
        prompts = prompt_set[start:start + 32]
        tokens = tokenizer(prompts, padding="max_length", truncation=True, max_length=args.text_len,
                           return_tensors="pt")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            hidden = encoder(input_ids=tokens.input_ids.to(args.device),
                             attention_mask=tokens.attention_mask.to(args.device)).last_hidden_state.float().cpu()
        for prompt, embedding, mask in zip(prompts, hidden, tokens.attention_mask):
            text_cache[prompt] = (embedding, mask)
    text_dim = encoder.config.d_model
    del encoder
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    def text_batch(prompts):
        embeddings, masks = zip(*(text_cache[prompt] for prompt in prompts))
        return torch.stack(embeddings).to(args.device), torch.stack(masks).to(args.device)

    model = ARVideoDiT(
        size=args.size, patch=args.patch, dim=args.dim, depth=args.depth, heads=args.heads,
        cond_ch=5, text_dim=text_dim,
    ).to(args.device)
    model.grad_ckpt = args.grad_ckpt
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
                                  fused=args.fast and args.device.startswith("cuda"))
    step = 0
    if args.resume:
        checkpoint_data = torch.load(args.resume, map_location=args.device, weights_only=False)
        if checkpoint_data.get("protocol") != "m4_signed_rope_block_ar_v1":
            raise ValueError("resume checkpoint is not M4 signed-RoPE block-AR v1")
        model.load_state_dict(checkpoint_data["model"])
        ema.load_state_dict(checkpoint_data["ema"])
        optimizer.load_state_dict(checkpoint_data["opt"])
        step = int(checkpoint_data["step"])
    else:
        checkpoint_data = torch.load(args.init, map_location=args.device, weights_only=False)
        for destination, key in ((model, "model"), (ema, "ema")):
            source = checkpoint_data.get(key) or checkpoint_data["ema"]
            selected = prepare_m3_warmstart(source, destination.state_dict())
            result = destination.load_state_dict(selected, strict=False)
            if result.missing_keys or result.unexpected_keys:
                raise RuntimeError(f"incomplete M3 warm start: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
            print(f"loaded {len(selected)} tensors into M4 {key}; dropped M3 absolute pos_t", flush=True)

    print(
        f"M4 {sum(parameter.numel() for parameter in model.parameters()) / 1e6:.1f}M params; "
        f"{len(train_set.clips)} train/{len(val_set.clips)} val clips; H={histories}, F={args.target_frames}; "
        f"{args.sample_steps} sampling steps/block",
        flush=True,
    )
    log = (out / "log.txt").open("a")
    iterator = iter(train_loader)
    start_time = time.time()
    start_step = step
    ema_loss = None

    if not args.no_previews and step == 0 and 0 in milestones:
        save_previews(ema, model, text_batch, out, step, args)
        _save_checkpoint(out / "ckpt_000000.pt", model, ema, optimizer, 0, vars(args), full=False)

    while step < args.steps:
        try:
            full, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            full, labels = next(iterator)
        history = random.choice(histories)
        clean = _select_history_window(full.to(args.device, non_blocking=True), history, args.target_frames,
                                       args.history_max)
        text, text_mask = text_batch(labels)
        null_text, null_mask = text_batch([""] * clean.shape[0])
        dropped = torch.rand(clean.shape[0], device=args.device) < args.cfg_drop
        text = torch.where(dropped[:, None, None], null_text, text)
        text_mask = torch.where(dropped[:, None], null_mask, text_mask)
        model_input, flow_target, timestep = _make_flow_batch(clean, history, args.shift)
        condition = history_condition(clean, history)

        progress = max(0.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        lr *= args.lr_final + (1 - args.lr_final) * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            prediction = model(model_input, timestep, cond=condition, text=text, text_mask=text_mask,
                               history_frames=history)
        target_error = (prediction[:, :, history:].float() - flow_target[:, :, history:]) ** 2
        target_clean = clean[:, :, history:]
        loss = foreground_weighted_mse(target_error, target_clean, args.fg_weight) / args.accum
        loss.backward()
        if (step + 1) % args.accum:
            step += 1
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            decay = min(0.999 if step < 5000 else 0.9995, (1 + step) / (10 + step))
            for ema_parameter, model_parameter in zip(ema.parameters(), model.parameters()):
                ema_parameter.lerp_(model_parameter, 1 - decay)
        step += 1
        value = float(loss.detach()) * args.accum
        ema_loss = value if ema_loss is None else 0.98 * ema_loss + 0.02 * value

        if step == 1 or step % 50 == 0:
            seconds = (time.time() - start_time) / max(1, step - start_step)
            message = f"step {step} loss {ema_loss:.5f} H {history} lr {lr:.2e} {seconds:.2f}s/it"
            if args.device.startswith("cuda"):
                message += f" peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB"
            message += f" ETA {(args.steps-step)*seconds/3600:.2f}h"
            if step % args.val_every == 0:
                values = validation_losses(
                    ema, val_loader, text_batch, history_max=args.history_max,
                    target_frames=args.target_frames, shift=args.shift, device=args.device,
                )
                message += f" val_first {values['first']:.5f} val_cont {values['continuation']:.5f}"
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        if step % args.save_every == 0 or step == args.steps:
            _save_checkpoint(out / "latest.pt", model, ema, optimizer, step, vars(args), full=True)
        if not args.no_previews and (step in milestones or step % args.sample_every == 0 or step == args.steps):
            save_previews(ema, model, text_batch, out, step, args)
            _save_checkpoint(out / f"ckpt_{step:06d}.pt", model, ema, optimizer, step, vars(args), full=False)
            print(f"wrote labeled M4 rollouts at step {step}", flush=True)


if __name__ == "__main__":
    main()
