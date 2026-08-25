"""M6 v9: rig co-generation — pixels and the cskel27 rig denoised jointly.

The v8 recipe (16-frame blocks, motion-weighted + foreground-weighted flow
loss) is kept; the deliberate change is one rig token per temporal latent
index appended to the spatiotemporal token sequence and denoised under the
same rectified flow.  The rig token carries the 2D screen coordinates of the
codec group's video frames (temporal_compression x 27 x 2, mapped to [-1,1]).

Everything lives in this module so the running v8/M6 code paths stay
untouched: dataset (rig-aligned windows), model subclass, flow batch, joint
sampler, previews, and the training entry point.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

from train.video_ddpm import VideoWindows, worker_init
from train.video_dit_ar import (
    FullSTARVideoDiT,
    _default_preview_prompts,
    _parse_histories,
    _parse_steps,
    history_condition,
    sample_t,
    signed_window_positions,
    timestep_embedding,
)
from train.latent_video_dit_ar import (
    LatentStandardizer,
    M6_SWITCH_PAIRS,
    decode_full,
    encode_video,
    file_sha256,
    flow_prediction_to_clean,
    foreground_latent_weight,
    load_codec,
    m6_switch_prompt_grid,
    select_video_history_window,
    validate_experiment_protocol,
)
from train.soft_skeleton_renderer import SoftSkeletonRenderer

RIG_JOINTS = 27
RIG_PARENTS = (-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 10, 10, 4, 13, 14, 15, 16, 16,
               0, 19, 20, 21, 0, 23, 24, 25)
RIG_NAMES = (
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandEnd", "RightHandThumb1",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandEnd", "LeftHandThumb1",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
)
# dataset limb palette (generator/render.py PALETTE; unlisted bones = ink)
RIG_INK = (40, 40, 40)
RIG_PALETTE = {
    "LeftForeArm": (232, 64, 48), "LeftHand": (255, 150, 40), "LeftHandEnd": (255, 150, 40),
    "RightForeArm": (40, 110, 230), "RightHand": (80, 200, 240), "RightHandEnd": (80, 200, 240),
    "LeftLeg": (200, 50, 160), "LeftFoot": (255, 120, 200), "LeftToeBase": (255, 120, 200),
    "RightLeg": (30, 150, 90), "RightFoot": (120, 220, 90), "RightToeBase": (120, 220, 90),
}
V9_PROTOCOL = "m6_latent_block_ar_v9_rig_cogen"


def build_rig_renderer(output_size: int) -> "SoftSkeletonRenderer":
    """Palette-faithful soft renderer matched to the dataset's drawn geometry.

    Thumb bones are undrawn in the data (dropped by re-rooting the thumb
    joints); the head sphere is approximated by a thick Neck-Head capsule.
    Radii approximate the median body (stroke 4 px, head_r 0.125 m at 128 px,
    halved for the 64 px cache).
    """
    parents = list(RIG_PARENTS)
    for name in ("RightHandThumb1", "LeftHandThumb1"):
        parents[RIG_NAMES.index(name)] = -1
    colors = [[c / 255 for c in RIG_PALETTE.get(name, RIG_INK)] for name in RIG_NAMES]
    radii = [3.4 if name == "Head" else 1.6 for name in RIG_NAMES]
    return SoftSkeletonRenderer(parents, bone_colors=colors, body_radius=radii,
                                image_size=output_size, normalized_coordinates=True)


def fg_weighted_alpha_mse(a: torch.Tensor, b: torch.Tensor, bg_weight: float = 0.02) -> torch.Tensor:
    """Alpha-map MSE weighted by the union foreground support.

    The transparent background dominates the 64x64 frame, so a plain MSE lets
    background-background agreement wash out the gradient (the same failure the
    foreground-weighted flow loss fixed).  Pixels where EITHER map claims
    foreground carry weight 1; empty-empty pixels carry bg_weight.  The weight
    is detached so it gates the loss without being optimized itself.
    """
    weight = torch.maximum(a, b).detach().clamp(0, 1)
    weight = weight + bg_weight * (1 - weight)
    return ((a - b).square() * weight).sum() / weight.sum().clamp_min(1e-8)


def fg_weighted_rgba_mse(
    alpha_a: torch.Tensor, rgb_a: torch.Tensor,
    alpha_b: torch.Tensor, rgb_b: torch.Tensor, bg_weight: float = 0.02,
) -> torch.Tensor:
    """Alpha plus premultiplied-RGB error under the union-foreground weight.

    RGB carries limb identity (the dataset's colour-coded limbs): a rig with
    swapped arms explains the silhouette but not the colours. Both error maps
    share the same detached union support so background agreement stays inert.
    """
    weight = torch.maximum(alpha_a, alpha_b).detach().clamp(0, 1)
    weight = weight + bg_weight * (1 - weight)
    err = (alpha_a - alpha_b).square() + (rgb_a - rgb_b).square().mean(dim=-3)
    return (err * weight).sum() / weight.sum().clamp_min(1e-8)


def rig_bone_lengths(rig_tokens: torch.Tensor, temporal: int) -> torch.Tensor:
    """[B, T_lat, temporal*27*2] rig tokens -> [B, T_video, 26] bone lengths."""
    batch, t_lat, _ = rig_tokens.shape
    xy = rig_tokens.reshape(batch, t_lat * temporal, RIG_JOINTS, 2)
    child = torch.arange(1, RIG_JOINTS, device=rig_tokens.device)
    parent = torch.tensor(RIG_PARENTS[1:], device=rig_tokens.device)
    return (xy[:, :, child] - xy[:, :, parent]).norm(dim=-1)


# ----------------------------------------------------------------- data
class RigVideoWindows(VideoWindows):
    """VideoWindows that also returns the aligned rig window [T, 27, 2]."""

    def __init__(self, cache, frames, split, stride, **kwargs):
        super().__init__(cache, frames, split, stride, **kwargs)
        self.rig = np.load(os.path.join(cache, "rig.npy"), mmap_mode="r")
        self.rig_depth = np.load(os.path.join(cache, "rig_depth.npy"), mmap_mode="r")
        if self.rig.shape[0] != self.frames.shape[0] or self.rig_depth.shape[0] != self.frames.shape[0]:
            raise ValueError("rig.npy/rig_depth.npy are not aligned with frames.npy")

    def __getitem__(self, i):
        c, repeat = self.items[i]
        max_offset = c["n"] - self.span
        if self.deterministic:
            offset = 0 if self.repeats == 1 else round(repeat * max_offset / (self.repeats - 1))
        else:
            offset = random.randint(0, max_offset)
        s = c["start"] + offset
        x = np.asarray(self.frames[s:s + self.span:self.stride]).astype(np.float32) / 255.0
        rig = np.asarray(self.rig[s:s + self.span:self.stride]).astype(np.float32)
        depth = np.asarray(self.rig_depth[s:s + self.span:self.stride]).astype(np.float32)
        if self.size != x.shape[1]:
            f = x.shape[1] // self.size
            x = np.concatenate([x[..., :3] * x[..., 3:4], x[..., 3:4]], -1)
            x = x.reshape(x.shape[0], self.size, f, self.size, f, 4).mean((2, 4))
        else:
            a = x[..., 3:4]
            x = np.concatenate([x[..., :3] * a, a], -1)
        video = torch.from_numpy(x).permute(3, 0, 1, 2) * 2 - 1
        rig = torch.from_numpy(rig) * 2 - 1          # normalised [0,1] -> [-1,1]
        return video, rig, torch.from_numpy(depth), c["text"]


def select_rig_history_window(
    rig: torch.Tensor, *, history_latents: int, target_latents: int,
    history_max: int, temporal_compression: int, initial_block: bool = False,
) -> torch.Tensor:
    """Slice the rig window exactly like select_video_history_window."""
    expected = (history_max + target_latents) * temporal_compression
    if rig.shape[1] != expected:
        raise ValueError(f"expected {expected} rig frames, got {rig.shape[1]}")
    keep = (history_latents + target_latents) * temporal_compression
    if initial_block:
        if history_latents:
            raise ValueError("only a zero-history sample can be an initial block")
        return rig[:, :keep]
    return rig[:, -keep:]


def rig_tokens_from_frames(rig: torch.Tensor, temporal: int) -> torch.Tensor:
    """[B, T_video, 27, 2] -> [B, T_lat, temporal*27*2] grouped like the codec."""
    batch, frames, joints, two = rig.shape
    if frames % temporal or joints != RIG_JOINTS or two != 2:
        raise ValueError("rig frames must group evenly into codec latents")
    return rig.reshape(batch, frames // temporal, temporal * joints * 2)


# ----------------------------------------------------------------- model
class RigFullSTARVideoDiT(FullSTARVideoDiT):
    """FullSTARVideoDiT plus one co-denoised rig token per temporal index."""

    def __init__(self, *, temporal_compression: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.rig_dim = temporal_compression * RIG_JOINTS * 2
        # input: noisy rig + clean-history rig + binary history mask
        self.rig_embed = nn.Linear(2 * self.rig_dim + 1, self.dim)
        self.rig_out = nn.Linear(self.dim, self.rig_dim)
        nn.init.zeros_(self.rig_out.weight); nn.init.zeros_(self.rig_out.bias)

    def forward(
        self, x, t, y=None, cond=None, text=None, text_mask=None, history_frames: int = 0,
        rig_input: torch.Tensor | None = None, rig_cond: torch.Tensor | None = None,
    ):
        if rig_input is None or rig_cond is None:
            raise ValueError("rig co-generation requires rig_input and rig_cond")
        batch, _, frames, _, _ = x.shape
        if rig_input.shape != (batch, frames, self.rig_dim):
            raise ValueError(f"rig_input must be [B,{frames},{self.rig_dim}]")
        if not 0 <= history_frames < frames:
            raise ValueError("invalid history_frames")
        if self.cond_ch:
            if cond is None:
                cond = torch.zeros(batch, self.cond_ch, *x.shape[2:], device=x.device, dtype=x.dtype)
            x = torch.cat((x, cond.to(x.dtype)), dim=1)
        h = self.embed(self.patchify(x)).reshape(batch, frames * self.N, self.dim)
        rig_h = self.rig_embed(torch.cat((rig_input, rig_cond), dim=-1).to(h.dtype))
        h = torch.cat((h, rig_h), dim=1)
        c = self.temb(timestep_embedding(t, 256))
        if self.text_proj is not None:
            if text is None:
                raise ValueError("text embeddings required")
            text = self.text_proj(text.to(h.dtype))
            weights = text_mask.to(text.dtype).unsqueeze(-1) if text_mask is not None else torch.ones_like(text[..., :1])
            c = c + (text * weights).sum(1) / weights.sum(1).clamp_min(1)

        signed = signed_window_positions(history_frames, frames - history_frames, device=x.device)
        temporal = torch.cat((signed.repeat_interleave(self.N), signed))
        yy, xx = torch.meshgrid(
            torch.arange(self.grid, device=x.device), torch.arange(self.grid, device=x.device), indexing="ij",
        )
        yy = torch.cat((yy.reshape(-1).repeat(frames), torch.full((frames,), self.grid, device=x.device)))
        xx = torch.cat((xx.reshape(-1).repeat(frames), torch.full((frames,), self.grid, device=x.device)))
        positions = torch.stack((temporal, yy, xx), dim=-1).unsqueeze(0).expand(batch, -1, -1)

        # immutable-prefix mask over patch tokens followed by rig tokens
        times = torch.cat((
            torch.arange(frames, device=x.device).repeat_interleave(self.N),
            torch.arange(frames, device=x.device),
        ))
        history_query = times[:, None] < history_frames
        target_key = times[None, :] >= history_frames
        mask = (~(history_query & target_key)).unsqueeze(0).expand(batch, -1, -1)

        for block in self.blocks:
            args = (h, c, text, text_mask, positions, mask)
            h = checkpoint(block, *args, use_reentrant=False) if (self.grad_ckpt and self.training) else block(*args)
        shift, scale = self.ada_f(c).chunk(2, -1)
        h = self.nf(h) * (1 + scale[:, None]) + shift[:, None]
        pixel_h, rig_h = h[:, :frames * self.N], h[:, frames * self.N:]
        return self.unpatchify(self.out(pixel_h), frames), self.rig_out(rig_h)


# ----------------------------------------------------------------- flow
def make_joint_flow_batch(clean: torch.Tensor, rig_clean: torch.Tensor, history: int, shift: float):
    """Noise pixel latents and rig tokens with one shared timestep per sample."""
    timestep = sample_t(clean.shape[0], clean.device, shift)
    amount = timestep[:, None, None, None, None]
    noise = torch.randn_like(clean)
    model_input = (1 - amount) * clean + amount * noise
    rig_noise = torch.randn_like(rig_clean)
    rig_input = (1 - timestep[:, None, None]) * rig_clean + timestep[:, None, None] * rig_noise
    if history:
        model_input[:, :, :history] = clean[:, :, :history]
        rig_input[:, :history] = rig_clean[:, :history]
    return model_input, noise - clean, rig_input, rig_noise - rig_clean, timestep


def rig_condition(rig_clean: torch.Tensor, history: int) -> torch.Tensor:
    """Clean rig history plus a binary mask, mirroring history_condition."""
    clean = torch.zeros_like(rig_clean)
    mask = torch.zeros_like(rig_clean[..., :1])
    if history:
        clean[:, :history] = rig_clean[:, :history]
        mask[:, :history] = 1
    return torch.cat((clean, mask), dim=-1)


# ----------------------------------------------------------------- sampling
@torch.no_grad()
def euler_sample_block_rig(
    model, history, rig_history, target_frames, *, steps, size, text, text_mask,
    null_text, null_mask, cfg, shift, generator,
):
    batch = text.shape[0]
    history_frames = 0 if history is None else history.shape[2]
    if history is None:
        history = torch.empty((batch, model.C, 0, size, size), device=text.device)
        rig_history = torch.empty((batch, 0, model.rig_dim), device=text.device)
    target = torch.randn((batch, model.C, target_frames, size, size), device=text.device, generator=generator)
    rig_target = torch.randn((batch, target_frames, model.rig_dim), device=text.device, generator=generator)
    schedule = torch.linspace(1, 0, steps + 1, device=text.device)
    schedule = shift * schedule / (1 + (shift - 1) * schedule)
    for now, nxt in zip(schedule[:-1], schedule[1:]):
        full = torch.cat((history, target), dim=2)
        rig_full = torch.cat((rig_history, rig_target), dim=1)
        cond = history_condition(full, history_frames)
        rcond = rig_condition(rig_full, history_frames)
        timestep = torch.full((batch,), float(now), device=text.device)
        vel, rvel = model(full, timestep, cond=cond, text=text, text_mask=text_mask,
                          history_frames=history_frames, rig_input=rig_full, rig_cond=rcond)
        vel, rvel = vel[:, :, history_frames:], rvel[:, history_frames:]
        if cfg > 0 and null_text is not None:
            uvel, urvel = model(full, timestep, cond=cond, text=null_text, text_mask=null_mask,
                                history_frames=history_frames, rig_input=rig_full, rig_cond=rcond)
            uvel, urvel = uvel[:, :, history_frames:], urvel[:, history_frames:]
            vel = uvel + cfg * (vel - uvel)
            rvel = urvel + cfg * (rvel - urvel)
        target = target + (nxt - now) * vel
        rig_target = rig_target + (nxt - now) * rvel
    return target, rig_target


@torch.no_grad()
def rollout_blocks_rig(
    model, prompts, *, total_frames, target_frames, history_max, steps,
    null_text, null_mask, cfg, shift, generator,
):
    output = rig_output = None
    block = 0
    while output is None or output.shape[2] < total_frames:
        text, text_mask = prompts[min(block, len(prompts) - 1)]
        history = None if output is None else output[:, :, -history_max:] if history_max else None
        rig_history = None if rig_output is None else rig_output[:, -history_max:] if history_max else None
        generated, rig_generated = euler_sample_block_rig(
            model, history, rig_history, target_frames, steps=steps, size=model.S,
            text=text, text_mask=text_mask, null_text=null_text, null_mask=null_mask,
            cfg=cfg, shift=shift, generator=generator,
        )
        output = generated if output is None else torch.cat((output, generated), dim=2)
        rig_output = rig_generated if rig_output is None else torch.cat((rig_output, rig_generated), dim=1)
        block += 1
    return output[:, :, :total_frames], rig_output[:, :total_frames]


# ----------------------------------------------------------------- previews
@torch.no_grad()
def save_previews(model, codec, standardizer, text_batch, out: Path, step: int, args):
    from eval.post_eval_t2v import save_labeled_gif

    model.eval()
    labels = _default_preview_prompts()
    text, mask = text_batch(labels)
    null_text, null_mask = text_batch([""] * len(labels))
    generator = torch.Generator(device=args.device).manual_seed(20260821)
    latent, rig = rollout_blocks_rig(
        model, [(text, mask)], total_frames=args.rollout_latents,
        target_frames=args.target_latents, history_max=args.history_max,
        steps=args.sample_steps, null_text=null_text, null_mask=null_mask,
        cfg=args.sample_cfg, shift=args.shift, generator=generator,
    )
    rgba = decode_full(codec, standardizer, latent, output_size=args.output_size) * 2 - 1
    save_labeled_gif(rgba.cpu(), str(out / f"fixed_prompt_{step:06d}_labeled.gif"), labels, fps=args.fps)
    np.save(out / f"fixed_prompt_{step:06d}_rig.npy",
            ((rig.cpu().float() + 1) / 2).reshape(rig.shape[0], -1, RIG_JOINTS, 2).numpy())
    model.train()


# ----------------------------------------------------------------- training
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--latent-stats", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--history-max", type=int, default=5)
    parser.add_argument("--target-latents", type=int, default=4)
    parser.add_argument("--history-choices", default="0,1,2,3,4,5")
    parser.add_argument("--rollout-latents", type=int, default=25)
    parser.add_argument("--output-size", type=int, default=64)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--patch", type=int, default=1)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--attention-mode", choices=("full",), default="full")
    parser.add_argument("--training-mode", choices=("block_ar",), default="block_ar")
    parser.add_argument("--start-aligned", action="store_true")
    parser.add_argument("--motion-weight-alpha", type=float, default=1.0)
    parser.add_argument("--fg-latent-weight", type=float, default=4.0)
    parser.add_argument("--rig-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--caption-variants", default="",
        help="JSON {canonical: [variants]} -- when set, each training sample "
             "draws a uniform random surface variant of its caption "
             "(validation and previews keep canonical captions)",
    )
    parser.add_argument(
        "--rig-render-weight", type=float, default=0.0,
        help="weight of the differentiable-render anchor: the recovered clean rig "
             "is rendered as soft capsules (ground-truth joint depth) and its alpha "
             "compared with the sample's ground-truth alpha (gradient to the rig path)",
    )
    parser.add_argument(
        "--rig-pixel-consistency-weight", type=float, default=0.0,
        help="weight of the rig<->pixel coupling: rendered clean-rig alpha compared "
             "with the alpha of the DECODED predicted clean pixels (frozen decoder; "
             "gradient flows into BOTH the rig and pixel paths; ~1.6x step cost)",
    )
    parser.add_argument(
        "--init-checkpoint", default="",
        help="initialize model+EMA weights from a prior v9 checkpoint (fresh optimizer "
             "and schedule) -- the post-training path; empty = train from scratch",
    )
    parser.add_argument(
        "--coupling-warmup-steps", type=int, default=0,
        help="linearly ramp the render and consistency weights from 0 to their "
             "declared values over this many steps (0 = constant from step 0); "
             "lets pixels settle before the coupling gradient arrives",
    )
    parser.add_argument(
        "--bone-length-weight", type=float, default=0.0,
        help="weight of the bone-length preservation loss on the recovered clean "
             "rig (x0 = input - t*v) against the sample's ground-truth bone lengths",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-final", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--cfg-drop", type=float, default=0.1)
    parser.add_argument("--text-encoder", default="google-t5/t5-small")
    parser.add_argument("--text-len", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--sample-cfg", type=float, default=2.0)
    parser.add_argument("--sample-milestones", default="0,500,1000,2000")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tensorboard-dir", default="")
    args = parser.parse_args()
    if not args.start_aligned:
        parser.error("v9 builds on the corrected start-aligned protocol")
    if args.rig_loss_weight <= 0:
        parser.error("--rig-loss-weight must be positive")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if args.fast and args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    histories = _parse_histories(args.history_choices, args.history_max)
    milestones = _parse_steps(args.sample_milestones)
    codec, standardizer, codec_checkpoint, stats = load_codec(args.codec, args.latent_stats, args.device)
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = json.loads(protocol_path.read_text())
    validate_experiment_protocol(protocol, args, codec, smoke=args.smoke)
    if not protocol["frozen_model"].get("rig_cogeneration"):
        raise ValueError("v9 launch requires a protocol declaring rig_cogeneration")
    if float(protocol["frozen_model"].get("rig_loss_weight", -1)) != args.rig_loss_weight:
        raise ValueError("launch differs from predeclared rig_loss_weight")
    if float(protocol["frozen_model"].get("bone_length_weight", 0.0)) != args.bone_length_weight:
        raise ValueError("launch differs from predeclared bone_length_weight")
    if bool(protocol["frozen_model"].get("caption_variants", False)) != bool(args.caption_variants):
        raise ValueError("launch differs from predeclared caption_variants treatment")
    for field, value in (("rig_render_weight", args.rig_render_weight),
                         ("rig_pixel_consistency_weight", args.rig_pixel_consistency_weight)):
        if float(protocol["frozen_model"].get(field, 0.0)) != value:
            raise ValueError(f"launch differs from predeclared {field}")
    if int(protocol["frozen_model"].get("coupling_warmup_steps", 0)) != args.coupling_warmup_steps:
        raise ValueError("launch differs from predeclared coupling_warmup_steps")
    variant_bank = {}
    if args.caption_variants:
        variant_bank = json.loads(Path(args.caption_variants).read_text())
    temporal = codec.temporal_compression
    video_frames = (args.history_max + args.target_latents) * temporal
    latent_size = args.output_size // codec.spatial_compression
    channels = codec.latent_channels

    def make_set(split, *, deterministic, repeats):
        return RigVideoWindows(args.cache, video_frames, split, 1, size=args.output_size,
                               return_text=True, deterministic=deterministic, repeats=repeats)

    train_set = make_set("train", deterministic=False, repeats=4)
    initial_train_set = make_set("train", deterministic=True, repeats=1)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              drop_last=True, pin_memory=True, persistent_workers=args.workers > 0,
                              worker_init_fn=worker_init)
    initial_train_loader = DataLoader(initial_train_set, batch_size=args.batch, shuffle=True,
                                      num_workers=args.workers, drop_last=True, pin_memory=True,
                                      persistent_workers=args.workers > 0, worker_init_fn=worker_init)

    from transformers import AutoTokenizer, T5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    encoder = T5EncoderModel.from_pretrained(args.text_encoder).to(args.device).eval().requires_grad_(False)
    prompt_set = sorted({row["text"] for row in train_set.clips} | {""} |
                        set(_default_preview_prompts()) |
                        {prompt for pair in M6_SWITCH_PAIRS for prompt in pair} |
                        {variant for row in train_set.clips
                         for variant in variant_bank.get(row["text"], [])})
    text_cache = {}
    for start in range(0, len(prompt_set), 32):
        prompts = prompt_set[start:start+32]
        tokens = tokenizer(prompts, padding="max_length", truncation=True,
                           max_length=args.text_len, return_tensors="pt")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = encoder(input_ids=tokens.input_ids.to(args.device),
                             attention_mask=tokens.attention_mask.to(args.device)).last_hidden_state.cpu()
        for prompt, embedding, mask in zip(prompts, hidden, tokens.attention_mask):
            text_cache[prompt] = (embedding, mask)
    text_dim = encoder.config.d_model
    del encoder
    torch.cuda.empty_cache()

    def text_batch(prompts):
        embeddings, masks = zip(*(text_cache[prompt] for prompt in prompts))
        return torch.stack(embeddings).to(args.device), torch.stack(masks).to(args.device)

    renderer = None
    if args.rig_render_weight > 0 or args.rig_pixel_consistency_weight > 0:
        renderer = build_rig_renderer(args.output_size).to(args.device)
    model = RigFullSTARVideoDiT(
        temporal_compression=temporal, size=latent_size, patch=args.patch, in_ch=channels,
        dim=args.dim, depth=args.depth, heads=args.heads, cond_ch=channels + 1, text_dim=text_dim,
    ).to(args.device)
    if args.init_checkpoint:
        init_data = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(init_data["model"])
        print(f"post-training init from {args.init_checkpoint} (step {init_data.get('step')})", flush=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if args.init_checkpoint and "ema" in init_data:
        ema.load_state_dict(init_data["ema"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.01, fused=args.fast)
    train_model = torch.compile(model) if args.compile else model

    codec_meta = {
        "checkpoint": str(Path(args.codec).resolve()), "checkpoint_sha256": file_sha256(Path(args.codec)),
        "stats": str(Path(args.latent_stats).resolve()),
        "latent_stats_sha256": file_sha256(Path(args.latent_stats)),
        "experiment_protocol": str(protocol_path),
        "experiment_protocol_sha256": file_sha256(protocol_path),
        "temporal_compression": temporal, "latent_channels": channels,
    }
    rig_meta_path = Path(args.cache) / "rig_meta.json"
    run_manifest = {
        "protocol": V9_PROTOCOL,
        "base_recipe": "v8 combined (16f blocks + motion weight + fg weight)",
        "rig_loss_weight": args.rig_loss_weight,
        "bone_length_weight": args.bone_length_weight,
        "rig_render_weight": args.rig_render_weight,
        "rig_pixel_consistency_weight": args.rig_pixel_consistency_weight,
        "caption_variants": {
            "enabled": bool(args.caption_variants),
            "path": str(Path(args.caption_variants).resolve()) if args.caption_variants else None,
            "sha256": file_sha256(Path(args.caption_variants)) if args.caption_variants else None,
        },
        "rig_dim_per_token": temporal * RIG_JOINTS * 2,
        "rig_meta_sha256": file_sha256(rig_meta_path),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "codec": codec_meta,
        "source_sha256": {"latent_video_dit_ar_rig.py": file_sha256(Path(__file__))},
    }
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    print(f"v9 {run_manifest['model_parameters']/1e6:.1f}M; rig token dim {run_manifest['rig_dim_per_token']}; "
          f"H={histories}; F={args.target_latents}", flush=True)
    log = (out / "log.txt").open("a")
    writer = None
    if args.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tensorboard_dir)

    def save_ckpt(path, step, full):
        payload = {"ema": ema.state_dict(), "step": step, "args": vars(args),
                   "arch": "latent_full_st_dit_fm_t2v_ar_rig", "protocol": V9_PROTOCOL, "codec": codec_meta}
        if full:
            payload.update(model=model.state_dict(), opt=optimizer.state_dict())
        torch.save(payload, path)

    iterator = iter(train_loader)
    initial_iterator = iter(initial_train_loader)
    started = time.time(); ema_loss = None
    step = 0
    if not args.no_previews and 0 in milestones:
        save_previews(ema, codec, standardizer, text_batch, out, 0, args)
        save_ckpt(out / "ckpt_000000.pt", 0, full=False)

    while step < args.steps:
        history = random.choice(histories)
        use_initial = history == 0
        try:
            video, rig, depth, labels = next(initial_iterator if use_initial else iterator)
        except StopIteration:
            if use_initial:
                initial_iterator = iter(initial_train_loader)
            else:
                iterator = iter(train_loader)
            video, rig, depth, labels = next(initial_iterator if use_initial else iterator)
        video = select_video_history_window(
            video.to(args.device, non_blocking=True), history_latents=history,
            target_latents=args.target_latents, history_max=args.history_max,
            temporal_compression=temporal, initial_block=use_initial,
        )
        rig = select_rig_history_window(
            rig.to(args.device, non_blocking=True), history_latents=history,
            target_latents=args.target_latents, history_max=args.history_max,
            temporal_compression=temporal, initial_block=use_initial,
        )
        depth = select_rig_history_window(
            depth.to(args.device, non_blocking=True), history_latents=history,
            target_latents=args.target_latents, history_max=args.history_max,
            temporal_compression=temporal, initial_block=use_initial,
        )
        clean = encode_video(codec, standardizer, video)
        rig_clean = rig_tokens_from_frames(rig, temporal)
        if clean.shape[2] != history + args.target_latents or rig_clean.shape[1] != clean.shape[2]:
            raise RuntimeError("latent/rig temporal misalignment")
        if variant_bank:
            labels = [random.choice(variant_bank.get(label, [label])) for label in labels]
        text, text_mask = text_batch(labels)
        null_text, null_mask = text_batch([""] * clean.shape[0])
        dropped = torch.rand(clean.shape[0], device=args.device) < args.cfg_drop
        text = torch.where(dropped[:, None, None], null_text, text)
        text_mask = torch.where(dropped[:, None], null_mask, text_mask)
        model_input, flow_target, rig_input, rig_target, timestep = make_joint_flow_batch(
            clean, rig_clean, history, args.shift,
        )
        progress = max(0.0, (step-args.warmup) / max(1, args.steps-args.warmup))
        lr = args.lr * min(1.0, (step+1)/max(1,args.warmup))
        lr *= args.lr_final + (1-args.lr_final)*0.5*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups: group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, rig_prediction = train_model(
                model_input, timestep, cond=history_condition(clean, history),
                text=text, text_mask=text_mask, history_frames=history,
                rig_input=rig_input, rig_cond=rig_condition(rig_clean, history),
            )
            flow_error = (prediction[:, :, history:].float() - flow_target[:, :, history:]).square()
            if args.fg_latent_weight > 1:
                flow_error = flow_error * foreground_latent_weight(
                    video, history, temporal, codec.spatial_compression, args.fg_latent_weight,
                )
            per_sample = flow_error.mean(dim=(1, 2, 3, 4))
            rig_per_sample = (
                rig_prediction[:, history:].float() - rig_target[:, history:]
            ).square().mean(dim=(1, 2))
            if args.motion_weight_alpha > 0:
                segment = clean[:, :, max(history - 1, 0):].float()
                motion = segment.diff(dim=2).abs().mean(dim=(1, 2, 3, 4))
                weight = (motion + 1e-4) ** args.motion_weight_alpha
                weight = (weight / weight.mean().clamp_min(1e-8)).clamp(0.25, 4.0).detach()
                per_sample = weight * per_sample
                rig_per_sample = weight * rig_per_sample
            pixel_loss = per_sample.mean()
            rig_loss = rig_per_sample.mean()
            loss = pixel_loss + args.rig_loss_weight * rig_loss
            bone_loss = render_loss = consistency_loss = None
            need_x0 = args.bone_length_weight > 0 or renderer is not None
            if need_x0:
                amount = timestep[:, None, None]
                rig_x0 = rig_input.float() - amount * rig_prediction.float()
            if args.bone_length_weight > 0:
                bones = rig_bone_lengths(rig_x0[:, history:], temporal)
                bones_gt = rig_bone_lengths(rig_clean[:, history:].float(), temporal)
                bone_loss = (bones - bones_gt).square().mean()
                loss = loss + args.bone_length_weight * bone_loss
            if renderer is not None:
                joints = rig_x0[:, history:].reshape(
                    video.shape[0], -1, RIG_JOINTS, 2)          # [-1,1], target frames
                target_depth = depth[:, history * temporal:].float()
                rendered = renderer(joints, target_depth)
                rendered_alpha = rendered["alpha"].squeeze(2)
                rendered_rgb = rendered["rgb"]                   # premultiplied, like the data
                coupling_scale = (1.0 if args.coupling_warmup_steps <= 0
                                  else min(1.0, step / args.coupling_warmup_steps))
                if args.rig_render_weight > 0:
                    # Target = the SAME soft renderer applied to the ground-truth
                    # rig, so the capsule-vs-procedural approximation cancels and
                    # the loss measures rig error alone (Jin's observation: the
                    # procedural renderer reproduces pixels exactly but is not
                    # differentiable; matching soft-to-soft removes its bias).
                    with torch.no_grad():
                        gt_joints = rig_clean[:, history:].float().reshape(
                            video.shape[0], -1, RIG_JOINTS, 2)
                        gt_rendered = renderer(gt_joints, target_depth)
                    render_loss = fg_weighted_rgba_mse(
                        rendered_alpha, rendered_rgb,
                        gt_rendered["alpha"].squeeze(2), gt_rendered["rgb"],
                    )
                    loss = loss + coupling_scale * args.rig_render_weight * render_loss
                if args.rig_pixel_consistency_weight > 0:
                    predicted_clean = flow_prediction_to_clean(
                        model_input, prediction, timestep,
                        clean_history=clean[:, :, :history] if history else None,
                    )
                    decoded = decode_full(codec, standardizer, predicted_clean,
                                          output_size=args.output_size)[:, :, history * temporal:]
                    consistency_loss = fg_weighted_rgba_mse(
                        rendered_alpha, rendered_rgb,
                        decoded[:, 3], decoded[:, :3].transpose(1, 2),
                    )
                    loss = loss + coupling_scale * args.rig_pixel_consistency_weight * consistency_loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        with torch.no_grad():
            decay = min(0.999 if step < 5000 else 0.9995, (1+step)/(10+step))
            for ep, mp in zip(ema.parameters(), model.parameters()): ep.lerp_(mp, 1-decay)
        step += 1
        value = float(loss.detach()); ema_loss = value if ema_loss is None else .98*ema_loss+.02*value
        if writer:
            writer.add_scalar("train/loss_total", value, step)
            writer.add_scalar("train/pixel_flow_mse", float(pixel_loss.detach()), step)
            writer.add_scalar("train/rig_flow_mse", float(rig_loss.detach()), step)
            if bone_loss is not None:
                writer.add_scalar("train/bone_length_mse", float(bone_loss.detach()), step)
            if render_loss is not None:
                writer.add_scalar("train/rig_render_mse", float(render_loss.detach()), step)
            if consistency_loss is not None:
                writer.add_scalar("train/rig_pixel_consistency_mse", float(consistency_loss.detach()), step)
            writer.add_scalar("train/learning_rate", lr, step)
        if step % 50 == 0 or step == args.steps:
            rate = (time.time()-started) / step
            print(f"step {step} loss {value:.5f} pixel {float(pixel_loss.detach()):.5f} rig {float(rig_loss.detach()):.5f} "
                  f"H {history} lr {lr:.2e} {rate:.3f}s/it ETA {(args.steps-step)*rate/3600:.2f}h", flush=True)
            log.write(f"step {step} loss {value:.5f} pixel {float(pixel_loss.detach()):.5f} rig {float(rig_loss.detach()):.5f} "
                      f"H {history} lr {lr:.2e} {rate:.3f}s/it\n"); log.flush()
        if step % args.save_every == 0 or step == args.steps:
            save_ckpt(out / "latest.pt", step, full=True)
        if step in milestones or step == args.steps:
            save_ckpt(out / f"ckpt_{step:06d}.pt", step, full=False)
            if not args.no_previews:
                save_previews(ema, codec, standardizer, text_batch, out, step, args)
    print("done", flush=True)


if __name__ == "__main__":
    main()
