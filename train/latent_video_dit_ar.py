"""M6: text-conditioned block-autoregressive rectified flow in frozen VAE space.

M6 keeps M4's signed local RoPE, immutable-prefix attention mask, teacher-forced
history, and replaceable per-block text condition.  The deliberate change is
the representation: continuous standardized codec latents replace RGBA pixels.
By default the model consumes ten f8t2 latent blocks (one second at 20 fps) and
generates two new latent blocks (four video frames) per sampling call.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from train.video_ddpm import VideoWindows, worker_init
from train.video_dit_ar import (
    ARVideoDiT,
    FullSTARVideoDiT,
    _make_flow_batch,
    _parse_histories,
    _parse_steps,
    _default_preview_prompts,
    history_condition,
    rollout_blocks,
)
from train.video_vae import DSFCausalVideoVAE


M6_SWITCH_PAIRS = [
    ("A person walks forward.", "A person runs forward."),
    ("A person sits down on a chair.", "A person remains seated and still."),
    ("A person waves hello with the left hand.", "A person stands still and breathes."),
    ("A person jumps up and down.", "A person stands still and breathes."),
]


def m6_switch_prompt_grid(blocks: int, switch_block: int) -> list[list[str]]:
    if blocks <= 1 or not 0 < switch_block < blocks:
        raise ValueError("switch block must lie inside a multi-block rollout")
    return [[before] * switch_block + [after] * (blocks-switch_block)
            for before, after in M6_SWITCH_PAIRS]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_experiment_protocol(
    protocol: dict, args, codec: DSFCausalVideoVAE, *, smoke: bool = False,
) -> None:
    """Fail before training if the launch drifts from the predeclared M6 task."""
    task, model = protocol["task"], protocol["frozen_model"]
    temporal = codec.temporal_compression
    expected = {
        "output_size": int(task["resolution"]),
        "fps": int(task["fps"]),
        "rollout_latents": int(task["generated_video_frames"]) // temporal,
        "target_latents": int(model["target_video_frames_per_block"]) // temporal,
        "history_max": int(model["history_seconds"] * task["fps"]) // temporal,
        "sample_steps": int(model["primary_sampling_steps_per_block"]),
        "steps": int(model["planned_optimizer_steps"]),
        "seed": int(model["training_seed"]),
    }
    if "attention_mode" in model:
        expected["attention_mode"] = str(model["attention_mode"])
    if "training_mode" in model:
        expected["training_mode"] = str(model["training_mode"])
    if "start_aligned" in model:
        expected["start_aligned"] = bool(model["start_aligned"])
    if "decoded_loss_weight" in model:
        expected["decoded_loss_weight"] = float(model["decoded_loss_weight"])
    if "history_noise_max" in model:
        expected["history_noise_max"] = float(model["history_noise_max"])
    if "decoded_background_alpha_weight" in model:
        expected["decoded_background_alpha_weight"] = float(model["decoded_background_alpha_weight"])
    if task["generated_video_frames"] % temporal or model["target_video_frames_per_block"] % temporal:
        raise ValueError("M6 video-frame lengths must divide exactly by codec temporal compression")
    checked = expected if not smoke else {key: value for key, value in expected.items() if key != "steps"}
    mismatches = {
        key: (expected_value, getattr(args, key))
        for key, expected_value in checked.items()
        if getattr(args, key) != expected_value
    }
    if mismatches:
        raise ValueError(f"launch differs from predeclared M6 protocol: {mismatches}")
    if smoke and not (0 < args.steps <= 2 and args.no_previews):
        raise ValueError("M6 smoke must use at most two steps and disable expensive previews")


class LatentStandardizer:
    def __init__(self, mean: list[float], std: list[float], device: str):
        if len(mean) != len(std) or not mean or min(std) <= 0:
            raise ValueError("latent mean/std must be non-empty, matched, and positive")
        self.mean = torch.tensor(mean, device=device).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, -1, 1, 1, 1)

    def encode(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.mean.to(latent.dtype)) / self.std.to(latent.dtype)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.std.to(latent.dtype) + self.mean.to(latent.dtype)


def load_codec(checkpoint_path: str, stats_path: str, device: str):
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    codec = DSFCausalVideoVAE(
        temporal_compression=int(saved["temporal_compression"]),
        spatial_compression=int(saved.get("spatial_compression", 4)),
        latent_channels=int(saved["latent_channels"]),
        base_channels=int(saved["base_channels"]),
        blocks_per_stage=int(saved["blocks_per_stage"]),
    )
    codec.load_state_dict(checkpoint["model"], strict=True)
    codec.to(device).eval().requires_grad_(False)
    stats = json.loads(Path(stats_path).read_text())
    if stats["codec_checkpoint_sha256"] != file_sha256(path):
        raise ValueError("latent stats were computed from a different codec checkpoint")
    standardizer = LatentStandardizer(stats["mean"], stats["std"], device)
    return codec, standardizer, checkpoint, stats


@torch.no_grad()
def encode_video(codec, standardizer: LatentStandardizer, video: torch.Tensor) -> torch.Tensor:
    rgba = ((video + 1) / 2).clamp(0, 1)
    mean, _ = codec.encode(rgba)
    return standardizer.encode(mean)


def select_video_history_window(
    video: torch.Tensor,
    *,
    history_latents: int,
    target_latents: int,
    history_max: int,
    temporal_compression: int,
    initial_block: bool = False,
) -> torch.Tensor:
    """Right-align visible video *before* causal encoding.

    Encoding the maximum window and slicing latents afterwards would leak the
    removed video prefix through the causal encoder's receptive field.  A
    zero-history first block must therefore encode only its target frames.
    """
    if not 0 <= history_latents <= history_max or target_latents <= 0:
        raise ValueError("invalid history/target latent lengths")
    expected = (history_max + target_latents) * temporal_compression
    if video.shape[2] != expected:
        raise ValueError(f"expected {expected} video frames, got {video.shape[2]}")
    keep = (history_latents + target_latents) * temporal_compression
    if initial_block:
        if history_latents != 0:
            raise ValueError("only a zero-history sample can be an initial block")
        return video[:, :, :keep]
    return video[:, :, -keep:]


@torch.no_grad()
def decode_sliding(
    codec: DSFCausalVideoVAE,
    standardizer: LatentStandardizer,
    latent: torch.Tensor,
    *,
    context_latents: int,
    commit_latents: int,
    output_size: int,
) -> torch.Tensor:
    """Bound codec history and commit every decoded frame exactly once."""
    if context_latents <= 0 or commit_latents <= 0:
        raise ValueError("context and commit must be positive")
    if latent.shape[2] < context_latents or (latent.shape[2] - context_latents) % commit_latents:
        raise ValueError("latent length must equal context plus whole commit blocks")
    raw = standardizer.decode(latent)
    temporal = codec.temporal_compression
    first = codec.decode(
        raw[:, :, :context_latents], output_frames=context_latents * temporal,
        output_size=(output_size, output_size),
    )
    parts = [first]
    for end in range(context_latents + commit_latents, latent.shape[2] + 1, commit_latents):
        # Decode the new commit *plus* its bounded past context.  The previous
        # implementation used a total window of context_latents, so a commit
        # larger than the context was silently truncated (H40 produced only
        # 60 of 100 requested video frames).
        start = end - commit_latents - context_latents
        window = raw[:, :, start:end]
        reconstruction = codec.decode(
            window, output_frames=window.shape[2] * temporal, output_size=(output_size, output_size),
        )
        parts.append(reconstruction[:, :, -commit_latents * temporal:])
    return torch.cat(parts, dim=2)


def decode_full(
    codec: DSFCausalVideoVAE, standardizer: LatentStandardizer, latent: torch.Tensor,
    *, output_size: int,
) -> torch.Tensor:
    """Decode the complete causal latent sequence without truncating codec state."""
    raw = standardizer.decode(latent)
    return codec.decode(
        raw, output_frames=latent.shape[2] * codec.temporal_compression,
        output_size=(output_size, output_size),
    )


def decoded_rgba_auxiliary_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    background_alpha_weight: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Foreground RGB plus weak-background alpha supervision in pixel space.

    Both tensors are premultiplied RGBA in ``[0,1]``. Transparent RGB carries
    no meaning, so RGB is scored only in proportion to ground-truth alpha.
    Alpha retains a small background weight to penalize hallucinated opacity
    without letting empty pixels dominate the objective.
    """
    if reconstruction.shape != target.shape or reconstruction.ndim != 5:
        raise ValueError("decoded reconstruction and target must be matched [B,4,T,H,W]")
    if reconstruction.shape[1] != 4 or not 0 <= background_alpha_weight <= 1:
        raise ValueError("decoded loss expects RGBA and a background weight in [0,1]")
    target_alpha = target[:, 3:4].clamp(0, 1)
    rgb_error = (reconstruction[:, :3] - target[:, :3]).abs()
    rgb_fg = (rgb_error * target_alpha).sum() / (target_alpha.sum() * 3).clamp_min(1e-8)
    alpha_error = (reconstruction[:, 3:4] - target_alpha).abs()
    alpha_weight = target_alpha + background_alpha_weight * (1 - target_alpha)
    alpha = (alpha_error * alpha_weight).sum() / alpha_weight.sum().clamp_min(1e-8)
    return {"total": rgb_fg + alpha, "rgb_foreground": rgb_fg, "alpha": alpha}


def flow_prediction_to_clean(
    model_input: torch.Tensor,
    prediction: torch.Tensor,
    timestep: torch.Tensor,
    *,
    clean_history: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recover x0 from a rectified-flow velocity prediction.

    The interpolation is ``x_t=(1-t)x0+t*noise`` and the target velocity is
    ``noise-x0``, hence ``x0=x_t-t*v``.  If teacher-forced history is supplied,
    retain it exactly and apply the reconstruction only to the target suffix.
    """
    if model_input.shape != prediction.shape or timestep.shape != (model_input.shape[0],):
        raise ValueError("flow tensors or timestep batch are incompatible")
    amount = timestep[:, None, None, None, None]
    recovered = model_input - amount * prediction
    if clean_history is not None:
        history = clean_history.shape[2]
        if clean_history.shape[:2] != model_input.shape[:2] or clean_history.shape[3:] != model_input.shape[3:]:
            raise ValueError("clean history must match the flow tensor geometry")
        if history > model_input.shape[2]:
            raise ValueError("clean history is longer than the flow sequence")
        recovered = torch.cat((clean_history, recovered[:, :, history:]), dim=2)
    return recovered


def _protocol_id(
    attention_mode: str, training_mode: str = "block_ar", start_aligned: bool = False,
    decoded_loss_weight: float = 0.0, history_noise_max: float = 0.0,
) -> str:
    if training_mode == "full_clip":
        return "r0_latent_full_clip_v1"
    if decoded_loss_weight > 0:
        if not start_aligned:
            raise ValueError("decoded-RGBA supervision requires the corrected start-aligned protocol")
        if history_noise_max > 0:
            raise ValueError("decoded-RGBA supervision and history-noise augmentation are separate treatments")
        return "m6_latent_block_ar_v4_decoded_rgba_aux"
    if history_noise_max > 0:
        if not start_aligned:
            raise ValueError("history-noise augmentation requires the corrected start-aligned protocol")
        return "m6_latent_block_ar_v5_noisy_history"
    if start_aligned:
        return "m6_latent_block_ar_v3_start_aligned"
    return "m6_latent_block_ar_v2_full_st" if attention_mode == "full" else "m6_latent_block_ar_v1"


def save_checkpoint(path: Path, model, ema, optimizer, step: int, args: dict, codec_meta: dict, *, full: bool):
    payload = {
        "ema": ema.state_dict(), "step": step, "args": args,
        "arch": "latent_full_st_dit_fm_t2v_ar" if args["attention_mode"] == "full" else "latent_dit_fm_t2v_ar",
        "protocol": _protocol_id(
            args["attention_mode"], args.get("training_mode", "block_ar"),
            args.get("start_aligned", False), args.get("decoded_loss_weight", 0.0),
            args.get("history_noise_max", 0.0),
        ),
        "codec": codec_meta,
    }
    if full:
        payload.update(
            model=model.state_dict(), opt=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        )
        if torch.cuda.is_available():
            payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


@torch.no_grad()
def validation_losses(
    model, codec, standardizer, loader, text_batch, args, *, batches=8, initial_loader=None,
):
    model.eval()
    result = {}
    cases = (("full_clip", 0),) if args.history_max == 0 else (("first", 0), ("continuation", args.history_max))
    for name, history in cases:
        values = []
        generator = torch.Generator(device=args.device).manual_seed(20260822)
        use_initial = name == "first" and initial_loader is not None
        case_loader = initial_loader if use_initial else loader
        for index, (video, labels) in enumerate(case_loader):
            if index >= batches:
                break
            video = select_video_history_window(
                video.to(args.device, non_blocking=True), history_latents=history,
                target_latents=args.target_latents, history_max=args.history_max,
                temporal_compression=codec.temporal_compression,
                initial_block=use_initial,
            )
            clean = encode_video(codec, standardizer, video)
            if clean.shape[2] != history + args.target_latents:
                raise RuntimeError("codec produced an unexpected temporal latent length")
            timestep = torch.sigmoid(torch.randn(clean.shape[0], device=args.device, generator=generator))
            amount = timestep[:, None, None, None, None]
            noise = torch.randn(clean.shape, device=args.device, generator=generator)
            model_input = (1 - amount) * clean + amount * noise
            if history:
                model_input[:, :, :history] = clean[:, :, :history]
            text, mask = text_batch(labels)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                pred = model(model_input, timestep, cond=history_condition(clean, history), text=text,
                             text_mask=mask, history_frames=history)
            values.append(float((pred[:, :, history:].float() - (noise-clean)[:, :, history:]).square().mean()))
        result[name] = sum(values) / max(len(values), 1)
    model.train()
    return result


@torch.no_grad()
def save_previews(model, codec, standardizer, text_batch, out: Path, step: int, args):
    from eval.post_eval_t2v import save_labeled_gif, save_strip

    model.eval()
    labels = _default_preview_prompts()
    text, mask = text_batch(labels)
    null_text, null_mask = text_batch([""] * len(labels))
    generator = torch.Generator(device=args.device).manual_seed(20260821)
    started = time.perf_counter()
    latent = rollout_blocks(
        model, [(text, mask)], total_frames=args.rollout_latents,
        target_frames=args.target_latents, history_max=args.history_max,
        steps=args.sample_steps, null_text=null_text, null_mask=null_mask,
        cfg=args.sample_cfg, shift=args.shift, generator=generator, sample_clamp=None,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    sample_seconds = time.perf_counter() - started
    decode_started = time.perf_counter()
    rgba = decode_full(codec, standardizer, latent, output_size=args.output_size) * 2 - 1
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    gif = f"fixed_prompt_{step:06d}_labeled.gif"
    save_labeled_gif(rgba.cpu(), str(out / gif), labels, fps=args.fps)
    save_strip(rgba.cpu(), str(out / f"fixed_prompt_{step:06d}_strip.png"), labels)

    blocks = math.ceil(args.rollout_latents / args.target_latents)
    video_frames_per_block = args.target_latents * codec.temporal_compression
    # Hold the first prompt for two seconds in the canonical five-second
    # preview, independent of codec temporal compression.
    switch_block = None
    schedules = []
    switch_gif = None
    if blocks > 1:
        switch_block = min(blocks - 1, max(1, round(2 * args.fps / video_frames_per_block)))
        schedules = m6_switch_prompt_grid(blocks, switch_block)
        conditions = []
        for block in range(blocks):
            prompts = [schedule[min(block, len(schedule)-1)] for schedule in schedules]
            conditions.append(text_batch(prompts))
        generator = torch.Generator(device=args.device).manual_seed(20260823)
        switched = rollout_blocks(
            model, conditions, total_frames=args.rollout_latents,
            target_frames=args.target_latents, history_max=args.history_max,
            steps=args.sample_steps, null_text=null_text, null_mask=null_mask,
            cfg=args.sample_cfg, shift=args.shift, generator=generator, sample_clamp=None,
        )
        switched_rgba = decode_full(codec, standardizer, switched, output_size=args.output_size) * 2 - 1
        switch_labels = [f"{row[0]} -> {row[-1]}" for row in schedules]
        switch_gif = f"prompt_switch_{step:06d}_labeled.gif"
        save_labeled_gif(switched_rgba.cpu(), str(out / switch_gif), switch_labels, fps=args.fps)
        save_strip(switched_rgba.cpu(), str(out / f"prompt_switch_{step:06d}_strip.png"), switch_labels)
    manifest = {
        "step": step, "seed_fixed": 20260821, "seed_switch": 20260823,
        "history_latents": args.history_max, "target_latents": args.target_latents,
        "video_frames_per_target": args.target_latents * codec.temporal_compression,
        "rollout_video_frames": args.rollout_latents * codec.temporal_compression,
        "sampling_steps_per_block": args.sample_steps, "cfg": args.sample_cfg,
        "fixed_sampling_seconds": sample_seconds,
        "fixed_decode_seconds": decode_seconds,
        "fixed_end_to_end_seconds": sample_seconds + decode_seconds,
        "generated_rows": len(labels),
        "seconds_per_generated_video_frame": (sample_seconds + decode_seconds) /
            (len(labels) * args.rollout_latents * codec.temporal_compression),
        "fps": args.fps,
        "fixed_prompts": labels, "switch_block": switch_block,
        "decode_policy": "full causal sequence; sliding-window decode is evaluated separately",
        "switch_time_seconds": None if switch_block is None else switch_block * video_frames_per_block / args.fps,
        "switch_schedules": schedules,
        "outputs": [name for name in (gif, switch_gif) if name is not None],
    }
    (out / f"sample_manifest_{step:06d}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model.train()
    return manifest


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--latent-stats", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--protocol", default="configs/m6_protocol_v2_full_st.json")
    parser.add_argument("--resume", default="")
    parser.add_argument("--history-max", type=int, default=10)
    parser.add_argument("--target-latents", type=int, default=2)
    parser.add_argument("--history-choices", default="0,2,4,6,8,10")
    parser.add_argument("--rollout-latents", type=int, default=50)
    parser.add_argument("--output-size", type=int, default=64)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--patch", type=int, default=1)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--attention-mode", choices=("full", "factorized"), default="full")
    parser.add_argument("--training-mode", choices=("block_ar", "full_clip"), default="block_ar")
    parser.add_argument(
        "--start-aligned", action="store_true",
        help="train H=0 on the true clip/command start instead of a random late crop",
    )
    parser.add_argument(
        "--decoded-loss-weight", type=float, default=0.0,
        help="weight of frozen-decoder foreground-RGB plus alpha auxiliary loss",
    )
    parser.add_argument(
        "--decoded-background-alpha-weight", type=float, default=0.02,
        help="relative alpha-loss weight for transparent target pixels",
    )
    parser.add_argument(
        "--history-noise-max", type=float, default=0.0,
        help="max flow-interpolation noise level applied to the teacher-forced history "
             "(uniform per sample; robustness augmentation, no extra conditioning)",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-final", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--cfg-drop", type=float, default=0.1)
    parser.add_argument("--text-encoder", default="google-t5/t5-small")
    parser.add_argument("--text-len", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--sample-cfg", type=float, default=2.0)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--sample-milestones", default="0,10,50,100,250,500,1000,2000,5000,10000")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true",
                        help="permit a <=2-step no-preview capacity check without changing the full protocol")
    parser.add_argument("--tensorboard-dir", default="")
    args = parser.parse_args()

    if args.decoded_loss_weight < 0:
        parser.error("--decoded-loss-weight must be nonnegative")
    if not 0 <= args.history_noise_max < 1:
        parser.error("--history-noise-max must lie in [0, 1)")
    if not 0 <= args.decoded_background_alpha_weight <= 1:
        parser.error("--decoded-background-alpha-weight must lie in [0,1]")

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
    temporal = codec.temporal_compression
    video_frames = (args.history_max + args.target_latents) * temporal
    latent_size = args.output_size // codec.spatial_compression
    channels = codec.latent_channels

    train_set = VideoWindows(args.cache, video_frames, "train", 1, size=args.output_size,
                             return_text=True, repeats=4)
    val_set = VideoWindows(args.cache, video_frames, "val", 1, size=args.output_size,
                           return_text=True, deterministic=True, repeats=1)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              drop_last=True, pin_memory=True, persistent_workers=args.workers > 0,
                              worker_init_fn=worker_init)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=min(2, args.workers), drop_last=True)
    initial_train_loader = initial_val_loader = None
    if args.start_aligned:
        # These loaders expose the same maximum-size source window, but pin its
        # crop offset to zero.  select_video_history_window then takes its left
        # edge for H=0.  Continuation examples keep the independently sampled
        # loader above, so this changes only the first-block distribution.
        initial_train_set = VideoWindows(
            args.cache, video_frames, "train", 1, size=args.output_size,
            return_text=True, deterministic=True, repeats=1,
        )
        initial_val_set = VideoWindows(
            args.cache, video_frames, "val", 1, size=args.output_size,
            return_text=True, deterministic=True, repeats=1,
        )
        initial_train_loader = DataLoader(
            initial_train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
            drop_last=True, pin_memory=True, persistent_workers=args.workers > 0,
            worker_init_fn=worker_init,
        )
        initial_val_loader = DataLoader(
            initial_val_set, batch_size=args.batch, shuffle=False,
            num_workers=min(2, args.workers), drop_last=True,
        )

    from transformers import AutoTokenizer, T5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    encoder = T5EncoderModel.from_pretrained(args.text_encoder).to(args.device).eval().requires_grad_(False)
    prompt_set = sorted({row["text"] for row in train_set.clips + val_set.clips} | {""} |
                        set(_default_preview_prompts()) |
                        {prompt for pair in M6_SWITCH_PAIRS for prompt in pair})
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

    model_class = FullSTARVideoDiT if args.attention_mode == "full" else ARVideoDiT
    model = model_class(size=latent_size, patch=args.patch, in_ch=channels, dim=args.dim,
                        depth=args.depth, heads=args.heads, cond_ch=channels+1, text_dim=text_dim).to(args.device)
    model.grad_ckpt = args.grad_ckpt
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.01, fused=args.fast)
    step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=args.device, weights_only=False)
        if saved.get("protocol") != _protocol_id(
            args.attention_mode, args.training_mode, args.start_aligned, args.decoded_loss_weight,
            args.history_noise_max,
        ):
            raise ValueError("resume checkpoint is not the selected M6 attention protocol")
        expected_codec_sha = file_sha256(Path(args.codec))
        if saved.get("codec", {}).get("checkpoint_sha256") != expected_codec_sha:
            raise ValueError("resume checkpoint uses a different frozen codec")
        expected_stats_sha = file_sha256(Path(args.latent_stats))
        if saved.get("codec", {}).get("latent_stats_sha256") != expected_stats_sha:
            raise ValueError("resume checkpoint uses different latent statistics")
        expected_protocol_sha = file_sha256(Path(args.protocol))
        if saved.get("codec", {}).get("experiment_protocol_sha256") != expected_protocol_sha:
            raise ValueError("resume checkpoint uses a different experiment protocol")
        compatible = (
            "history_max", "target_latents", "history_choices", "output_size", "patch",
            "dim", "depth", "heads", "shift", "text_encoder", "text_len",
            "attention_mode",
            "training_mode",
            "start_aligned",
            "decoded_loss_weight",
            "decoded_background_alpha_weight",
        )
        mismatches = {key: (saved["args"].get(key), getattr(args, key)) for key in compatible
                      if saved["args"].get(key) != getattr(args, key)}
        if mismatches:
            raise ValueError(f"resume architecture/protocol mismatch: {mismatches}")
        model.load_state_dict(saved["model"]); ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["opt"]); step = int(saved["step"])
        if "torch_rng" in saved: torch.set_rng_state(saved["torch_rng"].cpu())
        if "cuda_rng" in saved: torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if "numpy_rng" in saved: np.random.set_state(saved["numpy_rng"])
        if "python_rng" in saved: random.setstate(saved["python_rng"])
    train_model = torch.compile(model) if args.compile else model
    codec_meta = {
        "checkpoint": str(Path(args.codec).resolve()), "checkpoint_sha256": file_sha256(Path(args.codec)),
        "model_sha256": codec_checkpoint.get("model_sha256"), "variant": stats["variant"],
        "stats": str(Path(args.latent_stats).resolve()), "spatial_compression": codec.spatial_compression,
        "latent_stats_sha256": file_sha256(Path(args.latent_stats)),
        "temporal_compression": temporal, "latent_channels": channels,
        "experiment_protocol": str(protocol_path),
        "experiment_protocol_sha256": file_sha256(protocol_path),
    }
    parameters = sum(p.numel() for p in model.parameters())
    steps_per_epoch = len(train_set) / args.batch
    run_manifest = {
        "protocol": _protocol_id(
            args.attention_mode, args.training_mode, args.start_aligned, args.decoded_loss_weight,
            args.history_noise_max,
        ),
        "attention_mode": args.attention_mode,
        "training_mode": args.training_mode,
        "start_aligned": args.start_aligned,
        "history_noise_max": args.history_noise_max,
        "decoded_rgba_auxiliary": {
            "enabled": args.decoded_loss_weight > 0,
            "weight": args.decoded_loss_weight,
            "background_alpha_weight": args.decoded_background_alpha_weight,
            "decoder_parameters_frozen": True,
            "rgb_support": "ground-truth alpha",
        },
        "smoke_only": bool(args.smoke),
        "experiment_protocol": {
            "path": str(protocol_path), "sha256": file_sha256(protocol_path),
            "declared_before_training_results": bool(protocol.get("declared_before_training_results")),
        },
        "model_parameters": parameters,
        "codec": codec_meta,
        "train_clips": len(train_set.clips),
        "train_items_per_epoch": len(train_set),
        "batch": args.batch,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "planned_steps": args.steps,
        "planned_equivalent_epochs": args.steps / steps_per_epoch,
        "latent_window": [args.history_max + args.target_latents, latent_size, latent_size, channels],
        "tokens_per_max_window": (args.history_max + args.target_latents) *
            (latent_size // args.patch) ** 2,
        "target_video_frames": args.target_latents * temporal,
        "source_sha256": {
            "latent_video_dit_ar.py": file_sha256(Path(__file__)),
            "video_dit_ar.py": file_sha256(Path(__file__).with_name("video_dit_ar.py")),
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    print(f"M6 {sum(p.numel() for p in model.parameters())/1e6:.1f}M; codec {stats['variant']}; "
          f"latent {latent_size}x{latent_size}x{channels}; H={histories}; F={args.target_latents}", flush=True)
    log = (out / "log.txt").open("a")
    writer = None
    if args.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tensorboard_dir, purge_step=step or None)
        writer.add_text("run/manifest", "```json\n" + json.dumps(run_manifest, indent=2) + "\n```", step)
    iterator = iter(train_loader)
    initial_iterator = iter(initial_train_loader) if initial_train_loader is not None else None
    started = time.time(); initial_step = step; ema_loss = None
    if not args.no_previews and step == 0 and 0 in milestones:
        preview = save_previews(ema, codec, standardizer, text_batch, out, 0, args)
        if writer:
            for key in ("fixed_sampling_seconds", "fixed_decode_seconds", "fixed_end_to_end_seconds"):
                writer.add_scalar(f"sampling/{key}", preview[key], 0)
            writer.add_text("artifacts/fixed_prompt_gif", str(out / preview["outputs"][0]), 0)
            if len(preview["outputs"]) > 1:
                writer.add_text("artifacts/prompt_switch_gif", str(out / preview["outputs"][1]), 0)
        save_checkpoint(out / "ckpt_000000.pt", model, ema, optimizer, 0, vars(args), codec_meta, full=False)

    while step < args.steps:
        history = random.choice(histories)
        use_initial = history == 0 and initial_train_loader is not None
        if use_initial:
            try:
                video, labels = next(initial_iterator)
            except StopIteration:
                initial_iterator = iter(initial_train_loader)
                video, labels = next(initial_iterator)
        else:
            try:
                video, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                video, labels = next(iterator)
        video = select_video_history_window(
            video.to(args.device, non_blocking=True), history_latents=history,
            target_latents=args.target_latents, history_max=args.history_max,
            temporal_compression=temporal,
            initial_block=use_initial,
        )
        clean = encode_video(codec, standardizer, video)
        if clean.shape[2] != history + args.target_latents:
            raise RuntimeError("codec produced an unexpected temporal latent length")
        text, text_mask = text_batch(labels)
        null_text, null_mask = text_batch([""] * clean.shape[0])
        dropped = torch.rand(clean.shape[0], device=args.device) < args.cfg_drop
        text = torch.where(dropped[:, None, None], null_text, text)
        text_mask = torch.where(dropped[:, None], null_mask, text_mask)
        conditioned = clean
        if args.history_noise_max > 0 and history:
            strength = torch.rand(clean.shape[0], 1, 1, 1, 1, device=clean.device) * args.history_noise_max
            conditioned = clean.clone()
            conditioned[:, :, :history] = (
                (1 - strength) * clean[:, :, :history]
                + strength * torch.randn_like(clean[:, :, :history])
            )
        model_input, flow_target, timestep = _make_flow_batch(conditioned, history, args.shift)
        progress = max(0.0, (step-args.warmup) / max(1, args.steps-args.warmup))
        lr = args.lr * min(1.0, (step+1)/max(1,args.warmup))
        lr *= args.lr_final + (1-args.lr_final)*0.5*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups: group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = train_model(model_input, timestep, cond=history_condition(conditioned, history),
                                     text=text, text_mask=text_mask, history_frames=history)
            latent_loss = (
                prediction[:, :, history:].float() - flow_target[:, :, history:]
            ).square().mean()
            decoded_aux = None
            loss = latent_loss
            if args.decoded_loss_weight > 0:
                predicted_clean = flow_prediction_to_clean(
                    model_input, prediction, timestep,
                    clean_history=clean[:, :, :history] if history else None,
                )
                decoded = decode_full(
                    codec, standardizer, predicted_clean, output_size=args.output_size,
                )
                target_rgba = ((video + 1) / 2).clamp(0, 1)
                target_video_frames = args.target_latents * temporal
                decoded_aux = decoded_rgba_auxiliary_loss(
                    decoded[:, :, -target_video_frames:],
                    target_rgba[:, :, -target_video_frames:],
                    background_alpha_weight=args.decoded_background_alpha_weight,
                )
                loss = latent_loss + args.decoded_loss_weight * decoded_aux["total"]
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        with torch.no_grad():
            decay = min(0.999 if step < 5000 else 0.9995, (1+step)/(10+step))
            for ep, mp in zip(ema.parameters(), model.parameters()): ep.lerp_(mp, 1-decay)
        step += 1
        value = float(loss.detach()); ema_loss = value if ema_loss is None else .98*ema_loss+.02*value
        if writer:
            writer.add_scalar("train/loss_total", value, step)
            writer.add_scalar("train/loss_total_ema", ema_loss, step)
            writer.add_scalar("train/flow_mse", float(latent_loss.detach()), step)
            if decoded_aux is not None:
                writer.add_scalar("train/decoded_rgba", float(decoded_aux["total"].detach()), step)
                writer.add_scalar("train/decoded_rgb_foreground", float(decoded_aux["rgb_foreground"].detach()), step)
                writer.add_scalar("train/decoded_alpha", float(decoded_aux["alpha"].detach()), step)
            writer.add_scalar("train/learning_rate", lr, step)
            writer.add_scalar("train/history_latents", history, step)
            writer.add_scalar("train/equivalent_epoch", step / steps_per_epoch, step)
            if args.device.startswith("cuda"):
                writer.add_scalar("system/peak_allocated_gib", torch.cuda.max_memory_allocated()/2**30, step)
        if step == 1 or step % 50 == 0:
            seconds = (time.time()-started)/max(1,step-initial_step)
            epoch = step / steps_per_epoch
            message = (f"step {step} epoch {epoch:.2f} loss {ema_loss:.5f} H {history} "
                       f"lr {lr:.2e} {seconds:.3f}s/it")
            if decoded_aux is not None:
                message += (f" latent {float(latent_loss.detach()):.5f}"
                            f" decoded {float(decoded_aux['total'].detach()):.5f}")
            message += f" peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB ETA {(args.steps-step)*seconds/3600:.2f}h"
            if step % args.val_every == 0:
                values = validation_losses(
                    ema, codec, standardizer, val_loader, text_batch, args,
                    initial_loader=initial_val_loader,
                )
                message += " " + " ".join(f"val_{name} {value:.5f}" for name, value in values.items())
                if writer:
                    for name, value in values.items():
                        writer.add_scalar(f"validation/{name}_flow_mse", value, step)
            print(message, flush=True); log.write(message+"\n"); log.flush()
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(out / "latest.pt", model, ema, optimizer, step, vars(args), codec_meta, full=True)
        if not args.no_previews and (step in milestones or step % args.sample_every == 0 or step == args.steps):
            preview = save_previews(ema, codec, standardizer, text_batch, out, step, args)
            if writer:
                for key in ("fixed_sampling_seconds", "fixed_decode_seconds", "fixed_end_to_end_seconds"):
                    writer.add_scalar(f"sampling/{key}", preview[key], step)
                writer.add_text("artifacts/fixed_prompt_gif", str(out / preview["outputs"][0]), step)
                if len(preview["outputs"]) > 1:
                    writer.add_text("artifacts/prompt_switch_gif", str(out / preview["outputs"][1]), step)
                writer.flush()
            save_checkpoint(out / f"ckpt_{step:06d}.pt", model, ema, optimizer, step,
                            vars(args), codec_meta, full=False)

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
