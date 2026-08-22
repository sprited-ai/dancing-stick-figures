"""Train and visibly audit the matched DSF causal Video-VAE variants.

Example reconstruction pilot::

    python -m train.video_vae_train \
      --cache data/v1_cache --out runs/vae_t4 --temporal-compression 4 \
      --frames 20 --size 64 --batch 4 --steps 5000

The loop intentionally saves dense early milestones.  Every reconstruction GIF
contains the prompt plus side-by-side original/reconstruction columns, and each
milestone has an immutable checkpoint and JSON manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from train.video_ddpm import VideoWindows, worker_init
from train.video_vae import DSFCausalVideoVAE, video_vae_loss


DEFAULT_MILESTONES = "0,1,5,10,25,50,100,250,500,1000,2000,3000,5000,10000"


def parse_steps(spec: str) -> set[int]:
    try:
        values = {int(item.strip()) for item in spec.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(f"invalid milestone list: {spec!r}") from exc
    if any(step < 0 for step in values):
        raise ValueError("milestones must be non-negative")
    return values


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _composite_white(video: torch.Tensor) -> np.ndarray:
    """Premultiplied RGBA [4,T,H,W] in [0,1] -> uint8 RGB [T,H,W,3]."""
    value = video.detach().float().clamp(0, 1).cpu()
    rgb = value[:3] + (1 - value[3:4])
    return (rgb.permute(1, 2, 3, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)


def save_reconstruction_gif(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    prompts: list[str],
    path: Path,
    *,
    step: int,
    fps: int = 20,
) -> None:
    """Write labeled original/reconstruction pairs for a fixed validation batch."""
    originals = [_composite_white(video) for video in original]
    reconstructions = [_composite_white(video) for video in reconstruction]
    _, height, width, _ = originals[0].shape
    banner = 34
    frames = []
    for frame_index in range(original.shape[2]):
        canvas = Image.new("RGB", (2 * width, len(prompts) * (height + banner)), "white")
        draw = ImageDraw.Draw(canvas)
        for row, prompt in enumerate(prompts):
            top = row * (height + banner)
            draw.text((3, top + 2), f"step {step} | {prompt[:70]}", fill="black")
            draw.text((3, top + 17), "ORIGINAL", fill="black")
            draw.text((width + 3, top + 17), "RECONSTRUCTION", fill="black")
            canvas.paste(Image.fromarray(originals[row][frame_index]), (0, top + banner))
            canvas.paste(Image.fromarray(reconstructions[row][frame_index]), (width, top + banner))
        frames.append(np.asarray(canvas))
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)


@torch.no_grad()
def evaluate(
    model: DSFCausalVideoVAE,
    batch: torch.Tensor,
    *,
    alpha_background_weight: float,
    rgb_velocity_weight: float,
    alpha_velocity_weight: float,
    rgb_acceleration_weight: float,
    alpha_acceleration_weight: float,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    model.eval()
    output = model(batch, sample=False)
    losses = video_vae_loss(
        output,
        batch,
        alpha_background_weight=alpha_background_weight,
        rgb_velocity_weight=rgb_velocity_weight,
        alpha_velocity_weight=alpha_velocity_weight,
        rgb_acceleration_weight=rgb_acceleration_weight,
        alpha_acceleration_weight=alpha_acceleration_weight,
        kl_weight=kl_weight,
    )
    return output.reconstruction, {key: float(value.detach()) for key, value in losses.items()}


def save_milestone(
    model: DSFCausalVideoVAE,
    optimizer: torch.optim.Optimizer,
    fixed_video: torch.Tensor,
    fixed_prompts: list[str],
    out: Path,
    args: argparse.Namespace,
    step: int,
    kl_weight: float,
) -> dict[str, float]:
    reconstruction, metrics = evaluate(
        model,
        fixed_video,
        alpha_background_weight=args.alpha_background_weight,
        rgb_velocity_weight=args.rgb_velocity_weight,
        alpha_velocity_weight=args.alpha_velocity_weight,
        rgb_acceleration_weight=args.rgb_acceleration_weight,
        alpha_acceleration_weight=args.alpha_acceleration_weight,
        kl_weight=kl_weight,
    )
    gif_name = f"reconstruction_{step:06d}.gif"
    save_reconstruction_gif(
        fixed_video,
        reconstruction,
        fixed_prompts,
        out / gif_name,
        step=step,
        fps=args.fps,
    )
    checkpoint_name = f"ckpt_{step:06d}.pt"
    checkpoint = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
        "model_sha256": state_sha256(model.state_dict()),
    }
    torch.save(checkpoint, out / checkpoint_name)
    manifest = {
        "step": step,
        "variant": f"f{args.spatial_compression}t{args.temporal_compression}d{args.latent_channels}",
        "input_shape": list(fixed_video.shape),
        "prompts": fixed_prompts,
        "posterior": "mean",
        "fps": args.fps,
        "kl_weight": kl_weight,
        "metrics": metrics,
        "checkpoint": checkpoint_name,
        "model_sha256": checkpoint["model_sha256"],
        "gif": gif_name,
    }
    (out / f"manifest_{step:06d}.json").write_text(json.dumps(manifest, indent=2))
    model.train()
    return metrics


def choose_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--temporal-compression", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--spatial-compression", type=int, choices=(4, 8), default=4)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--blocks-per-stage", type=int, default=2)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", default="", help="checkpoint to continue from")
    parser.add_argument(
        "--allow-window-change",
        action="store_true",
        help="allow only --frames to differ when fine-tuning a verified checkpoint",
    )
    parser.add_argument("--posterior-mode", choices=("mean", "sample"), default="sample")
    parser.add_argument("--milestones", default=DEFAULT_MILESTONES)
    parser.add_argument("--fixed-samples", type=int, default=2)
    parser.add_argument("--alpha-background-weight", type=float, default=1.0)
    parser.add_argument("--rgb-velocity-weight", type=float, default=0.0)
    parser.add_argument("--alpha-velocity-weight", type=float, default=0.0)
    parser.add_argument("--rgb-acceleration-weight", type=float, default=0.0)
    parser.add_argument("--alpha-acceleration-weight", type=float, default=0.0)
    parser.add_argument("--kl-max", type=float, default=3e-6)
    parser.add_argument("--kl-warmup", type=int, default=5000)
    parser.add_argument("--tensorboard-dir", default="",
                        help="optional SummaryWriter directory; scalar logging must never replace JSON artifacts")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    train_data = VideoWindows(
        args.cache,
        frames=args.frames,
        split="train",
        size=args.size,
        repeats=8,
        return_text=True,
    )
    val_data = VideoWindows(
        args.cache,
        frames=args.frames,
        split="val",
        size=args.size,
        deterministic=True,
        repeats=1,
        return_text=True,
    )
    if not train_data or not val_data:
        raise RuntimeError("cache must contain non-empty train and val clips")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_data,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init if args.workers else None,
        generator=generator,
        drop_last=True,
    )
    fixed_loader = DataLoader(val_data, batch_size=args.fixed_samples, shuffle=False, num_workers=0)
    fixed_video, fixed_prompts = next(iter(fixed_loader))
    fixed_video = ((fixed_video.to(device) + 1) / 2).clamp(0, 1)
    fixed_prompts = list(fixed_prompts)

    model = DSFCausalVideoVAE(
        temporal_compression=args.temporal_compression,
        spatial_compression=args.spatial_compression,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        blocks_per_stage=args.blocks_per_stage,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)
    start_step = 0
    resume_record = None
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        required = {"step", "model", "optimizer", "args", "model_sha256"}
        missing = required - checkpoint.keys()
        if missing:
            raise ValueError(f"resume checkpoint is missing keys: {sorted(missing)}")
        compatible = (
            "temporal_compression", "spatial_compression", "latent_channels", "base_channels",
            "blocks_per_stage", "frames", "size", "fps", "posterior_mode",
            "alpha_background_weight", "rgb_velocity_weight", "alpha_velocity_weight",
            "rgb_acceleration_weight", "alpha_acceleration_weight", "kl_max", "kl_warmup",
        )
        mismatches = {
            key: (checkpoint["args"].get(key), getattr(args, key))
            for key in compatible
            if checkpoint["args"].get(key) != getattr(args, key)
        }
        if args.allow_window_change:
            mismatches.pop("frames", None)
        if mismatches:
            raise ValueError(f"resume architecture/data mismatch: {mismatches}")
        if state_sha256(checkpoint["model"]) != checkpoint["model_sha256"]:
            raise ValueError("resume checkpoint model checksum does not match")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        if start_step >= args.steps:
            raise ValueError(f"resume step {start_step} must be below target steps {args.steps}")
        resume_record = {
            "path": str(resume_path),
            "step": start_step,
            "model_sha256": checkpoint["model_sha256"],
            "note": "Legacy checkpoint has no RNG/DataLoader cursor; data order restarts from the declared seed.",
            "window_change": {
                "from_frames": checkpoint["args"].get("frames"),
                "to_frames": args.frames,
            } if checkpoint["args"].get("frames") != args.frames else None,
        }
        (out / "resume.json").write_text(json.dumps(resume_record, indent=2))

    initialization_sha = state_sha256(model.state_dict())
    (out / "initialization.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "sha256": initialization_sha,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "resumed_from": resume_record,
            },
            indent=2,
        )
    )
    train_model = torch.compile(model) if args.compile else model
    use_amp = device.type == "cuda"
    milestones = parse_steps(args.milestones) | {0, args.steps}
    log_path = out / "log.jsonl"
    writer = None
    if args.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tensorboard_dir, purge_step=start_step or None)
        writer.add_text("run/args", "```json\n" + json.dumps(vars(args), indent=2) + "\n```", start_step)

    if start_step == 0 and 0 in milestones:
        metrics = save_milestone(model, optimizer, fixed_video, fixed_prompts, out, args, 0, 0.0)
        print("milestone", 0, metrics, flush=True)
        if writer:
            for key, value in metrics.items():
                writer.add_scalar(f"fixed_validation/{key}", value, 0)
            writer.add_text("artifacts/reconstruction_gif", str(out / "reconstruction_000000.gif"), 0)

    iterator = iter(loader)
    start = time.time()
    for step in range(start_step + 1, args.steps + 1):
        try:
            video, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            video, _ = next(iterator)
        video = ((video.to(device, non_blocking=True) + 1) / 2).clamp(0, 1)
        kl_weight = args.kl_max * min(1.0, step / max(1, args.kl_warmup))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            output = train_model(video, sample=args.posterior_mode == "sample")
            losses = video_vae_loss(
                output,
                video,
                alpha_background_weight=args.alpha_background_weight,
                rgb_velocity_weight=args.rgb_velocity_weight,
                alpha_velocity_weight=args.alpha_velocity_weight,
                rgb_acceleration_weight=args.rgb_acceleration_weight,
                alpha_acceleration_weight=args.alpha_acceleration_weight,
                kl_weight=kl_weight,
            )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        record = {
            "step": step,
            "elapsed": time.time() - start,
            "kl_weight": kl_weight,
            **{key: float(value.detach()) for key, value in losses.items()},
        }
        if device.type == "cuda":
            record["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if writer:
            for key, value in record.items():
                if key not in ("step", "elapsed") and isinstance(value, (int, float)):
                    writer.add_scalar(f"train/{key}", value, step)
            writer.add_scalar("system/elapsed_seconds", record["elapsed"], step)
        if step == 1 or step % 25 == 0:
            print(json.dumps(record), flush=True)
        if step in milestones:
            metrics = save_milestone(model, optimizer, fixed_video, fixed_prompts, out, args, step, kl_weight)
            print("milestone", step, metrics, flush=True)
            if writer:
                for key, value in metrics.items():
                    writer.add_scalar(f"fixed_validation/{key}", value, step)
                writer.add_text(
                    "artifacts/reconstruction_gif", str(out / f"reconstruction_{step:06d}.gif"), step,
                )
                writer.flush()

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
