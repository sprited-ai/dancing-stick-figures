"""Generate the Figure 1 prompt panel from a latent Mini-Wan checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from eval.post_eval_t2v import (
    encode_prompts,
    load_prompts,
    save_labeled_gif,
    save_strip,
    _straight_rgba_uint8,
)
from scripts.encode_latent_cache import load_codec
from train.video_ddpm import to_gif
from train.wan_mini import build_model, euler_sample


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint_path = Path(args.ckpt)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(prompts_file=args.prompts_file)[:4]
    if len(prompts) != 4:
        raise ValueError("Figure 1 requires exactly four prompts")

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    train_args = checkpoint["args"]
    if checkpoint.get("arch") != "wan_mini_lat":
        raise ValueError("checkpoint is not a latent Mini-Wan model")

    model = build_model(Namespace(**train_args)).to(args.device)
    model.load_state_dict(checkpoint["ema"])
    model.eval().requires_grad_(False)

    latent_meta_path = Path(train_args["cache"]) / "meta.json"
    latent_meta = json.loads(latent_meta_path.read_text())
    codec, _, _ = load_codec(latent_meta["codec_ckpt"], args.device)
    codec.eval().requires_grad_(False)
    mean = torch.tensor(latent_meta["mean"], device=args.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(latent_meta["std"], device=args.device).view(1, -1, 1, 1, 1)

    text, _, null_text, _, encoder_name = encode_prompts(prompts, train_args, args.device)
    seeds = [args.seed + 1000 + index for index in range(4)]
    decoded = []
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=args.device).manual_seed(seed)
        latent = euler_sample(
            model,
            (1, train_args["channels"], train_args["frames"], train_args["size"], train_args["size"]),
            args.device,
            text[index : index + 1],
            None,
            null_text[index : index + 1],
            None,
            steps=args.steps,
            cfg=args.cfg,
            generator=generator,
        )
        rgba = codec.decode(
            latent * std + mean,
            output_frames=train_args["frames"] * int(latent_meta["temporal_compression"]),
            output_size=(train_args["size"] * int(latent_meta["spatial_compression"]),) * 2,
        ).clamp(0, 1)
        decoded.append(rgba * 2 - 1)
    videos = torch.cat(decoded)

    gif_path = output / "prompt_diverse_grid.gif"
    labeled_path = output / "prompt_diverse_grid_labeled.gif"
    strip_path = output / "prompt_diverse_grid_strip.png"
    rgba_path = output / "prompt_diverse_grid_rgba.npz"
    to_gif(videos, str(gif_path), fps=args.fps)
    save_labeled_gif(videos, str(labeled_path), prompts, fps=args.fps)
    frames = save_strip(videos, str(strip_path), prompts, (0, 21, 42, 63))
    np.savez_compressed(
        rgba_path,
        rgba=_straight_rgba_uint8(videos),
        prompts=np.asarray(prompts),
        seeds=np.asarray(seeds, dtype=np.int64),
    )
    manifest = {
        "protocol": "figure1_mini_wan_qualitative_v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "architecture": checkpoint["arch"],
        "codec": latent_meta["codec_ckpt"],
        "codec_sha256": latent_meta.get("codec_ckpt_sha256"),
        "shape": [4, videos.shape[2], videos.shape[3], videos.shape[4]],
        "sampler": {"name": "rectified_flow_euler", "steps": args.steps, "cfg": args.cfg, "fps": args.fps},
        "prompts": prompts,
        "noise_seeds": seeds,
        "strip_frames": frames,
        "text_encoder": encoder_name,
        "outputs": [gif_path.name, labeled_path.name, strip_path.name, rgba_path.name],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
