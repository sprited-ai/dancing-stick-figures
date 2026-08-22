"""Compute deterministic per-channel statistics for a frozen DSF video codec.

M6 trains rectified flow in standardized continuous latent space.  Statistics
are estimated from train-only windows and stored with the codec checksum so a
run cannot silently mix a codec and an incompatible normalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train.video_ddpm import VideoWindows
from train.video_vae import DSFCausalVideoVAE


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ChannelMoments:
    def __init__(self, channels: int):
        self.count = 0
        self.sum = torch.zeros(channels, dtype=torch.float64)
        self.square_sum = torch.zeros(channels, dtype=torch.float64)
        self.minimum = torch.full((channels,), float("inf"), dtype=torch.float64)
        self.maximum = torch.full((channels,), -float("inf"), dtype=torch.float64)

    def update(self, latent: torch.Tensor) -> None:
        if latent.ndim != 5 or latent.shape[1] != self.sum.numel():
            raise ValueError("expected [B,C,T,H,W] latent tensor")
        flat = latent.detach().double().permute(1, 0, 2, 3, 4).flatten(1).cpu()
        self.count += flat.shape[1]
        self.sum += flat.sum(1)
        self.square_sum += flat.square().sum(1)
        self.minimum = torch.minimum(self.minimum, flat.min(1).values)
        self.maximum = torch.maximum(self.maximum, flat.max(1).values)

    def result(self) -> dict[str, object]:
        if not self.count:
            raise ValueError("no latent values observed")
        mean = self.sum / self.count
        variance = (self.square_sum / self.count - mean.square()).clamp_min(1e-12)
        return {
            "values_per_channel": self.count,
            "mean": mean.tolist(),
            "std": variance.sqrt().tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
        }


@torch.no_grad()
def compute(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    model = DSFCausalVideoVAE(
        temporal_compression=int(saved["temporal_compression"]),
        spatial_compression=int(saved.get("spatial_compression", 4)),
        latent_channels=int(saved["latent_channels"]),
        base_channels=int(saved["base_channels"]),
        blocks_per_stage=int(saved["blocks_per_stage"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    dataset = VideoWindows(
        args.cache, frames=args.frames, split="train", size=int(saved["size"]),
        deterministic=True, repeats=args.repeats, return_text=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    moments = ChannelMoments(int(saved["latent_channels"]))
    windows = 0
    for video, _ in loader:
        if windows >= args.windows:
            break
        video = ((video.to(device) + 1) / 2).clamp(0, 1)
        if windows + video.shape[0] > args.windows:
            video = video[:args.windows - windows]
        mean, _ = model.encode(video)
        moments.update(mean)
        windows += video.shape[0]
    return {
        "version": 1,
        "codec_checkpoint": str(checkpoint_path),
        "codec_checkpoint_sha256": file_sha256(checkpoint_path),
        "codec_model_sha256": checkpoint.get("model_sha256"),
        "codec_step": int(checkpoint["step"]),
        "variant": f"f{saved.get('spatial_compression', 4)}t{saved['temporal_compression']}d{saved['latent_channels']}",
        "split": "train",
        "deterministic": True,
        "windows": windows,
        "input_frames": args.frames,
        "latent_shape_per_window": [
            int(saved["latent_channels"]),
            (args.frames + int(saved["temporal_compression"]) - 1) // int(saved["temporal_compression"]),
            int(saved["size"]) // int(saved.get("spatial_compression", 4)),
            int(saved["size"]) // int(saved.get("spatial_compression", 4)),
        ],
        **moments.result(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--windows", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = compute(args)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
