"""Encode a pixel cache's first-N-frame windows into a frozen-codec latent cache.

    PYTHONPATH=. python3 scripts/encode_latent_cache.py \
        --cache cache/mini_v02 --ckpt results/vae_final_f8t2d32_mirrored_40k/ckpt_040000.pt \
        --out cache/latent_v02_f8t2d32 --frames 64

Output layout mirrors the pixel cache contract consumed by VideoWindows:
frames.npy float16 [n_clips * T_lat, H_lat, W_lat, C+1] (C raw codec channels
plus one foreground-footprint weight channel), clips.json (n = T_lat per clip),
meta.json {latent: true, channels, mean, std, ...}. Latent channels are stored
raw; the loader normalises with the recorded train-split statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from train.video_vae import DSFCausalVideoVAE


def load_codec(ckpt_path: str, device: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    model = DSFCausalVideoVAE(
        temporal_compression=int(saved["temporal_compression"]),
        spatial_compression=int(saved.get("spatial_compression", 4)),
        latent_channels=int(saved["latent_channels"]),
        base_channels=int(saved["base_channels"]),
        blocks_per_stage=int(saved["blocks_per_stage"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, saved, checkpoint


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=64, help="pixel frames per clip to encode (first N)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha_threshold", type=float, default=0.05)
    a = ap.parse_args()

    codec, saved, checkpoint = load_codec(a.ckpt, a.device)
    t_comp = int(saved["temporal_compression"])
    s_comp = int(saved.get("spatial_compression", 4))
    c_lat = int(saved["latent_channels"])
    if a.frames % t_comp:
        raise ValueError(f"--frames must divide the codec temporal compression {t_comp}")

    frames = np.load(os.path.join(a.cache, "frames.npy"), mmap_mode="r")
    clips = json.load(open(os.path.join(a.cache, "clips.json")))
    size = frames.shape[1]
    t_lat, h_lat = a.frames // t_comp, size // s_comp

    os.makedirs(a.out, exist_ok=True)
    order = sorted(clips)
    n_clips = len(order)
    mm = np.lib.format.open_memmap(
        os.path.join(a.out, "frames.npy"), mode="w+", dtype=np.float16,
        shape=(n_clips * t_lat, h_lat, h_lat, c_lat + 1),
    )

    def encode_batch(ids):
        xs, ws = [], []
        for cid in ids:
            c = clips[cid]
            clip = np.asarray(frames[c["start"]:c["start"] + a.frames]).astype(np.float32) / 255.0
            alpha = clip[..., 3:4]
            pre = np.concatenate([clip[..., :3] * alpha, alpha], -1)          # premultiplied [0,1]
            xs.append(torch.from_numpy(pre).permute(3, 0, 1, 2))              # [4,T,H,W]
            fg = (alpha[..., 0] > a.alpha_threshold)
            fg = fg.reshape(t_lat, t_comp, h_lat, s_comp, h_lat, s_comp).max((1, 3, 5))
            ws.append(fg.astype(np.float16))                                  # [T_lat,H_lat,W_lat]
        x = torch.stack(xs).to(a.device)
        mean, _ = codec.encode(x)                                             # [B,C,T_lat,H,W]
        return mean.permute(0, 2, 3, 4, 1).cpu().numpy(), np.stack(ws)

    stats_sum = np.zeros(c_lat, np.float64)
    stats_sq = np.zeros(c_lat, np.float64)
    stats_n = 0
    for start in range(0, n_clips, a.batch):
        ids = order[start:start + a.batch]
        z, w = encode_batch(ids)
        for j, cid in enumerate(ids):
            k = (start + j) * t_lat
            mm[k:k + t_lat, ..., :c_lat] = z[j].astype(np.float16)
            mm[k:k + t_lat, ..., c_lat] = w[j]
            if clips[cid]["split"] == "train":
                flat = z[j].reshape(-1, c_lat).astype(np.float64)
                stats_sum += flat.sum(0); stats_sq += (flat ** 2).sum(0); stats_n += flat.shape[0]
        if start % (a.batch * 32) == 0:
            print(f"  {start + len(ids)}/{n_clips} clips", flush=True)
    mm.flush()

    mean = stats_sum / stats_n
    std = np.sqrt(np.maximum(stats_sq / stats_n - mean ** 2, 1e-12))
    out_clips = {
        cid: dict(start=i * t_lat, n=t_lat, split=clips[cid]["split"], group=clips[cid]["group"],
                  text=clips[cid]["text"], qa=clips[cid].get("qa") or "")
        for i, cid in enumerate(order)
    }
    json.dump(out_clips, open(os.path.join(a.out, "clips.json"), "w"))
    json.dump(dict(
        latent=True, channels=c_lat, mean=mean.tolist(), std=std.tolist(),
        size=h_lat, frames=n_clips * t_lat, clips=n_clips,
        source_cache=a.cache, source_frames_per_clip=a.frames,
        codec_ckpt=a.ckpt, codec_ckpt_sha256=file_sha256(a.ckpt), codec_step=int(checkpoint["step"]),
        temporal_compression=t_comp, spatial_compression=s_comp,
        posterior="mean", weight_channel="last channel = foreground footprint (alpha>threshold max-pooled per latent cell)",
        alpha_threshold=a.alpha_threshold,
    ), open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"{n_clips} clips -> {a.out} (latent [{t_lat},{h_lat},{h_lat},{c_lat}+1])")


if __name__ == "__main__":
    main()
