"""Codec-floor row: encode->decode the real reference windows through the frozen
video codec and score the reconstructions with the model-comparison protocol.

Decomposes any latent-model score into (codec reconstruction limit) + (latent
model skill): if this row sits near the real reference, latent-vs-pixel
comparisons are about the models; if not, the codec is the binding constraint.

    PYTHONPATH=. python3 scripts/codec_floor.py \
        --cache cache/mini_v02 --manifest results/v02c_eval/win64_manifest.json \
        --ckpt <codec ckpt> --out results/v02c_eval/codec_floor_64f_n128.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval.baselines import score_set, tensors_to_premult_rgb, tensors_to_rgba
from eval.fvd import fvd
from eval.protocol import load_manifest_windows
from scripts.encode_latent_cache import file_sha256, load_codec


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    manifest = json.load(open(a.manifest))
    codec, saved, checkpoint = load_codec(a.ckpt, a.device)
    ref_a = load_manifest_windows(a.cache, manifest["reference_a"], size=a.size)   # premult [-1,1]
    ref_b = load_manifest_windows(a.cache, manifest["reference_b"], size=a.size)

    recon = torch.empty_like(ref_b)
    for start in range(0, len(ref_b), a.batch):
        x01 = ((ref_b[start:start + a.batch] + 1) / 2).to(a.device)               # premult [0,1]
        mean, _ = codec.encode(x01)
        y = codec.decode(mean, output_frames=x01.shape[2], output_size=(a.size, a.size))
        recon[start:start + a.batch] = (y.clamp(0, 1) * 2 - 1).cpu()
        print(f"  {min(start + a.batch, len(ref_b))}/{len(ref_b)}", flush=True)

    recon_rgba = tensors_to_rgba(recon)
    row = score_set(recon_rgba)
    row["fvd"] = float(fvd(tensors_to_premult_rgb(ref_a), tensors_to_premult_rgb(recon), device=a.device))
    result = {
        "protocol_version": 1,
        "n": len(recon_rgba),
        "frames": manifest["frames"],
        "stride": manifest["stride"],
        "note": "reference_b encoded+decoded by the frozen codec, FVD vs reference_a (same convention as corruptions)",
        "codec_ckpt": a.ckpt,
        "codec_ckpt_sha256": file_sha256(a.ckpt),
        "codec_step": int(checkpoint["step"]),
        "codec_config": {k: saved[k] for k in ("temporal_compression", "spatial_compression", "latent_channels")},
        "baselines": {"codec_recon": row},
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"fvd": row["fvd"], "tvr": row["tvr"]["mean"], "angle_jerk": row["angle_jerk"]["mean"]}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
