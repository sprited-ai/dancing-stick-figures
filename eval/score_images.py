"""Oracle metrics for IMAGE models (T=1): sample N images from a checkpoint (EMA), score tvr/lie/cpe per
frame, compare with real val frames (floor). Works for UNet (video_ddpm) and DiT-FM checkpoints.

    python -m eval.score_images --ckpt runs/ia64/ckpt.pt --cache data/v1_cache [--n 512] [--steps 50] [--out out/ia64_scores.json]
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, torch
from train.video_ddpm import UNet3D, VideoWindows, alphas_cumprod, sample, to_gif
from train.video_dit_fm import VideoDiT, euler_sample
from eval.oracle import score_frame
from eval.run_ckpt import to_uint8_rgba, boot_ci

KEYS = ("tvr", "lie", "cpe", "fg")


def load(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False); a = ck["args"]; S = a.get("size", 128)
    n_cls = len(ck.get("groups", [])) if a.get("cond") == "group" else 0
    if ck.get("arch") == "dit_fm":
        m = VideoDiT(size=S, frames=a.get("frames", 1), patch=a.get("patch", 4), dim=a.get("dim", 384), depth=a.get("depth", 12), heads=a.get("heads", 6), n_classes=n_cls)
    else:
        m = UNet3D(ch=a.get("ch", 64), n_classes=n_cls, size=S)
    m.load_state_dict(ck["ema"])
    if ck.get("arch") == "dit_fm":
        for b in m.blocks: b.t1_skip = bool(a.get("t1_skip", True))
    return m.to(dev).eval(), ck, a, S


@torch.no_grad()
def sample_images(m, ck, a, S, n, steps, dev, chunk=64, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed); outs = []
    ac = alphas_cumprod().to(dev)
    for i in range(0, n, chunk):
        b = min(chunk, n - i); noise = torch.randn((b, 4, 1, S, S), device=dev, generator=g)
        if ck.get("arch") == "dit_fm":
            outs.append(euler_sample(m, noise.shape, dev, steps=steps, noise=noise, shift=a.get("shift", 1.0)))
        else:
            outs.append(sample(m, noise.shape, ac, dev, steps=steps, noise=noise))
    return torch.cat(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=512); ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default=""); ap.add_argument("--grid", default="", help="also save a PNG grid of the first 64 samples")
    a = ap.parse_args(); dev = "cuda"
    m, ck, args, S = load(a.ckpt, dev)
    xs = sample_images(m, ck, args, S, a.n, a.steps, dev)
    rgba, _ = to_uint8_rgba(xs)                                   # [N,1,H,W,4]
    per = [score_frame(f[0]) for f in rgba]
    res = {"ckpt": a.ckpt, "step": int(ck["step"]), "n": a.n, "steps": a.steps}
    for k in KEYS:
        v = [p[k] for p in per]; res[k] = float(np.mean(v)); res[k + "_ci"] = boot_ci(v)
    # floor: real val frames at the same size
    ds = VideoWindows(a.cache, 1, "val", 1, size=S); idx = np.random.RandomState(1).permutation(len(ds))[:a.n]
    real = torch.stack([ds[int(i)][0] for i in idx]); rr, _ = to_uint8_rgba(real)
    perr = [score_frame(f[0]) for f in rr]
    res["floor"] = {k: float(np.mean([p[k] for p in perr])) for k in KEYS}
    res["clean_frac"] = float(np.mean([(p["tvr"] == 0 and p["lie"] == 0) for p in per]))       # fraction of "perfect skeleton" samples
    res["floor"]["clean_frac"] = float(np.mean([(p["tvr"] == 0 and p["lie"] == 0) for p in perr]))
    print(json.dumps(res, indent=1))
    if a.out: os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    if a.grid: to_gif(xs[:64], a.grid[:-4] + ".gif")              # to_gif writes PNG when T==1


if __name__ == "__main__":
    main()
