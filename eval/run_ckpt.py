"""Evaluate a checkpoint: sample N videos -> oracle metrics + FVD vs real val windows -> TensorBoard + json.

    python -m eval.run_ckpt --run runs/a0 --cache data/v1_cache [--n 64] [--watch]

--watch: loop; re-evaluate whenever runs/<run>/ckpt.pt changes (its `step` field). Writes
runs/<run>/eval/<step>.json and TB scalars eval/{tvr,lie,cpe,mass_drift,fvd}.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
from train.video_ddpm import UNet3D, VideoWindows, alphas_cumprod, sample
from eval.oracle import score_video
from eval.fvd import fvd, rgba_premult_to_rgb


def to_uint8_rgba(x):
    """[B,4,T,H,W] in [-1,1] premultiplied -> uint8 straight-ish RGBA [B,T,H,W,4] for the oracle
    (oracle uses alpha>127 as fg and rgb distance; un-premultiply where alpha>0)."""
    v = ((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 4, 1).cpu().numpy()   # [B,T,H,W,4]
    a = v[..., 3:4]
    rgb = np.where(a > 0.05, v[..., :3] / np.maximum(a, 0.05), 0.0)
    return (np.clip(np.concatenate([rgb, a], -1), 0, 1) * 255).astype(np.uint8), v


def evaluate(run, cache, n=64, dev="cuda", real_cache=None):
    ck = torch.load(os.path.join(run, "ckpt.pt"), map_location=dev)
    a = ck["args"]; step = ck["step"]
    n_cls = len(ck.get("groups", [])) if a.get("cond") == "group" else 0
    model = UNet3D(ch=a.get("ch", 64), n_classes=n_cls, size=a.get("size", 128)).to(dev)
    model.load_state_dict(ck["ema"]); model.eval()
    ac = alphas_cumprod().to(dev)
    T, S = a.get("frames", 16), a.get("size", 128)
    outs = []
    for i in range(0, n, 8):
        ys = (torch.arange(8, device=dev) % n_cls) if n_cls else None
        with torch.no_grad():
            xs = sample(model, (8, 4, T, S, S), ac, dev, steps=50, y=ys, cfg=2.0 if n_cls else 0.0,
                        null_y=torch.full((8,), n_cls, device=dev) if n_cls else None)
        outs.append(xs)
    xs = torch.cat(outs, 0)[:n]
    rgba, prem = to_uint8_rgba(xs)
    per = [score_video(v) for v in rgba]
    m = {k: float(np.mean([p[k] for p in per])) for k in ("tvr", "lie", "cpe", "fg", "mass_drift")}
    # real val windows for FVD
    ds = VideoWindows(cache, T, "val", 1, size=S) if real_cache is None else real_cache
    idx = np.random.RandomState(0).choice(len(ds), n, replace=len(ds) < n)
    real = torch.stack([ds[int(i)][0] for i in idx])
    real_rgb = rgba_premult_to_rgb(((real.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 4, 1).numpy())
    fake_rgb = rgba_premult_to_rgb(prem)
    m["fvd"] = fvd(real_rgb, fake_rgb, device=dev)
    m["step"] = int(step); m["n"] = int(n)
    del model; torch.cuda.empty_cache()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=64); ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.run, "eval"), exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(a.run, "tb_eval"))
    last = -1
    while True:
        p = os.path.join(a.run, "ckpt.pt")
        if os.path.exists(p):
            try:
                step = torch.load(p, map_location="cpu", weights_only=False)["step"]
            except Exception:
                step = last
            if step != last:
                try:
                    m = evaluate(a.run, a.cache, a.n)
                    json.dump(m, open(os.path.join(a.run, "eval", f"{m['step']:06d}.json"), "w"), indent=1)
                    for k in ("tvr", "lie", "cpe", "mass_drift", "fvd"): tb.add_scalar(f"eval/{k}", m[k], m["step"])
                    tb.flush()
                    print(f"step {m['step']}: " + " ".join(f"{k} {m[k]:.3f}" for k in ("tvr", "lie", "cpe", "mass_drift", "fvd")), flush=True)
                    last = m["step"]
                except Exception as e:
                    print("eval failed:", e, flush=True); time.sleep(60)
        if not a.watch: break
        time.sleep(120)


if __name__ == "__main__":
    main()
