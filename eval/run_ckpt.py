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
from train.video_dit_fm import VideoDiT, euler_sample, mixed_noise
from eval.oracle import score_video
from eval.fvd import fvd, rgba_premult_to_rgb


def to_uint8_rgba(x):
    """[B,4,T,H,W] in [-1,1] premultiplied -> uint8 straight-ish RGBA [B,T,H,W,4] for the oracle
    (oracle uses alpha>127 as fg and rgb distance; un-premultiply where alpha>0)."""
    v = ((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 4, 1).cpu().numpy()   # [B,T,H,W,4]
    a = v[..., 3:4]
    rgb = np.where(a > 0.05, v[..., :3] / np.maximum(a, 0.05), 0.0)
    return (np.clip(np.concatenate([rgb, a], -1), 0, 1) * 255).astype(np.uint8), v


TEMPORAL = ("mass_drift", "head_jitter", "angle_jerk", "height_var")
FRAME = ("tvr", "lie", "cpe")


def boot_ci(vals, B=500, seed=0):
    r = np.random.RandomState(seed); v = np.asarray(vals, np.float64)
    if len(v) < 2: return (float(v.mean()), float(v.mean()))
    m = [r.choice(v, len(v), replace=True).mean() for _ in range(B)]
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


_floor_cache = {}


def real_floor(cache, T, S, n, dev):
    """oracle floor on real val windows + FVD real-vs-real (two disjoint halves)."""
    key = (cache, T, S, n)
    if key in _floor_cache: return _floor_cache[key]
    ds = VideoWindows(cache, T, "val", 1, size=S)
    idx = np.random.RandomState(1).permutation(len(ds))
    a_ = torch.stack([ds[int(i)][0] for i in idx[:n]]); b_ = torch.stack([ds[int(i)][0] for i in idx[n:2 * n]]) if len(ds) >= 2 * n else None
    rgba, _ = to_uint8_rgba(a_)
    per = [score_video(v) for v in rgba]
    fl = {k: float(np.mean([p[k] for p in per])) for k in FRAME + TEMPORAL}
    ra = rgba_premult_to_rgb(((a_.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 4, 1).numpy())
    if b_ is not None:
        rb = rgba_premult_to_rgb(((b_.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 4, 1).numpy())
        fl["fvd_real_real"] = fvd(ra, rb, device=dev)
    fl["_real_rgb"] = ra
    _floor_cache[key] = fl
    return fl


def evaluate(run, cache, n=64, dev="cuda", real_cache=None, seeds=(0, 1, 2)):
    ck = torch.load(os.path.join(run, "ckpt.pt"), map_location=dev)
    a = ck["args"]; step = ck["step"]
    n_cls = len(ck.get("groups", [])) if a.get("cond") == "group" else 0
    T, S = a.get("frames", 16), a.get("size", 128)
    is_dit = ck.get("arch") == "dit_fm"
    if is_dit:
        model = VideoDiT(size=S, frames=T, patch=a.get("patch", 4), dim=a.get("dim", 384), depth=a.get("depth", 12), heads=a.get("heads", 6), n_classes=n_cls).to(dev)
    else:
        model = UNet3D(ch=a.get("ch", 64), n_classes=n_cls, size=S).to(dev)
    model.load_state_dict(ck["ema"]); model.eval()
    if is_dit:
        for b in model.blocks: b.t1_skip = bool(a.get("t1_skip", True))
    ac = alphas_cumprod().to(dev)
    fl = real_floor(cache, T, S, n, dev)
    m = {"step": int(step), "n": int(n), "seeds": list(seeds), "floor": {k: v for k, v in fl.items() if not k.startswith("_")}}
    per_seed = []
    for sd in seeds:
        g = torch.Generator(device=dev).manual_seed(1000 + sd)
        outs = []
        for i in range(0, n, 8):
            ys = (torch.arange(8, device=dev) % n_cls) if n_cls else None
            with torch.no_grad():
                if is_dit:
                    noise = mixed_noise((8, 4, T, S, S), dev, a.get("noise_corr", 0.0), g)
                    xs = euler_sample(model, noise.shape, dev, steps=50, y=ys, cfg=2.0 if n_cls else 0.0,
                                      null_y=torch.full((8,), n_cls, device=dev) if n_cls else None, noise=noise, shift=a.get("shift", 1.0))
                else:
                    noise = torch.randn((8, 4, T, S, S), device=dev, generator=g)
                    xs = sample(model, noise.shape, ac, dev, steps=50, y=ys, cfg=2.0 if n_cls else 0.0,
                                null_y=torch.full((8,), n_cls, device=dev) if n_cls else None, noise=noise)
            outs.append(xs)
        xs = torch.cat(outs, 0)[:n]
        rgba, prem = to_uint8_rgba(xs)
        per = [score_video(v) for v in rgba]
        r = {k: [p[k] for p in per] for k in FRAME + TEMPORAL + ("fg",)}
        r["fvd"] = fvd(fl["_real_rgb"], rgba_premult_to_rgb(prem), device=dev)
        per_seed.append(r)
    for k in FRAME + TEMPORAL + ("fg",):
        allv = [v for r in per_seed for v in r[k]]
        m[k] = float(np.mean(allv)); m[k + "_ci"] = boot_ci(allv)
    fv = [r["fvd"] for r in per_seed]
    m["fvd"] = float(np.mean(fv)); m["fvd_std"] = float(np.std(fv))
    if "fvd_real_real" in fl: m["dfvd"] = m["fvd"] - fl["fvd_real_real"]
    del model; torch.cuda.empty_cache()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=64); ap.add_argument("--watch", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
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
                snap = os.path.join(a.run, f"ckpt_{int(step):06d}.pt")
                if not os.path.exists(snap):
                    try:
                        ck = torch.load(p, map_location="cpu", weights_only=False)
                        torch.save({k: ck[k] for k in ck if k != "opt"}, snap)
                    except Exception as e:
                        print("snapshot failed:", e, flush=True)
                try:
                    m = evaluate(a.run, a.cache, a.n, seeds=tuple(range(a.seeds)))
                    json.dump(m, open(os.path.join(a.run, "eval", f"{m['step']:06d}.json"), "w"), indent=1)
                    for k in FRAME + TEMPORAL + ("fvd",): tb.add_scalar(f"eval/{k}", m[k], m["step"])
                    if "dfvd" in m: tb.add_scalar("eval/dfvd", m["dfvd"], m["step"])
                    for k in FRAME + TEMPORAL:
                        tb.add_scalar(f"floor/{k}", m["floor"][k], m["step"])
                    tb.flush()
                    print(f"step {m['step']}: " + " ".join(f"{k} {m[k]:.3f}" for k in FRAME + TEMPORAL + ("fvd",)) +
                          (f" dfvd {m['dfvd']:.1f}" if "dfvd" in m else ""), flush=True)
                    last = m["step"]
                except Exception as e:
                    print("eval failed:", e, flush=True); time.sleep(60)
        if not a.watch: break
        time.sleep(120)


if __name__ == "__main__":
    main()
