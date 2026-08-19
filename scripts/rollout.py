"""Generate long dances from an autoregressive checkpoint (chunked rollout) -> GIF (+ optional oracle temporal scores).

    python scripts/rollout.py --ckpt unet_ar64.pt --seconds 5 --n 8 --out out/dance.gif [--steps 50] [--score --cache data/cache]

Downloads the checkpoint from sprited/dancing-stick-figures-baselines if the path is not a local file.
"""
from __future__ import annotations
import argparse, math, os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.video_ddpm import UNet3D, alphas_cumprod, rollout, to_gif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="unet_ar64.pt"); ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--steps", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/dance.gif"); ap.add_argument("--score", action="store_true"); ap.add_argument("--cache", default="")
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    path = a.ckpt
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download; path = hf_hub_download("sprited/dancing-stick-figures-baselines", a.ckpt)
    ck = torch.load(path, map_location=dev, weights_only=False); args = ck["args"]
    K, F, S, stride = args["ar_ctx"], args["frames"], args["size"], args.get("stride", 1)
    assert K > 0, "not an autoregressive checkpoint (ar_ctx == 0)"
    m = UNet3D(ch=args.get("ch", 64), size=S, cond_ch=5).to(dev); m.load_state_dict(ck["ema"]); m.eval()
    fps = 20 / stride; total = int(round(a.seconds * fps)); n_chunks = max(1, math.ceil((total - (K + F)) / F) + 1)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    x = rollout(m, a.n, K, F, n_chunks, alphas_cumprod().to(dev), dev, steps=a.steps, S=S, generator=g)[:, :, :total]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); to_gif(x.cpu(), a.out, fps=int(round(fps)))
    print(f"wrote {a.out}: {a.n} samples x {x.shape[2]} frames = {x.shape[2] / fps:.1f} s @ {fps:.0f} fps ({n_chunks} chunks of {F} new frames, {K} context)")
    if a.score:
        from eval.oracle import score_video
        from eval.run_ckpt import to_uint8_rgba, TEMPORAL, FRAME
        rgba, _ = to_uint8_rgba(x); per = [score_video(v) for v in rgba]
        res = {k: float(np.nanmean([p[k] for p in per])) for k in FRAME + TEMPORAL}
        print(json.dumps(res, indent=1))
        if a.cache:
            from train.video_ddpm import VideoWindows
            ds = VideoWindows(a.cache, x.shape[2], "val", stride, size=S); idx = np.random.RandomState(1).permutation(len(ds))[:a.n]
            real = torch.stack([ds[int(i)][0] for i in idx]); rr, _ = to_uint8_rgba(real); perr = [score_video(v) for v in rr]
            print("floor (real val clips, same length):", json.dumps({k: float(np.nanmean([p[k] for p in perr])) for k in FRAME + TEMPORAL}, indent=1))


if __name__ == "__main__":
    main()
