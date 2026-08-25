"""Oracle metrics for IMAGE models (T=1): sample N images from a checkpoint (EMA), score tvr/lie/cpe per
frame, and compare with real validation frames. Works for UNet (video_ddpm) and DiT-FM checkpoints.

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
    text_dim = 0
    if a.get("cond") == "text":
        from transformers import AutoConfig
        text_dim = AutoConfig.from_pretrained(a.get("text_encoder", "google-t5/t5-small")).d_model
    is_dit = ck.get("arch") in {"dit_fm", "dit_fm_t2v"}
    if is_dit:
        m = VideoDiT(size=S, frames=a.get("frames", 1), patch=a.get("patch", 4), dim=a.get("dim", 384), depth=a.get("depth", 12), heads=a.get("heads", 6), n_classes=n_cls, cond_ch=5 if a.get("i2v_frac", 0) > 0 else 0, text_dim=text_dim, local_3d=bool(a.get("local_3d", False)))
    else:
        m = UNet3D(ch=a.get("ch", 64), n_classes=n_cls, size=S,
                  cond_ch=5 if a.get("ar_ctx", 0) > 0 else 0, text_dim=text_dim,
                  temporal_neighbors=int(a.get("temporal_neighbors", 0)),
                  temporal_pos_bias=bool(a.get("temporal_pos_bias", False)))
    m.load_state_dict(ck["ema"])
    if is_dit:
        for b in m.blocks: b.t1_skip = bool(a.get("t1_skip", True))
    return m.to(dev).eval(), ck, a, S


@torch.no_grad()
def sample_images(m, ck, a, S, n, steps, dev, chunk=64, seed=0, prompts=None, cfg=3.0):
    g = torch.Generator(device=dev).manual_seed(seed); outs = []
    ac = alphas_cumprod().to(dev)
    text = text_mask = null_text = null_text_mask = None
    if a.get("cond") == "text":
        if not prompts or len(prompts) != n:
            raise ValueError("text-conditioned scoring requires one prompt per sample")
        from transformers import AutoTokenizer, T5EncoderModel
        name = a.get("text_encoder", "google-t5/t5-small")
        tok = AutoTokenizer.from_pretrained(name)
        enc = T5EncoderModel.from_pretrained(name).to(dev).eval().requires_grad_(False)
        tokens = tok(prompts + [""], padding="max_length", truncation=True,
                     max_length=a.get("text_len", 32), return_tensors="pt")
        hidden = []
        for j in range(0, len(prompts) + 1, chunk):
            hidden.append(enc(input_ids=tokens.input_ids[j:j + chunk].to(dev),
                              attention_mask=tokens.attention_mask[j:j + chunk].to(dev)).last_hidden_state.cpu())
        hidden = torch.cat(hidden).to(dev)
        text, text_mask = hidden[:-1], tokens.attention_mask[:-1].to(dev)
        null_text = hidden[-1:].expand(n, -1, -1)
        null_text_mask = tokens.attention_mask[-1:].to(dev).expand(n, -1)
        del enc
    for i in range(0, n, chunk):
        b = min(chunk, n - i); noise = torch.randn((b, 4, 1, S, S), device=dev, generator=g)
        if ck.get("arch") in {"dit_fm", "dit_fm_t2v"}:
            outs.append(euler_sample(
                m, noise.shape, dev, steps=steps, noise=noise, shift=a.get("shift", 1.0),
                cfg=cfg if text is not None else 0.0,
                text=text[i:i + b] if text is not None else None,
                text_mask=text_mask[i:i + b] if text_mask is not None else None,
                null_text=null_text[i:i + b] if null_text is not None else None,
                null_text_mask=null_text_mask[i:i + b] if null_text_mask is not None else None,
            ))
        else:
            outs.append(sample(m, noise.shape, ac, dev, steps=steps, noise=noise,
                               cfg=cfg if text is not None else 0.0,
                               text=text[i:i + b] if text is not None else None,
                               text_mask=text_mask[i:i + b] if text_mask is not None else None,
                               null_text=null_text[i:i + b] if null_text is not None else None,
                               null_text_mask=null_text_mask[i:i + b] if null_text_mask is not None else None))
    return torch.cat(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=512); ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0, help="guidance for full-prompt UNet checkpoints")
    ap.add_argument("--out", default=""); ap.add_argument("--grid", default="", help="also save a PNG grid of the first 64 samples")
    a = ap.parse_args(); dev = "cuda"
    m, ck, args, S = load(a.ckpt, dev)
    prompt_pool = []
    if args.get("cond") == "text":
        prompt_ds = VideoWindows(a.cache, 1, "val", 1, size=S, return_text=True)
        prompt_pool = sorted({c["text"] for c in prompt_ds.clips})
    prompts = [prompt_pool[(i * len(prompt_pool)) // a.n] for i in range(a.n)] if prompt_pool else None
    xs = sample_images(m, ck, args, S, a.n, a.steps, dev, prompts=prompts, cfg=a.cfg)
    rgba, _ = to_uint8_rgba(xs)                                   # [N,1,H,W,4]
    per = [score_frame(f[0]) for f in rgba]
    res = {"ckpt": a.ckpt, "step": int(ck["step"]), "n": a.n, "steps": a.steps}
    if prompts is not None:
        res.update(conditioning="full_prompt", cfg=a.cfg, unique_prompts=len(set(prompts)))
    for k in KEYS:
        v = [p[k] for p in per]; res[k] = float(np.mean(v)); res[k + "_ci"] = boot_ci(v)
    # Legacy JSON key ``floor`` stores the real-frame reference at the same size.
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
