"""mini-Wan pixel baseline: diffusers WanTransformer3DModel retrofitted to 64x64 RGBA,
no VAE -- the simplest design (pixels -> tokens -> one full-attention transformer).

Verbatim Wan block stack (full 3D attention, 3D RoPE, cross-attention text) minified to
~39M (dim 384 / 16 layers / patch 1x8x8 -> 4,096 tokens on 64f), trained with the same
rectified-flow objective, first-64 view, fg weighting, and win64 evaluation as the
paper-1 ladder so its row is directly comparable to L2/L3.

    PYTHONPATH=. python3 -m train.wan_mini --cache cache/mini_v02 \
        --out results/paper1_v02c_wanmini_pix10k_s0 --steps 10000
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from train.video_ddpm import VideoWindows
from train.video_dit_fm import foreground_weighted_mse, sample_t


def build_model(a):
    return WanTransformer3DModel(
        patch_size=(a.patch_t, a.patch, a.patch), num_attention_heads=a.heads,
        attention_head_dim=a.dim // a.heads, in_channels=a.channels, out_channels=a.channels,
        text_dim=a.text_dim, freq_dim=256, ffn_dim=4 * a.dim, num_layers=a.depth,
        rope_max_seq_len=8192,
    )


@torch.no_grad()
def euler_sample(model, shape, dev, text, mask, null_text, null_mask, steps=50, cfg=3.0, generator=None):
    x = torch.randn(shape, device=dev, generator=generator)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    for i in range(steps):
        t = ts[i].expand(shape[0])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(hidden_states=x, timestep=t * 1000, encoder_hidden_states=text, return_dict=False)[0]
            if cfg > 0:
                vu = model(hidden_states=x, timestep=t * 1000, encoder_hidden_states=null_text, return_dict=False)[0]
                v = vu + cfg * (v - vu)
        x = x + (ts[i + 1] - ts[i]) * v.float()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--first_frames", type=int, default=64)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--patch_t", type=int, default=4, help="temporal patch: one token spans this many frames (VAE-style 4-frame block, learned)")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--text_encoder", default="google-t5/t5-small")
    ap.add_argument("--text_dim", type=int, default=512)
    ap.add_argument("--text_len", type=int, default=32)
    ap.add_argument("--cfg_drop", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_final", type=float, default=0.1)
    ap.add_argument("--fg_weight", type=float, default=2.0)
    ap.add_argument("--img_frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--channels", type=int, default=4,
                    help="model channels: 4 for pixel RGBA, latent_channels for a latent cache")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--video_every", type=int, default=1000,
                    help="log EMA inference samples to TensorBoard every N steps (0 disables)")
    ap.add_argument("--resume", default="", help="ckpt.pt with model+ema+step to continue from")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    ds = VideoWindows(a.cache, a.frames, "train", 1, size=a.size, return_text=True,
                      first_frames=a.first_frames)
    vds = VideoWindows(a.cache, a.frames, "val", 1, size=a.size, return_text=True,
                       deterministic=True, repeats=1, first_frames=a.first_frames)
    latent_mode = bool(getattr(ds, "latent", False))
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                                     pin_memory=True, drop_last=True, persistent_workers=True)
    vdl = torch.utils.data.DataLoader(vds, batch_size=a.batch, shuffle=False, num_workers=2)

    from transformers import AutoTokenizer, T5EncoderModel
    tok = AutoTokenizer.from_pretrained(a.text_encoder)
    enc = T5EncoderModel.from_pretrained(a.text_encoder).to(dev).eval().requires_grad_(False)

    def text_batch(labels):
        z = tok(list(labels), padding="max_length", truncation=True, max_length=a.text_len,
                return_tensors="pt").to(dev)
        with torch.no_grad():
            return enc(**z).last_hidden_state.float()

    model = build_model(a).to(dev)
    model.enable_gradient_checkpointing()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    start_step = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        start_step = int(ck["step"])
        print(f"resumed from {a.resume} at step {start_step}", flush=True)
    if a.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, fused=True)
    os.makedirs(a.out, exist_ok=True)
    arch = "wan_mini_lat" if latent_mode else "wan_mini_pix"
    json.dump(vars(a) | {"arch": arch, "params_m": n_params},
              open(os.path.join(a.out, "args.json"), "w"), indent=1)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(a.out, "tb"))
    log = open(os.path.join(a.out, "launcher.log"), "a")
    print(f"wan-mini params {n_params:.1f}M · {len(ds.clips)} train clips · patch {a.patch_t}x{a.patch}x{a.patch} · "
          f"tokens {(a.frames // a.patch_t) * (a.size // a.patch) ** 2}", flush=True)

    null_emb_cache = {}

    codec = None
    if a.video_every > 0:
        video_prompts = [vds[i][1] for i in range(0, min(len(vds), 60), 20)]
        if latent_mode:
            from scripts.encode_latent_cache import load_codec
            meta = json.load(open(os.path.join(a.cache, "meta.json")))
            codec, _, _ = load_codec(meta["codec_ckpt"], dev)
            lat_mean = torch.tensor(meta["mean"], device=dev).view(1, -1, 1, 1, 1)
            lat_std = torch.tensor(meta["std"], device=dev).view(1, -1, 1, 1, 1)

    @torch.no_grad()
    def tb_video(step):
        # sample a few val prompts with the EMA model, decode, composite on gray
        txt = text_batch(video_prompts)
        nul = text_batch([""] * len(video_prompts))
        g = torch.Generator(device=dev).manual_seed(7)
        z = euler_sample(ema, (len(video_prompts), a.channels, a.frames, a.size, a.size),
                         dev, txt, None, nul, None, steps=50, cfg=3.0, generator=g)
        if codec is not None:
            tc, sc = int(meta["temporal_compression"]), int(meta["spatial_compression"])
            x = codec.decode(z * lat_std + lat_mean, output_frames=a.frames * tc,
                             output_size=(a.size * sc, a.size * sc)).clamp(0, 1)
        else:
            x = ((z + 1) / 2).clamp(0, 1)
        rgb = x[:, :3] + (1 - x[:, 3:4]) * 0.5                      # gray composite
        tb.add_video("inference/val_prompts", rgb.permute(0, 2, 1, 3, 4).cpu(), step, fps=20)

    def vloss():
        ema.eval(); tot = 0; n = 0
        with torch.no_grad():
            for i, (x, labels) in enumerate(vdl):
                if i >= 8: break
                x = x.to(dev)
                if latent_mode: x = x[:, :-1]
                t = sample_t(x.shape[0], dev)
                eps = torch.randn_like(x)
                tt = t[:, None, None, None, None]
                txt = text_batch(labels)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = ema(hidden_states=(1 - tt) * x + tt * eps, timestep=t * 1000,
                               encoder_hidden_states=txt, return_dict=False)[0]
                tot += F.mse_loss(pred.float(), eps - x).item(); n += 1
        return tot / max(n, 1)

    step, t0 = start_step, time.time()
    it = iter(dl)
    while step < a.steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            try: x, labels = next(it)
            except StopIteration: it = iter(dl); x, labels = next(it)
            x = x.to(dev, non_blocking=True)
            if a.img_frac > 0 and np.random.random() < a.img_frac:
                span = max(1, a.patch_t)
                fi = np.random.randint(x.shape[2] - span + 1); x = x[:, :, fi:fi + span]
            fg_map = None
            if latent_mode:
                fg_map, x = x[:, -1:], x[:, :-1]
            txt = text_batch(labels)
            drop = torch.rand(x.shape[0], device=dev) < a.cfg_drop
            if drop.any():
                null = text_batch([""] * int(drop.sum()))
                txt[drop] = null
            t = sample_t(x.shape[0], dev)
            eps = torch.randn_like(x)
            tt = t[:, None, None, None, None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(hidden_states=(1 - tt) * x + tt * eps, timestep=t * 1000,
                             encoder_hidden_states=txt, return_dict=False)[0]
            err = (pred.float() - (eps - x)) ** 2
            if fg_map is not None:
                w = 1 + (a.fg_weight - 1) * fg_map.clamp(0, 1)
                loss = ((err * w).mean() / w.mean().clamp_min(1e-8)) / a.accum
            else:
                loss = foreground_weighted_mse(err, x, a.fg_weight) / a.accum
            loss.backward()
        step += 1
        lr = a.lr * min(1.0, step / 1000) * (a.lr_final + (1 - a.lr_final) * 0.5 *
              (1 + math.cos(math.pi * max(0.0, (step - 1000) / max(1, a.steps - 1000)))))
        for gp in opt.param_groups: gp["lr"] = lr
        opt.step()
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - d)
        if step % 50 == 0:
            spi = (time.time() - t0) / 50; t0 = time.time()
            vl = vloss() if step % a.val_every == 0 else None
            msg = (f"step {step} loss {loss.item() * a.accum:.4f} lr {lr:.2e} {spi:.2f}s/it "
                   f"peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB "
                   f"ETA {(a.steps - step) * spi / 3600:.1f}h" + (f" val {vl:.4f}" if vl is not None else ""))
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            tb.add_scalar("loss/train", loss.item() * a.accum, step)
            if vl is not None: tb.add_scalar("loss/val", vl, step)
        if a.video_every > 0 and step % a.video_every == 0:
            tb_video(step)
        if step % 5000 == 0 or step == a.steps:
            torch.save(dict(ema=ema.state_dict(), step=step, args=vars(a), arch=arch),
                       os.path.join(a.out, f"ckpt_{step:06d}.pt"))
            torch.save(dict(model=getattr(model, '_orig_mod', model).state_dict(), ema=ema.state_dict(), step=step,
                            args=vars(a), arch=arch), os.path.join(a.out, "ckpt.pt"))
    print("training done", flush=True)


if __name__ == "__main__":
    main()
