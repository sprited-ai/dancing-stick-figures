"""Track B — reference recipe: pixel-space DiT + flow matching on stickdance-128. Pure PyTorch.

    python -m train.video_dit_fm --cache data/v1_cache --out runs/b0 --preset 4090-full

Model: factorised video DiT (Seedance/CogVideoX-style decoupled layers): patchify each frame (p×p, no
temporal patching) → alternate SPATIAL blocks (attention within a frame) and TEMPORAL blocks (attention
across frames at the same token position). adaLN-Zero conditioning on (t, optional class), learned
spatial + temporal positional embeddings, QK-norm. ~30M params at dim 384 / depth 12.
Diffusion: rectified flow. x_t = (1-t)·x0 + t·ε, target v = ε − x0, t ~ logit-normal(0,1), optional
timestep shift for larger res/frames. Sampler: Euler, 50 steps (also try 10). EMA, bf16, fixed-seed
sample grid, raw+EMA samples early, val loss, TB. Shares the data pipeline / presets with video_ddpm.
"""
from __future__ import annotations
import argparse, copy, json, math, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from train.video_ddpm import VideoWindows, to_gif, worker_init, PRESETS


# ----------------------------------------------------------------- model
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads; self.qkv = nn.Linear(dim, 3 * dim); self.o = nn.Linear(dim, dim)
        self.qn = nn.LayerNorm(dim // heads); self.kn = nn.LayerNorm(dim // heads)

    def forward(self, x):                      # x [N, L, D]
        N, L, D = x.shape
        q, k, v = self.qkv(x).reshape(N, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = self.qn(q), self.kn(k)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(N, L, D)
        return self.o(o)


class Block(nn.Module):
    """DiT block with adaLN-Zero. axis='spatial' attends within frame, 'temporal' across frames."""
    def __init__(self, dim, heads, axis, mlp=4):
        super().__init__()
        self.axis = axis
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6); self.attn = Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp * dim), nn.GELU(approximate="tanh"), nn.Linear(mlp * dim, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    t1_skip = True                             # ib64 (trained before the skip existed) carries args["t1_skip"]=False; loaders honour it

    def forward(self, x, c):                   # x [B, T, N, D], c [B, D]
        B, T, N, D = x.shape
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, -1)
        if self.axis == "spatial":
            h = x.reshape(B * T, N, D); rep = lambda z: z.repeat_interleave(T, 0)
        else:
            h = x.permute(0, 2, 1, 3).reshape(B * N, T, D); rep = lambda z: z.repeat_interleave(N, 0)
        if not (self.axis == "temporal" and T == 1 and self.t1_skip):   # image mode: nothing to attend across (gate stays 0 -> identity when video-initialised)
            h = h + rep(g1).unsqueeze(1) * self.attn(modulate(self.n1(h), rep(s1), rep(sc1)))
        h = h + rep(g2).unsqueeze(1) * self.mlp(modulate(self.n2(h), rep(s2), rep(sc2)))
        if self.axis == "spatial":
            return h.reshape(B, T, N, D)
        return h.reshape(B, N, T, D).permute(0, 2, 1, 3)


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * 1000.0 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


class VideoDiT(nn.Module):
    def __init__(self, size=128, frames=16, patch=4, in_ch=4, dim=384, depth=12, heads=6, n_classes=0):
        super().__init__()
        self.p, self.T, self.S, self.C, self.dim = patch, frames, size, in_ch, dim
        self.N = (size // patch) ** 2
        self.embed = nn.Linear(in_ch * patch * patch, dim)
        self.pos_s = nn.Parameter(torch.zeros(1, 1, self.N, dim)); self.pos_t = nn.Parameter(torch.zeros(1, frames, 1, dim))
        nn.init.normal_(self.pos_s, std=0.02); nn.init.normal_(self.pos_t, std=0.02)
        self.temb = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.cls = nn.Embedding(n_classes + 1, dim) if n_classes else None
        self.blocks = nn.ModuleList([Block(dim, heads, "spatial" if i % 2 == 0 else "temporal") for i in range(depth)])
        self.nf = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, in_ch * patch * patch)
        nn.init.zeros_(self.ada_f[-1].weight); nn.init.zeros_(self.ada_f[-1].bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.grad_ckpt = False

    def patchify(self, x):                     # [B,C,T,H,W] -> [B,T,N,C*p*p]
        B, C, T, H, W = x.shape; p = self.p
        x = x.reshape(B, C, T, H // p, p, W // p, p).permute(0, 2, 3, 5, 1, 4, 6)
        return x.reshape(B, T, (H // p) * (W // p), C * p * p)

    def unpatchify(self, x):                   # [B,T,N,C*p*p] -> [B,C,T,H,W]
        B, T, N, _ = x.shape; p = self.p; h = w = int(math.sqrt(N))
        x = x.reshape(B, T, h, w, self.C, p, p).permute(0, 4, 1, 2, 5, 3, 6)
        return x.reshape(B, self.C, T, h * p, w * p)

    def forward(self, x, t, y=None):
        h = self.embed(self.patchify(x)) + self.pos_s + self.pos_t[:, : x.shape[2]]
        c = self.temb(timestep_embedding(t, 256))
        if self.cls is not None: c = c + self.cls(y)
        for blk in self.blocks:
            h = checkpoint(blk, h, c, use_reentrant=False) if (self.grad_ckpt and self.training) else blk(h, c)
        s, sc = self.ada_f(c).chunk(2, -1)
        h = self.nf(h) * (1 + sc[:, None, None]) + s[:, None, None]
        return self.unpatchify(self.out(h))


# ----------------------------------------------------------------- flow matching
def sample_t(B, device, shift=1.0):
    """logit-normal(0,1) then optional shift toward noise: t' = shift*t / (1 + (shift-1)*t)."""
    t = torch.sigmoid(torch.randn(B, device=device))
    return shift * t / (1 + (shift - 1) * t)


@torch.no_grad()
def mixed_noise(shape, device, corr, generator=None):
    e = torch.randn(shape, device=device, generator=generator)
    if corr > 0 and shape[2] > 1:
        s = torch.randn(shape[:2] + (1,) + shape[3:], device=device, generator=generator).expand(shape)
        e = math.sqrt(1 - corr) * s + math.sqrt(corr) * e
    return e


@torch.no_grad()
def euler_sample(model, shape, device, steps=50, y=None, cfg=0.0, null_y=None, noise=None, shift=1.0):
    x = torch.randn(shape, device=device) if noise is None else noise.clone()
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    ts = shift * ts / (1 + (shift - 1) * ts)
    for i in range(steps):
        t, tn = ts[i], ts[i + 1]
        tt = torch.full((shape[0],), float(t), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(x, tt, y)
            if cfg > 0 and y is not None:
                vu = model(x, tt, null_y); v = vu + cfg * (v - vu)
        x = x + (tn - t) * v.float()          # dx/dt = v = ε − x0 ; integrate t: 1 → 0
    return x.clamp(-1, 1)


# ----------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=16); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12); ap.add_argument("--heads", type=int, default=6); ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--cond", default="none", choices=["none", "group"]); ap.add_argument("--cfg_drop", type=float, default=0.1)
    ap.add_argument("--sample_every", type=int, default=2000); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", default=""); ap.add_argument("--preset", default="", choices=[""] + list(PRESETS))
    ap.add_argument("--lr_final", type=float, default=0.1, help="cosine decays to this fraction of --lr")
    ap.add_argument("--init", default="", help="warm-start weights (model+ema) from an image/other checkpoint; step/opt fresh (Seedance stage 2)")
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--grad_ckpt", action="store_true"); ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--fast", action="store_true", help="cudnn.benchmark, tf32-high, channels_last_3d (UNet), fused AdamW, foreach EMA")
    ap.add_argument("--compile", action="store_true", help="torch.compile per block (regional); test on sm_120 first")
    ap.add_argument("--shift", type=float, default=1.0, help="timestep shift (>1 = more noise; try 3 at 128/16f)")
    ap.add_argument("--img_frac", type=float, default=0.0, help="fraction of batches that are single frames (T=1 image warm-up mix)")
    ap.add_argument("--noise_corr", type=float, default=0.0, help="PYoCo mixed noise: eps = sqrt(1-b)*shared + sqrt(b)*per-frame; 0 = iid")
    a = ap.parse_args()
    if a.preset:
        for k, v in PRESETS[a.preset].items():
            if k != "ch": setattr(a, k, v)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    if a.fast:
        torch.backends.cudnn.benchmark = True; torch.set_float32_matmul_precision("high")
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    dev = "cuda"
    ds = VideoWindows(a.cache, a.frames, "train", a.stride, size=a.size)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True,
                                     persistent_workers=True, worker_init_fn=worker_init)
    vds = VideoWindows(a.cache, a.frames, "val", a.stride, size=a.size)
    vdl = torch.utils.data.DataLoader(vds, batch_size=a.batch, shuffle=True, num_workers=2, drop_last=True, worker_init_fn=worker_init) if len(vds) else None
    n_cls = len(ds.groups) if a.cond == "group" else 0
    a.t1_skip = True                                          # recorded in ckpt args; see Block.t1_skip
    model = VideoDiT(size=a.size, frames=a.frames, patch=a.patch, dim=a.dim, depth=a.depth, heads=a.heads, n_classes=n_cls).to(dev)
    model.grad_ckpt = a.grad_ckpt
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if a.compile:   # compile forward only -> module identity / state_dict keys unchanged
        torch._dynamo.config.cache_size_limit = 64          # train batch / sample chunk / val batch shapes
        for b in model.blocks: b.forward = torch.compile(b.forward, dynamic=False)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"DiT-FM params {n_params:.1f}M · {len(ds.clips)} train / {len(vds.clips)} val clips · {a.size}px p{a.patch} · frames {a.frames} · batch {a.batch}×{a.accum} · ckpt {a.grad_ckpt} · shift {a.shift}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01, fused=a.fast)
    step = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev); model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    elif a.init:
        ck = torch.load(a.init, map_location=dev)
        for tgt, key in ((model, "model"), (ema, "ema")):
            src = ck.get(key) or ck["ema"]; own = tgt.state_dict()
            src = {k: v for k, v in src.items() if k in own and (v.shape == own[k].shape or k.endswith("pos_t"))}
            if "pos_t" in src and src["pos_t"].shape != own["pos_t"].shape:      # DiT: image pos_t [1,1,1,D] -> tile over frames
                src["pos_t"] = src["pos_t"].expand_as(own["pos_t"]).clone()
            miss = tgt.load_state_dict(src, strict=False)
            print(f"init {key} from {a.init} step {ck.get('step')}: {len(src)} tensors, missing {len(miss.missing_keys)}, unexpected {len(miss.unexpected_keys)}", flush=True)
    log = open(os.path.join(a.out, "log.txt"), "a")
    try:
        from torch.utils.tensorboard import SummaryWriter; tb = SummaryWriter(os.path.join(a.out, "tb"))
    except Exception: tb = None

    def vloss():
        if vdl is None: return None
        ema.eval(); tot = 0; n = 0
        with torch.no_grad():
            for i, (x, _) in enumerate(vdl):
                if i >= 16: break
                x = x.to(dev); t = sample_t(x.shape[0], dev, a.shift); eps = torch.randn_like(x)
                tt = t[:, None, None, None, None]
                with torch.autocast("cuda", dtype=torch.bfloat16): pred = ema((1 - tt) * x + tt * eps, t, None)
                tot += F.mse_loss(pred.float(), eps - x).item(); n += 1
        return tot / max(n, 1)

    t0 = time.time(); step0 = step; it = iter(dl); ema_loss = None
    while step < a.steps:
        try: x, y = next(it)
        except StopIteration: it = iter(dl); x, y = next(it)
        x = x.to(dev, non_blocking=True); y = y.to(dev)
        if a.img_frac > 0 and random.random() < a.img_frac:      # image warm-up mix: a random single frame per clip
            fi = random.randrange(x.shape[2]); x = x[:, :, fi:fi + 1]
        if n_cls:
            drop = torch.rand(y.shape[0], device=dev) < a.cfg_drop
            y = torch.where(drop, torch.full_like(y, n_cls), y)
        else:
            y = None
        t = sample_t(x.shape[0], dev, a.shift); tt = t[:, None, None, None, None]
        eps = torch.randn_like(x)
        if a.noise_corr > 0 and x.shape[2] > 1:      # PYoCo mixed noise (valid: still Gaussian, forward process unchanged)
            shared = torch.randn_like(x[:, :, :1]).expand_as(x)
            eps = math.sqrt(1 - a.noise_corr) * shared + math.sqrt(a.noise_corr) * eps
        xt = (1 - tt) * x + tt * eps
        prog = max(0.0, (step - 1000) / max(1, a.steps - 1000))
        lr = a.lr * min(1.0, (step + 1) / 1000) * (a.lr_final + (1 - a.lr_final) * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups: g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(xt, t, y)
        err = (pred.float() - (eps - x)) ** 2
        loss = err.mean() / a.accum
        with torch.no_grad():                                   # diagnostics only
            fgm = (x[:, 3:4] > -0.9).float()
            fg_loss = float((err * fgm).sum() / fgm.sum().clamp_min(1))
            per = err.flatten(1).mean(1)
            lb = float(per[t > 0.8].mean()) if (t > 0.8).any() else float("nan")   # near noise
            hb = float(per[t < 0.2].mean()) if (t < 0.2).any() else float("nan")   # near data
        loss.backward()
        if (step + 1) % a.accum != 0:
            step += 1; continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        loss = loss * a.accum
        with torch.no_grad():
            d = min(0.999 if step < 5000 else 0.9995, (1 + step) / (10 + step))
            if a.fast: torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - d)
            else:
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - d)
        step += 1
        ema_loss = loss.item() if ema_loss is None else 0.98 * ema_loss + 0.02 * loss.item()
        if step == 10 or step % 50 == 0:
            spi = (time.time() - t0) / max(1, step - step0)
            msg = f"step {step} loss {ema_loss:.4f} fg {fg_loss:.4f} t>.8 {lb:.4f} t<.2 {hb:.4f} lr {lr:.2e} {spi:.2f}s/it peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB ETA {(a.steps-step)*spi/3600:.1f}h"
            vl = vloss() if (vdl is not None and step % a.val_every == 0) else None
            if vl is not None: msg += f" val {vl:.4f}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            if tb:
                tb.add_scalar("loss/train_ema", ema_loss, step); tb.add_scalar("lr", lr, step)
                tb.add_scalar("loss/fg", fg_loss, step); tb.add_scalar("loss/t_gt_0.8", lb, step); tb.add_scalar("loss/t_lt_0.2", hb, step)
                tb.add_scalar("perf/s_per_it", spi, step); tb.add_scalar("perf/peak_gb", torch.cuda.max_memory_allocated() / 1e9, step)
                if vl is not None: tb.add_scalar("loss/val", vl, step)
        if step % a.sample_every == 0 or step == a.steps:
            NS, CH = (64, 32) if a.frames == 1 else (16, 8)                          # chunks of 8 for 24 GB cards; 64 samples for image models
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            ys = torch.arange(NS, device=dev) % n_cls if n_cls else None
            g = torch.Generator(device=dev).manual_seed(1234)
            noise = mixed_noise((NS, 4, a.frames, a.size, a.size), dev, a.noise_corr, g)
            for name, m_ in (("", ema), ("raw_", model)):
                if name and step > 10000: continue
                m_.eval()
                outs = []
                for i in range(0, NS, CH):
                    yy = ys[i:i + CH] if ys is not None else None
                    outs.append(euler_sample(m_, noise[i:i + CH].shape, dev, steps=50, y=yy, cfg=2.0 if n_cls else 0.0,
                                             null_y=torch.full((CH,), n_cls, device=dev) if n_cls else None, noise=noise[i:i + CH], shift=a.shift).cpu())
                xs = torch.cat(outs, 0)
                m_.train() if m_ is model else None
                to_gif(xs, os.path.join(a.out, f"sample_{name}{step:06d}.gif"))
                if not name and tb:
                    v = ((xs.clamp(-1, 1) + 1) / 2); rgb = (v[:, :3] + (1 - v[:, 3:4])).clamp(0, 1)
                    try: tb.add_video("samples", rgb.permute(0, 2, 1, 3, 4).cpu(), step, fps=10)
                    except Exception: pass
                torch.cuda.empty_cache()
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, args=vars(a), groups=ds.groups, arch="dit_fm"),
                       os.path.join(a.out, "ckpt.pt"))
            torch.save(dict(ema=ema.state_dict(), step=step, args=vars(a), groups=ds.groups, arch="dit_fm"),
                       os.path.join(a.out, f"ckpt_{step:06d}.pt"))
            print(f"  wrote sample_{step:06d}.gif", flush=True)


if __name__ == "__main__":
    main()
