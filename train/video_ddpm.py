"""Track A — pixel-space video diffusion on stickdance-128. Pure PyTorch, no diffusers.

    python -m train.video_ddpm --cache data/v1_cache --out runs/a0 [--frames 16] [--batch 8] [--steps 100000]

Model: factorised 3D UNet (spatial ResBlocks + spatial self-attn at <=32px + temporal attention at
every level), ~50M params. Diffusion: cosine schedule, v-prediction, 1000 train steps, DDIM sampling.
EMA 0.9995. bf16 autocast. Input: 4ch premultiplied RGBA in [-1,1], [B,4,T,128,128].
Optional group conditioning (--cond group) with classifier-free guidance.
Logs loss; every --sample_every steps writes a GIF grid of samples and a checkpoint.
"""
from __future__ import annotations
import argparse, copy, json, math, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# ----------------------------------------------------------------- data
class VideoWindows(torch.utils.data.Dataset):
    def __init__(self, cache, frames=16, split="train", stride=1, drop_flags=("levitation",)):
        self.frames = np.load(os.path.join(cache, "frames.npy"), mmap_mode="r")
        clips = json.load(open(os.path.join(cache, "clips.json")))
        self.clips = [c for c in clips.values() if c["split"] == split and c["n"] >= frames * stride
                      and not any(f in (c["qa"] or "") for f in drop_flags)]
        self.groups = sorted({c["group"] for c in clips.values()})
        self.T, self.stride = frames, stride
        # epoch = every clip a few times
        self.items = [c for c in self.clips for _ in range(4)]

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        c = self.items[i]
        span = self.T * self.stride
        s = c["start"] + random.randint(0, c["n"] - span)
        x = np.asarray(self.frames[s:s + span:self.stride]).astype(np.float32) / 255.0   # [T,H,W,4]
        a = x[..., 3:4]
        x = np.concatenate([x[..., :3] * a, a], -1)          # premultiply
        x = torch.from_numpy(x).permute(3, 0, 1, 2) * 2 - 1  # [4,T,H,W]
        return x, self.groups.index(c["group"])


# ----------------------------------------------------------------- model
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, temb):
        super().__init__()
        self.n1 = nn.GroupNorm(32, cin); self.c1 = nn.Conv3d(cin, cout, (1, 3, 3), padding=(0, 1, 1))
        self.t = nn.Linear(temb, cout)
        self.n2 = nn.GroupNorm(32, cout); self.c2 = nn.Conv3d(cout, cout, (3, 3, 3), padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(F.silu(temb))[:, :, None, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return self.skip(x) + h


class Attn(nn.Module):
    """self-attention over spatial (per frame) or temporal (per pixel) axis."""
    def __init__(self, c, axis, heads=4):
        super().__init__()
        self.axis, self.h = axis, heads
        self.n = nn.GroupNorm(32, c); self.qkv = nn.Linear(c, 3 * c); self.o = nn.Linear(c, c)
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)

    def forward(self, x):
        B, C, T, H, W = x.shape
        h = self.n(x)
        if self.axis == "spatial":
            h = h.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
        else:
            h = h.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, C)
        q, k, v = self.qkv(h).chunk(3, -1)
        sp = lambda z: z.reshape(z.shape[0], z.shape[1], self.h, C // self.h).transpose(1, 2)
        outs = []
        for s0 in range(0, h.shape[0], 32768):   # chunk: kernel grid limit on the batch dim
            sl = slice(s0, s0 + 32768)
            outs.append(F.scaled_dot_product_attention(sp(q[sl]), sp(k[sl]), sp(v[sl])).transpose(1, 2).reshape(-1, h.shape[1], C))
        o = self.o(torch.cat(outs, 0))
        if self.axis == "spatial":
            o = o.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        else:
            o = o.reshape(B, H, W, T, C).permute(0, 4, 3, 1, 2)
        return x + o


class UNet3D(nn.Module):
    def __init__(self, ch=64, mult=(1, 2, 4, 4), attn_res=(32, 16), tattn_res=(64, 32, 16), n_res=2, in_ch=4, n_classes=0, size=128):
        super().__init__()
        temb = ch * 4
        self.temb = nn.Sequential(nn.Linear(ch, temb), nn.SiLU(), nn.Linear(temb, temb))
        self.ch = ch
        self.cls = nn.Embedding(n_classes + 1, temb) if n_classes else None   # last = null class
        self.inp = nn.Conv3d(in_ch, ch, 3, padding=1)
        self.down = nn.ModuleList(); chans = [ch]; c = ch; res = size
        for i, m in enumerate(mult):
            for _ in range(n_res):
                blocks = nn.ModuleList([ResBlock(c, ch * m, temb)]); c = ch * m
                if res in attn_res: blocks.append(Attn(c, "spatial"))
                if res in tattn_res: blocks.append(Attn(c, "temporal"))
                self.down.append(blocks); chans.append(c)
            if i < len(mult) - 1:
                self.down.append(nn.ModuleList([nn.Conv3d(c, c, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))])); chans.append(c); res //= 2
        self.mid = nn.ModuleList([ResBlock(c, c, temb), Attn(c, "spatial"), Attn(c, "temporal"), ResBlock(c, c, temb)])
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mult))):
            for _ in range(n_res + 1):
                blocks = nn.ModuleList([ResBlock(c + chans.pop(), ch * m, temb)]); c = ch * m
                if res in attn_res: blocks.append(Attn(c, "spatial"))
                if res in tattn_res: blocks.append(Attn(c, "temporal"))
                self.up.append(blocks)
            if i > 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"), nn.Conv3d(c, c, (1, 3, 3), padding=(0, 1, 1))])); res *= 2
        self.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv3d(c, in_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, y=None):
        temb = self.temb(timestep_embedding(t, self.ch))
        if self.cls is not None: temb = temb + self.cls(y)
        h = self.inp(x); hs = [h]
        for blocks in self.down:
            if isinstance(blocks[0], ResBlock):
                h = blocks[0](h, temb)
                for b in blocks[1:]: h = b(h)
            else:
                h = blocks[0](h)
            hs.append(h)
        for b in self.mid: h = b(h, temb) if isinstance(b, ResBlock) else b(h)
        for blocks in self.up:
            if isinstance(blocks[0], ResBlock):
                h = blocks[0](torch.cat([h, hs.pop()], 1), temb)
                for b in blocks[1:]: h = b(h)
            else:
                for b in blocks: h = b(h)
        return self.out(h)


# ----------------------------------------------------------------- diffusion (v-pred, cosine)
def alphas_cumprod(T=1000, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ac = (f / f[0]).clamp(1e-5, 1.0)
    return ac[1:].float()


@torch.no_grad()
def sample(model, shape, ac, device, steps=50, y=None, cfg=0.0, null_y=None):
    x = torch.randn(shape, device=device)
    ts = torch.linspace(len(ac) - 1, 0, steps + 1).long().to(device)
    for i in range(steps):
        t, tn = ts[i], ts[i + 1]
        a, an = ac[t], (ac[tn] if i < steps - 1 else torch.tensor(1.0, device=device))
        tt = torch.full((shape[0],), int(t), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(x, tt, y)
            if cfg > 0 and y is not None:
                vu = model(x, tt, null_y); v = vu + cfg * (v - vu)
        v = v.float()
        x0 = a.sqrt() * x - (1 - a).sqrt() * v
        eps = (1 - a).sqrt() * x + a.sqrt() * v
        x0 = x0.clamp(-1, 1)
        x = an.sqrt() * x0 + (1 - an).sqrt() * eps
    return x


def to_gif(x, path, fps=10):
    """x [B,4,T,H,W] in [-1,1] premultiplied -> composite over white, grid, GIF."""
    import imageio
    x = ((x.clamp(-1, 1) + 1) / 2).cpu().numpy()
    B, C, T, H, W = x.shape
    cols = int(math.ceil(math.sqrt(B))); rows = int(math.ceil(B / cols))
    frames = []
    for t in range(T):
        canvas = np.ones((rows * H, cols * W, 3), np.float32)
        for b in range(B):
            rgb, a = x[b, :3, t].transpose(1, 2, 0), x[b, 3:4, t].transpose(1, 2, 0)
            img = rgb + (1 - a)                       # premultiplied over white
            r, c = divmod(b, cols); canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = img
        frames.append((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)


# ----------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=16); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--cond", default="none", choices=["none", "group"]); ap.add_argument("--cfg_drop", type=float, default=0.1)
    ap.add_argument("--sample_every", type=int, default=2000); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    ds = VideoWindows(a.cache, a.frames, "train", a.stride)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=True)
    n_cls = len(ds.groups) if a.cond == "group" else 0
    model = UNet3D(ch=a.ch, n_classes=n_cls).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params {n_params:.1f}M · {len(ds.clips)} clips · frames {a.frames} · batch {a.batch} · cond {a.cond}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.01)
    ac = alphas_cumprod().to(dev)
    step = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev); model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    log = open(os.path.join(a.out, "log.txt"), "a")
    t0 = time.time(); it = iter(dl); ema_loss = None
    while step < a.steps:
        try: x, y = next(it)
        except StopIteration: it = iter(dl); x, y = next(it)
        x = x.to(dev, non_blocking=True); y = y.to(dev)
        if n_cls:
            drop = torch.rand(y.shape[0], device=dev) < a.cfg_drop
            y = torch.where(drop, torch.full_like(y, n_cls), y)
        else:
            y = None
        t = torch.randint(0, len(ac), (x.shape[0],), device=dev)
        at = ac[t][:, None, None, None, None]
        eps = torch.randn_like(x)
        xt = at.sqrt() * x + (1 - at).sqrt() * eps
        v = at.sqrt() * eps - (1 - at).sqrt() * x
        # warmup
        for g in opt.param_groups: g["lr"] = a.lr * min(1.0, (step + 1) / 1000)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(xt, t, y)
        loss = F.mse_loss(pred.float(), v)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            d = min(0.9995, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - d)
        step += 1
        ema_loss = loss.item() if ema_loss is None else 0.98 * ema_loss + 0.02 * loss.item()
        if step % 50 == 0:
            msg = f"step {step} loss {ema_loss:.4f} {(time.time()-t0)/step:.2f}s/it {torch.cuda.max_memory_allocated()/1e9:.1f}GB"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if step % a.sample_every == 0 or step == a.steps:
            ys = torch.arange(8, device=dev) % n_cls if n_cls else None
            xs = sample(ema, (8, 4, a.frames, 128, 128), ac, dev, steps=50, y=ys, cfg=2.0 if n_cls else 0.0,
                        null_y=torch.full((8,), n_cls, device=dev) if n_cls else None)
            to_gif(xs, os.path.join(a.out, f"sample_{step:06d}.gif"))
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, args=vars(a), groups=ds.groups),
                       os.path.join(a.out, "ckpt.pt"))
            print(f"  wrote sample_{step:06d}.gif", flush=True)


if __name__ == "__main__":
    main()
