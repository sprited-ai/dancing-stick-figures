"""Track A — pixel-space video diffusion on stickdance-128. Pure PyTorch, no diffusers.

    python -m train.video_ddpm --cache data/v1_cache --out runs/a0 [--frames 16] [--batch 8] [--steps 100000]

Model: factorised 3D UNet (spatial ResBlocks + spatial self-attn at <=32px + temporal attention at
every level), ~50M params. Diffusion: cosine schedule, v-prediction, 1000 train steps, DDIM sampling.
EMA 0.9995. bf16 autocast. Input: 4ch premultiplied RGBA in [-1,1], [B,4,T,128,128].
Optional group or full-prompt conditioning (--cond group/text) with classifier-free guidance.
Logs loss; every --sample_every steps writes a GIF grid of samples and a checkpoint.
"""
from __future__ import annotations
import argparse, copy, json, math, os, random, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ----------------------------------------------------------------- data
class VideoWindows(torch.utils.data.Dataset):
    def __init__(self, cache, frames=16, split="train", stride=1, drop_flags=("levitation",), size=128,
                 deterministic=False, repeats=4, return_text=False, first_frames=0):
        # first_frames > 0 restricts every sampled window to lie inside the
        # first `first_frames` frames of each clip (e.g. the action-dense span).
        self.first_frames = first_frames
        meta_path = os.path.join(cache, "meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        # Latent caches store float16 [N,H,W,C+1]: C codec channels plus one
        # foreground-footprint weight channel; meta carries per-channel stats.
        self.latent = bool(meta.get("latent"))
        if self.latent:
            self.latent_mean = np.asarray(meta["mean"], np.float32)
            self.latent_std = np.asarray(meta["std"], np.float32)
        self.frames = np.load(os.path.join(cache, "frames.npy"), mmap_mode="r")
        self.size = size
        clips = json.load(open(os.path.join(cache, "clips.json")))
        self.span = (frames - 1) * stride + 1
        self.clips = [{"clip_id": clip_id, **c} for clip_id, c in clips.items()
                      if c["split"] == split and c["n"] >= self.span
                      and not any(f in (c.get("qa") or "") for f in drop_flags)]
        self.groups = sorted({c["group"] for c in clips.values()})
        self.T, self.stride, self.deterministic, self.repeats = frames, stride, deterministic, repeats
        self.return_text = return_text
        # epoch = every clip a few times
        self.items = [(c, repeat) for c in self.clips for repeat in range(repeats)]

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        c, repeat = self.items[i]
        max_offset = c["n"] - self.span
        if self.first_frames:
            max_offset = max(0, min(max_offset, self.first_frames - self.span))
        if self.deterministic:
            offset = 0 if self.repeats == 1 else round(repeat * max_offset / (self.repeats - 1))
        else:
            offset = random.randint(0, max_offset)
        s = c["start"] + offset
        if self.latent:
            x = np.asarray(self.frames[s:s + self.span:self.stride]).astype(np.float32)       # [T,H,W,C+1]
            x[..., :-1] = (x[..., :-1] - self.latent_mean) / self.latent_std
            x = torch.from_numpy(x).permute(3, 0, 1, 2)                                       # [C+1,T,H,W]
            return x, c["text"] if self.return_text else self.groups.index(c["group"])
        x = np.asarray(self.frames[s:s + self.span:self.stride]).astype(np.float32) / 255.0   # [T,H,W,4]
        if self.size != x.shape[1]:   # area downsample (premultiply first so colour doesn't bleed from bg)
            f = x.shape[1] // self.size
            x = np.concatenate([x[..., :3] * x[..., 3:4], x[..., 3:4]], -1)
            x = x.reshape(x.shape[0], self.size, f, self.size, f, 4).mean((2, 4))
            a = x[..., 3:4]
            x = torch.from_numpy(x).permute(3, 0, 1, 2) * 2 - 1
            return x, c["text"] if self.return_text else self.groups.index(c["group"])
        a = x[..., 3:4]
        x = np.concatenate([x[..., :3] * a, a], -1)          # premultiply
        x = torch.from_numpy(x).permute(3, 0, 1, 2) * 2 - 1  # [4,T,H,W]
        return x, c["text"] if self.return_text else self.groups.index(c["group"])


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
        h = conv3d_t1(self.c2, F.silu(self.n2(h)))
        return self.skip(x) + h


def conv3d_t1(conv, x):
    """3x3x3 conv; when T == 1 (image mode) only the centre time-slice of the kernel touches non-padding,
    so use it alone (identical result, 3x fewer FLOPs). Off-centre slices stay untouched (zero-init) ->
    an image checkpoint later loads into the video model as 'same image every frame'."""
    if x.shape[2] == 1 and conv.kernel_size[0] == 3:
        return F.conv3d(x, conv.weight[:, :, 1:2], conv.bias, padding=(0, 1, 1))
    return conv(x)


class Attn(nn.Module):
    """Factorised attention with optional motion-aware temporal keys/values.

    The original temporal path compares the same pixel coordinate across time.
    ``neighbor_radius`` first gathers a learned spatial neighbourhood at every
    frame, so a query can match a limb that moved by a few pixels.  A relative
    time bias makes the otherwise permutation-equivariant temporal attention
    aware of ordering.  Both additions are inactive for T=1 image training.
    """
    def __init__(self, c, axis, heads=4, neighbor_radius=0, relative_time=False, max_frames=64):
        super().__init__()
        self.axis, self.h = axis, heads
        self.n = nn.GroupNorm(32, c); self.qkv = nn.Linear(c, 3 * c); self.o = nn.Linear(c, c)
        self.neighbor_radius = neighbor_radius if axis == "temporal" else 0
        if self.neighbor_radius:
            kernel = 2 * self.neighbor_radius + 1
            self.neighbor = nn.Conv3d(
                c, c, (1, kernel, kernel), padding=(0, self.neighbor_radius, self.neighbor_radius),
                groups=c, bias=False,
            )
            nn.init.zeros_(self.neighbor.weight)
            self.neighbor.weight.data[:, 0, 0, self.neighbor_radius, self.neighbor_radius] = 1
        else:
            self.neighbor = None
        self.max_frames = max_frames
        self.relative_time = relative_time and axis == "temporal"
        self.time_bias = nn.Parameter(torch.zeros(heads, 2 * max_frames - 1)) if self.relative_time else None
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)

    def _temporal_bias(self, frames, dtype, device):
        if self.time_bias is None:
            return None
        if frames > self.max_frames:
            raise ValueError(f"temporal attention received {frames} frames; max_frames={self.max_frames}")
        positions = torch.arange(frames, device=device)
        offsets = positions[None, :] - positions[:, None] + self.max_frames - 1
        return self.time_bias[:, offsets].to(dtype=dtype).unsqueeze(0)

    def forward(self, x):
        B, C, T, H, W = x.shape
        if self.axis == "temporal" and T == 1: return x     # image mode: nothing to attend across; keep layer at init
        h = self.n(x)
        if self.axis == "spatial":
            h = h.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
            q, k, v = self.qkv(h).chunk(3, -1)
        else:
            kv_source = self.neighbor(h) if self.neighbor is not None else h
            h = h.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, C)
            kv_source = kv_source.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, C)
            wq, wk, wv = self.qkv.weight.chunk(3, 0)
            bq, bk, bv = self.qkv.bias.chunk(3, 0)
            q = F.linear(h, wq, bq)
            k = F.linear(kv_source, wk, bk)
            v = F.linear(kv_source, wv, bv)
        sp = lambda z: z.reshape(z.shape[0], z.shape[1], self.h, C // self.h).transpose(1, 2)
        attention_bias = self._temporal_bias(T, q.dtype, q.device) if self.axis == "temporal" else None
        outs = []
        for s0 in range(0, h.shape[0], 32768):   # chunk: kernel grid limit on the batch dim
            sl = slice(s0, s0 + 32768)
            outs.append(F.scaled_dot_product_attention(
                sp(q[sl]), sp(k[sl]), sp(v[sl]), attn_mask=attention_bias,
            ).transpose(1, 2).reshape(-1, h.shape[1], C))
        o = self.o(torch.cat(outs, 0))
        if self.axis == "spatial":
            o = o.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        else:
            o = o.reshape(B, H, W, T, C).permute(0, 4, 3, 1, 2)
        return x + o


class UNet3D(nn.Module):
    def __init__(self, ch=64, mult=(1, 2, 4, 4), attn_res=(32, 16), tattn_res=(64, 32, 16), n_res=2, in_ch=4, n_classes=0, size=128, cond_ch=0,
                 text_dim=0, temporal_neighbors=0, temporal_pos_bias=False):
        super().__init__()
        self.cond_ch = cond_ch                       # autoregressive / I2V conditioning: clean past frames (in_ch) + binary mask (1)
        temb = ch * 4
        self.temb = nn.Sequential(nn.Linear(ch, temb), nn.SiLU(), nn.Linear(temb, temb))
        self.ch = ch
        self.cls = nn.Embedding(n_classes + 1, temb) if n_classes else None   # last = null class
        self.text_proj = nn.Linear(text_dim, temb) if text_dim else None
        self.inp = nn.Conv3d(in_ch + cond_ch, ch, 3, padding=1)
        self.down = nn.ModuleList(); chans = [ch]; c = ch; res = size
        for i, m in enumerate(mult):
            for _ in range(n_res):
                blocks = nn.ModuleList([ResBlock(c, ch * m, temb)]); c = ch * m
                if res in attn_res: blocks.append(Attn(c, "spatial"))
                if res in tattn_res: blocks.append(Attn(c, "temporal", neighbor_radius=temporal_neighbors, relative_time=temporal_pos_bias))
                self.down.append(blocks); chans.append(c)
            if i < len(mult) - 1:
                self.down.append(nn.ModuleList([nn.Conv3d(c, c, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))])); chans.append(c); res //= 2
        self.mid = nn.ModuleList([ResBlock(c, c, temb), Attn(c, "spatial"),
                                  Attn(c, "temporal", neighbor_radius=temporal_neighbors, relative_time=temporal_pos_bias),
                                  ResBlock(c, c, temb)])
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mult))):
            for _ in range(n_res + 1):
                blocks = nn.ModuleList([ResBlock(c + chans.pop(), ch * m, temb)]); c = ch * m
                if res in attn_res: blocks.append(Attn(c, "spatial"))
                if res in tattn_res: blocks.append(Attn(c, "temporal", neighbor_radius=temporal_neighbors, relative_time=temporal_pos_bias))
                self.up.append(blocks)
            if i > 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"), nn.Conv3d(c, c, (1, 3, 3), padding=(0, 1, 1))])); res *= 2
        self.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv3d(c, in_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    grad_ckpt = False

    def _run(self, blocks, h, temb):
        h = blocks[0](h, temb)
        for b in blocks[1:]: h = b(h)
        return h

    def forward(self, x, t, y=None, cond=None, text=None, text_mask=None):
        if self.cond_ch:
            if cond is None: cond = torch.zeros(x.shape[0], self.cond_ch, *x.shape[2:], device=x.device, dtype=x.dtype)
            x = torch.cat([x, cond.to(x.dtype)], 1)
        temb = self.temb(timestep_embedding(t, self.ch))
        if self.cls is not None: temb = temb + self.cls(y)
        if self.text_proj is not None:
            if text is None: raise ValueError("text-conditioned UNet3D requires text embeddings")
            weights = text_mask.to(text.dtype).unsqueeze(-1) if text_mask is not None else torch.ones_like(text[..., :1])
            pooled = (text * weights).sum(1) / weights.sum(1).clamp_min(1)
            temb = temb + self.text_proj(pooled.to(temb.dtype))
        ck = (lambda f, *a: checkpoint(f, *a, use_reentrant=False)) if (self.grad_ckpt and self.training) else (lambda f, *a: f(*a))
        h = conv3d_t1(self.inp, x); hs = [h]
        for blocks in self.down:
            h = ck(self._run, blocks, h, temb) if isinstance(blocks[0], ResBlock) else blocks[0](h)
            hs.append(h)
        for b in self.mid: h = b(h, temb) if isinstance(b, ResBlock) else b(h)
        for blocks in self.up:
            if isinstance(blocks[0], ResBlock):
                h = ck(self._run, blocks, torch.cat([h, hs.pop()], 1), temb)
            else:
                for b in blocks: h = b(h)
        return conv3d_t1(self.out[2], F.silu(self.out[0](h)))


# ----------------------------------------------------------------- diffusion (v-pred, cosine)
def alphas_cumprod(T=1000, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ac = (f / f[0]).clamp(1e-5, 1.0)
    return ac[1:].float()


@torch.no_grad()
def ar_cond(x0, K):
    """Conditioning tensor for chunked autoregressive diffusion: clean first K frames of the window + mask -> [B,5,T,H,W]
    (K=0 -> all zeros = unconditional first chunk)."""
    c = torch.zeros_like(x0); m = torch.zeros_like(x0[:, :1])
    if K > 0: c[:, :, :K] = x0[:, :, :K]; m[:, :, :K] = 1
    return torch.cat([c, m], 1)


@torch.no_grad()
def rollout(model, B, K, F, n_chunks, ac, device, steps=50, S=64, y=None, cfg=0.0, null_y=None, generator=None,
            text=None, text_mask=None, null_text=None, null_text_mask=None):
    """Chunked autoregressive generation: chunk 1 = a full K+F window with no context; each later chunk conditions on the
    last K generated frames and contributes F new frames. Returns [B,4,K+F+(n_chunks-1)*F,S,S]."""
    T = K + F; frames = None
    for c in range(n_chunks):
        noise = torch.randn((B, 4, T, S, S), device=device, generator=generator)
        if frames is None:
            cond = ar_cond(noise, 0)
        else:
            ctx = torch.zeros((B, 4, T, S, S), device=device); ctx[:, :, :K] = frames[:, :, -K:]
            cond = ar_cond(ctx, K)
        x = sample(model, noise.shape, ac, device, steps=steps, y=y, cfg=cfg, null_y=null_y, noise=noise, cond=cond,
                   text=text, text_mask=text_mask, null_text=null_text, null_text_mask=null_text_mask)
        frames = x if frames is None else torch.cat([frames, x[:, :, K:]], 2)
    return frames


@torch.no_grad()
def sample(model, shape, ac, device, steps=50, y=None, cfg=0.0, null_y=None, noise=None, cond=None,
           text=None, text_mask=None, null_text=None, null_text_mask=None):
    x = torch.randn(shape, device=device) if noise is None else noise.clone()
    ts = torch.linspace(len(ac) - 1, 0, steps + 1).long().to(device)
    for i in range(steps):
        t, tn = ts[i], ts[i + 1]
        a, an = ac[t], (ac[tn] if i < steps - 1 else torch.tensor(1.0, device=device))
        tt = torch.full((shape[0],), int(t), device=device)
        with autocast():
            v = model(x, tt, y, cond, text, text_mask)
            if cfg > 0 and (y is not None or text is not None):
                vu = model(x, tt, null_y, cond, null_text, null_text_mask); v = vu + cfg * (v - vu)
        v = v.float()
        x0 = (a.sqrt() * x - (1 - a).sqrt() * v).clamp(-1, 1)
        eps = (x - a.sqrt() * x0) / (1 - a).sqrt().clamp_min(1e-8)   # recomputed from the clamped x0
        x = an.sqrt() * x0 + (1 - an).sqrt() * eps
    return x


def to_gif(x, path, fps=10):
    """Write a sample grid and return its actual path (PNG for T=1, GIF otherwise)."""
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
    if T == 1:                                     # image model: write a PNG grid instead of a 1-frame GIF
        destination = path[:-4] + ".png"
        imageio.imwrite(destination, frames[0])
        return destination
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)
    return path


# ----------------------------------------------------------------- train
PRESETS = {  # name: overrides. VRAM/speed numbers filled in from measured runs (see REPORT.md §5).
    "4090-fast":   dict(size=64, frames=8, batch=16, ch=64, grad_ckpt=False, accum=1),
    "4090-full":   dict(size=128, frames=16, batch=4, ch=64, grad_ckpt=True, accum=2),
    "4090-mid":    dict(size=128, frames=8, batch=8, ch=64, grad_ckpt=True, accum=1),
    "runpod-96gb": dict(size=128, frames=16, batch=8, ch=64, grad_ckpt=False, accum=1),
}


def worker_init(_):
    seed = torch.utils.data.get_worker_info().seed % 2**32
    random.seed(seed); np.random.seed(seed)


@torch.no_grad()
def val_losses(model, dl, ac, dev, ar_ctx=0, ctx_noise=0.0, n_batches=16, seed=0, text_batch=None):
    """Deterministic validation for unconditional and AR-continuation paths.

    ``first_chunk`` denoises the complete window with no context.  For AR
    models, ``continuation`` receives a deterministically corrupted ground-
    truth prefix and scores only the newly generated frames.
    """
    was_training = model.training
    model.eval(); totals = {"first_chunk": 0.0}; n = 0
    if ar_ctx > 0:
        totals["continuation"] = 0.0
    generator = torch.Generator(device=dev).manual_seed(seed)
    for i, (x, labels) in enumerate(dl):
        if i >= n_batches: break
        x = x.to(dev)
        t = torch.randint(0, len(ac), (x.shape[0],), device=dev, generator=generator)
        at = ac[t][:, None, None, None, None]
        eps = torch.randn(x.shape, device=dev, dtype=x.dtype, generator=generator)
        yy = labels.to(dev) if model.cls is not None else None                  # class-conditional: use true labels
        text, text_mask = text_batch(labels) if getattr(model, "text_proj", None) is not None else (None, None)
        with autocast():
            xt = at.sqrt() * x + (1 - at).sqrt() * eps
            target = at.sqrt() * eps - (1 - at).sqrt() * x
            pred = model(xt, t, yy, None, text, text_mask) if text is not None else model(xt, t, yy, None)
        totals["first_chunk"] += F.mse_loss(pred.float(), target).item()
        if ar_ctx > 0:
            if ar_ctx >= x.shape[2]:
                raise ValueError(f"ar_ctx={ar_ctx} must be smaller than validation window T={x.shape[2]}")
            strength = torch.rand((x.shape[0], 1, 1, 1, 1), device=dev, generator=generator) * ctx_noise
            ctx_eps = torch.randn(x.shape, device=dev, dtype=x.dtype, generator=generator)
            ctx = (1 - strength).sqrt() * x + strength.sqrt() * ctx_eps
            cond = ar_cond(ctx, ar_ctx)
            with autocast():
                pred_cont = model(xt, t, yy, cond, text, text_mask) if text is not None else model(xt, t, yy, cond)
            totals["continuation"] += F.mse_loss(pred_cont[:, :, ar_ctx:].float(), target[:, :, ar_ctx:]).item()
        n += 1
    model.train(was_training)
    return {key: value / max(n, 1) for key, value in totals.items()}


@torch.no_grad()
def val_loss(model, dl, ac, dev, n_batches=16, seed=0):
    """Backward-compatible unconditional/first-chunk validation scalar."""
    return val_losses(model, dl, ac, dev, n_batches=n_batches, seed=seed)["first_chunk"]


def adapt_warm_start_state(src, own, image_source=False):
    """Adapt a checkpoint state dict to a model without temporal leakage.

    A T=1 image run only trains the centre slice of 3-frame temporal kernels.
    The input convolution is not zero-initialised, so its untouched outer
    slices must be cleared before the weights are activated on video.
    """
    src = {k: v.clone() for k, v in src.items()
           if k in own and (v.shape == own[k].shape or k.endswith("pos_t") or k == "inp.weight")}
    if "inp.weight" in src and src["inp.weight"].shape != own["inp.weight"].shape:
        w = torch.zeros_like(own["inp.weight"])
        w[:, : src["inp.weight"].shape[1]] = src["inp.weight"]
        src["inp.weight"] = w
    if image_source and "inp.weight" in src and src["inp.weight"].shape[2] == 3:
        src["inp.weight"][:, :, 0].zero_()
        src["inp.weight"][:, :, 2].zero_()
    if "pos_t" in src and src["pos_t"].shape != own["pos_t"].shape:
        src["pos_t"] = src["pos_t"].expand_as(own["pos_t"]).clone()
    return src


@torch.no_grad()
def initialize_video_input(model, image_channels=4):
    """Give scratch AR models the same structural zero-init as warm starts.

    Only the centre spatial kernel over image channels starts random. Temporal
    outer slices and newly introduced context channels start inactive in both
    conditions, isolating learned image weights as the experimental variable.
    """
    weight = model.inp.weight
    if weight.shape[2] == 3:
        weight[:, :, 0].zero_()
        weight[:, :, 2].zero_()
    if model.cond_ch:
        weight[:, image_channels:].zero_()


AMP = {"dtype": torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16, "on": True}   # T4/V100: fp16


def autocast():
    return torch.autocast("cuda", dtype=AMP["dtype"], enabled=AMP["on"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=16); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--cond", default="none", choices=["none", "group", "text"]); ap.add_argument("--cfg_drop", type=float, default=0.1)
    ap.add_argument("--text_encoder", default="google-t5/t5-small", help="frozen Hugging Face text encoder for --cond text")
    ap.add_argument("--text_len", type=int, default=32)
    ap.add_argument("--sample_every", type=int, default=2000); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", default="")
    ap.add_argument("--ar_ctx", type=int, default=0, help="chunked autoregressive diffusion: condition on K clean past frames (window = K + --frames); 0 = off")
    ap.add_argument("--ctx_drop", type=float, default=0.2, help="fraction of AR training samples with no context (first-chunk mode)")
    ap.add_argument("--ctx_noise", type=float, default=0.1, help="max noise variance mixed into context frames during training (drift robustness)")
    ap.add_argument("--rollout", type=int, default=5, help="chunks generated for AR sample GIFs (K+F+(n-1)F frames)")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"], help="mixed precision: bf16 (Ampere+), fp16 + GradScaler (T4/V100/Colab), off")
    ap.add_argument("--min_snr", type=float, default=0.0, help="min-SNR-gamma loss weight for v-pred (Hang et al. 2023), e.g. 5; 0 = off")
    ap.add_argument("--lr_final", type=float, default=0.1, help="cosine decays to this fraction of --lr")
    ap.add_argument("--init", default="", help="warm-start weights (model+ema) from an image/other checkpoint; step/opt fresh (Seedance stage 2)")
    ap.add_argument("--preset", default="", choices=[""] + list(PRESETS)); ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--grad_ckpt", action="store_true"); ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--temporal_neighbors", type=int, default=0,
                    help="spatial radius gathered into temporal attention keys/values; 1 enables a 3x3 neighbourhood")
    ap.add_argument("--temporal_pos_bias", action="store_true",
                    help="learn a relative-time attention bias (inactive for T=1)")
    ap.add_argument("--fast", action="store_true", help="cudnn.benchmark, tf32-high, channels_last_3d (UNet), fused AdamW, foreach EMA")
    ap.add_argument("--compile", action="store_true", help="torch.compile per block (regional); test on sm_120 first")
    a = ap.parse_args()
    AMP["dtype"] = torch.float16 if a.amp == "fp16" else torch.bfloat16; AMP["on"] = a.amp != "off"
    scaler = torch.amp.GradScaler("cuda", enabled=(a.amp == "fp16"))
    if a.preset:
        for k, v in PRESETS[a.preset].items(): setattr(a, k, v)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    if a.fast:
        torch.backends.cudnn.benchmark = True; torch.set_float32_matmul_precision("high")
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    dev = "cuda"
    T_win = a.frames + a.ar_ctx                                              # AR: window = K context + F new frames
    ds = VideoWindows(a.cache, T_win, "train", a.stride, size=a.size, return_text=a.cond == "text")
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True,
                                     persistent_workers=True, worker_init_fn=worker_init)
    vds = VideoWindows(a.cache, T_win, "val", a.stride, size=a.size, return_text=a.cond == "text", deterministic=True)
    vdl = torch.utils.data.DataLoader(vds, batch_size=a.batch, shuffle=False, num_workers=2, drop_last=True,
                                      worker_init_fn=worker_init) if len(vds) else None
    n_cls = len(ds.groups) if a.cond == "group" else 0
    text_cache = {}; text_dim = 0
    if a.cond == "text":
        from transformers import AutoTokenizer, T5EncoderModel
        tokenizer = AutoTokenizer.from_pretrained(a.text_encoder)
        encoder = T5EncoderModel.from_pretrained(a.text_encoder).to(dev).eval().requires_grad_(False)
        prompts = sorted({c["text"] for c in ds.clips + vds.clips} | {""})
        for start in range(0, len(prompts), 32):
            batch_prompts = prompts[start:start + 32]
            tokens = tokenizer(batch_prompts, padding="max_length", truncation=True,
                               max_length=a.text_len, return_tensors="pt")
            mask = tokens.attention_mask
            with torch.no_grad(), autocast():
                hidden = encoder(input_ids=tokens.input_ids.to(dev),
                                 attention_mask=mask.to(dev)).last_hidden_state.float().cpu()
            for prompt, hidden_, mask_ in zip(batch_prompts, hidden, mask):
                text_cache[prompt] = (hidden_, mask_)
        text_dim = encoder.config.d_model
        del encoder; torch.cuda.empty_cache()
        print(f"cached {len(text_cache) - 1} prompts with {a.text_encoder} ({text_dim}d, frozen)", flush=True)

    def text_batch(prompts):
        if not text_cache:
            raise ValueError("text_batch called without --cond text")
        hidden, masks = zip(*(text_cache[prompt] for prompt in prompts))
        return (torch.stack(hidden).to(dev, non_blocking=True),
                torch.stack(masks).to(dev, non_blocking=True))

    model = UNet3D(ch=a.ch, n_classes=n_cls, size=a.size, cond_ch=5 if a.ar_ctx > 0 else 0,
                   text_dim=text_dim, temporal_neighbors=a.temporal_neighbors,
                   temporal_pos_bias=a.temporal_pos_bias).to(dev)
    if a.ar_ctx > 0:
        initialize_video_input(model)
    model.grad_ckpt = a.grad_ckpt
    if a.fast: model = model.to(memory_format=torch.channels_last_3d)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if a.compile:   # compile forward only -> state_dict keys unchanged (do not combine with --grad_ckpt)
        for blocks in list(model.down) + list(model.up):
            if isinstance(blocks[0], ResBlock):
                torch._dynamo.config.cache_size_limit = 64          # train batch / sample chunk / val batch shapes
                for b in blocks: b.forward = torch.compile(b.forward, dynamic=False)
        for b in model.mid: b.forward = torch.compile(b.forward, dynamic=False)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params {n_params:.1f}M · {len(ds.clips)} train clips / {len(vds.clips)} val · {a.size}px · frames {a.frames} · batch {a.batch}×{a.accum} · ckpt {a.grad_ckpt} · cond {a.cond}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.01, fused=a.fast)
    ac = alphas_cumprod().to(dev)
    step = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev); model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    elif a.init:
        ck = torch.load(a.init, map_location=dev)
        image_source = ck.get("args", {}).get("frames", 1) == 1
        for tgt, key in ((model, "model"), (ema, "ema")):
            src = ck.get(key) or ck["ema"]; own = tgt.state_dict()
            src = adapt_warm_start_state(src, own, image_source=image_source)
            miss = tgt.load_state_dict(src, strict=False)
            print(f"init {key} from {a.init} step {ck.get('step')}: {len(src)} tensors, missing {len(miss.missing_keys)}, unexpected {len(miss.unexpected_keys)}, image_source {image_source}", flush=True)
    log = open(os.path.join(a.out, "log.txt"), "a")
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(a.out, "tb"))
    except Exception:
        tb = None
    t0 = time.time(); step0 = step; it = iter(dl); ema_loss = None
    while step < a.steps:
        try: x, labels = next(it)
        except StopIteration: it = iter(dl); x, labels = next(it)
        x = x.to(dev, non_blocking=True)
        if a.fast: x = x.contiguous(memory_format=torch.channels_last_3d)
        text, text_mask = (None, None)
        if a.cond == "text":
            text, text_mask = text_batch(labels)
            null_text, null_text_mask = text_batch([""] * x.shape[0])
            drop = torch.rand(x.shape[0], device=dev) < a.cfg_drop
            text = torch.where(drop[:, None, None], null_text, text)
            text_mask = torch.where(drop[:, None], null_text_mask, text_mask)
            y = None
        elif n_cls:
            y = labels.to(dev)
            drop = torch.rand(y.shape[0], device=dev) < a.cfg_drop
            y = torch.where(drop, torch.full_like(y, n_cls), y)
        else:
            y = None
        t = torch.randint(0, len(ac), (x.shape[0],), device=dev)
        at = ac[t][:, None, None, None, None]
        eps = torch.randn_like(x)
        xt = at.sqrt() * x + (1 - at).sqrt() * eps
        v = at.sqrt() * eps - (1 - at).sqrt() * x
        cond = None; lmask = None
        if a.ar_ctx > 0:                                                            # chunked AR: clean (lightly noised) past K frames as conditioning
            K = a.ar_ctx; has_ctx = (torch.rand(x.shape[0], device=dev) >= a.ctx_drop).float()[:, None, None, None, None]
            s_ = torch.rand(x.shape[0], device=dev)[:, None, None, None, None] * a.ctx_noise
            ctx = (1 - s_).sqrt() * x + s_.sqrt() * torch.randn_like(x)
            cond = ar_cond(ctx, K) * has_ctx                                          # ctx dropped -> zeros + mask 0 (first-chunk mode)
            lmask = torch.ones_like(x[:, :1]); lmask[:, :, :K] = 1 - has_ctx          # loss only on the new F frames when context is given
        # warmup 1000 then cosine decay to 10 %
        prog = max(0.0, (step - 1000) / max(1, a.steps - 1000))
        lr = a.lr * min(1.0, (step + 1) / 1000) * (a.lr_final + (1 - a.lr_final) * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups: g["lr"] = lr
        with autocast():
            pred = model(xt, t, y, cond, text, text_mask)
        err = (pred.float() - v) ** 2
        if lmask is not None: err = err * lmask * (lmask.numel() / lmask.sum().clamp_min(1))
        if a.min_snr > 0:                                        # v-pred min-SNR-γ: w = min(SNR,γ)/(SNR+1)
            snr_ = (at / (1 - at)).flatten()
            w = (snr_.clamp(max=a.min_snr) / (snr_ + 1))[:, None, None, None, None]
            loss = (err * w).mean() / a.accum
        else:
            loss = err.mean() / a.accum
        with torch.no_grad():                                   # diagnostics only
            fgm = (x[:, 3:4] > -0.9).float()                    # alpha > 0.05
            fg_loss = float((err * fgm).sum() / fgm.sum().clamp_min(1))
            snr = (at / (1 - at)).flatten(); lb = float(err.flatten(1).mean(1)[snr < 0.1].mean()) if (snr < 0.1).any() else float("nan")
            hb = float(err.flatten(1).mean(1)[snr > 10].mean()) if (snr > 10).any() else float("nan")
        scaler.scale(loss).backward()
        if (step + 1) % a.accum != 0:
            step += 1; continue
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        loss = loss * a.accum
        with torch.no_grad():
            d = min(0.999 if step < 5000 else 0.9995, (1 + step) / (10 + step))   # EMA warmup: shorter horizon early
            if a.fast: torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - d)
            else:
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - d)
        step += 1
        ema_loss = loss.item() if ema_loss is None else 0.98 * ema_loss + 0.02 * loss.item()
        if step == 10 or step % 50 == 0:
            spi = (time.time() - t0) / max(1, step - step0)
            msg = f"step {step} loss {ema_loss:.4f} fg {fg_loss:.4f} lowsnr {lb:.4f} highsnr {hb:.4f} lr {lr:.2e} {spi:.2f}s/it peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB ETA {(a.steps-step)*spi/3600:.1f}h"
            validation = None
            if vdl is not None and step % a.val_every == 0:
                validation = val_losses(ema, vdl, ac, dev, ar_ctx=a.ar_ctx, ctx_noise=a.ctx_noise,
                                        text_batch=text_batch if a.cond == "text" else None)
                msg += f" val_first {validation['first_chunk']:.4f}"
                if "continuation" in validation:
                    msg += f" val_cont {validation['continuation']:.4f}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            if tb:
                tb.add_scalar("loss/train_ema", ema_loss, step); tb.add_scalar("lr", lr, step)
                tb.add_scalar("loss/fg", fg_loss, step); tb.add_scalar("loss/low_snr", lb, step); tb.add_scalar("loss/high_snr", hb, step)
                tb.add_scalar("perf/s_per_it", spi, step); tb.add_scalar("perf/peak_gb", torch.cuda.max_memory_allocated() / 1e9, step)
                if validation is not None:
                    tb.add_scalar("loss/val_first_chunk", validation["first_chunk"], step)
                    if "continuation" in validation:
                        tb.add_scalar("loss/val_continuation", validation["continuation"], step)
        if step % a.sample_every == 0 or step == a.steps:
            NS, CH = (64, 32) if a.frames == 1 else (16, 8)                          # 16 fixed-noise samples in chunks of 8 (24 GB cards); 64 for image models
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            ys = torch.arange(NS, device=dev) % n_cls if n_cls else None
            prompt_pool = sorted({c["text"] for c in (vds.clips or ds.clips)}) if a.cond == "text" else []
            sample_prompts = [prompt_pool[(i * len(prompt_pool)) // NS] for i in range(NS)] if prompt_pool else None
            sample_text, sample_text_mask = text_batch(sample_prompts) if sample_prompts else (None, None)
            sample_null, sample_null_mask = text_batch([""] * NS) if sample_prompts else (None, None)
            sample_cfg = 3.0 if sample_prompts else (2.0 if n_cls else 0.0)
            def _samp(m_):
                outs = []
                g = torch.Generator(device=dev).manual_seed(1234)                   # FIXED noise -> comparable across steps/models
                for i in range(0, NS, CH):
                    yy = ys[i:i + CH] if ys is not None else None
                    B_ = min(CH, NS - i)
                    null = torch.full((B_,), n_cls, device=dev) if n_cls else None
                    if a.ar_ctx > 0:
                        # The first chunk is K+F frames with no visual context.
                        # not the F-frame continuation target used in training.
                        out = rollout(m_, B_, a.ar_ctx, a.frames, 1, ac, dev, steps=50, S=a.size,
                                      y=yy, cfg=sample_cfg, null_y=null, generator=g,
                                      text=sample_text[i:i + CH] if sample_text is not None else None,
                                      text_mask=sample_text_mask[i:i + CH] if sample_text_mask is not None else None,
                                      null_text=sample_null[i:i + CH] if sample_null is not None else None,
                                      null_text_mask=sample_null_mask[i:i + CH] if sample_null_mask is not None else None)
                    else:
                        noise = torch.randn((B_, 4, a.frames, a.size, a.size), device=dev, generator=g)
                        out = sample(m_, noise.shape, ac, dev, steps=50, y=yy, cfg=sample_cfg,
                                     null_y=null, noise=noise,
                                     text=sample_text[i:i + CH] if sample_text is not None else None,
                                     text_mask=sample_text_mask[i:i + CH] if sample_text_mask is not None else None,
                                     null_text=sample_null[i:i + CH] if sample_null is not None else None,
                                     null_text_mask=sample_null_mask[i:i + CH] if sample_null_mask is not None else None)
                    outs.append(out.cpu())
                return torch.cat(outs, 0)
            xs = _samp(ema)
            sample_name = f"first_chunk_{step:06d}.gif" if a.ar_ctx > 0 else f"sample_{step:06d}.gif"
            sample_path = to_gif(xs, os.path.join(a.out, sample_name))
            sample_name = os.path.basename(sample_path)
            if a.ar_ctx > 0:                                                        # long dance: chunked rollout, 8 samples
                gr = torch.Generator(device=dev).manual_seed(4321)
                xr_ = rollout(ema, 8, a.ar_ctx, a.frames, a.rollout, ac, dev, steps=50, S=a.size,
                              cfg=sample_cfg, generator=gr,
                              text=sample_text[:8] if sample_text is not None else None,
                              text_mask=sample_text_mask[:8] if sample_text_mask is not None else None,
                              null_text=sample_null[:8] if sample_null is not None else None,
                              null_text_mask=sample_null_mask[:8] if sample_null_mask is not None else None).cpu()
                to_gif(xr_, os.path.join(a.out, f"rollout_{step:06d}.gif"), fps=int(round(20 / a.stride)))
            if step <= 10000:                                                       # early: also raw weights (EMA lags)
                model.eval(); xr = _samp(model); model.train()
                to_gif(xr, os.path.join(a.out, f"sample_raw_{step:06d}.gif"))
            torch.cuda.empty_cache()
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, args=vars(a), groups=ds.groups),
                       os.path.join(a.out, "ckpt.pt"))
            torch.save(dict(ema=ema.state_dict(), step=step, args=vars(a), groups=ds.groups),          # history (EMA only)
                       os.path.join(a.out, f"ckpt_{step:06d}.pt"))
            json.dump(dict(step=step, seed=1234, prompts=sample_prompts, cfg=sample_cfg,
                           sampler="DDIM", steps=50, output=sample_name),
                      open(os.path.join(a.out, f"sample_manifest_{step:06d}.json"), "w"), indent=2)
            print(f"  wrote {sample_name}", flush=True)
            if tb:   # first frame of the sample grid as an image, whole clip as video
                v = ((xs.clamp(-1, 1) + 1) / 2)
                rgb = (v[:, :3] + (1 - v[:, 3:4])).clamp(0, 1)          # premult over white -> [B,3,T,H,W]
                try: tb.add_video("samples", rgb.permute(0, 2, 1, 3, 4).cpu(), step, fps=10)
                except Exception: pass
            # Sampling can take minutes on a T4. Exclude that pause from the
            # subsequent seconds/iteration and ETA measurements.
            t0, step0 = time.time(), step


if __name__ == "__main__":
    main()
