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
from train.video_ddpm import VideoWindows, UNet3D, to_gif, worker_init, PRESETS


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


class TextCrossAttention(nn.Module):
    """Token-level text cross-attention. Zero output init preserves the unconditional DiT at step zero."""
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads; self.q = nn.Linear(dim, dim); self.kv = nn.Linear(dim, 2 * dim); self.o = nn.Linear(dim, dim)
        self.qn = nn.LayerNorm(dim // heads); self.kn = nn.LayerNorm(dim // heads)
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)

    def forward(self, x, text, text_mask=None):               # x [B,Q,D], text [B,L,D]
        B, Q, D = x.shape; L = text.shape[1]
        q = self.q(x).reshape(B, Q, self.h, D // self.h).transpose(1, 2)
        k, v = self.kv(text).reshape(B, L, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = self.qn(q), self.kn(k)
        mask = text_mask[:, None, None, :].bool() if text_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, Q, D)
        return self.o(out)


class Local3DMixer(nn.Module):
    """Residual 3x3x3 mixer over neighbouring frames and patch locations.

    Factorised attention exchanges information either within one frame or
    along one fixed patch trajectory. This zero-initialised branch adds a
    direct local spatiotemporal path without changing a warm-started model's
    initial function.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.depthwise = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.pointwise = nn.Conv3d(dim, dim, 1)
        nn.init.zeros_(self.pointwise.weight); nn.init.zeros_(self.pointwise.bias)

    def forward(self, x):                    # x [B,T,N,D]
        B, T, N, D = x.shape
        side = math.isqrt(N)
        if side * side != N:
            raise ValueError(f"local 3D mixer requires a square patch grid, got {N} tokens")
        h = self.norm(x).reshape(B, T, side, side, D).permute(0, 4, 1, 2, 3)
        h = self.pointwise(F.silu(self.depthwise(h)))
        h = h.permute(0, 2, 3, 4, 1).reshape(B, T, N, D)
        return x + h


class ConvStem(nn.Module):
    """Learned f8t4 front-end: overlapping conv downsampling (t x4, s x8) with SiLU,
    giving VAE-style contextual spatio-temporal mixing without a separately trained
    codec -- trained end-to-end under the diffusion objective."""
    def __init__(self, in_ch, dim):
        super().__init__()
        c = dim // 4
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, c, 3, stride=(2, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(c, 2 * c, 3, stride=(2, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(2 * c, dim, 3, stride=(1, 2, 2), padding=1),
        )

    def forward(self, x):                        # [B,C,T,H,W] -> [B,T/4,N,dim]
        h = self.net(x)                          # [B,dim,T/4,H/8,W/8]
        B, D, T, H, W = h.shape
        return h.permute(0, 2, 3, 4, 1).reshape(B, T, H * W, D)


class ConvHead(nn.Module):
    """Mirror of ConvStem: upsample tokens back to pixels (t x4, s x8)."""
    def __init__(self, dim, out_ch):
        super().__init__()
        c = dim // 4
        def up(ci, co, scale):
            return nn.Sequential(nn.Upsample(scale_factor=scale, mode="nearest"),
                                 nn.Conv3d(ci, co, 3, padding=1))
        self.u1 = up(dim, 2 * c, (1, 2, 2)); self.a1 = nn.SiLU()
        self.u2 = up(2 * c, c, (2, 2, 2)); self.a2 = nn.SiLU()
        self.u3 = up(c, out_ch, (2, 2, 2))
        nn.init.zeros_(self.u3[1].weight); nn.init.zeros_(self.u3[1].bias)

    def forward(self, h, side):                  # [B,T/4,N,dim] -> [B,C,T,H,W]
        B, T, N, D = h.shape
        x = h.reshape(B, T, side, side, D).permute(0, 4, 1, 2, 3)
        return self.u3(self.a2(self.u2(self.a1(self.u1(x)))))


class Block(nn.Module):
    """DiT block with adaLN-Zero. axis='spatial' attends within frame, 'temporal' across frames."""
    def __init__(self, dim, heads, axis, mlp=4, text_cond=False, local_3d=False):
        super().__init__()
        self.axis = axis
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6); self.attn = Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp * dim), nn.GELU(approximate="tanh"), nn.Linear(mlp * dim, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.text_attn = TextCrossAttention(dim, heads) if text_cond else None
        self.text_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) if text_cond else None
        self.local_mixer = Local3DMixer(dim) if local_3d else None
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    t1_skip = True                             # ib64 (trained before the skip existed) carries args["t1_skip"]=False; loaders honour it

    def forward(self, x, c, text=None, text_mask=None):       # x [B,T,N,D], c [B,D]
        B, T, N, D = x.shape
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, -1)
        if self.axis == "spatial":
            h = x.reshape(B * T, N, D); rep = lambda z: z.repeat_interleave(T, 0)
        elif self.axis == "full":              # joint spatio-temporal attention over all T*N tokens
            h = x.reshape(B, T * N, D); rep = lambda z: z
        else:
            h = x.permute(0, 2, 1, 3).reshape(B * N, T, D); rep = lambda z: z.repeat_interleave(N, 0)
        if not (self.axis == "temporal" and T == 1 and self.t1_skip):   # image mode: nothing to attend across (gate stays 0 -> identity when video-initialised)
            h = h + rep(g1).unsqueeze(1) * self.attn(modulate(self.n1(h), rep(s1), rep(sc1)))
        h = h + rep(g2).unsqueeze(1) * self.mlp(modulate(self.n2(h), rep(s2), rep(sc2)))
        if self.axis == "spatial" or self.axis == "full":
            h = h.reshape(B, T, N, D)
        else:
            h = h.reshape(B, N, T, D).permute(0, 2, 1, 3)
        if self.text_attn is not None and text is not None:
            flat = h.reshape(B, T * N, D)
            flat = flat + self.text_attn(self.text_norm(flat), text, text_mask)
            h = flat.reshape(B, T, N, D)
        if self.local_mixer is not None:
            h = self.local_mixer(h)
        return h


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * 1000.0 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


class VideoDiT(nn.Module):
    def __init__(self, size=128, frames=16, patch=4, in_ch=4, dim=384, depth=12, heads=6, n_classes=0, cond_ch=0,
                 text_dim=0, local_3d=False, full_st=False, patch_t=1, conv_stem=False):
        super().__init__()
        self.conv_stem = conv_stem
        if conv_stem:
            patch, patch_t = 8, 4                                  # stem geometry is fixed f8t4
        if frames % patch_t:
            raise ValueError("frames must divide patch_t")
        self.full_st = full_st
        self.p, self.pt, self.T, self.S, self.C, self.dim = patch, patch_t, frames, size, in_ch, dim
        self.cond_ch = cond_ch                                     # I2V (Seedance §2.2): concat clean/zero frames + binary mask -> in_ch+in_ch+1
        self.N = (size // patch) ** 2
        if conv_stem:
            self.stem = ConvStem(in_ch + cond_ch, dim)
            self.head = ConvHead(dim, in_ch)
        self.embed = nn.Linear((in_ch + cond_ch) * patch * patch * patch_t, dim)
        self.pos_s = nn.Parameter(torch.zeros(1, 1, self.N, dim)); self.pos_t = nn.Parameter(torch.zeros(1, frames // patch_t, 1, dim))
        nn.init.normal_(self.pos_s, std=0.02); nn.init.normal_(self.pos_t, std=0.02)
        self.temb = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.cls = nn.Embedding(n_classes + 1, dim) if n_classes else None
        self.text_proj = nn.Linear(text_dim, dim) if text_dim else None
        self.blocks = nn.ModuleList([
            Block(dim, heads, "full" if full_st else ("spatial" if i % 2 == 0 else "temporal"),
                  text_cond=bool(text_dim), local_3d=local_3d)
            for i in range(depth)
        ])
        self.nf = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, in_ch * patch * patch * patch_t)
        nn.init.zeros_(self.ada_f[-1].weight); nn.init.zeros_(self.ada_f[-1].bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.grad_ckpt = False

    def patchify(self, x):                     # [B,C,T,H,W] -> [B,T/pt,N,C*pt*p*p]
        B, C, T, H, W = x.shape; p, pt = self.p, self.pt
        x = x.reshape(B, C, T // pt, pt, H // p, p, W // p, p).permute(0, 2, 4, 6, 1, 3, 5, 7)
        return x.reshape(B, T // pt, (H // p) * (W // p), C * pt * p * p)

    def unpatchify(self, x):                   # [B,T/pt,N,C*pt*p*p] -> [B,C,T,H,W]
        B, Tp, N, _ = x.shape; p, pt = self.p, self.pt; h = w = int(math.sqrt(N))
        x = x.reshape(B, Tp, h, w, self.C, pt, p, p).permute(0, 4, 1, 5, 2, 6, 3, 7)
        return x.reshape(B, self.C, Tp * pt, h * p, w * p)

    def forward(self, x, t, y=None, cond=None, text=None, text_mask=None):
        if self.cond_ch:
            if cond is None: cond = torch.zeros(x.shape[0], self.cond_ch, *x.shape[2:], device=x.device, dtype=x.dtype)
            x = torch.cat([x, cond.to(x.dtype)], 1)
        h = (self.stem(x) if self.conv_stem else self.embed(self.patchify(x))) + self.pos_s + self.pos_t[:, : x.shape[2] // self.pt]
        c = self.temb(timestep_embedding(t, 256))
        if self.cls is not None: c = c + self.cls(y)
        if self.text_proj is not None:
            if text is None: raise ValueError("text-conditioned VideoDiT requires text embeddings")
            text = self.text_proj(text.to(h.dtype))
            weights = text_mask.to(text.dtype).unsqueeze(-1) if text_mask is not None else torch.ones_like(text[..., :1])
            c = c + (text * weights).sum(1) / weights.sum(1).clamp_min(1)
        for blk in self.blocks:
            h = checkpoint(blk, h, c, text, text_mask, use_reentrant=False) if (self.grad_ckpt and self.training) else blk(h, c, text, text_mask)
        s, sc = self.ada_f(c).chunk(2, -1)
        h = self.nf(h) * (1 + sc[:, None, None]) + s[:, None, None]
        if self.conv_stem:
            return self.head(h, self.S // self.p)
        return self.unpatchify(self.out(h))


# ----------------------------------------------------------------- flow matching
def sample_t(B, device, shift=1.0):
    """logit-normal(0,1) then optional shift toward noise: t' = shift*t / (1 + (shift-1)*t)."""
    t = torch.sigmoid(torch.randn(B, device=device))
    return shift * t / (1 + (shift - 1) * t)


def latent_weighted_mse(err, footprint, foreground_weight=1.0):
    """Latent-space analogue of foreground weighting: ``footprint`` is the
    per-latent-cell foreground occupancy in [0,1] carried by the cache's extra
    channel, so the weighting matches the pixel route's fg emphasis."""
    if foreground_weight < 1:
        raise ValueError("foreground_weight must be >= 1")
    if foreground_weight == 1:
        return err.mean()
    weights = 1 + (foreground_weight - 1) * footprint.to(err.dtype).clamp(0, 1)
    return (err * weights).mean() / weights.mean().clamp_min(1e-8)


def foreground_weighted_mse(err, clean_video, foreground_weight=1.0):
    """Average flow error while optionally upweighting visible RGBA pixels.

    ``foreground_weight`` is the absolute per-pixel multiplier (1 disables the
    ablation).  Dividing by the mean weight keeps the overall loss scale close
    to the unweighted baseline, so the learning-rate comparison stays useful.
    Soft ground-truth alpha preserves antialiased limb edges rather than
    converting them to a brittle foreground/background threshold.
    """
    if foreground_weight < 1:
        raise ValueError("foreground_weight must be >= 1")
    if foreground_weight == 1:
        return err.mean()
    alpha = ((clean_video[:, 3:4].to(err.dtype) + 1) / 2).clamp(0, 1)
    weights = 1 + (foreground_weight - 1) * alpha
    return (err * weights).mean() / weights.mean().clamp_min(1e-8)


def rgba_x0_disagreement(predicted_clean, clean_video, t):
    """Observable-space auxiliary used by the Mini-Wan decode-loss run.

    Pixel models already operate in RGBA space, so this applies the same alpha
    mismatch, foreground-colour, and weak background-colour terms directly to
    predicted x0 instead of passing it through a codec. Inputs use the
    trainer's [-1, 1] range; ``t`` is one scalar per video.
    """
    predicted = ((predicted_clean.float() + 1) / 2).clamp(0, 1)
    target = ((clean_video.float() + 1) / 2).clamp(0, 1)
    alpha_target, alpha_predicted = target[:, 3:4], predicted[:, 3:4]
    mismatch = (alpha_target - alpha_predicted).abs()
    weights = 1 + 9 * (alpha_target + mismatch).clamp(0, 1)
    rgb_error = (target[:, :3] - predicted[:, :3]) ** 2
    dims = (1, 2, 3, 4)
    normalizer = weights.mean(dims).clamp_min(1e-8)
    alpha_term = (weights * mismatch).mean(dims) / normalizer
    foreground_colour = (weights * alpha_target * alpha_predicted * rgb_error).mean(dims) / normalizer
    background_colour = 0.1 * ((1 - alpha_target) * rgb_error).mean(dims)
    per_video = alpha_term + foreground_colour + background_colour
    return ((1 - t.float()) * per_video).mean()


def optimizer_step_due(micro_step, accumulation):
    """Whether this zero-based microbatch completes an optimizer update."""
    if accumulation < 1:
        raise ValueError("accumulation must be >= 1")
    return (micro_step + 1) % accumulation == 0


def parse_step_set(spec):
    """Parse a comma-separated, non-negative milestone list."""
    if not spec:
        return set()
    try:
        steps = {int(item.strip()) for item in spec.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(f"invalid step list: {spec!r}") from exc
    if any(step < 0 for step in steps):
        raise ValueError("sample steps must be non-negative")
    return steps


def prepare_warmstart_state(source, target):
    """Select compatible tensors and adapt learned video-time positions.

    A one-frame checkpoint has no information about the identity of 50 video
    positions.  Copy spatial/common weights, but retain the target model's
    independently initialised temporal positions instead of tiling one value.
    When both checkpoints are video models, linearly interpolate the learned
    temporal positions so a 50-frame model can initialise a 60-frame model.
    """
    selected = {k: v for k, v in source.items()
                if k in target and (v.shape == target[k].shape or k == "embed.weight")}
    if "embed.weight" in selected and selected["embed.weight"].shape != target["embed.weight"].shape:
        source_weight = selected["embed.weight"]
        if source_weight.shape[0] != target["embed.weight"].shape[0] or source_weight.shape[1] > target["embed.weight"].shape[1]:
            selected.pop("embed.weight")
        else:
            weight = torch.zeros_like(target["embed.weight"])
            weight[:, :source_weight.shape[1]] = source_weight
            selected["embed.weight"] = weight
    if (
        "pos_t" in source and "pos_t" in target
        and source["pos_t"].shape != target["pos_t"].shape
        and source["pos_t"].shape[1] > 1 and target["pos_t"].shape[1] > 1
        and source["pos_t"].shape[2:] == target["pos_t"].shape[2:]
    ):
        value = source["pos_t"].squeeze(2).transpose(1, 2)
        value = F.interpolate(value.float(), size=target["pos_t"].shape[1], mode="linear", align_corners=True)
        selected["pos_t"] = value.transpose(1, 2).unsqueeze(2).to(dtype=target["pos_t"].dtype)
    return selected


@torch.no_grad()
def mixed_noise(shape, device, corr, generator=None):
    e = torch.randn(shape, device=device, generator=generator)
    if corr > 0 and shape[2] > 1:
        s = torch.randn(shape[:2] + (1,) + shape[3:], device=device, generator=generator).expand(shape)
        e = math.sqrt(1 - corr) * s + math.sqrt(corr) * e
    return e


def i2v_cond(x0, first=1):
    """Seedance-style conditioning tensor from clean clip x0 [B,4,T,H,W]: clean first `first` frames + zeros, and a binary
    frame mask -> [B,5,T,H,W]. For unconditional / T2V samples use zeros (mask 0)."""
    c = torch.zeros_like(x0); m = torch.zeros_like(x0[:, :1]); c[:, :, :first] = x0[:, :, :first]; m[:, :, :first] = 1
    return torch.cat([c, m], 1)


def diverse_text_prompts(ds, n):
    """First `n` unique prompts in dataset order (avoids adjacent camera/seed duplicates in sample grids)."""
    out = []
    for clip in ds.clips:
        prompt = clip["text"]
        if prompt not in out: out.append(prompt)
        if len(out) == n: break
    if not out: return [""] * n
    return [out[i % len(out)] for i in range(n)]


@torch.no_grad()
def euler_sample(model, shape, device, steps=50, y=None, cfg=0.0, null_y=None, noise=None, shift=1.0, cond=None,
                 text=None, text_mask=None, null_text=None, null_text_mask=None):
    x = torch.randn(shape, device=device) if noise is None else noise.clone()
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    ts = shift * ts / (1 + (shift - 1) * ts)
    for i in range(steps):
        t, tn = ts[i], ts[i + 1]
        tt = torch.full((shape[0],), float(t), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(x, tt, y, cond, text, text_mask)
            if cfg > 0 and (y is not None or text is not None):
                vu = model(x, tt, null_y, cond, null_text, null_text_mask); v = vu + cfg * (v - vu)
        x = x + (tn - t) * v.float()          # dx/dt = v = ε − x0 ; integrate t: 1 → 0
    return x.clamp(-1, 1)


# ----------------------------------------------------------------- train
def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="dit", choices=["dit", "resunet"], help="spatial backbone; both use this trainer's rectified-flow protocol")
    ap.add_argument("--frames", type=int, default=16); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--first_frames", type=int, default=0,
                    help="restrict training/validation windows to the first N frames of each clip (0 = whole clip)")
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--optimizer_beta2", type=float, default=0.95,
                    help="AdamW beta2; use .999 to match Mini-Wan")
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="gradient-norm clip; 0 disables clipping")
    ap.add_argument("--ema_max", type=float, default=0.0,
                    help="fixed EMA cap; 0 preserves the legacy .999/.9995 schedule")
    ap.add_argument("--optimizer_steps", action="store_true",
                    help="count --steps as optimizer updates rather than microbatches")
    ap.add_argument("--ch", type=int, default=64, help="ResUNet base channels (ignored by DiT)")
    ap.add_argument("--depth", type=int, default=12); ap.add_argument("--heads", type=int, default=6); ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--patch_t", type=int, default=1, help="temporal patch: one token spans this many frames (learned f-t patchify, no codec)")
    ap.add_argument("--cond", default="none", choices=["none", "group", "text"]); ap.add_argument("--cfg_drop", type=float, default=0.1)
    ap.add_argument("--text_encoder", default="google-t5/t5-small", help="frozen Hugging Face text encoder for --cond text")
    ap.add_argument("--text_len", type=int, default=32)
    ap.add_argument("--sample_every", type=int, default=2000); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--checkpoint_every", type=int, default=0,
                    help="write the rolling full-state ckpt.pt at this interval without running inference (0 = sample checkpoints only)")
    ap.add_argument("--early_sample_steps", default="", help="comma-separated extra inference/checkpoint steps, e.g. 0,1,5,10,25,50,100,250")
    ap.add_argument("--resume", default=""); ap.add_argument("--preset", default="", choices=[""] + list(PRESETS))
    ap.add_argument("--lr_final", type=float, default=0.1, help="cosine decays to this fraction of --lr")
    ap.add_argument("--init", default="", help="warm-start weights (model+ema) from an image/other checkpoint; step/opt fresh (Seedance stage 2)")
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--grad_ckpt", action="store_true"); ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--fast", action="store_true", help="cudnn.benchmark, tf32-high, channels_last_3d (UNet), fused AdamW, foreach EMA")
    ap.add_argument("--compile", action="store_true", help="torch.compile per block (regional); test on sm_120 first")
    ap.add_argument("--timing_breakdown", action="store_true",
                    help="log low-overhead data/prep/forward/loss/backward/optimizer/EMA timings every 50 updates")
    ap.add_argument("--shift", type=float, default=1.0, help="timestep shift (>1 = more noise; try 3 at 128/16f)")
    ap.add_argument("--img_frac", type=float, default=0.0, help="fraction of batches that are single frames (T=1 image warm-up mix)")
    ap.add_argument("--i2v_frac", type=float, default=0.0, help="fraction of video batches conditioned on the clean first frame (Seedance I2V, channel-concat + mask); enables cond channels")
    ap.add_argument("--noise_corr", type=float, default=0.0, help="PYoCo mixed noise: eps = sqrt(1-b)*shared + sqrt(b)*per-frame; 0 = iid")
    ap.add_argument("--fg_weight", type=float, default=1.0, help="soft-alpha foreground loss multiplier; 1 preserves the baseline")
    ap.add_argument("--rgba_aux_loss", type=float, default=0.0,
                    help="weight of Mini-Wan-matched x0 RGBA disagreement loss (pixel mode only)")
    ap.add_argument("--local_3d", action="store_true", help="add a zero-initialised local 3x3x3 token mixer to every DiT block")
    ap.add_argument("--full_st", action="store_true", help="replace factorised spatial/temporal attention with joint attention over all T*N tokens in every block")
    ap.add_argument("--conv_stem", action="store_true", help="learned f8t4 conv stem/head instead of linear patchify: VAE-style contextual mixing, end-to-end, no codec")
    a = ap.parse_args()
    early_sample_steps = parse_step_set(a.early_sample_steps)
    if a.preset:
        for k, v in PRESETS[a.preset].items():
            if k != "ch": setattr(a, k, v)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    if a.fast:
        torch.backends.cudnn.benchmark = True; torch.set_float32_matmul_precision("high")
    # Conv3d gradients can carry non-standard layouts under AMP. PyTorch's
    # fused AdamW requires every optimizer tensor to share an identical
    # layout, so use the regular CUDA implementation for the local-3D arm.
    a.fused_adamw = bool(a.fast and not a.local_3d)
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    dev = "cuda"
    ds = VideoWindows(a.cache, a.frames, "train", a.stride, size=a.size, return_text=a.cond == "text",
                      first_frames=a.first_frames)
    latent_mode = ds.latent
    if latent_mode and a.rgba_aux_loss > 0:
        raise ValueError("--rgba_aux_loss is for direct-RGBA models; latent models must decode before applying it")
    in_ch = int(json.load(open(os.path.join(a.cache, "meta.json")))["channels"]) if latent_mode else 4
    if latent_mode and (a.i2v_frac > 0 or a.noise_corr > 0):
        raise ValueError("latent caches do not support --i2v_frac or --noise_corr")
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True,
                                     persistent_workers=True, worker_init_fn=worker_init)
    vds = VideoWindows(a.cache, a.frames, "val", a.stride, size=a.size, return_text=a.cond == "text", deterministic=True, repeats=1,
                       first_frames=a.first_frames)
    vdl = torch.utils.data.DataLoader(vds, batch_size=a.batch, shuffle=True, num_workers=2, drop_last=True, worker_init_fn=worker_init) if len(vds) else None
    n_cls = len(ds.groups) if a.cond == "group" else 0
    text_cache = {}; text_dim = 0
    if a.cond == "text":
        from transformers import AutoTokenizer, T5EncoderModel
        tok = AutoTokenizer.from_pretrained(a.text_encoder)
        enc = T5EncoderModel.from_pretrained(a.text_encoder).to(dev).eval().requires_grad_(False)
        prompts = sorted({c["text"] for c in ds.clips + vds.clips} | {""})
        for start in range(0, len(prompts), 32):
            batch_prompts = prompts[start:start + 32]
            z = tok(batch_prompts, padding="max_length", truncation=True, max_length=a.text_len, return_tensors="pt")
            mask = z.attention_mask
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = enc(input_ids=z.input_ids.to(dev), attention_mask=mask.to(dev)).last_hidden_state.float().cpu()
            for prompt, h_, m_ in zip(batch_prompts, hidden, mask): text_cache[prompt] = (h_, m_)
        text_dim = enc.config.d_model
        del enc; torch.cuda.empty_cache()
        print(f"cached {len(text_cache) - 1} prompts with {a.text_encoder} ({text_dim}d, frozen)", flush=True)

    def text_batch(prompts):
        h, m = zip(*(text_cache[p] for p in prompts))
        return torch.stack(h).to(dev, non_blocking=True), torch.stack(m).to(dev, non_blocking=True)
    a.t1_skip = True                                          # recorded in ckpt args; see Block.t1_skip
    if a.arch == "dit":
        model = VideoDiT(size=a.size, frames=a.frames, patch=a.patch, in_ch=in_ch, dim=a.dim, depth=a.depth, heads=a.heads, n_classes=n_cls,
                         cond_ch=5 if a.i2v_frac > 0 else 0, text_dim=text_dim, local_3d=a.local_3d,
                         full_st=a.full_st, patch_t=a.patch_t, conv_stem=a.conv_stem).to(dev)
    else:
        model = UNet3D(ch=a.ch, n_classes=n_cls, size=a.size, cond_ch=5 if a.i2v_frac > 0 else 0,
                       text_dim=text_dim).to(dev)
    model.grad_ckpt = a.grad_ckpt
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if a.compile:   # compile forward only -> module identity / state_dict keys unchanged
        torch._dynamo.config.cache_size_limit = 64          # train batch / sample chunk / val batch shapes
        if a.arch == "dit":
            for b in model.blocks: b.forward = torch.compile(b.forward, dynamic=False)
        else:
            raise ValueError("--compile is currently supported only for --arch dit")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    model_name = "DiT-FM" if a.arch == "dit" else "ResUNet-FM"
    detail = (("convstem-f8t4" if a.conv_stem else f"p{a.patch}" + (f"+pt{a.patch_t}" if a.patch_t > 1 else "")) + ("+local3d" if a.local_3d else "") + ("+fullst" if a.full_st else "")) if a.arch == "dit" else f"ch{a.ch}"
    print(f"{model_name} params {n_params:.1f}M · {len(ds.clips)} train / {len(vds.clips)} val clips · {a.size}px {detail} · frames {a.frames} · batch {a.batch}×{a.accum} · ckpt {a.grad_ckpt} · shift {a.shift}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, a.optimizer_beta2), weight_decay=0.01, fused=a.fused_adamw)
    step = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev); model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    elif a.init:
        ck = torch.load(a.init, map_location=dev)
        for tgt, key in ((model, "model"), (ema, "ema")):
            src = ck.get(key) or ck["ema"]; own = tgt.state_dict()
            src = prepare_warmstart_state(src, own)
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
            for i, (x, labels) in enumerate(vdl):
                if i >= 16: break
                x = x.to(dev)
                if latent_mode: x = x[:, :-1]
                t = sample_t(x.shape[0], dev, a.shift); eps = torch.randn_like(x)
                tt = t[:, None, None, None, None]; yy = labels.to(dev) if n_cls else None
                txt, txt_mask = text_batch(labels) if a.cond == "text" else (None, None)
                with torch.autocast("cuda", dtype=torch.bfloat16): pred = ema((1 - tt) * x + tt * eps, t, yy, None, txt, txt_mask)
                tot += F.mse_loss(pred.float(), eps - x).item(); n += 1
        return tot / max(n, 1)

    if step == 0 and 0 in early_sample_steps and not latent_mode:
        preview_n = 4
        preview_ext = "png" if a.frames == 1 else "gif"
        generator = torch.Generator(device=dev).manual_seed(1234)
        preview_noise = mixed_noise((preview_n, 4, a.frames, a.size, a.size), dev, a.noise_corr, generator).cpu()
        to_gif(preview_noise, os.path.join(a.out, "sample_000000.gif"))
        torch.save(dict(ema=ema.state_dict(), step=0, args=vars(a), groups=ds.groups,
                        arch=("dit_fm_t2v" if a.cond == "text" else "dit_fm") if a.arch == "dit" else ("resunet_fm_t2v" if a.cond == "text" else "resunet_fm")),
                   os.path.join(a.out, "ckpt_000000.pt"))
        json.dump(dict(step=0, kind="zero-output initialization (noise is unchanged)", seed=1234,
                       prompts=diverse_text_prompts(vds, preview_n) if a.cond == "text" else None,
                       output=f"sample_000000.{preview_ext}"),
                  open(os.path.join(a.out, "sample_manifest_000000.json"), "w"), indent=2)
        print("  wrote sample_000000.gif", flush=True)

    t0 = time.time(); step0 = step; micro_step = 0; it = iter(dl); ema_loss = None
    timing_records = []
    timing_data_ms = 0.0
    timing_updates = 0
    while step < a.steps:
        data_started = time.perf_counter()
        try: x, labels = next(it)
        except StopIteration: it = iter(dl); x, labels = next(it)
        if a.timing_breakdown:
            timing_data_ms += (time.perf_counter() - data_started) * 1000.0
            timing_marks = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
            timing_marks[0].record()
        x = x.to(dev, non_blocking=True)
        fg_map = None
        if latent_mode:
            fg_map = x[:, -1:]; x = x[:, :-1]
        if a.img_frac > 0 and random.random() < a.img_frac:      # image warm-up mix: one temporal patch worth of frames
            span = 4 if a.conv_stem else max(1, a.patch_t)
            fi = random.randrange(x.shape[2] - span + 1); x = x[:, :, fi:fi + span]
            if fg_map is not None: fg_map = fg_map[:, :, fi:fi + span]
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
        cond = None
        if a.i2v_frac > 0:
            cond = i2v_cond(x) if (x.shape[2] > 1 and random.random() < a.i2v_frac) else torch.zeros(x.shape[0], 5, *x.shape[2:], device=dev)
        t = sample_t(x.shape[0], dev, a.shift); tt = t[:, None, None, None, None]
        eps = torch.randn_like(x)
        if a.noise_corr > 0 and x.shape[2] > 1:      # PYoCo mixed noise (valid: still Gaussian, forward process unchanged)
            shared = torch.randn_like(x[:, :, :1]).expand_as(x)
            eps = math.sqrt(1 - a.noise_corr) * shared + math.sqrt(a.noise_corr) * eps
        xt = (1 - tt) * x + tt * eps
        schedule_step = step + 1
        cosine_step = schedule_step if a.optimizer_steps else step
        prog = max(0.0, (cosine_step - 1000) / max(1, a.steps - 1000))
        lr = a.lr * min(1.0, schedule_step / 1000) * (a.lr_final + (1 - a.lr_final) * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups: g["lr"] = lr
        if a.timing_breakdown: timing_marks[1].record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(xt, t, y, cond, text, text_mask)
        if a.timing_breakdown: timing_marks[2].record()
        err = (pred.float() - (eps - x)) ** 2
        loss = (latent_weighted_mse(err, fg_map, a.fg_weight) if latent_mode
                else foreground_weighted_mse(err, x, a.fg_weight)) / a.accum
        rgba_aux = None
        if a.rgba_aux_loss > 0:
            x0_hat = xt - tt * pred.float()
            rgba_aux = a.rgba_aux_loss * rgba_x0_disagreement(x0_hat, x, t) / a.accum
            loss = loss + rgba_aux
        with torch.no_grad():                                   # diagnostics only
            fgm = (fg_map > 0.5).float() if latent_mode else (x[:, 3:4] > -0.9).float()
            fg_loss = float((err * fgm).sum() / fgm.sum().clamp_min(1))
            per = err.flatten(1).mean(1)
            lb = float(per[t > 0.8].mean()) if (t > 0.8).any() else float("nan")   # near noise
            hb = float(per[t < 0.2].mean()) if (t < 0.2).any() else float("nan")   # near data
        if a.timing_breakdown: timing_marks[3].record()
        loss.backward()
        if a.timing_breakdown: timing_marks[4].record()
        if a.optimizer_steps:
            if not optimizer_step_due(micro_step, a.accum):
                if a.timing_breakdown: timing_records.append(timing_marks[:5])
                micro_step += 1
                continue
            micro_step += 1
        elif (step + 1) % a.accum != 0:
            if a.timing_breakdown: timing_records.append(timing_marks[:5])
            step += 1
            continue
        if a.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step(); opt.zero_grad(set_to_none=True)
        if a.timing_breakdown: timing_marks[5].record()
        loss = loss * a.accum
        if a.optimizer_steps:
            step += 1
        with torch.no_grad():
            if a.ema_max > 0:
                d = min(a.ema_max, (1 + step) / (10 + step))
            else:
                d = min(0.999 if step < 5000 else 0.9995, (1 + step) / (10 + step))
            if a.fast: torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - d)
            else:
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - d)
        if a.timing_breakdown:
            timing_marks[6].record()
            timing_records.append(timing_marks)
            timing_updates += 1
        if not a.optimizer_steps:
            step += 1
        ema_loss = loss.item() if ema_loss is None else 0.98 * ema_loss + 0.02 * loss.item()
        if step == 10 or step % 50 == 0:
            spi = (time.time() - t0) / max(1, step - step0)
            msg = f"step {step} loss {ema_loss:.4f} fg {fg_loss:.4f} t>.8 {lb:.4f} t<.2 {hb:.4f} lr {lr:.2e} {spi:.2f}s/it peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB ETA {(a.steps-step)*spi/3600:.1f}h"
            timing_values = None
            if a.timing_breakdown and timing_updates:
                torch.cuda.synchronize()
                sums = dict(prep=0.0, fwd=0.0, loss=0.0, bwd=0.0, opt=0.0, ema=0.0)
                for marks in timing_records:
                    sums["prep"] += marks[0].elapsed_time(marks[1])
                    sums["fwd"] += marks[1].elapsed_time(marks[2])
                    sums["loss"] += marks[2].elapsed_time(marks[3])
                    sums["bwd"] += marks[3].elapsed_time(marks[4])
                    if len(marks) == 7:
                        sums["opt"] += marks[4].elapsed_time(marks[5])
                        sums["ema"] += marks[5].elapsed_time(marks[6])
                timing_values = {k: v / timing_updates for k, v in sums.items()}
                timing_values["data"] = timing_data_ms / timing_updates
                msg += (" timing_ms " + " ".join(
                    f"{key}={timing_values[key]:.1f}"
                    for key in ("data", "prep", "fwd", "loss", "bwd", "opt", "ema")
                ))
                timing_records.clear(); timing_data_ms = 0.0; timing_updates = 0
            if rgba_aux is not None: msg += f" rgba_aux {float(rgba_aux * a.accum):.4f}"
            vl = vloss() if (vdl is not None and step % a.val_every == 0) else None
            if vl is not None: msg += f" val {vl:.4f}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            if tb:
                tb.add_scalar("loss/train_ema", ema_loss, step); tb.add_scalar("lr", lr, step)
                tb.add_scalar("loss/fg", fg_loss, step); tb.add_scalar("loss/t_gt_0.8", lb, step); tb.add_scalar("loss/t_lt_0.2", hb, step)
                tb.add_scalar("perf/s_per_it", spi, step); tb.add_scalar("perf/peak_gb", torch.cuda.max_memory_allocated() / 1e9, step)
                if timing_values is not None:
                    for key, value in timing_values.items(): tb.add_scalar(f"timing_ms/{key}", value, step)
                if vl is not None: tb.add_scalar("loss/val", vl, step)
        sample_due = step in early_sample_steps or step % a.sample_every == 0 or step == a.steps
        if a.checkpoint_every > 0 and step % a.checkpoint_every == 0 and not sample_due:
            arch_tag = ((('dit_fm_fullst_t2v' if a.cond == 'text' else 'dit_fm_fullst') if a.full_st
                         else ('dit_fm_local3d_t2v' if a.cond == 'text' else 'dit_fm_local3d') if a.local_3d
                         else ('dit_fm_t2v' if a.cond == 'text' else 'dit_fm')) if a.arch == 'dit'
                        else ('resunet_fm_t2v' if a.cond == 'text' else 'resunet_fm'))
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step,
                            args=vars(a), groups=ds.groups, arch=arch_tag),
                       os.path.join(a.out, "ckpt.pt"))
            print(f"  wrote rolling ckpt.pt at step {step}", flush=True)
        if sample_due and not latent_mode:
            early_preview = step in early_sample_steps and step % a.sample_every != 0 and step != a.steps
            NS, CH = ((4, 4) if early_preview else ((64, 32) if a.frames == 1 else ((4, 1) if a.frames >= 32 else (16, 8))))
            sample_nfe = 20 if early_preview else 50
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            ys = torch.arange(NS, device=dev) % n_cls if n_cls else None
            sample_prompts = diverse_text_prompts(vds, NS) if a.cond == "text" else None
            sample_text, sample_text_mask = text_batch(sample_prompts) if sample_prompts else (None, None)
            sample_null, sample_null_mask = text_batch([""] * NS) if sample_prompts else (None, None)
            g = torch.Generator(device=dev).manual_seed(1234)
            noise = mixed_noise((NS, 4, a.frames, a.size, a.size), dev, a.noise_corr, g)
            scond = None
            if a.i2v_frac > 0 and a.frames > 1 and vds is not None:                  # rows 0-1: T2V, rows 2-3: I2V from fixed val first frames
                vx = torch.stack([vds[int(i)][0] for i in np.random.RandomState(7).permutation(len(vds))[:NS // 2]]).to(dev)
                scond = torch.cat([torch.zeros(NS - NS // 2, 5, *vx.shape[2:], device=dev), i2v_cond(vx)], 0)
            for name, m_ in (("", ema), ("raw_", model)):
                if name and step > 10000: continue
                m_.eval()
                outs = []
                for i in range(0, NS, CH):
                    yy = ys[i:i + CH] if ys is not None else None
                    outs.append(euler_sample(m_, noise[i:i + CH].shape, dev, steps=sample_nfe, y=yy,
                                             null_y=torch.full((CH,), n_cls, device=dev) if n_cls else None, noise=noise[i:i + CH], shift=a.shift,
                                             cond=scond[i:i + CH] if scond is not None else None,
                                             text=sample_text[i:i + CH] if sample_text is not None else None,
                                             text_mask=sample_text_mask[i:i + CH] if sample_text_mask is not None else None,
                                             null_text=sample_null[i:i + CH] if sample_null is not None else None,
                                             null_text_mask=sample_null_mask[i:i + CH] if sample_null_mask is not None else None,
                                             cfg=3.0 if sample_text is not None else (2.0 if n_cls else 0.0)).cpu())
                xs = torch.cat(outs, 0)
                m_.train() if m_ is model else None
                to_gif(xs, os.path.join(a.out, f"sample_{name}{step:06d}.gif"))
                if not name and tb:
                    v = ((xs.clamp(-1, 1) + 1) / 2); rgb = (v[:, :3] + (1 - v[:, 3:4])).clamp(0, 1)
                    try: tb.add_video("samples", rgb.permute(0, 2, 1, 3, 4).cpu(), step, fps=10)
                    except Exception: pass
                torch.cuda.empty_cache()
            sample_ext = "png" if a.frames == 1 else "gif"
            json.dump(dict(step=step, seed=1234, prompts=sample_prompts, sampler="euler",
                           nfe=sample_nfe, cfg=3.0 if sample_text is not None else (2.0 if n_cls else 0.0),
                           shift=a.shift, raw_and_ema=True,
                           outputs=[f"sample_{step:06d}.{sample_ext}", f"sample_raw_{step:06d}.{sample_ext}"]),
                      open(os.path.join(a.out, f"sample_manifest_{step:06d}.json"), "w"), indent=2)
            arch_tag = ((("dit_fm_fullst_t2v" if a.cond == "text" else "dit_fm_fullst") if a.full_st
                         else ("dit_fm_local3d_t2v" if a.cond == "text" else "dit_fm_local3d") if a.local_3d
                         else ("dit_fm_t2v" if a.cond == "text" else "dit_fm")) if a.arch == "dit"
                        else ("resunet_fm_t2v" if a.cond == "text" else "resunet_fm"))
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, args=vars(a), groups=ds.groups, arch=arch_tag),
                       os.path.join(a.out, "ckpt.pt"))
            torch.save(dict(ema=ema.state_dict(), step=step, args=vars(a), groups=ds.groups, arch=arch_tag),
                       os.path.join(a.out, f"ckpt_{step:06d}.pt"))
            print(f"  wrote sample_{step:06d}.gif", flush=True)
        elif latent_mode and sample_due:
            # Latent runs skip pixel previews (decode happens in offline evaluation) but still checkpoint.
            arch_tag = (("dit_fm_fullst_t2v_latent" if a.full_st else "dit_fm_t2v_latent") if a.cond == "text"
                        else ("dit_fm_fullst_latent" if a.full_st else "dit_fm_latent"))
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, args=vars(a), groups=ds.groups, arch=arch_tag),
                       os.path.join(a.out, "ckpt.pt"))
            torch.save(dict(ema=ema.state_dict(), step=step, args=vars(a), groups=ds.groups, arch=arch_tag),
                       os.path.join(a.out, f"ckpt_{step:06d}.pt"))
            print(f"  wrote ckpt_{step:06d}.pt", flush=True)


if __name__ == "__main__":
    main()
