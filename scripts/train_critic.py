"""Corruption-supervised video quality critic (v0) -- auxiliary ladder statistic.

Trains a small 3D CNN to regress corruption severity on first-64 windows of the
curated cache: real clips are severity 0, corrupted copies carry the strength of
the corruption applied. reverse_time and loop_first_8 are NEVER trained on --
they are held-out families that measure whether the critic generalises beyond
the corruptions it was taught. Model samples are never seen during training, so
the frozen critic can serve as an evaluation-only auxiliary score.

    PYTHONPATH=. python3 scripts/train_critic.py \
        --cache cache/mini_v02 --out results/critic_v0 --steps 6000

Writes ckpt_final.pt and validation.json (per-family mean scores on the val
split, trained and held-out families separately).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TRAINED_FAMILIES = ("freeze_tail", "window_shuffle", "frame_jitter", "warp_static", "warp_flicker")
HELDOUT_FAMILIES = ("reverse_time", "loop_first_8")


def smooth_field(rng, size, cells=4, amp=1.0):
    """Smooth per-pixel displacement field, amplitude in pixels."""
    coarse = rng.standard_normal((2, cells, cells)).astype(np.float32) * amp
    field = torch.from_numpy(coarse)[None]
    return F.interpolate(field, size=(size, size), mode="bicubic", align_corners=False)[0].numpy()


def warp_frames(clip, field):
    """clip [T,H,W,4] float, field [2,H,W] pixel offsets -> warped clip."""
    T, H, W, _ = clip.shape
    x = torch.from_numpy(clip).permute(0, 3, 1, 2)
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    gx = (xs + torch.from_numpy(field[0])) / (W - 1) * 2 - 1
    gy = (ys + torch.from_numpy(field[1])) / (H - 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)[None].expand(T, -1, -1, -1)
    return F.grid_sample(x, grid, align_corners=True, padding_mode="zeros").permute(0, 2, 3, 1).numpy()


def corrupt(clip, family, s, rng):
    """clip [T,H,W,4] float32 in [0,1]; s in (0,1] severity; returns corrupted copy."""
    T = clip.shape[0]
    out = clip.copy()
    if family == "freeze_tail":                      # freeze the last s of the clip
        k = max(1, int(round((1 - s) * (T - 1))))
        out[k:] = out[k]
    elif family == "window_shuffle":                 # shuffle within windows growing with s
        w = max(2, int(round(2 + s * (T - 2))))
        for a in range(0, T, w):
            idx = np.arange(a, min(a + w, T)); rng.shuffle(idx)
            out[a:min(a + w, T)] = clip[idx]
    elif family == "frame_jitter":                   # swap adjacent frames with prob s
        for t in range(0, T - 1, 2):
            if rng.random() < s:
                out[[t, t + 1]] = out[[t + 1, t]]
    elif family == "warp_static":                    # one smooth deformation held over time
        out = warp_frames(out, smooth_field(rng, clip.shape[1], amp=s * 6.0))
    elif family == "warp_flicker":                   # independent deformation per frame
        for t in range(T):
            out[t] = warp_frames(out[t:t + 1], smooth_field(rng, clip.shape[1], amp=s * 3.0))[0]
    elif family == "reverse_time":
        out = out[::-1].copy()
    elif family == "loop_first_8":
        out = np.concatenate([out[:8]] * (T // 8 + 1))[:T]
    else:
        raise ValueError(family)
    return out


class Critic(nn.Module):
    def __init__(self, ch=(32, 64, 128, 256)):
        super().__init__()
        layers, c_in = [], 4
        for c in ch:
            layers += [nn.Conv3d(c_in, c, 3, stride=2, padding=1), nn.GroupNorm(8, c), nn.SiLU()]
            c_in = c
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, 1)

    def forward(self, x):                            # x [B,4,T,H,W] premultiplied [0,1]
        h = self.body(x).mean(dim=(2, 3, 4))
        return torch.sigmoid(self.head(h)).squeeze(-1)


def load_clip(frames, meta, T):
    x = np.asarray(frames[meta["start"]:meta["start"] + T]).astype(np.float32) / 255.0
    x[..., :3] *= x[..., 3:4]                        # premultiply
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    frames = np.load(os.path.join(a.cache, "frames.npy"), mmap_mode="r")
    clips = json.load(open(os.path.join(a.cache, "clips.json")))
    train_ids = sorted(c for c, m in clips.items() if m["split"] == "train")
    val_ids = sorted(c for c, m in clips.items() if m["split"] == "val")

    model = Critic().to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    os.makedirs(a.out, exist_ok=True)

    def make_batch(ids):
        xs, ys = [], []
        for cid in rng.choice(ids, a.batch):
            clip = load_clip(frames, clips[cid], a.frames)
            if rng.random() < 0.5:                   # real: severity 0
                xs.append(clip); ys.append(0.0)
            else:
                s = float(rng.uniform(0.25, 1.0))
                fam = TRAINED_FAMILIES[rng.integers(len(TRAINED_FAMILIES))]
                xs.append(corrupt(clip, fam, s, rng)); ys.append(s)
        x = torch.from_numpy(np.stack(xs)).permute(0, 4, 1, 2, 3).to(a.device)
        return x, torch.tensor(ys, device=a.device)

    for step in range(1, a.steps + 1):
        x, y = make_batch(train_ids)
        loss = F.mse_loss(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)

    model.eval()
    report = {"trained": {}, "heldout": {}, "real": None}
    with torch.no_grad():
        def score_family(fam, s):
            vals = []
            for cid in val_ids:
                clip = load_clip(frames, clips[cid], a.frames)
                if fam is not None:
                    clip = corrupt(clip, fam, s, np.random.default_rng(hash(cid) % 2**32))
                x = torch.from_numpy(clip[None]).permute(0, 4, 1, 2, 3).to(a.device)
                vals.append(float(model(x)))
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        report["real"] = score_family(None, 0.0)
        for fam in TRAINED_FAMILIES:
            report["trained"][fam] = score_family(fam, 1.0)
        for fam in HELDOUT_FAMILIES:
            report["heldout"][fam] = score_family(fam, 1.0)

    real = report["real"]
    report["kill_criterion"] = {
        "rule": "drop the critic column if any TRAINED family mean <= real mean + 2*real std on val",
        "real_bar": real["mean"] + 2 * real["std"],
        "passed": all(v["mean"] > real["mean"] + 2 * real["std"] for v in report["trained"].values()),
    }
    torch.save({"model": model.state_dict(), "args": vars(a),
                "trained_families": TRAINED_FAMILIES, "heldout_families": HELDOUT_FAMILIES},
               os.path.join(a.out, "ckpt_final.pt"))
    json.dump(report, open(os.path.join(a.out, "validation.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
