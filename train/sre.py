"""SRE — Skeleton Regression Evaluator: single RGBA frame -> cskel27 2D joints.

    python -m train.sre --cache cache/mini --out results/sre_v1 --steps 20000

Design (declared in paper/refs/sre_design.md before implementation): conv encoder
(4 stages, 32->256, stride 2) -> 4x4 map -> MLP -> 54 sigmoid outputs (27 joints x 2,
normalized [0,1]). ~2.5M params. Loss: L2 over VISIBLE joints only (both coords inside
[0,1]); off-screen joints are masked, not clamped targets. Input: premultiplied RGBA
in [0,1] at cache resolution. Splits follow clips.json (prompt-disjoint), same
drop_flags discipline as VideoWindows. Validation gates live in eval/sre_validate.py.
"""
from __future__ import annotations
import argparse, json, math, os, random, subprocess, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

RIG_JOINTS = 27


# ----------------------------------------------------------------- data
class RigFrames(torch.utils.data.Dataset):
    """Every individual frame of every clip in the split, with its rig labels."""

    def __init__(self, cache, split="train", drop_flags=("levitation",)):
        self.frames = np.load(os.path.join(cache, "frames.npy"), mmap_mode="r")
        self.rig = np.load(os.path.join(cache, "rig.npy"), mmap_mode="r")
        if self.rig.shape[0] != self.frames.shape[0]:
            raise ValueError("rig.npy is not aligned with frames.npy")
        clips = json.load(open(os.path.join(cache, "clips.json")))
        self.index = np.concatenate([
            np.arange(c["start"], c["start"] + c["n"])
            for c in clips.values()
            if c["split"] == split and not any(f in (c.get("qa") or "") for f in drop_flags)
        ]) if any(c["split"] == split for c in clips.values()) else np.zeros(0, np.int64)
        self.size = self.frames.shape[1]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j = int(self.index[i])
        x = np.asarray(self.frames[j]).astype(np.float32) / 255.0        # [H,W,4] straight
        a = x[..., 3:4]
        x = np.concatenate([x[..., :3] * a, a], -1)                      # premultiply, keep [0,1]
        rig = np.asarray(self.rig[j]).astype(np.float32)                 # [27,2] normalized
        visible = ((rig >= 0.0) & (rig <= 1.0)).all(-1)                  # both coords on-frame
        return (torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(rig),
                torch.from_numpy(visible))


# ----------------------------------------------------------------- model
class SRE(nn.Module):
    def __init__(self, size=64, widths=(32, 64, 128, 256), hidden=512):
        super().__init__()
        stages, cin = [], 4
        for w in widths:
            stages += [nn.Conv2d(cin, w, 3, stride=2, padding=1), nn.GroupNorm(8, w), nn.SiLU(),
                       nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU()]
            cin = w
        self.encoder = nn.Sequential(*stages)
        feat = widths[-1] * (size // 2 ** len(widths)) ** 2
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(feat, hidden), nn.SiLU(),
                                  nn.Linear(hidden, RIG_JOINTS * 2))

    def forward(self, x):                                                # [B,4,H,W] -> [B,27,2]
        return torch.sigmoid(self.head(self.encoder(x))).view(-1, RIG_JOINTS, 2)


def masked_joint_l2(pred, target, visible):
    """Mean squared error over visible joints' coords; zero-visible batches contribute 0."""
    err = (pred - target).square().sum(-1)                               # [B,27]
    m = visible.float()
    return (err * m).sum() / m.sum().clamp_min(1.0)


# ----------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, loader, device, size):
    """Mean joint error (px at cache resolution) + PCK@2px/@4px over visible joints."""
    model.eval()
    dist_sum = n = pck2 = pck4 = 0.0
    for x, rig, visible in loader:
        pred = model(x.to(device))
        d = (pred - rig.to(device)).norm(dim=-1) * size                  # [B,27] px
        m = visible.to(device).float()
        dist_sum += (d * m).sum().item(); n += m.sum().item()
        pck2 += ((d <= 2.0) * m).sum().item(); pck4 += ((d <= 4.0) * m).sum().item()
    model.train()
    n = max(n, 1.0)
    return dist_sum / n, pck2 / n, pck4 / n


# ----------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-final", type=float, default=0.05, help="final lr as a fraction of --lr")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--val-batches", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=512)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    train_ds = RigFrames(args.cache, "train")
    val_ds = RigFrames(args.cache, "val")
    size = train_ds.size
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=(device == "cuda"), drop_last=True, persistent_workers=args.workers > 0)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_ds, list(range(0, len(val_ds),
                                max(1, len(val_ds) // (args.val_batches * args.batch))))),
        batch_size=args.batch, shuffle=False, num_workers=2)

    model = SRE(size=size, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())

    try:
        git_rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
    except Exception:
        git_rev = "unknown"
    manifest = {"protocol": "sre_v1", "design": "paper/refs/sre_design.md",
                "model_parameters": n_params, "cache": args.cache, "cache_size": size,
                "train_frames": len(train_ds), "val_frames": len(val_ds),
                "git_rev": git_rev, "args": vars(args)}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"SRE {n_params / 1e6:.2f}M; {len(train_ds)} train / {len(val_ds)} val frames "
          f"at {size}x{size}; device {device}", flush=True)

    log = (out / "log.txt").open("a")
    metrics, it, step, t0 = [], iter(loader), 0, time.time()
    amp = torch.autocast("cuda", torch.bfloat16) if device == "cuda" else None
    while step < args.steps:
        try:
            x, rig, visible = next(it)
        except StopIteration:
            it = iter(loader); continue
        step += 1
        lr = args.lr * (step / args.warmup if step < args.warmup else
                        args.lr_final + (1 - args.lr_final) * 0.5 *
                        (1 + math.cos(math.pi * (step - args.warmup) / max(1, args.steps - args.warmup))))
        for g in opt.param_groups:
            g["lr"] = lr
        x, rig, visible = x.to(device), rig.to(device), visible.to(device)
        if amp:
            with amp:
                loss = masked_joint_l2(model(x), rig, visible)
        else:
            loss = masked_joint_l2(model(x), rig, visible)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        if step % 50 == 0:
            rate = 50 / (time.time() - t0); t0 = time.time()
            line = (f"step {step} loss {loss.item():.6f} lr {lr:.2e} "
                    f"{rate:.1f}it/s ETA {(args.steps - step) / rate / 3600:.2f}h")
            print(line, flush=True); log.write(line + "\n"); log.flush()
        if step % args.val_every == 0 or step == args.steps:
            px, p2, p4 = evaluate(model, val_loader, device, size)
            metrics.append({"step": step, "val_px": px, "pck2": p2, "pck4": p4})
            line = f"step {step} VAL px {px:.3f} PCK@2 {p2:.4f} PCK@4 {p4:.4f}"
            print(line, flush=True); log.write(line + "\n"); log.flush()
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "step": step, "size": size,
                        "hidden": args.hidden, "manifest": manifest}, out / "latest.pt")
    torch.save({"model": model.state_dict(), "step": step, "size": size,
                "hidden": args.hidden, "manifest": manifest}, out / "ckpt_final.pt")
    print(f"done: {out / 'ckpt_final.pt'}", flush=True)


if __name__ == "__main__":
    main()
