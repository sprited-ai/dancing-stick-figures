"""SRE: single-frame pixels -> cskel27 2D rig regressor.

Design and pre-declared validation gates: paper/refs/sre_design.md.
Trains on the cache's exact rig labels (train split), validates on the test
split. Deliberately small (~2M params): an instrument, not a model.

Gates (declared before training):
  G1 held-out mean joint error < 1.6 px at 64^2 (capsule limb radius);
     report PCK@2px and PCK@4px.
  G2 corruption sanity: swapped-limb / extra-arm renders must raise error.
     (Scored by scripts/sre_corruption_check.py once G1 passes.)
  G3 off-screen joints (outside [-0.5, 1.5]) are excluded from the loss and
     the error report; they must not destabilize visible-joint predictions.
"""
import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class FrameRigSet(Dataset):
    """Single frames (premultiplied RGBA [0,1]) with normalized joint targets."""

    def __init__(self, cache: str, split: str):
        self.frames = np.load(os.path.join(cache, "frames.npy"), mmap_mode="r")
        self.rig = np.load(os.path.join(cache, "rig.npy"), mmap_mode="r")
        clips = json.loads(Path(cache, "clips.json").read_text())
        self.index = []
        for row in clips.values():
            if row["split"] != split:
                continue
            self.index.extend(range(row["start"], row["start"] + row["n"]))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j = self.index[i]
        x = np.asarray(self.frames[j], np.float32) / 255.0
        x = np.concatenate([x[..., :3] * x[..., 3:4], x[..., 3:4]], -1)
        rig = np.asarray(self.rig[j], np.float32)              # [27, 2] in [0,1]-ish
        visible = ((rig > -0.5) & (rig < 1.5)).all(-1)         # G3: off-screen mask
        return (torch.from_numpy(x).permute(2, 0, 1),
                torch.from_numpy(rig),
                torch.from_numpy(visible))


class SRE(nn.Module):
    def __init__(self, joints: int = 27):
        super().__init__()
        chans = [4, 32, 64, 128, 256]
        blocks = []
        for a, b in zip(chans[:-1], chans[1:]):
            blocks += [nn.Conv2d(a, b, 3, stride=2, padding=1), nn.GroupNorm(8, b), nn.SiLU()]
        self.encoder = nn.Sequential(*blocks)                  # 64 -> 4, 256 ch
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 4 * 4, 512), nn.SiLU(),
            nn.Linear(512, joints * 2),
        )
        self.joints = joints

    def forward(self, x):
        return self.head(self.encoder(x)).reshape(-1, self.joints, 2)


def masked_l2(pred, target, visible):
    err = (pred - target).square().sum(-1)                     # [B, J]
    mask = visible.float()
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model, loader, device, image_size):
    model.eval()
    errors, hits2, hits4, count = 0.0, 0, 0, 0
    for x, rig, visible in loader:
        pred = model(x.to(device)).cpu()
        dist_px = (pred - rig).norm(dim=-1) * image_size       # [B, J] pixels
        mask = visible
        errors += float(dist_px[mask].sum())
        hits2 += int((dist_px[mask] < 2.0).sum())
        hits4 += int((dist_px[mask] < 4.0).sum())
        count += int(mask.sum())
    model.train()
    return errors / count, hits2 / count, hits4 / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--val-frames", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    train_set = FrameRigSet(args.cache, "train")
    test_set = FrameRigSet(args.cache, "test")
    generator = torch.Generator().manual_seed(args.seed)
    val_index = torch.randperm(len(test_set), generator=generator)[: args.val_frames].tolist()
    val_set = torch.utils.data.Subset(test_set, val_index)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=2)

    model = SRE().to(args.device)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"train {len(train_set)} val {len(val_set)}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    step, history = 0, []
    while step < args.steps:
        for x, rig, visible in train_loader:
            loss = masked_l2(model(x.to(args.device)), rig.to(args.device), visible.to(args.device))
            optimizer.zero_grad(); loss.backward(); optimizer.step(); schedule.step()
            step += 1
            if step % 200 == 0:
                print(f"step {step} loss {float(loss):.5f} lr {schedule.get_last_lr()[0]:.2e}", flush=True)
            if step % args.val_every == 0 or step == args.steps:
                err, pck2, pck4 = evaluate(model, val_loader, args.device, args.image_size)
                history.append({"step": step, "mean_err_px": err, "pck2": pck2, "pck4": pck4})
                print(f"VAL step {step} mean_err {err:.3f}px PCK@2 {pck2:.3f} PCK@4 {pck4:.3f}", flush=True)
                torch.save({"model": model.state_dict(), "step": step, "history": history,
                            "gate_G1": {"mean_err_px": err, "pass": err < 1.6}},
                           out / f"sre_{step:06d}.pt")
            if step >= args.steps:
                break
    (out / "history.json").write_text(json.dumps(history, indent=2))
    final = history[-1]
    print(f"G1 {'PASS' if final['mean_err_px'] < 1.6 else 'FAIL'} "
          f"({final['mean_err_px']:.3f}px, gate < 1.6px)")


if __name__ == "__main__":
    main()
