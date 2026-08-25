"""Confidence-aware SRE: one RGBA frame -> 27 2D joints + uncertainty.

The coordinate mean is paired with an isotropic Gaussian scale per joint.
After training, a held-out validation temperature calibrates that scale.  The
reported confidence is the calibrated probability that a joint lies within
the renderer's 1.6-pixel limb radius; it is coordinate confidence, not a
guarantee that an out-of-distribution generated frame is anatomically valid.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train.sre import RIG_JOINTS, RigFrames


class SREConfidence(nn.Module):
    def __init__(self, size=64, widths=(32, 64, 128, 256), hidden=512,
                 initial_sigma_px=1.5):
        super().__init__()
        stages, channels = [], 4
        for width in widths:
            stages += [
                nn.Conv2d(channels, width, 3, stride=2, padding=1),
                nn.GroupNorm(8, width), nn.SiLU(),
                nn.Conv2d(width, width, 3, padding=1),
                nn.GroupNorm(8, width), nn.SiLU(),
            ]
            channels = width
        self.encoder = nn.Sequential(*stages)
        features = widths[-1] * (size // 2 ** len(widths)) ** 2
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(features, hidden), nn.SiLU())
        self.coordinate_head = nn.Linear(hidden, RIG_JOINTS * 2)
        self.log_sigma_head = nn.Linear(hidden, RIG_JOINTS)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.constant_(self.log_sigma_head.bias, math.log(initial_sigma_px / size))
        self.size = int(size)

    def forward(self, value):
        features = self.trunk(self.encoder(value))
        joints = torch.sigmoid(self.coordinate_head(features)).view(-1, RIG_JOINTS, 2)
        low = math.log(0.05 / self.size)
        high = math.log(16.0 / self.size)
        log_sigma = self.log_sigma_head(features).clamp(low, high)
        return joints, log_sigma


def gaussian_joint_nll(pred, log_sigma, target, visible):
    """Isotropic 2D Gaussian NLL over on-frame joints."""
    squared = (pred - target).square().sum(-1)
    nll = 0.5 * squared * torch.exp(-2.0 * log_sigma) + 2.0 * log_sigma
    mask = visible.float()
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)


def warm_start_from_v1(model: SREConfidence, checkpoint: dict) -> None:
    """Copy the v1 encoder, hidden layer, and coordinate head exactly."""
    state = checkpoint["model"]
    model.encoder.load_state_dict({key.removeprefix("encoder."): value
                                   for key, value in state.items() if key.startswith("encoder.")})
    model.trunk[1].load_state_dict({
        "weight": state["head.1.weight"], "bias": state["head.1.bias"],
    })
    model.coordinate_head.load_state_dict({
        "weight": state["head.3.weight"], "bias": state["head.3.bias"],
    })


def _ece(probability: np.ndarray, outcome: np.ndarray, bins=10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(probability), 1)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        selected = (probability >= left) & (
            probability <= right if right == 1.0 else probability < right
        )
        if selected.any():
            value += selected.sum() / total * abs(probability[selected].mean() - outcome[selected].mean())
    return float(value)


@torch.no_grad()
def evaluate_confidence(model, loader, device, size, limb_radius_px=1.6):
    model.eval()
    distances, sigmas, joint_ids = [], [], []
    for value, target, visible in loader:
        mean, log_sigma = model(value.to(device))
        distance = (mean - target.to(device)).norm(dim=-1) * size
        sigma = log_sigma.exp() * size
        mask = visible.to(device)
        indices = torch.arange(RIG_JOINTS, device=device)[None].expand_as(mask)
        distances.append(distance[mask].cpu().numpy())
        sigmas.append(sigma[mask].cpu().numpy())
        joint_ids.append(indices[mask].cpu().numpy())
    distance = np.concatenate(distances)
    raw_sigma = np.maximum(np.concatenate(sigmas), 1e-6)
    ids = np.concatenate(joint_ids)
    temperature = float(np.sqrt(np.mean((distance / raw_sigma) ** 2) / 2.0))
    sigma = raw_sigma * max(temperature, 1e-6)
    probability = 1.0 - np.exp(-(limb_radius_px ** 2) / (2.0 * sigma ** 2))
    outcome = distance <= limb_radius_px
    per_joint = {}
    for joint in range(RIG_JOINTS):
        selected = ids == joint
        per_joint[str(joint)] = {
            "mean_error_px": float(distance[selected].mean()),
            "mean_sigma_px": float(sigma[selected].mean()),
            "mean_confidence_within_limb": float(probability[selected].mean()),
            "empirical_within_limb": float(outcome[selected].mean()),
        }
    report = {
        "mean_px": float(distance.mean()),
        "pck2": float((distance <= 2.0).mean()),
        "pck4": float((distance <= 4.0).mean()),
        "temperature": temperature,
        "mean_sigma_px": float(sigma.mean()),
        "coverage_68": float((distance <= 1.51 * sigma).mean()),
        "coverage_95": float((distance <= 2.45 * sigma).mean()),
        "confidence_radius_px": limb_radius_px,
        "mean_confidence_within_limb": float(probability.mean()),
        "empirical_within_limb": float(outcome.mean()),
        "brier_within_limb": float(np.mean((probability - outcome) ** 2)),
        "ece_within_limb": _ece(probability, outcome),
        "per_joint": per_joint,
    }
    model.train()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init-v1", default="")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-final", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=512)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    train_data = RigFrames(args.cache, "train")
    validation_data = RigFrames(args.cache, "val")
    size = train_data.size
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=device == "cuda", drop_last=True, persistent_workers=args.workers > 0,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=args.batch, shuffle=False, num_workers=2,
    )
    model = SREConfidence(size=size, hidden=args.hidden).to(device)
    if args.init_v1:
        initial = torch.load(args.init_v1, map_location="cpu", weights_only=False)
        warm_start_from_v1(model, initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
    except Exception:
        revision = "unknown"
    manifest = {
        "protocol": "sre_confidence_v1", "coordinate_base": "SRE v1",
        "confidence": "temperature-calibrated isotropic Gaussian scale per joint",
        "claim_limit": "held-out in-domain coordinate confidence; not certified OOD confidence",
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "cache": args.cache, "cache_size": size, "train_frames": len(train_data),
        "val_frames": len(validation_data), "git_rev": revision, "args": vars(args),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    scaler_context = torch.autocast("cuda", torch.bfloat16) if device == "cuda" else None
    iterator, step, started = iter(train_loader), 0, time.time()
    log = (output / "log.txt").open("a")
    while step < args.steps:
        try:
            value, target, visible = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            continue
        step += 1
        progress = max(0.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        learning_rate = args.lr * min(1.0, step / args.warmup) * (
            args.lr_final + (1.0 - args.lr_final) * 0.5 * (1.0 + math.cos(math.pi * progress))
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        value, target, visible = value.to(device), target.to(device), visible.to(device)
        if scaler_context:
            with scaler_context:
                mean, log_sigma = model(value)
                loss = gaussian_joint_nll(mean, log_sigma, target, visible)
        else:
            mean, log_sigma = model(value)
            loss = gaussian_joint_nll(mean, log_sigma, target, visible)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step % 50 == 0:
            elapsed = time.time() - started
            rate = step / max(elapsed, 1e-6)
            line = (f"step {step} nll {loss.item():.5f} lr {learning_rate:.2e} "
                    f"{rate:.1f}it/s ETA {(args.steps-step)/rate/3600:.2f}h")
            print(line, flush=True); log.write(line + "\n"); log.flush()
        if step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "step": step, "size": size,
                        "hidden": args.hidden, "manifest": manifest}, output / "latest.pt")

    calibration = evaluate_confidence(model, validation_loader, device, size)
    (output / "validation_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    checkpoint = {"model": model.state_dict(), "step": step, "size": size,
                  "hidden": args.hidden, "temperature": calibration["temperature"],
                  "manifest": manifest}
    torch.save(checkpoint, output / "ckpt_final.pt")
    print(json.dumps({key: value for key, value in calibration.items() if key != "per_joint"}, indent=2))


if __name__ == "__main__":
    main()
