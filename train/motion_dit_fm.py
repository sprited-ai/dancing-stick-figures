"""S0: a small text-conditioned flow-matching DiT for 50-frame cskel27 motion.

This is intentionally a minimal, auditable structured baseline rather than a
claim of a novel motion architecture.  It consumes caches built by
``train.motion_data`` and predicts hips-centred joints, root displacement and
relative root heading.  Foot contacts are an optional auxiliary prediction.

Example pilot::

    python -m train.motion_dit_fm --cache data/motion_cache --out runs/s0 \
        --steps 3000 --batch 32 --dim 256 --depth 6 --heads 4
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train.motion_data import MotionCache


MOTION_DIM = 27 * 3 + 3 + 2


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    phase = 1000.0 * t.float()[:, None] * freq[None]
    out = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
    return F.pad(out, (0, dim - out.shape[-1]))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class MotionBlock(nn.Module):
    """adaLN-Zero temporal self-attention plus token-level text cross-attention."""
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_text = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(approximate="tanh"),
                                 nn.Linear(mlp_ratio * dim, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, text: torch.Tensor,
                text_mask: torch.Tensor | None = None) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(condition).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift1, scale1)
        h = self.self_attn(h, h, h, need_weights=False)[0]
        x = x + gate1[:, None] * h
        key_padding = ~text_mask.bool() if text_mask is not None else None
        x = x + self.cross_attn(self.norm_text(x), text, text,
                                key_padding_mask=key_padding, need_weights=False)[0]
        x = x + gate2[:, None] * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x


class MotionDiT(nn.Module):
    def __init__(self, frames: int = 50, motion_dim: int = MOTION_DIM, text_dim: int = 512,
                 dim: int = 256, depth: int = 6, heads: int = 4, predict_contacts: bool = False):
        super().__init__()
        self.frames = frames
        self.motion_dim = motion_dim
        self.predict_contacts = predict_contacts
        self.input = nn.Linear(motion_dim, dim)
        self.position = nn.Parameter(torch.randn(1, frames, dim) * 0.02)
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.text_proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([MotionBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.output = nn.Linear(dim, motion_dim)
        self.contact_head = nn.Linear(dim, 4) if predict_contacts else None
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if self.contact_head is not None:
            nn.init.zeros_(self.contact_head.weight)
            nn.init.zeros_(self.contact_head.bias)

    def forward(self, motion: torch.Tensor, t: torch.Tensor, text: torch.Tensor,
                text_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if motion.ndim != 3 or motion.shape[1:] != (self.frames, self.motion_dim):
            raise ValueError(f"expected [B,{self.frames},{self.motion_dim}], got {tuple(motion.shape)}")
        text = self.text_proj(text.to(motion.dtype))
        mask = text_mask.to(text.dtype)[..., None] if text_mask is not None else torch.ones_like(text[..., :1])
        pooled = (text * mask).sum(1) / mask.sum(1).clamp_min(1)
        condition = self.time(timestep_embedding(t, self.position.shape[-1])) + pooled
        h = self.input(motion) + self.position
        for block in self.blocks:
            h = block(h, condition, text, text_mask)
        shift, scale = self.final_ada(condition).chunk(2, dim=-1)
        h = modulate(self.norm(h), shift, scale)
        return self.output(h), self.contact_head(h) if self.contact_head is not None else None


def pack_motion(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    joints = batch["joints"].flatten(2)
    return torch.cat([joints, batch["root"], batch["heading"]], dim=-1)


def unpack_motion(motion: np.ndarray, stats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Denormalize a packed model sample into renderable motion components."""
    motion = np.asarray(motion, dtype=np.float32)
    joints = motion[..., :81].reshape(motion.shape[:-1] + (27, 3))
    root = motion[..., 81:84]
    heading = motion[..., 84:86]
    joints = joints * stats["joints_std"] + stats["joints_mean"]
    root = root * stats["root_std"] + stats["root_mean"]
    heading /= np.maximum(np.linalg.norm(heading, axis=-1, keepdims=True), 1e-8)
    return joints.astype(np.float32), root.astype(np.float32), heading.astype(np.float32)


def write_inference_artifacts(joints: np.ndarray, root: np.ndarray, prompts: list[str], out_dir: str,
                              step: int, noise_seed: int) -> dict[str, str]:
    """Write a canonical 64px GIF/strip and manifest for the first fixed sample."""
    from PIL import Image
    from generator.render import render
    from generator.skeleton import Body, Camera, NAMES, project

    artifact_dir = os.path.join(out_dir, f"inference_{step:06d}")
    os.makedirs(artifact_dir, exist_ok=True)
    body = Body()
    camera = Camera(yaw=math.radians(50))
    frames = []
    for frame in range(joints.shape[1]):
        xyz = joints[0, frame] + root[0, frame, None]
        skeleton = {name: tuple(xyz[i]) for i, name in enumerate(NAMES)}
        image = render(*project(skeleton, camera, body.px_per_m), body,
                       bg=(255, 255, 255, 255)).convert("RGB")
        frames.append(image.resize((64, 64), Image.Resampling.LANCZOS))
    gif_path = os.path.join(artifact_dir, "sample.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    selected = frames[::5]
    strip = Image.new("RGB", (64 * len(selected), 64), "white")
    for i, image in enumerate(selected):
        strip.paste(image, (64 * i, 0))
    strip_path = os.path.join(artifact_dir, "strip.png")
    strip.save(strip_path)
    manifest_path = os.path.join(artifact_dir, "manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump({"step": step, "prompt": prompts[0], "noise_seed": noise_seed,
                   "frames": 50, "fps": 10, "render": "canonical diagnostic render at 64px",
                   "gif": "sample.gif", "strip": "strip.png"}, handle, indent=2)
    return {"gif": gif_path, "strip": strip_path, "manifest": manifest_path}


def flow_loss(model: MotionDiT, clean: torch.Tensor, text: torch.Tensor,
              text_mask: torch.Tensor | None = None, contacts: torch.Tensor | None = None,
              contacts_available: torch.Tensor | None = None, contact_weight: float = 0.0,
              generator: torch.Generator | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    batch = clean.shape[0]
    t = torch.sigmoid(torch.randn(batch, device=clean.device, generator=generator))
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    xt = (1 - t[:, None, None]) * clean + t[:, None, None] * noise
    prediction, contact_logits = model(xt, t, text, text_mask)
    flow = F.mse_loss(prediction.float(), (noise - clean).float())
    total = flow
    contact = torch.zeros((), device=clean.device)
    if contact_weight > 0 and contact_logits is not None and contacts is not None:
        available = contacts_available.bool() if contacts_available is not None else torch.ones(batch, dtype=torch.bool, device=clean.device)
        if available.any():
            contact = F.binary_cross_entropy_with_logits(contact_logits[available].float(), contacts[available].float())
            total = total + contact_weight * contact
    return total, {"flow": float(flow.detach()), "contact": float(contact.detach())}


@torch.no_grad()
def euler_sample(model: MotionDiT, shape: tuple[int, int, int], text: torch.Tensor,
                 text_mask: torch.Tensor | None = None, steps: int = 30,
                 noise: torch.Tensor | None = None, null_text: torch.Tensor | None = None,
                 null_mask: torch.Tensor | None = None, cfg: float = 1.0) -> torch.Tensor:
    device = text.device
    x = torch.randn(shape, device=device) if noise is None else noise.clone()
    times = torch.linspace(1, 0, steps + 1, device=device)
    for i in range(steps):
        t = torch.full((shape[0],), float(times[i]), device=device)
        velocity, _ = model(x, t, text, text_mask)
        if cfg != 1.0 and null_text is not None:
            unconditional, _ = model(x, t, null_text, null_mask)
            velocity = unconditional + cfg * (velocity - unconditional)
        x = x + (times[i + 1] - times[i]) * velocity
    return x


class TorchMotionCache(torch.utils.data.Dataset):
    def __init__(self, path: str, split: str):
        self.cache = MotionCache(path, split, normalize=True)

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, item: int):
        row = self.cache[item]
        return {"joints": torch.from_numpy(row["joints"]), "root": torch.from_numpy(row["root"]),
                "heading": torch.from_numpy(row["heading"]), "contacts": torch.from_numpy(row["contacts"]),
                "contacts_available": row["contacts_available"], "text": row["metadata"]["text"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--text-encoder", default="google-t5/t5-small")
    parser.add_argument("--text-len", type=int, default=32)
    parser.add_argument("--cfg-drop", type=float, default=0.1)
    parser.add_argument("--contact-weight", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="write resumable latest.pt (milestones still follow --sample-every)")
    parser.add_argument("--sample-seed", type=int, default=12345)
    parser.add_argument("--resume", default="auto", help="checkpoint path, 'auto' for OUT/latest.pt, or empty")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)

    train_set = TorchMotionCache(args.cache, "train")
    val_set = TorchMotionCache(args.cache, "val")
    if not len(train_set):
        raise ValueError("motion cache has no training clips")
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, drop_last=len(train_set) >= args.batch, pin_memory=device.type == "cuda")

    from transformers import AutoTokenizer, T5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    encoder = T5EncoderModel.from_pretrained(args.text_encoder).to(device).eval().requires_grad_(False)
    prompts = sorted({train_set.cache.records[i]["text"] for i in train_set.cache.indices} |
                     {val_set.cache.records[i]["text"] for i in val_set.cache.indices} | {""})
    text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for start in range(0, len(prompts), 32):
        strings = prompts[start:start + 32]
        token = tokenizer(strings, padding="max_length", truncation=True, max_length=args.text_len, return_tensors="pt")
        with torch.no_grad():
            hidden = encoder(input_ids=token.input_ids.to(device), attention_mask=token.attention_mask.to(device)).last_hidden_state.cpu()
        for string, h, mask in zip(strings, hidden, token.attention_mask):
            text_cache[string] = h, mask
    text_dim = int(encoder.config.d_model)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def text_batch(strings: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, masks = zip(*(text_cache[string] for string in strings))
        return torch.stack(hidden).to(device), torch.stack(masks).to(device)

    model = MotionDiT(text_dim=text_dim, dim=args.dim, depth=args.depth, heads=args.heads,
                      predict_contacts=args.contact_weight > 0).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    start_step = 0
    resume_path = os.path.join(args.out, "latest.pt") if args.resume == "auto" else args.resume
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        if "torch_rng" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        if "numpy_rng" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng"])
        if "python_rng" in checkpoint:
            random.setstate(checkpoint["python_rng"])
        print(f"resumed {resume_path} at step {start_step}", flush=True)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"S0 MotionDiT {parameters / 1e6:.2f}M params; {len(train_set)} train / {len(val_set)} val", flush=True)
    iterator = iter(loader)
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    log_path = os.path.join(args.out, "log.jsonl")
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else contextlib.nullcontext

    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        clean = pack_motion({k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)})
        text, mask = text_batch(list(batch["text"]))
        null_text, null_mask = text_batch([""] * len(batch["text"]))
        drop = torch.rand(clean.shape[0], device=device) < args.cfg_drop
        text = torch.where(drop[:, None, None], null_text, text)
        mask = torch.where(drop[:, None], null_mask, mask)
        with amp():
            loss, parts = flow_loss(model, clean, text, mask, batch["contacts"].to(device),
                batch["contacts_available"].to(device), args.contact_weight)
            scaled_loss = loss / args.accum
        scaled_loss.backward()
        if step % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                decay = min(0.999, (1 + step) / (10 + step))
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1 - decay)

        record = {"step": step, "loss": float(loss.detach()), **parts,
                  "seconds": time.time() - start_time}
        if step == 1 or step % 50 == 0:
            print(json.dumps(record), flush=True)
            with open(log_path, "a") as handle:
                handle.write(json.dumps(record) + "\n")

        if step % args.sample_every == 0 and len(val_set):
            rows = [val_set[i] for i in range(min(4, len(val_set)))]
            prompt = [row["text"] for row in rows]
            txt, txt_mask = text_batch(prompt)
            fixed_generator = torch.Generator(device=device).manual_seed(args.sample_seed)
            fixed_noise = torch.randn((len(rows), 50, MOTION_DIM), device=device, generator=fixed_generator)
            with torch.no_grad(), amp():
                sample = euler_sample(ema, (len(rows), 50, MOTION_DIM), txt, txt_mask,
                                      steps=30, noise=fixed_noise)
            packed = sample.float().cpu().numpy()
            joints, root, heading = unpack_motion(packed, val_set.cache.stats)
            np.savez_compressed(os.path.join(args.out, f"samples_{step:06d}.npz"),
                motion_normalized=packed, joints=joints, root=root, heading=heading,
                prompts=np.asarray(prompt))
            artifacts = write_inference_artifacts(joints, root, prompt, args.out, step, args.sample_seed)
            print(f"fixed inference artifacts: {artifacts}", flush=True)

        if step % args.val_every == 0 and len(val_set):
            ema.eval()
            values = []
            with torch.no_grad():
                for i in range(0, min(len(val_set), 64), args.batch):
                    rows = [val_set[j] for j in range(i, min(i + args.batch, len(val_set), 64))]
                    clean = torch.stack([pack_motion({
                        "joints": row["joints"][None], "root": row["root"][None],
                        "heading": row["heading"][None]})[0] for row in rows]).to(device)
                    txt, txt_mask = text_batch([row["text"] for row in rows])
                    generator = torch.Generator(device=device).manual_seed(10_000 + i)
                    value, _ = flow_loss(ema, clean, txt, txt_mask, generator=generator)
                    values.append(float(value))
            val = sum(values) / len(values)
            with open(log_path, "a") as handle:
                handle.write(json.dumps({"step": step, "val_flow": val}) + "\n")
            print(f"step {step}: val_flow={val:.6f}", flush=True)

        if step % args.checkpoint_every == 0 or step == args.steps:
            state = {"step": step, "model": model.state_dict(), "ema": ema.state_dict(),
                     "optimizer": optimizer.state_dict(), "args": vars(args),
                     "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                     "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
            if step % args.sample_every == 0 or step == args.steps:
                milestone = os.path.join(args.out, f"ckpt_{step:06d}.pt")
                torch.save(state, milestone)
            temporary = os.path.join(args.out, ".latest.pt.tmp")
            torch.save(state, temporary)
            os.replace(temporary, os.path.join(args.out, "latest.pt"))


if __name__ == "__main__":
    main()
