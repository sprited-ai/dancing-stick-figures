"""Qualitative post-evaluation for a text-conditioned VideoDiT checkpoint.

Produces three controlled panels:

* fixed noise, varied prompts: does text change the generated motion?
* fixed prompt, varied noise: is the model stochastic rather than memorising one clip?
* diverse prompt grid: a compact qualitative overview for a paper or lab note.

Each panel is saved as an animated GIF and a frame strip.  ``manifest.json``
records every prompt, noise seed, checkpoint hash, and sampling option.

Example (run beside a collected K1 checkpoint on a CUDA machine)::

    python -m eval.post_eval_t2v \
      --ckpt pod_results/k1_t2v50_64px_4090_b16_3k/ckpt_003000.pt \
      --cache data/cache --out output/k1_post_eval --n 6 --steps 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from train.video_dit_fm import VideoDiT, euler_sample, mixed_noise
from train.video_ddpm import to_gif


def checkpoint_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def unique_prompts(prompts: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for prompt in prompts:
        prompt = " ".join(str(prompt).split())
        if prompt and prompt not in seen:
            seen.add(prompt)
            out.append(prompt)
    return out


def load_prompts(cache: str = "", prompts_file: str = "") -> list[str]:
    """Load prompts from a newline/JSON file or a dataset cache manifest."""
    prompts = []
    if prompts_file:
        text = Path(prompts_file).read_text()
        if prompts_file.endswith(".json"):
            value = json.loads(text)
            if isinstance(value, dict):
                value = value.get("prompts", [])
            prompts.extend(value)
        else:
            prompts.extend(
                line for line in text.splitlines()
                if not line.lstrip().startswith("#")
            )
    if cache:
        clips_path = Path(cache) / "clips.json"
        if not clips_path.exists():
            raise FileNotFoundError(f"dataset prompt manifest not found: {clips_path}")
        clips = json.loads(clips_path.read_text())
        prompts.extend(row.get("text", "") for row in clips.values())
    result = unique_prompts(prompts)
    if not result:
        raise ValueError("no prompts found; pass --cache and/or --prompts_file")
    return result


def _rgba_over_white(videos: torch.Tensor) -> np.ndarray:
    """[B,4,T,H,W] in [-1,1] premultiplied RGBA -> uint8 RGB."""
    value = ((videos.detach().clamp(-1, 1) + 1) / 2).cpu().numpy()
    rgb, alpha = value[:, :3], value[:, 3:4]
    return (np.clip(rgb + 1 - alpha, 0, 1) * 255).astype(np.uint8)


def _straight_rgba_uint8(videos: torch.Tensor) -> np.ndarray:
    """[B,4,T,H,W] premultiplied [-1,1] -> straight uint8 [B,T,H,W,4]."""
    value = ((videos.detach().clamp(-1, 1) + 1) / 2).cpu().numpy()
    rgb, alpha = value[:, :3], value[:, 3:4]
    straight = np.where(alpha > 0.05, rgb / np.maximum(alpha, 0.05), 0.0)
    rgba = np.concatenate((np.clip(straight, 0, 1), np.clip(alpha, 0, 1)), axis=1)
    return (rgba.transpose(0, 2, 3, 4, 1) * 255).astype(np.uint8)


def save_strip(
    videos: torch.Tensor,
    path: str,
    labels: Sequence[str],
    frame_indices: Optional[Sequence[int]] = None,
) -> list[int]:
    """Save rows=samples, columns=selected frames with short prompt labels."""
    b, _, t, h, w = videos.shape
    if len(labels) != b:
        raise ValueError("labels must contain one entry per video")
    if frame_indices is None:
        frame_indices = np.linspace(0, t - 1, min(6, t), dtype=int).tolist()
    frame_indices = [int(i) for i in frame_indices]
    if not frame_indices or min(frame_indices) < 0 or max(frame_indices) >= t:
        raise ValueError("strip frame index is outside the generated video")

    rgb = _rgba_over_white(videos)  # [B,3,T,H,W]
    label_width = max(120, 3 * w)
    canvas = Image.new("RGB", (label_width + len(frame_indices) * w, b * h), "white")
    draw = ImageDraw.Draw(canvas)
    for row in range(b):
        text = labels[row]
        if len(text) > 34:
            text = text[:31] + "..."
        draw.text((4, row * h + 4), text, fill="black")
        for col, frame in enumerate(frame_indices):
            cell = Image.fromarray(rgb[row, :, frame].transpose(1, 2, 0))
            canvas.paste(cell, (label_width + col * w, row * h))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return frame_indices


def save_labeled_gif(videos: torch.Tensor, path: str, labels: Sequence[str], fps: int = 10) -> None:
    """Save an animated contact sheet with a persistent prompt beside each row."""
    b, _, t, h, w = videos.shape
    if len(labels) != b:
        raise ValueError("labels must contain one entry per video")
    rgb = _rgba_over_white(videos)
    label_width = max(220, 3 * w)
    frames = []
    for frame_index in range(t):
        canvas = Image.new("RGB", (label_width + w, b * h), "white")
        draw = ImageDraw.Draw(canvas)
        for row, label in enumerate(labels):
            wrapped = "\n".join(textwrap.wrap(label, width=34)[:3])
            draw.multiline_text((5, row * h + 5), wrapped, fill="black", spacing=2)
            cell = Image.fromarray(rgb[row, :, frame_index].transpose(1, 2, 0))
            canvas.paste(cell, (label_width, row * h))
        frames.append(canvas)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(destination, save_all=True, append_images=frames[1:],
                   duration=round(1000 / fps), loop=0, disposal=2)


def make_noise_batch(
    shape: Sequence[int],
    device: str,
    seeds: Sequence[int],
    noise_corr: float = 0.0,
) -> torch.Tensor:
    """Create one reproducible noise tensor per seed."""
    if shape[0] != len(seeds):
        raise ValueError("shape batch and number of noise seeds differ")
    rows = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        rows.append(mixed_noise((1,) + tuple(shape[1:]), device, noise_corr, generator))
    return torch.cat(rows, dim=0)


@torch.no_grad()
def sample_in_chunks(
    model: VideoDiT,
    noise: torch.Tensor,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    null_text: torch.Tensor,
    null_text_mask: torch.Tensor,
    steps: int,
    cfg: float,
    shift: float,
    batch: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, noise.shape[0], batch):
        stop = min(start + batch, noise.shape[0])
        outputs.append(
            euler_sample(
                model,
                noise[start:stop].shape,
                str(noise.device),
                steps=steps,
                noise=noise[start:stop],
                shift=shift,
                text=text[start:stop],
                text_mask=text_mask[start:stop],
                null_text=null_text[start:stop],
                null_text_mask=null_text_mask[start:stop],
                cfg=cfg,
            ).cpu()
        )
    return torch.cat(outputs)


def build_model(ckpt: dict, device: str) -> tuple[VideoDiT, dict]:
    args = ckpt["args"]
    if args.get("cond") != "text" or ckpt.get("arch") != "dit_fm_t2v":
        raise ValueError("post_eval_t2v requires a text-conditioned dit_fm_t2v checkpoint")
    state = ckpt.get("ema") or ckpt.get("model")
    if state is None or "text_proj.weight" not in state:
        raise ValueError("checkpoint does not contain EMA/model text projection weights")
    text_dim = int(state["text_proj.weight"].shape[1])
    model = VideoDiT(
        size=args.get("size", 128),
        frames=args.get("frames", 16),
        patch=args.get("patch", 4),
        dim=args.get("dim", 384),
        depth=args.get("depth", 12),
        heads=args.get("heads", 6),
        cond_ch=5 if args.get("i2v_frac", 0) > 0 else 0,
        text_dim=text_dim,
        local_3d=bool(args.get("local_3d", False)),
    ).to(device)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    for block in model.blocks:
        block.t1_skip = bool(args.get("t1_skip", True))
    return model, args


def encode_prompts(prompts: Sequence[str], args: dict, device: str):
    from transformers import AutoTokenizer, T5EncoderModel

    encoder_name = args.get("text_encoder", "google-t5/t5-small")
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    encoder = T5EncoderModel.from_pretrained(encoder_name).to(device).eval().requires_grad_(False)
    tokens = tokenizer(
        list(prompts) + [""],
        padding="max_length",
        truncation=True,
        max_length=args.get("text_len", 32),
        return_tensors="pt",
    )
    mask = tokens.attention_mask.to(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            hidden = encoder(tokens.input_ids.to(device), attention_mask=mask).last_hidden_state.float()
    del encoder
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    n = len(prompts)
    null_hidden = hidden[n:n + 1].expand(n, -1, -1)
    null_mask = mask[n:n + 1].expand(n, -1)
    return hidden[:n], mask[:n], null_hidden, null_mask, encoder_name


def experiment_specs(prompts: Sequence[str], n: int, base_seed: int, same_prompt: str = "") -> dict:
    if n <= 0:
        raise ValueError("n must be positive")
    chosen = [prompts[i % len(prompts)] for i in range(n)]
    anchor = same_prompt or chosen[0]
    return {
        "fixed_noise_varied_prompt": {
            "prompts": chosen,
            "noise_seeds": [base_seed] * n,
            "note": "Identical initial noise; only the text prompt changes.",
        },
        "fixed_prompt_varied_noise": {
            "prompts": [anchor] * n,
            "noise_seeds": [base_seed + i for i in range(n)],
            "note": "Identical text prompt; initial noise changes.",
        },
        "prompt_diverse_grid": {
            "prompts": chosen,
            "noise_seeds": [base_seed + 1000 + i for i in range(n)],
            "note": "Different prompts and independently seeded noise.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", default="", help="cache directory containing clips.json")
    parser.add_argument("--prompts_file", "--prompts-file", default="", help="newline text or JSON prompt list")
    parser.add_argument("--same_prompt", "--same-prompt", default="", help="override anchor prompt for varied-noise panel")
    parser.add_argument("--n", type=int, default=6, help="samples per panel")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strip_frames", "--strip-frames", default="", help="comma-separated frame indices; default is six even samples")
    parser.add_argument("--save_rgba", "--save-rgba", action="store_true",
                        help="also save straight-alpha uint8 RGBA archives for quantitative diagnostics")
    options = parser.parse_args()

    output = Path(options.out)
    output.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(options.cache, options.prompts_file)
    specs = experiment_specs(prompts, options.n, options.seed, options.same_prompt)
    checkpoint = torch.load(options.ckpt, map_location=options.device, weights_only=False)
    model, train_args = build_model(checkpoint, options.device)
    size, frames = int(train_args["size"]), int(train_args["frames"])
    frame_indices = [int(v) for v in options.strip_frames.split(",") if v.strip()] or None
    all_eval_prompts = unique_prompts(prompt for spec in specs.values() for prompt in spec["prompts"])
    all_text, all_mask, all_null, all_null_mask, encoder_name = encode_prompts(
        all_eval_prompts, train_args, options.device
    )
    prompt_index = {prompt: i for i, prompt in enumerate(all_eval_prompts)}

    manifest = {
        "protocol": "k1_t2v_qualitative_v1",
        "checkpoint": str(Path(options.ckpt).resolve()),
        "checkpoint_sha256": checkpoint_sha256(options.ckpt),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "architecture": checkpoint.get("arch"),
        "shape": [4, frames, size, size],
        "sampler": {"name": "rectified_flow_euler", "steps": options.steps, "cfg": options.cfg,
                    "shift": train_args.get("shift", 1.0), "fps": options.fps},
        "experiments": {},
    }
    for name, spec in specs.items():
        indices = torch.tensor([prompt_index[prompt] for prompt in spec["prompts"]], device=all_text.device)
        text, mask = all_text[indices], all_mask[indices]
        null_text = all_null[:1].expand(options.n, -1, -1)
        null_mask = all_null_mask[:1].expand(options.n, -1)
        noise = make_noise_batch(
            (options.n, 4, frames, size, size),
            options.device,
            spec["noise_seeds"],
            train_args.get("noise_corr", 0.0),
        )
        videos = sample_in_chunks(
            model, noise, text, mask, null_text, null_mask,
            options.steps, options.cfg, train_args.get("shift", 1.0), options.batch,
        )
        gif_path, strip_path = output / f"{name}.gif", output / f"{name}_strip.png"
        labeled_gif_path = output / f"{name}_labeled.gif"
        to_gif(videos, str(gif_path), fps=options.fps)
        save_labeled_gif(videos, str(labeled_gif_path), spec["prompts"], fps=options.fps)
        used_frames = save_strip(videos, str(strip_path), spec["prompts"], frame_indices)
        rgba_path = output / f"{name}_rgba.npz"
        if options.save_rgba:
            np.savez_compressed(rgba_path, rgba=_straight_rgba_uint8(videos),
                                prompts=np.asarray(spec["prompts"]),
                                seeds=np.asarray(spec["noise_seeds"], dtype=np.int64))
        manifest["text_encoder"] = encoder_name
        manifest["experiments"][name] = {
            **spec,
            "gif": gif_path.name,
            "labeled_gif": labeled_gif_path.name,
            "strip": strip_path.name,
            "strip_frames": used_frames,
            "rgba": rgba_path.name if options.save_rgba else None,
        }
        if options.device.startswith("cuda"):
            torch.cuda.empty_cache()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote 3 GIFs, 3 strips, and {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
