"""Controlled qualitative evaluation for a full-prompt factorised-UNet checkpoint.

The three panels separate text sensitivity from stochastic variation:

* fixed noise, varied prompts;
* fixed prompt, varied noise; and
* varied prompts with independently seeded noise.

The fixed-noise panel replays the complete AR noise stream per prompt, so text
is the only changed input. This establishes conditioning sensitivity, not
semantic prompt correctness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval.post_eval_t2v import (
    checkpoint_sha256,
    experiment_specs,
    load_prompts,
    save_labeled_gif,
    save_strip,
    unique_prompts,
    _straight_rgba_uint8,
)
from train.video_ddpm import UNet3D, alphas_cumprod, rollout, to_gif


def rollout_chunks(context_frames: int, new_frames: int, target_frames: int) -> int:
    """Return the AR chunk count without importing the full metric stack."""
    first = context_frames + new_frames
    return 1 if target_frames <= first else 1 + int(np.ceil((target_frames - first) / new_frames))


def mean_pairwise_l1(videos: torch.Tensor) -> float:
    """Mean L1 distance over unordered sample pairs."""
    if videos.shape[0] < 2:
        return 0.0
    value = videos.float().flatten(1)
    pairs = []
    for left in range(value.shape[0]):
        for right in range(left + 1, value.shape[0]):
            pairs.append(torch.mean(torch.abs(value[left] - value[right])))
    return float(torch.stack(pairs).mean())


def build_model(checkpoint: dict, device: str):
    args = checkpoint["args"]
    if args.get("cond") != "text" or checkpoint.get("arch") in {"dit_fm", "dit_fm_t2v"}:
        raise ValueError("post_eval_unet requires a full-prompt UNet checkpoint")
    state = checkpoint.get("ema") or checkpoint.get("model")
    if state is None or "text_proj.weight" not in state:
        raise ValueError("checkpoint does not contain text-projection weights")
    context = int(args.get("ar_ctx", 0))
    if context <= 0:
        raise ValueError("a multi-second prompt suite requires an --ar_ctx checkpoint")
    model = UNet3D(
        ch=int(args.get("ch", 64)),
        size=int(args.get("size", 64)),
        cond_ch=5,
        text_dim=int(state["text_proj.weight"].shape[1]),
        temporal_neighbors=int(args.get("temporal_neighbors", 0)),
        temporal_pos_bias=bool(args.get("temporal_pos_bias", False)),
    ).to(device)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    return model, args


def encode_prompts(prompts, args, device):
    from transformers import AutoTokenizer, T5EncoderModel

    name = args.get("text_encoder", "google-t5/t5-small")
    tokenizer = AutoTokenizer.from_pretrained(name)
    encoder = T5EncoderModel.from_pretrained(name).to(device).eval().requires_grad_(False)
    tokens = tokenizer(
        list(prompts) + [""], padding="max_length", truncation=True,
        max_length=int(args.get("text_len", 32)), return_tensors="pt",
    )
    mask = tokens.attention_mask.to(device)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=amp_dtype, enabled=str(device).startswith("cuda")
    ):
        hidden = encoder(
            input_ids=tokens.input_ids.to(device), attention_mask=mask
        ).last_hidden_state.float()
    del encoder
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    count = len(prompts)
    return hidden[:count], mask[:count], hidden[count:count + 1], mask[count:count + 1], name


@torch.no_grad()
def sample_panel(model, args, prompts, seeds, text, mask, null_text, null_mask,
                 target_frames, sample_steps, cfg, device):
    context = int(args["ar_ctx"])
    new_frames = int(args["frames"])
    chunks = rollout_chunks(context, new_frames, target_frames)
    schedule = alphas_cumprod().to(device)
    outputs = []
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(int(seed))
        value = rollout(
            model, 1, context, new_frames, chunks, schedule, device,
            steps=sample_steps, S=int(args.get("size", 64)), generator=generator,
            text=text[index:index + 1], text_mask=mask[index:index + 1],
            null_text=null_text, null_text_mask=null_mask, cfg=cfg,
        )[:, :, :target_frames]
        outputs.append(value.cpu())
    return torch.cat(outputs, dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", default="")
    parser.add_argument("--prompts_file", "--prompts-file", default="")
    parser.add_argument("--same_prompt", "--same-prompt", default="")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strip_frames", "--strip-frames", default="")
    parser.add_argument("--save_rgba", "--save-rgba", action="store_true")
    options = parser.parse_args()

    destination = Path(options.out)
    destination.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(options.cache, options.prompts_file)
    specs = experiment_specs(prompts, options.n, options.seed, options.same_prompt)
    checkpoint = torch.load(options.ckpt, map_location=options.device, weights_only=False)
    model, train_args = build_model(checkpoint, options.device)
    all_prompts = unique_prompts(prompt for spec in specs.values() for prompt in spec["prompts"])
    all_text, all_mask, null_text, null_mask, encoder_name = encode_prompts(
        all_prompts, train_args, options.device
    )
    prompt_index = {prompt: index for index, prompt in enumerate(all_prompts)}
    frame_indices = [int(value) for value in options.strip_frames.split(",") if value.strip()] or None
    manifest = {
        "protocol": "full_prompt_unet_qualitative_v1",
        "claim": "conditioning sensitivity; not semantic prompt correctness",
        "checkpoint": str(Path(options.ckpt).resolve()),
        "checkpoint_sha256": checkpoint_sha256(options.ckpt),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "text_encoder": encoder_name,
        "shape": [4, options.frames, int(train_args.get("size", 64)), int(train_args.get("size", 64))],
        "sampler": {"name": "DDIM chunked rollout", "steps": options.steps,
                    "cfg": options.cfg, "fps": options.fps},
        "experiments": {},
    }
    generated = {}
    for name, spec in specs.items():
        indices = torch.tensor([prompt_index[prompt] for prompt in spec["prompts"]], device=all_text.device)
        videos = sample_panel(
            model, train_args, spec["prompts"], spec["noise_seeds"],
            all_text[indices], all_mask[indices], null_text, null_mask,
            options.frames, options.steps, options.cfg, options.device,
        )
        generated[name] = videos
        gif_path = destination / f"{name}.gif"
        labeled_path = destination / f"{name}_labeled.gif"
        strip_path = destination / f"{name}_strip.png"
        to_gif(videos, str(gif_path), fps=options.fps)
        save_labeled_gif(videos, str(labeled_path), spec["prompts"], fps=options.fps)
        used_frames = save_strip(videos, str(strip_path), spec["prompts"], frame_indices)
        rgba_path = destination / f"{name}_rgba.npz"
        if options.save_rgba:
            np.savez_compressed(
                rgba_path, rgba=_straight_rgba_uint8(videos),
                prompts=np.asarray(spec["prompts"]), seeds=np.asarray(spec["noise_seeds"]),
            )
        manifest["experiments"][name] = {
            **spec, "gif": gif_path.name, "labeled_gif": labeled_path.name,
            "strip": strip_path.name, "strip_frames": used_frames,
            "rgba": rgba_path.name if options.save_rgba else None,
            "mean_pairwise_l1": mean_pairwise_l1(videos),
        }
    prompt_l1 = manifest["experiments"]["fixed_noise_varied_prompt"]["mean_pairwise_l1"]
    noise_l1 = manifest["experiments"]["fixed_prompt_varied_noise"]["mean_pairwise_l1"]
    manifest["prompt_to_noise_l1_ratio"] = prompt_l1 / max(noise_l1, 1e-12)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote prompt suite to {destination}; prompt/noise L1 ratio {manifest['prompt_to_noise_l1_ratio']:.3f}")


if __name__ == "__main__":
    main()
