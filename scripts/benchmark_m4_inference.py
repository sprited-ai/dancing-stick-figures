#!/usr/bin/env python3
"""Benchmark M4 block sampling without writing generated videos."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from train.video_dit_ar import ARVideoDiT, rollout_blocks


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", default="4,10,20")
    parser.add_argument("--frames", default="10,100")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--prompt", default="A person walks forward.")
    parser.add_argument("--synthetic-text", action="store_true",
                        help="benchmark diffusion only with shape-correct cached-like embeddings")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved = checkpoint.get("args", {})
    encoder_name = saved.get("text_encoder", "google-t5/t5-small")
    text_length = int(saved.get("text_len", 32))
    text_dim = int(checkpoint["ema"]["text_proj.weight"].shape[1])
    if args.synthetic_text:
        generator = torch.Generator(device=args.device).manual_seed(80001)
        embeddings = torch.randn((2, text_length, text_dim), device=args.device, generator=generator)
        masks = torch.ones((2, text_length), dtype=torch.long, device=args.device)
        text_encoding_seconds = None
    else:
        from transformers import AutoTokenizer, T5EncoderModel
        tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        encoder = T5EncoderModel.from_pretrained(encoder_name).to(args.device).eval().requires_grad_(False)
        tokens = tokenizer([args.prompt, ""], padding="max_length", truncation=True,
                           max_length=text_length, return_tensors="pt")
        synchronize(args.device)
        encode_start = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            embeddings = encoder(input_ids=tokens.input_ids.to(args.device),
                                 attention_mask=tokens.attention_mask.to(args.device)).last_hidden_state
        synchronize(args.device)
        text_encoding_seconds = time.perf_counter() - encode_start
        masks = tokens.attention_mask.to(args.device)
        del encoder
    text, null_text = embeddings[:1], embeddings[1:]
    text_mask, null_mask = masks[:1], masks[1:]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    model = ARVideoDiT(
        size=int(saved.get("size", 64)), patch=int(saved.get("patch", 4)),
        dim=int(saved.get("dim", 384)), depth=int(saved.get("depth", 12)),
        heads=int(saved.get("heads", 6)), cond_ch=5, text_dim=text_dim,
    ).to(args.device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["ema"])
    target_frames = int(saved.get("target_frames", 10))
    history_max = int(saved.get("history_max", 40))
    fps = 20
    step_values = [int(value) for value in args.steps.split(",")]
    frame_values = [int(value) for value in args.frames.split(",")]
    results = []

    def run(sample_steps: int, total_frames: int, seed: int) -> None:
        generator = torch.Generator(device=args.device).manual_seed(seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=args.device.startswith("cuda")):
            rollout_blocks(
                model, [(text, text_mask)], total_frames=total_frames,
                target_frames=target_frames, history_max=history_max, steps=sample_steps,
                null_text=null_text, null_mask=null_mask, cfg=args.cfg,
                shift=float(saved.get("shift", 1.0)), generator=generator,
            )

    for sample_steps in step_values:
        for total_frames in frame_values:
            for index in range(args.warmup):
                run(sample_steps, total_frames, 91000 + index)
            durations = []
            for index in range(args.repeats):
                synchronize(args.device)
                started = time.perf_counter()
                run(sample_steps, total_frames, 92000 + index)
                synchronize(args.device)
                durations.append(time.perf_counter() - started)
            median = statistics.median(durations)
            video_seconds = total_frames / fps
            results.append({
                "sampling_steps_per_block": sample_steps,
                "frames": total_frames,
                "blocks": (total_frames + target_frames - 1) // target_frames,
                "video_seconds_at_20fps": video_seconds,
                "durations_seconds": durations,
                "median_seconds": median,
                "generated_video_seconds_per_wall_second": video_seconds / median,
                "realtime_factor": median / video_seconds,
            })
            print(json.dumps(results[-1]), flush=True)

    payload = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else "CPU",
        "dtype": "bfloat16 autocast" if args.device.startswith("cuda") else "float32",
        "cfg": args.cfg,
        "model_forwards_per_sampling_step": 2 if args.cfg > 0 else 1,
        "text_encoder": encoder_name,
        "text_encoding_seconds_for_prompt_and_null": text_encoding_seconds,
        "prompt": args.prompt,
        "fps": fps,
        "target_frames_per_block": target_frames,
        "history_max": history_max,
        "warmup_runs": args.warmup,
        "timed_repeats": args.repeats,
        "results": results,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
