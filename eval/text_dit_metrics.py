"""Quantitative evaluation for text-conditioned image/video DiT checkpoints.

The evaluator keeps two claims separate:

* the existing palette oracle measures anatomy and motion in this synthetic
  domain;
* a same-noise prompt swap measures *prompt sensitivity*, not whether the
  generated motion is semantically correct.

Example::

    python -m eval.text_dit_metrics \
      --ckpt runs/k1/ckpt_003000.pt --cache data/cache \
      --out runs/k1/eval_text_003000.json --split test --n 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from eval.oracle import score_video
from eval.post_eval_t2v import (
    build_model,
    checkpoint_sha256,
    encode_prompts,
    make_noise_batch,
    sample_in_chunks,
    unique_prompts,
)
from eval.run_ckpt import FRAME, TEMPORAL, boot_ci, to_uint8_rgba


def load_split_prompts(cache: str, split: str) -> list[str]:
    """Return unique prompts from exactly one frozen dataset split."""
    rows = json.loads((Path(cache) / "clips.json").read_text())
    rows = rows.values() if isinstance(rows, dict) else rows
    prompts = unique_prompts(row.get("text", "") for row in rows if row.get("split") == split)
    if not prompts:
        raise ValueError(f"no prompts found for split {split!r}")
    return prompts


def mean_pairwise_l1(videos: torch.Tensor) -> float:
    """Mean L1 over all unordered output pairs; zero means identical outputs."""
    if videos.shape[0] < 2:
        raise ValueError("pairwise sensitivity requires at least two samples")
    flat = videos.float().flatten(1)
    distance = torch.cdist(flat, flat, p=1) / flat.shape[1]
    tri = torch.triu_indices(len(flat), len(flat), offset=1)
    return float(distance[tri[0], tri[1]].mean())


def oracle_summary(videos: torch.Tensor, bootstrap: int = 500) -> dict:
    rgba, _ = to_uint8_rgba(videos)
    per_video = [score_video(video) for video in rgba]
    result = {}
    for key in FRAME + TEMPORAL + ("fg",):
        values = [row[key] for row in per_video]
        result[key] = float(np.mean(values))
        result[f"{key}_ci95"] = list(boot_ci(values, B=bootstrap))
    return result


def _select(items: Iterable[str], n: int) -> list[str]:
    values = list(items)
    return [values[i % len(values)] for i in range(n)]


@torch.no_grad()
def generate(model, args, prompts, seeds, device, steps, cfg, batch):
    text, mask, null, null_mask, encoder_name = encode_prompts(prompts, args, device)
    shape = (len(prompts), 4, int(args["frames"]), int(args["size"]), int(args["size"]))
    noise = make_noise_batch(shape, device, seeds, args.get("noise_corr", 0.0))
    videos = sample_in_chunks(
        model, noise, text, mask, null, null_mask, steps, cfg,
        args.get("shift", 1.0), batch,
    )
    return videos, encoder_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--sensitivity_n", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_rgba", default="", help="optional compressed uint8 generated-video archive")
    options = parser.parse_args()

    if options.n < 2 or options.sensitivity_n < 2:
        raise ValueError("--n and --sensitivity_n must be at least 2")
    checkpoint = torch.load(options.ckpt, map_location=options.device, weights_only=False)
    model, args = build_model(checkpoint, options.device)
    prompts = load_split_prompts(options.cache, options.split)

    eval_prompts = _select(prompts, options.n)
    eval_seeds = [options.seed + i for i in range(options.n)]
    generated, encoder_name = generate(
        model, args, eval_prompts, eval_seeds, options.device,
        options.steps, options.cfg, options.batch,
    )

    sensitivity_prompts = _select(prompts, options.sensitivity_n)
    fixed_noise, _ = generate(
        model, args, sensitivity_prompts, [options.seed] * options.sensitivity_n,
        options.device, options.steps, options.cfg, options.batch,
    )
    anchor_prompts = [sensitivity_prompts[0]] * options.sensitivity_n
    varied_noise, _ = generate(
        model, args, anchor_prompts,
        [options.seed + 10_000 + i for i in range(options.sensitivity_n)],
        options.device, options.steps, options.cfg, options.batch,
    )
    prompt_l1 = mean_pairwise_l1(fixed_noise)
    noise_l1 = mean_pairwise_l1(varied_noise)

    output = {
        "protocol": "text_dit_metrics_v1",
        "checkpoint": str(Path(options.ckpt).resolve()),
        "checkpoint_sha256": checkpoint_sha256(options.ckpt),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "split": options.split,
        "n": options.n,
        "shape": [4, int(args["frames"]), int(args["size"]), int(args["size"])],
        "sampler": {"name": "rectified_flow_euler", "steps": options.steps,
                    "cfg": options.cfg, "shift": args.get("shift", 1.0)},
        "text_encoder": encoder_name,
        "prompts": eval_prompts,
        "noise_seeds": eval_seeds,
        "oracle": oracle_summary(generated),
        "prompt_sensitivity": {
            "n": options.sensitivity_n,
            "fixed_noise_prompts": sensitivity_prompts,
            "fixed_noise_seed": options.seed,
            "same_noise_different_prompt_l1": prompt_l1,
            "same_prompt_different_noise_l1": noise_l1,
            "prompt_to_noise_l1_ratio": prompt_l1 / max(noise_l1, 1e-12),
            "claim_limit": "Sensitivity only; this does not measure semantic prompt correctness.",
        },
    }
    destination = Path(options.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2))
    if options.save_rgba:
        rgba, _ = to_uint8_rgba(generated)
        Path(options.save_rgba).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(options.save_rgba, rgba=rgba, prompts=np.asarray(eval_prompts), seeds=np.asarray(eval_seeds))
    print(json.dumps({"out": str(destination), "step": output["checkpoint_step"],
                      "oracle": output["oracle"], "prompt_sensitivity": output["prompt_sensitivity"]}, indent=2))


if __name__ == "__main__":
    main()
