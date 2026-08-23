"""Evaluate an M6 latent block-AR checkpoint at matched 10/20-step sampling.

The evaluator generates each video from an explicit per-sample seed so sampler
comparisons and prompt/noise sensitivity panels are paired.  It reports the
palette oracle, long-horizon drift/repetition/seam diagnostics, sampling time,
and CUDA peak memory.  Prompt distance remains a sensitivity measurement, not
semantic correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from eval.long_horizon_metrics import aggregate as aggregate_long, score_long_horizon
from eval.post_eval_t2v import save_labeled_gif, save_strip
from eval.run_ckpt import to_uint8_rgba
from eval.text_dit_metrics import load_split_prompts, mean_pairwise_l1, oracle_summary
from train.latent_video_dit_ar import decode_full, decode_sliding, load_codec
from train.video_dit_ar import ARVideoDiT, FullSTARVideoDiT, rollout_blocks


PROMPT_MOTION_PATHS = {
    "centroid_speed": ("whole_video", "centroid_speed"),
    "motion_fraction": ("whole_video", "motion_fraction"),
    "motion_decay": ("time_drift", "slope", "motion_fraction"),
    "limb_relative_motion": ("part_motion", "mean_limb_relative_motion"),
}


def _nested_metric(row: dict, path: tuple[str, ...]) -> float:
    value = row
    for key in path:
        value = value[key]
    return float(value)


def prompt_motion_alignment(prompts: list[str], generated_rows: list[dict], real_rows: list[dict]) -> dict:
    """Compare prompt-level motion profiles without claiming semantic correctness.

    Generated and real videos share a prompt but are not motion-paired. We
    average within prompt before comparing prompt-level variation.
    """
    if not prompts or len(prompts) != len(generated_rows) or len(prompts) != len(real_rows):
        raise ValueError("prompt-motion alignment requires non-empty matched prompt/row lists")
    names = sorted(set(prompts))
    per_prompt: dict[str, dict] = {name: {"n": prompts.count(name)} for name in names}
    summaries = {}
    for metric, path in PROMPT_MOTION_PATHS.items():
        generated_means, real_means = [], []
        for name in names:
            indices = [index for index, prompt in enumerate(prompts) if prompt == name]
            generated = float(np.mean([_nested_metric(generated_rows[index], path) for index in indices]))
            real = float(np.mean([_nested_metric(real_rows[index], path) for index in indices]))
            per_prompt[name][metric] = {"generated": generated, "real": real}
            generated_means.append(generated)
            real_means.append(real)
        generated_array = np.asarray(generated_means, dtype=np.float64)
        real_array = np.asarray(real_means, dtype=np.float64)
        correlation = None
        if len(names) >= 2 and generated_array.std() > 1e-12 and real_array.std() > 1e-12:
            correlation = float(np.corrcoef(generated_array, real_array)[0, 1])
        summaries[metric] = {
            "pearson_across_prompt_means": correlation,
            "normalized_mae": float(
                np.mean(np.abs(generated_array-real_array)) /
                max(np.mean(np.abs(real_array)), 1e-12)
            ),
        }
    return {
        "n_videos": len(prompts), "n_prompts": len(names),
        "metrics": summaries, "per_prompt": per_prompt,
        "claim_limit": (
            "This compares motion profiles for shared prompt labels against unpaired real motions; "
            "it is not a general text-video semantic correctness score."
        ),
    }


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block_boundaries(video_frames: int, block_frames: int) -> list[int]:
    if video_frames <= 0 or block_frames <= 0:
        raise ValueError("frame counts must be positive")
    return list(range(block_frames, video_frames, block_frames))


def load_real_references(cache: str, split: str, n: int, frames: int, size: int):
    """Load deterministic, source-motion-distinct camera-0 references.

    Keeping one camera per source motion avoids quietly treating alternate
    views of the same motion as independent reference videos.  The returned
    prompts are also used for generation, so every generated sample has a
    prompt-matched real reference without claiming pairwise motion alignment.
    """
    cache_path = Path(cache)
    clip_map = json.loads((cache_path / "clips.json").read_text())
    eligible = [
        (clip_id, row) for clip_id, row in sorted(clip_map.items())
        if row.get("split") == split and not row.get("qa")
        and int(row.get("n", 0)) >= frames and clip_id.endswith("/c0")
    ]
    if len(eligible) < n:
        raise ValueError(f"requested {n} source-motion references but found {len(eligible)}")
    # Round-robin over prompts before taking a second seed from any prompt.
    # A lexicographic prefix would otherwise fill n=64 with only the first few
    # prompt directories even though the test split contains many prompts.
    by_prompt: dict[str, list[tuple[str, dict]]] = {}
    for item in eligible:
        by_prompt.setdefault(item[1]["text"], []).append(item)
    balanced = []
    for seed_index in range(max(len(rows) for rows in by_prompt.values())):
        for prompt in sorted(by_prompt):
            if seed_index < len(by_prompt[prompt]):
                balanced.append(by_prompt[prompt][seed_index])
    eligible = balanced
    store = np.load(cache_path / "frames.npy", mmap_mode="r")
    videos, prompts, clip_ids = [], [], []
    for clip_id, row in eligible[:n]:
        value = np.asarray(store[int(row["start"]):int(row["start"])+frames])
        if value.shape != (frames, size, size, 4):
            raise ValueError(
                f"reference {clip_id} has shape {value.shape}; expected {(frames, size, size, 4)}"
            )
        videos.append(value.copy())
        prompts.append(row["text"])
        clip_ids.append(clip_id)
    return np.stack(videos), prompts, clip_ids


def real_tensor_for_oracle(rgba: np.ndarray) -> torch.Tensor:
    """Convert straight-alpha uint8 references to model-style premultiplied RGBA."""
    value = torch.from_numpy(rgba.astype(np.float32) / 255.0)
    value[..., :3] *= value[..., 3:4]
    return value.permute(0, 4, 1, 2, 3).mul(2).sub(1)


def build(checkpoint: dict, device: str):
    if checkpoint.get("protocol") not in (
        "m6_latent_block_ar_v1", "m6_latent_block_ar_v2_full_st",
        "m6_latent_block_ar_v3_start_aligned", "m6_latent_block_ar_v4_decoded_rgba_aux",
        "m6_latent_block_ar_v5_noisy_history",
        "m6_latent_block_ar_v6_motion_weighted",
        "m6_latent_block_ar_v7_fg_weighted",
        "m6_latent_block_ar_v8_combined",
        "m6_latent_block_ar_v9_rig_cogen",
        "r0_latent_full_clip_v1",
    ):
        raise ValueError("checkpoint is not a supported latent video protocol")
    args = checkpoint["args"]
    codec_meta = checkpoint["codec"]
    for path_key, sha_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("stats", "latent_stats_sha256"),
        ("experiment_protocol", "experiment_protocol_sha256"),
    ):
        if sha256(codec_meta[path_key]) != codec_meta[sha_key]:
            raise ValueError(f"M6 {path_key} checksum no longer matches checkpoint metadata")
    codec, standardizer, codec_checkpoint, stats = load_codec(
        codec_meta["checkpoint"], codec_meta["stats"], device,
    )
    state = checkpoint.get("ema") or checkpoint.get("model")
    text_dim = int(state["text_proj.weight"].shape[1])
    latent_size = int(args["output_size"]) // codec.spatial_compression
    attention_mode = args.get("attention_mode", "factorized")
    if checkpoint.get("protocol") == "m6_latent_block_ar_v9_rig_cogen":
        from train.latent_video_dit_ar_rig import RigFullSTARVideoDiT

        model = RigFullSTARVideoDiT(
            temporal_compression=codec.temporal_compression,
            size=latent_size, patch=int(args["patch"]), in_ch=codec.latent_channels,
            dim=int(args["dim"]), depth=int(args["depth"]), heads=int(args["heads"]),
            cond_ch=codec.latent_channels + 1, text_dim=text_dim,
        ).to(device).eval().requires_grad_(False)
        model.load_state_dict(state, strict=True)
        return model, codec, standardizer, codec_checkpoint, stats, args
    model_class = FullSTARVideoDiT if attention_mode == "full" else ARVideoDiT
    model = model_class(
        size=latent_size, patch=int(args["patch"]), in_ch=codec.latent_channels,
        dim=int(args["dim"]), depth=int(args["depth"]), heads=int(args["heads"]),
        cond_ch=codec.latent_channels + 1, text_dim=text_dim,
    ).to(device).eval().requires_grad_(False)
    model.load_state_dict(state, strict=True)
    return model, codec, standardizer, codec_checkpoint, stats, args


def encode_text(prompts: list[str], args: dict, device: str):
    from transformers import AutoTokenizer, T5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(args["text_encoder"])
    encoder = T5EncoderModel.from_pretrained(args["text_encoder"]).to(device).eval().requires_grad_(False)
    tokens = tokenizer(prompts, padding="max_length", truncation=True,
                       max_length=int(args["text_len"]), return_tensors="pt")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        hidden = encoder(input_ids=tokens.input_ids.to(device),
                         attention_mask=tokens.attention_mask.to(device)).last_hidden_state
    return hidden, tokens.attention_mask.to(device)


def build_text_cache(prompts: list[str], args: dict, device: str):
    unique = list(dict.fromkeys([*prompts, ""]))
    hidden, masks = encode_text(unique, args, device)
    return {prompt: (embedding.unsqueeze(0), mask.unsqueeze(0))
            for prompt, embedding, mask in zip(unique, hidden, masks)}


@torch.no_grad()
def generate_one(model, codec, standardizer, args, prompt: str, text_cache, seed: int, steps: int,
                 cfg: float, device: str, *, decode_mode: str = "full",
                 rig_sink: list | None = None) -> tuple[torch.Tensor, float]:
    text, mask = text_cache[prompt]
    null_text, null_mask = text_cache[""]
    generator = torch.Generator(device=device).manual_seed(seed)
    if hasattr(model, "rig_dim"):
        # v9 rig co-generation: pixel and rig noise both come from the seeded
        # generator, so paired-noise semantics across prompts are preserved.
        from train.latent_video_dit_ar_rig import rollout_blocks_rig

        if device.startswith("cuda"):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        latent, rig = rollout_blocks_rig(
            model, [(text, mask)], total_frames=int(args["rollout_latents"]),
            target_frames=int(args["target_latents"]), history_max=int(args["history_max"]),
            steps=steps, null_text=null_text, null_mask=null_mask, cfg=cfg,
            shift=float(args["shift"]), generator=generator,
        )
        if rig_sink is not None:
            rig_sink.append(rig.cpu().float())
        rgba = decode_full(codec, standardizer, latent, output_size=int(args["output_size"]))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        return rgba.mul(2).sub(1).cpu(), time.perf_counter() - started
    required_noise_frames = (
        (int(args["rollout_latents"]) + int(args["target_latents"]) - 1)
        // int(args["target_latents"])
    ) * int(args["target_latents"])
    paired_noise = torch.randn(
        (1, model.C, required_noise_frames, model.S, model.S),
        device=device, generator=generator,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    latent = rollout_blocks(
        model, [(text, mask)], total_frames=int(args["rollout_latents"]),
        target_frames=int(args["target_latents"]), history_max=int(args["history_max"]),
        steps=steps, null_text=null_text, null_mask=null_mask, cfg=cfg,
        shift=float(args["shift"]), initial_noise=paired_noise, sample_clamp=None,
    )
    if decode_mode == "full":
        rgba = decode_full(codec, standardizer, latent, output_size=int(args["output_size"]))
    elif decode_mode == "sliding":
        # The bounded decode context is a cap on codec history, not part of the
        # generation protocol; shrink it so the remaining latents split into
        # whole commit blocks (e.g. history_max 12 over 25 rollout latents).
        commit = int(args["target_latents"])
        context = int(args["history_max"])
        context = max(commit, context - (latent.shape[2] - context) % commit)
        rgba = decode_sliding(
            codec, standardizer, latent, context_latents=context,
            commit_latents=commit, output_size=int(args["output_size"]),
        )
    else:
        raise ValueError(f"unknown decode mode: {decode_mode}")
    if device.startswith("cuda"): torch.cuda.synchronize()
    return rgba.mul(2).sub(1).cpu(), time.perf_counter() - started


def evaluate_sampler(
    model, codec, standardizer, args, prompts, text_cache, seeds, steps, cfg, device,
    *, comparison_block_frames: int, rig_sink: list | None = None,
):
    videos, seconds, peaks = [], [], []
    for prompt, seed in zip(prompts, seeds):
        video, elapsed = generate_one(
            model, codec, standardizer, args, prompt, text_cache, seed, steps, cfg, device,
            rig_sink=rig_sink,
        )
        videos.append(video); seconds.append(elapsed)
        peaks.append(torch.cuda.max_memory_allocated()/2**20 if device.startswith("cuda") else 0.0)
    videos = torch.cat(videos)
    rgba, _ = to_uint8_rgba(videos)
    native_block_frames = int(args["target_latents"]) * codec.temporal_compression
    boundaries = block_boundaries(rgba.shape[1], comparison_block_frames)
    long_rows = [score_long_horizon(video, boundaries=boundaries) for video in rgba]
    return videos, {
        "steps": steps, "n": len(videos), "oracle": oracle_summary(videos),
        "decode_policy": "full causal sequence",
        "long_horizon": aggregate_long(long_rows), "per_video_long_horizon": long_rows,
        "comparison_block_video_frames": comparison_block_frames,
        "comparison_boundary_frames": boundaries,
        "native_block_video_frames": native_block_frames,
        "native_boundary_frames": block_boundaries(rgba.shape[1], native_block_frames),
        "mean_seconds_per_video": float(np.mean(seconds)),
        "mean_seconds_per_video_frame": float(np.sum(seconds)/(len(videos)*videos.shape[2])),
        "peak_allocated_mib": float(max(peaks)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--sensitivity-n", type=int, default=6)
    parser.add_argument("--steps", default="10,20")
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--evaluation-protocol", default="")
    parser.add_argument(
        "--comparison-block-frames", type=int, default=4,
        help="common virtual boundary interval used for both full-clip R0 and block-AR M6",
    )
    parser.add_argument("--device", default="cuda")
    args_cli = parser.parse_args()
    destination = Path(args_cli.out); destination.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args_cli.ckpt, map_location=args_cli.device, weights_only=False)
    model, codec, standardizer, codec_checkpoint, stats, args = build(checkpoint, args_cli.device)
    rollout_video_frames = int(args["rollout_latents"]) * codec.temporal_compression
    real_rgba, prompts, real_clip_ids = load_real_references(
        args_cli.cache, args_cli.split, args_cli.n, rollout_video_frames,
        int(args["output_size"]),
    )
    available = load_split_prompts(args_cli.cache, args_cli.split)
    sens_prompts = [available[i % len(available)] for i in range(args_cli.sensitivity_n)]
    text_cache = build_text_cache([*prompts, *sens_prompts], args, args_cli.device)
    seeds = [args_cli.seed+i for i in range(args_cli.n)]
    step_counts = [int(value) for value in args_cli.steps.split(",")]
    samplers = {}
    videos_by_step = {}
    for steps in step_counts:
        rig_sink = [] if hasattr(model, "rig_dim") else None
        videos, result = evaluate_sampler(
            model, codec, standardizer, args, prompts, text_cache, seeds, steps,
            args_cli.cfg, args_cli.device,
            comparison_block_frames=args_cli.comparison_block_frames,
            rig_sink=rig_sink,
        )
        if rig_sink:
            np.save(destination / f"rigs_{steps:02d}step.npy",
                    ((torch.cat(rig_sink) + 1) / 2).numpy().astype(np.float16))
        videos_by_step[steps] = videos
        samplers[str(steps)] = result
        labels = [f"{steps} steps | {prompt}" for prompt in prompts[:4]]
        save_labeled_gif(videos[:4], str(destination/f"samples_{steps:02d}step.gif"), labels,
                         fps=int(args["fps"]))
        save_strip(videos[:4], str(destination/f"samples_{steps:02d}step_strip.png"), labels)

    anchor = sens_prompts[0]
    sensitivity = {}
    primary_steps = step_counts[0]
    fixed_noise = torch.cat([
        generate_one(model, codec, standardizer, args, prompt, text_cache, args_cli.seed+50_000,
                     primary_steps, args_cli.cfg, args_cli.device)[0]
        for prompt in sens_prompts
    ])
    varied_noise = torch.cat([
        generate_one(model, codec, standardizer, args, anchor, text_cache, args_cli.seed+60_000+i,
                     primary_steps, args_cli.cfg, args_cli.device)[0]
        for i in range(args_cli.sensitivity_n)
    ])
    fixed_noise_labels = [f"same noise | {prompt}" for prompt in sens_prompts]
    varied_noise_labels = [f"same prompt | seed {args_cli.seed+60_000+i} | {anchor}"
                           for i in range(args_cli.sensitivity_n)]
    save_labeled_gif(
        fixed_noise, str(destination / "same_noise_different_prompts.gif"),
        fixed_noise_labels, fps=int(args["fps"]),
    )
    save_strip(
        fixed_noise, str(destination / "same_noise_different_prompts_strip.png"),
        fixed_noise_labels,
    )
    save_labeled_gif(
        varied_noise, str(destination / "same_prompt_different_noise.gif"),
        varied_noise_labels, fps=int(args["fps"]),
    )
    save_strip(
        varied_noise, str(destination / "same_prompt_different_noise_strip.png"),
        varied_noise_labels,
    )
    prompt_l1, noise_l1 = mean_pairwise_l1(fixed_noise), mean_pairwise_l1(varied_noise)
    sensitivity.update(
        same_noise_different_prompt_l1=prompt_l1,
        same_prompt_different_noise_l1=noise_l1,
        prompt_to_noise_l1_ratio=prompt_l1/max(noise_l1, 1e-12),
        prompts=sens_prompts, fixed_noise_seed=args_cli.seed+50_000,
        claim_limit="Sensitivity only; this does not measure semantic prompt correctness.",
        artifacts=[
            "same_noise_different_prompts.gif", "same_noise_different_prompts_strip.png",
            "same_prompt_different_noise.gif", "same_prompt_different_noise_strip.png",
        ],
    )
    comparison = None
    if len(step_counts) >= 2:
        a, b = videos_by_step[step_counts[0]], videos_by_step[step_counts[1]]
        comparison = {
            "paired_output_l1": float((a.float()-b.float()).abs().mean()),
            "speed_ratio": samplers[str(step_counts[1])]["mean_seconds_per_video"] /
                max(samplers[str(step_counts[0])]["mean_seconds_per_video"], 1e-12),
        }
    # The primary metrics use exact full causal decoding. A small paired audit
    # separately measures the bounded-window approximation intended for future
    # low-memory streaming; it is not silently mixed into the quality score.
    sliding_pairs = []
    if int(args["history_max"]) > 0:
        for prompt, seed in zip(prompts[: min(4, len(prompts))], seeds):
            full = generate_one(
                model, codec, standardizer, args, prompt, text_cache, seed,
                primary_steps, args_cli.cfg, args_cli.device, decode_mode="full",
            )[0]
            sliding = generate_one(
                model, codec, standardizer, args, prompt, text_cache, seed,
                primary_steps, args_cli.cfg, args_cli.device, decode_mode="sliding",
            )[0]
            sliding_pairs.append(float((full.float() - sliding.float()).abs().mean()))
    native_block_frames = int(args["target_latents"]) * codec.temporal_compression
    real_boundaries = block_boundaries(rollout_video_frames, args_cli.comparison_block_frames)
    real_long_rows = [score_long_horizon(video, boundaries=real_boundaries) for video in real_rgba]
    motion_alignment = prompt_motion_alignment(
        prompts, samplers[str(primary_steps)]["per_video_long_horizon"], real_long_rows,
    )
    output = {
        "protocol": "m6_eval_v2", "checkpoint": str(Path(args_cli.ckpt).resolve()),
        "evaluation_protocol": None if not args_cli.evaluation_protocol else {
            "path": str(Path(args_cli.evaluation_protocol).resolve()),
            "sha256": sha256(args_cli.evaluation_protocol),
        },
        "attention_mode": args.get("attention_mode", "factorized"),
        "checkpoint_sha256": sha256(args_cli.ckpt), "checkpoint_step": checkpoint["step"],
        "codec": checkpoint["codec"], "latent_stats": stats,
        "split": args_cli.split, "prompts": prompts, "noise_seeds": seeds,
        "cfg": args_cli.cfg, "samplers": samplers,
        "step_comparison": comparison, "prompt_sensitivity": sensitivity,
        "prompt_motion_alignment": motion_alignment,
        "streaming_decode_audit": {
            "n": len(sliding_pairs),
            "full_vs_bounded_sliding_rgba_l1": None if not sliding_pairs else float(np.mean(sliding_pairs)),
            "bounded_context_latents": int(args["history_max"]),
            "claim_limit": "The bounded sliding decoder truncates the codec's theoretical receptive field; it is an approximation, not state-cache-equivalent streaming."
        },
        "real_reference": {
            "policy": "same split, one camera (c0) per distinct source motion; prompt matched but not motion paired",
            "clip_ids": real_clip_ids,
            "oracle": oracle_summary(real_tensor_for_oracle(real_rgba)),
            "long_horizon": aggregate_long(real_long_rows),
            "per_video_long_horizon": real_long_rows,
            "comparison_block_video_frames": args_cli.comparison_block_frames,
            "comparison_boundaries_are_virtual_controls": real_boundaries,
            "model_native_block_video_frames": native_block_frames,
        },
    }
    (destination/"metrics.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({"out": str(destination), "step": checkpoint["step"],
                      "step_comparison": comparison, "prompt_sensitivity": sensitivity}, indent=2))


if __name__ == "__main__":
    main()
