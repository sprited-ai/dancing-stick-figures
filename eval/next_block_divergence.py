"""Next-block divergence: the AR-native, perplexity-style metric.

Teacher-force ground-truth history from a held-out clip, generate the next
block N times, and measure the foreground-weighted RGBA distance between the
generated block and the clip's ACTUAL next block. Paired, prompt-conditioned,
and model-agnostic: any block-AR checkpoint (v8 pixel-only or v9 rig-cogen)
is scored by the same ground-truth continuation, so it is the common judge
for cross-family comparisons. Predicting the true continuation of "does
squats" requires actually squatting, so semantic following is priced in.

Reported per checkpoint:
  teacher_forced: per block position, best-of-N and mean divergence
  free_running:  the same comparison with the model's OWN prefix (exposure gap)
  real_floor:    same-prompt different-clip real blocks scored the same way
                 (the irreducible divergence of a legitimately different take)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval.eval_m6 import build, build_text_cache
from train.latent_video_dit_ar import decode_full, encode_video
from train.video_dit_ar import euler_sample_block


def fg_weighted_rgba_distance(a: torch.Tensor, b: torch.Tensor, bg_weight: float = 0.02) -> float:
    """a, b: [4, T, H, W] RGBA in [0,1] (premultiplied). Union-fg weighted MSE."""
    weight = torch.maximum(a[3], b[3]).clamp(0, 1)
    weight = weight + bg_weight * (1 - weight)
    err = (a[3] - b[3]).square() + (a[:3] - b[:3]).square().mean(dim=0)
    return float((err * weight).sum() / weight.sum().clamp_min(1e-8))


def load_test_clips(cache: Path, frames_needed: int, limit: int):
    clips = json.loads((cache / "clips.json").read_text())
    frames = np.load(cache / "frames.npy", mmap_mode="r")
    rows = sorted(
        ((cid, row) for cid, row in clips.items()
         if row["split"] == "test" and row["n"] >= frames_needed),
        key=lambda item: item[0],
    )[:limit]
    return rows, frames


def to_video(frames: np.ndarray, size: int) -> torch.Tensor:
    """uint8 [T,H,W,4] straight-alpha -> [4,T,h,w] premultiplied in [0,1]."""
    x = frames.astype(np.float32) / 255.0
    if x.shape[1] != size:
        f = x.shape[1] // size
        x = np.concatenate([x[..., :3] * x[..., 3:4], x[..., 3:4]], -1)
        x = x.reshape(x.shape[0], size, f, size, f, 4).mean((2, 4))
    else:
        x = np.concatenate([x[..., :3] * x[..., 3:4], x[..., 3:4]], -1)
    return torch.from_numpy(x).permute(3, 0, 1, 2)


@torch.no_grad()
def generate_block(model, codec, standardizer, args, history_rgba01, text, mask,
                   null_text, null_mask, steps, cfg, seed, device, rig_history=None):
    """history_rgba01 [4,Th,H,W] in [0,1] -> generated block RGBA [4,Tb,H,W] in [0,1]."""
    history_video = history_rgba01.unsqueeze(0).to(device) * 2 - 1
    history_latents = encode_video(codec, standardizer, history_video)
    generator = torch.Generator(device=device).manual_seed(seed)
    rig_block = None
    if hasattr(model, "rig_dim"):
        from train.latent_video_dit_ar_rig import euler_sample_block_rig

        block, rig_block = euler_sample_block_rig(
            model, history_latents, rig_history, int(args["target_latents"]),
            steps=steps, size=model.S, text=text, text_mask=mask,
            null_text=null_text, null_mask=null_mask, cfg=cfg,
            shift=float(args["shift"]), generator=generator,
        )
    else:
        block = euler_sample_block(
            model, history_latents, int(args["target_latents"]), steps=steps,
            size=model.S, text=text, text_mask=mask, null_text=null_text,
            null_mask=null_mask, cfg=cfg, shift=float(args["shift"]),
            generator=generator, clamp=None,
        )
    full = torch.cat((history_latents, block), dim=2)
    rgba = decode_full(codec, standardizer, full, output_size=int(args["output_size"]))
    block_frames = int(args["target_latents"]) * codec.temporal_compression
    return rgba[0, :, -block_frames:].cpu(), rig_block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--clips", type=int, default=32)
    parser.add_argument("--samples", type=int, default=4, help="best-of-N per block")
    parser.add_argument("--blocks", type=int, default=6, help="block positions per clip")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda")
    args_cli = parser.parse_args()

    checkpoint = torch.load(args_cli.ckpt, map_location=args_cli.device, weights_only=False)
    model, codec, standardizer, _, _, args = build(checkpoint, args_cli.device)
    temporal = codec.temporal_compression
    size = int(args["output_size"])
    history_frames = int(args["history_max"]) * temporal
    block_frames = int(args["target_latents"]) * temporal
    frames_needed = history_frames + args_cli.blocks * block_frames

    cache = Path(args_cli.cache)
    rows, frames = load_test_clips(cache, frames_needed, args_cli.clips)
    if not rows:
        raise SystemExit("no test clips long enough")
    rig = None
    if hasattr(model, "rig_dim"):
        rig = np.load(cache / "rig.npy", mmap_mode="r")

    prompts = [row["text"] for _, row in rows]
    text_cache = build_text_cache(prompts, args, args_cli.device)

    def rig_hist(row_start, offset):
        if rig is None:
            return None
        window = np.asarray(rig[row_start + offset - history_frames: row_start + offset], np.float32)
        tokens = (torch.from_numpy(window) * 2 - 1).reshape(
            1, int(args["history_max"]), temporal * 27 * 2)
        return tokens.to(args_cli.device)

    teacher = {b: {"best": [], "mean": []} for b in range(args_cli.blocks)}
    free = {b: [] for b in range(args_cli.blocks)}
    by_prompt_gt = {}
    for (cid, row), prompt in zip(rows, prompts):
        start = int(row["start"])
        text, mask = text_cache[prompt]
        null_text, null_mask = text_cache[""]
        gt_all = to_video(np.asarray(frames[start:start + frames_needed]), size)
        by_prompt_gt.setdefault(prompt, []).append(gt_all)
        # teacher-forced: GT prefix at every block position
        for b in range(args_cli.blocks):
            offset = history_frames + b * block_frames
            history = gt_all[:, offset - history_frames: offset]
            gt_block = gt_all[:, offset: offset + block_frames]
            distances = []
            for n in range(args_cli.samples):
                gen, _ = generate_block(model, codec, standardizer, args, history,
                                        text, mask, null_text, null_mask,
                                        args_cli.steps, args_cli.cfg,
                                        args_cli.seed + 1000 * b + n, args_cli.device,
                                        rig_history=rig_hist(start, offset))
                distances.append(fg_weighted_rgba_distance(gen, gt_block))
            teacher[b]["best"].append(min(distances))
            teacher[b]["mean"].append(float(np.mean(distances)))
        # free-running: model's own prefix from the GT start
        rollout = gt_all[:, :history_frames].clone()
        rig_state = rig_hist(start, history_frames)
        history_latents_count = int(args["history_max"])
        for b in range(args_cli.blocks):
            history = rollout[:, -history_frames:]
            gen, rig_block = generate_block(model, codec, standardizer, args, history,
                                            text, mask, null_text, null_mask,
                                            args_cli.steps, args_cli.cfg,
                                            args_cli.seed + 77_000 + b, args_cli.device,
                                            rig_history=rig_state)
            if rig_state is not None and rig_block is not None:
                rig_state = torch.cat((rig_state, rig_block), dim=1)[:, -history_latents_count:]
            offset = history_frames + b * block_frames
            free[b].append(fg_weighted_rgba_distance(gen, gt_all[:, offset: offset + block_frames]))
            rollout = torch.cat((rollout, gen), dim=1)

    # real floor: for prompts with >=2 clips, score clip B's block against clip A's
    floor = {b: [] for b in range(args_cli.blocks)}
    for prompt, gts in by_prompt_gt.items():
        if len(gts) < 2:
            continue
        for a in range(len(gts)):
            other = gts[(a + 1) % len(gts)]
            for b in range(args_cli.blocks):
                offset = history_frames + b * block_frames
                floor[b].append(fg_weighted_rgba_distance(
                    gts[a][:, offset: offset + block_frames],
                    other[:, offset: offset + block_frames]))

    def summary(store, key=None):
        return {str(b): float(np.mean(vals[key] if key else vals))
                for b, vals in store.items() if (vals[key] if key else vals)}

    report = {
        "ckpt": str(Path(args_cli.ckpt).resolve()),
        "protocol": checkpoint.get("protocol"),
        "clips": len(rows), "samples": args_cli.samples, "blocks": args_cli.blocks,
        "steps": args_cli.steps, "cfg": args_cli.cfg, "seed": args_cli.seed,
        "block_video_frames": block_frames, "history_video_frames": history_frames,
        "teacher_forced_best_of_n": summary(teacher, "best"),
        "teacher_forced_mean": summary(teacher, "mean"),
        "free_running_mean": summary(free),
        "real_floor_mean": summary(floor),
        "note": "fg-union weighted RGBA MSE vs the clip's actual continuation; "
                "v9 free-running carries its own generated rig history after block 0",
    }
    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("teacher_forced_best_of_n", "free_running_mean", "real_floor_mean")}, indent=2))


if __name__ == "__main__":
    main()
