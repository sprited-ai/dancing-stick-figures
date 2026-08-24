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

With --sre CKPT the same generated blocks are also scored in RIG SPACE (the
third evaluation layer): SRE joints of the generated block vs the true
continuation's EXACT rig labels (px at output size), the label-space real
floor, SRE's own instrument noise on real renders, free-running bone-length
drift, and — for rig-co-generating models — self-consistency between SRE of
the generated pixels and the model's own rig tokens. Generation seeds and the
pixel metric are unchanged by --sre.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval.eval_m6 import build, build_text_cache
from train.latent_video_dit_ar import decode_full, encode_video
from train.video_dit_ar import euler_sample_block


# ------------------------------------------------------------- rig space (--sre)
@torch.no_grad()
def sre_joints(sre_model, rgba01: torch.Tensor, device) -> torch.Tensor:
    """[4,T,H,W] premultiplied [0,1] -> [T,27,2] normalized joints (cpu)."""
    return sre_model(rgba01.permute(1, 0, 2, 3).to(device)).cpu()


def joint_px(a: torch.Tensor, b: torch.Tensor, size: int) -> float:
    """Mean joint distance in px between two [T,27,2] normalized joint arrays."""
    return float((a - b).norm(dim=-1).mean() * size)


def bone_rel_error(pred: torch.Tensor, ref_lengths: torch.Tensor, parents) -> float:
    """Mean relative bone-length error of pred [T,27,2] against per-bone reference lengths [26]."""
    child = torch.arange(1, pred.shape[1])
    lengths = (pred[:, child] - pred[:, list(parents[1:])]).norm(dim=-1)      # [T,26]
    return float(((lengths - ref_lengths).abs() / ref_lengths.clamp_min(1e-6)).mean())


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
    parser.add_argument("--sre", default=None, help="SRE checkpoint: also score blocks in rig space")
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
    if hasattr(model, "rig_dim") or args_cli.sre:
        rig = np.load(cache / "rig.npy", mmap_mode="r")
    sre = None
    if args_cli.sre:
        from eval.sre_validate import load_model as load_sre
        from train.latent_video_dit_ar_rig import RIG_PARENTS
        sre, sre_size = load_sre(args_cli.sre, args_cli.device)
        if sre_size != size:
            raise SystemExit(f"SRE trained at {sre_size}, checkpoint decodes at {size}")

    prompts = [row["text"] for _, row in rows]
    text_cache = build_text_cache(prompts, args, args_cli.device)

    def rig_hist(row_start, offset):
        if rig is None:
            return None
        window = np.asarray(rig[row_start + offset - history_frames: row_start + offset], np.float32)
        tokens = (torch.from_numpy(window) * 2 - 1).reshape(
            1, int(args["history_max"]), temporal * 27 * 2)
        return tokens.to(args_cli.device)

    def own_rig_xy(rig_block_tokens):
        """[1,T_lat,temporal*27*2] in [-1,1] -> [T,27,2] normalized, cpu."""
        t_lat = rig_block_tokens.shape[1]
        return ((rig_block_tokens.cpu().float().reshape(t_lat * temporal, 27, 2) + 1) / 2)

    teacher = {b: {"best": [], "mean": []} for b in range(args_cli.blocks)}
    free = {b: [] for b in range(args_cli.blocks)}
    rig_teacher = {b: {"best": [], "mean": []} for b in range(args_cli.blocks)}
    rig_free = {b: [] for b in range(args_cli.blocks)}
    bone_free = {b: [] for b in range(args_cli.blocks)}
    sre_noise, selfcons_tf, selfcons_free = [], [], []
    by_prompt_gt, by_prompt_rig = {}, {}
    for (cid, row), prompt in zip(rows, prompts):
        start = int(row["start"])
        text, mask = text_cache[prompt]
        null_text, null_mask = text_cache[""]
        gt_all = to_video(np.asarray(frames[start:start + frames_needed]), size)
        by_prompt_gt.setdefault(prompt, []).append(gt_all)
        gt_rig_all = ref_bones = None
        if sre is not None:
            gt_rig_all = torch.from_numpy(
                np.asarray(rig[start:start + frames_needed], np.float32))
            by_prompt_rig.setdefault(prompt, []).append(gt_rig_all)
            child = torch.arange(1, 27)
            hist_len = (gt_rig_all[:history_frames, child]
                        - gt_rig_all[:history_frames, list(RIG_PARENTS[1:])]).norm(dim=-1)
            ref_bones = hist_len.mean(0)                                  # [26] exact-label reference
        # teacher-forced: GT prefix at every block position
        for b in range(args_cli.blocks):
            offset = history_frames + b * block_frames
            history = gt_all[:, offset - history_frames: offset]
            gt_block = gt_all[:, offset: offset + block_frames]
            gt_rig_block = None if sre is None else gt_rig_all[offset: offset + block_frames]
            if sre is not None:
                sre_noise.append(joint_px(sre_joints(sre, gt_block, args_cli.device),
                                          gt_rig_block, size))
            distances, rig_distances = [], []
            for n in range(args_cli.samples):
                gen, rig_block = generate_block(model, codec, standardizer, args, history,
                                                text, mask, null_text, null_mask,
                                                args_cli.steps, args_cli.cfg,
                                                args_cli.seed + 1000 * b + n, args_cli.device,
                                                rig_history=rig_hist(start, offset))
                distances.append(fg_weighted_rgba_distance(gen, gt_block))
                if sre is not None:
                    rj = sre_joints(sre, gen, args_cli.device)
                    rig_distances.append(joint_px(rj, gt_rig_block, size))
                    if rig_block is not None:
                        selfcons_tf.append(joint_px(rj, own_rig_xy(rig_block), size))
            teacher[b]["best"].append(min(distances))
            teacher[b]["mean"].append(float(np.mean(distances)))
            if rig_distances:
                rig_teacher[b]["best"].append(min(rig_distances))
                rig_teacher[b]["mean"].append(float(np.mean(rig_distances)))
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
            if sre is not None:
                rj = sre_joints(sre, gen, args_cli.device)
                rig_free[b].append(joint_px(rj, gt_rig_all[offset: offset + block_frames], size))
                bone_free[b].append(bone_rel_error(rj, ref_bones, RIG_PARENTS))
                if rig_block is not None:
                    selfcons_free.append(joint_px(rj, own_rig_xy(rig_block), size))
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
    if sre is not None:
        rig_floor = {b: [] for b in range(args_cli.blocks)}
        for prompt, gts in by_prompt_rig.items():
            if len(gts) < 2:
                continue
            for a in range(len(gts)):
                other = gts[(a + 1) % len(gts)]
                for b in range(args_cli.blocks):
                    offset = history_frames + b * block_frames
                    rig_floor[b].append(joint_px(gts[a][offset: offset + block_frames],
                                                 other[offset: offset + block_frames], size))
        report["rig_space"] = {
            "sre_ckpt": str(Path(args_cli.sre).resolve()),
            "teacher_forced_best_of_n_px": summary(rig_teacher, "best"),
            "teacher_forced_mean_px": summary(rig_teacher, "mean"),
            "free_running_mean_px": summary(rig_free),
            "real_floor_label_space_px": summary(rig_floor),
            "sre_noise_real_px": float(np.mean(sre_noise)),
            "bone_rel_error_free_running": summary(bone_free),
            "self_consistency_tf_px": float(np.mean(selfcons_tf)) if selfcons_tf else None,
            "self_consistency_free_px": float(np.mean(selfcons_free)) if selfcons_free else None,
            "note": "SRE joints of generated blocks vs EXACT rig labels of the true "
                    "continuation (mean joint distance, px at output size); the floor is "
                    "computed purely in label space; bone error is relative to the clip's "
                    "exact history bone lengths; self-consistency compares SRE of generated "
                    "pixels with the model's own rig tokens",
        }
    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    keys = ["teacher_forced_best_of_n", "free_running_mean", "real_floor_mean"]
    print(json.dumps({k: report[k] for k in keys}, indent=2))
    if "rig_space" in report:
        print(json.dumps({k: v for k, v in report["rig_space"].items()
                          if k not in ("sre_ckpt", "note")}, indent=2))


if __name__ == "__main__":
    main()
