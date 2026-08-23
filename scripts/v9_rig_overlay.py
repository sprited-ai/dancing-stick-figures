"""v9 verdict probe: roll out pixels+rig jointly, overlay the emitted rig on
the decoded frames, and score rig plausibility.

Outputs per prompt: overlay GIF (pixels with skeleton edges drawn from the
model's own rig output), plus a JSON report with (a) on-figure rate — fraction
of emitted joints landing on generated foreground alpha, (b) bone-length mean
and temporal jitter versus the ground-truth rig cache statistics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from train.latent_video_dit_ar import decode_full, load_codec
from train.latent_video_dit_ar_rig import (
    RIG_JOINTS, RigFullSTARVideoDiT, rollout_blocks_rig,
)

PARENTS = [-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 10, 10, 4, 13, 14, 15, 16, 16,
           0, 19, 20, 21, 0, 23, 24, 25]


def bone_lengths(rig01: np.ndarray) -> np.ndarray:
    """rig01 [T,27,2] in [0,1] -> [T,26] bone lengths (child order, root skipped)."""
    child = np.arange(1, RIG_JOINTS)
    parent = np.array(PARENTS[1:])
    return np.linalg.norm(rig01[:, child] - rig01[:, parent], axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--latent-stats", required=True)
    parser.add_argument("--cache", required=True, help="for GT bone-length reference (rig.npy)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompts", default="A person walks forward.|A person runs forward.|"
                        "A person waves hello with the left hand.|A person does squats.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--scale", type=int, default=4, help="overlay upscale factor")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    saved = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    ta = saved["args"]
    codec, standardizer, _, _ = load_codec(args.codec, args.latent_stats, args.device)
    model = RigFullSTARVideoDiT(
        temporal_compression=codec.temporal_compression, size=ta["output_size"] // codec.spatial_compression,
        patch=ta["patch"], in_ch=codec.latent_channels, dim=ta["dim"], depth=ta["depth"],
        heads=ta["heads"], cond_ch=codec.latent_channels + 1, text_dim=512,
    ).to(args.device)
    model.load_state_dict(saved["ema"]); model.eval()

    from transformers import AutoTokenizer, T5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(ta["text_encoder"])
    encoder = T5EncoderModel.from_pretrained(ta["text_encoder"]).to(args.device).eval().requires_grad_(False)

    def embed(prompts):
        tokens = tokenizer(prompts, padding="max_length", truncation=True,
                           max_length=ta["text_len"], return_tensors="pt")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = encoder(input_ids=tokens.input_ids.to(args.device),
                             attention_mask=tokens.attention_mask.to(args.device)).last_hidden_state
        return hidden.float(), tokens.attention_mask.to(args.device)

    prompts = [p.strip() for p in args.prompts.split("|") if p.strip()]
    text, mask = embed(prompts)
    null_text, null_mask = embed([""] * len(prompts))
    generator = torch.Generator(device=args.device).manual_seed(20260821)
    with torch.no_grad():
        latent, rig_tokens = rollout_blocks_rig(
            model, [(text, mask)], total_frames=ta["rollout_latents"],
            target_frames=ta["target_latents"], history_max=ta["history_max"],
            steps=args.steps, null_text=null_text, null_mask=null_mask,
            cfg=args.cfg, shift=ta["shift"], generator=generator,
        )
        rgba = decode_full(codec, standardizer, latent, output_size=ta["output_size"])
    temporal = codec.temporal_compression
    rig = ((rig_tokens.cpu().float() + 1) / 2).reshape(
        len(prompts), -1, temporal, RIG_JOINTS, 2).reshape(len(prompts), -1, RIG_JOINTS, 2).numpy()
    frames = rgba.cpu().float().numpy()          # [B,4,T,H,W] in [0,1]
    size = frames.shape[-1]

    gt = np.load(Path(args.cache) / "rig.npy", mmap_mode="r")
    picks = np.random.default_rng(0).choice(gt.shape[0], size=20000, replace=False)
    gt_bones = bone_lengths(np.asarray(gt[np.sort(picks)], np.float32))
    gt_mean = gt_bones.mean(0)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = {}
    for b, prompt in enumerate(prompts):
        images = []
        hits = total = 0
        for t in range(frames.shape[2]):
            rgb = (frames[b, :3, t].transpose(1, 2, 0) * 255).astype(np.uint8)
            alpha = frames[b, 3, t]
            img = Image.fromarray(rgb).resize((size*args.scale,)*2, Image.NEAREST)
            draw = ImageDraw.Draw(img)
            xy = rig[b, t] * size * args.scale
            for child in range(1, RIG_JOINTS):
                parent = PARENTS[child]
                draw.line([tuple(xy[parent]), tuple(xy[child])], fill=(0, 255, 0), width=1)
            for j in range(RIG_JOINTS):
                px = np.floor(rig[b, t, j] * size).astype(int)
                if (0 <= px).all() and (px < size).all():
                    total += 1
                    hits += int(alpha[px[1], px[0]] > 0.1)
            images.append(img)
        images[0].save(out / f"overlay_{b}.gif", save_all=True, append_images=images[1:],
                       duration=1000 // ta["fps"], loop=0, disposal=2)
        bones = bone_lengths(rig[b])
        report[prompt] = {
            "on_figure_rate": hits / max(total, 1),
            "bone_length_rel_error_vs_gt": float(np.abs(bones.mean(0) - gt_mean).mean() / gt_mean.mean()),
            "bone_length_temporal_jitter": float(bones.std(0).mean() / gt_mean.mean()),
        }
    (out / "rig_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
