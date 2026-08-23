"""Ad-hoc M6/v8 inference for a custom prompt list.

Loads a latent block-AR checkpoint (EMA weights), rolls out the canonical
five-second clip per prompt, and writes one labeled GIF per seed.  Sampling
matches the training previews: 10-step Euler, CFG 2, block rollout with the
checkpoint's own history/target geometry.
"""
import argparse
import json
from pathlib import Path

import torch

from train.latent_video_dit_ar import LatentStandardizer, decode_full, load_codec
from train.video_dit_ar import ARVideoDiT, FullSTARVideoDiT, rollout_blocks
from eval.post_eval_t2v import save_labeled_gif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--latent-stats", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompts", required=True, help="'|'-separated prompt list")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--rollout-latents", type=int, default=0,
                        help="override the checkpoint's rollout length (0 = use training default)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    saved = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    train_args = saved["args"]
    codec, standardizer, _, _ = load_codec(args.codec, args.latent_stats, args.device)
    model_class = FullSTARVideoDiT if train_args["attention_mode"] == "full" else ARVideoDiT
    latent_size = train_args["output_size"] // codec.spatial_compression
    model = model_class(
        size=latent_size, patch=train_args["patch"], in_ch=codec.latent_channels,
        dim=train_args["dim"], depth=train_args["depth"], heads=train_args["heads"],
        cond_ch=codec.latent_channels + 1, text_dim=512,
    ).to(args.device)
    model.load_state_dict(saved["ema"])
    model.eval()

    from transformers import AutoTokenizer, T5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(train_args["text_encoder"])
    encoder = T5EncoderModel.from_pretrained(train_args["text_encoder"]).to(args.device).eval().requires_grad_(False)

    def embed(prompts):
        tokens = tokenizer(prompts, padding="max_length", truncation=True,
                           max_length=train_args["text_len"], return_tensors="pt")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = encoder(input_ids=tokens.input_ids.to(args.device),
                             attention_mask=tokens.attention_mask.to(args.device)).last_hidden_state
        return hidden.float(), tokens.attention_mask.to(args.device)

    prompts = [p.strip() for p in args.prompts.split("|") if p.strip()]
    text, mask = embed(prompts)
    null_text, null_mask = embed([""] * len(prompts))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        generator = torch.Generator(device=args.device).manual_seed(20260821 + seed)
        with torch.no_grad():
            latent = rollout_blocks(
                model, [(text, mask)],
                total_frames=args.rollout_latents or train_args["rollout_latents"],
                target_frames=train_args["target_latents"], history_max=train_args["history_max"],
                steps=args.steps, null_text=null_text, null_mask=null_mask,
                cfg=args.cfg, shift=train_args["shift"], generator=generator, sample_clamp=None,
            )
            rgba = decode_full(codec, standardizer, latent,
                               output_size=train_args["output_size"]) * 2 - 1
        path = out / f"infer_seed{seed}.gif"
        save_labeled_gif(rgba.cpu(), str(path), prompts, fps=train_args["fps"])
        print("wrote", path)
    (out / "infer_manifest.json").write_text(json.dumps({
        "ckpt": str(Path(args.ckpt).resolve()), "step": saved.get("step"),
        "protocol": saved.get("protocol"), "prompts": prompts,
        "steps": args.steps, "cfg": args.cfg, "seeds": args.seeds,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
