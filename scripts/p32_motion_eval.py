"""Alpha-mask centroid-speed evaluation for the 32^2 pixel arms.

Same measurement as the real-reference computation (alpha>0.15 mask on
premultiplied RGBA), so generated and real numbers are directly comparable.
"""
import argparse, json
import numpy as np
import torch
from train.video_dit_ar import ARVideoDiT, rollout_blocks
from transformers import AutoTokenizer, T5EncoderModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16); ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=2.0); ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    args = ck["args"]
    model = ARVideoDiT(size=args["size"], patch=args["patch"], dim=args["dim"],
                       depth=args["depth"], heads=args["heads"]).to(a.device)
    model.load_state_dict(ck["ema"]); model.eval()
    tok = AutoTokenizer.from_pretrained(args["text_encoder"])
    enc = T5EncoderModel.from_pretrained(args["text_encoder"]).to(a.device).eval()
    prompts = ["A person walks forward.", "A person runs forward.", "A person does jumping jacks.",
               "A person waves hello with the left hand.", "A person does squats.",
               "A person dances energetically.", "A person balances on one leg.",
               "A person jogs in place."][: a.n]
    speeds = {}
    gen = torch.Generator(device=a.device).manual_seed(a.seed)
    with torch.no_grad():
        b = tok([""], return_tensors="pt", padding="max_length", truncation=True, max_length=args["text_len"])
        null = enc(input_ids=b.input_ids.to(a.device), attention_mask=b.attention_mask.to(a.device)).last_hidden_state
        null_mask = b.attention_mask.to(a.device).bool()
        for p in prompts:
            t = tok([p], return_tensors="pt", padding="max_length", truncation=True, max_length=args["text_len"])
            emb = enc(input_ids=t.input_ids.to(a.device), attention_mask=t.attention_mask.to(a.device)).last_hidden_state
            video = rollout_blocks(model, [(emb, t.attention_mask.to(a.device).bool())],
                                   total_frames=100, target_frames=args["target_frames"],
                                   history_max=args["history_max"], steps=a.steps,
                                   null_text=null, null_mask=null_mask, cfg=a.cfg, generator=gen)
            x = (video[0].permute(1, 2, 3, 0).float().cpu().numpy() + 1) / 2   # [T,H,W,4], straight? model outputs premultiplied-trained space
            cents = []
            for fr in x:
                mask = fr[..., 3] > 0.15
                ys, xs = np.nonzero(mask)
                if len(xs) < 5: continue
                cents.append((xs.mean(), ys.mean()))
            c = np.array(cents)
            speeds[p] = float(np.mean(np.linalg.norm(np.diff(c, axis=0), axis=1))) if len(c) > 10 else None
    valid = [v for v in speeds.values() if v is not None]
    report = {"ckpt": a.ckpt, "mean_centroid_speed": float(np.mean(valid)), "per_prompt": speeds,
              "measure": "alpha>0.15 mask centroid, same as real-reference computation (real 32^2 = 0.2477)"}
    json.dump(report, open(a.out, "w"), indent=2)
    print(json.dumps({"mean": report["mean_centroid_speed"]}, indent=0))

if __name__ == "__main__":
    main()
