"""Evaluate a checkpoint on frozen long-horizon videos.

    python -m eval.run_ckpt --run runs/a64AR --cache data/v1_cache --frames 50 --stride 2 --n 128

--watch: loop; re-evaluate whenever runs/<run>/ckpt.pt changes (its `step` field). Writes
runs/<run>/eval/<step>.json and TB scalars eval/{tvr,lie,cpe,mass_drift,fvd}.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
from train.video_ddpm import UNet3D, alphas_cumprod, rollout, sample
from train.video_dit_fm import VideoDiT, euler_sample, mixed_noise
from eval.oracle import score_seams, score_video
from eval.fvd import fvd, rgba_premult_to_rgb
from eval.protocol import build_reference_manifest, load_manifest_windows, save_manifest


def to_uint8_rgba(x):
    """[B,4,T,H,W] in [-1,1] premultiplied -> uint8 straight-ish RGBA [B,T,H,W,4] for the oracle
    (oracle uses alpha>127 as fg and rgb distance; un-premultiply where alpha>0)."""
    v = ((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 4, 1).cpu().numpy()   # [B,T,H,W,4]
    a = v[..., 3:4]
    rgb = np.where(a > 0.05, v[..., :3] / np.maximum(a, 0.05), 0.0)
    return (np.clip(np.concatenate([rgb, a], -1), 0, 1) * 255).astype(np.uint8), v


TEMPORAL = ("mass_drift", "centroid_speed", "centroid_accel", "motion_fraction", "angle_speed", "angle_jerk", "height_var",
            "ang_path_total")
FRAME = ("tvr", "lie", "cpe")
SEAM_METRICS = (
    "seam_centroid_speed", "within_centroid_speed",
    "seam_centroid_accel", "within_centroid_accel",
    "seam_angle_speed", "within_angle_speed",
    "seam_angle_jerk", "within_angle_jerk",
)


def boot_ci(vals, B=500, seed=0):
    r = np.random.RandomState(seed); v = np.asarray(vals, np.float64)
    if len(v) < 2: return (float(v.mean()), float(v.mean()))
    m = [r.choice(v, len(v), replace=True).mean() for _ in range(B)]
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


_reference_cache = {}


def rollout_chunks(context_frames, new_frames, target_frames):
    """Number of AR chunks required to produce at least ``target_frames``."""
    first = context_frames + new_frames
    return 1 if target_frames <= first else 1 + int(np.ceil((target_frames - first) / new_frames))


def ar_seam_frames(context_frames, new_frames, target_frames):
    """Destination-frame indices of true generated-chunk transitions."""
    first = context_frames + new_frames
    return list(range(first, target_frames, new_frames))


def real_reference(cache, frames, stride, S, n, dev, manifest=None):
    """Metrics and FVD for two frozen, source-motion-disjoint real sets."""
    if manifest is None:
        manifest = build_reference_manifest(cache, frames=frames, stride=stride, n_per_half=n, seed=1)
    manifest_key = json.dumps(manifest, sort_keys=True)
    key = (cache, frames, stride, S, n, manifest_key)
    if key in _reference_cache:
        return _reference_cache[key]
    if len(manifest["reference_a"]) != n or len(manifest["reference_b"]) != n:
        raise ValueError("manifest reference sizes do not match --n")
    a_ = load_manifest_windows(cache, manifest["reference_a"], size=S)
    b_ = load_manifest_windows(cache, manifest["reference_b"], size=S)
    rgba, _ = to_uint8_rgba(a_)
    per = [score_video(v) for v in rgba]
    limb_keys = tuple(sorted(k for k in per[0] if k.startswith("ang_path_")))
    ref = {k: float(np.mean([p[k] for p in per])) for k in FRAME + TEMPORAL + limb_keys}
    ra = rgba_premult_to_rgb(((a_.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 4, 1).numpy())
    rb = rgba_premult_to_rgb(((b_.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 4, 1).numpy())
    ref["fvd_real_real"] = fvd(ra, rb, device=dev)
    ref["_real_rgb"] = ra
    ref["_real_rgba"] = rgba
    ref["_manifest"] = manifest
    _reference_cache[key] = ref
    return ref


def evaluate(run, cache, n=128, dev="cuda", seeds=(0, 1, 2), target_frames=50,
             reference_stride=2, manifest=None, sample_steps=50, batch=8, text_cfg=3.0):
    ck = torch.load(os.path.join(run, "ckpt.pt"), map_location=dev)
    a = ck["args"]; step = ck["step"]
    n_cls = len(ck.get("groups", [])) if a.get("cond") == "group" else 0
    model_frames, model_size = a.get("frames", 16), a.get("size", 128)
    ar_context = a.get("ar_ctx", 0)
    is_dit = str(ck.get("arch", "")).startswith("dit_fm")
    is_text = a.get("cond") == "text"
    latent_meta = codec = None
    if str(ck.get("arch", "")).endswith("_latent"):
        latent_meta = json.load(open(os.path.join(a["cache"], "meta.json")))
        from scripts.encode_latent_cache import load_codec
        codec, _, _ = load_codec(latent_meta["codec_ckpt"], dev)
    in_ch = int(latent_meta["channels"]) if latent_meta else 4
    # References always load at pixel resolution; a latent model generates at
    # its latent grid and is decoded back to pixels before scoring.
    S = model_size * int(latent_meta["spatial_compression"]) if latent_meta else model_size
    text_dim = 0
    if is_text:
        from transformers import AutoConfig
        text_dim = AutoConfig.from_pretrained(a.get("text_encoder", "google-t5/t5-small")).d_model
    if is_dit:
        model = VideoDiT(size=model_size, frames=model_frames, patch=a.get("patch", 4), in_ch=in_ch, dim=a.get("dim", 384), depth=a.get("depth", 12), heads=a.get("heads", 6), n_classes=n_cls, cond_ch=5 if a.get("i2v_frac", 0) > 0 else 0, text_dim=text_dim, local_3d=bool(a.get("local_3d", False)),
                         full_st=bool(a.get("full_st", False))).to(dev)
    else:
        model = UNet3D(ch=a.get("ch", 64), n_classes=n_cls, size=S,
                      cond_ch=5 if a.get("ar_ctx", 0) > 0 else 0, text_dim=text_dim,
                      temporal_neighbors=int(a.get("temporal_neighbors", 0)),
                      temporal_pos_bias=bool(a.get("temporal_pos_bias", False))).to(dev)
    model.load_state_dict(ck["ema"]); model.eval()
    if is_dit:
        for b in model.blocks: b.t1_skip = bool(a.get("t1_skip", True))
    ac = alphas_cumprod().to(dev)
    ref = real_reference(cache, target_frames, reference_stride, S, n, dev, manifest=manifest)
    prompt_text = prompt_mask = null_text = null_mask = None
    prompts = None
    if is_text:
        clips = json.load(open(os.path.join(cache, "clips.json")))
        prompts = [clips[item["clip_id"]]["text"] for item in ref["_manifest"]["reference_a"]]
        from transformers import AutoTokenizer, T5EncoderModel
        name = a.get("text_encoder", "google-t5/t5-small")
        tokenizer = AutoTokenizer.from_pretrained(name)
        encoder = T5EncoderModel.from_pretrained(name).to(dev).eval().requires_grad_(False)
        tokens = tokenizer(prompts + [""], padding="max_length", truncation=True,
                           max_length=a.get("text_len", 32), return_tensors="pt")
        encoded = []
        for start in range(0, n + 1, batch):
            with torch.no_grad():
                encoded.append(encoder(input_ids=tokens.input_ids[start:start + batch].to(dev),
                                       attention_mask=tokens.attention_mask[start:start + batch].to(dev)).last_hidden_state.cpu())
        encoded = torch.cat(encoded).to(dev)
        prompt_text, prompt_mask = encoded[:-1], tokens.attention_mask[:-1].to(dev)
        null_text = encoded[-1:].expand(n, -1, -1)
        null_mask = tokens.attention_mask[-1:].to(dev).expand(n, -1)
        del encoder
    m = {
        "protocol_version": 2,
        "step": int(step),
        "n": int(n),
        "sampling_seeds": list(seeds),
        "target_frames": int(target_frames),
        "reference_stride": int(reference_stride),
        "sample_steps": int(sample_steps),
        "reference": {k: v for k, v in ref.items() if not k.startswith("_")},
        "manifest": ref["_manifest"],
    }
    if prompts is not None:
        m.update(conditioning="full_prompt", cfg=float(text_cfg), prompts=prompts)
    if latent_meta is not None:
        m.update(latent_codec=latent_meta["codec_ckpt"], latent_codec_sha256=latent_meta.get("codec_ckpt_sha256"),
                 latent_grid=[model_frames, model_size, model_size, in_ch],
                 note="generated latents are decoded with the frozen codec before scoring; codec reconstruction cost is included")
    per_seed = []
    seam_frames = ar_seam_frames(ar_context, model_frames, target_frames) if ar_context > 0 else []
    if seam_frames:
        ref_seams = [score_seams(v, seam_frames) for v in ref["_real_rgba"]]
        m["ar_seam_frames"] = seam_frames
        m["reference_seams"] = {
            key: float(np.nanmean([row[key] for row in ref_seams])) for key in SEAM_METRICS
        }
    for sd in seeds:
        g = torch.Generator(device=dev).manual_seed(1000 + sd)
        outs = []
        for i in range(0, n, batch):
            B = min(batch, n - i)
            ys = (torch.arange(B, device=dev) % n_cls) if n_cls else None
            with torch.no_grad():
                if latent_meta is not None:
                    noise = torch.randn((B, in_ch, model_frames, model_size, model_size), device=dev, generator=g)
                    z = euler_sample(model, noise.shape, dev, steps=sample_steps, y=ys,
                                     cfg=text_cfg if is_text else (2.0 if n_cls else 0.0),
                                     null_y=torch.full((B,), n_cls, device=dev) if n_cls else None,
                                     noise=noise, shift=a.get("shift", 1.0),
                                     text=prompt_text[i:i + B] if prompt_text is not None else None,
                                     text_mask=prompt_mask[i:i + B] if prompt_mask is not None else None,
                                     null_text=null_text[i:i + B] if null_text is not None else None,
                                     null_text_mask=null_mask[i:i + B] if null_mask is not None else None)
                    mean_t = torch.tensor(latent_meta["mean"], device=dev, dtype=z.dtype).view(1, -1, 1, 1, 1)
                    std_t = torch.tensor(latent_meta["std"], device=dev, dtype=z.dtype).view(1, -1, 1, 1, 1)
                    pre = codec.decode(z * std_t + mean_t, output_frames=target_frames, output_size=(S, S)).clamp(0, 1)
                    xs = pre * 2 - 1
                elif is_dit:
                    chunks = []
                    for _ in range(int(np.ceil(target_frames / model_frames))):
                        noise = mixed_noise((B, 4, model_frames, S, S), dev, a.get("noise_corr", 0.0), g)
                        chunks.append(euler_sample(model, noise.shape, dev, steps=sample_steps, y=ys,
                                      cfg=text_cfg if is_text else (2.0 if n_cls else 0.0),
                                      null_y=torch.full((B,), n_cls, device=dev) if n_cls else None,
                                      noise=noise, shift=a.get("shift", 1.0),
                                      text=prompt_text[i:i + B] if prompt_text is not None else None,
                                      text_mask=prompt_mask[i:i + B] if prompt_mask is not None else None,
                                      null_text=null_text[i:i + B] if null_text is not None else None,
                                      null_text_mask=null_mask[i:i + B] if null_mask is not None else None))
                    xs = torch.cat(chunks, 2)[:, :, :target_frames]
                elif ar_context > 0:
                    chunks = rollout_chunks(ar_context, model_frames, target_frames)
                    xs = rollout(model, B, ar_context, model_frames, chunks, ac, dev, steps=sample_steps,
                                 S=S, y=ys,
                                 null_y=torch.full((B,), n_cls, device=dev) if n_cls else None,
                                 generator=g,
                                 text=prompt_text[i:i + B] if prompt_text is not None else None,
                                 text_mask=prompt_mask[i:i + B] if prompt_mask is not None else None,
                                 null_text=null_text[i:i + B] if null_text is not None else None,
                                 null_text_mask=null_mask[i:i + B] if null_mask is not None else None,
                                 cfg=text_cfg if is_text else (2.0 if n_cls else 0.0))[:, :, :target_frames]
                else:
                    chunks = []
                    for _ in range(int(np.ceil(target_frames / model_frames))):
                        noise = torch.randn((B, 4, model_frames, S, S), device=dev, generator=g)
                        chunks.append(sample(model, noise.shape, ac, dev, steps=sample_steps, y=ys,
                                      cfg=text_cfg if is_text else (2.0 if n_cls else 0.0),
                                      null_y=torch.full((B,), n_cls, device=dev) if n_cls else None, noise=noise,
                                      text=prompt_text[i:i + B] if prompt_text is not None else None,
                                      text_mask=prompt_mask[i:i + B] if prompt_mask is not None else None,
                                      null_text=null_text[i:i + B] if null_text is not None else None,
                                      null_text_mask=null_mask[i:i + B] if null_mask is not None else None))
                    xs = torch.cat(chunks, 2)[:, :, :target_frames]
            outs.append(xs.cpu())
            print(
                f"eval seed {sd + 1}/{len(seeds)}: generated {min(i + B, n)}/{n} "
                f"videos ({target_frames} frames)", flush=True,
            )
        xs = torch.cat(outs, 0)
        rgba, prem = to_uint8_rgba(xs)
        per = [score_video(v) for v in rgba]
        limb_keys = tuple(sorted(k for k in per[0] if k.startswith("ang_path_")))
        r = {k: [p[k] for p in per] for k in FRAME + TEMPORAL + ("fg",) + limb_keys}
        if seam_frames:
            seam_rows = [score_seams(v, seam_frames) for v in rgba]
            for key in SEAM_METRICS:
                r[key] = [row[key] for row in seam_rows]
        print(f"eval seed {sd + 1}/{len(seeds)}: extracting FVD features", flush=True)
        r["fvd"] = fvd(ref["_real_rgb"], rgba_premult_to_rgb(prem), device=dev)
        print(f"eval seed {sd + 1}/{len(seeds)}: FVD {r['fvd']:.3f}", flush=True)
        per_seed.append(r)
    agg_keys = FRAME + TEMPORAL + ("fg",) + tuple(sorted(k for k in per_seed[0] if k.startswith("ang_path_")))
    for k in agg_keys:
        allv = [v for r in per_seed for v in r[k]]
        m[k] = float(np.mean(allv)); m[k + "_ci"] = boot_ci(allv)
    fv = [r["fvd"] for r in per_seed]
    m["fvd"] = float(np.mean(fv)); m["fvd_std"] = float(np.std(fv))
    m["dfvd"] = m["fvd"] - ref["fvd_real_real"]
    if seam_frames:
        for key in SEAM_METRICS:
            values = np.asarray([v for row in per_seed for v in row[key]], np.float64)
            values = values[np.isfinite(values)]
            m[key] = float(values.mean()) if values.size else float("nan")
            m[key + "_ci"] = boot_ci(values) if values.size else (float("nan"), float("nan"))
    del model; torch.cuda.empty_cache()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--watch", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--frames", type=int, default=50, help="canonical output length")
    ap.add_argument("--stride", type=int, default=2, help="real-reference temporal stride")
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=3.0, help="guidance for full-prompt UNet checkpoints")
    ap.add_argument("--manifest", default="", help="frozen reference manifest; created if absent")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.run, "eval"), exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(a.run, "tb_eval"))
    last = -1
    while True:
        p = os.path.join(a.run, "ckpt.pt")
        if os.path.exists(p):
            try:
                step = torch.load(p, map_location="cpu", weights_only=False)["step"]
            except Exception:
                step = last
            if step != last:
                snap = os.path.join(a.run, f"ckpt_{int(step):06d}.pt")
                if not os.path.exists(snap):
                    try:
                        ck = torch.load(p, map_location="cpu", weights_only=False)
                        torch.save({k: ck[k] for k in ck if k != "opt"}, snap)
                    except Exception as e:
                        print("snapshot failed:", e, flush=True)
                try:
                    manifest = json.load(open(a.manifest)) if a.manifest and os.path.exists(a.manifest) else None
                    m = evaluate(a.run, a.cache, a.n, seeds=tuple(range(a.seeds)), target_frames=a.frames,
                                 reference_stride=a.stride, manifest=manifest, sample_steps=a.sample_steps,
                                 batch=a.batch, text_cfg=a.cfg)
                    if a.manifest and not os.path.exists(a.manifest):
                        save_manifest(m["manifest"], a.manifest)
                    json.dump(m, open(os.path.join(a.run, "eval", f"{m['step']:06d}.json"), "w"), indent=1)
                    for k in FRAME + TEMPORAL + ("fvd",): tb.add_scalar(f"eval/{k}", m[k], m["step"])
                    if "dfvd" in m: tb.add_scalar("eval/dfvd", m["dfvd"], m["step"])
                    for k in FRAME + TEMPORAL:
                        tb.add_scalar(f"reference/{k}", m["reference"][k], m["step"])
                    tb.flush()
                    print(f"step {m['step']}: " + " ".join(f"{k} {m[k]:.3f}" for k in FRAME + TEMPORAL + ("fvd",)) +
                          (f" dfvd {m['dfvd']:.1f}" if "dfvd" in m else ""), flush=True)
                    last = m["step"]
                except Exception as e:
                    print("eval failed:", e, flush=True); time.sleep(60)
        if not a.watch: break
        time.sleep(120)


if __name__ == "__main__":
    main()
