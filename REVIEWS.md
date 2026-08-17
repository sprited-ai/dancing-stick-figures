# stickdance-128 — five-perspective review, merged (2026-08-17)

Reviews run by sub-agents on the state at commit `f24db8e`+: (1) code reviewer on `train/video_ddpm.py`,
(2) ordinary RTX 4090 hobbyist, (3) senior ML engineer on the whole project, (4) "Intermediate DL"
professor evaluating for a course, (5) Seedance-style caption/data practitioner. Full texts in the
session log; this file keeps the decisions and the ranked backlog.

## Consensus (what all five said, in different words)

1. **The trainer as shipped does not fit the target GPU** (54 GB @ batch 8). Need activation
   checkpointing, `--batch 4 --accum 2`, a real `--size` flag, and named presets. Print VRAM/ETA in
   the first 10 steps. Ship a **64 px** config natively rendered (stroke ≥ 2 px), not on-the-fly
   downsample; and a `mini` (~5k frames) and `color-only` config.
2. **The oracle exists only as a spec.** Implement `stickdance.eval`, regressor-free metrics first
   (LIE/BLD from colour masks, TVR via per-colour components), then a regressor with reported PCK.
   Validate: ≈0 on real held-out frames, high on 4 controlled corruptions (L/R colour swap, bone
   +30 %, delete a hand, add an arm), and correlate with FVD across checkpoints. Without this it's
   a tweet, not a metric. Don't claim "first" (Kubric/CLEVR-style oracles exist).
3. **SPARK.md is stale vs. code** (12 keyed styles / 19 joints / image DDPM v0 ≠ ARDY-only /
   cskel27 / 46 M video UNet). Rewrite the plan and the card to match `build.py`.
4. **Scale + prompt QA.** 465 clips is classification-sized; a 46 M video model memorises it.
   Motion+render are ~free → ~500 prompts × 10 seeds × 3 cams (~15 k clip-cams). Per-prompt contact
   sheet, `off_prompt` flag, re-caption from labels for mismatches, hold out groups ARDY can do.
5. **Captions ≠ ARDY prompts.** Generate captions deterministically from labels (view, screen
   direction, speed, facing, hand, posture, temporal phase), per-window with clip context,
   1 template + 3–4 LLM paraphrases, length mix 30/50/20. Rename `text` → `ardy_prompt`.
6. **Split leakage** across seeds of one prompt — **fixed** (`build.py` hashes the prompt slug).

## Disagreement to resolve (Jin's call)

- **Reference architecture.** Jin: "SDXL-class, not FLUX" (UNet + own f4 VAE + CFG). ML reviewer:
  in 2026 the hello-world should be **pixel-space DiT + flow matching** (SD3/Wan/CogVideoX/LTX
  lineage); UNet+VAE reads as legacy, own VAE is a week-long distraction. Claudia agrees with the
  reviewer: DiT-FM as the reference (shorter code, no ε/v/schedule zoo, Euler sampler); keep the
  UNet as Track A "classic" for comparison. Both trained, both on the card.

## Ranked backlog

| # | item | effort | source | status |
|---|---|---|---|---|
| 1 | Split by prompt slug | 1 h | ML | ✅ done, rebuilt |
| 2 | Trainer: `--grad_ckpt`, `--accum`, `--size`, presets (`4090-fast` 64²/8f/b16, `4090-full` 128²/16f/b4/ckpt/accum2, `runpod-96gb`), VRAM+ETA at step 10, `requirements.txt`, `worker_init_fn`, seed, LR cosine decay, val loss, DDIM eps-recompute after clamp | 4 h | code, 4090, ML | ☐ |
| 3 | 64 px native render config (`stickdance-64`), `mini`, `color-only`; `cache.py --size` | 3 h | 4090, prof | ☐ |
| 4 | Oracle v0 (regressor-free LIE/BLD/TVR) + corruption validation | 8 h | all | ☐ |
| 5 | Caption generator from labels + extra categorical labels (facing_rel_cam, screen_motion_dir, speed_bucket, posture, hands_above_head, foot contacts, gait_phase, motion_segments, prompt_match) | 6 h | caption | ☐ |
| 6 | Prompt QA contact sheets → `off_prompt`; move held-out to ARDY-capable groups | 3 h | ML, prof, caption | ☐ |
| 7 | Scale: LLM-expand to ~500 prompts w/ verb whitelist + round-trip validation; 10 seeds; 3 cams; `full` + `mini` | 8–12 h (compute) | ML | ☐ pending Jin |
| 8 | DiT + flow-matching pixel trainer as reference; UNet stays as classic | 8 h | ML | ☐ pending Jin |
| 9 | Rewrite SPARK.md + dataset card to match code; honest ARDY/OML provenance; no overclaims; stats + QA + baseline table + failures section | 4 h | all | ☐ |
| 10 | Course pack: A0 pose regression, A1 64px DDPM notebooks (Colab T4), baseline numbers | 6 h | prof | ☐ later |

## Things confirmed fine (don't relitigate)

Skip/attention bookkeeping in UNet3D; v-pred target + DDIM algebra; premultiplied RGBA in
[-1,1] as input; nearest-upsample + conv; zero-inits; bf16 autocast + fp32 master; depth/normal/
seg + cskel27 labels are worth shipping (as a separate config); camera distribution.

## Round 2 — mid-training review of run a0 @ steps 2000/4000 (2026-08-17)

Three personas: diffusion practitioner ("on track, normal-to-ahead; EMA lag hides early progress"),
animator/AD ("a figure, not yet animation — head/hips re-roll per frame; torso too heavy; feet not
on a ground plane"), skeptical ML reviewer ("nothing claimable yet; FVD n=64 uninterpretable; fixed
seeds, CIs, real-real floor, NN-to-train required").

Actions taken / queued:
- trainer (next run): fixed noise seeds for the sample grid; sample raw AND EMA early; 16 samples;
  EMA warmup 0.999→0.9995 @5k.
- eval watcher: n=256 videos, 3 sampling seeds, bootstrap 95 % CI over videos; FVD real-vs-real
  floor (split) → report ΔFVD; per-frame (tvr/lie/cpe) vs per-video (temporal) metrics separated.
- oracle: temporal metrics from the animator's checks — head/torso jitter across frames, per-colour
  limb-angle smoothness, figure-height variance; plus mass_drift.
- memorisation: nearest-neighbour-to-train grid per sample (todo).
- claims: no "consumer GPU" until a 4090 preset is measured; no "motion" until temporal metrics
  approach the real floor; FVD only as Δ over floor with CI.
- v1.1 render candidates (AD): ink stroke ≈ limb stroke, head +10 %, larger palette value gaps
  (magenta↔red-orange, green↔cyan), faint ground line. Decide after a1.

## Round 3 — loss/objective review (PhD-level agent, 2026-08-17)

Verdict: both objectives sound; the loss is just a bad proxy here (λ<−4 irreducible variance ≈ all of the
0.009). Keep A as classic (min-SNR would kill the *layout* regime our benchmark cares about; ZT-SNR a
footnote). B: `shift s` ≡ logit-normal μ=ln s; **shift 3 → promote if B1 wins** (b0 already runs it).
**Foreground weighting is biased** (weights depending on x0 tilt the posterior → halos/ghost limbs);
only (t, position) weights are valid. Alpha: premultiplied MSE ideal; ink lives only in α — log per-channel.
Video: frame-difference weighting valid; **PYoCo mixed noise (β≈0.5)** is the one trick with evidence for
per-frame re-roll. Diagnostics: fg-loss in 4 λ-buckets; fixed-grid val x̂0 fg-MSE + fg-IoU at stratified λ;
bg-haze + |Δ_t α| from samples. Ablations: A0/A1(min-SNR)/A2(w_bg=0.2, negative lesson)/B0(shift1)/B1(shift3)/
B2(shift3+mixed noise) — 6×7k steps ≈ one night. → E14.
