# Dancing Stick Figures — research roadmap

## Fixed question

Given a natural-language motion prompt, generate a **64×64, 50-frame, 10 fps** video.

Every serious comparison uses the same prompt-disjoint split, data, frame count, seed list, training-example budget, sampler,
and evaluation prompts. The 16×16 and 32×32 runs are engineering/teaching exercises, not quality results.

## Architecture ladder

### K0 — archived 3D U-Net

Status: historical infrastructure run, not the official T2V baseline.

- [x] Train image and chunk-autoregressive U-Net variants.
- [x] Learn checkpointing, rollout, and failure-evaluation infrastructure.
- [ ] Report only as project history unless it receives the same text condition and budget as K1/K2.

### K1 — bidirectional full-clip Video DiT (current)

One model sees all 50 frames together. Spatial and temporal attention are factorised. Text comes from frozen T5-small tokens
through cross-attention. The model predicts rectified-flow velocity in pixel space.

- [x] Implement token-level text conditioning and classifier-free guidance.
- [x] Pass 16×16, 32×32, and 64×64 end-to-end smoke tests.
- [ ] Finish the 4090 batch/throughput benchmark.
- [x] Run a 3,000-step scratch pilot with samples every 500 steps. This saved
      `stride=1` and is therefore a 20-fps engineering pilot, not a result on
      the fixed 10-fps track.
- [ ] Run the canonical image-pretrained K1 pilot with `stride=2`.
- [ ] Test whether changing only the prompt changes the generated motion.
- [ ] Decide go/no-go for a longer K1 run.

Success gate: recognisable figures, non-static motion, and measurable prompt sensitivity on held-out prompts.
Stop gate: collapse, prompt-insensitive motion, or no useful improvement across the last two sample milestones.

### K2 — chunk-causal Video DiT

Generate a first chunk, then generate later chunks from prior generated context. Keep the DiT width, text encoder, objective,
data, and training-example budget matched to K1 wherever possible.

- [ ] Add block-causal temporal attention and an explicit clean/noised context interface.
- [ ] Train 8-frame chunks and roll out exactly 50 frames.
- [ ] Measure boundary seams separately from within-chunk motion.
- [ ] Compare drift, memory, speed, and prompt adherence with K1.

Research question: does causal chunking buy efficient long-horizon generation without unacceptable seams or drift?

### S0/S1 — modular motion-first generation

Before building a joint model, separate motion generation from rendering. This creates interpretable upper bounds and shows
whether joint training is actually necessary.

```text
S0: text → learned motion diffusion → existing deterministic stick renderer
S1: text → learned motion diffusion → learned skeleton-conditioned video renderer
```

- [ ] Train a small text-to-motion diffusion model on figure-frame `[50,27,3]` joints, root motion, and contacts.
- [ ] S0: render predicted motion with the exact released renderer; attribute failures to motion generation.
- [ ] S1-upper: train/evaluate the learned renderer on ground-truth motion; attribute failures to rendering.
- [ ] S1-rollout: feed predicted motion to the learned renderer; measure pipeline distribution shift/error propagation.
- [ ] Reuse identical motion samples across renderers so appearance and motion errors remain separable.

For the current stick figures, the exact renderer makes S0 the rational structured baseline; learning the same renderer is
deliberately redundant. S1 becomes more meaningful for the paired chibi domain or renderer variants where deterministic
pixels are not the final goal.

Research question: how much of text-to-video difficulty comes from motion generation versus mapping known motion to pixels?

### M1 — joint skeleton-and-video generation

Generate an explicit frame-aligned 3D skeleton together with the video. The skeleton provides an occlusion-independent
structural target; a differentiable renderer couples that target back to pixels so the model cannot output a valid skeleton
and an unrelated video.

```text
text → shared generative model → 3D joints/root/contact → soft differentiable renderer ┐
                              └→ video tokens -----------------------------------------┴→ consistency
```

- [ ] Start with figure-frame joints `[50,27,3]`, root trajectory/heading, and four foot-contact bits.
- [ ] Do not initially predict raw 3×3 rotation matrices; add a valid rotation representation only if needed.
- [ ] Implement a PyTorch soft capsule renderer with differentiable projection, soft coverage, and depth-aware visibility.
- [ ] Condition projection on the released camera/body parameters.
- [ ] Add skeleton denoising plus bone-length, joint-range, velocity/acceleration, and contact objectives.
- [ ] Add physics-inspired kinematic losses: bone rigidity, floor penetration, contact-conditioned foot skating/hover,
      angular velocity, acceleration, and jerk.
- [ ] Validate every physics loss on controlled corruptions before using it for training.
- [ ] Use train-split motion quantiles and contact masks; do not penalise legitimate jumps or fast dance motion globally.
- [ ] Compare the soft-rendered RGBA/parts/depth with the ground-truth buffers.
- [ ] Require generated-video ↔ generated-skeleton render consistency to prevent a disconnected auxiliary output.
- [ ] Validate that skeleton-space scores remain correct under controlled visual occlusion.

Candidate objective:

```text
L = L_video_flow + λm L_motion_flow + λb L_bone + λj L_joint_range
  + λt L_velocity/acceleration/jerk + λc L_contact
  + λp L_penetration/foot_skating + λr L_soft_render_reconstruction
```

These are physics-inspired kinematic constraints, not a full dynamics simulator: the release has joints and contacts but
does not provide mass, inertia, forces, or torques. Centre-of-mass/support-polygon losses are optional approximations and
must not be described as physical validity without additional body parameters.

Research question: can privileged trace supervision improve structural validity and make failure measurement robust to
occlusion without reducing the model to a disconnected pose predictor?

Joint training is not assumed to be better. Compare it with S0/S1 to test whether shared training reduces pipeline error and
improves pixel learning, or instead sacrifices the modular system's editability, diagnosis, and clean supervision.

Important control: rendering a generated skeleton with the original deterministic renderer is a text-to-motion-plus-renderer
system, not a learned video generator. Report it as a useful structured baseline; retain the video branch when claiming joint
video generation.

### W1 — official Wan family transfer (deferred)

Decision: **do not run this track in the current 64×64 project.** It is retained as a future high-resolution transfer idea,
not part of the immediate experiment queue. The official models' 480p/720p training distributions and strong VAE spatial
compression would make the result primarily a resolution-mismatch test rather than a clean architecture comparison.

| Track | Official model | Native setting | Decision |
|---|---|---|---|
| W1a | Wan2.1-T2V-1.3B | 480p; VAE stride 4×8×8 | zero-shot reconstruction/inference gate |
| W1b | Wan2.1-T2V-1.3B LoRA | upscaled/letterboxed training | defer until a higher-resolution dataset exists |
| W2a | Wan2.2-TI2V-5B | 720p; VAE stride 4×16×16 | modern 4090 zero-shot reference |
| W2b | Wan2.2-TI2V-5B LoRA | upscaled/letterboxed training | only if the VAE gate preserves structure |
| W3 | Wan2.1-T2V-14B | 480p/720p | defer: redundant cost for this exercise |
| W4 | Wan2.2-T2V-A14B | 480p/720p; 27B total/14B active | defer: official single-GPU path needs ≥80GB |

Wan2.1 uses spatial VAE stride 8, while Wan2.2 TI2V-5B uses stride 16. At native 64×64 these would leave only 8×8 and 4×4
latent grids respectively, so direct low-resolution fine-tuning is a severe resolution/distribution mismatch.

- [ ] Reconsider only after a higher-resolution character dataset exists.
- [ ] If reconsidered, begin with VAE reconstruction—not expensive fine-tuning.
- [ ] Define a reversible 64×64 → 480×832 letterbox/upscale transform for our data.
- [ ] Compare Wan2.1 and Wan2.2 VAE reconstructions of thin limbs before any LoRA training.
- [ ] Run a small Wan2.1-1.3B LoRA feasibility experiment; do not full-fine-tune before this gate passes.
- [ ] Attempt Wan2.2-5B LoRA only if its higher-compression VAE passes the reconstruction gate.
- [ ] Preserve native outputs and also downsample them to 64×64 for shared metrics.
- [ ] Report external pretraining data/model scale and extra compute separately from K1/K2.
- [ ] Stop if the VAE destroys thin-limb colours or LoRA learns appearance without prompt-sensitive motion.

Research question: does large-scale open video pretraining transfer usefully to a tiny synthetic motion domain despite the
large resolution and representation mismatch?

### K3 — Diffusion-Forcing causal DiT

Give different temporal chunks different noise levels so the model learns prediction, interpolation, and correction within one
causal formulation.

- [ ] Implement only after K2 exposes a measurable rollout/seam problem.
- [ ] Keep K2 architecture fixed and change the training/noise schedule only.
- [ ] Compare against K2 at a matched training-example and sampling budget.

Research question: does asynchronous temporal noise make the causal model more robust to its own imperfect history?

### K4 — Self-Forcing training

Train with some context produced by the model itself rather than only ground-truth context.

- [ ] Implement only if K2/K3 show exposure-bias drift.
- [ ] Define a reproducible generated-context schedule.
- [ ] Measure quality gain against added compute cost.

Research question: does training on self-generated context reduce long-rollout drift enough to justify its cost?

### O1 — our first evidence-driven modification

O1 is intentionally undefined today.

- [ ] Select one repeated, measured failure shared by or separating K1–K4.
- [ ] Write one causal hypothesis before changing code.
- [ ] Change one mechanism only.
- [ ] Run a matched ablation and accept or reject the hypothesis.

O1 is not “combine every modern trick.” It is the smallest change supported by our own evidence.

## Shared evaluation work

- [ ] Freeze and hash prompt-disjoint train/validation/test manifests.
- [ ] Freeze 16–32 held-out evaluation prompts and sampling seeds.
- [ ] Save prompt/output pairs at every sample milestone.
- [ ] Add a prompt-swap test: same noise, different prompt.
- [ ] Measure anatomy/structure, temporal motion, drift, and FVD.
- [ ] For causal models, measure seam transitions separately.
- [ ] Report parameters, peak VRAM, examples/second, GPU-hours, and dollars.
- [ ] Repeat promising final comparisons across at least three seeds.

## Educational deliverables

- [ ] One notebook: data → text encoding → noising → one training step → sampling.
- [ ] One notebook: compare K1 and K2 failure modes on identical prompts.
- [ ] Architecture diagrams showing full-clip versus chunk-causal generation.
- [ ] A failure journal: observation → hypothesis → experiment → result → next decision.
- [ ] Small pretrained checkpoints so a student can evaluate without paying for training.

## Immediate queue

1. Finish the 64×64 K1 throughput test on RTX 4090.
2. Launch the K1 3,000-step pilot.
3. Inspect fixed-prompt samples at 500, 1,000, 2,000, and 3,000 steps.
4. Run same-noise/different-prompt tests.
5. Make the K1 go/no-go decision.
6. Build the smaller S0 text-to-motion + exact-renderer control.
7. Use S0 to decide whether an S1 learned renderer and M1 joint model answer a real observed failure.
8. Choose M1 or K2 next based on whether K1's dominant failure is anatomy/occlusion or long-horizon temporal drift.
   Wan transfer remains deferred because it does not cleanly answer the current 64×64 question.

## Primary design references

- Wan: <https://arxiv.org/abs/2503.20314>
- MAGI-1: <https://arxiv.org/abs/2505.13211>
- Seedance 1.0: <https://arxiv.org/abs/2506.09113>
- Diffusion Forcing: <https://arxiv.org/abs/2407.01392>
- Self Forcing: <https://arxiv.org/abs/2506.08009>
