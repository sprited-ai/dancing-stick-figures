# Seedance 1.0 (ByteDance Seed, arXiv 2506.09113, Jun 2025) — notes for Dancing Stick Figures

Read 2026-08-17. What transfers to us, what doesn't.

## Recipe facts
- VAE: temporally-causal 3D conv (MAGVIT-style), (r_t,r_h,r_w)=(4,16,16), C=48; L1+KL+LPIPS+GAN. Image = T'=0 case.
- DiT: decoupled spatial (MMDiT w/ text, window attn) and temporal (visual-only) blocks; MM-RoPE; QK-norm.
- Diffusion: flow matching, velocity prediction, logit-normal timesteps, resolution/duration-aware timestep shift.
- **Progressive training**: init from low-res T2I (256px) → image+video joint (256px, 3–12 s @12 fps) → 640px → 24 fps.
  Keep a small T2I fraction during video training. I2V 20% via channel-concat conditioning + frame masks.
- Captions: dense, dynamic (actions, camera) + static (appearance); PE model rewrites user prompts into caption format.
- Data: shot segmentation, overlay rectification, quality/safety filter, semantic dedup, distribution rebalancing.
- Post-training: SFT (curated, model merging), RLHF with 3 reward models (foundational VLM, motion, aesthetic), refiner RLHF; distillation (TSCD, RayFlow, APT).
- Eval: SeedVideoBench 300 prompts × taxonomy; human Motion Quality includes **structural accuracy: extra limbs, truncation,
  unnatural bending, inhuman postures** — human-rated.

## Transfers to us
1. Curriculum: T=1 image warm-up then video; mix single frames into video batches (a1).
2. DiT + FM + v-pred + logit-normal t = our Track B reference; timestep shift when we go 64→128 or 8f→16f.
3. Caption spec: dynamic+static split; normalise inference prompts into the template format (label→caption grammar).
4. Distribution rebalancing over prompt groups; dedup near-identical clips (seed collapse).
5. Framing line for the report: the structural-accuracy dimension they pay humans to judge is a function in our world.

## Doesn't transfer
RLHF, refiner cascade, distillation, infra (HSDP/Ulysses/MLAC), 1080p.
