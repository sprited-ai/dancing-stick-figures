# DRAFT — HF model card for the v9.0 reference release (2026-08-25)

Target repo: `sprited/dancing-stick-figures-baselines` (new subfolder `v9-rig-cogen/`).
Status: DRAFT — not uploaded. Jin decides release timing; numbers below are
EXPERIMENT_LOG-backed (2026-08-24/25 entries).

---

## Rig-Co-Generating Block-Autoregressive Video Model (v9.0)

The reference long-horizon model of the Dancing Stick Figures testbed: a
39.8M-parameter full-spatiotemporal latent DiT that autoregressively generates
$64^2$ RGBA video in 16-frame blocks at 20 fps (rollouts of 100+ frames), and
**co-generates its own 27-joint 2D skeleton** — one 216-dim rig token per
latent frame, denoised jointly with the pixels under a shared rectified flow.

**Why the rig channel:** at matched parameters, data, optimizer, and budget,
adding the co-generated rig improves pixel-space next-block prediction
(teacher-forced divergence .287 vs .314 for the pixel-only recipe at 100k;
free-running .440 vs .458), replicated across two training seeds. See the
companion paper for the full evaluation.

### Checkpoints

| file | steps | seed | notes |
|---|---|---|---|
| `v9_rig_cogen_100k_s0.pt` | 100k | 0 | primary reference (TVR .126, centroid speed .275) |
| `v9_rig_cogen_100k_s1.pt` | 100k | 1 | seed replication (TVR .129, speed .253) |

Requires the frozen codec checkpoint + latent stats (SHA-256-pinned, included).

### Training recipe

v8 base (16-frame generation blocks, motion-weighted flow loss α=1,
foreground latent weight 4, 1 s clean-history prefix, start-aligned windows)
+ rig tokens (rig flow loss weight 1). Varied-caption augmentation (a null
treatment by a matched study — kept for the released run's provenance).
100k steps, batch 16, lr 2e-4 cosine, bf16, single RTX PRO 6000; ~2 h.

### Honest limits

- **Semantic alignment is open.** Prompt-conditioned motion retrieval sits at
  chance for this and every tested model on the testbed; the model moves like
  the data but does not reliably obey the prompt. Every solo-model lever we
  tested (encoder scale, rig-loss strength, budget, caption variants) failed
  to move it.
- Residual motion gaps versus real references: height variance under-shoots
  (compound vertical motions), angular jerk ~1.5x real at 10-step sampling
  (partly sampler: 20 steps removes ~40%).
- Single visual domain (the testbed's renderer), $64^2$, orthographic cameras.
- The co-generated rig is faithful to the model's own pixels
  (self-consistency 0.76 px vs 0.32 px instrument noise via SRE) — it is a
  *description of the generation*, not a guarantee of correct motion.

### Not released as flagship

A render-coupled variant (v9.3-warmup: differentiable soft-capsule coupling,
5k warmup) achieves stronger structure (TVR .103) and near-perfect rig-pixel
binding (.994) but dampens motion to ~68% of real at the 100k budget; it is
documented in the paper as a coupling study, not shipped as the reference.
(Checkpoint available on request.)

### License / provenance

Code MIT, data CC0-1.0; source motions generated with NVIDIA ARDY under the
NVIDIA Open Model Agreement — downstream users must check fit for their use.
