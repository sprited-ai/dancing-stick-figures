# ARDY vs. Dancing Stick Figures M4

Status: design and literature note, 2026-08-21. This is not yet paper prose.

## Source

- Zhao et al., *ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation*, ACM TOG / SIGGRAPH 2026.
- Official paper: https://research.nvidia.com/labs/sil/projects/ardy/assets/ardy_paper.pdf
- Official code: https://github.com/nv-tlabs/ardy
- Local source PDF SHA-256: `d615b59e465da5a01371decf601b780c9af34c79e40a2dc24d36a488b2670c97`

## Executive verdict

ARDY is the right **streaming-control design reference** for M4, but it is not a
drop-in architecture baseline. ARDY generates compact 3D motion tokens. We generate
rendered RGBA pixels. That difference changes the dimensionality, inductive biases,
training scale, useful losses, evaluation metrics, and number of practical sampling
steps.

The defensible description is therefore:

> M4 adapts ARDY-like variable-history, windowed autoregressive diffusion to a small
> pixel-video teaching domain, while deliberately omitting ARDY's motion tokenizer,
> root/body factorization, and spatial-constraint machinery.

We must not describe M4 as “reproducing ARDY,” “ARDY for video,” or real-time until
the corresponding evidence exists.

## Side-by-side comparison

| Property | ARDY | Dancing Stick Figures |
|---|---|---|
| Output domain | 3D articulated human motion | 64×64 premultiplied RGBA rendered video |
| Native rate | 20 fps | M1–M3 evidence was sampled at 10 fps; M4 target is **20 fps** |
| Representation | Explicit global root plus compressed body-motion latent | Raw pixel patches; no learned video latent |
| Compression | Causal motion tokenizer; default FSQ, four frames per token | None; spatial 4×4 patches are continuous linear embeddings, not VQ tokens |
| Denoiser | Two interleaved transformers: root first, body second | One factorized spatial/temporal VideoDiT |
| Position | Sinusoidal temporal encoding on motion tokens | Learned spatial positions and, in M4 candidate A, signed temporal RoPE |
| Generation unit | Default 40 motion frames (2 seconds) per window | Candidate A predicts 10 frames (0.5 seconds at 20 fps) per block |
| History | Variable length, including zero; truncated at runtime | Candidate A trains with H∈{0,10,20,30,40}; runtime keeps the latest ≤40 frames |
| First window | Same model with zero history | Same candidate model with zero history |
| Current target block | Denoised jointly within its window | Denoised jointly within its block; it is not frame-by-frame AR |
| Text | LLM2Vec on Llama-3-8B-Instruct | Frozen T5-small |
| Prompt changes | Online controller triggers immediate replanning | Prompt can change at block boundaries; mid-block discard/replan is TODO |
| Future control | Sparse joint position/rotation goals, including beyond the current window | Text only; no explicit future spatial goals |
| Training target | DDPM clean-motion prediction plus decoded-motion, goal, and FK consistency losses | Rectified-flow velocity prediction in pixel space |
| Teacher forcing | Ground-truth history during training; generated history at deployment | Same basic train/test distinction in candidate A |
| Sampling | Default 10 DDPM steps; 4 is usable in its compressed motion space | Candidate target is 20 Euler steps per block; 4/10/20 is unverified |
| Scale | ~156M deployed parameters; denoiser trained 1M steps with batch 512 on 4×A100-80GB | Small educational model and dataset; exact final M4 scale/result still pending |
| Data | Bones Rigplay: about 700 hours, 315k/35k clips; also HumanML3D | Small synthetic, fully traced rendered dataset |
| Evaluation | Motion FID, TMR R-precision, foot skating, constraint errors, human study | Pixel/video distribution, color-part trajectories, temporal structure, prompt tests |
| Known failures | Foot skating, jitter, compute cost, extreme-history inefficiency | Jitter, repetition/buffering, error accumulation, weak prompt completion, raw-pixel cost |

## What M4 candidate A already borrows successfully

1. **Windowed autoregression.** It predicts a future block rather than one frame at a
   time.
2. **Variable clean history.** Ground-truth history is supplied during training,
   including the H=0 first-window case.
3. **Target-only corruption and loss.** History is context; only the new block is
   diffused and scored.
4. **Truncated sliding continuation.** Generated frames can be appended indefinitely
   while retaining a bounded recent history.
5. **Per-window text conditioning.** A later block can receive a new prompt.
6. **Causal boundary with intra-block cooperation.** Target tokens cannot rewrite the
   clean history, while target frames can interact bidirectionally with one another.

These points make M4 meaningfully closer to ARDY than M3, whose whole 50-frame clip is
generated bidirectionally in one shot.

## Important deviations

### 1. Pixels instead of motion tokens

ARDY's four-frame token is a learned, low-dimensional body-motion representation. Our
4×4 “patch” is merely a continuous projection of local pixels. It does not provide the
same temporal compression or kinematic prior. This is the largest reason ARDY's speed
and few-step results do not transfer directly.

### 2. No root/body decomposition

ARDY explicitly separates globally meaningful root motion from body articulation and
predicts root first. Our model must discover body position, articulation, visibility,
and appearance jointly in pixels. The color-coded body parts help evaluation, but do
not give the generator ARDY's structural representation.

### 3. No explicit physics or goal constraints

ARDY can overwrite sparse root/joint constraints and trains with decoded-motion and
forward-kinematics consistency. Candidate A has no skeleton output, foot-contact loss,
or future waypoint input. Pixel trajectory metrics diagnose failures; they do not fix
them during training.

### 4. Different diffusion formulation

ARDY uses DDPM-style clean-motion prediction. M4 uses rectified flow and Euler
integration. A shared number such as “10 steps” therefore does not imply comparable
quality or compute.

### 5. Prompt switching is less complete

ARDY's interactive controller can discard an obsolete predicted future and replan from
the current played state, using a small playback buffer to hide latency. Candidate A
can change prompts between blocks, but does not yet implement asynchronous mid-block
replanning or a latency buffer.

### 6. ARDY does not have an explicit action countdown

Its model receives the active prompt, motion history, diffusion step, and optional
future constraints. The interactive application manages when prompts become active.
Long history helps the model infer whether a non-cyclic action has already occurred;
there is no model-side literal `once` timer described in the paper. We should likewise
evaluate general finite-action completion rather than hard-code the word “once.”

## What to borrow now

1. Keep variable history, including H=0, in a single M4 model.
2. Measure first-window generation separately from continuation.
3. Compare ground-truth-history continuation with generated-history rollout to expose
   teacher-forcing distribution shift.
4. Keep a truncated sliding history and record the exact retained horizon.
5. Condition every new block on the currently active prompt.
6. Add controller-level future discard/replan for prompt changes after the basic model
   is validated.
7. Test finite, non-cyclic instructions such as “sit down,” “turn around,” and “wave,
   then rest,” measuring unwanted repetition rather than parsing a magic keyword.
8. Ablate generation horizon and sampling steps instead of assuming shorter is better.

## What not to borrow yet

- A custom FSQ/AE/VAE tokenizer: potentially valuable, but it would turn M4 into a
  separate representation-learning project and weaken the simple pixel-space ladder.
- Root/body two-stage prediction: valuable only after we intentionally expose a motion
  representation to the model.
- Sparse 3D joint constraints and FK losses: outside the current pixel-only baseline.
- Llama-3-8B/LLM2Vec text conditioning: too costly for the educational thesis, and not
  yet shown necessary for this limited prompt vocabulary.
- ARDY's million-step, multi-A100 recipe: inappropriate as an accessibility baseline.
- A forced switch from rectified flow to DDPM solely to resemble ARDY.

## Experiments motivated by ARDY

These are proposed experiments, not completed results.

### E1 — History and teacher-forcing gap (highest priority)

- Evaluate H={0,10,20,40} at 20 fps.
- Report first-window, ground-truth-history continuation, and generated-history rollout
  separately.
- Track degradation by block index over a 100-frame (5-second) rollout.

### E2 — Generation horizon

- Compare F=10 (0.5 s) with F=20 (1.0 s) while holding data and compute reporting
  fixed.
- ARDY's results warn that an extremely short horizon can drift or ignore text, whereas
  a long horizon reacts more slowly. We must measure this trade-off in pixels.

### E3 — Sampling steps

- Evaluate 4/10/20 Euler steps only after a checkpoint has learned recognizable motion.
- Report latency together with quality; do not claim ARDY-equivalent few-step behavior.

### E4 — Prompt replacement

- At a fixed frame, replace prompts such as `walking → sitting` and `running → idle`.
- Measure response latency, discontinuity at the switch, and adherence before/after.
- Compare block-boundary switching first; controller-level mid-block replanning follows.

### E5 — Finite-action repetition

- Prompts: “wave and lower the arm,” “turn around and stand,” “sit down and remain
  seated.”
- Measure whether the action terminates, how often it repeats, and whether the terminal
  pose persists. This tests the underlying problem more honestly than a literal `once`
  token.

### E6 — Temporal position encoding (secondary)

- Candidate A's signed RoPE anchors the new block at positions 0…F−1 and history at
  negative offsets, which naturally survives sliding-window continuation.
- Compare it with a relative-bias or sinusoidal baseline only if continuation failures
  implicate position encoding. ARDY's sinusoidal encoding is evidence that fixed learned
  absolute length is unnecessary, not evidence that RoPE must win.

## Metric implications

ARDY's motion FID, TMR R-precision, foot-skating, and 3D constraint errors cannot be
reported directly on our rendered videos. Our color-part centroids provide a useful
low-cost substitute for selected motion diagnostics:

- joint/part visibility and disappearance,
- centroid velocity, acceleration, and jerk,
- bone-length and relative-position consistency where colored parts are visible,
- freeze and duplicate-window detection,
- action repetition/autocorrelation,
- boundary discontinuity between generated blocks,
- prompt-switch response latency.

These metrics remain an approximation. Occlusion and overlapping colors can hide or
shift visible centroids, so they need controlled-corruption validation and real-data
floors before being treated as a benchmark.

## Claims we may and may not make

Defensible after successful experiments:

- “an ARDY-inspired streaming pixel-video baseline”
- “variable-history block-autoregressive rectified-flow generation”
- “a small teaching implementation of prompt-conditioned sliding-window generation”

Not defensible now:

- “we reproduce ARDY”
- “ARDY for video”
- “real-time generation”
- “four-step sampling works”
- “prompt replacement is solved”
- direct performance comparison to ARDY's latency or motion metrics

## Immediate M4 decision

Finish and inspect the current 1k candidate-A pilot rather than redesign it mid-run.
Display stride-1 outputs at 20 fps. Candidate A should advance only if it produces
recognizable motion and passes basic first-window/continuation checks. Before a longer
paper run, freeze the 20-fps protocol, add generated-history evaluation, and preserve
the horizon, sampling-step, prompt-switch, and repetition experiments above as explicit
gates.


## M6 addendum (2026-08-22, Claudia) — techniques for the motion-collapse fix

Verified in the ARDY source (`gin:~/dev/ardy/ardy/model/backbone.py`,
`auto_latent_twostage_denoiser.py`):

1. **Boundary-anchored positional encoding.** Mode
   `learned_prefix_zero_at_first_generation`: motion tokens receive a sinusoidal
   encoding (`PositionalEncodingNegativeIndex`) whose index is
   `arange(num_tokens) - history_len_tokens` — index 0 sits at the *first
   generation token*, history tokens carry negative indices. Every token
   therefore knows its absolute distance from the generation boundary, and the
   encoding is consistent across rollout windows. Our M6 uses purely relative
   signed RoPE, which cannot express this. Porting this additive PE is fix
   candidate C' (model change; after the 2k pilot wave).
2. **Long variable history is ARDY's answer to action-completion ambiguity.**
   No clip-absolute time and no "once" timer: the model infers whether a
   non-cyclic action already happened from the history itself. Our 1.0 s latent
   history cannot contain a 1-3 s action; hence the resting-context ambiguity
   that plausibly drives the collapse. Fix pilot E (`hist24`, 2.4 s history) is
   the direct port, declared in
   `configs/m6_protocol_v3_start_aligned_h8_hist24_pilot.json`.
3. Not ported now (unchanged from the M4 verdict): motion tokenizer, root/body
   two-stage, FK/goal losses, LLM2Vec text, million-step recipe.
