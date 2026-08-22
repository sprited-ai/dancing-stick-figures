# ARDY — Distilled Essence

Source: Zhao, Petrovich, Zhang, Wang, Tang, Rempe. *ARDY: Autoregressive Diffusion
with Hybrid Representation for Interactive Human Motion Generation.* ACM TOG 45(4),
Art. 86, SIGGRAPH/July 2026. Distilled 2026-08-22 from the complete extracted text
(`paper/refs/ardy_paper_full.md`, lines 1-962, abstract through references; the
extraction has no appendix — the published paper is 14 pages and ends at Sec. 8 +
references). Section/table citations below refer to the paper. Companion note:
`paper/refs/ardy_vs_dsf_m4.md` (side-by-side with our M4/M6); this file deepens it
with exact recipe numbers and an M6-freeze portability analysis, and avoids
re-listing what that note already tables.

## 1. One-paragraph thesis

ARDY is a streaming (block-autoregressive) motion diffusion model that generates 2 s
windows of 3D human motion in ~33-63 ms, conditioned on online text prompts and
arbitrarily sparse kinematic constraints that may lie beyond the current window
(Abstract; Sec. 1). It works because of three interlocking choices: (a) a **hybrid
representation** — explicit global root features concatenated with a learned latent
body embedding — keeps the globally-meaningful, constraint-facing part of motion
directly overwritable while compressing the high-dimensional body articulation into
something a few-step denoiser can actually model (Sec. 3.1; Tab. 2 shows the
fully-explicit alternative is 2-8x worse); (b) a **two-stage denoiser** predicts
clean root first, then body conditioned on that root, inside every denoising step,
which buys precise spatial control without hurting text-only fidelity (Sec. 3.4;
Tab. 2); (c) **variable-length history training** (H drawn from 0 up to 8 s by
random window placement in ≤10 s clips) plus **long-horizon goal conditioning**
gives the model enough context to disambiguate whether a non-cyclic action has
already happened and enough foresight to start moving toward far-future goals
natively, with no RL policy or test-time optimization (Sec. 3.3, 3.5; Tab. 1,
Tab. 5).

## 2. Hybrid representation

- **Explicit per-frame pose** (Sec. 3.1, Eq. 1): `m = (m_root, m_body)`.
  - `m_root = (p, cos ψ, sin ψ) ∈ R^5`: global 3D root position + heading angle.
    Root is *global-coordinate*, not velocity-integrated.
  - `m_body = (θ, J, J̇, c)`: 6D global joint rotations (R^{6j}, root included),
    non-root joint positions minus planar root (R^{3j−3}), global joint velocities
    (R^{3j}), binary foot-contact labels (R^4). Skeleton: 27 joints on Bones
    Rigplay (Sec. 5.1).
- **Hybrid pose** (Eq. 2-3): body part replaced by tokenizer latent
  `x_body ∈ R^L`; per-token hybrid `x^{1:T} = [m_root^{1:T}; x_body^{1:T}] ∈
  R^{T×D}`, `D = L + 5P`. Default `L = 128`, `P = 4` ⇒ D = 148.
- **Frames per token**: patch size `P = 4` frames at 20 fps = 0.2 s/token
  (Sec. 3.5). A 2 s generation window = 10 tokens.
- **Tokenizer** (Sec. 3.2, 3.5; Fig. 2): asymmetric conditional autoencoder.
  Encoder and decoder are 8-layer *causal* transformers, width 512. Encoder sees
  patchified explicit body motion only. Decoder reconstructs body motion from the
  hybrid tokens, but with the root converted **global→local** first — decoder-side
  root is `(ψ̇, ṗ_x, ṗ_z, p_y)` (heading angular velocity, planar linear velocity,
  height). This conversion exists solely to fight foot skating (Sec. 3.2; ablated
  in Tab. 2: skipping it raises text-only skate 0.264→0.303 m/s and waypoint error
  0.024→0.060 m).
- **Quantization**: FSQ on the encoder output, 64 discrete levels per feature,
  128 dims ("FSQ 64-128"); quantized vectors used directly as the latent
  (Sec. 3.5). VAE (KL weight 1e-6) and vanilla AE perform comparably (Tab. 3,
  tokenizer-type block), but the vanilla AE **diverges** when trained at 40-frame
  horizons; FSQ is chosen for training stability, not quality (Sec. 5.3).
- **Why root stays explicit** (Sec. 3.1): (i) global coordinates avoid compounding
  error from integrating local velocities; (ii) sparse spatial constraints live in
  global scene space, and an explicit root can be *directly overwritten* by them;
  (iii) only the body — the high-dimensional part — needs compression for
  generative efficiency.
- Tokenizer training: clips of 1-10 s; reconstruction loss + foot-skating loss
  (Eq. 6: contact-weighted mean predicted foot-joint velocity, weight **0.01**);
  AdamAtan2, lr 2e-5, batch 128, cosine schedule with 10k linear warmup, **4M
  steps**, 1×A100-80GB (Sec. 3.5).

## 3. Denoiser architecture

- **Problem form** (Sec. 3.3, Eq. 4): generate next `C` hybrid tokens given text
  `s`, history `x^{(−H+1):0}` (H variable, 0..max), and goals `g^{1:(C+F)}` —
  goals both inside the window (1..C) and beyond it (C+1..C+F). Long history is
  motivated explicitly by non-cyclic-action ambiguity: with short history, a model
  seeing recent walking cannot tell whether "bend over and pick something up" has
  already occurred, "leading to inaccurate generations with missing or duplicated
  actions" (Sec. 3.3).
- **Token layout / positional encoding**: the index frame is boundary-anchored by
  construction — history occupies indices −H+1..0, the generation window 1..C,
  future goal tokens C+1..C+F (Eq. 4-5; Fig. 3 left axis). Motion tokens get
  **sinusoidal** positional encodings on this window-relative index; text and
  diffusion-step tokens get separate learned positional embeddings; all token
  types pass through linear projections to a common width (Sec. 3.4). The paper
  ablates **no** PE alternatives — the boundary-anchored indexing is presented as
  the design, not compared against clip-absolute or purely relative schemes. (The
  released code names this `learned_prefix_zero_at_first_generation` /
  `PositionalEncodingNegativeIndex`, index = `arange(tokens) − history_len`;
  verified in the M6 addendum of `ardy_vs_dsf_m4.md`.)
- **Conditioning pathways** (Sec. 3.4; Fig. 3):
  - *Text*: one token from LLM2Vec (embedding model on Llama-3-8B-Instruct),
    fed alongside the motion sequence (Sec. 3.5).
  - *Diffusion step*: one token, same treatment.
  - *In-window spatial goals*: goals are a masked explicit-representation sequence
    `g` + binary mask `v` (zeros where unconstrained), patchified to `R^{C×MP}`.
    The **root part of the noisy input tokens is overwritten** with the constraint
    root through the mask (`m̃_root = (1−v_root)⊙m_root + v_root⊙g_root`); body
    goals and the full mask are **concatenated along the feature dim** with the
    noisy tokens: `[m̃_root; x_body; g_body; v]` (Sec. 3.4, "Spatial Goal
    Conditioning").
  - *Out-of-window goals*: patchified `g^{(C+1):(C+F)}` + masks appended as extra
    transformer tokens; variable length and sparsity, unconstrained tokens masked
    out at attention time (Sec. 3.4).
  - There is **no first-heading-angle prefix token in the paper text** — heading
    enters only through `m_root` (cos ψ, sin ψ) and the history-translation
    convention at rollout (Sec. 4.1). (If a heading prefix exists it is code-only;
    do not cite the paper for it.)
- **Two-stage interleaved denoiser** (Sec. 3.4; Fig. 3 right): at every denoising
  step k, the **root transformer** predicts clean global root `m̂_root^{1:C}`;
  this is *detached* and fed into the **body transformer**, which predicts clean
  latent body `x̂_body^{1:C}`; concatenated ⇒ clean hybrid prediction; re-noised
  and fed back for step k−1. So root and body exchange influence every step
  ("interleaved"), not once. Hypothesis stated: body-given-clean-root is an easier
  task than joint prediction (Sec. 3.4).
- **Size**: each of the two transformers is 8 layers, 8 heads, width 1024; the
  deployed two-stage denoiser totals **~156M parameters** (Sec. 3.5).

## 4. Training recipe

- **Framework**: DDPM [Ho et al. 2020], x0-prediction ("clean hybrid prediction"),
  modified simplified loss (Sec. 3.5). Notably, **10 diffusion steps are used at
  both train and test time** — the few-step schedule is trained directly, not
  distilled (Sec. 3.5). No beta-schedule details are given.
- **History/future sampling** (Sec. 3.5): sequences clipped to **max 10 s** at
  20 fps. Per sample, a fixed-size generation window of `G` frames is placed
  **randomly** within the clip ⇒ `H` and `F` "vary dynamically, ranging from 0 to
  the maximum sequence length minus G". So H is (implicitly uniform) in
  [0, N−G]; H = 0 (cold-start generation) is a first-class training case. Max
  usable history at deployment: 8 s (= 10 s − 2 s window; Sec. 4.1).
- **History is clean ground truth.** The paper describes no history-noising, no
  scheduled sampling, no rollout-in-training, no self-conditioning anywhere.
  Teacher-forced GT history + variable H is the entire exposure-bias story.
- **Augmentation**: random rotations about the y-axis (Sec. 3.5).
- **Constraint sampling** (training; Sec. 3.5): sampled from the GT motion itself,
  drawn from "common use cases": 2D root keyframes, 2D root trajectories,
  full-body sparse keyframes, full-body keyframe blocks, sparse end-effector
  keyframes, foot-contact keyframes; both in-horizon and out-of-horizon.
- **CFG dropout**: text prompts *and* spatial constraints independently dropped
  with **10%** probability to enable classifier-free guidance (Sec. 3.5). No
  guidance scale is reported in the paper.
- **Losses** (all applied to the denoiser; Eq. 7-11, weights all 1 in Eq. 11):
  1. `L_hybrid = ||x̂0 − x0||_1` — smooth-L1 on hybrid tokens (Eq. 7).
  2. `L_dec = ||D(x̂0) − m_body||_1` — decode predicted tokens through the frozen
     tokenizer decoder, penalize in *explicit body space* (Eq. 8).
  3. `L_goal = ||v ⊙ (m̂0 − g)||_1` — extra weight on exactly the constrained
     entries of the full explicit prediction (Eq. 9).
  4. `L_consist = ||Ĵ0 − FK(θ̂0)||_2` — predicted joint positions vs positions
     implied by predicted rotations through forward kinematics (Eq. 10).
- **No dropout in the denoiser** — dropout would randomly destroy the overwritten
  root-constraint inputs (Sec. 3.5).
- **Optimizer/scale**: AdamAtan2, lr 2e-5, batch 512, **1M steps**, 4×A100-80GB
  (Sec. 3.5).
- **Data**: Bones Rigplay, ~700 h studio mocap, >150 participants, 27-joint
  unified retarget, 20 fps, clips 1-180 s cut to ≤10 s; LLM-paraphrased text
  labels; split by semantic action group 90/10 ⇒ ~315k train / ~35k test clips,
  test categories fully unseen (Sec. 5.1). HumanML3D experiments (Sec. 6) use a
  40-frame-horizon, 10-step model with a *vanilla AE* tokenizer and a retarget
  that preserves native SMPL rotations (HumanAct12 subset excluded).

## 5. Sampling & rollout

- **Steps**: 10 default (train=test); 4 is the deployable floor — Tab. 3 shows 4
  steps ≈ 10 steps on constraints (0.034 vs 0.025 m joint pos) while 1-2 steps
  collapse (1-step: FID 0.079, joint pos 1.04 m). Latency on RTX 4090: 33 ms
  (4-step), 63 ms (10-step) for a G=40-frame (2 s) window (Sec. 4.2).
- **Chaining** (Sec. 4.1): window 1 is generated with **zero history**; each
  subsequent window conditions on previously *generated* tokens as history.
  Truncated sliding window on both history and future-goal context, configurable
  up to 8 s (the max seen in training). Future constraints beyond the truncation
  horizon are **excluded entirely** until the advancing window brings them inside
  it.
- **Coordinate convention**: before each window, history root motion is translated
  so the last history frame sits at the origin; the offset is stored and re-applied
  to the output (Sec. 4.1). (Heading is carried in the root features; the paper
  describes only translation normalization, not rotation normalization.)
- **Interactive controller** (Sec. 4.1; Fig. 5): replanning triggers on any new
  user input (text or constraints) or when the playback buffer nears depletion.
  Latency-aware replan: the next `B` already-generated frames become a **replan
  buffer** — played back to the user *while simultaneously serving as history* for
  the asynchronous generation thread, hiding inference latency. Deployed: B=0
  buffer frames for the 4-step model, **1 buffer frame** for the 10-step model
  (Sec. 4.1, end).
- **Constraint overwriting at inference**: same as training — root constraints
  overwrite the noisy root channel each denoising step; mouse waypoints are
  linearly interpolated + smoothed into dense trajectories; keyboard velocity is
  interpolated and integrated into a root trajectory input (Sec. 4.2).
- **CFG at inference**: enabled by the 10% dropout; scale unreported.

## 6. Ablations & numbers

All on Bones Rigplay unless noted. Default config: FSQ 64-128, patch 4, horizon
40 frames, 10 steps.

**Tab. 2 — architecture** (text-only: skate m/s / R-prec / FID; constraint errors
in m or deg):
- ARDY default: 0.264 / 65.47 / 0.027; joint rot 2.23°, joint pos 0.025, keyframe
  0.023, traj 0.015, waypoint 0.024. (Dataset reference: skate 0.255, R-prec
  76.56.)
- **Fully explicit representation** (no tokenizer, same patching, masked
  overwriting for all constraints): the biggest effect in the paper. R-prec
  −11.6 pts (53.90), FID 2.4× (0.065), joint pos 5× (0.130), keyframe 6× (0.136),
  waypoint 8× (0.203). Interpretation given: high-dimensional explicit features
  cripple generative learning "particularly under our few-step denoising setting"
  (Sec. 5.2).
- **Global-root-conditioned decoder** (skip global→local): mainly a foot-skating
  and control hit — skate 0.264→0.303 text-only, waypoint 0.024→0.060.
- **One-stage denoiser** (joint root+body prediction): text-only metrics
  *unchanged or trivially better* (FID 0.029, R-prec 65.84), but constraint
  adherence craters: joint pos 4× (0.101), keyframe 3.4× (0.079), waypoint 6.8×
  (0.164). Two-stage is a *control* mechanism, not a fidelity mechanism
  (Sec. 5.2).

**Tab. 3 — hyperparameters** (Sec. 5.3):
- **Generation horizon** (frames; = C·P): 4 frames is catastrophic — FID 0.224,
  R-prec 33.42, all constraint errors ~0.85 m; described as training instability
  producing **drifting motions that fail to respond to text**, with a
  "misleadingly low" skate number (0.151) because the model barely responds.
  8 frames already works (FID 0.037, R-prec 56.70, traj 0.013, waypoint 0.020);
  12 → 0.033/59.54; 20 → 0.030/63.80; 40 → 0.027/65.47. FID and R-prec improve
  **monotonically** with horizon; constraints are best at 8 and 40. Qualitative:
  8-frame model switches actions faster on prompt updates and learns constraint
  adherence faster; 40-frame is more semantically faithful.
- **Diffusion steps** (at horizon 40): 1 → 0.079 FID / ~1 m constraint errors;
  2 → 0.052 / ~0.17; 3 → 0.041 / ~0.05; 4 → 0.034 / ~0.03; 10 → 0.027 / ~0.025;
  100 → 0.025 / ~0.028. Knee at 4; ≥10 saturates.
- **Tokenizer patch size**: 1 → fast early learning, later instability, terrible
  final (FID 0.152, R-prec 44.45, joint pos 0.764); 4 → default; 8 → slightly
  better FID/R-prec (0.022/68.01) but worse skate (0.317) and constraints
  (waypoint 0.100) — too much per-token compression loses fine pose detail.
- **FSQ capacity**: 16 levels-32 dims slightly better FID/R-prec under the 1M-step
  budget but degrades fine constraint detail (joint rot 4.57° vs 2.23°); 256 dims
  slows convergence with no gain. 64-128 is the compromise.
- **Tokenizer type** (at horizon 20): AE 0.033 / VAE 0.031 / FSQ 0.030 FID —
  statistical tie; FSQ wins on stability only (AE diverges at horizon 40).

**Tab. 4 — offline comparison, HumanML3D** (vs MaskControl): without test-time
optimization, ARDY joint error **4.15 cm vs 46.18 cm** at 0.15 s vs 0.46 s
latency; with optimization 0.30 vs 0.45 cm at 9.25 vs 68.65 s. FID 0.044 vs
0.050; R-prec 0.729 vs 0.760.

**Tab. 5 — autoregressive comparison, HumanML3D** (vs DiP; 9 s sequences, 1 s GT
history): in-horizon goals — R-prec 0.690 vs 0.609, FID 0.092 vs 0.967, error
2.48 vs 9.20 cm. Out-of-horizon goals — ARDY barely moves (0.684 / 0.100 /
2.92 cm) while DiP degrades sharply (0.599 / 1.453 / **17.64 cm**). This is the
direct evidence that long history + out-of-window goal conditioning enables
long-horizon planning that short-context AR diffusion (DiP: 1 s history / 2 s
window) cannot do.

**Tab. 6 — perceptual study** (240 pairwise, vs DiP, out-of-horizon): ARDY
preferred 65.8/67.5/64.6% (quality/semantics/goal accuracy) vs DiP 9.2/7.5/4.2%.

## 7. Failure modes & limitations (Sec. 7)

1. **Unstructured memory**: all past frames are kept as explicit history context;
   inefficient for extremely long horizons. Structured memory is future work.
2. **Multi-step cost**: diffusion still needs 4-10 iterations; shortcut/consistency
   models cited as the way forward.
3. **No physics**: purely kinematic — foot skating and jitter "can sometimes be
   observed". (Also implicit in Tab. 2/3: skate never reaches the dataset floor.)
4. From the ablations, ARDY's own degenerate regime: **too-short generation
   horizons (4 frames / 1 token) produce drifting, text-unresponsive motion with
   deceptively good motion-magnitude-adjacent metrics** (Sec. 5.3) — the paper's
   closest analog to a degenerate-continuation collapse. Idle/freeze collapse is
   never reported as an ARDY failure mode.

## 8. Portability map for the M6 freeze problem

Context (ours): M6 = 39.8M block-AR latent video DiT; frozen causal video VAE
f8t4d16; 64×64 RGBA stick-figure dance; rectified flow; T5-small text; clean
teacher-forced history up to 1 s; 8-video-frame generation blocks (= 2 latent
frames); 10 Euler steps; CFG 2. Failure: motion collapses over training (centroid
speed 0.142 vs real 0.314 at 20k steps) while anatomy improves; data has a
rest→action→rest profile; a matched full-clip control (R0) does **not** freeze —
so the collapse is specific to the block-AR continuation task, i.e., the model
learns "rest history → rest continuation" as the dominant conditional mode.

Per-technique judgment (paper evidence → relevance → cost):

1. **Longer generation horizon** (piloted: 16f). *Strongest ARDY-supported fix.*
   Tab. 3 horizon block is the paper's clearest dose-response: 1-token windows are
   degenerate (drifting, text-ignoring — the mirror image of our freeze: a
   too-short window cannot contain enough of an action for the loss to reward
   initiating it), and quality rises monotonically to 2 s windows. Our current
   block (2 latent frames ≈ ARDY's 8-frame setting) sits just above their
   catastrophe threshold; ARDY at that setting already shows depressed semantic
   adherence (R-prec 56.7 vs 65.5 at 2 s). Prediction: 16f (4 latent frames)
   helps, and the ARDY curve says gains continue to ~1-2 s windows — a 32f arm is
   worth queuing if 16f moves the centroid-speed metric but doesn't close it.
   Cost: compute only. Trade-off documented: longer windows respond to prompt
   changes more slowly (Sec. 5.3) — irrelevant while we have no online prompt
   switching in the training objective.

2. **Long variable history** (piloted: 2.4 s, `hist24`). Directly ARDY's stated
   answer to action-completion ambiguity (Sec. 3.3) — with only rest in a short
   history, "action already done" and "action not yet started" are
   indistinguishable, and the marginal (rest) wins. ARDY never ablates H, but
   Sec. 6.3/Tab. 5 shows the 1 s-history baseline (DiP — exactly our current
   history) failing at long-horizon behavior while 8 s-history ARDY does not.
   Cost: context length. Note the paper's H is not a fixed length: it *varies per
   sample* from 0 to N−G via random window placement — see item 3.

3. **Uniform random window placement (H,F sampling)** — *not yet in our fix list;
   flag.* ARDY trains by dropping the generation window uniformly at random inside
   the clip (Sec. 3.5), so the model sees every phase of the rest→action→rest arc
   as a target: initiation (rest history → action window), continuation, and
   completion, each with enough history to disambiguate. If M6's protocol is
   start-aligned (cf. `m6_protocol_v3_start_aligned_*` configs), the
   (rest-history → action-onset) case may be under-represented or systematically
   confounded with H=0. Matching ARDY here is a **zero-parameter data-level
   change**: sample block offsets uniformly, let history length vary 0..max per
   sample. Cheap, and it is the mechanism by which ARDY's clean teacher forcing
   stays honest.

4. **Clean teacher forcing is not the culprit per se** — *negative evidence worth
   recording.* ARDY uses pure clean-GT history: no noisy history, no scheduled
   sampling, no self-conditioning, no rollout training appear anywhere in the
   paper (Sec. 3.5, 4.1). Yet no idle collapse. What ARDY has that M6 lacks when
   collapsing: long variable history, longer windows, uniform window placement,
   and per-window conditioning strong enough (text + GT-sampled goals) that a
   frozen output is *wrong by construction* on most samples. Implication: our
   noisy-history pilot (U[0,0.2]) and magnitude-weighted flow loss are reasonable
   but have no ARDY precedent either way; the ARDY-predicted fixes are items 1-3
   and 6. If 1-3 fix the freeze, exposure-bias tricks may be unnecessary.

5. **Boundary-anchored positional encoding** (planned). The paper's indexing is
   boundary-anchored by construction (−H+1..0 | 1..C; Eq. 4-5, Fig. 3) with
   sinusoidal PE on that index (Sec. 3.4), and the code confirms the
   negative-index implementation — but there is **no PE ablation**: the paper is
   an existence proof that boundary-anchored absolute indexing works across
   variable H and unbounded rollout, not evidence it beats relative RoPE. Port is
   cheap (additive PE); rank it below items 1-3 on expected effect, and treat any
   gain as our finding, not ARDY's.

6. **Future-keyframe goal conditioning + goal loss** — *not yet considered; flag
   as the strongest untried idea for action initiation.* ARDY samples future
   constraints from the GT clip itself (Sec. 3.5), conditions on them even beyond
   the generation window, and adds `L_goal` (Eq. 9). Sec. 3.3 states the
   anti-freeze mechanism verbatim: an out-of-window destination "will determine in
   which direction the human should start moving from the first step". M6 analog:
   with 10% (or higher) probability, feed a GT latent frame from t+Δ (Δ ~ 0.5-2 s
   ahead) plus its time offset as extra conditioning tokens, and add a small
   goal-consistency loss; drop it for CFG so inference can run unconditioned.
   Whenever the future frame shows movement, rest-continuation becomes explicitly
   penalized — a supervised, targeted attack on rest-bias rather than a loss
   reweighting. Tab. 5's out-of-horizon results (2.92 cm vs DiP 17.64 cm) show
   this conditioning trains cleanly. Cost: moderate — a conditioning path + GT
   frame sampler; no new networks.

7. **Explicit root channel / two-stage prediction.** ARDY's Tab. 2 says the
   explicit-vs-latent split matters enormously (representation) and root-first
   two-stage matters for *constraint adherence only*, not text-only fidelity. M6
   analog: predict an explicit global-motion summary (per-frame figure centroid
   ± heading) as a first stage or auxiliary output, with the pixel/latent stage
   conditioned on it. Attractive because our collapse metric *is* centroid speed —
   an explicit trajectory channel makes freezing directly supervisable and even
   overwritable at inference (drag the centroid, get motion). But per Tab. 2 the
   two-stage split is a control mechanism, not an anti-collapse mechanism; and the
   M4-note verdict (don't port root/body machinery into the pixel ladder) still
   holds for the thesis scope. Rank: interesting M7+ direction, not an M6 fix.
   Cost: high (new head, centroid extraction in the loss loop, conditioning path).

8. **Decoded-motion loss (`L_dec`, Eq. 8).** ARDY decodes predicted latents
   through the frozen tokenizer decoder every training step and penalizes in
   explicit space, alongside the latent-space loss. M6 analog: decode predicted
   video latents through the frozen VAE decoder and apply the
   motion-magnitude-weighted loss in *pixel/centroid* space instead of (or on top
   of) latent space. Our current magnitude weighting operates on latent flow
   targets, where "motion magnitude" is a proxy; ARDY's pattern grounds the
   weighting where the failure is measured. Cost: one VAE-decoder forward per step
   (small at 64×64 with f8), plus loss plumbing. Un-ablated in ARDY (never removed
   in Tab. 2/3), so treat as plausible, not proven.

9. **FK-consistency loss (Eq. 10), foot-skate loss (Eq. 6), constraint
   overwriting, replan buffer, LLM2Vec text, 4M-step tokenizer / 1M-step denoiser
   recipe.** No mechanism relevant to freeze; per the M4 note these stay
   un-ported. One transferable detail: **no dropout in the denoiser** when
   conditioning channels carry overwritten values (Sec. 3.5) — remember this if
   item 6's goal tokens are added.

10. **Diffusion-step count.** Tab. 3 says 4-10 steps saturate in ARDY's compressed
    token space; this does not transfer to pixel-latent flow (M4-note deviation
    #4), and steps are unrelated to the freeze (R0 doesn't freeze at the same
    10-step budget).

### What ARDY does NOT contain (searched for explicitly)

No scheduled sampling, no noisy-history augmentation, no self-conditioning, no
rollout/consistency training, no anti-idle regularizer, no motion-magnitude loss
weighting, no action-completion token or timer. ARDY's entire defense against
degenerate continuation is architectural and data-level: long variable history
(Sec. 3.3), uniform window placement (Sec. 3.5), sufficient window length
(Tab. 3), and goal conditioning sampled from GT (Sec. 3.3, 3.5).
