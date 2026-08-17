# Dancing Stick Figures: a fully-labelled synthetic motion-video benchmark with render-based anatomy metrics for small-scale video diffusion

*(working title per Jin: "Dancing Stick Figures"; the dataset id stays `stickdance-128`)*

**Draft v0 — skeleton with holes.** `[TODO]` = needs an experiment or a number. `[CLAIM?]` = only
if the evidence lands. Target: arXiv tech report (6–8 pp) + HF dataset + code. Owner: Sprited.

---

## Abstract  `[write last]`

Training and evaluating video generation models is inaccessible to most researchers: data is
large and unlabelled, evaluation relies on FVD, and reference recipes assume clusters. We
introduce **stickdance**, a synthetic benchmark of colour-coded stick-figure videos at 128×128 in
which every pixel is a deterministic function of a known 3D skeleton (27 joints, NVIDIA ARDY
`cskel27`), camera, and body parameters. Motion comes from a text-to-motion foundation model
(ARDY), so clips carry text; because we own the renderer, each frame ships with exact 3D/2D pose,
depth, normals, and part segmentation, and captions are generated from labels rather than
guessed. We propose **render-based anatomy metrics** — skeleton reprojection error, bone-length
deviation, topology violation, limb-identity error — and validate them against controlled
corruptions and FVD `[TODO numbers]`. We release reference pixel-space video diffusion baselines
(a classic UNet recipe and a DiT + flow-matching recipe) that train on a single 24–32 GB GPU in
hours, with curves, wall-clock and memory `[TODO]`, and show `[CLAIM?]` that colour-coding limbs
reduces limb-identity errors by N %. Data CC0, code MIT, motion generated with ARDY under the
NVIDIA Open Model License.

## 1  Introduction

- The gap: video-gen research needs (i) small, labelled, *legible* data, (ii) evaluation that
  doesn't reduce to FVD, (iii) recipes that fit one consumer GPU. Toy video datasets exist
  (Moving MNIST, KTH, CLEVRER, Kubric/MoVi) but none combine human motion, text, exact anatomy
  labels and a checkable oracle. `[TODO Table 1: toy video datasets × {frames, res, text, 3D
  labels, oracle, license}]`
- Key idea: **own the generator, borrow the motion.** A text-to-motion FM supplies natural,
  captioned motion; a deterministic renderer supplies exact labels and, crucially, a way to
  score generated frames without labels.
- Contributions: (1) dataset + generator + captions from labels; (2) validated anatomy oracle;
  (3) consumer-GPU baselines and analyses (colour vs mono, pixel vs latent, 64 vs 128, seed/camera
  vs prompt diversity, temporal failure modes).
- Fig. 1: pipeline (prompt → ARDY → cskel27 → camera → G-buffer) + a hero grid. `[TODO fig]`

## 2  Related work

- Synthetic datasets with oracles: dSprites, Shapes3D, CLEVR/CLEVRER, Kubric & MoVi, Moving MNIST,
  KTH. Position: same philosophy, human motion + text + anatomy metrics. `[TODO cite]`
- Text-to-motion & skeleton data: HumanML3D, AMASS, KIT-ML; ARDY (Zhao et al. 2026); Rigplay.
  Licensing note: why we use ARDY outputs and not AMASS renders. `[TODO cite]`
- Verifiable evaluation of generative models: GenEval-style object/attribute checks, VBench,
  human-preference vs. metric critiques; the "hands" problem. `[TODO cite]`
- Small-scale diffusion references: DDPM/IDDPM, v-pred/zero-SNR, DiT, flow matching / rectified
  flow, SVD-style factorised temporal layers. `[TODO cite]`

## 3  The stickdance dataset

### 3.1 Generation pipeline
- Prompts: 143 (6 groups: idle, locomotion, gesture, dance, sport, transitions), 10 seeds each,
  6 s @ 20 fps via ARDY (`generate.py`), Core skeleton. `[TODO final counts after scale-up]`
- Canonical skeleton: cskel27 (names, hierarchy, rest proportions) — same label space as ARDY.
- Body sampling: per-clip bone-scale jitter ±8 %, stroke 3–5 px, scale 50–58 px/m.
- Camera: orthographic; 3 per clip; yaw canonical (0/±45/±90/±135/180 ± 6°) 70 % / uniform 30 %;
  pitch −3…10°.
- Renderer: z-buffer capsule rasteriser at 4× supersample → colour (RGBA, transparent),
  depth (16-bit, metres), normals (camera space), part segmentation (27 ids). One pass.
- Determinism: `regenerate.py --seed` reproduces bytes. `[TODO verify + state]`

### 3.2 Labels
Per frame: `joint_xyz` [27,3] m (figure frame), `joint_xy` [27,2] (normalised), `joint_depth`,
`joint_visible`, camera (yaw, pitch, px/m, centre), body params, `root_pos/vel/heading`,
categorical: `facing_rel_cam`, `screen_motion_dir`, `speed_bucket`, `posture`,
`hand_{L,R}_above_head`, `foot_{L,R}_contact`, `gait_phase` `[TODO implement]`; per clip:
`ardy_prompt`, `group`, `seed`, `qa_flags`, `prompt_match` `[TODO]`, `motion_segments` `[TODO]`,
`caption_{short,long}` + paraphrases `[TODO]`.

### 3.3 Captions from labels
Deterministic template (view · action · screen direction · speed · facing · hands · posture ·
temporal phase) → 1 canonical + 3–4 LLM paraphrases with slot-preservation check; per-window
captions with clip context. `[TODO implement; Fig: examples]`

### 3.4 Splits and QA
Split by prompt (all seeds/cameras of a prompt share a split): train 90 / val 5 / test 5 within
groups; **`sport` held out entirely** (33 prompts) as a concept-generalisation probe. QA flags:
`levitation`, `frozen`, `off_prompt` (contact-sheet audit) `[TODO audit + counts]`. Known ARDY
failure classes (crawl, crouch-walk, stairs, acrobatics) removed with reasons.

### 3.5 Statistics
`[TODO Table 2: clips, prompts, seeds, cameras, frames, size per config (color / gbuffer / 64px /
mini), split sizes, QA-flag rates, per-group counts, root-speed and yaw histograms (Fig)]`

### 3.6 Configs and access
`stickdance-128` (color default; `gbuffer`; `mini`), `stickdance-64` (native render, stroke ≥ 2 px).
`load_dataset` one-liner; window loader; viewer. License: data CC0, code MIT; motion generated
with `nvidia/ARDY` (NVIDIA Open Model License; NVIDIA claims no ownership of outputs), ARDY
trained on Rigplay mocap.

## 4  Render-based anatomy metrics ("the oracle")

### 4.1 Definitions
- **LIE** limb-identity error: per-colour masks → is each coloured chain attached at the correct
  joint / side? (regressor-free)
- **BLD** bone-length deviation: per-colour PCA axes → bone lengths, normalised by figure scale;
  deviation from cskel27 proportions and *temporal drift* within a clip. (regressor-free)
- **TVR** topology violation: per-colour connected components / expected counts; extremities.
  (regressor-free)
- **SRE** skeleton reprojection error: pose regressor (image → 27×2) → re-render → pixel/mask
  agreement with the input; regressor PCK reported. `[TODO train regressor; PCK@2px]`

### 4.2 Validation  `[the core table]`
- Floor: real held-out frames → all metrics ≈ 0. `[TODO]`
- Ceiling / specificity: four controlled corruptions of real frames — L/R colour swap, +30 % bone
  stretch, delete a hand, add a third arm — each metric should respond to its intended corruption
  and not to the others. `[TODO Table 3 / heatmap]`
- Relation to FVD: metrics vs FVD across training checkpoints of the baselines; report
  correlation and at least one disagreement (Fig). `[TODO]`
- Failure modes of the oracle itself (thin strokes, occlusion, off-manifold samples). `[TODO]`

## 5  Baselines on one consumer GPU

### 5.1 Setup
16-frame windows (0.8 s), 128² × 4 ch premultiplied RGBA; also 64² / 8 f preset. GPUs: RTX 4090
(24 GB), RTX PRO 6000 (96 GB, reference); report s/it, peak GB, GPU-hours. Fixed seeds, val loss.
`[TODO presets + numbers]`

### 5.2 Tracks
- **A. Classic UNet** (46 M factorised 3D UNet, v-pred, cosine, DDIM, EMA) — "SD-era recipe".
- **B. DiT + flow matching, pixel space** — reference recipe. `[TODO implement]`
- **C. Latent (own f4 4-ch VAE + DiT)** `[optional; TODO]` — VAE reconstruction floor measured
  with SRE/LIE.
Conditioning: unconditional → group (CFG) → text (frozen text encoder) `[TODO]`; held-out
`sport` evaluation.

### 5.3 Results
`[TODO Table 4: track × config → params, GPU-h, s/it, GB, val loss, FVD, LIE, BLD, TVR, SRE]`
`[TODO Fig 3: metrics vs GPU-hours on 4090]` `[TODO Fig: sample grids at checkpoints]`

## 6  Analyses
1. **Colour vs mono (alpha-only)**: same model, LIE / identity-swap rate. `[CLAIM?]` `[TODO]`
2. **128 vs 64**: quality/time trade-off; when to start at 64. `[TODO]`
3. **Pixel vs latent**: VAE floor vs speed. `[TODO if Track C]`
4. **Data diversity**: seeds×cameras vs prompts — train on {1,3,10} seeds; memorisation
   (nearest-neighbour distance to train) vs held-out metrics. `[TODO]`
5. **Temporal failures**: identity swap rate, bone-length drift, jitter over 16 frames. `[TODO]`

## 7  Limitations
Single figure; orthographic; 128 px; ARDY's motion distribution (no floor contact beyond
sit/kneel, no acrobatics); 143 prompts; captions template-derived; synthetic-only — pair with a
real dataset for anything beyond mechanism.

## 8  Release
HF: `sprited/stickdance-128`, `-64`; code: generator, builder, eval, trainers, viewer; regenerate
script; model cards for baselines. `[TODO links]`

---

## Appendix (planned)
A. cskel27 joint table & palette · B. caption grammar · C. corruption protocol · D. training
hyper-parameters · E. per-prompt QA table · F. compute log.

## Experiment backlog (→ fills the holes above)
| id | experiment | fills | GPU-h | status |
|---|---|---|---|---|
| E1 | scale-up build (10 seeds × 3 cams) + stats | 3.5 | 0 | 🟡 running |
| E2 | oracle v0 (regressor-free) + corruption table | 4.1–4.2 | 0 | ☐ |
| E3 | pose regressor + PCK | 4.1 | 1 | ☐ |
| E4 | FVD pipeline (I3D) | 4.2, 5.3 | 0.5 | ☐ |
| E5 | Track A on 4090 preset + 96 GB reference | 5.3 | 20 | 🟡 a0 running |
| E6 | Track B DiT-FM | 5.3 | 20 | ☐ |
| E7 | colour vs mono | 6.1 | 10 | ☐ |
| E8 | 64 vs 128 | 6.2 | 6 | ☐ |
| E9 | diversity ablation (1/3/10 seeds) | 6.4 | 15 | ☐ |
| E10 | temporal metrics on checkpoints | 6.5 | 0 | ☐ |
| E11 | group/text conditioning + held-out sport | 5.2 | 15 | ☐ |
| E12 | captions + categorical labels + prompt_match audit | 3.2–3.4 | 0 | ☐ |
