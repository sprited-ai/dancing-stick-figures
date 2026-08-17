# SPARK — `stickdance-128`

**A procedurally-generated dancing stick figure dataset, and the diffusion model we train on it.**

Status: proposal · Author: Claudia · Date: 2026-08-17 · Owner: Jin

---

## 0. The one-liner

> A synthetic 128×128 dataset of animated stick figures where every pixel is a
> deterministic function of ~20 known parameters — so for the first time you can
> **script a check for whether the generated hands are wrong.**

Two artifacts ship, in this order:

1. `sprited/stickdance-128` — the dataset + the generator that made it (**guaranteed**, ships day 4)
2. `sprited/stickdance-ddpm-v0` — a from-scratch DDPM trained on it (**best effort**, days 5–7)

If (2) fails, (1) still stands on its own and the week was not wasted.

---

## 1. Why this, and why it's not a toy

The strategic goal is repositioning Sprited from wrapper-company to model-company.
That repositioning is not achieved by training *a* model. It's achieved by having a
**public research line** — a sequence of artifacts under `sprited/` that visibly
build on each other.

You already have artifact #1: `sprited/sprite-dx-anti-corruption-v1` — a 13.3M-param
U-Net, 128×128 RGBA, pixel-art restoration. That is not a random project; it's the
first point of a line. `stickdance-128` is chosen to be **the second point on the same
line**, not a new line:

| | `sprite-dx-anti-corruption-v1` | `stickdance-128` |
|---|---|---|
| canvas | 128×128 RGBA | 128×128 RGBA |
| domain | pixel art sprites | pixel art sprites (animated) |
| params | 13.3M | ~15–30M |
| architecture | U-Net + residual | U-Net + residual (+ time embedding) |
| task | restore | generate |

Same canvas, same aesthetic, same parameter scale, same architecture family. A visitor
landing on the `sprited` org page sees **a lab with a thesis** ("small models for sprite
generation and repair") rather than two unrelated uploads. That coherence is worth more
than either artifact alone, and it costs nothing extra because you already made the
choice once.

There is also a compounding benefit: the anti-corruption model becomes a *post-processing
step* for the generator's output. Diffusion samples at 128×128 come out soft; running them
through your existing quantizer/sharpener is a two-line demo that makes both models look
better. Ship that in the Space.

---

## 2. What we are NOT doing, and why

You floated: NVIDIA "ARDY" for motion generation, or Seedance → video → pose extraction →
retarget to rigs → animate in Unity → render.

**Killing that path for week 1.** Four reasons, in order of how much they'd hurt:

### 2.1 It has no oracle, which defeats the entire purpose

You said the point is to *reverse-engineer a diffusion model for your own practice*. The
single most valuable property a practice dataset can have is a **known ground truth
manifold**. When your DDPM emits a blurry mess at epoch 40, you need to answer "is this
model broken or is this data hard?" With mocap-derived video frames you cannot answer that.
With procedural data you can, exactly: you know the data lives on a ~20-dimensional manifold,
you know its parameterization, and you can render the nearest true sample for comparison.

A retargeted-mocap dataset is *more realistic and less informative*. For learning, that trade
is backwards.

### 2.2 The licensing makes it unpublishable

This is the hard blocker, not a soft one.

- **AMASS** (the standard mocap corpus) is research-only, non-commercial, and its component
  datasets have individually restrictive terms. Publishing a derived public dataset is
  not clean.
- **Mixamo** animations are Adobe-licensed; redistribution of the animation data (as opposed
  to using it in a product) is restricted.
- **Seedance / any commercial video model** — outputs used as training data for a
  redistributed dataset land you in unresolved terms-of-service territory. Nobody wants to
  be the test case.

Procedural generation from your own code has **zero** of this. You can license the data CC0
and the generator MIT, and no one ever has to think about it again. For an org trying to
build public credibility, "provenance is trivially clean" is a feature you can put on the card.

### 2.3 The pipeline is 3 weeks, not 1

Video → pose extraction → retarget → rig → render has a failure mode at every arrow:
temporal jitter in extracted keypoints, depth ambiguity in monocular pose, retarget
artifacts on limb-length mismatch, Unity render setup, batch export. Each is a day minimum
and several are multi-day when they go wrong. You'd spend the whole week debugging a
*data pipeline* and train zero diffusion models.

### 2.4 "Careful artistic direction" is easier in code than in a retarget

You explicitly named artistic direction as the differentiator. Counterintuitively,
procedural gives you *more* control, not less — overlapping action, squash, and arc quality
become named float parameters you can sweep and A/B, instead of properties baked into
someone else's mocap take. See §5.

### What survives from your idea

Motion realism is a real axis and mocap wins on it. It's just **v2**. Once the procedural
pipeline exists, swapping the motion source is a single interface — `pose(t) -> joint angles` —
and you can drop AMASS-derived or model-generated motion behind it later for a
`stickdance-natural` variant, with the licensing question answered separately at that time.

### 2.5 ARDY, evaluated properly

**ARDY** = *Autoregressive Diffusion with hYbrid representation for Interactive Human Motion
Generation*, NVIDIA (SIGGRAPH 2026). Text → streaming full-body motion, 326M params, 4-step
diffusion at ~33 ms/step. Facts that matter for us:

| | |
|---|---|
| output | 27-joint "Core" skeleton, 20 fps, up to 8 s. World-space joint positions + local/global rotations + foot contacts. |
| batch | `scripts/generate.py` — headless, text prompt → `.npz`, `--num_samples`. |
| code | Apache-2.0 |
| checkpoints | NVIDIA Open Model License — *"NVIDIA claims no ownership rights in outputs."* Outputs are ours to license. |
| training data | Bones Rigplay 1 (630 h mocap w/ text) — **not** AMASS. |
| requirements | Ubuntu 22.04, RTX 4090+, driver ≥ 575, PyTorch ≥ 2.4 — **`gin` meets every line.** |

So the §2.2 licensing blocker **does not apply to ARDY.** Unlike AMASS/Mixamo/commercial video
outputs, ARDY-generated motion can go into a CC0 dataset cleanly. That's a real correction.

What still holds: §2.1 (no oracle over the *motion* manifold — but note the *anatomy* oracle in
§6 still works, since we render) and §2.3 (time). So the decision becomes:

- **Week 1: hand-authored loops.** Small, legible, cyclic, named classes, zero infra.
- **Week 2: ARDY behind the same `pose(t)` seam.** `.npz` → 27→15 joint mapping → our renderer
  → `stickdance-natural`. Free text labels come with it. Same renderer, same oracle, same card
  format — the only thing that changes is the motion source, which is exactly what the seam
  was for.

That's a better story than either alone: *"v1 is hand-keyed, v2 swaps in an NVIDIA foundation
model for motion, and the anatomy oracle scores both."*

Sources: [ARDY paper](https://arxiv.org/abs/2607.08741) · [nv-tlabs/ardy](https://github.com/nv-tlabs/ardy) · [HF: nvidia/ARDY-Core-RP-20FPS-Horizon40](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon40) · [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)

### Why stick figures over emojis

You floated emojis as the alternative. Stick figures win:

- **Emoji has no motion structure.** The whole research value here is that a temporal
  dataset gives you image-gen *and* video-gen *and* frame-interpolation from one corpus.
- **Emoji is IP-encumbered.** Apple's set is copyrighted; Google's Noto and Twemoji are
  permissive but derivative-dataset provenance gets murky, and "we trained on emoji" invites
  a takedown conversation you don't need.
- **Emoji has no oracle.** No parameterization, no anatomy to check, no metric.
- Stick figures are **zero-IP, infinitely parameterizable, and anatomically checkable.**

Your instinct that "dancing stick figure" kills decision paralysis was correct. It's also
just the better dataset.

---

## 3. Dataset specification

### 3.1 Canvas

- **128 × 128, RGBA, PNG.** Matches the existing org format exactly.
- Figure occupies ~100px of the 128px height, centered, with jitterable framing.
- Nearest-neighbour / box downsample from a 4× supersampled vector render → crisp edges,
  controlled aliasing, no mushy anti-aliasing that a small model will waste capacity learning.

### 3.1b Colour coding — every body part has a fixed colour (default config)

Jin's call, and it's the right one. Each of the 8 limb segments gets a fixed, distinct colour;
head + torso stay ink-dark. Warm = left, cool = right, proximal darker than distal:

| segment | L | R |
|---|---|---|
| upper arm | red `#E84030` | blue `#286EE6` |
| forearm | orange `#FF9628` | cyan `#50C8F0` |
| thigh | magenta `#C832A0` | green `#1E965A` |
| shin | pink `#FF78C8` | lime `#78DC5A` |
| torso / head | ink `#282828` | |

Why this is more than cosmetic:

1. **Kills mirror ambiguity.** A black stick figure and its reflection are indistinguishable,
   so image ↔ label is many-to-one. With colour it's one-to-one. This matters for the pose
   oracle *and* for the diffusion model — the training target is no longer secretly bimodal.
2. **The oracle gets simpler and stronger.** Colour segmentation → per-limb masks → PCA axis →
   joints. Nearly deterministic; the CNN regressor becomes a cross-check rather than the
   instrument. And a new failure mode becomes measurable: *"left-arm colour appearing on the
   right side"* — call it **Limb Identity Error (LIE)**. Generators confuse left/right constantly;
   nobody can measure it on real data. Here it's a pixel count.
3. **Matches the OpenPose / ControlNet skeleton convention.** Per-limb-coloured skeletons are
   already the lingua franca of pose conditioning. `stickdance` samples are drop-in condition
   images for downstream character generators — the dataset gets a second life as a control
   signal.
4. **Occlusion legibility for free.** Halo separates depth; colour separates identity. `floss`
   at 128px goes from "readable" to "obvious." Verified: `scratch/color_probe.py`.

`mono` (ink-only) stays as a secondary config for the "harder, ambiguous" variant — useful
precisely *because* it's bimodal — but colour is the default and the one on the hero grid.

### 3.2 Skeleton — with hands and feet

19 joints, 18 segments, 2D. Hands and feet are **in** (Jin's call; also see §3.2b):

```
             head
              |
            neck ── shoulder_L ── elbow_L ── wrist_L ── hand_L
              |   └─ shoulder_R ── elbow_R ── wrist_R ── hand_R
            spine
              |
            pelvis ── hip_L ── knee_L ── ankle_L ── toe_L
                   └─ hip_R ── knee_R ── ankle_R ── toe_R
```

### 3.2b Hands and feet at 128px

Probed in `scratch/hands_probe.py`. Findings:

- **Feet: a 7px segment from ankle, pointing in the facing direction, shin colour.** Reads
  instantly. Foot direction is a free "which way is the figure facing" cue that black-stick
  figures don't have.
- **Hands: a plain blob at the wrist is too small to read at 128px** — barely distinguishable
  from a bare wrist. Rejected.
- **Hands: mitten = palm blob + 3 fingers (2px each, ~10px hand width), forearm colour.**
  Fingers individually smear at 128px but the *hand* gestalt is unmistakable. **This is the
  default.** It's on-brand for pixel art, and it makes the §6 pitch literally true.
- New param `wrist_angle` (±30°, per hand, animated) so hands aren't just forearm extensions.
  Adds a hand-pose dimension for the oracle: finger count + wrist angle + which arm it's on.

Fingers at 2px are borderline by design — that borderline *is* the interesting difficulty.
A model that gets fingers right at 128px has learned something real. If it turns out to be
too hard for DDPM v0, `hands=blob-large` (r=5px) is a one-flag fallback that keeps hands
detectable while removing finger structure.

Bone lengths are sampled per-figure (within anatomical ratios) and held constant for the
whole clip. **This is deliberate**: constant bone length within a clip is what makes
"Bone Length Deviation" a valid metric later (§6). A model that has learned the domain
keeps limbs rigid; a model that hasn't, stretches them.

### 3.3 Motion library — 12 hand-authored loops

Every loop is 32 frames and seamlessly cyclic. Hand-authored keyframes, not sinusoid soup —
sinusoids give you technical diversity and semantic mush. Named classes are what make the
dataset legible on the card and usable for conditioning experiments.

| id | name | why it's in the set |
|---|---|---|
| 0 | `idle_bob` | baseline; weight shift + breathing, near-stationary |
| 1 | `walk` | the canonical 8-key cycle; every animator's Hello World |
| 2 | `run` | high amplitude, both feet airborne — tests extreme poses |
| 3 | `jumping_jack` | full symmetric extension; limbs at frame edges |
| 4 | `wave` | asymmetric; one limb moves, rest is idle |
| 5 | `robot` | **stepped interpolation, no easing** — deliberate contrast case |
| 6 | `twist` | hips counter-rotate against shoulders |
| 7 | `disco_point` | alternating arm extension, iconic silhouette |
| 8 | `jump` | anticipation → launch → land squash; the timing showcase |
| 9 | `floss` | arms cross the body — **self-occlusion stress case** |
| 10 | `moonwalk` | 🔒 held out | 
| 11 | `helicopter_kick` | 🔒 held out |

Styles 10 and 11 appear **only in the test split**. This gives the dataset a built-in
compositional-generalization probe: can a pose-conditioned model render a style it never saw
as a class label? Most synthetic datasets don't bother with this and it costs us nothing.

`floss` earns its place specifically because limbs crossing the torso is the hardest thing
to render legibly and the hardest thing for a generator to get right. It's the "hands" of
this dataset.

### 3.4 Parametric variation

Each sample draws from a seeded parameter vector:

| param | range | effect |
|---|---|---|
| `tempo_scale` | 0.6 – 1.6 | frames-per-cycle |
| `amp_scale` | 0.7 – 1.3 | global motion amplitude |
| `asym` | 0.0 – 0.3 | left/right amplitude imbalance |
| `lean` | ±12° | whole-body tilt |
| `bounce_amp` | 0 – 6 px | root vertical oscillation at 2× tempo |
| `squash_gain` | 0 – 0.25 | torso compression coupled to vertical velocity |
| `lag_gain` | 0 – 0.12 | distal joint phase delay (see §5) |
| `bone_scale[8]` | ±15% | per-limb length |
| `wrist_angle_L/R` | ±30° | hand pose, animated per style |
| `hand_style` | `mitten` (default) / `blob-large` | difficulty fallback |
| `stroke_width` | 3 – 6 px | line weight |
| `mirrored` | bool | horizontal flip |
| `stroke_rgb` / `bg_rgba` | palette | appearance (styled config only) |

12 discrete styles × a 20-D continuous space = effectively unbounded, with clean labels
throughout.

### 3.5 Label schema (one row per frame)

```jsonc
{
  "sample_id":    "0007f3a1",
  "seed":         2847119,
  "clip_id":      "c00412",     // null for single-frame configs
  "frame_idx":    17,           // 0..31
  "phase":        0.53125,      // frame_idx / 32, cyclic
  "style_id":     9,
  "style_name":   "floss",
  "params":       { "tempo_scale": 1.12, "amp_scale": 0.94, ... },
  "joint_angles": [18 floats, radians],
  "joint_xy":     [[x,y] × 19, normalized to [0,1] in image space],
  "bone_lengths": [12 floats, px],
  "visible":      [19 bools],   // occluded-by-torso flag
  "appearance":   { "stroke_rgb": [17,17,17], "bg_rgba": [0,0,0,0], "stroke_width": 4 }
}
```

`joint_xy` in image space is the important one — it's directly usable as a ControlNet-style
conditioning signal, and as the regression target for the pose oracle in §6.

### 3.6 Configs and splits

| config | contents | purpose |
|---|---|---|
| `color-frames` | 65,536 frames · fixed limb palette (§3.1b) · transparent bg | **default**; hero grid, DDPM v0, oracle |
| `mono-frames` | 65,536 frames · ink stroke only | the mirror-ambiguous variant; harder |
| `clips-32` | 4,096 clips × 32 frames = 131,072 · colour | video-gen, frame interp, next-frame |

(`styled-frames` — random palettes per figure — dropped from week 1. With a fixed semantic
palette it's a different, less useful axis. Revisit as `stickdance-styled` if there's demand.)

Splits: `train` 90% / `val` 5% / `test` 5%, **split by clip and by figure, never by frame** —
otherwise near-duplicate frames leak across the boundary and every metric lies to you.
Styles 10–11 are test-only.

Total ≈ 262k frames ≈ **~1.0 GB**. Generation cost is minutes of CPU, not GPU-hours.

### 3.7 Format

Parquet shards with embedded PNG bytes + an `image` column, so `load_dataset("sprited/stickdance-128")`
works with no custom loader. Plus:

- `generator/` — the full source, MIT
- `regenerate.py --config mono-frames --seed 0` → **byte-identical output**

Determinism is not a nicety here. It means the dataset is really "a seed plus 800 lines of
code," anyone can regenerate it at 10× scale for free, and the repo is honest about being
synthetic rather than pretending to be found data.

**License: CC0 for the data, MIT for the generator.**

---

## 4. Why this is the right substrate for learning diffusion

Concretely, what this dataset buys you that CIFAR or a scraped set does not:

1. **You know the manifold.** ~20 intrinsic dimensions embedded in 49,152 pixel dimensions.
   When training diverges you can distinguish model bugs from data difficulty.
2. **Labels are exact and free.** No annotation noise. Pose conditioning, class conditioning,
   and CFG experiments all become available with zero labeling cost.
3. **Difficulty is a dial.** Start with `mono-frames`, one style, fixed bone lengths — a
   near-trivial target that a working DDPM *must* nail in an hour. If it can't, your DDPM is
   broken and you know it on day 5, not day 7. Then turn on styles, then appearance, then temporal.
4. **It trains on one consumer GPU.** 128×128, ~15–30M params, 65k images. Hours, not weeks.
   And you can prototype at 64×64 by downsampling on the fly — same data, 4× faster loop.
   **Do this on day 5.**
5. **Temporal structure comes free.** The same corpus supports unconditional image gen →
   pose-conditioned gen → frame interpolation → next-frame video, so the research line has
   somewhere to go in weeks 2–4 without a new dataset.
6. **It's visually legible.** You can tell at a glance whether a sample is good. Debugging
   generative models on data you can't eyeball is misery.

---

## 5. Artistic direction

This is the section that decides whether the dataset gets likes or gets ignored, so it's
worth being specific. Bad stick figure animation reads as clip art. Good stick figure
animation reads as *alive*. The difference is four things, all of which are expressible
as parameters:

### 5.1 Overlapping action — the single biggest "alive vs dead" factor

Distal joints lag proximal ones. Shoulder leads, elbow follows a few frames later, wrist
later still. Implement as a per-joint phase delay that increases with distance from the root:

```
delay(joint) = lag_gain × depth(joint)     # shoulder 0, elbow 1, wrist 2
angle(joint, t) = keyframe(joint, t - delay(joint))
```

One float (`lag_gain`) and the whole figure stops looking like a puppet.

### 5.2 Arcs, not lines

Joints must travel in curves. Catmull-Rom interpolation between keyframes instead of linear.
Free once you write it, and `robot` gets its character by explicitly *disabling* it (stepped
interpolation) — which is why `robot` is in the library.

### 5.3 Weight — squash, bounce, and the drop

- Root drops on the downbeat, `bounce_amp` at 2× tempo.
- Torso compresses proportional to vertical velocity (`squash_gain`) — critical on `jump`'s landing.
- Head tilt leads the direction of motion by a few degrees.

### 5.4 Silhouette readability at 128px — the halo trick

At 128×128 with a 4px stroke, a limb crossing in front of the torso becomes an unreadable
blob. Fix: draw each limb with a 1–2px **background-coloured outline** underneath the stroke,
depth-ordered front to back. This is what makes hand-drawn stick figure animation legible,
it's ~10 lines of code, and without it the `floss` and `twist` styles are mud.

This is the detail I'd most want in a v0. It's the difference between "cute" and "someone
who knows animation made this."

### 5.5 Line quality

Rounded caps and joins. Slight taper — thicker at the torso, thinner at extremities. Head as
a filled circle with a 1px gap at the neck join so it reads as a head rather than a lollipop.

---

## 6. The differentiator: an anatomy oracle

This is the part that turns "a pile of PNGs" into something people bookmark.

Because we own the renderer, we can evaluate generated samples against ground truth
**even though generated samples have no labels.** Three metrics, all cheap:

### 6.1 Skeleton Reprojection Error (SRE)

1. Train a small CNN pose regressor on the GT data (image → 15 joint xy). Labels are exact,
   so this is trivially accurate — it's not research, it's a measuring instrument.
2. Given a *generated* sample: predict its pose, **re-render that pose with our renderer**,
   and compare the re-render to the generated image.

A valid figure reprojects almost perfectly. A three-armed mutant does not. One number,
no human eval, no FID hand-waving.

### 6.2 Bone Length Deviation (BLD)

From predicted joints, check whether bone lengths are mutually consistent with a single
figure scale. Diffusion models notoriously produce rubber limbs; this catches it directly.

### 6.3 Topology Violation Rate (TVR)

Skeletonize the generated image, count connected components and endpoints. A correct figure
has exactly **1 component and 5 extremities** (head, 2 hands, 2 feet). Anything else is a
violation. Brutal, cheap, uninterpretable-by-nobody. With mitten hands, extremities are real
blobs rather than line-ends, so this is robust to skeletonization noise — and a **finger count
per hand** (expected 3) becomes a sub-metric. Yes, we can literally count the fingers.

### 6.4 Limb Identity Error (LIE) — colour config only

Per-limb colour masks → is the red/orange chain attached at the left shoulder and the
blue/cyan chain at the right? Is each chain's proximal colour actually proximal? Left/right
confusion and limb-swaps are among the most common generative failures and, on real data,
completely unmeasurable. Here: pixel counting.

Report all four alongside standard FID against a held-out split.

### The pitch line

> *"AI can't draw hands" is a meme because nobody can measure it.
> Here it's `stickdance.eval(samples)` and it returns a number — including how many fingers.*

That framing is the distribution hook. It's true, it's useful, and it's funny — which is
what actually gets a dataset shared.

---

## 7. Model plan

Staged, so that each stage is independently shippable:

| ver | task | conditioning | when |
|---|---|---|---|
| **v0** | unconditional DDPM | none | day 5 |
| v1 | class-conditional | `style_id` + CFG | day 6 if v0 lands |
| v2 | pose-conditional | `joint_xy` heatmaps | week 2 |
| v3 | frame-pair / next-frame | prev frame + Δphase | week 3+ |

v0 spec: U-Net, ~15–30M params, sinusoidal timestep embedding, 1000-step linear-β DDPM,
128×128 (prototype at 64×64), `mono-frames` only, single style first then all 10 train styles.
Deliberately the most vanilla DDPM possible — you're reverse-engineering the mechanism, and
the point is to feel where it's fragile, not to win a benchmark.

Then: run the samples through `sprite-dx-anti-corruption-v1` for the crisping pass, and show
before/after in the Space. Two Sprited models in one demo.

---

## 8. Week plan

**Hard gate: the dataset ships on day 4, before any model work begins.** That guarantees the
week produces an artifact regardless of how training goes.

| day | work | gate |
|---|---|---|
| **1** | Skeleton, renderer, halo pass, `walk` cycle. Contact sheet + GIF. | *Does it look alive?* If not, fix before scaling. |
| **2** | Remaining 11 motion loops. Artistic direction pass (§5). | 12 GIFs that each read instantly at 128px. |
| **3** | Variation sampler, label schema, generator CLI, determinism test. | Same seed → byte-identical output. |
| **4** | Generate 262k frames, shard to parquet, dataset card w/ hero grid, **push to HF**. | 🚢 **Dataset is live.** |
| **5** | Baseline DDPM v0. Prototype at 64×64, then 128. | Recognizable figures from noise. |
| **6** | Eval harness: pose regressor, SRE, BLD, TVR. v1 conditioning if v0 landed. | Three real numbers on the card. |
| **7** | Model card, HF Space demo (live generator + before/after crisping), writeup, post. | 🚢 **Model + Space live.** |

---

## 9. Distribution — the "3 likes" problem

Likes are a *legibility* problem, not a quality problem. Someone decides in about ten seconds.

Required on the dataset card, in order:

1. **A hero GIF grid at the very top** — 12 styles animating side by side, above the fold.
   Nothing else matters as much as this. A dataset card that opens with a wall of YAML is dead.
2. **One-sentence "what is this"** and one-sentence "why you'd use it."
3. **A copy-pasteable `load_dataset` snippet** that works with no arguments.
4. **The eval hook** (§6) stated in two lines, with the meme framing.
5. Honest provenance: 100% synthetic, CC0, generator included, regenerate at any scale.

Plus:
- **An HF Space** where you drag a slider and it renders a stick figure live. Interactive
  Spaces get an order of magnitude more engagement than static cards, and this one is
  ~50 lines of Gradio because the generator is already pure-CPU and fast.
- Cross-link the dataset ⇄ the DDPM ⇄ `sprite-dx-anti-corruption-v1` so the org reads as a line.
- One post to r/MachineLearning or X with the "measurable hands" framing.

Honest note: three likes is achievable but it is noise, and optimizing for it past the above
is a waste of your week. The card and the Space are worth doing because they make the artifact
*usable*, and usability is what you're actually building a reputation on.

---

## 10. Risks and kill criteria

| risk | mitigation | kill criterion |
|---|---|---|
| Figures look like clip art, nobody cares | §5 artistic direction is day 1–2, gated | If day 2 GIFs don't read as alive, stop and rethink the visual before generating 262k frames |
| DDPM doesn't converge in the time available | Prototype at 64×64, single style first | Day 6 EOD: if no recognizable figures, ship dataset only and write up the failure honestly |
| Dataset is *too* easy → model memorizes, learns nothing | Variation space is large; check train/test nearest-neighbour distance | If val loss ≈ train loss and samples are near-duplicates of training images, widen the parameter ranges |
| Scope creep into video gen | v2/v3 are explicitly week 2+ | Any day-5+ work on temporal models before v0 works is scope creep |
| A week spent on infrastructure | Parquet + `load_dataset`, no custom tooling | If day 4 isn't a push, cut the `styled-frames` config |

The "if all fails" case you named is already the design: the dataset ships on day 4 and is
independently useful. Worst realistic outcome is a clean, well-documented, permissively
licensed synthetic dataset with a nice Space, and a writeup about why a from-scratch DDPM
was harder than expected. That's still a credible week for a lab.

---

## 11. Naming

Proposed:

- Dataset: **`sprited/stickdance-128`**
- Model: **`sprited/stickdance-ddpm-v0`**
- Local repo: `dancing-stick-figure` (as-is)
- Benchmark metrics: `SRE`, `BLD`, `TVR`, `LIE`

The `-128` suffix does real work — it signals the canvas, matches the existing model's
resolution, and leaves room for `stickdance-256` and `stickdance-natural` later without
renaming anything.

---

## 12. Open questions for Jin

1. ~~ARDY~~ — resolved, see §2.5. Week-2 motion source.
2. ~~GPU~~ — `gin`: RTX PRO 6000 Blackwell, 96 GB, Ubuntu 22.04, torch 2.8+cu128. Skip the
   64×64 prototype; go straight to 128. v1 conditioning fits in the week. ARDY runs there too.
3. **Scale** — 262k frames is my proposal. Generation is nearly free, so the real constraint
   is your upload bandwidth and how much you want to eyeball for quality. Bigger or smaller?
4. ~~Mono vs styled~~ — resolved: colour is default (§3.1b), `styled-frames` cut, `mono` kept
   as the hard variant.
5. **Ship cadence** — dataset on day 4 as a hard gate, or would you rather hold everything
   and ship dataset + model together on day 7? I strongly favour day 4; shipping early is
   what makes the week un-loseable.

---

## 13. The longer arc — "touch of genius" datasets, and the moddable game

Jin's framing mid-discussion: Sprited should be able to make *touch-of-genius datasets* —
and, further out, a full game that researchers can mod and train on.

Both of those are the same idea at two scales, and `stickdance-128` is the smallest honest
instance of it. Naming the pattern so we can repeat it:

**A touch-of-genius dataset is one where the generator is a first-class artifact and ships
with its own oracle.** Not "here are 100k images," but "here is a world, here is the code that
renders it, and here is a function that tells you whether *your* model's output is a valid
inhabitant of that world." The dataset is a by-product; the *world + oracle* is the product.

That's what §6 is. `stickdance.eval(samples)` is the touch. Everything else in this document is
competent execution; §6 is the reason someone forwards the link.

The moddable game is the same shape, bigger:

| | stickdance-128 (this week) | "the game" (quarter+) |
|---|---|---|
| world | 15-joint 2D skeleton | a real playable environment |
| generator | 800 lines of Python | the game engine + headless render mode |
| oracle | SRE / BLD / TVR | game-state checks: physics validity, rule consistency, agent legality |
| moddable via | `pose(t)` interface, params | scripting layer, asset packs, level format |
| research uses | image/video gen, pose cond. | world models, video gen with controllable state, RL, agent eval |
| labels | exact pose | exact full game state per frame — the thing nobody else can give you |

The game version's pitch writes itself: every generated frame comes with the *complete*
underlying state, and modders can add mechanics/characters/styles and instantly get a new
labelled dataset with the same oracle. That's a Sprited-shaped product — sprites, games,
small models — and it's a genuine moat because the oracle is only possible if you own the
renderer.

Don't build the game this week. Do build this week's thing so that its bones are the game's
bones: **the `pose(t) → render()` seam, the seeded param vector, and the eval hook are the
three interfaces to keep clean.** If those three survive contact with the DDPM, the game is
"replace the renderer" not "start over."

## Appendix A — generator interface sketch

```python
# generator/motion.py
def pose(style_id: int, t: float, params: Params) -> JointAngles:
    """t in [0,1), cyclic. The ONLY seam between motion source and renderer —
    swap this for mocap/model-driven motion in v2 without touching anything else."""

# generator/render.py
def render(angles: JointAngles, params: Params, size: int = 128) -> Image:
    """Vector -> 4x supersample -> box downsample. Halo pass for depth legibility."""

# generator/sample.py
def sample(seed: int) -> Params: ...

# CLI
$ python -m generator --config mono-frames --n 65536 --seed 0 --out shards/
$ python -m generator --config clips-32   --n 4096  --seed 1 --out shards/
$ python -m generator --verify --seed 0        # asserts byte-identical reproduction
```

## Appendix B — first thing to build

Day 1, first hour, before anything else: render **one** figure in a T-pose at 128×128 with
the halo pass, and look at it. If a static stick figure doesn't read cleanly at that
resolution with a 4px stroke, every downstream assumption in this document is wrong and we
need to know immediately.

**Done** — `scratch/tpose_probe.py` → `scratch/probe_128.png` (and `_x3` for viewing).
Result: T-pose and a floss-style crossing pose both read cleanly at 128px / 4px stroke.
The halo pass visibly separates the crossing arms from the torso.

One finding to carry into day 1: a **naive per-bone halo erases the joint it shares with the
previous bone** (T-pose arm gets a nick at the shoulder). Fix is to halo per *chain*
(shoulder→elbow→wrist as one polyline, halo pass then stroke pass), not per bone. Ten-minute
fix, but it would have been a nasty artifact across 262k frames if it went unnoticed.
