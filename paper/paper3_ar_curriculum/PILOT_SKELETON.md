# Paper 3 pilot — From Data to Rollout

Working title:

> **From Data to Rollout: A Course-Scale Laboratory for Autoregressive Video Diffusion**

Alternative title:

> **Learning Autoregressive Video Diffusion by Building It**

## One-sentence thesis

A complete block-autoregressive video-diffusion experiment—including the data, causal training example, generated-history rollout, structural state, and failure measurements—can be made small enough to train, modify, and diagnose on one conventional GPU.

## What this paper is

Paper 3 is the educational and systems synthesis of the first two papers. It does not merge their claims.

- Paper 1 supplies the dataset, renderer, exact rig, reconstruction contract, and diagnostic measurements.
- Paper 2 supplies the block-autoregressive research result: the static-copy failure, next-block evaluation, and rig co-generation.
- Paper 3 turns those components into a coherent route through the learning problem, beginning with the autoregressive factorization rather than ending with it.

The artifact is not one long Colab. It is a small sequence of runnable labs with one shared configuration and checkpoint contract.

## What this paper does not claim yet

- It does not claim that students learn better without a controlled learner study.
- It does not call the model competitive with frontier video generators.
- It does not call training “from scratch” when the route uses a frozen text encoder or a supplied codec.
- It does not use visual inspection as evidence of prompt following.
- It does not present Paper 1 or Paper 2 results as new results.

## Pilot abstract

Training a video generator is rarely the entry point for learning how video generation works. Public routes commonly begin with inference or adaptation of a pretrained system, leaving the data construction, temporal factorization, train–test gap, and long-horizon failure modes hidden. We present a course-scale laboratory in which the first video model is block-autoregressive: it denoises a bounded future block from an immutable history and then continues from its own generated output. The laboratory uses Dancing Stick Figures, whose compact videos retain the exact rig and rendering state behind every frame, and decomposes the experiment into runnable stages for inspecting the training target, fitting the generator, rolling it beyond one block, measuring accumulated error, and adding an explicit structural channel. A 32-pixel route supports short smoke tests, while the 64-pixel reference route trains a roughly 40M-parameter model on one conventional GPU. The system includes matched pixel-only and rig-co-generating checkpoints, structural and motion diagnostics, and an autoregressive next-block evaluation that separates teacher-forced prediction from free-running continuation. This report specifies the learning sequence, resource envelope, reproducibility contract, and the evidence still required before making claims about educational effectiveness.

## Central research questions

1. **Feasibility:** Can the complete AR training and rollout path fit one conventional GPU and produce an inspectable multi-second result within hours rather than days?
2. **Inspectability:** Can every important state—the clean history, noisy future block, loss mask, generated history, predicted rig, and evaluator output—be exposed without changing the reference implementation?
3. **Diagnostic value:** Can a learner reproduce a failure that ordinary validation hides, such as good teacher-forced loss with poor free-running continuation?
4. **Modifiability:** Can one bounded architectural change be made and evaluated in the same session or course unit?
5. **Learning effectiveness:** Do learners understand the AR factorization and train–test gap better after using the laboratory? **[Requires a learner study; not answered by model metrics.]**

## Reference learning path

### Lab 0 — Read one video example

- Inspect one 120-frame clip, its prompt, rig, camera, and rendering labels.
- Render the same motion from another released camera.
- Identify which quantities are observations and which are known generating state.

**Artifact:** one clip filmstrip plus aligned rig and part labels.

### Lab 1 — See the autoregressive training example

- Construct an immutable history prefix and a noisy future block.
- Display which frames enter the loss.
- Show teacher forcing as a data operation, not only as prose.
- Run one denoising step and inspect tensor shapes.

**Artifact:** training-example diagram generated from the actual batch.

### Lab 2 — Train the first block-AR generator

- Use the 32² smoke configuration or the 64² reference configuration.
- Train the generator directly on future-block prediction.
- Keep the architecture source visible and ordinary; do not expose a menu of speculative backbone toggles.
- Save validation loss, a fixed-noise block sample, peak memory, and wall time.

**Artifact:** first-block checkpoint and generated block.

### Lab 3 — Roll from generated history

- Generate several blocks while carrying only the model’s completed history.
- Compare teacher-forced next blocks with free-running rollout.
- Mark block boundaries in the viewer without modifying the generated pixels.

**Artifact:** five-second rollout, teacher-forced/free-running pair, and boundary diagnostics.

### Lab 4 — Diagnose the static-copy shortcut

- Plot topology, motion fraction, angular jerk, and next-block divergence over training.
- Demonstrate that a lower reconstruction or flow loss does not certify useful continuation.
- Compare the short-horizon baseline with the repaired 16-frame-block recipe.

**Artifact:** one failure curve and one matched rollout comparison.

### Lab 5 — Add explicit structure

- Append one rig token per latent frame.
- Train pixel-only and rig-co-generating variants under a matched budget.
- Decode the pixels and render the co-generated rig separately.
- Measure pixel continuation, recovered-rig continuation, bone drift, and rig–pixel self-consistency.

**Artifact:** v8/v9 comparison with the changed code region highlighted.

### Lab 6 — Make and test one change

Suggested assignments change one variable, not a grid of exposed options:

- history length;
- future-block length;
- generated-history corruption;
- rig-token removal; or
- one additional conditioning signal.

The submission includes the hypothesis, fixed protocol, generated video, measurements, and failure interpretation.

## System design

### Default route

- Resolution: 64² RGBA; 32² smoke route.
- Dataset: released Dancing Stick Figures mini tier.
- Codec: supplied, frozen causal video VAE.
- Generator: approximately 40M-parameter full-spatiotemporal DiT.
- Training mode: block autoregression with clean history and a noisy future block.
- Generation block: 16 video frames at 20 fps.
- History: variable, up to approximately one second.
- Objective: rectified-flow prediction with visible foreground and motion weighting.
- Text: frozen T5-small conditioning.
- Optional structural channel: one 27-joint 2D rig token per latent frame.

### Optional codec route

The codec is a separate lab, not a prerequisite for the first AR experiment. Learners may train a small video VAE and quantify what thin limbs, temporal compression, and causal decoding lose. The main route supplies the validated codec so that generator learning is not blocked by a second long training run.

### Meaning of “from scratch”

The safe claim is:

> The learner trains the autoregressive video-diffusion backbone from random initialization.

“Complete stack from scratch” is reserved for the optional route that also trains the codec. Neither phrase implies training the text encoder.

## Paper outline

### 1. Introduction

- Open weights often expose inference but not the complete learning problem.
- Autoregression is usually introduced after image or fixed-window generation, although it changes the training example, deployment interface, and evaluation.
- State the feasibility, inspectability, and diagnostic questions.
- Contributions: the learning sequence, reference implementation, resource/reproducibility envelope, and evaluation tasks.

### 2. What must be visible to learn autoregression

- Immutable history versus denoised future.
- Teacher forcing versus generated context.
- Block boundaries and accumulated error.
- Why fixed-window quality scores are insufficient.
- Exact structural state as an answer key.

### 3. The laboratory

- Dataset and reconstruction contract, citing Paper 1.
- Codec and generator.
- Runnable lab sequence.
- Shared configuration, checkpoint, seed, and manifest contracts.
- 32² smoke and 64² reference paths.

### 4. Autoregressive evaluation

- Teacher-forced next-block divergence.
- Free-running divergence.
- Same-prompt real-pair floor.
- TVR, LIE, CPE, motion fraction, jerk, and boundary excess.
- Rig-space evaluation and confidence limitations.
- Motion-semantic scoring remains outside the current metric set unless the learned motion encoder is validated.

### 5. Reference experiments

- Naive short-horizon AR and the static-copy shortcut.
- Repaired pixel-only v8.
- Rig-co-generating v9.
- Two-seed continuation replication.
- Resource measurements on the intended GPU classes.

### 6. Reproducibility and modifiability

- Clean-environment execution.
- Time to first sample and time to reference checkpoint.
- Peak memory and checkpoint sizes.
- One documented student-scale modification completed end to end.

### 7. Educational evaluation

Two admissible versions:

1. **Systems paper without learner claims:** report completion, resource, reproducibility, and experiment-modification evidence only.
2. **Education paper:** add a preregistered learner study measuring whether participants can explain and diagnose teacher forcing, exposure bias, and block-boundary failure.

Do not substitute Reddit response, downloads, or anecdotal enthusiasm for learning evidence.

### 8. Limitations

- Synthetic single-character domain.
- Supplied codec and pretrained text encoder in the default route.
- Small model and short course-scale training budgets.
- No validated open-vocabulary prompt-adherence metric.
- Rig supervision is unusually exact and does not transfer automatically to natural video.

### 9. Conclusion

The contribution is a complete, repeatable learning experiment: build the AR training example, fit it, roll it from its own output, observe the gap, change the representation, and measure whether the change helped.

## Planned figures

1. **The AR training example:** real frames, immutable history, noisy target block, loss mask, and predicted clean block.
2. **The laboratory path:** data → block construction → training → free rollout → rig co-generation → evaluation.
3. **Teacher-forced versus free-running:** the same checkpoint and prompt, with block boundaries visible.
4. **Static-copy progression:** structure improves while motion/continuation degrades.
5. **One-code-change comparison:** pixel-only v8 versus v9 with rig tokens.
6. **Student-facing output:** one compact report containing video, protocol, resource use, and diagnostic interpretation.

## Planned tables

### Table 1 — Resource envelope

| Route | GPU | Resolution | Steps | Peak VRAM | Wall time | Time to first rollout |
|---|---|---:|---:|---:|---:|---:|
| 32² smoke | [TODO] | 32² | [TODO] | [TODO] | [TODO] | [TODO] |
| 64² reference | [TODO] | 64² | [TODO] | [TODO] | [TODO] | [TODO] |
| 64² full reference | Gin | 64² | 100k | [TODO] | 1.9 h on the recorded training host | [TODO] |

### Table 2 — What each stage demonstrates

| Stage | Model state | Required output | Concept tested |
|---|---|---|---|
| First block | random-init AR | one denoised block | future-block objective |
| Rollout | same checkpoint | multi-block clip | generated context |
| v8 | repaired pixel-only AR | metrics + video | static-copy repair |
| v9 | rig co-generation | pixels + rig + metrics | explicit structural state |

### Table 3 — Reference AR results

Reuse the frozen Paper 2 numbers with explicit citation; do not imply they are newly produced by Paper 3.

| Model | Budget | TVR | Jerk | TF divergence | Free-running divergence |
|---|---:|---:|---:|---:|---:|
| v8 pixel-only | 100k | .122 | .095 | .314 | .458 |
| v9 + rig, seed 0 | 100k | .126 | .084 | .287 | .440 |
| v9 + rig, seed 1 | 100k | .129 | .091 | .292 | .439 |

### Table 4 — Reproduction and modification

| Attempt | Environment | Unmodified run complete? | Metrics within tolerance? | One modification complete? |
|---|---|---|---|---|
| Author reference | [TODO] | [TODO] | [TODO] | [TODO] |
| Clean independent run | [TODO] | [TODO] | [TODO] | [TODO] |
| Learner study, if run | [TODO] | [TODO] | [TODO] | [TODO] |

## Evidence ledger

### Already available

- Dataset, renderer, reconstruction, and diagnostic contracts from Paper 1.
- v8/v9 matched 10k and 100k continuation evaluations from Paper 2.
- v9 100k seed replication.
- SRE validation and rig-space rescoring.
- Fixed-prompt five-second AR rollout artifacts.
- Recorded 100k v9 training time on Gin.

### Required for a credible pilot release

- [TODO] One clean, scripted 32² end-to-end run.
- [TODO] One clean, scripted 64² end-to-end run on the advertised hardware class.
- [TODO] Peak memory, wall time, and time-to-first-rollout manifests.
- [TODO] A notebook or viewer that exposes the actual history/target/loss-mask tensors.
- [TODO] A minimal assignment that changes one variable and regenerates the report.
- [TODO] Independent reproduction from release instructions.

### Required only for an education claim

- [TODO] Target learner population and prerequisites.
- [TODO] Preregistered concepts and scoring rubric.
- [TODO] Pre/post or controlled comparison.
- [TODO] Completion rate and qualitative error analysis.
- [TODO] Consent and privacy procedure if human participants are involved.

## Pilot go/no-go gate

Proceed from skeleton to full Paper 3 only if all of the following are true:

1. A clean user can complete the 32² path without undocumented intervention.
2. The 64² path fits the advertised GPU and produces a multi-block rollout in the declared time envelope.
3. The laboratory visibly exposes the AR training example and generated-history transition.
4. At least one bounded modification can be trained and evaluated without editing infrastructure code.
5. Every numerical result is generated into a manifest rather than copied by hand into the paper.

If these gates fail, release the material as a course/lab package without forcing a third research paper.

