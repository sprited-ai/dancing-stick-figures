# v9 design: rig co-generation (pixels + cskel27 jointly)

Jin's direction (2026-08-23): no ARDY at inference, training data unchanged.
The model emits the rig (27-joint 2D screen coordinates per frame) alongside
the video. "Reasoning, metric and loss all become easier." Not a pure video
model any more — positioned as a structured / analysis-by-synthesis baseline
family in the testbed; the black-box interface is still text→video with the
rig as a bonus output.

## Why each of the three gets easier

- **Loss**: joint-space MSE is free supervision (synthetic data ⇒ perfect
  labels). v8's motion-weighting and foreground-weighting are workarounds for
  the objective not seeing the figure; a rig loss injects the motion signal
  directly (targets the height-variance and hold-still failures measured in
  the weakness report).
- **Metric**: semantic correctness becomes measurable on the emitted rig
  (e.g. wave = periodic wrist oscillation vs raise-and-lower = single bump)
  without first building a pose regressor. Per-prompt joint-trajectory
  comparison against ground truth (retrieval / DTW / Fréchet) runs on model
  output directly.
- **Reasoning**: the model is structurally encouraged to plan in rig space.

## Variants

- (a) **Rig readout head**: auxiliary regression head over DiT features →
  joints of the recovered clean target; aux MSE loss. Cheap; rig is an
  observation, not part of the generative state.
- (b) **Rig co-denoising** (closer to intent): one rig token per temporal
  latent index (27×2×4 = 216 dims projected to model dim), appended to the
  token sequence and denoised jointly under the same rectified flow. Rig is
  generative state; flow loss covers it natively.

Pilot order: (b) primary, (a) as fallback/ablation if (b) destabilizes.

## Integrity guard (self-report loophole)

Scoring the model with its own emitted rig is self-reported. Guard: render
the emitted rig with the procedural renderer and compare with the generated
pixels (alpha IoU / foreground overlap) — a consistency metric that must be
reported next to any rig-based semantic metric.

## Prerequisite: rig cache

cache/mini has frames+text only. Need rig.npy aligned with frames.npy
(per-frame 27×2 screen coords; bone_scale applied; per-clip camera
projection reproduced from the generator). Scoping in progress (subagent):
exact projection call chain, clip-id ↔ npz ↔ camera mapping, gotchas.

## Protocol discipline

- v9 protocol id family in `_protocol_id` (rig treatments are separate from
  v5–v8 loss treatments; combos with v8's declared winners allowed as the
  v9 base recipe = v8 + rig, since v8 is the standing best).
- Matched pilot: v9 2k/10k vs v8 same-budget arm; verdict on (i) standard
  oracle metrics unchanged or better, (ii) rig-pixel consistency, (iii)
  per-prompt rig-trajectory alignment vs GT.

## Open questions (pending scoping)

- Is the projection deterministic and exactly frame-aligned with rendered
  pixels (no re-render needed)?
- 2D screen coords vs 3D + camera: start with 2D (what the pixels show).
- Rig token normalization (coords in [-1,1] over the 64² frame).
