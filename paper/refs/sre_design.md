# SRE (Skeleton Regression Evaluator) — design sketch (2026-08-24)

Purpose: pixels → cskel27 2D rig regressor, so ANY model (rig-free v8, rig-cogen
v9) is scorable in rig space. Third layer of the evaluation stack (latent
flow-NLL / pixel next-block divergence / rig-space divergence). Promoted to
REQUIRED evidence for paper2 (Jin, 2026-08-24): enables the "rig-space
divergence" table and validates v9's self-reported rig against its own pixels.

## Training data — free and exact

- `cache/mini`: frames.npy [514800, 64, 64, 4] + rig.npy [514800, 27, 2]
  (bit-exact normalized joint_xy) + rig_depth.npy. Same prompt-disjoint split
  discipline as everything else: train on train-split frames only.
- Input: single RGBA frame (premultiplied, [0,1]). Optionally a 3-frame window
  later for occlusion disambiguation — v1 is single-frame.

## Architecture (v1, deliberately small)

- Conv encoder (4 stages, 32→256 ch, stride 2) → 4x4 feature map → flatten →
  MLP → 54 outputs (27 joints × 2, in [0,1]).
- ~2M params. Loss: L2 on visible joints; optionally weight by inverse depth
  order later. No heatmaps in v1 (64² is small; direct regression suffices as
  a first instrument — upgrade to heatmap head only if PCK is poor).

## Validation gates (declare before training)

1. Held-out real renders: mean joint error in px at 64²; PCK@2px / PCK@4px.
   Target: mean error well under the capsule radius (1.6 px limbs).
2. Malformed-render sanity (reuse the corruption harness): swapped-limb and
   extra-arm renders must produce large, localized joint errors — the
   regressor must not hallucinate a clean skeleton on broken inputs.
3. Off-screen handling: sit-up clip frames with joints outside [-0.5, 1.5]
   must not destabilize nearby-joint predictions.

## Usage once validated

- Rig-space next-block divergence: run SRE on generated + GT blocks from
  eval/next_block_divergence.py outputs → per-joint distance, bone-length
  stability, end-effector trajectory divergence.
- v9 self-consistency: SRE(decoded pixels) vs the model's own rig tokens —
  measures whether the co-generated rig actually describes the pixels.
- Retrieval-style prompt alignment in rig space (existing z-scored signature
  machinery in scripts/rig_alignment_metric.py applies unchanged).

## Boundary with the testbed paper

The testbed report lists a "geometry track (TODO)" — SRE is that track's
first instrument. Coordinate with Pixel before claiming it there; paper2 uses
it first. One checkpoint, cited by both, no duplicated result tables.

## Cost estimate

Training: minutes-to-an-hour on gin (2M params, 64² frames, memmap reads).
Fits any GPU gap; does not contend with flagship runs.
