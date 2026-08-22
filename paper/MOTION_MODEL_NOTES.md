# S0 motion track: experiment contract

S0 is the modular structured control for the direct pixel-video K1 model:

`text -> 50-frame cskel27 motion -> existing exact renderer -> 64x64 video`

It is not a joint video-motion model. Its purpose is to separate motion-generation
errors from rendering errors and establish whether explicit structure is already a
strong solution for the current synthetic domain.

## Fixed target

- 50 frames at 10 fps, starting at source time 0, aligned with the canonical video task.
- Local pose: hips-centred 27-joint XYZ in the frame-0 figure basis, metres.
- Global motion: separate root displacement XYZ and relative heading `(cos yaw, sin yaw)`.
- Four binary foot-contact channels when the ARDY record contains them.
- Train-only, deterministic mean/std normalization for local joints and root. Heading
  remains on the unit circle; contacts remain binary.

## First model and losses

Start with a small text-conditioned 1D motion DiT using flow matching over the continuous
`[local joints, root, heading]` sequence. Contact is either an auxiliary BCE head or a
continuous channel thresholded only at evaluation.

The matched baseline objective is flow loss alone. Add the following one at a time:

1. bone-length deviation from the source body;
2. joint velocity/acceleration against the target, avoiding a static-motion shortcut;
3. contact-conditioned foot skating and foot hover;
4. joint-limit penalties only after limits are defined for cskel27;
5. exact-render consistency only in the later joint model, not S0.

Report both normalized training losses and denormalized metric values in metres/seconds.
Do not call kinematic regularizers “full physics”: the dataset has no mass, force, torque,
or inertial parameters.

## Required controls

- K1: `text -> direct pixel video` under the same prompts, split, 50 frames and 10 fps.
- S0-oracle: ground-truth motion through the exact renderer (rendering upper bound).
- S0: predicted motion through the exact renderer (motion-model error only).
- Text ablation: correct prompt versus shuffled/null prompt.
- Loss ablations: flow only, then each accepted kinematic term under matched compute.
- Evaluation: text-motion retrieval/alignment, MPJPE only where paired-seed comparison is
  meaningful, bone error, foot skate/contact, velocity/acceleration distribution, and
  rendered-video metrics used for K1.

## Missing prerequisites (do not fabricate)

- The released `motion` parquet shards are not currently present in the workspace. There
  are 465 local raw ARDY NPZs, fewer than the released 1,430-clip motion set. The cache CLI
  can join these NPZs to the local frame-parquet metadata for a leak-free *subset pilot*,
  but paper numbers require the complete released motion shards.
- A frozen text encoder/cache shared with K1 has not yet been specified for S0.
- The small motion-DiT architecture, parameter budget, optimizer budget, and seed count
  are not yet frozen.
- cskel27 joint-angle limits and a differentiable renderer have not been implemented.
- Exact camera/body-jitter metadata must be joined before rendering predicted S0 motions
  for a pixel-matched comparison. The motion generator output alone does not contain the
  three per-clip camera variants.
- Prompt-alignment metrics for motion require a declared evaluator; no numerical claim
  should be made until that evaluator and its held-out protocol are frozen.

`train/motion_data.py` implements the auditable data contract and cache.
`train/motion_dit_fm.py` implements the minimal S0 text-conditioned temporal DiT/flow
baseline. Neither changes or interrupts the active K1 training run. S0 numbers remain
TODO until the full leak-free cache and a declared GPU pilot are run.

The S0 CLI validates every 250 steps, refreshes its full resumable `latest.pt` every 100
steps, and every 500 steps stores a milestone checkpoint, fixed-prompt/fixed-noise NPZ,
64px diagnostic GIF and frame strip with a JSON manifest. `latest.pt` contains the
optimizer, EMA and RNG state and is resumed automatically.
