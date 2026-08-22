# Training run protocol

Every paper-facing training run must be observable while it is running and
recoverable after interruption. A final checkpoint without intermediate
evidence is not considered a completed run.

## Dataset preflight gate

Before a paid run starts, load the exact cache from the training pod and save
a deterministic grid of at least 64 ground-truth samples.  Record the array
shape, dtype, clip count, metadata/manifest hashes, and visually verify the
intended character style, channels, compositing, and resolution.  A filename
or shape check alone is insufficient: the gray-mini cache and colored
stick-figure cache can both be valid 64x64 RGBA arrays.  Preserve a failed
preflight as evidence and do not use its training outputs in model-quality
comparisons.

For K1, run `scripts/dataset_preflight.py --profile colored_k1_v1` against the
exact pod cache before training **and** evaluation. The gate verifies full-file
SHA256 values plus array shape, dtype, clip count, and split counts, writes a
pass/fail JSON record, and optionally emits the deterministic visual grid. A
run without a passing preflight record is not paper-facing evidence.

## Required artifacts

- `args.json`: complete resolved configuration, seed, parameter count, code
  revision/diff identifier, dataset manifest hash, GPU type, and start time.
- `log.txt` or structured metrics: training loss, validation loss, learning
  rate, throughput, peak VRAM, elapsed time, and estimated remaining time.
- `latest.pt`: resumable model, optimizer, scheduler/scaler, RNG, and step
  state, written atomically.
- `ckpt_STEP.pt`: immutable milestone weights retained at a fixed interval.
- `sample_STEP.*`: inference from a frozen evaluation manifest at the same
  milestones.
- `sample_manifest.json`: exact prompts, sample seeds/noise seeds, sampler,
  guidance, checkpoint, and output filenames.
- `COMPLETE`: written only after the final checkpoint and artifacts pass an
  integrity check and are copied off the training pod.

## Inference panels

Each milestone must contain three controlled panels:

1. fixed noise, different held-out prompts (text sensitivity);
2. fixed prompt, different noise seeds (sample diversity);
3. a stable prompt-diverse grid reused across runs (visual progression and
   architecture comparison).

Do not select only attractive samples. Preserve the complete frozen panel and
label every output with prompt and seed.

## Default cadence

Cadence is expressed as a fraction of the total budget so short pilots and long
runs remain comparable:

- log: approximately every 1% of training;
- validation: every 10%;
- inference panel: at 0%, 10%, 25%, 50%, 75%, and 100%;
- immutable checkpoint: at 10%, 25%, 50%, 75%, and 100%;
- resumable latest checkpoint: at least every 10% and before planned shutdown.

For a newly introduced architecture or training stage, preserve a dense early
visual trace at steps `0, 1, 5, 10, 25, 50, 100, 250, 500`, then at regular
milestones. Early previews may use four frozen prompts and a reduced sampler
step count to avoid making inference dominate training; milestone previews use
the full sampler. Save and display both raw-model and EMA outputs while they
meaningfully differ. Show failed and malformed generations as well as good
ones—these are diagnostics, not a curated demo reel.

For the current 3,000-step K1 pilot, the existing 500-step cadence is stricter
than this minimum and remains unchanged.

## Retention and stopping

- Never delete or overwrite an immutable milestone checkpoint during a run.
- Copy final artifacts and hashes off the pod before termination.
- Stop early only using a recorded criterion such as non-finite loss, collapse,
  prompt insensitivity, or no improvement across two consecutive milestones.
- Preserve failed runs and their manifests; failures are part of the research
  trail.
