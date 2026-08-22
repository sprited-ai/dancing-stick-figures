# Experiment log

This file records protocol mistakes and negative runs as well as successful
experiments.  A run is canonical only when its saved `args.json` matches the
fixed 64x64, 50-frame, 10-fps protocol.  The source videos are 20 fps, so a
canonical 10-fps run must use `stride=2`.

## K1 scratch pilot -- useful but non-canonical

- Architecture: factorised text-conditioned pixel Video DiT, patch 4, 39.9M
  parameters, rectified flow, frozen T5-small.
- Budget: 3,000 steps, batch 16, one seed.
- Saved arguments: 64x64, 50 frames, **`stride=1` (20 fps)**.
- Outcome: validation loss improved from 0.0486 at step 1,000 to 0.0283 at
  step 3,000; samples visibly improved but did not pass the quality gate.
- Interpretation: an under-trained engineering pilot and scratch reference.
  It must not be reported as a result on the canonical 10-fps track.
- Artifacts: `pod_results/k1_t2v50_64px_4090_b16_3k/`.

## T2I gray-data mismatch -- excluded

- Intended task: patch-4, text-conditioned 64x64 T2I curriculum stage.
- Actual cache: a gray mini/chibi figure cache, not the colored stick-figure
  cache used by the video experiments.
- Action: stopped near step 2,000, preserved for debugging, and excluded from
  every quality or architecture comparison.
- Evidence: ground-truth grid and `DATASET_MISMATCH.md` in
  `pod_results/t2i64_text_p4_fg2_30k/`.
- Protocol change: every long run now requires a rendered ground-truth grid
  to pass visual review before training starts.

## T2I colored-data foundation -- completed

- Architecture: same 39.9M Video DiT operated at `T=1`; temporal attention is
  skipped, while spatial, text cross-attention, time-conditioning, and output
  weights train.
- Data preflight: `(514800, 64, 64, 4)` uint8 frames, 4,290 clips; the visual
  grid confirms black heads and colored thin limbs.
- Configuration: patch 4, batch 128, text CFG dropout 0.1, foreground weight
  2, 30,000 steps, seed 0.
- Early result: val 0.0947 at step 500 and 0.0589 at step 1,000. Human and
  colored-limb structure is visible by step 1,000. The verified final
  checkpoint reached step 30,000 and seeded the matched video run below.
- Artifacts: `pod_results/t2i64_color_text_p4_fg2_30k/`.

## Canonical scratch vs T2I-warm-start video run -- completed

- Both matched configurations reached 10,000 video-stage steps at 64x64,
  50 frames, `stride=2`, with 70% T2V, 20% I2V, and 10% T2I draws.
- Correct colored-cache n=64 evaluation: warm-start lowers TVR from 0.406 to
  0.107 (paired delta CI95 `[-0.336,-0.261]`) and final validation loss from
  0.0159 to 0.0118. It also worsens centroid speed, acceleration, and motion
  fraction relative to scratch and the real-data floor.
- Prompt/noise sensitivity ratio rises from 0.360 to 0.550, but semantic text
  correctness is not measured.
- The warm run recovered from a verified step-6,000 full checkpoint after
  remote disk exhaustion; RNG/dataloader ordering restarted after recovery.
- Full results and the invalid grayscale-cache audit are in
  `pod_results/k1_final_eval_n64/REPORT.md`.

## Matched 10k scratch control -- completed

- Purpose: provide a matched control for the image-pretrained video run; the
  older K1 pilot cannot serve this role because it used `stride=1`.
- Configuration: 64x64, 50 frames, `stride=2`, patch 4, batch 16, seed 0,
  foreground weight 2, 10% image batches, 20% conditional I2V draws among
  video batches, and otherwise T2V.
- Budget: 10,000 steps with immutable 0/1/5/10/25/50/100/250/500/... artifacts.
- Data/code preflight: colored reference grid passed; 514,800 frames and
  4,290 clips; `video_dit_fm.py` SHA256
  `511ba614a391680050d7f5c2dffc4981fa7b15588c098130ceef071b3572e08c`.
- Status: step 10,000 full and EMA checkpoints, logs, manifests, and fixed
  samples were checksum-verified locally before the pod was terminated.

### Comparison limits

The video-stage architecture, data mixture, optimizer schedule, seed, and
10,000-step budget are matched, but the result is still one paired seed. Do
not claim broad architecture ranking or statistical generality. Also do not
claim total-compute efficiency: the warm-start arm consumes an additional
30,000 T2I steps. The warm run's step-6,000 recovery is disclosed above.

## M6 H8 20k milestone evaluation -- completed (2026-08-22)

- Rule declared at training step 15,050, before any late-milestone result
  existed (`configs/m6_evaluation_v1_h8_5k_20k.json`): evaluate 5k/10k/15k/20k
  under identical n=64 prompts, noise seeds (20260824+), 10-step sampler, CFG 2.
- Result: frame structure improves monotonically (TVR .745@2k -> .349@20k, LIE
  .261 -> .188; real floors .133/.110) while motion collapses by 10k and
  flatlines: centroid speed .300 -> .142 (real .314), motion fraction .403 ->
  .155 (real .371), height variance -> .010 (real .061). Prompt/noise L1 ratio
  oscillates (.766/.628/.852/.800) with no monotone trend.
- Pareto front over topology/motion is {2k-5k, 20k}; no checkpoint approaches
  both floors. Checked-in summary: `paper/results/m6_h8_milestones_n64.json`.
- Provenance: evaluations ran on gin under the after-complete watcher script
  (`scripts/eval_m6_h8_20k_after_complete.sh`); all four output directories
  passed SHA256 manifests; collected to `pod_results/m6v3_start_aligned_h8_20k_s0/`.

## R0 full-clip latent control -- completed (2026-08-22)

- Matched control per `configs/r0_full_clip_latent_protocol_v1.json`: same
  frozen f8t4d16 codec + stats hashes, same 39.8M full-ST DiT family, text
  encoder, flow objective, optimizer, batch 16, seed 0; denoises all 25
  latents (100 frames) jointly, no history prefix; declared 10k endpoint.
  Final val_full_clip 0.349. Trained on gin (~50 min, 0.30 s/it, 9.1 GB peak).
- n=64 evaluation, same prompts/noise/sampler as the M6 milestones (10-step):
  TVR .456, LIE .205, centroid speed .382, motion fraction .540, height
  variance .043, angle jerk .220 (real .059). 20-step sampler: TVR .426,
  speed .338.
- Reading: R0 does not freeze -- it mildly overshoots motion with worse
  topology than M6@20k (.456 vs .349) and high jerk. With parameters,
  objective, codec, and data shared, the M6 motion collapse is attributable to
  teacher-forced short-horizon block factorization, consistent with the
  h4/h8/h40 horizon dial. Diagnostic comparison, not a ranking: single seed
  per arm, and R0's budget follows its own declared endpoint (10k vs 20k).
- Artifacts: gin `results/r0_f8t4d16_fullst_10k_s0/` (+ `eval_n64/` with
  SHA256SUMS), collected to `pod_results/r0_f8t4d16_fullst_10k_s0/`.
