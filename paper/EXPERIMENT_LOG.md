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

## M6 motion-collapse fix pilots -- completed (2026-08-22)

Five matched single-variable 2k pilots against the start-aligned H8 2k control
(TVR .745, centroid speed .300, motion fraction .403, height var .023; real
floors .133/.314/.371/.061). Each declared its protocol config before results;
identical n=64 evaluation (seed 20260824, 10-step, CFG 2).

| pilot | treatment | speed | mfrac | TVR | verdict |
|---|---|---|---|---|---|
| A h16 | 16-frame generation blocks | .507 | .682 | .760 | pass (overshoots) |
| B v5 | history-noise augmentation 0.2 | .213 | .276 | -- | fail |
| D v6 | motion-weighted flow loss a=1 | .383 | .523 | .738 | pass |
| E hist24 | 2.4 s teacher-forced history | .187 | .232 | .748 | fail |
| F v7 | foreground-alpha loss weight 4 | .327 | .455 | .757 | pass |

- Reading: both history-side interventions suppress motion further, while all
  three interventions that reallocate training signal toward moving figure
  content revive it. Consistent with the ARDY horizon dose-response and with
  its report that exposure-bias tricks were weak; the 2.4 s-history failure at
  this budget shows long history alone (without ARDY's boundary-anchored
  positional encoding and budget) is not the lever at 2k steps.
- v7 detail: the transparent background dominates the 64x64 frame, so
  unweighted latent MSE spends most of its gradient on cells that never move;
  upweighting figure cells (max-pooled alpha footprint, weight 4, per-sample
  mean-normalized) recovers real-level speed with control-level TVR.
- Infrastructure note: the hist24 evaluation first crashed in the bounded
  sliding decode (history_max 12 over 25 rollout latents leaves a partial
  commit block); eval_m6 now shrinks the decode context to alignment. Verdict
  above is from the fixed rerun.
- Artifacts: gin `results/m6v{3,5,6,7}_*_2k_s0/` with `eval_n64/metrics.json`.

## v9.3 render-coupling pair -- mechanism validated, tuning needed (2026-08-24)

- Pair on the v9 recipe, 10k each, declared before results
  (`configs/m6_protocol_v9p3_{anchor,coupled}_10k.json`): soft-capsule
  renderer matched to the dataset geometry (palette RGB, thumbs dropped,
  head capsule), union-foreground weighting, anchor target = soft-render of
  the GT rig (renderer bias cancels); coupled adds rendered-rig vs
  decoded-predicted-pixel RGBA matching (gradient through both streams).
- anchor (weight 2): pixel tax (TVR .538 -> .645) with no rig gain
  (on-figure .678 == v9.0) -- rejected at this weight.
- coupled (+ weight 2): **on-figure .93** -- rig-pixel binding at 10k that
  the uncoupled recipe only reaches by 100k of convergence -- so the
  cross-modal gradient does exactly what it was designed to do. Costs:
  TVR .721, jerk .307, bone error .213 at this budget/weight; retrieval
  stays at chance (semantic alignment is untouched by coupling, consistent
  with every other lever).
- Reading: Jin's coupling mechanism is real but over-weighted at 2.0 for a
  10k budget; next tuning = lower weight or coupling warmup, or the 32px
  pixel-space pilot to study the coupling without the frozen-VAE decode in
  the loop. Alignment remains a conditioning/data problem.
- Artifacts: gin `results/m6v9p3_{anchor,coupled}_10k_s0/`.

## Seed-1 replication + v9 convergence -- completed (2026-08-24)

- v8 seed-1 100k (`configs/m6_protocol_v8_combined_h16_100k_seed1.json`,
  prediction declared before results: within +-0.03-0.05 of seed 0): at 100k
  TVR .122 / speed .282 / hvar .028 / jerk .095 versus seed-0's
  .122/.286/.028/.095 -- agreement to +-0.005. The Pareto escape replicates;
  the paper's single-seed caveat for this result is retired (n=2 seeds).
- v9 rig-cogen convergence 100k (varied captions per the predeclared winner
  rule): pixel metrics track the rig-free v8 at every milestone (100k: TVR
  .126 / speed .275 / jerk .084) -- the emitted rig remains free at
  convergence, jerk slightly better. Rig-space retrieval stays at chance
  throughout (top-1 0-4.7%), confirming the caption-diversity null at scale.
- Series `v8_100k_seed1_run` and `v9_conv_100k_run` in
  `paper/results/m6_h8_milestones_n64.json`.

## Caption-diversity pair -- syntactic variation rejected (2026-08-23)

- Matched v9-recipe 40k pair: canonical single caption vs 9 semantically
  identical syntactic variants per action drawn per sample
  (`configs/m6_protocol_v9_captions_{canon,varied}_40k.json`).
- Result: no meaningful alignment gain. At 40k, retrieval top-1/5 = 4.7%/15.6%
  (canon) vs 6.2%/17.2% (varied), both near chance (2.7%/13.5%); speed
  pearson .18 vs .15; pixel metrics equivalent. The winner rule (declared
  before results) selects varied on the top-5 tie-break, and the 100k
  convergence run proceeds with it, but the honest verdict is a null result.
- Reading: the alignment floor is not caused by surface-form memorization.
  Rejected levers so far: encoder scale (t5-base), rig-loss strength (v9.1),
  training budget (300k), syntactic caption diversity. Remaining hypotheses:
  semantically richer captions (describing motion content/phases), goal/
  keyframe conditioning (ARDY Sec 3.3), model/data scale.
- Artifacts: gin `results/m6v9_captions_{canon,varied}_40k_s0/`.

## v8 300k extended-budget run -- completed (2026-08-23)

- Fresh 300k cosine run of the v8 recipe (3x budget, Jin's directive),
  milestones per the predeclared protocol config, n=64 each.
- Predictions (written before results): TVR/speed match the 100k run;
  hvar/jerk do not close. Outcome: half right. Angular jerk NEARLY CLOSES
  (.107@40k -> .069@300k, real .059) and TVR holds below floor (.089-.105),
  but centroid speed bleeds back (.283@100k -> .245@300k, mfrac .424->.351)
  and height variance worsens (.026->.022). Per-prompt motion pearson
  collapses (.27@20k -> ~.06 beyond 100k) while prompt/noise sensitivity
  rises to .86: over-training re-freezes the model into smooth, generic,
  prompt-agnostic motion. val_first upticked from ~212k, consistent.
- Answer to "does longer training teach prompt following": no -- it degrades
  past ~40k. Practical operating range for the recipe: 40k-100k steps.
- Series `v8_300k_run` in `paper/results/m6_h8_milestones_n64.json`; dotted
  extension in fig m6_fix_comparison; paper paragraph updated.

## v9.1 rig-strengthening pilot -- rejected (2026-08-23)

- Single-variable pair vs the v9.0 pilot: rig flow weight 1->4 plus a
  bone-length preservation loss (weight 5) on the recovered clean rig
  (`configs/m6_protocol_v9p1_rigweight_bones_10k.json`).
- Result: bone-length error tightens only modestly (10-19% -> 10-12%) while
  pixel quality regresses sharply at the same budget (TVR .538 -> .708,
  angle jerk .230 -> .341) and prompt retrieval stays at chance (top-1 3.1%).
- Reading: rig-side loss strength is not the alignment lever and taxes the
  shared trunk. v9.0 (weight 1, no bone loss) remains the standing recipe.
  Together with the t5-base rejection, the alignment floor points at
  caption/conditioning data, not loss shaping or encoder scale.
- Artifacts: gin `results/m6v9p1_rigweight_bones_10k_s0/`.

## Rig-space prompt alignment metric -- validated (2026-08-23)

- New metric (`scripts/rig_alignment_metric.py`): per-joint signature (speed
  profile, amplitude profile, end-effector periodicity) from 2D rig
  trajectories, z-scored over the prompt reference set; retrieval asks
  whether a clip's motion is nearest its own prompt's reference.
- Validation: held-out REAL clips retrieve their own prompt at top-1 59% /
  top-5 89% over 37 test prompts (chance 2.7%/13.5%) -- the signature is
  discriminative on real motion. A naive uncentered cosine was NOT (matched
  .871 vs shuffled .866, dominated by the generic hands-move-more profile);
  caught by a shuffled-prompt sanity check and replaced.
- First reading on v9 10k (self-emitted rig, n=64): top-1 4.7% / top-5 14.1%
  -- at chance. The model's motion carries almost no prompt-specific
  signature, consistent with the pixel-side pearson ~.20. Semantic following
  now has a measurable floor (chance) and ceiling (real).
- v9 n=64 pixel verdict vs matched control: statistically indistinguishable
  (TVR .538 vs .527, speed .385 vs .368) -- the rig output remains free.

## v9 rig co-generation pilot -- proof of concept (2026-08-23)

- 10k run of the v8 recipe plus co-denoised rig tokens
  (`configs/m6_protocol_v9_rig_cogen_10k_pilot.json`; module
  `train/latent_video_dit_ar_rig.py`, existing code paths untouched).
- Pixel cost: none measurable -- final pixel flow loss .410 vs .403 for the
  matched rig-free control (t5-small text-pilot arm, same fresh 10k cosine).
- Rig quality (self-emitted, overlay probe on 4 prompts): rig flow loss .020;
  on-figure rate .54-.79; bone-length error 10-19% of mean bone; temporal
  jitter 12-16%. The rig tracks the figure but is loose.
- Next levers: bone-length preservation loss, higher rig loss weight, longer
  training; eval_m6 v9 support for the full pixel metric set; rig-space
  prompt alignment metric (the point of the track).
- Artifacts: gin `results/m6v9_rig_cogen_10k_s0/` (+ `rig_probe/` overlays
  and `rig_report.json`).

## Text-encoder pilot pair -- t5-base rejected (2026-08-23)

- Matched fresh 10k runs of the v8 recipe differing only in the frozen text
  encoder (declared `configs/m6_protocol_v8_text_pilot_t5{small,base}_10k.json`).
- Result: t5-base does not improve prompt-motion alignment (speed pearson
  .196 vs .263 for t5-small; motion-fraction pearson ~0 for both; prompt/noise
  ratio .706 vs .655). Encoder scale is not the conditioning bottleneck at
  this data/model scale; remaining suspects are the templated single-sentence
  captions (one fixed string per action) and cross-attention capacity.
- Artifacts: gin `results/m6v8_text_t5{small,base}_10k_s0/`.

## v8 weakness report -- probes concluded (2026-08-23)

Taxonomy of the v8 100k model's remaining weaknesses, each probed and
attributed:

1. Vertical-motion deficit (data/objective): undershoot concentrates on
   compound vertical motions (squats -.47, burpee -.41, lunges -.37) --
   height variance .028 vs real .061. Not budget-limited (flat 60k-100k).
   Lever: future-keyframe goal conditioning (ARDY Sec 3.3) or vertical-motion
   data upweighting.
2. Hold-still failure (conditioning): overshoot concentrates on static-pose
   prompts (balance +.23, yoga ~+.16) -- regression toward mean motion.
   Lever: caption diversity; conditioning pathway (t5-base already rejected).
3. Weak semantic following (conditioning/data): prompt-motion pearson rises
   to ~.30 by 40k then saturates while prompt/noise sensitivity keeps rising
   (.46->.79) -- the model listens more but does not understand better.
   CFG amplifies (pearson .24->.30 at CFG 4, no quality cost). Quantity
   metrics cannot judge intent (e.g. wave vs raise-and-lower); action
   classifier + pose-regressor retrieval metrics designed, not yet built.
4. Jerk (sampler + training, split measured): 20-step sampling cuts angle
   jerk .095 -> .079 (real .059) at mild motion damping (speed .286 -> .256)
   -- ~40% of the excess is sampler, the rest trained-in.
5. Mild in-horizon decay: motion-fraction drift slope -.067 per block within
   the 5 s rollout. 10 s (2x horizon) probe recorded for visual reference.
6. OOD prompts (cartwheel, ballet, crawl, spin) probe saved at
   `infer_ood/` -- expected to fall back to in-distribution motions.

## v8 weakness probes (2026-08-23, superseded by report above)

- Per-prompt alignment on the v8 100k checkpoint splits systematically:
  undershoot concentrates on vertical/full-body compound motions (squats
  -.47, burpee -.41, lunges -.37 centroid-speed error) -- the height-variance
  deficit localized -- while overshoot concentrates on hold-still prompts
  (balance on one leg +.23, yoga poses +.16): regression toward mean motion.
- Within the 5 s rollout, motion decays mildly block-over-block
  (motion-fraction drift slope -.067).
- 10 s (2x training horizon) rollout probe saved to
  `results/m6v8_combined_h16_100k_s0/infer_10s/`.

## v8 100k CFG sweep -- sampler diagnostic (2026-08-23)

- Question (Jin): does the frozen VAE hurt prompt following? Sampler-only
  sweep on the v8 100k checkpoint, same n=64 evaluation at CFG 2/3/4.
- Result: prompt-motion alignment rises monotonically with CFG (speed pearson
  .242 -> .271 -> .302; motion-fraction pearson .101 -> .184; prompt/noise L1
  ratio .77 -> .86) at no quality cost (TVR improves .122 -> ~.105, motion and
  jerk essentially unchanged).
- Reading: the conditioning signal survives the latent space and is amplified
  by CFG -- evidence against the VAE hypothesis -- but even amplified it is
  weak (pearson .30), pointing at the conditioning pathway (60M t5-small over
  templated single-sentence captions) rather than the representation. The
  declared primary sampler stays CFG 2; this sweep is diagnostic.
- Artifacts: gin `results/m6v8_combined_h16_100k_s0/eval_n64_100000_cfg{3.0,4.0}/`.

## M6 h8 100k convergence run -- completed (2026-08-22)

- Independent fresh 100k run of the h8 start-aligned protocol
  (`configs/m6_protocol_v3_start_aligned_h8_100k.json`), evaluated at
  10k/20k/40k/60k/80k/100k under the convergence rule declared before results
  (`configs/m6_r0_convergence_evaluation_v1.json`), n=64 each.
- Result: frame structure converges through the real floors (TVR .434 -> .097,
  LIE .193 -> .075 versus real .133/.110) while motion stays collapsed the
  whole way: centroid speed .134 -> .115 (real .314), motion fraction
  .137 -> .111 (real .371), height variance ~.008 (real .061). Prompt/noise
  L1 ratio stays in .67-.82 with no trend.
- Reading: the freeze is structural, not under-training -- 100k steps buy
  ever-cleaner statues. Milestone series checked into
  `paper/results/m6_h8_milestones_n64.json` (`h8_100k_run`); figure and paper
  text updated (fig:m6 dotted squares).

## R0 full-clip 50k convergence run -- completed (2026-08-22)

- Independent fresh 50k full-clip run
  (`configs/r0_full_clip_latent_protocol_v2_50k.json`), milestones 10k-50k
  per the predeclared convergence rule, n=64 each.
- Result: motion stays calibrated (centroid speed .375 -> .293, real .314;
  motion fraction .543 -> .440, real .371) but structure plateaus: TVR
  .438 -> .223 with no improvement after 40k (real floor .133), LIE ~.14,
  angle jerk ~.12 (real .059).
- Reading: full-clip denoising avoids the static shortcut but does not reach
  the structural floor at this budget; the v8 fixed block-AR run dominates it
  on structure at comparable motion (.122 TVR / .286 speed at 100k vs .223 /
  .293 at 50k -- budgets differ, single seeds, diagnostic only).
- Series `r0_50k_run` in `paper/results/m6_h8_milestones_n64.json`; paper R0
  paragraph extended.

## M6 v8 combined main run -- completed (2026-08-22): freeze broken

- 100k-step run of the winner combination (16-frame blocks + motion-weighted
  + foreground-weighted flow loss), milestones evaluated after completion
  under the predeclared rule (n=64 at 2k/5k/10k/20k/40k/60k/80k/100k).
- Result: escapes the h8 structure-motion Pareto front. At 100k: TVR .122,
  LIE .073 (real floors .133/.110) with centroid speed .286 (91% of real
  .314), motion fraction .434 (real .371), against the h8 baseline's frozen
  .115/.111 at the same budget. Motion is stable from 20k onward while
  structure keeps converging.
- Residual gaps: height variance .028 (real .061), angle jerk .095 (real
  .059, ~1.6x), prompt/noise ratio .77 at 100k. Single seed; component
  attribution rests on the matched 2k pilots.
- Series in `paper/results/m6_h8_milestones_n64.json` (`v8_100k_run`);
  figure `paper/figs/m6_fix_comparison.*`; paper paragraph "Breaking the
  freeze". Superseded 40k declaration noted in the 100k protocol config.

## M6 v8 combined main run -- declared 2026-08-22, superseded 40k note

- `configs/m6_protocol_v8_combined_h16_40k.json` declared before results:
  h16 blocks + motion weighting + foreground weighting, 40k steps, milestone
  n=64 evaluations at 2k/5k/10k/20k/30k/40k (config'd before any result).
- Purpose: test whether the winner combination escapes the h8 structure-motion
  Pareto front (no h8 checkpoint approaches both floors). Single seed;
  component attribution rests on the matched pilots above.

## Next-block divergence: first common-judge scores (2026-08-24)

- New metric `eval/next_block_divergence.py` (declared in
  `paper/refs/v02_direction_next_token.md` idea #4 before implementation):
  teacher-force GT history from held-out test clips, generate the next
  16-frame block (best-of-4), score fg-union weighted RGBA MSE against the
  clip's ACTUAL continuation; free-running variant carries the model's own
  prefix (exposure gap); real floor = same-prompt different-clip pairs.
  32 clips, 6 block positions, 10 steps, CFG 2, seed 20260824.
- 100k-budget scores (TF best-of-4 avg / free-running avg; floor .5545):
  v8 100k .3139/.4584 | v9.0 conv 100k .2870/.4399 | v9.3 coupled 10k
  .3091/.4968.
- Matched 10k-budget scores: v8 .3007/.4653 | v9.0 varied-captions .2947/.4471
  | v9.0 rig-cogen (canon) .2966/.4504 | v9.3 anchor .3009/.4486 | v9.3
  coupled .3091/.4968.
- Readings: (1) v9.0 beats v8 on BOTH teacher-forced and free-running at both
  budgets -- first quantitative evidence that rig co-generation improves pixel
  prediction for free. (2) v9.3 coupled w2.0 is the WORST arm at matched 10k
  (pixel tax visible in the common judge), so the earlier "coupled 10k ~= v8
  100k" framing is retired; the weight sweep (declared
  configs/m6_protocol_v9p3_sweep_*.json with judgment rule) must show a
  weight that keeps the binding gain without this tax. (3) v8's TF divergence
  WORSENS 10k->100k (.3007->.3139) while v9.0's improves (.2947->.2870) --
  training v8 longer buys structure but not continuation fidelity.
- All models sit well below the real floor: they track the specific
  continuation better than a legitimately different real take (metric sane).
- Artifacts: `results/divergence/*.json` (local + gin).

## v9.3 coupling-weight sweep -- completed (2026-08-24): warmup wins

- Four arms, protocols + judgment rule declared before any result
  (`configs/m6_protocol_v9p3_sweep_*.json`): constant render+consistency
  weight 0.25/0.5/1.0, plus the full 2.0 weight linearly warmed up over the
  first 5k steps (new `--coupling-warmup-steps` trainer flag). 10k steps
  each, matched to the coupled pilot and v9.0 control.
- Gate: on-figure >= .85 AND TVR <= v9.0-10k .538; winner = lowest
  teacher-forced divergence among passers.
- Results (on-figure / TVR / TF div / FR div):
  w0.25 .724 / .518 / .2959 / .4510 -- binding lost, pixels fine
  w0.5  .799 / .563 / .2953 / .4519 -- both gates missed
  w1.0  .876 / .597 / .2994 / .4625 -- binding passes, pixel tax clear
  w2.0-warmup .912 / .493 / .2977 / .4522 -- BOTH GATES PASS
  (constant w2.0 pilot: .93 / .721 / .3091 / .4968)
- Reading: constant coupling weight is a strictly monotone binding-vs-pixels
  tradeoff with NO feasible point; the 5k warmup breaks the tradeoff --
  full-strength coupling applied after pixels settle keeps ~all of the
  binding gain (.912 vs .93) while IMPROVING structure over the uncoupled
  control (TVR .493 vs .538). Mechanistically consistent with the coupled
  arm's failure mode: early in training the decoded pixel x0 is noise, so
  the consistency loss drags the pixel path toward the (also-noisy) rendered
  rig; ramping the weight lets attention-level co-generation establish the
  representation first.
- Verdict per declared rule: winner = w2.0 + 5k warmup. Flagship recipe =
  v9.3-warmup at 100k x 2 seeds.
- Artifacts: gin `results/m6v9p3_sweep_*`, divergence jsons mirrored to
  `results/divergence/`.

## SRE v1 trained + validation gates run (2026-08-24, autonomous session)

- Implemented per the pre-declared design (`paper/refs/sre_design.md`, commit
  79853a8): `train/sre.py` (3.30M-param conv regressor, single premultiplied
  RGBA frame -> 27x2 sigmoid joints, L2 masked to joints inside [0,1]) and
  `eval/sre_validate.py` (three gates; numeric thresholds written into the
  script before training). Unit tests `tests/test_sre.py`.
- Training: gin, cache/mini, 20k steps batch 256 (~3 min as a nice-15 second
  job beside the flagship; 360k train / 18k val frames).
- Gate 1 (held-out real renders) PASS: val 0.711 px mean, PCK@2 .914,
  PCK@4 .975; test 0.657 px, PCK@2 .934, PCK@4 .976. Well under the 1.6 px
  capsule radius target.
- Gate 3 (off-screen stability) PASS but VACUOUS at the declared tier: no
  frame in cache/mini has any joint outside [-0.5,1.5] (actual rig range
  [-0.494, 1.157]), so the design's sit-up premise does not occur in mini.
  Informative tier instead: frames with edge-clipped joints (outside [0,1])
  degrade visible-joint error to 0.96 px on test (2076 frames, 1.48x
  in-frame) and 2.48 px on val (98 frames, 3.5x — small sample, flagged).
- Gate 2 (corruption localization, 128 val/test frames, seed 20260824):
  extra_arm PASS (affected 2.77 px = 4.2x baseline, unaffected 0.89 px =
  1.35x). swap_LR_partial FAIL under the v1 rule (affected {fore-arms only}
  5.09 px = 7.7x; "unaffected" 1.64 px = 2.5x > 2x limit). Per-joint
  diagnosis: the entire bleed is the arm chains (hands +2.8, upper arms
  +2.65, hand-ends +2.6, thumbs +2.1; non-arm joints unmoved) — the
  mechanistically expected response of a colour-identity regressor to an
  arm-colour swap, i.e. a too-narrow affected set in MY gate implementation,
  not skeleton hallucination. v1 verdict recorded as-run, not relitigated.
- PROPOSED for sign-off (Jin): gate2 v2 with chain-complete affected sets
  (swap_LR_partial -> full left+right arm chains incl. shoulders/thumbs);
  rerun only after the rule is blessed.
- Notable extra: swap_LR_full — invisible to the oracle by design (adjacency
  preserved) — moves SRE affected joints to 8.38 px (12.8x baseline): the
  learned instrument catches a corruption class the rule-based oracle cannot.
- Artifacts: gin `results/sre_v1/` (ckpt_final.pt, validation_{val,test}.json);
  mirrored `paper/results/sre_v1_validation_{val,test,corruptions}.json`.

## SRE v1 -- trained and G1-passed (2026-08-24)

- Single-frame pixels -> cskel27 regressor per the declared design
  (`paper/refs/sre_design.md`): 2.1M-param conv encoder + MLP, masked L2
  excluding off-screen joints (gate G3 by construction), train split only.
- 20k steps, batch 256, ~3 minutes on gin ALONGSIDE the running flagship
  (no visible slowdown; flagship held 0.17s/it).
- Held-out test split (4096 frames): mean joint error 0.796 px at 64^2
  (gate G1: < 1.6 px -- PASS), PCK@2px .915, PCK@4px .970.
- Gate G2 (corruption sanity: swapped-limb / extra-arm renders must raise
  error) still owed before SRE-based claims enter the paper.
- Artifacts: gin `results/sre_v1/` (checkpoints + history.json + log).
