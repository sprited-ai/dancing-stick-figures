# Dancing Stick Figures — paper TODO

The primary task is: **text prompt → 64×64 RGBA video, 50 frames at 10 fps**.
The 16×16 and 32×32 runs are cheap pipeline/teaching checks, not quality settings. The measured 32×32 anatomy loss makes
64×64 the smallest canonical setting that preserves the dataset's structural signal.
Claims stay `TODO` until the listed evidence exists.

All new runs must follow `paper/TRAINING_PROTOCOL.md`: periodic validation,
fixed-manifest inference panels, immutable milestone checkpoints, resumable
latest state, and off-pod artifact verification are mandatory.

The architecture sequence and per-model gates live in `paper/RESEARCH_ROADMAP.md`.

## P0 — start the first real T2V baseline

- [x] Choose K1: bidirectional, factorised Video DiT + flow matching.
- [x] Add token-level frozen T5-small conditioning and classifier-free dropout/guidance.
- [x] Pass CPU unit tests and a 16×16, 50-frame end-to-end GPU pipeline test.
- [x] Pass short 16×16 and 64×64, 50-frame GPU smoke runs; preserve them as engineering evidence.
- [x] Pass the 32×32 batch/throughput teaching run.
- [x] Pass the canonical 64×64, 50-frame batch-16 capacity run on RTX 4090
      (14.5 GB observed); an A100 is not required for K1.
- [x] Start a separate RunPod 4090 canonical scratch K1 pilot; preserve the
      existing U-Net runs unchanged.
- [ ] Record pod, GPU, command, git diff/hash, seed, parameter count, and hourly cost.
- [ ] Inspect samples at 100/500/1,000 steps before committing to the long run.
- [x] Preserve the scratch 3k K1 pilot as non-canonical engineering evidence:
      its saved `stride=1` is 20 fps, while the fixed task requires `stride=2`.
- [ ] Run every canonical 50-frame baseline with `stride=2` and reject a run
      at preflight if its saved arguments disagree.

## P1 — make the number publishable

- [ ] Freeze the prompt-disjoint train/validation/test manifest and publish its hash.
- [ ] Run K1 with at least 3 seeds under one fixed budget.
- [ ] Save fixed-seed/fixed-prompt samples and exact checkpoints.
- [ ] Report mean ± confidence interval, parameters, peak VRAM, steps, and GPU-hours.
- [ ] Measure prompt adherence, anatomy/structure, temporal motion, drift, and FVD.
- [ ] Validate the text-adherence evaluator on controlled prompt swaps.

## P2 — comparison, not model collecting

- [ ] Implement K2: chunk-causal text-conditioned Video DiT under the same budget.
- [ ] Compare K1 and K2 on the identical 50-frame protocol.
- [ ] Separate chunk-boundary errors from within-chunk errors for K2.
- [ ] Add one simple non-generative/replay control so metric floors are visible.
- [ ] Only define O1 after a measured K1/K2 failure yields a concrete hypothesis.

## Paper and educational release

- [ ] Rewrite Abstract and Introduction around the T2V testbed; do not claim unfinished results.
- [ ] Add modern primary references: Wan, MAGI-1, Seedance 1.0, Diffusion Forcing, Self Forcing, VBench-2.0, T2V-CompBench.
- [ ] Replace the broken pipeline figure and use “testbed” rather than mature “benchmark”.
- [ ] Add a compact architecture/results table instead of many sticker-like figures.
- [ ] Create a Colab path that trains a reduced lesson model but evaluates at the same concepts.
- [ ] Publish configs, evaluator version, checkpoints, samples, and failure reports.
- [ ] Ask a professor for a focused technical review once K1/K2 evidence is complete.

## Explicitly out of scope for the first submission

- 128×128 training.
- Claiming Seedance-level quality.
- Hidden leaderboard infrastructure.
- Unmeasured O1/O2 architecture inventions.
- Calling image pretraining itself the paper thesis.
