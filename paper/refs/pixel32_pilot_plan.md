# 32² pixel-space block-AR pilot — PROPOSAL for sign-off (2026-08-24, Claudia)

Purpose (paper2 "Generality" subsection): defuse the "static-copy is an artifact
of the frozen latent space" objection. Prediction to declare before the run: the
freeze reproduces in pixel space, and the signal-reallocation repair transfers.

## Why this needs Jin's sign-off (not run autonomously)

Two design choices materially shape what the section can claim, and the natural
trainer (`train/video_dit_ar.py`, M4) does not currently support the M6-matched
configuration:

1. **Init policy.** M4 requires `--init` (an M3 image warm-start). M6 trains
   from scratch, and the testbed paper documents warm-starting as a confound
   (better connectivity, worse motion calibration). A warm-started pilot could
   freeze "because of the init" — the objection we are trying to kill would just
   move. Proposal: allow `--init none` (fresh init) in M4; run from scratch.
2. **Repair-arm flags.** M4 has `--fg-weight` but no motion-weighted flow loss.
   Proposal: port `--motion-weight-alpha` from the latent trainer (identical
   formula, pixel-space application) so the repair arm is the same treatment.

Both are small trainer diffs but they touch a training path used by released
M4/M5 results — hence flagged rather than done.

## Proposed protocol (to be frozen as configs/pixel32_blockar_pilot_{base,fix}.json)

- Data: cache/mini at 32² (area-downsampled in the loader), full prompts, t5-small.
- Model: M4 DiT, patch 2, dim 256, depth 10, heads 4 (~12M — small on purpose),
  from scratch, seed 0.
- Factorization matched to the M6 h8 baseline: target 8 frames, history max 16
  frames, variable history, start-aligned equivalent.
- Budget: 2k steps, batch 32 — the same budget class as the five fix pilots.
- Arms: (base) uniform loss; (fix) motion-weight α=1 + fg-weight 4 + 16-frame
  target blocks. Two runs total, ~30-45 min each on gin as nice-15 second jobs.
- Declared predictions: base shows centroid speed < .20 vs real .314 with TVR
  improving (freeze signature); fix restores speed ≥ .27 at comparable TVR.
  Verdict rule: eval_m6-style n=64 milestone eval at 2k, seed 20260824.
- Failure reading: if base does NOT freeze in pixel space, the frozen codec is
  implicated after all — that outcome would REWRITE the generality subsection,
  which is exactly why the design is pre-declared.

## Cost

Two ~40-min GPU jobs + ~30 lines of trainer diff + one eval pass. No contention
risk: run after flagship s0 finishes or nice-15 beside s1.
