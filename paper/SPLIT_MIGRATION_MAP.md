# Testbed / mechanism paper split — content migration map (2026-08-24)

Approved by Jin (relayed by Pixel); executed by Claudia. Recovery point:
git tag `paper-v01-mixed` + verbatim copy `paper/paper_mixed_v01.tex`.

## Documents

- **Testbed report**: `paper/paper.tex` (6 pp after cut; was 8 pp mixed).
- **Mechanism paper skeleton**: `paper/mechanism/static_copy_shortcut.tex`
  ("The Static-Copy Shortcut: Diagnosing and Repairing Motion Collapse in
  Block-Autoregressive Video Diffusion"), compiles to 4 pp with both M6
  figures; TODO blocks typeset loudly so a draft can't pass as finished.

## Moved (mixed v0.1 → mechanism paper, verbatim transplant)

| Content | From (mixed v0.1) | To (mechanism) |
|---|---|---|
| Frozen codec description + IoU/PSNR + decode audit + selection gate | "Latent long-horizon track" ¶ | §Setup |
| M6 architecture description | first half of M6 ¶ | §Setup |
| Horizon dial (4/8/40-frame), decoded-RGBA aux rejection | M6 ¶ | §Phenomenon |
| 20k milestone curve + 100k convergence run + fig:m6 | M6 ¶ + figure | §Phenomenon |
| Five matched pilots + tab:pilots + gradient-allocation reading | "Breaking the freeze" | §Competing hypotheses |
| Combined fix 100k + seed-1 replication + 300k over-training bound + fig:m6fix | combined-fix ¶ + figure | §Combined repair |
| R0 10k control + 50k full-clip run + attribution | "Full-clip latent control R0" ¶ | §Attribution |
| Per-prompt failure taxonomy + CFG/steps diagnostics + t5-base null | "What remains broken" ¶ | §Residual failure taxonomy |

Figures `paper/figs/m6_milestone_tradeoff.pdf` and
`paper/figs/m6_fix_comparison.pdf` are now referenced ONLY by the mechanism
paper (removed from the testbed arXiv bundle).

## Rewritten in the testbed report

- "Latent long-horizon track" ¶ → release/interface-only, explicit
  no-performance-claim sentence + companion-study pointer.
- tab:status Baselines row → "block-AR/codec evaluation (companion study)"
  moved to the not-claimed column.
- Limitations item 4 (long-horizon track) → released + companion pointer.
- Abstract, intro contributions, conclusion: checked — contained no M6
  numbers; unchanged.

## Kept in the testbed report (quantitative evidence, final list)

1. Controlled corruption stress test (freeze/shuffle/reverse/loop × metric
   complementarity) — fig:failures.
2. Image baselines (M0–M3 ladder, tab:ladder).
3. Matched K1 random-vs-warm-start 50 f pair — fig:k1.
4. Short-video track: release-only, withdrawn numbers stay withdrawn.
5. AR track + frozen codec: release/interface-only, zero numbers.

## Still owed to the mechanism paper (explicit TODO blocks in the tex)

- §Generality: 32² pixel-space no-VAE block-AR pilot (freeze reproduction +
  repair transfer); protocol to declare before the run.
- §AR-native evaluation: next-block divergence table (first results already
  in `results/divergence/`, see EXPERIMENT_LOG 2026-08-24 entry).
- Intro, related work, limitations, conclusion prose.
- `\cite{cho2026dsf}` arXiv id after the testbed report is submitted.

## Unresolved decisions (for Jin/Pixel review)

1. Testbed title still says "for Diagnosing Long-Horizon Video Generation" —
   defensible via the corruption suite + protocol, but "Diagnosing" now leans
   less on M6; consider "for Small-Scale Video Generation Research".
2. The "Why full-clip bidirectionality stops being enough" essay ¶ was KEPT
   in the testbed report (conceptual, zero numbers, motivates the released AR
   track) — trim or move if the report should read leaner.
3. Codec reconstruction numbers (IoU .952 / 34 dB) now appear only in the
   mechanism paper; if the testbed release page needs them, they'd live in the
   HF model card, not the report.
4. Page 6 of the testbed PDF is mostly whitespace after references — fine for
   arXiv, could be balanced later.
5. arXiv bundle regenerated as DRAFT (`output/arxiv_bundle_testbed_draft.tar.gz`,
   compile-verified standalone) — NOT submitted, per instruction.

## Pixel review round 1 (2026-08-24) — resolved

- **P0** mechanism Table 1 overflow into the right column: fixed (narrow
  treatment column, tabcolsep 3pt, verdict abbreviated; overshoot note moved
  to the caption). All 4 pages re-QA'd visually; 0 overfull warnings.
- **P1** title overclaim: renamed to "Dancing Stick Figures: A Traced
  Synthetic Testbed for Video Generation Diagnostics" (title, pdftitle,
  pdfsubject, mechanism \cite{cho2026dsf}); abstract "long-horizon sampling"
  -> "video sampling". Resolves unresolved decision #1.
- **P1** boundary: "Why full-clip bidirectionality stops being enough" essay
  moved to the mechanism paper's Introduction (K1 now introduced via
  \cite{cho2026dsf}); testbed keeps a 4-sentence "Beyond fixed-length
  clips" interface rationale. Resolves unresolved decision #2.
- Testbed still 6 pp; mechanism still 4 pp; both compile clean (0 overfull).
  arXiv draft bundle regenerated (still NOT submitted).
