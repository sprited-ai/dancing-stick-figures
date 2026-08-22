# STATUS — 2026-08-22 07:00 PDT (Claudia) — latent long-horizon evidence complete, paper updated

## 2026-08-22 overnight (Claudia, taking over the ChatGPT/Codex paper session on Jin's ask)
- ChatGPT session (08-20→22, died 05:31 08-22) built: paper.tex repositioning, K1 canonical protocol + warm-start
  result, video-VAE codec ablation (**f8t4d16 40k frozen, sha256-pinned**; alpha IoU .952, PSNR ~34 dB), M6 block-AR
  latent track (start-aligned fix, h4/h8/h40 horizon dial, decoded-RGBA aux NO-GO at n=64), H8 fresh 20k run.
  Snapshot committed as-is on branch `paper-m6` (7866ad9). Workspace on gin: `/data/dancing-stick-figure-paper`.
- Milestone evals (declared before results; 5k/10k/15k/20k, n=64): structure → floor monotonically
  (TVR .745→.349, floor .133) while **motion collapses by 10k and flatlines** (speed .300→.142 vs real .314;
  height var → 1/6 real). `paper/results/m6_h8_milestones_n64.json`, figure `paper/figs/m6_milestone_tradeoff.*`.
- **R0 full-clip control trained (10k, gin, ~50 min) + evaluated: no freeze** (speed .382, fraction .540, mild
  overshoot; TVR .456; jerk .220). → M6 freeze is the teacher-forced short-horizon block factorization, not
  capacity and not the latent flow loss. Full log entries in `paper/EXPERIMENT_LOG.md`.
- Paper updated: latent-track paragraphs (codec / M6 trade-off / R0 control), fig:m6, status table, future item 4
  → "started". 7 pages now (refs spill to p7). PDF page-QA'd: `output/pdf/paper.pdf`.
- Open: 2 EXITED RunPod pods (`dsf-ar-pilot-*-seed1`) hold volumes — terminate after confirming collection (Jin).
  T4 clean notebook rerun still owed. VAE 96-frame-window idea from the side session was superseded by the main
  session's "codec validated, don't retrain" decision.

# (2026-08-18 state below)

## Public
- Dataset https://huggingface.co/datasets/sprited/dancing-stick-figures (public; configs frames 4.6 GB / mini 0.85 GB / motion 0.35 GB; viewer live)
- Baselines https://huggingface.co/sprited/dancing-stick-figures-baselines — image 64²/128² (UNet, DiT, cond), video 8f (scratch 85k, image-init 61k),
  **autoregressive `unet_ar64.pt`** (5 s rollouts), DiT video interim; README has the oracle/FVD tables.
- Code https://github.com/sprited-ai/dancing-stick-figures — route-first README, `scripts/rollout.py`, `scripts/compare.py`, notebook with outputs (answer sheet).
- Colab notebook: verified on a real T4 2026-08-18 evening (data/cache/image/AR-video all within stated times); 4 T4-only bugs fixed
  (batch 4×2 for the AR cell, grep error surfacing, line-buffered grep, fp16 auto on non-bf16 GPUs). **One more clean end-to-end T4 run
  with the fixed code is still owed** (last two cells were slow under the old bf16 path).
- Blog draft parked in blog/ (Jin: skip for now). Paper: full v0.1 tech report now exists as Markdown
  (`paper/REPORT.md`), editable LaTeX (`paper/paper.tex`), and a visually verified 6-page PDF
  (`output/pdf/dancing-stick-figures-tech-report.pdf`). Public submission has not started.

## Results (v0.1)
- Image 64²: UNet 100k at oracle floor (tvr .134/.136 real, lie .116/.103); DiT p2 slightly worse; 128² both ~floor+0.02.
- Video 8f: scratch 85k vs image-init 61k — image-init reaches same loss ~2.5× sooner, ends equal (FVD 199 vs 213; real-real ~115).
- AR 5 s: per-frame anatomy within ~0.02 of real; temporal jitter/jerk ~1.2–1.4× real (seams). paper/results/*.json.

## Spend
RunPod today ≈ $75 (A100 ×2 sessions, 4090 ×2, H100 5.5 h). Jin topped up; no pods running now.

## Data (final for v0.1)
gin `~/dev/stickdance/data/v1_final` (parquet, meta version 0.1.0), `data/v1_final/motion`, cache `data/v1_final_cache` (frames.npy + clips.json, 4,290 clip-windows).
ARDY npz: `~/dev/ardy/outputs/v1` 1,430/1,430. bone_scale is now applied to joints (was only recorded before).
Older `data/v1*` / `v1_cache` (1,320 clips) still used by running trainers — do not delete until they finish.

## Running (all unattended; watchdogs restart ≤3×; collectors pull + auto-terminate pods)
| where | runs | purpose | results land in |
|---|---|---|---|
| gin (`watchdog_img.sh`, /tmp/watchdog.log) | a64 (video UNet 64² scratch, →85k), a64i (same, --init from ia64L, →61k) | image-init vs scratch (UNet) | gin runs/a64, runs/a64i |
| RunPod A100 c7b7aplyx0v6ey $1.39/h | b64i (DiT stage-2 from ib64L), b64 (scratch), patch 2, shift 2, img .1, i2v .2, →61k | Seedance §4.1 recipe, init vs scratch (DiT) | `pod_results/dit2/` via `scripts/pod_collect_dit2.sh` (mac, 20 h cap) |
| RunPod 4090 k203e8o9tfm37c $0.74/h | ic64 (UNet), id64 (DiT p2) class-conditional 64² image, →30k | "give the category, generate" | `pod_results/imgcond/` via `scripts/pod_collect_imgcond.sh` (8 h cap) |
Check pods: `python3 scripts/runpod.py list`; terminate manually if a collector dies: `python3 scripts/runpod.py terminate <id>`.
Finished image models on gin: runs/ia64, ib64 (30k), ia64L (100k, min-SNR), ib64L (50k, p2); 128²: pod_results/img128 (ia128 20k, ib128 40k). Scores: paper/results/score_*.json.

## v0.1 → v0.2 backlog (in order)
1. Replace card video GIF with a64 final ckpt; add video table (a64 vs a64i vs b64 vs b64i: loss/val/oracle temporal/FVD) — `eval/run_ckpt.py`.
2. Templated dense captions from labels (`caption`, `caption_static`; Seedance dynamic/static) → new `frames` columns; more prompts (143 → 500+, ARDY overnight).
3. Learned pose regressor (SRE) → geometry-aware oracle; anomaly config (rendered malformations, multi-label). Then adversarial aux-loss ablation (Jin's GAN point) judged by independent metrics.
4. Review the 6-page paper draft, settle the author line, record one clean post-fix T4 notebook run, then make the
   arXiv decision. SPARK.md cleanup remains separate.
5. ARDY web demo at ardy.sprited.ai (Jin delegating to another agent; see notes in chat 2026-08-18).
6. Teaching pack (Jin 08-18): repo README quickstart verified on a fresh pod; `mini` config (64², ~1 GB); **Colab notebook in ELI11 tone**
   (elementary/middle-school friendly): ① visualise data ② train image model ③ train video model warm-started from a released
   checkpoint ④ oracle scoring; baseline ckpts on HF `sprited/dancing-stick-figures-baselines`.
7. **Diversity wishlist (v0.2+, Jin 08-18 — grow if the data feels samey):** more prompts (143 → 500+, incl. two-part
   "A then B" motions); wider body variation (bone_scale beyond ±8 %, head size, limb thickness, child/adult proportions);
   more cameras per clip and a perspective-camera option (currently orthographic only), camera roll, moving camera;
   longer clips / variable length, 30 fps variant; multiple figures per frame (interaction, occlusion between people);
   colour-palette / background variants as separate configs (keep the canonical palette for the oracle); optional props
   (chair, ball) as distractors; more seeds only where prompt diversity is exhausted (seed collapse check first).

## Gotchas
`pgrep -f`/`pkill -f` over ssh matches your own shell — use `[p]attern`. Watchdog `running()` must match `--out runs/x` at end of cmdline. `@torch.no_grad()` must sit directly above `euler_sample`. Don't stack >2 GPU jobs on gin. `~/dev/monet-machi/PAUSE` exists at Jin's request — ask before removing.
