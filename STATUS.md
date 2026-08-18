# STATUS — 2026-08-18 12:00 PDT (Claudia) — v0.1 pushed (private)

## Released (private, awaiting Jin's public flip)
https://huggingface.co/datasets/sprited/dancing-stick-figures — 275 files / 5.2 GB: `frames/` (514,800 rows) + `motion/` (1,430 clips)
+ card `hf/README.md` (same as repo). Verified `load_dataset` roundtrip from gin (user `sprited`). Staging: gin `~/dev/stickdance/hf_stage`.
Re-upload after edits: `hf upload sprited/dancing-stick-figures hf_stage . --repo-type dataset` (or single file `hf_stage/README.md README.md`).

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
4. REPORT.md v0 fill (§3, §4.2 table, §5.3), SPARK.md cleanup, arXiv decision.
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
