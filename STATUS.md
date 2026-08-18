# STATUS — night of 2026-08-17 → 18 (written 22:40 PDT by Claudia)

## What is running
| where | run | what | steps | ETA |
|---|---|---|---|---|
| gin | ia64 | UNet 64² **image** (T=1), b128 | 30k | ~00:20 |
| gin | ib64 | DiT-FM 64² image, p4, b128 | 30k | ~00:20 |
| gin (queued) | ia64L | UNet 64² image, **min-SNR-γ=5**, LR→2 % | 100k | ~09:00–10:00 |
| gin (queued) | ib64L | DiT-FM 64² image, **patch 2**, LR→2 % | 50k | ~09:00–10:00 |
| gin (queued after) | a64 → b64 | video 64² 8f: resume UNet from 24k, then DiT fresh (sampler OOM fixed) | 85k | afternoon |
| RunPod A100-80GB `bs03yg22p0vrnf` ($1.39/h) | ia128 + ib128 | same image recipe at **128²** (UNet min-SNR / DiT p4), b128 | 40k each | ~09:00; auto-collect + terminate (`scripts/pod_collect_img128.sh`, results → `pod_results/img128/`) |

Watchdogs restart crashed runs ≤3×: gin `~/dev/stickdance/watchdog_img.sh` (log `/tmp/watchdog.log`), pod `/root/stickdance/pod_watchdog_img128.sh` (`/root/watchdog.log`).
Samples: `runs/<r>/sample_XXXXXX.png` (64 fixed-noise EMA samples; `sample_raw_` = non-EMA, ≤10k). TB: `gin:6006`.

## Early evidence (see out/img64_early.png)
UNet image at 2k steps (10 min) = clean colour-coded stick figures; DiT at 4k = formed but arms fragmented (patch-4 blockiness → ib64L uses patch 2).
Video UNet needed 22k steps for a similar level → image-first is the right stage 1.

## Fixed tonight
* `euler_sample` had no `@torch.no_grad()` → 77 GB at sampling. This killed b64 (twice) and every RunPod ablation on 4090. Fixed.
* UNet/DiT T=1 fast paths (skip temporal attn, centre-slice 3×3×3 convs) — exact, 5× faster; also avoids CUDA grid-limit crash at patch 2.
* `--init` (warm-start video model from image ckpt; UNet exactly = image model per frame), `--min_snr`, `--lr_final`.

## Answers to Jin's questions (asked 22:00)
**"Do we simply provide the category and ask it to generate?"** — Currently no: all runs are **unconditional** (`--cond none`): pure noise → stick figure, no label. `--cond group` exists (6 prompt groups + CFG, null class) but isn't used tonight; text (CLIP/T5) conditioning is not implemented yet. For the *image* milestone unconditional is the honest baseline; class/text conditioning is the next step (and needed for the "text-to-motion" story of the paper).

**"How are we fighting disconnected / extra limbs?"** — Today: *measured, not yet directly fought.* The oracle (tvr = topology violation rate, lie = limb-existence error, cpe) counts them; what reduces them in practice is (1) more steps + EMA + LR annealed low (the long runs), (2) finer tokens for the DiT (patch 2), (3) min-SNR weighting (better use of mid-noise steps where structure is decided), (4) later: CFG with group/text conditioning usually sharpens structure. The *direct* lever is a structural auxiliary loss / pose-regressor SRE (backlog E3/E15) — that's the research item for the paper, and the oracle gives us the metric to prove it works. Anything else (rejection-sampling with the oracle) hides the problem rather than fixing the model.

**Hashnode** — noted, you said nvm; the draft can come from this file when you want it.

## Morning 2026-08-18 07:30 — results
All overnight runs completed without a crash (gin: ia64/ib64 30k, ia64L 100k, ib64L 50k; RunPod A100: ia128 20k, ib128 40k, pod
auto-terminated 03:46, ~5.5 h ≈ $7.6). Grids: `out/img_final_sheet.png`, per-run `out/img/<run>/`. Scores `out/scores/`.

Oracle on 512 EMA samples (50 sampler steps) — generated vs **real val frames at the same resolution** (the "floor"):

| run | arch | res | steps | tvr | lie | cpe | clean-skeleton frac |
|---|---|---|---|---|---|---|---|
| ia64 | UNet | 64 | 30k | .159 (.142) | .113 (.106) | .041 (.039) | .42 (.40) |
| ib64 | DiT p4 | 64 | 30k | .176 (.143) | .122 (.108) | .040 (.040) | .38 (.37) |
| **ia64L** | UNet min-SNR | 64 | 96k | **.134 (.136)** | .116 (.103) | .039 (.040) | **.43 (.40)** |
| ib64L | DiT p2 | 64 | 50k | .164 (.139) | .114 (.106) | .043 (.039) | .40 (.37) |
| ia128 | UNet min-SNR | 128 | 20k | .226 (.203) | .073 (.047) | .020 (.019) | .22 (.23) |
| ib128 | DiT p4 | 128 | 40k | .251 (.209) | .065 (.048) | .020 (.020) | .23 (.21) |

Reading: every model is within a few points of the real-data floor on all three regressor-free metrics; the long UNet is *at* the
floor. The floor itself is nonzero because of self-occlusion (and rises at 128² where the oracle resolves more components).
So the "we have a good stick-figure image diffusion model" claim holds for both architectures at both resolutions.
Caveat (unchanged): the oracle can't see proportion/geometry errors — the pose-regressor SRE (E3) is what would.
Visually: UNet strokes are cleaner; DiT patch-2 fixed the arm fragmentation of patch-4; 128² samples are crisp.

Note: ib64 was trained before the T=1 temporal-attn skip; its ckpt args carry `t1_skip=False` and loaders honour it.

## Now running (video phase, gin)
* a64 — video UNet 64² 8f from scratch, resumed 24k → 85k.
* **a64i** — same model, **--init from ia64L** (image model), 0 → 61k = same video-step budget. This is the Seedance-stage-2 controlled test.
* then b64 (video DiT, sampler OOM fixed).

## Morning checklist (remaining)
1. `out/`: build 30k grids ia64 vs ib64 + evolution strips; oracle image metrics on 512 samples each (script TODO: `eval/score_images.py`).
2. Look at ia64L/ib64L mid-run grids; ia128/ib128 from `pod_results/img128/`.
3. Decide: which image model seeds the video model → `--init` experiment vs from-scratch (a64).
