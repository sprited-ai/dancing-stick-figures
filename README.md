# Dancing Stick Figures — train your first video generation model, end to end

Code behind the dataset **[sprited/dancing-stick-figures](https://huggingface.co/datasets/sprited/dancing-stick-figures)**:
1,430 six-second clips of colour-coded stick figures dancing, 514,800 labelled frames, small enough to learn video
diffusion on one GPU. This repo gives you **one route** from the data to a working (toy) video generation model, with
a checkpoint at every step and a scorer that tells you whether your dancers have the right number of limbs.

<p align="center"><img src="hf/figs/dataset_contact_sheet.png" width="800"></p>

## The route (Colab T4: ~1 h · RTX 4090: ~20 min)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sprited-ai/dancing-stick-figures/blob/main/notebooks/dancing_stick_figures_colab.ipynb)
— the same five steps below, with pictures and plain-language explanations (written so a curious 11-year-old can follow). The notebook ships **with outputs from a real run**, so you can read the answer sheet before pressing anything.

| step | command | what you get |
|---|---|---|
| **0 · setup** | `git clone https://github.com/sprited-ai/dancing-stick-figures && cd dancing-stick-figures && pip install -r train/requirements.txt` | code + deps |
| **1 · data** (0.85 GB) | `hf download sprited/dancing-stick-figures --repo-type dataset --include "mini/*" --local-dir data/hf` | 64² frames + skeleton labels |
| **2 · cache** (~3 min) | `python -m train.cache --data data/hf/mini --out data/cache --splits train,val` | one fast uint8 file |
| **3 · image model** (T4 ~15 min / 4090 ~4 min) | `python -m train.video_ddpm --cache data/cache --out runs/img64 --size 64 --frames 1 --batch 64 --steps 1500 --sample_every 500 --amp fp16` | `runs/img64/sample_001500.png` — 64 figures drawn from noise |
| **4 · video model** (T4 ~35 min / 4090 ~6 min) | `python -m train.video_ddpm --cache data/cache --out runs/vid64 --size 64 --frames 8 --ar_ctx 8 --stride 2 --batch 4 --accum 2 --steps 1200 --sample_every 400 --amp fp16 --init runs/img64/ckpt.pt` | `runs/vid64/rollout_001200.gif` — 8 dances of 5.6 s, generated chunk by chunk (autoregressive diffusion) |
| **5 · score & compare** | `python scripts/compare.py --ckpt runs/img64/ckpt.pt --cache data/cache` | your model vs our reference vs real frames (limb count / attachment / colour) |

Short on time? Replace step 3 with our fully-trained image model:
`hf download sprited/dancing-stick-figures-baselines unet_img64.pt --local-dir ckpts` and use `--init ckpts/unet_img64.pt` in step 4
— dancing figures within a few hundred steps. Or skip training entirely and roll out our finished autoregressive model:
`python scripts/rollout.py --ckpt unet_ar64.pt --seconds 5 --n 8 --out out/dance.gif` (downloads the checkpoint; add `--score --cache data/cache` for the oracle table). On a 24 GB card drop `--amp fp16` (bf16 default), raise `--batch 16`, add `--fast --compile`,
and let the video model run 20k+ steps for clean motion (`--steps 60000` is our reference run).

**How does it make 5-second dances on a small GPU?** The video model only ever looks at a 16-frame window: 8 clean *past*
frames (extra input channels + a mask) and 8 new frames it has to draw. To generate, it draws a first window from nothing,
then feeds its own last 8 frames back as context and draws the next 8, and so on — *autoregressive diffusion*. Length is
free; the cost is a slight flicker at chunk seams (oracle temporal jitter ≈ 1.3× real clips at v0.1). `--ar_ctx 8` turns it on.

**Why image first?** A video is 8 pictures in a row. Teaching one network to draw a good *frame* first, then adding the
"what changes between frames" part on top, is how large systems (Seedance-style) do it too — and on a small GPU it is
roughly 2.5× faster than training video from scratch. Our `--init` makes the video model start out *exactly* as the image
model repeated 8 times (the time-mixing layers begin at zero), so nothing is lost.

## Appendix — everything else in this repo

- **Second architecture, same route:** `train/video_dit_fm.py` — a DiT + flow-matching model (patch 2, logit-normal t, timestep
  shift, image-to-video conditioning via `--i2v_frac`); e.g. `--size 64 --frames 1 --batch 128 --patch 2` then `--frames 8 --batch 8 --accum 2 --init ...`.
- **Bigger:** the full `frames` config (128², depth, normals, segmentation; 4.6 GB): `--include "frames/*"`, then `--size 128 --grad_ckpt` (24 GB cards).
- **Conditional:** `--cond group` trains on the 5 prompt groups with classifier-free guidance.
- **Baselines & numbers:** `hf/README.md` (the dataset card) lists six trained image models and their oracle scores; checkpoints at
  [sprited/dancing-stick-figures-baselines](https://huggingface.co/sprited/dancing-stick-figures-baselines).
- **The oracle:** `eval/oracle.py` parses a rendered figure by colour and counts limbs / checks attachment / colour purity;
  `eval/corrupt.py` validates it on synthetic corruptions; `eval/run_ckpt.py` adds temporal metrics + FVD for video checkpoints.
- **Regenerating the data:** NVIDIA ARDY text-to-motion → `generator/build.py` (see below).
- **Tech-report draft** in `paper/` — frozen; the dataset card is the reference document.

## Layout

| path | what |
|---|---|
| `generator/` | `skeleton.py` (cskel27, FK, orthographic camera), `render.py` (z-buffer capsule rasteriser → colour/depth/normal/seg), `ardy_adapter.py` (NVIDIA ARDY `.npz` → figure frame, bone-length jitter), `build.py` (→ parquet shards, prompt-based splits, QA flags), `export_motion.py` (raw motion config), `motion.py` (hand-keyed styles, tooling only) |
| `train/` | `cache.py`, `video_ddpm.py` (factorised 3D UNet, v-pred, DDIM, EMA; T=1 = image model), `video_dit_fm.py` (factorised video DiT, rectified flow, logit-normal t, shift, I2V conditioning), `log2tb.py` |
| `eval/` | `oracle.py` (rule-based anatomy metrics), `corrupt.py` (oracle validation on synthetic corruptions), `score_images.py`, `run_ckpt.py` (checkpoint watcher: oracle + FVD → TensorBoard), `fvd.py` |
| `scripts/` | `contact_sheet.py`, `evolution.py`, RunPod helpers, watchdogs used for the paper runs |
| `hf/` | the dataset card and figures as published |
| `paper/` | tech report draft (`REPORT.md`), reviews, results JSON |
| `prompts/v1.txt` | the 143 motion prompts (`*` marks held-out groups) |
| `viewer/` | Vite/React clip viewer (`npm --prefix viewer run dev`) |

## Regenerating the data

Requires [NVIDIA ARDY](https://github.com/nv-tlabs/ardy) (text-to-motion) and its gated Llama-3 text encoder.

```bash
bash scripts/ardy_batch_v1.sh                                   # prompts × 10 seeds → ardy_out/v1/<group>/<slug>_s<seed>.npz
python -m generator.build --npz ardy_out/v1 --prompts prompts/v1.txt --out data/v1     # 1,430 clips × 3 cams → parquet (~25 min, CPU)
python -m generator.export_motion --npz ardy_out/v1 --prompts prompts/v1.txt --out data/v1/motion
python scripts/contact_sheet.py data/v1 out/
```

Everything is deterministic from `clip_id` (body, cameras, split).

## Status

v0.1 (2026-08-18): dataset public, route above verified on a fresh machine, image baselines at 64²/128², video baselines
finishing. `STATUS.md` = live state. Roadmap (v0.2): video checkpoints + numbers, templated dense captions, more prompts,
a learned pose regressor / anomaly detector.

## License

Code MIT. Dataset CC0-1.0 (motion generated with NVIDIA ARDY under the NVIDIA Open Model License, which claims no
ownership of outputs).
