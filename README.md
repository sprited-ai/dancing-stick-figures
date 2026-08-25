# Dancing Stick Figures — train your first video generation model, end to end

Code behind the dataset **[sprited/dancing-stick-figures](https://huggingface.co/datasets/sprited/dancing-stick-figures)**:
1,340 six-second motions rendered from three cameras (4,020 videos), 482,400 labelled frames, small enough to learn video
diffusion on one GPU. This repo gives you **one route** from the data to a working (toy) video generation model, with
a checkpoint at every step and a scorer that tells you whether each coloured limb remains present and connected.

<p align="center"><img src="hf/figs/dataset_contact_sheet.png" width="800"></p>

> **Sibling dataset:** the same source motion collection rendered as a volumetric chibi character with depth/normals/segmentation and motion-grounded captions — [sprited/dancing-chibi-figures](https://huggingface.co/datasets/sprited/dancing-chibi-figures) ([code](https://github.com/sprited-ai/dancing-chibi-figures)). Same `clip_id`s and cameras: every frame here has a paired chibi frame.

## The route (one GPU; 32² sanity tier or 64² reference tier)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sprited-ai/dancing-stick-figures/blob/main/notebooks/dancing_stick_figures_colab_v0_3.ipynb)
— the same five steps below, with pictures and a guided walkthrough of the model and its outputs.

| step | command | what you get |
|---|---|---|
| **0 · setup** | `git clone https://github.com/sprited-ai/dancing-stick-figures && cd dancing-stick-figures && pip install -r train/requirements.txt` | code + deps |
| **1 · data** (0.79 GB) | `hf download sprited/dancing-stick-figures --repo-type dataset --include "mini/*" --local-dir data/hf` | 64² frames + skeleton labels |
| **2 · cache** (~3 min) | `python -m train.cache --data data/hf/mini --out data/cache --splits train,val` | one fast uint8 file |
| **3 · image DiT** | `python -m train.video_dit_fm --cache data/cache --out runs/dit_img64 --arch dit --size 64 --frames 1 --first_frames 64 --batch 64 --steps 2000 --patch 4 --cond text --fg_weight 2 --grad_ckpt --fast` | fixed-noise image samples and a reusable spatial checkpoint |
| **4 · video DiT** | `python -m train.video_dit_fm --cache data/cache --out runs/dit_vid40 --arch dit --size 64 --frames 40 --stride 1 --first_frames 64 --batch 4 --steps 2000 --patch 4 --cond text --fg_weight 2 --img_frac .1 --i2v_frac .2 --init runs/dit_img64/ckpt.pt --grad_ckpt --fast` | jointly generated two-second windows at the native 20 fps |
| **5 · generate & diagnose** | `python -m eval.post_eval_t2v --ckpt runs/dit_vid40/ckpt.pt --out out/prompt_suite --prompts_file prompts/v1.txt --n 4 --steps 30 --cfg 3 --fps 20` | fixed prompt/noise comparisons, frame strips, and manifests |

The 2,000-step cells are a teaching run, not the released model's full budget. Fixed intermediate samples make it easy
to decide whether another 1,000 steps changed the result before spending more compute.

**How does it remain small enough for one GPU?** The public videos remain 120 frames at 20 fps. The reference training
protocol samples a 40-frame native-cadence window (two seconds at 20 fps) at a random offset inside the first 64
frames of each clip — the same window length ARDY itself generates with, restricted to the span where the prompted
action concentrates (`--first_frames 64`). Native dataset evaluation still uses complete 120-frame clips. Within
each transformer block, spatial attention exchanges information inside one frame and temporal attention exchanges
information across all 40 frames at the same patch position. Patchifying each frame and training on windows keeps the
baseline practical without a Video VAE.

**Why image first?** A video is a sequence of pictures. Teaching one network to draw a good *frame* first, then adding
the "what changes between frames" part, separates spatial learning from temporal learning while keeping one backbone.
Our `--init` copies every compatible spatial and text-conditioning tensor. A one-frame image model keeps fresh video
time positions; when extending an existing video checkpoint to a new length, the learned temporal positions are
interpolated. The image and video stages therefore use one readable backbone and one direct pixel representation.

## Appendix — everything else in this repo

- **Compact convolutional alternative:** `train/video_ddpm.py` — a factorised 3D UNet with v-prediction, DDIM, image warm-starting, and optional autoregressive continuation. It is retained for architecture experiments rather than used for the main qualitative figure.
- **Bigger:** the full `frames` config (128², depth, normals, segmentation; 4.4 GB): `--include "frames/*"`, then `--size 128 --grad_ckpt` (24 GB cards).
- **Conditioning:** `--cond text` uses complete prompts through frozen T5-small features; `--cond group` is the coarser category baseline.
- **Baselines & numbers:** `hf/README.md` (the dataset card) lists the historical image models and structural scores; checkpoints at
  [sprited/dancing-stick-figures-baselines](https://huggingface.co/sprited/dancing-stick-figures-baselines).
- **Structural evaluator:** `eval/oracle.py` parses a rendered figure by colour and counts limbs / checks attachment / colour purity;
  `eval/corrupt.py` validates it on synthetic corruptions; `eval/run_ckpt.py` adds temporal metrics + FVD for video checkpoints.
- **Reconstructing the data:** rebuild the released visual data from the public motion table with `generator/rebuild_from_motion.py`; ARDY is not required (see below).
- **Tech report** in `paper/` — native 120-frame diagnostics, reconstruction evidence, and the factorised-DiT reference route; the dataset card remains the schema reference. `paper/EXPERIMENT_MATRIX.md` separates paper evidence from additional UNet, VAE, and autoregressive research.

## Layout

| path | what |
|---|---|
| `generator/` | `skeleton.py` (cskel27, FK, orthographic camera), `render.py` (z-buffer capsule rasteriser → colour/depth/normal/seg), `rebuild_from_motion.py` (public motion → frame dataset), `build.py` (ARDY output → parquet), `export_motion.py` (raw motion config) |
| `train/` | `cache.py`, `video_ddpm.py` (factorised 3D UNet, v-pred, DDIM, EMA; T=1 = image model), `video_dit_fm.py` (factorised video DiT, rectified flow, logit-normal t, shift, I2V conditioning), `log2tb.py` |
| `eval/` | `oracle.py` (rule-based structural metrics), `corrupt.py` (controlled metric validation), `score_images.py`, `run_ckpt.py` (structural + motion metrics and FVD), `fvd.py` |
| `scripts/` | `contact_sheet.py`, `evolution.py`, RunPod helpers, watchdogs used for the paper runs |
| `hf/` | the dataset card and figures as published |
| `paper/` | current dataset report (`paper.tex`), review guide, experiment matrix, and results JSON; `REPORT.md` is the archived v0.1 draft |
| `prompts/v1.txt` | the 143 generated motion prompts (134 released; `prompts/v02_excluded.txt` lists the nine removed by visual QA, with reasons) |
| `viewer/` | Vite/React clip viewer (`npm --prefix viewer run dev`) |

## Reconstructing the released data

The released `motion` configuration is sufficient to recreate the rendered dataset; ARDY is not required for this
route:

```bash
hf download sprited/dancing-stick-figures --repo-type dataset \
  --include "motion/*" --include "mini/*" --local-dir data/hf
python -m generator.rebuild_from_motion --motion data/hf/motion --out data/rebuilt_frames --workers 8
python scripts/verify_rebuild.py --rebuilt data/rebuilt_frames --mini data/hf/mini --out out/rebuild_check.json
```

The verifier requires discrete motion, camera, body, split, and projected-joint fields to agree exactly; recomputed
root headings use an explicit numeric tolerance. It compares colour and segmentation after the published 64px
downsampling transform. `joint_visible` means that a part owns at least one raster pixel, so it uses a declared bit
disagreement tolerance: a single boundary pixel can flip the stored boolean. PNG and parquet byte hashes are not a
stable contract across numerical-library versions.

For a course-specific rendering, instructors can keep a private seed and apply a small renderer variation without
changing motions or the seed split:

```bash
python -m generator.rebuild_from_motion --motion data/hf/motion --out data/course_frames --workers 8 \
  --variant_seed course-private-2026 --variant_config configs/instructor_variant.example.json
```

To generate a new source-motion collection, install [NVIDIA ARDY](https://github.com/nv-tlabs/ardy) and its gated
Llama-3 text encoder, then run the commands below. This is a variation path, not a bit-exact reconstruction of the
released motions:

```bash
bash scripts/ardy_batch_v1.sh                                   # 143 prompts × seeds 0..9 → ardy_out/v1/<group>/<slug>_s<seed>.npz
python -m generator.build --npz ardy_out/v1 --prompts prompts/v1.txt --out data/v1     # clips × 3 cams → parquet (~25 min, CPU); apply scripts/curate_prompts.py for the released 134-prompt cut
python -m generator.export_motion --npz ardy_out/v1 --prompts prompts/v1.txt --out data/v1/motion
python scripts/contact_sheet.py data/v1 out/
```

The batch script writes `ardy_out/v1/generation_manifest.json` with the ARDY source revision, resolved model name,
Hugging Face checkpoint revision, prompt count, seeds, and run timestamps. Set `ARDY_ROOT`, `ARDY_PYTHON`, or
`ARDY_OUT` when ARDY, its environment, or the output directory lives elsewhere.

The public body and cameras are deterministic from `clip_id`; an instructor seed intentionally selects a different,
repeatable rendering.

## Status

v0.2 is in final release QA. The full-prompt factorised UNet has completed its three-seed, 120-frame evaluation, and
the public-motion reconstruction passes all 514,800 frames in both released tiers. On a Tesla T4, the released 64²
lesson completes its 2k image and 1.2k video stages at 0.76/1.12 seconds per update, peaks at 6.9/11.3 GB, and produces
a 120-frame rollout. The same release source completed setup through typed-prompt generation and scoring on an RTX
4090 in 944 seconds; hardware-specific timings are kept separate.
`STATUS.md` records the live engineering state.

## License

Code MIT. Dataset CC0-1.0. Motions were generated with ARDY's 20-fps Core model; the original generation record did
not retain the checkpoint revision. ARDY's code is Apache-2.0, and its released checkpoints are governed by the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/),
which states that NVIDIA claims no ownership of generated outputs.
