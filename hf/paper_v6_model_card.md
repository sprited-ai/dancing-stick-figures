---
license: mit
tags:
- diffusion
- video-diffusion
- synthetic
- stick-figure
- dit
- flow-matching
datasets:
- sprited/dancing-stick-figures
pipeline_tag: text-to-video
---

# Dancing Stick Figures — reference models

Reference checkpoints for [Dancing Stick Figures](https://huggingface.co/datasets/sprited/dancing-stick-figures),
the dataset introduced in *Dancing Stick Figures: An Introductory Dataset for Training Video Generation Models*.
The current paper release is under [`paper-v6/`](tree/main/paper-v6). Historical image, short-video, and
autoregressive baselines remain at the repository root.

**Start here:** [Dataset](https://huggingface.co/datasets/sprited/dancing-stick-figures) ·
[Colab](https://colab.research.google.com/github/sprited-ai/dancing-stick-figures/blob/main/notebooks/dancing_stick_figures_colab_v0_3.ipynb) ·
[Code](https://github.com/sprited-ai/dancing-stick-figures)

## Paper v6 checkpoints

All learned video rows below generate text-conditioned 64-frame, 64×64 clips at 20 fps. The Pixel DiTs operate
directly on RGBA pixels. Mini-Wan generates normalized video latents and must be used with the released f4t4d8 codec
and latent statistics.

| file | model | parameters | initialization | training |
|---|---|---:|---|---:|
| `paper-v6/image_dit_30k.pt` | single-frame Pixel DiT | 39.9M | random | 30k updates |
| `paper-v6/pixel_dit_factorised_random_30k.pt` | factorized Pixel DiT | 39.9M | random | 30k updates |
| `paper-v6/pixel_dit_factorised_image_30k.pt` | factorized Pixel DiT | 39.9M | image checkpoint | 30k updates |
| `paper-v6/pixel_dit_local_mixer_image_30k.pt` | factorized Pixel DiT + local 3×3×3 mixer | 41.8M | image checkpoint | 30k updates |
| `paper-v6/mini_wan_40m_decode_30k.pt` | compact Wan-style latent DiT | 39.4M | random | 30k updates |
| `paper-v6/vae_f4t4d8_10k.pt` | frozen video codec for Mini-Wan | — | random | 10k updates |

The Pixel video runs use batch 8 × accumulation 2, velocity-prediction flow matching, foreground-weighted pixel
loss, and an RGBA clean-prediction auxiliary term of weight 1. Mini-Wan uses a latent flow objective plus its
decoded-RGBA auxiliary. These auxiliary constructions are analogous but not mathematically identical.

Exact hashes, architecture fields, provenance, and evaluation-file mappings are in
[`paper-v6/release_manifest.json`](resolve/main/paper-v6/release_manifest.json). Canonical evaluation JSONs are under
[`paper-v6/evaluations/`](tree/main/paper-v6/evaluations), and fixed-seed 64-frame samples are under
[`paper-v6/samples/`](tree/main/paper-v6/samples).

## Table 3 reference results

The video protocol uses 128 held-out source motions, sampling seeds 0/1/2, 64 frames at stride 1, 50 Euler steps,
and CFG 3. Lower is better for TVR, LIE, CPE, jerk, and FVD. Speed and motion fraction are two-sided diagnostic
signals to compare with the real-reference row.

| model | TVR↓ | LIE↓ | CPE↓ | speed | motion fraction | jerk | FVD↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| real reference windows | .116 | .093 | .037 | .373 | .501 | .073 | 114.7 / ref. |
| codec reconstruction floor | .153 | .118 | .053 | .383 | .507 | .106 | 127.1 |
| factorized Pixel DiT, random | .259 | .042 | .032 | .366 | .431 | .172 | 483.7 |
| factorized Pixel DiT, image init | .161 | .033 | .029 | .350 | .447 | .151 | 319.0 |
| Pixel DiT + local mixer, image init | .149 | .050 | .035 | .355 | .470 | .146 | 282.9 |
| Mini-Wan 40M, decode loss | .199 | .101 | .058 | .423 | .534 | .148 | 182.6 |

## Download the current release

```bash
hf download sprited/dancing-stick-figures-baselines \
  --include "paper-v6/*" --local-dir checkpoints
```

Checkpoint dictionaries contain EMA model weights, the optimizer step, and saved architecture arguments. The codec
checkpoint additionally contains its model state and training metadata. Use the matching trainer/evaluator revision
from the linked code repository; these are research checkpoints rather than a packaged inference API.

## Historical baselines

The repository root retains the completed v0.1 image, 8-frame UNet, and long autoregressive UNet references for
reproducing earlier dataset-card results. Files explicitly labelled `interim`, superseded 10k Mini-Wan checkpoints,
and the no-decode-loss Mini-Wan checkpoint are not part of the current release.

Data are CC0-1.0 and code/checkpoint wrappers are MIT. See the dataset card and paper for scope and limitations.
