---
title: "I built an MNIST for video diffusion — and trained a 5-second dancer on a free Colab"
subtitle: "Dancing Stick Figures: a tiny, fully-labelled video dataset, one beginner route from noise to motion, and a robot that counts limbs"
tags: machine-learning, generative-ai, deep-learning, dataset, opensource
cover: https://huggingface.co/sprited/dancing-stick-figures-baselines/resolve/main/unet_ar64_interim_rollout.gif
---

![Eight stick figures dancing for 5.6 seconds, generated chunk by chunk by a 46 M-parameter diffusion model](https://huggingface.co/sprited/dancing-stick-figures-baselines/resolve/main/unet_ar64_interim_rollout.gif)

*Every one of these dancers was drawn from pure noise by a model that trained for a few hours on one GPU. And for every frame in the training data, we know exactly where each arm and leg really is — so a small program can grade the model instead of a human squinting at GIFs.*

This post is also a note to my future self about why I made this.

I could have kept using video models as finished products. But the part I wanted to understand was hidden behind their scale: how motion becomes data, how noise becomes a frame, how separate frames learn to belong to the same moment, and how you tell whether a generated body is actually correct. On real video, every one of those questions quickly turns into terabytes of data, expensive training, and evaluation by eye.

I wanted a world small enough to hold the whole thing in my head and run it end to end — from a sentence, to a moving skeleton, to rendered frames, to noise, and back to motion. The stick figure is not just a shortcut. Its simplicity makes the important parts visible. Colour-coded limbs make failures countable. Tiny frames make experiments possible on one ordinary GPU. And a beginner can watch the same model first learn what a body looks like, then learn how that body changes through time.

For Sprited, this was a line in the sand. We do not only want to build around models made by other people. We want to learn how models are made, build the data and measurements ourselves, and leave behind things another curious person can inspect, rerun, question, and improve. Last week Sprited was a company that shipped sprite tools. This week we shipped a dataset, six baseline models, a Colab notebook and a code repo — our first step toward doing the model work ourselves.

What follows is the build log: what we made, the one route we recommend, the two things that surprised us, and what is still broken.

## The problem: you can't *learn* video diffusion on real data

If you want to understand how a video model like Seedance or Sora works — not read about it, actually train one and watch it fail — you hit two walls immediately.

1. **Data.** Real video datasets are terabytes. A single "small" experiment eats a weekend of downloading before the first gradient step.
2. **Measurement.** When your model produces a person with three arms, how do you *count* that? Real video has no labels for "number of arms". You end up eyeballing samples, which doesn't scale and doesn't teach.

MNIST solved the equivalent problem for image classification in 1998: tiny, clean, one number tells you how you're doing. Video generation never got its MNIST.

## What we made

**[Dancing Stick Figures](https://huggingface.co/datasets/sprited/dancing-stick-figures)** — 1,430 six-second clips of colour-coded stick figures dancing, 514,800 frames at 128×128 (and a 64×64 "mini" version that fits in 0.85 GB), CC0.

![32 random frames from the dataset with their prompts](https://raw.githubusercontent.com/sprited-ai/dancing-stick-figures/main/hf/figs/dataset_contact_sheet.png)

The figures aren't drawn by hand. Each clip starts as a text prompt ("A person does the running man dance"), goes through NVIDIA's ARDY text-to-motion model to become a 27-joint 3D skeleton moving for six seconds, and is then photographed by a virtual camera and rasterised by our own renderer. Because we own the whole pipeline, **every frame ships with everything the renderer knew**:

![colour · segmentation · depth · normals · the skeleton overlay](https://raw.githubusercontent.com/sprited-ai/dancing-stick-figures/main/hf/figs/dataset_labels_row.png)

- the 3D and 2D positions of all 27 joints, and whether each is visible or hidden behind the body,
- the camera (yaw, pitch, scale), the body proportions, the line width,
- a depth map, camera-space normals, and a per-pixel "which bone is this" segmentation,
- the raw motion (rotation matrices, foot contacts) as a separate `motion` config for anyone who wants to retarget it.

And one design choice that turns out to matter more than any of that: **every body part has its own colour**. Left upper arm is red, left forearm orange, right upper arm blue, and so on. That is what makes the next section possible.

## The oracle: a robot that counts limbs

Because the colours are fixed, a rendered figure can be *parsed* by a hundred lines of numpy: find the red pixels, count the connected blobs, check the orange blob touches the red one. We call this the **oracle** and use it to score generated frames on three things:

- **lie** — limb existence error: a limb is missing, or there's an extra one
- **tvr** — topology violation: the limb is there but not attached where it should be
- **clean** — the fraction of frames with zero mistakes

There's a catch that took us a day to accept: real frames don't score zero. When a dancer turns sideways, an arm disappears behind the torso, and to a pixel parser that looks exactly like a missing arm. So we always report the score of *real* frames at the same resolution — the floor — and the goal is to reach it, not zero.

Our best 64² image model does:

| | lie ↓ | tvr ↓ | clean ↑ |
|---|---|---|---|
| UNet, 100k steps | 0.116 | 0.134 | 0.43 |
| real frames (floor) | 0.103 | 0.136 | 0.40 |

i.e. it makes limb-count and attachment mistakes no more often than the data itself. (What the oracle *cannot* see: proportions and joint angles. A figure with a thigh longer than its shin passes. That's next.)

## The route: five commands from noise to motion

We deliberately publish **one** path, not a menu:

1. **data** — `hf download sprited/dancing-stick-figures --include "mini/*"` (0.85 GB)
2. **cache** — decode the frames once into a fast uint8 file (3 min)
3. **image model** — a small UNet learns to draw a single stick figure from noise (T4: 20 min)
4. **video model** — the *same* network, warm-started from step 3, learns what changes between frames (T4: 35 min)
5. **score & compare** — the oracle grades your model against ours and against real frames

The whole thing runs on a free Colab T4 in about an hour, and the notebook ships **with the outputs of a real run** so you can read the answer sheet before you press anything:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sprited-ai/dancing-stick-figures/blob/main/notebooks/dancing_stick_figures_colab.ipynb)

Written so that a curious 11-year-old can follow it. Every cell says what is happening in plain language ("a picture covered in TV static; if you had seen a million stick figures you could guess which specks are probably a leg…").

## Two things that surprised us

**1. Image first, then video — it's a real 2.5×.** Seedance and friends pre-train a text-to-image model and then "add time". On our toy scale that's easy to test cleanly: same UNet, same data, one run from scratch, one initialised from the image model (the time-mixing layers start at exactly zero, so at step 0 the video model *is* the image model repeated 8 times). The warm-started run reached the from-scratch run's 10k-step loss at 4k steps. Not a huge claim, but a clean one, and it's why step 4 of the route takes 35 minutes instead of two hours.

**2. Autoregressive diffusion is what makes clips long enough to be dances.** A fixed 8-frame model produces 0.4 seconds — a twitch, not a dance. Doubling frames doubles memory. So the video model above is trained *chunked*: it sees 8 clean past frames as extra input channels and learns to draw the next 8. At generation time you roll: draw a chunk, feed its tail back as context, draw the next. The GIF at the top is seven chunks — 5.6 seconds — at the memory cost of a 16-frame model. It is still the same diffusion model; the autoregression only changes how you get length. The price, visible if you look closely at that GIF, is a slight flicker at chunk seams. We're measuring that with the oracle's temporal metrics as this post goes out.

## What's still broken (v0.1 honesty)

- **Captions.** There are only 143 distinct prompts, so a text-conditioned model would effectively see 143 classes. Dense, templated captions (camera, body, root motion — Seedance's dynamic/static split) are v0.2.
- **The oracle is blind to geometry.** A learned pose regressor is the fix, and the labels to train it are already in the dataset.
- **Seams.** See above.
- **It's stick figures.** One body preset (jittered ±8 %), no clothes, no scene. That's the point, but don't mistake it for a human-motion dataset.

## Links

- Dataset (3 configs, viewer): https://huggingface.co/datasets/sprited/dancing-stick-figures
- Baseline checkpoints (image 64²/128², video, autoregressive): https://huggingface.co/sprited/dancing-stick-figures-baselines
- Code — generator, trainers, oracle, notebook: https://github.com/sprited-ai/dancing-stick-figures
- The notebook: [Open in Colab](https://colab.research.google.com/github/sprited-ai/dancing-stick-figures/blob/main/notebooks/dancing_stick_figures_colab.ipynb)

If you teach a deep-learning class and try it with students, we'd love to hear what broke. If you're a video-model person and think the oracle is naïve — it is; tell us how you'd score a stick figure. Issues and discussions are open on both repos.

*Motion generated with NVIDIA ARDY (NVIDIA Open Model License; outputs are ours to license). Data CC0, code MIT.*
