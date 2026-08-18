# Dancing Stick Figures — generator, trainers, oracle

Code behind the dataset **[sprited/dancing-stick-figures](https://huggingface.co/datasets/sprited/dancing-stick-figures)**:
a small, fully-labelled synthetic video dataset (1,430 clips · 514,800 RGBA frames · 3D skeleton, camera and
G-buffer per frame) for learning video diffusion on one consumer GPU — plus two reference trainers (UNet /
DiT-flow-matching), the rule-based "oracle" that scores anatomical errors in generated frames, and the renderer
that made the data.

<p align="center"><img src="hf/figs/dataset_contact_sheet.png" width="800"></p>

## Quickstart (one GPU, ~15 min to first samples)

```bash
git clone https://github.com/sprited-ai/dancing-stick-figures && cd dancing-stick-figures
pip install -r train/requirements.txt

# 1. data (5 GB parquet; frames/ + motion/)
hf download sprited/dancing-stick-figures --repo-type dataset --local-dir data/hf

# 2. decode the colour frames once into a uint8 memmap (≈2 min, 30 GB; add --splits train,val to skip test)
python -m train.cache --data data/hf/frames --out data/cache

# 3a. image model, 64², unconditional (UNet, v-pred; ~7 GB, 0.3 s/it on a 4090; figures by step ~2k)
python -m train.video_ddpm --cache data/cache --out runs/img64 --size 64 --frames 1 --batch 128 --steps 30000 --sample_every 1000 --fast --compile

# 3b. video model, 64² × 8 frames (UNet; ~13 GB, 0.36 s/it; dancing figures by ~20k steps)
python -m train.video_ddpm --cache data/cache --out runs/vid64 --size 64 --frames 8 --batch 16 --steps 60000 --fast --compile
#     ...or warm-start it from the image model (Seedance-style stage 2; same quality ~2.5× sooner)
python -m train.video_ddpm --cache data/cache --out runs/vid64i --size 64 --frames 8 --batch 16 --steps 60000 --fast --compile --init runs/img64/ckpt.pt

# 3c. the DiT + flow-matching track (patch 2, ~23 GB at batch 128 images / 16 GB at batch 8×2 clips)
python -m train.video_dit_fm --cache data/cache --out runs/dit_img64 --size 64 --frames 1 --batch 128 --patch 2 --steps 50000 --fast --compile
python -m train.video_dit_fm --cache data/cache --out runs/dit_vid64 --size 64 --frames 8 --batch 8 --accum 2 --patch 2 --shift 2 --img_frac 0.1 --i2v_frac 0.2 --steps 60000 --fast --compile --init runs/dit_img64/ckpt.pt

# 4. score generated frames with the oracle (limb existence / topology / colour purity vs the real-frame floor)
python -m eval.score_images --ckpt runs/img64/ckpt.pt --cache data/cache --n 512
python -m eval.run_ckpt --run runs/vid64 --cache data/cache --n 64      # video: temporal metrics + FVD, TensorBoard
```

Every run writes `runs/<name>/sample_XXXXXX.{png,gif}` (64 or 16 fixed-noise samples), `log.txt`, `ckpt.pt`,
TensorBoard under `tb/`. Flags: `--grad_ckpt` (24 GB cards at 128²), `--cond group` (class-conditional on the 5 prompt
groups + CFG), `--min_snr 5` (min-SNR-γ loss weight, UNet), `--preset 4090-fast|4090-full|4090-mid` (memory-safe presets).
Drop `--compile` if your torch/GPU combination complains; it is only a speed-up.

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

v0.1 (2026-08-18): dataset public; image baselines at 64²/128²; video baselines training. See `STATUS.md` for the live
state and `paper/REPORT.md` for the report skeleton. Roadmap: templated dense captions, more prompts, learned pose
regressor / anomaly detector, Colab notebook, baseline checkpoints on HF.

## License

Code MIT. Dataset CC0-1.0 (motion generated with NVIDIA ARDY under the NVIDIA Open Model License, which claims no
ownership of outputs).
