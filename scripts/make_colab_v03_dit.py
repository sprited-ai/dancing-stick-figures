"""Create the DiT-first Colab workflow from the maintained v0.2 notebook."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/dancing_stick_figures_colab_v0_2.ipynb"
OUTPUT = ROOT / "notebooks/dancing_stick_figures_colab_v0_3.ipynb"


def lines(text: str) -> list[str]:
    return text.strip().splitlines(keepends=True)


REPLACEMENTS = {
    0: r"""
# 🕺 Dancing Stick Figures v0.3 — train an image DiT, then a video DiT

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sprited-ai/dancing-stick-figures/blob/main/notebooks/dancing_stick_figures_colab_v0_3.ipynb)

*A hands-on notebook for readers with basic Python and neural-network familiarity. The reference `64²` route targets a 16 GB Tesla T4; `32²` is a faster sanity check.*

**What you'll do**
1. 👀 Inspect the videos and their hidden skeleton labels.
2. 🎨 Train a small **image DiT** from noise.
3. 🎬 Reuse those weights in a **video DiT** that jointly generates a 3.2-second action at the native 20 fps.
4. 🤖 Generate from your own prompt and diagnose visible structural failures.

The released clips contain 120 frames at 20 fps. The workflow trains on the first 64 frames of each clip (3.2 seconds at the native 20 fps): the source motions concentrate their prompted action early, so the later frames often continue or idle. The paper's benchmarks and diagnostics use this same fixed first-64-frame window; the complete 120-frame clips remain available in the dataset.

Before you start: **Runtime → Change runtime type → GPU** (the conservative batch targets a 16 GB T4; the final verification record reports the actual device and peak allocation).
""",
    1: r"""
#@title 0. Setup (≈2 min) — grab the code and the small version of the dataset
import os, sys, subprocess, glob, time
V03_STARTED = time.time()
if not os.path.exists("dancing-stick-figures"):
    !git clone -q https://github.com/sprited-ai/dancing-stick-figures
else:
    !git -C dancing-stick-figures pull -q  # refresh a clone left over from an earlier session
%cd dancing-stick-figures
!pip install -q -r train/requirements.txt 2>&1 | tail -1
DATA_DOWNLOAD_ATTEMPTS = 3
download_cmds = [
    ["hf", "download", "sprited/dancing-stick-figures", "--repo-type", "dataset", "--include", pattern, "--local-dir", "data/hf"]
    for pattern in ("mini/*", "motion/val-*")
]
for attempt in range(1, DATA_DOWNLOAD_ATTEMPTS + 1):
    results = [subprocess.run(cmd, text=True, capture_output=True) for cmd in download_cmds]
    mini_files = glob.glob("data/hf/mini/*.parquet"); motion_files = glob.glob("data/hf/motion/*.parquet")
    if all(result.returncode == 0 for result in results) and mini_files and motion_files: break
    print(f"dataset download attempt {attempt} did not finish; retrying")
else:
    raise RuntimeError("Dataset download did not produce mini parquet files after three attempts")
print(len(mini_files), "mini shards,", len(motion_files), "motion shard(s)")
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — switch the runtime to GPU!")

#@title 0b. Choose a resolution
IMAGE_SIZE = "64" #@param ["32", "64"]
IMAGE_SIZE = int(IMAGE_SIZE)
VIDEO_FRAMES, FRAME_STRIDE = 64, 1
IMAGE_BATCH = 128 if IMAGE_SIZE == 32 else 64
VIDEO_BATCH = 8 if IMAGE_SIZE == 32 else 4
RUN_TAG = f"{IMAGE_SIZE}px"
IMAGE_RUN, VIDEO_RUN = f"runs/dit_img_{RUN_TAG}", f"runs/dit_vid64_{RUN_TAG}"
print(f"resolution={IMAGE_SIZE}² · factorised DiT · {VIDEO_FRAMES} frames at 20 fps")
print("32² is a faster sanity check; 64² is the reference setting.")
""",
    2: r"""
## 🧩 The fixed reference backbone

The same **factorised video DiT** is used for both stages. At `T=1`, it is an image generator. At `T=64`, it becomes a video generator.

```text
noisy RGBA frames [4, T, size, size]
             │
             ▼  split each frame into 4×4 patches
spatial attention: patches within the same frame exchange information
temporal attention: the same patch position exchanges information across all frames
text cross-attention: the complete prompt conditions every block
             │
             ▼
predicted flow from noise toward clean video [4, T, size, size]
```

Spatial and temporal attention alternate. This makes the division of labour visible in code and lets each generated time point use both earlier and later context inside the 3.2-second training window. A frozen T5-small encoder supplies prompt tokens; the DiT itself learns the rendered figure and motion from this dataset.

The exercise fixes the architecture and exposes its source. The only initial choice is `32²` for a quick sanity check or `64²` for the main run.
""",
    3: r"""
#@title Read the actual backbone and image-to-video initialization code (optional)
import inspect
from train.video_dit_fm import Attention, TextCrossAttention, Block, VideoDiT, prepare_warmstart_state
for component in (Attention, TextCrossAttention, Block, VideoDiT, prepare_warmstart_state):
    print(f"\n# --- {component.__name__} ---")
    print(inspect.getsource(component))
print("To experiment later, edit train/video_dit_fm.py in Colab's Files pane. The workflow below uses this reference code unchanged.")
""",
    8: r"""
## 2 · 🎨 Train an image model

First we unpack the frames into a fast cache. Then we train the DiT with one frame at a time. Flow matching draws a noisy point between a real frame and Gaussian noise; the model learns the direction that leads back toward the clean frame.

This stage teaches spatial structure before temporal modelling begins. Samples are saved every 500 steps so you can inspect the learning curve.
""",
    10: r"""
#@title Train the image DiT — show validation and samples every 500 steps
import glob, os, subprocess, sys, time
from IPython.display import display
from PIL import Image
STEPS = 5000  #@param {type:"integer"}
REPORT_EVERY = 500
image_started = time.perf_counter()
cmd = [sys.executable, "-u", "-m", "train.video_dit_fm",
       "--cache", "data/cache", "--out", IMAGE_RUN, "--arch", "dit",
       "--size", str(IMAGE_SIZE), "--frames", "1", "--stride", "1",
       "--first_frames", "64", "--batch", str(IMAGE_BATCH),
       "--steps", str(STEPS), "--dim", "384", "--depth", "12",
       "--heads", "6", "--patch", "4", "--cond", "text",
       "--cfg_drop", "0.1", "--fg_weight", "2", "--fast", "--compile",
       "--workers", "2", "--sample_every", str(REPORT_EVERY),
       "--val_every", str(REPORT_EVERY)]
shown_mtime = {p: os.path.getmtime(p) for p in glob.glob(f"{IMAGE_RUN}/sample_0*.png")}
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
for line in process.stdout:
    if line.startswith(("cached", "init", "step", "wrote", "params", "Error", "Traceback")):
        print(line, end="", flush=True)
    current = {p: os.path.getmtime(p) for p in glob.glob(f"{IMAGE_RUN}/sample_0*.png")}
    for sample_path in sorted(p for p, mtime in current.items() if mtime > shown_mtime.get(p, 0)):
        if "raw" not in sample_path:
            print(f"\n{os.path.basename(sample_path)}")
            display(Image.open(sample_path).resize((512, 512), Image.Resampling.NEAREST))
    shown_mtime = current
return_code = process.wait()
if return_code:
    raise subprocess.CalledProcessError(return_code, cmd)
IMAGE_WALL_SECONDS = time.perf_counter() - image_started
print(f"image stage wall time: {IMAGE_WALL_SECONDS / 60:.1f} min")
""",
    11: r"""
#@title Look at what it learned — fixed-noise drawings at each checkpoint
from PIL import Image
for f in sorted(glob.glob(f"{IMAGE_RUN}/sample_0*.png")):
    if "raw" in f: continue
    print(f.split("/")[-1]); display(Image.open(f).resize((512, 512), Image.NEAREST))
""",
    12: r"""
> **Why do early samples look rough?** The default 5,000-step exercise exposes the complete training loop but does not match the released 30,000-step reference checkpoint. Your image checkpoint is still useful: it supplies the spatial weights for the video stage below.
""",
    13: r"""
## 3 · 🎬 Make it move — train a 3.2-second video generator

We now build the same DiT with 64 temporal positions and initialise every compatible spatial, text, and output weight from your image model. Temporal positions and temporal attention then learn from video.

Each source clip remains available in full at 120 frames and 20 fps. For this training exercise we keep the first 64 frames — 3.2 seconds at the native 20 fps. The source motions perform their prompted action early (the generator does not pace an action to the requested duration), so this window concentrates training on the action itself. This is the protocol used by the released prompt-conditioned DiT reference.

> **Prompt conditioning.** Both stages use complete prompts through frozen T5-small token features. The model is prompt-conditioned; the notebook demonstrates sensitivity to prompt changes but does not claim a calibrated semantic adherence score.

> **Why no Video VAE?** At `32²`–`64²`, direct pixel training is practical. Keeping generated pixels in the renderer's representation avoids adding codec reconstruction errors to the first experiment.

Unlike the earlier autoregressive baseline, this model generates the complete 64-frame window jointly. Every temporal-attention layer can coordinate earlier and later frames at a shared patch position.
""",
    14: r"""
#@title Train the 64-frame video DiT on top of your image DiT
INIT_CKPT = f"{IMAGE_RUN}/ckpt.pt"
assert os.path.exists(INIT_CKPT), f"Missing {INIT_CKPT}; finish image training first."
VSTEPS = 2000  #@param {type:"integer"}
video_started = time.perf_counter()
video_cmd = [sys.executable, "-u", "-m", "train.video_dit_fm",
             "--cache", "data/cache", "--out", VIDEO_RUN, "--arch", "dit",
             "--size", str(IMAGE_SIZE), "--frames", str(VIDEO_FRAMES),
             "--stride", str(FRAME_STRIDE), "--first_frames", "64",
             "--batch", str(VIDEO_BATCH), "--steps", str(VSTEPS),
             "--dim", "384", "--depth", "12", "--heads", "6", "--patch", "4",
             "--cond", "text", "--cfg_drop", "0.1", "--fg_weight", "2",
             "--img_frac", "0.1", "--i2v_frac", "0.2", "--init", INIT_CKPT,
             "--grad_ckpt", "--fast", "--workers", "2",
             "--sample_every", str(VSTEPS), "--val_every", "500"]
video_process = subprocess.Popen(video_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
for line in video_process.stdout:
    if line.startswith(("cached", "init", "step", "wrote", "params", "Error", "Traceback")):
        print(line, end="", flush=True)
video_return_code = video_process.wait()
if video_return_code:
    raise subprocess.CalledProcessError(video_return_code, video_cmd)
VIDEO_WALL_SECONDS = time.perf_counter() - video_started
print(f"video stage wall time: {VIDEO_WALL_SECONDS / 60:.1f} min")
""",
    15: r"""
#@title Type a prompt, then watch four 3.2-second samples
from pathlib import Path
from IPython.display import Image as IPImage
PROMPT = "A person runs forward."  #@param {type:"string"}
PROMPT_DIR = Path(f"out/dit_prompt_{RUN_TAG}")
PROMPT_DIR.mkdir(parents=True, exist_ok=True)
(PROMPT_DIR / "prompt.txt").write_text(PROMPT + "\n")
!python -m eval.post_eval_t2v --ckpt $VIDEO_RUN/ckpt.pt --out $PROMPT_DIR --prompts_file $PROMPT_DIR/prompt.txt --same_prompt "$PROMPT" --n 4 --steps 30 --cfg 3 --batch 1 --seed 1234 --fps 20 --strip_frames 0,21,42,63 --save_rgba 2>&1 | grep -v FutureWarning
MINE_GIF = str(PROMPT_DIR / "fixed_prompt_varied_noise_labeled.gif")
print(f"yours — prompt: {PROMPT!r}"); display(IPImage(filename=MINE_GIF))
""",
    16: r"""
> The image stage teaches the model how a clean figure is assembled. The video stage teaches how those parts change together over a 3.2-second window. Compare the four samples: changing noise should change the motion while preserving a coherent figure. If they remain rough, the first useful experiment is simply to train longer and compare the fixed-noise strips again.
""",
    18: r"""
#@title Measure visible structure in your image DiT and real validation frames
import json
MINE_JSON = f"out/dit_image_{RUN_TAG}.json"
!python -m eval.score_images --ckpt $IMAGE_RUN/ckpt.pt --cache data/cache --n 128 --steps 30 --cfg 3 --out $MINE_JSON 2>&1 | tail -1
mine = json.load(open(MINE_JSON))
print(f"{'':22s} {'lie':>6s} {'tvr':>6s} {'clean':>6s}")
print(f"{'your image DiT':22s} {mine['lie']:6.3f} {mine['tvr']:6.3f} {mine['clean_frac']:6.2f}")
print(f"{'real reference':22s} {mine['floor']['lie']:6.3f} {mine['floor']['tvr']:6.3f} {mine['floor']['clean_frac']:6.2f}")
""",
    19: r"""
#@title Verification record — timings, memory, and required artifacts
from pathlib import Path
import re
def peak_gb(path):
    values = [float(x) for x in re.findall(r"peak ([0-9.]+)GB", Path(path).read_text())]
    return max(values) if values else float("nan")
required = [f"{IMAGE_RUN}/ckpt.pt", f"{VIDEO_RUN}/ckpt.pt", MINE_GIF, MINE_JSON]
missing = [path for path in required if not Path(path).exists()]
assert not missing, f"missing required artifacts: {missing}"
verification = {
    "image_wall_seconds": IMAGE_WALL_SECONDS,
    "video_wall_seconds": VIDEO_WALL_SECONDS,
    "image_peak_gb": peak_gb(IMAGE_RUN + "/log.txt"),
    "video_peak_gb": peak_gb(VIDEO_RUN + "/log.txt"),
    "total_wall_seconds": time.time() - V03_STARTED,
    "gpu": torch.cuda.get_device_name(0),
    "resolution": IMAGE_SIZE,
    "image_steps": STEPS,
    "video_steps": VSTEPS,
    "video_frames": VIDEO_FRAMES,
    "frame_stride": FRAME_STRIDE,
}
Path("out").mkdir(exist_ok=True)
Path("out/v03_completion.json").write_text(json.dumps(verification, indent=2))
print(json.dumps(verification, indent=2))
print("V03_COMPLETE=1")
""",
    20: r"""
## 🚀 Where to go next

- Train the same fixed backbone longer and compare the fixed-noise strips.
- Read the printed `VideoDiT` and `Block` source to see exactly where spatial, temporal, and text attention occur.
- Change the training input or add a conditioning signal once you understand the reference run.
- Use the exact state labels for another task, such as estimating a skeleton from an image.

Dataset: https://huggingface.co/datasets/sprited/dancing-stick-figures · Code: https://github.com/sprited-ai/dancing-stick-figures · Made by Sprited.
""",
}


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    for index, replacement in REPLACEMENTS.items():
        notebook["cells"][index]["source"] = lines(replacement)
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
