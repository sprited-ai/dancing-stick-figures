"""stickdance oracle v0 — regressor-free anatomy metrics on colour frames.

    from eval.oracle import score_frame, score_frames
    m = score_frame(rgba_uint8_128x128x4)   # dict of metrics (0 = perfect)

Metrics (all from the colour image alone, no labels needed):
  tvr   topology violation rate: per limb colour, #connected components != 1 (0..1 over the 12 colours),
        plus missing extremities (head, 2 hands, 2 feet colours absent)
  lie   limb-identity error v0: adjacency graph of colours vs skeleton adjacency. Each distal colour must
        touch its proximal colour; each proximal colour must touch ink. Score = fraction of the 12
        expected adjacencies that are missing. NOTE: a full left<->right chain swap preserves adjacency and
        is NOT detected here (needs geometry -> SRE with the pose regressor). Partial swaps are.
  cpe   colour purity error: fraction of foreground (alpha>0.5) pixels farther than TAU from every palette
        colour (smear / wrong colours).
  fg    foreground fraction (sanity; ~0.03..0.08 for real frames)
Ink = torso/head/clavicles/pelvis colour.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage

INK = (40, 40, 40)
PAL = {  # colour name -> rgb, and skeleton adjacency by name
    "ink": INK,
    "arm_L": (232, 64, 48), "fore_L": (255, 150, 40),
    "arm_R": (40, 110, 230), "fore_R": (80, 200, 240),
    "leg_L": (200, 50, 160), "shin_L": (255, 120, 200),
    "leg_R": (30, 150, 90), "shin_R": (120, 220, 90),
}
NAMES = list(PAL)
COLS = np.array([PAL[n] for n in NAMES], np.float32)          # [9,3]
LIMBS = [n for n in NAMES if n != "ink"]
# expected adjacencies (proximal -> distal). hand/foot share the forearm/shin colour.
ADJ = [("ink", "arm_L"), ("arm_L", "fore_L"), ("ink", "arm_R"), ("arm_R", "fore_R"),
       ("ink", "leg_L"), ("leg_L", "shin_L"), ("ink", "leg_R"), ("leg_R", "shin_R")]
TAU = 60.0     # rgb distance for "this pixel is that colour"
DIL = 2        # dilation (px) when testing adjacency


def label_colours(rgba: np.ndarray):
    """rgba uint8 [H,W,4] (straight alpha) -> label map [H,W] (index into NAMES, -1 = none/impure), fg mask."""
    a = rgba[..., 3] > 127
    rgb = rgba[..., :3].astype(np.float32)
    d = np.linalg.norm(rgb[..., None, :] - COLS[None, None], axis=-1)   # [H,W,9]
    lab = d.argmin(-1)
    pure = d.min(-1) < TAU
    lab = np.where(a & pure, lab, -1)
    return lab, a


def score_frame(rgba: np.ndarray) -> dict:
    lab, fg = label_colours(rgba)
    out = {"fg": float(fg.mean())}
    if fg.sum() < 20:
        return {**out, "tvr": 1.0, "lie": 1.0, "cpe": 1.0, "n_missing_colours": len(LIMBS) + 1}
    # cpe
    out["cpe"] = float(((lab == -1) & fg).sum() / fg.sum())
    # components per colour
    masks = {n: (lab == i) for i, n in enumerate(NAMES)}
    ncomp = {}
    missing = 0
    for n in NAMES:
        m = masks[n]
        if m.sum() < 4:
            ncomp[n] = 0; missing += 1
        else:
            _, k = ndimage.label(m, structure=np.ones((3, 3)))
            ncomp[n] = k
    viol = sum(1 for n in LIMBS if ncomp[n] != 1)
    out["tvr"] = viol / len(LIMBS)
    out["n_missing_colours"] = missing
    out["ncomp"] = ncomp
    # adjacency
    st = np.ones((3, 3), bool)
    dil = {n: ndimage.binary_dilation(masks[n], structure=st, iterations=DIL) for n in NAMES}
    miss = 0
    for a_, b_ in ADJ:
        if masks[a_].sum() < 4 or masks[b_].sum() < 4:
            miss += 1; continue
        if not (dil[a_] & masks[b_]).any():
            miss += 1
    out["lie"] = miss / len(ADJ)
    return out


def score_frames(frames, keys=("tvr", "lie", "cpe", "fg")) -> dict:
    """frames: iterable of rgba uint8 arrays -> mean of each metric."""
    acc = {k: [] for k in keys}
    for f in frames:
        m = score_frame(np.asarray(f))
        for k in keys: acc[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def score_video(frames_thw4: np.ndarray) -> dict:
    """[T,H,W,4] -> per-frame means + temporal: colour-mass drift (std over time of each colour's pixel count
    relative to its mean; large = limbs appearing/disappearing)."""
    per = [score_frame(f) for f in frames_thw4]
    out = {k: float(np.mean([p[k] for p in per])) for k in ("tvr", "lie", "cpe", "fg")}
    counts = np.array([[(label_colours(f)[0] == i).sum() for i in range(len(NAMES))] for f in frames_thw4], np.float32)
    rel = counts.std(0) / np.maximum(counts.mean(0), 1)
    out["mass_drift"] = float(rel[1:].mean())
    return out
