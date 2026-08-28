"""stickdance oracle v0 — regressor-free anatomy metrics on colour frames.

    from eval.oracle import score_frame, score_frames
    m = score_frame(rgba_uint8_128x128x4)   # dict of metrics (0 = perfect)

Metrics (all from the colour image alone, no labels needed):
  tvr   topology violation rate: per limb colour, #connected components != 1 (0..1 over the 8 limb colours),
        plus missing extremities (head, 2 hands, 2 feet colours absent)
  lie   limb-identity error v0: adjacency graph of colours vs skeleton adjacency. Each distal colour must
        touch its proximal colour; each proximal colour must touch ink. Score = fraction of the 8
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


def _axis_angle(mask):
    """principal-axis angle (rad, in [0,pi)) of a binary mask, or nan."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 6: return np.nan
    x = xs - xs.mean(); y = ys - ys.mean()
    cov = np.array([[np.mean(x * x), np.mean(x * y)], [np.mean(x * y), np.mean(y * y)]])
    w, v = np.linalg.eigh(cov)
    a = np.arctan2(v[1, 1], v[0, 1])
    return a % np.pi


def transition_series(frames_thw4: np.ndarray) -> dict:
    """Return frame-transition signals for seam/within-chunk diagnosis.

    Velocity arrays are aligned to their destination frame (indices 1..T-1),
    and acceleration/jerk arrays to indices 2..T-1.
    """
    labs = [label_colours(f) for f in frames_thw4]
    cents = []
    for _, fg in labs:
        ys, xs = np.nonzero(fg)
        cents.append((xs.mean(), ys.mean()) if len(xs) >= 10 else (np.nan, np.nan))
    c = np.asarray(cents, np.float64)
    centroid_speed = np.linalg.norm(np.diff(c, axis=0), axis=1)
    centroid_accel = np.linalg.norm(np.diff(c, n=2, axis=0), axis=1)

    angle_steps = []
    for i, name in enumerate(NAMES):
        if name == "ink": continue
        ang = np.asarray([_axis_angle(lab == i) for lab, _ in labs])
        d = np.diff(ang)
        angle_steps.append((d + np.pi / 2) % np.pi - np.pi / 2)
    angle_steps = np.asarray(angle_steps)
    def finite_mean_axis0(values):
        finite = np.isfinite(values)
        return np.divide(np.nansum(values, axis=0), finite.sum(axis=0),
                         out=np.full(values.shape[1], np.nan), where=finite.sum(axis=0) > 0)
    angle_speed = finite_mean_axis0(np.abs(angle_steps))
    angle_jerk = finite_mean_axis0(np.abs(np.diff(angle_steps, axis=1)))
    return {
        "centroid_speed": centroid_speed,
        "centroid_accel": centroid_accel,
        "angle_speed": angle_speed,
        "angle_jerk": angle_jerk,
    }


def score_seams(frames_thw4: np.ndarray, seam_frames) -> dict:
    """Split transition signals at true AR seams versus within chunks.

    ``seam_frames`` contains destination frame indices, e.g. 16 denotes the
    transition 15->16. Missing/undefined signals remain NaN rather than being
    silently replaced by a favourable value.
    """
    seam_frames = set(int(x) for x in seam_frames)
    series = transition_series(frames_thw4)
    out = {}
    for name, values in series.items():
        start = 1 if name.endswith("speed") else 2
        indices = np.arange(start, start + len(values))
        seam_mask = np.asarray([i in seam_frames for i in indices])
        for split, mask in (("seam", seam_mask), ("within", ~seam_mask)):
            chosen = values[mask]
            finite = chosen[np.isfinite(chosen)]
            out[f"{split}_{name}"] = float(finite.mean()) if finite.size else np.nan
    return out


def score_video(frames_thw4: np.ndarray) -> dict:
    """[T,H,W,4] -> per-frame means (tvr/lie/cpe/fg) + per-video temporal metrics:
      mass_drift    std/mean over time of each colour's pixel count (limbs appearing/disappearing)
      centroid_speed mean |Δ foreground centroid|, px; this is motion amount, not a one-sided error
      centroid_accel mean |Δ² foreground centroid|, px; high values indicate abrupt motion
      motion_fraction fraction of transitions whose centroid moves by more than 0.25 px
      head_jitter   legacy alias for centroid_speed (kept for checkpoint/result compatibility)
      angle_speed   mean absolute first angular difference of limb principal axes
      angle_jerk    mean over limb colours of mean |second difference| of the limb's principal-axis angle, rad
                    (smooth arcs: small)
      height_var    std over time of the figure's vertical extent / mean extent (figure shouldn't change size)
    Real-data floors: measure with eval.corrupt --video or on cached val windows."""
    per = [score_frame(f) for f in frames_thw4]
    out = {k: float(np.mean([p[k] for p in per])) for k in ("tvr", "lie", "cpe", "fg")}
    labs = [label_colours(f) for f in frames_thw4]
    counts = np.array([[(l == i).sum() for i in range(len(NAMES))] for l, _ in labs], np.float32)
    rel = counts.std(0) / np.maximum(counts.mean(0), 1)
    out["mass_drift"] = float(rel[1:].mean())
    cents, heights = [], []
    for l, fg in labs:
        ys, xs = np.nonzero(fg)
        if len(xs) < 10: cents.append((np.nan, np.nan)); heights.append(np.nan); continue
        cents.append((xs.mean(), ys.mean())); heights.append(ys.max() - ys.min())
    c = np.array(cents); h = np.array(heights)
    d = np.linalg.norm(np.diff(c, axis=0), axis=1)
    speed = float(np.nanmean(d)) if np.isfinite(d).any() else 1e3
    out["centroid_speed"] = speed
    out["head_jitter"] = speed
    accel = np.linalg.norm(np.diff(c, n=2, axis=0), axis=1)
    out["centroid_accel"] = float(np.nanmean(accel)) if np.isfinite(accel).any() else 1e3
    valid_d = d[np.isfinite(d)]
    out["motion_fraction"] = float(np.mean(valid_d > 0.25)) if valid_d.size else 0.0
    out["height_var"] = float(np.nanstd(h) / max(np.nanmean(h), 1)) if np.isfinite(h).any() else 1.0
    speeds, jerks, paths = [], [], []
    for i, n in enumerate(NAMES):
        if n == "ink": continue
        ang = np.array([_axis_angle(l == i) for l, _ in labs])
        if np.isfinite(ang).sum() < 3:
            out[f"ang_path_{n}"] = 0.0
            continue
        # unwrap on the half-circle
        dd = np.diff(ang); dd = (dd + np.pi / 2) % np.pi - np.pi / 2
        v = np.abs(dd); v = v[np.isfinite(v)]
        if v.size: speeds.append(float(v.mean()))
        j = np.abs(np.diff(dd)); j = j[np.isfinite(j)]
        if j.size: jerks.append(float(j.mean()))
        # angular odometer: total principal-axis angle travelled over the clip, rad.
        # Absence and freezing both read 0 -- read beside mass_drift/tvr.
        path = float(v.sum()) if v.size else 0.0
        out[f"ang_path_{n}"] = path
        paths.append(path)
    out["angle_speed"] = float(np.mean(speeds)) if speeds else 0.0
    out["angle_jerk"] = float(np.mean(jerks)) if jerks else 1.0
    out["ang_path_total"] = float(np.sum(paths)) if paths else 0.0
    return out
