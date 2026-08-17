"""Controlled corruptions rendered FROM LABELS (we own the renderer), for validating the oracle.

    python -m eval.corrupt --data data/v1 --n 300 --out out/oracle_validation.md

For n held-out (val/test) frames: re-render the true frame from its stored joints/camera/body, then
render 4 corruptions and score all with eval.oracle:
  swap_LR_partial  swap the colours of the two UPPER arms only (red<->blue) — adjacency breaks
  swap_LR_full     swap whole left/right chains (colour-only) — adjacency preserved (expected: NOT caught)
  stretch_bone     LeftForeArm +30 % (re-render with the wrist/hand moved along the bone)
  delete_hand      remove RightHand+RightHandEnd (forearm ends at the wrist)
  extra_arm        add a third arm: duplicate the left arm chain rotated 40 deg about the shoulder
Prints a table: metric x condition (mean), i.e. floor (real) and specificity.
"""
from __future__ import annotations
import argparse, glob, io, json, math, os, random
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from generator.skeleton import Body, Camera, project, NAMES, PARENT
from generator import render as R
from generator.render import render_all
from eval.oracle import score_frame


def rows_from_parquet(data, n, splits=("val", "test"), seed=0):
    files = sorted(glob.glob(os.path.join(data, "*.parquet")))
    files = [f for f in files if any(os.path.basename(f).startswith(s) for s in splits)]
    rng = random.Random(seed); rng.shuffle(files)
    out = []
    for f in files:
        t = pq.read_table(f, columns=["sample_id", "joint_xyz", "cam_yaw", "cam_pitch", "cam_center_x", "cam_center_y",
                                      "px_per_m", "stroke", "bone_scale", "color"])
        idx = list(range(t.num_rows)); rng.shuffle(idx)
        for i in idx[: max(1, n // max(1, len(files)) + 1)]:
            out.append(t.slice(i, 1).to_pylist()[0])
            if len(out) >= n: return out
    return out


def joints_of(row):
    xyz = np.frombuffer(row["joint_xyz"], np.float32).reshape(27, 3)
    return {n: tuple(map(float, xyz[i])) for i, n in enumerate(NAMES)}


def body_cam(row):
    body = Body(px_per_m=float(row["px_per_m"]), stroke=float(row["stroke"]), bone_scale=json.loads(row["bone_scale"]))
    cam = Camera(yaw=float(row["cam_yaw"]), pitch=float(row["cam_pitch"]), center=(float(row["cam_center_x"]), float(row["cam_center_y"])))
    return body, cam


def render_joints(j3, body, cam, palette=None, extra=None):
    j2, dep = project(j3, cam, body.px_per_m)
    old = dict(R.PALETTE)
    if palette: R.PALETTE.update(palette)
    try:
        if extra is None:
            o = render_all(j2, dep, body)
        else:
            o = extra(j2, dep, body)
    finally:
        R.PALETTE.clear(); R.PALETTE.update(old)
    return np.asarray(o["color"])


def corruptions(j3, body, cam):
    out = {}
    out["real"] = render_joints(j3, body, cam)
    # partial swap: upper arms only
    p = {"LeftForeArm": R.PALETTE["RightForeArm"], "RightForeArm": R.PALETTE["LeftForeArm"]}
    out["swap_LR_partial"] = render_joints(j3, body, cam, palette=p)
    # full swap: whole chains
    p = {}
    for a, b in [("LeftForeArm", "RightForeArm"), ("LeftHand", "RightHand"), ("LeftHandEnd", "RightHandEnd"),
                 ("LeftLeg", "RightLeg"), ("LeftFoot", "RightFoot"), ("LeftToeBase", "RightToeBase")]:
        p[a] = R.PALETTE[b]; p[b] = R.PALETTE[a]
    out["swap_LR_full"] = render_joints(j3, body, cam, palette=p)
    # stretch LeftForeArm +30 %: move LeftHand (and children) along the bone
    js = dict(j3)
    e, w = np.array(j3["LeftForeArm"]), np.array(j3["LeftHand"])
    d = (w - e) * 0.3
    for n in ("LeftHand", "LeftHandEnd", "LeftHandThumb1"):
        js[n] = tuple(np.array(j3[n]) + d)
    out["stretch_bone"] = render_joints(js, body, cam)
    # delete right hand: collapse RightHandEnd/Thumb onto RightHand and draw no hand -> emulate by moving
    # hand end to the wrist (mitten of zero length renders as a small blob) -> instead drop the bones:
    def no_hand(j2, dep, b):
        keep = R.DRAWN
        R.DRAWN = [n for n in keep if n not in ("RightHandEnd",)]
        try: return render_all(j2, dep, b)
        finally: R.DRAWN = keep
    out["delete_hand"] = render_joints(j3, body, cam, extra=no_hand)
    # extra arm: third arm = left arm chain rotated 40 deg about the shoulder around the camera axis (approx: about y)
    def third_arm(j2, dep, b):
        o = render_all(j2, dep, b)
        # render the extra arm on top by re-projecting a rotated copy and rasterising only those bones
        return o
    sh = np.array(j3["LeftArm"]); ang = math.radians(40); c, s = math.cos(ang), math.sin(ang)
    jx = dict(j3)
    for n in ("LeftForeArm", "LeftHand", "LeftHandEnd", "LeftHandThumb1"):
        v = np.array(j3[n]) - sh
        v2 = np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]])   # rotate in the x-y (frontal) plane
        jx[n] = tuple(sh + v2)
    base = out["real"].astype(np.float32)
    arm = render_joints(jx, body, cam)
    # composite: draw the rotated arm over the real frame where the rotated arm has alpha (only arm bones differ)
    j2a, depa = project(jx, cam, body.px_per_m)
    def only_arm(j2, dep, b):
        keep = R.DRAWN
        R.DRAWN = ["LeftForeArm", "LeftHand", "LeftHandEnd"]
        try: return render_all(j2, dep, b)
        finally: R.DRAWN = keep
    armonly = render_joints(jx, body, cam, extra=only_arm).astype(np.float32)
    a = armonly[..., 3:4] / 255.0
    comp = base.copy()
    comp[..., :3] = armonly[..., :3] * a + base[..., :3] * (1 - a)
    comp[..., 3] = np.maximum(base[..., 3], armonly[..., 3])
    out["extra_arm"] = comp.astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1"); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="out/oracle_validation.md"); ap.add_argument("--sheet", default="out/_oracle_corruptions.png")
    a = ap.parse_args()
    rows = rows_from_parquet(a.data, a.n)
    conds = ["real", "swap_LR_partial", "swap_LR_full", "stretch_bone", "delete_hand", "extra_arm"]
    acc = {c: {k: [] for k in ("tvr", "lie", "cpe", "fg")} for c in conds}
    stored = {k: [] for k in ("tvr", "lie", "cpe", "fg")}
    sheet = None
    for i, r in enumerate(rows):
        j3 = joints_of(r); body, cam = body_cam(r)
        imgs = corruptions(j3, body, cam)
        for c in conds:
            m = score_frame(imgs[c])
            for k in acc[c]: acc[c][k].append(m[k])
        m = score_frame(np.asarray(Image.open(io.BytesIO(r["color"]["bytes"])).convert("RGBA")))
        for k in stored: stored[k].append(m[k])
        if i < 6:
            row = np.concatenate([imgs[c] for c in conds], 1)
            sheet = row if sheet is None else np.concatenate([sheet, row], 0)
    lines = [f"# Oracle v0 validation — {len(rows)} held-out frames, re-rendered from labels\n",
             "| condition | tvr | lie | cpe | fg |", "|---|---|---|---|---|"]
    lines.append(f"| stored parquet frame | {np.mean(stored['tvr']):.3f} | {np.mean(stored['lie']):.3f} | {np.mean(stored['cpe']):.3f} | {np.mean(stored['fg']):.3f} |")
    for c in conds:
        lines.append(f"| {c} | " + " | ".join(f"{np.mean(acc[c][k]):.3f}" for k in ("tvr", "lie", "cpe", "fg")) + " |")
    lines += ["", "tvr = fraction of the 8 limb colours with #components != 1 · lie = fraction of 8 expected colour adjacencies missing (v0; full L/R chain swap is *not* expected to be caught) · cpe = impure foreground fraction · fg = foreground fraction."]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(lines))
    print("\n".join(lines))
    if sheet is not None:
        bg = np.full_like(sheet, 255); al = sheet[..., 3:4] / 255.0
        comp = (sheet[..., :3] * al + 255 * (1 - al)).astype(np.uint8)
        Image.fromarray(comp).resize((comp.shape[1] * 2, comp.shape[0] * 2), Image.NEAREST).save(a.sheet)


if __name__ == "__main__":
    main()
