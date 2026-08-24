"""SRE validation gates (declared in paper/refs/sre_design.md before training).

    python -m eval.sre_validate --ckpt results/sre_v1/ckpt_final.pt --cache cache/mini \
        --split val --data data/v1 --corrupt-n 128 --out results/sre_v1/validation.json

Gate 1 — held-out real renders: mean joint error (px at cache resolution), PCK@2px/@4px,
per-joint means. Target: mean error well under the 1.6 px capsule radius.
Gate 2 — malformed-render sanity (corruption harness of eval/corrupt.py): swapped-limb and
extra-arm renders must produce LARGE errors on the corrupted joints and near-baseline errors
elsewhere — no hallucinated clean skeleton. Thresholds declared here, before any training
result existed: affected >= 3x the real-render baseline, unaffected <= 2x.
Gate 3 — off-screen handling: frames with any joint outside [0,1] must keep visible-joint
error within 1.5x of the all-visible subset.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from train.sre import SRE, RigFrames, RIG_JOINTS
from generator.skeleton import NAMES, project

AFFECTED = {
    "swap_LR_partial": ["LeftForeArm", "RightForeArm"],
    "swap_LR_full": ["LeftForeArm", "LeftHand", "LeftHandEnd", "RightForeArm", "RightHand",
                     "RightHandEnd", "LeftLeg", "LeftFoot", "LeftToeBase", "RightLeg",
                     "RightFoot", "RightToeBase"],
    "stretch_bone": ["LeftHand", "LeftHandEnd", "LeftHandThumb1"],
    "delete_hand": ["RightHandEnd"],
    "extra_arm": ["LeftForeArm", "LeftHand", "LeftHandEnd"],
}
GATED = ("swap_LR_partial", "extra_arm")   # the two corruptions the design gates on


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SRE(size=ckpt["size"], hidden=ckpt.get("hidden", 512)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, ckpt["size"]


def to_model_input(rgba_uint8, model_size, device):
    """Straight-RGBA uint8 [H,W,4] -> premultiplied [0,1] [1,4,S,S], area-downsampled."""
    x = rgba_uint8.astype(np.float32) / 255.0
    a = x[..., 3:4]
    x = np.concatenate([x[..., :3] * a, a], -1)
    if x.shape[0] != model_size:
        f = x.shape[0] // model_size
        x = x.reshape(model_size, f, model_size, f, 4).mean((1, 3))
    return torch.from_numpy(x).permute(2, 0, 1)[None].to(device)


# ------------------------------------------------------- gates 1 + 3 (one pass)
@torch.no_grad()
def gate_real_and_offscreen(model, cache, split, size, device, batch=256):
    ds = RigFrames(cache, split)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=4)
    per_joint_sum = np.zeros(RIG_JOINTS); per_joint_n = np.zeros(RIG_JOINTS)
    pck2 = pck4 = 0.0
    on_sum = on_n = off_sum = off_n = 0.0     # gate 3: all-visible frames vs frames w/ off-screen joints
    off_frames = 0
    for x, rig, visible in loader:
        d = (model(x.to(device)) - rig.to(device)).norm(dim=-1) * size   # [B,27] px
        m = visible.to(device).float()
        dm = (d * m).cpu().numpy(); mc = m.cpu().numpy()
        per_joint_sum += dm.sum(0); per_joint_n += mc.sum(0)
        pck2 += ((d <= 2.0) * m).sum().item(); pck4 += ((d <= 4.0) * m).sum().item()
        allvis = mc.all(1)
        off_frames += int((~allvis).sum())
        on_sum += dm[allvis].sum(); on_n += mc[allvis].sum()
        off_sum += dm[~allvis].sum(); off_n += mc[~allvis].sum()
    n = max(per_joint_n.sum(), 1.0)
    per_joint = {NAMES[i]: round(per_joint_sum[i] / max(per_joint_n[i], 1.0), 4)
                 for i in range(RIG_JOINTS)}
    mean_px = per_joint_sum.sum() / n
    on = on_sum / max(on_n, 1.0); off = off_sum / max(off_n, 1.0)
    return {
        "split": split, "frames": len(ds), "mean_px": round(mean_px, 4),
        "pck2": round(pck2 / n, 4), "pck4": round(pck4 / n, 4), "per_joint_px": per_joint,
        "gate1_pass": bool(mean_px < 1.6),
        "offscreen": {"all_visible_px": round(on, 4), "offscreen_frames_px": round(off, 4),
                      "offscreen_frames": off_frames,
                      "ratio": round(off / max(on, 1e-8), 3) if off_n else None,
                      "gate3_pass": bool(off_n == 0 or off / max(on, 1e-8) <= 1.5)},
    }


# ------------------------------------------------------------------ gate 2
@torch.no_grad()
def gate_corruptions(model, model_size, data, n, seed, device):
    from eval.corrupt import rows_from_parquet, joints_of, body_cam, corruptions
    rows = rows_from_parquet(data, n, seed=seed)
    sums = {}                                  # cond -> [aff_sum, aff_n, unaff_sum, unaff_n]
    for row in rows:
        j3, (body, cam) = joints_of(row), body_cam(row)
        renders = corruptions(j3, body, cam)
        native = next(iter(renders.values())).shape[0]
        j2, _ = project(j3, cam, body.px_per_m)
        gt = np.array([j2[nm] for nm in NAMES], np.float32) / native     # [27,2] normalized
        vis = ((gt >= 0) & (gt <= 1)).all(-1)
        for cond, img in renders.items():
            pred = model(to_model_input(img, model_size, device))[0].cpu().numpy()
            d = np.linalg.norm(pred - gt, axis=-1) * model_size          # px at model scale
            aff = np.isin(NAMES, AFFECTED.get(cond, [])) & vis
            unaff = ~np.isin(NAMES, AFFECTED.get(cond, [])) & vis
            s = sums.setdefault(cond, [0.0, 0, 0.0, 0])
            s[0] += d[aff].sum(); s[1] += aff.sum()
            s[2] += d[unaff].sum(); s[3] += unaff.sum()
    base = sums["real"][2] / max(sums["real"][3], 1)                     # real, all joints ~ unaffected set
    out = {"n_frames": len(rows), "real_baseline_px": round(base, 4), "conditions": {}}
    for cond, (asum, an, usum, un) in sums.items():
        if cond == "real":
            continue
        aff = asum / max(an, 1); unaff = usum / max(un, 1)
        entry = {"affected_px": round(aff, 4), "unaffected_px": round(unaff, 4),
                 "affected_joints": AFFECTED.get(cond, [])}
        if cond in GATED:
            entry["gate2_pass"] = bool(aff >= 3.0 * base and unaff <= 2.0 * base)
        out["conditions"][cond] = entry
    out["gate2_pass"] = all(out["conditions"][c].get("gate2_pass", True) for c in GATED
                            if c in out["conditions"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--data", default=None, help="parquet dir for the corruption gate")
    ap.add_argument("--corrupt-n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, size = load_model(args.ckpt, device)

    report = {"ckpt": args.ckpt, "gates": {}}
    g13 = gate_real_and_offscreen(model, args.cache, args.split, size, device)
    report["gates"]["real_renders"] = g13
    print(f"[gate 1] {args.split}: mean {g13['mean_px']:.3f} px, PCK@2 {g13['pck2']:.4f}, "
          f"PCK@4 {g13['pck4']:.4f}  -> {'PASS' if g13['gate1_pass'] else 'FAIL'} (target < 1.6 px)")
    o = g13["offscreen"]
    print(f"[gate 3] off-screen frames: {o['offscreen_frames_px']} px vs all-visible "
          f"{o['all_visible_px']} px (ratio {o['ratio']}) -> "
          f"{'PASS' if o['gate3_pass'] else 'FAIL'} (target <= 1.5x)")

    if args.data:
        g2 = gate_corruptions(model, size, args.data, args.corrupt_n, args.seed, device)
        report["gates"]["corruptions"] = g2
        print(f"[gate 2] real baseline {g2['real_baseline_px']:.3f} px")
        for cond, e in g2["conditions"].items():
            tag = "" if "gate2_pass" not in e else ("  PASS" if e["gate2_pass"] else "  FAIL")
            print(f"  {cond:16s} affected {e['affected_px']:7.3f}  unaffected {e['unaffected_px']:7.3f}{tag}")
        print(f"[gate 2] -> {'PASS' if g2['gate2_pass'] else 'FAIL'} "
              f"(gated on {', '.join(GATED)}; affected >= 3x, unaffected <= 2x baseline)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
