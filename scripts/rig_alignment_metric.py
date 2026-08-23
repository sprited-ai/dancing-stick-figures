"""Rig-space prompt alignment: does the emitted rig perform the intended motion?

Compares the model's own rig output (eval_m6 `rigs_10step.npy`, one video per
prompt) against ground-truth rig windows of the same prompts from the training
cache.  Signatures are computed per joint from 2D screen trajectories:

  speed      mean per-frame displacement, per joint            [27]
  amplitude  positional std over time, per joint               [27]
  periodicity dominant-frequency power ratio (0.5-5 Hz band)
             for end effectors (hands, feet, head)             [5]

Reported per prompt: cosine similarity of the speed and amplitude profiles
(does the RIGHT set of joints move?), relative magnitude error, and the
periodicity match for end effectors (wave = oscillatory wrist, not a single
raise).  Quantity metrics cannot see any of this; this metric runs entirely
on the model's self-emitted rig, so it must be read next to the rig-pixel
consistency probe (`v9_rig_overlay.py`).
"""
import argparse
import json
from pathlib import Path

import numpy as np

FPS = 20
END_EFFECTORS = {"RightHand": 10, "LeftHand": 16, "RightFoot": 21, "LeftFoot": 25, "Head": 6}


def signature(rig: np.ndarray) -> dict:
    """rig [T,27,2] in [0,1] -> per-joint signature dict."""
    velocity = np.linalg.norm(np.diff(rig, axis=0), axis=-1)          # [T-1,27]
    speed = velocity.mean(0)                                          # [27]
    amplitude = rig.std(axis=0).mean(-1)                              # [27]
    period = {}
    freqs = np.fft.rfftfreq(rig.shape[0], d=1 / FPS)
    band = (freqs >= 0.5) & (freqs <= 5.0)
    for name, j in END_EFFECTORS.items():
        centred = rig[:, j] - rig[:, j].mean(0)
        power = (np.abs(np.fft.rfft(centred, axis=0)) ** 2).sum(-1)   # [F]
        total = power[1:].sum()
        period[name] = float(power[band].max() / total) if total > 1e-12 else 0.0
    return {"speed": speed, "amplitude": amplitude, "periodicity": period}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, help="eval_m6 output dir with metrics.json + rigs_10step.npy")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", default="", help="output JSON (default: <eval-dir>/rig_alignment.json)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    metrics = json.loads((eval_dir / "metrics.json").read_text())
    prompts = metrics["prompts"]
    rigs = np.load(eval_dir / "rigs_10step.npy").astype(np.float32)    # [N, T_lat, temporal*27*2]
    n, t_lat, dim = rigs.shape
    temporal = dim // (27 * 2)
    rigs = rigs.reshape(n, t_lat * temporal, 27, 2)
    if n != len(prompts):
        raise SystemExit("rig count does not match prompt count")
    frames_needed = rigs.shape[1]

    clips = json.loads((Path(args.cache) / "clips.json").read_text())
    gt_rig = np.load(Path(args.cache) / "rig.npy", mmap_mode="r")
    by_prompt: dict[str, list] = {}
    for cid, row in clips.items():
        if row["split"] == metrics["split"] and row["n"] >= frames_needed:
            by_prompt.setdefault(row["text"], []).append(int(row["start"]))

    def feature(sig: dict) -> np.ndarray:
        return np.concatenate([sig["speed"], sig["amplitude"],
                               np.array([sig["periodicity"][k] for k in END_EFFECTORS])])

    scored = [(p, r) for p, r in zip(prompts, rigs) if p in by_prompt]
    if not scored:
        raise SystemExit("no scored prompts (split mismatch?)")
    ref_features = {}
    for prompt in {p for p, _ in scored}:
        refs = [feature(signature(np.asarray(gt_rig[s:s + frames_needed], np.float32)))
                for s in by_prompt[prompt]]
        ref_features[prompt] = np.mean(refs, axis=0)
    ref_names = list(ref_features)
    ref_matrix = np.stack([ref_features[p] for p in ref_names])
    # z-score every dimension over the reference set so no single scale
    # dominates; the SAME transform is applied to generated signatures.
    mu, sd = ref_matrix.mean(0), ref_matrix.std(0) + 1e-9
    ref_z = (ref_matrix - mu) / sd

    per_prompt = {}
    top1 = top5 = 0
    centered = []
    for prompt, rig in scored:
        gen = (feature(signature(rig)) - mu) / sd
        distance = np.linalg.norm(ref_z - gen, axis=1)
        order = np.argsort(distance)
        rank = int(np.where(np.array(ref_names)[order] == prompt)[0][0]) + 1
        top1 += rank == 1
        top5 += rank <= 5
        target = ref_z[ref_names.index(prompt)]
        centered.append(cosine(gen, target))
        per_prompt[prompt] = {"retrieval_rank": rank,
                              "centered_cosine": centered[-1],
                              "nearest": ref_names[order[0]]}
    n_scored = len(scored)
    report = {
        "n_prompts_scored": n_scored,
        "retrieval_top1": top1 / n_scored,
        "retrieval_top5": top5 / n_scored,
        "chance_top1": 1 / len(ref_names),
        "chance_top5": 5 / len(ref_names),
        "mean_centered_cosine": float(np.mean(centered)),
        "note": "self-emitted rig, z-scored signatures (speed+amplitude+end-effector "
                "periodicity); retrieval asks whether the rig performs a motion nearest "
                "to its own prompt's reference. Read next to the rig-pixel consistency probe.",
        "per_prompt": per_prompt,
    }
    out = Path(args.out) if args.out else eval_dir / "rig_alignment.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_prompt"}, indent=2))


if __name__ == "__main__":
    main()
