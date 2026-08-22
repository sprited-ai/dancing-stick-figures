"""Repeated-subset stability check for paired controlled FVD comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.fvd import fvd_from_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidate", default="reverse_time")
    parser.add_argument("--subset", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.features)
    a, b, candidate = data["real_reference_a"], data["real_reference_b"], data[args.candidate]
    if not (len(a) == len(b) == len(candidate)):
        raise ValueError("feature sets must have equal sizes")
    rng = np.random.default_rng(args.seed)
    real_real, corrupted = [], []
    for _ in range(args.repeats):
        ia = rng.choice(len(a), args.subset, replace=False)
        ib = rng.choice(len(b), args.subset, replace=False)
        real_real.append(fvd_from_features(a[ia], b[ib]))
        corrupted.append(fvd_from_features(a[ia], candidate[ib]))
    real_real, corrupted = np.asarray(real_real), np.asarray(corrupted)
    delta = corrupted - real_real

    def summary(x):
        return {
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)),
            "range": [float(x.min()), float(x.max())],
        }

    result = {
        "candidate": args.candidate,
        "source_n": int(len(a)),
        "subset_n": args.subset,
        "repeats": args.repeats,
        "seed": args.seed,
        "real_real": summary(real_real),
        "corrupted": summary(corrupted),
        "paired_delta": {
            **summary(delta),
            "positive_fraction": float(np.mean(delta > 0)),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
