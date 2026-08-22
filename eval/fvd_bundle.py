"""Compute FVD for a frozen bundle while embedding the real reference once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.fvd import features, fvd_from_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--features_out", default="", help="optional NPZ of per-video I3D features")
    args = parser.parse_args()

    bundle = np.load(args.bundle)
    if "real_reference_a" not in bundle:
        raise ValueError("bundle must contain real_reference_a")
    print("embedding real_reference_a ...", flush=True)
    reference = features(bundle["real_reference_a"], device=args.device, bs=args.batch)
    all_features = {"real_reference_a": reference}
    result = {"n": int(len(reference)), "feature_dim": int(reference.shape[1]), "fvd": {}}
    for name in bundle.files:
        if name == "real_reference_a":
            candidate = reference
        else:
            print(f"embedding {name} ...", flush=True)
            candidate = features(bundle[name], device=args.device, bs=args.batch)
        all_features[name] = candidate
        result["fvd"][name] = float(fvd_from_features(reference, candidate))
        print(f"{name}: {result['fvd'][name]:.3f}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    if args.features_out:
        np.savez_compressed(args.features_out, **all_features)
        print(f"wrote {args.features_out}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
