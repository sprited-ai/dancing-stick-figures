"""Per-clip body parameters for the rig renderer, recovered deterministically.

Body proportions are a pure function of the clip id (generator/build.py seeds
one RNG per motion with sha256(clip stem)), so stroke and head radius can be
reconstructed without touching the parquet source. Verified bit-exact by the
2026-08-23 pipeline scoping. Radii are emitted for the 64 px cache (half the
128 px render values); head_r_m is a non-randomized constant (0.125 m).
"""
import argparse
import json
import random
from pathlib import Path

from generator.build import _h, sample_body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    clips = json.loads(Path(args.clips).read_text())
    out = {}
    for cid in clips:
        group, stem, _cam = cid.split("/")
        rng = random.Random(_h(f"{group}/{stem}"))
        body = sample_body(rng)
        out[cid] = {
            "px_per_m": body.px_per_m,
            "stroke_px64": body.stroke / 2,
            "head_r_px64": 0.125 * body.px_per_m / 2,
        }
    Path(args.out).write_text(json.dumps(out, indent=0) + "\n")
    print(f"{len(out)} clips")


if __name__ == "__main__":
    main()
