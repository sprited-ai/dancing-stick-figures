"""Compare protocol-matched 64-frame and full-clip 120-frame stress tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BASELINES = ("real_reference_b", "repeat_first", "shuffle_frames", "reverse_time", "loop_first_8")
METRICS = (
    "tvr", "lie", "cpe", "mass_drift", "centroid_speed", "centroid_accel",
    "motion_fraction", "angle_speed", "angle_jerk", "height_var",
)


def mean(row, metric):
    value = row[metric]
    return float(value["mean"] if isinstance(value, dict) else value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics64", type=Path, required=True)
    parser.add_argument("--fvd64", type=Path, required=True)
    parser.add_argument("--metrics120", type=Path, required=True)
    parser.add_argument("--reverse64", type=Path, required=True)
    parser.add_argument("--reverse120", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metrics64 = json.loads(args.metrics64.read_text())
    fvd64 = json.loads(args.fvd64.read_text())["fvd"]
    metrics120 = json.loads(args.metrics120.read_text())
    reverse64 = json.loads(args.reverse64.read_text())
    reverse120 = json.loads(args.reverse120.read_text())
    rows = {}
    for baseline in BASELINES:
        row64 = metrics64["baselines"][baseline]
        row120 = metrics120["baselines"][baseline]
        fvd120 = float(row120["fvd"])
        values = {}
        for metric in METRICS:
            v64, v120 = mean(row64, metric), mean(row120, metric)
            values[metric] = {"frames64": v64, "frames120": v120,
                              "ratio_64_over_120": v64 / v120 if v120 else None}
        values["fvd"] = {"frames64": float(fvd64[baseline]), "frames120": fvd120,
                         "ratio_64_over_120": float(fvd64[baseline]) / fvd120}
        rows[baseline] = values
    result = {
        "protocol": {
            "n_per_set": 128,
            "statistical_unit": "source_motion",
            "cadence": "native 20 fps, stride 1",
            "relationship": "same source motions and camera views; 64-frame protocol truncates each 120-frame clip to frames 0..63",
        },
        "rows": rows,
        "reversal_subset_uncertainty": {
            "frames64": reverse64["paired_delta"],
            "frames120": reverse120["paired_delta"],
        },
        "interpretation": (
            "FVD and across-window statistics are horizon sensitive. The 64-frame result is the protocol-matched "
            "reference for 64-frame model rows; the 120-frame result remains a full-clip dataset stress test."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
