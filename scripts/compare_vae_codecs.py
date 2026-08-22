"""Apply a predeclared candidate-versus-baseline codec selection protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def summarize(short: dict, long: dict) -> dict[str, float]:
    sm, lm = short["metrics"], long["metrics"]
    within = lm["sliding_within_transition_l1"]
    return {
        "short_rgba_l1": sm["rgba_l1"],
        "short_rgb_support_l1": sm["rgb_support_l1"],
        "short_rgb_edge_l1": sm["rgb_edge_l1"],
        "short_alpha_iou": sm["alpha_iou_0.5"],
        "short_white_psnr": sm["white_composite_psnr"],
        "long_sliding_rgba_l1": lm["sliding_rgba_l1"],
        "long_sliding_white_psnr": lm["sliding_white_psnr"],
        "long_sliding_seam_l1": lm["sliding_boundary_transition_l1"],
        "long_sliding_within_l1": within,
        "long_sliding_seam_to_within_ratio": lm["sliding_boundary_transition_l1"] / max(within, 1e-12),
        "causal_prefix_max_abs": lm["causal_prefix_max_abs"],
        "sliding_seconds_per_clip": long["timing"]["sliding_seconds"],
    }


def compare(protocol: dict, baseline_short: dict, baseline_long: dict,
            candidate_short: dict, candidate_long: dict) -> dict:
    baseline = summarize(baseline_short, baseline_long)
    candidate = summarize(candidate_short, candidate_long)
    gates = protocol["candidate_hard_gates"]
    checks = {
        "short_rgba_l1": candidate["short_rgba_l1"] <= gates["short_rgba_l1_max"],
        "short_rgb_edge_l1": candidate["short_rgb_edge_l1"] <= gates["short_rgb_edge_l1_max"],
        "short_alpha_iou": candidate["short_alpha_iou"] >= gates["short_alpha_iou_min"],
        "long_sliding_rgba_l1": candidate["long_sliding_rgba_l1"] <= gates["long_sliding_rgba_l1_max"],
        "seam_ratio": candidate["long_sliding_seam_to_within_ratio"] <= gates["long_sliding_seam_to_within_ratio_max"],
        "causal_prefix": candidate["causal_prefix_max_abs"] <= gates["causal_prefix_max_abs_max"],
        "runtime_ratio": candidate["sliding_seconds_per_clip"] / max(
            baseline["sliding_seconds_per_clip"], 1e-12
        ) <= gates["sliding_runtime_ratio_vs_baseline_max"],
    }
    return {
        "protocol_version": protocol["version"],
        "baseline": baseline,
        "candidate": candidate,
        "candidate_vs_baseline": {
            key: candidate[key] / max(value, 1e-12) for key, value in baseline.items()
        },
        "gate_checks": checks,
        "quantitative_gate_pass": all(checks.values()),
        "visual_audit_required": True,
        "selection": "pending_visual_audit" if all(checks.values()) else protocol["baseline"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--baseline-short", required=True)
    parser.add_argument("--baseline-long", required=True)
    parser.add_argument("--candidate-short", required=True)
    parser.add_argument("--candidate-long", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = compare(
        _load(args.protocol), _load(args.baseline_short), _load(args.baseline_long),
        _load(args.candidate_short), _load(args.candidate_long),
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
