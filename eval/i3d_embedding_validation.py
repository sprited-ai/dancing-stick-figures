"""Validate whether the released I3D embedding separates useful video pairs.

This is deliberately separate from FVD.  FVD compares two feature
distributions; this module tests the per-video feature geometry that FVD is
built from.  It asks whether clean clips generated from the same prompt are
closer than clips from different prompts, and how far deterministic temporal
corruptions move a clip from its clean source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


SEED_SUFFIX = re.compile(r"_s\d+$")


def prompt_id(row: dict) -> str:
    """Recover the prompt identity from a released motion id."""
    return SEED_SUFFIX.sub("", row["motion_id"])


def _pair_distances(features: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    delta = features[pairs[:, 0]] - features[pairs[:, 1]]
    return np.linalg.norm(delta, axis=1)


def _summary(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ValueError("cannot summarise an empty distance set")
    medians = np.median(
        values[rng.integers(0, len(values), size=(draws, len(values)))], axis=1
    )
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "ci95_median": [float(x) for x in np.quantile(medians, [0.025, 0.975])],
    }


def _clean_pairs(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    same, different = [], []
    prompts = [prompt_id(row) for row in rows]
    groups = [row["motion_id"].split("/", 1)[0] for row in rows]
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if prompts[left] == prompts[right]:
                same.append((left, right))
            elif groups[left] == groups[right]:
                # Same-group negatives are harder and avoid making the result
                # depend only on broad categories such as idle vs sport.
                different.append((left, right))
    if not same or not different:
        raise ValueError("manifest must contain repeated prompts and same-group negatives")
    return np.asarray(same, dtype=np.int64), np.asarray(different, dtype=np.int64)


def validate_embeddings(
    feature_sets: dict[str, np.ndarray],
    manifest: dict,
    *,
    seed: int = 0,
    bootstrap_draws: int = 2000,
) -> dict:
    clean_a = np.asarray(feature_sets["real_reference_a"], dtype=np.float64)
    clean_b = np.asarray(feature_sets["real_reference_b"], dtype=np.float64)
    rows = list(manifest["reference_a"]) + list(manifest["reference_b"])
    clean = np.concatenate([clean_a, clean_b], axis=0)
    if len(rows) != len(clean):
        raise ValueError("manifest rows and clean feature rows differ")

    rng = np.random.default_rng(seed)
    same_pairs, different_pool = _clean_pairs(rows)
    # Balance the comparison so the probability below is not dominated by the
    # much larger number of available negative pairs.
    take = rng.choice(len(different_pool), size=len(same_pairs), replace=False)
    different_pairs = different_pool[take]
    same = _pair_distances(clean, same_pairs)
    different = _pair_distances(clean, different_pairs)

    # Probability that a randomly paired same-prompt clip is closer than a
    # randomly paired same-group, different-prompt clip.
    separation_probability = float(np.mean(same[:, None] < different[None, :]))

    distance_matrix = np.linalg.norm(clean[:, None, :] - clean[None, :, :], axis=2)
    np.fill_diagonal(distance_matrix, np.inf)
    nearest = np.argmin(distance_matrix, axis=1)
    prompts = np.asarray([prompt_id(row) for row in rows])
    top1_prompt_accuracy = float(np.mean(prompts == prompts[nearest]))
    _, prompt_counts = np.unique(prompts, return_counts=True)
    random_top1_prompt_accuracy = float(
        np.sum(prompt_counts * (prompt_counts - 1)) / (len(prompts) * (len(prompts) - 1))
    )

    corruption = {}
    for name, candidate in feature_sets.items():
        if name in {"real_reference_a", "real_reference_b", "train_replay"}:
            continue
        candidate = np.asarray(candidate, dtype=np.float64)
        if candidate.shape != clean_b.shape:
            raise ValueError(f"{name} shape {candidate.shape} != clean B {clean_b.shape}")
        paired = np.linalg.norm(clean_b - candidate, axis=1)
        corruption[name] = {
            **_summary(paired, rng, bootstrap_draws),
            "fraction_above_clean_same_prompt_median": float(np.mean(paired > np.median(same))),
        }

    return {
        "protocol_version": 1,
        "question": "Does the I3D feature geometry separate prompt identity and controlled temporal corruption?",
        "statistical_unit": manifest.get("statistical_unit", "source_motion"),
        "clean_videos": int(len(clean)),
        "feature_dim": int(clean.shape[1]),
        "negative_pair_policy": "different prompt, same broad motion group",
        "seed": seed,
        "bootstrap_draws": bootstrap_draws,
        "clean_pair_distance": {
            "same_prompt_different_seed": _summary(same, rng, bootstrap_draws),
            "different_prompt_same_group": _summary(different, rng, bootstrap_draws),
            "probability_same_prompt_is_closer": separation_probability,
            "nearest_neighbour_prompt_accuracy": top1_prompt_accuracy,
            "random_neighbour_prompt_accuracy": random_top1_prompt_accuracy,
        },
        "paired_clean_to_corruption_distance": corruption,
        "interpretation_guardrail": (
            "These are per-video I3D embedding distances, not prompt-level FVD. "
            "FVD remains a distribution-level comparison and requires substantially larger sets."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="NPZ written by eval.fvd_bundle")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap_draws", type=int, default=2000)
    args = parser.parse_args()

    bundle = np.load(args.features)
    feature_sets = {name: bundle[name] for name in bundle.files}
    manifest = json.loads(Path(args.manifest).read_text())
    result = validate_embeddings(
        feature_sets,
        manifest,
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
    )
    result["inputs"] = {
        "features": str(Path(args.features)),
        "features_sha256": hashlib.sha256(Path(args.features).read_bytes()).hexdigest(),
        "manifest": str(Path(args.manifest)),
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
