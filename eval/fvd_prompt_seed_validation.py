"""Test FVD stability across seeds and sensitivity to prompt composition.

FVD is distribution-level: this protocol never computes it from one clip pair.
It embeds every c0 clip in the final 120-frame mini release, then compares
(1) complementary seed halves over the same prompt roster and (2) disjoint,
group-balanced prompt rosters.  Repeated deterministic partitions expose how
much the conclusion depends on the particular split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.fvd import features, fvd_from_features


SEED_CAMERA = re.compile(r"_s(\d+)/c(\d+)$")


def clip_inventory(clips: dict, camera: int = 0, frames: int = 120) -> list[dict]:
    rows = []
    by_prompt: dict[str, set[int]] = defaultdict(set)
    for clip_id, row in clips.items():
        match = SEED_CAMERA.search(clip_id)
        if not match or int(match.group(2)) != camera:
            continue
        seed = int(match.group(1))
        rows.append({
            "clip_id": clip_id,
            "start": int(row["start"]),
            "frames": int(row["n"]),
            "prompt": row["text"],
            "group": row["group"],
            "seed": seed,
        })
        by_prompt[row["text"]].add(seed)
    expected = set(range(10))
    incomplete = {prompt: sorted(seeds) for prompt, seeds in by_prompt.items() if seeds != expected}
    if incomplete:
        raise ValueError(f"prompts without seeds 0..9: {incomplete}")
    if any(row["frames"] < frames for row in rows):
        raise ValueError(f"protocol requires clips with at least {frames} frames")
    return sorted(rows, key=lambda row: (row["group"], row["prompt"], row["seed"]))


def embed_inventory(cache: Path, rows: list[dict], frames: int, device: str, batch: int) -> np.ndarray:
    frame_store = np.load(cache / "frames.npy", mmap_mode="r")
    output = []
    for start in range(0, len(rows), batch):
        batch_rows = rows[start:start + batch]
        rgba = np.stack([
            np.asarray(frame_store[row["start"]:row["start"] + frames]) for row in batch_rows
        ])
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        rgb = np.clip(rgba[..., :3] * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
        output.append(features(rgb, device=device, bs=batch))
        print(f"embedded {min(start + batch, len(rows))}/{len(rows)} clips", flush=True)
    return np.concatenate(output, axis=0)


def grouped_prompt_halves(rows: list[dict], rng: np.random.Generator) -> tuple[set[str], set[str], list[str]]:
    prompts_by_group: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["prompt"] not in prompts_by_group[row["group"]]:
            prompts_by_group[row["group"]].append(row["prompt"])
    left, right, dropped = set(), set(), []
    for group in sorted(prompts_by_group):
        prompts = np.asarray(sorted(prompts_by_group[group]), dtype=object)
        rng.shuffle(prompts)
        half = len(prompts) // 2
        left.update(map(str, prompts[:half]))
        right.update(map(str, prompts[half:2 * half]))
        dropped.extend(map(str, prompts[2 * half:]))
    return left, right, dropped


def indices_for(rows: list[dict], *, prompts: set[str] | None = None,
                seeds: set[int] | None = None) -> np.ndarray:
    return np.asarray([
        index for index, row in enumerate(rows)
        if (prompts is None or row["prompt"] in prompts)
        and (seeds is None or row["seed"] in seeds)
    ], dtype=np.int64)


def run_partitions(rows: list[dict], embedded: np.ndarray, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    same_values, different_values, random_values = [], [], []
    partitions = []
    all_seeds = np.arange(10)
    all_indices = np.arange(len(rows))
    for repeat in range(repeats):
        shuffled_seeds = rng.permutation(all_seeds)
        seeds_a, seeds_b = set(map(int, shuffled_seeds[:5])), set(map(int, shuffled_seeds[5:]))
        same_a = indices_for(rows, seeds=seeds_a)
        same_b = indices_for(rows, seeds=seeds_b)

        prompts_a, prompts_b, dropped = grouped_prompt_halves(rows, rng)
        different_a = indices_for(rows, prompts=prompts_a)
        different_b = indices_for(rows, prompts=prompts_b)
        n = min(len(same_a), len(same_b), len(different_a), len(different_b))
        same_a = rng.choice(same_a, n, replace=False)
        same_b = rng.choice(same_b, n, replace=False)
        different_a = rng.choice(different_a, n, replace=False)
        different_b = rng.choice(different_b, n, replace=False)
        random_split = rng.permutation(all_indices)[:2 * n]
        random_a, random_b = random_split[:n], random_split[n:]

        same_fvd = fvd_from_features(embedded[same_a], embedded[same_b])
        different_fvd = fvd_from_features(embedded[different_a], embedded[different_b])
        random_fvd = fvd_from_features(embedded[random_a], embedded[random_b])
        same_values.append(same_fvd)
        different_values.append(different_fvd)
        random_values.append(random_fvd)
        partitions.append({
            "repeat": repeat,
            "n_per_set": int(n),
            "same_prompt_seed_halves": [sorted(seeds_a), sorted(seeds_b)],
            "prompt_halves": [sorted(prompts_a), sorted(prompts_b)],
            "dropped_prompts": sorted(dropped),
            "fvd": {
                "same_prompt_roster_different_seeds": float(same_fvd),
                "different_prompt_rosters_group_balanced": float(different_fvd),
                "random_video_halves": float(random_fvd),
            },
        })
        print(f"partition {repeat + 1}/{repeats}: same {same_fvd:.2f}, "
              f"different {different_fvd:.2f}, random {random_fvd:.2f}", flush=True)

    def summary(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "range": [float(values.min()), float(values.max())],
            "values": values.tolist(),
        }

    same = np.asarray(same_values)
    different = np.asarray(different_values)
    return {
        "same_prompt_roster_different_seeds": summary(same_values),
        "different_prompt_rosters_group_balanced": summary(different_values),
        "random_video_halves": summary(random_values),
        "fraction_same_below_different": float(np.mean(same < different)),
        "median_ratio_same_over_different": float(np.median(same) / np.median(different)),
        "partitions": partitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--features_out", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    clips_path = args.cache / "clips.json"
    clips = json.loads(clips_path.read_text())
    rows = clip_inventory(clips, camera=0, frames=args.frames)
    embedded = embed_inventory(args.cache, rows, args.frames, args.device, args.batch)
    if args.features_out:
        args.features_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.features_out, features=embedded,
                            clip_ids=np.asarray([row["clip_id"] for row in rows]))
    result = {
        "protocol_version": 1,
        "question": "Is FVD lower across different seeds of the same prompt roster than across different prompt rosters?",
        "metric": "standard Kinetics-400 I3D-logit FVD, 224-pixel white composite",
        "camera": "c0 only",
        "frames": args.frames,
        "feature_dim": int(embedded.shape[1]),
        "clips": len(rows),
        "prompts": len({row["prompt"] for row in rows}),
        "groups": sorted({row["group"] for row in rows}),
        "clips_sha256": hashlib.sha256(clips_path.read_bytes()).hexdigest(),
        "interpretation_guardrail": (
            "This tests distributional seed stability and prompt-composition sensitivity. "
            "It does not turn FVD into a per-video or prompt-correctness score."
        ),
        **run_partitions(rows, embedded, args.repeats, args.seed),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "clips", "prompts", "same_prompt_roster_different_seeds",
        "different_prompt_rosters_group_balanced", "fraction_same_below_different",
        "median_ratio_same_over_different")}, indent=2))


if __name__ == "__main__":
    main()
