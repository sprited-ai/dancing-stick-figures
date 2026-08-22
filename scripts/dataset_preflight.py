#!/usr/bin/env python3
"""Fingerprint and visually preflight an exact training/evaluation cache.

This gate deliberately hashes frame bytes, not just shape or metadata. The
excluded gray-mini cache and canonical colored K1 cache are both valid 64px
RGBA arrays, so shape-only checks cannot prevent a renderer mix-up.

Example:
    python scripts/dataset_preflight.py \
      --cache /data/dancing-stick-figure-paper/cache/mini \
      --profile colored_k1_v1 --out preflight.json --grid reference_grid.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = ROOT / "configs" / "dataset_fingerprints.json"
REQUIRED_FILES = ("frames.npy", "clips.json", "meta.json")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_cache(cache: Path | str) -> dict[str, Any]:
    cache = Path(cache)
    missing = [name for name in REQUIRED_FILES if not (cache / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cache is missing required files: {', '.join(missing)}")
    frames = np.load(cache / "frames.npy", mmap_mode="r")
    clips_value = json.loads((cache / "clips.json").read_text())
    rows = list(clips_value.values()) if isinstance(clips_value, dict) else list(clips_value)
    splits = dict(sorted(Counter(str(row.get("split", "")) for row in rows).items()))
    return {
        "files": {
            name: {"sha256": sha256_file(cache / name), "size": (cache / name).stat().st_size}
            for name in REQUIRED_FILES
        },
        "array": {"shape": list(frames.shape), "dtype": str(frames.dtype)},
        "clips": {"count": len(rows), "splits": splits},
        "meta": json.loads((cache / "meta.json").read_text()),
    }


def _compare(actual: Any, expected: Any, path: str, errors: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            errors.append(f"{path}: expected mapping, got {type(actual).__name__}")
            return
        for key, value in expected.items():
            child = f"{path}.{key}" if path else key
            if key not in actual:
                errors.append(f"{child}: missing")
            else:
                _compare(actual[key], value, child, errors)
    elif actual != expected:
        errors.append(f"{path}: expected {expected!r}, got {actual!r}")


def validate_fingerprint(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    errors: list[str] = []
    _compare(actual, expected, "", errors)
    if errors:
        raise ValueError("dataset fingerprint mismatch:\n- " + "\n- ".join(errors))


def render_reference_grid(cache: Path | str, destination: Path | str, count: int = 64) -> None:
    cache, destination = Path(cache), Path(destination)
    frames = np.load(cache / "frames.npy", mmap_mode="r")
    clips_value = json.loads((cache / "clips.json").read_text())
    rows = list(clips_value.values()) if isinstance(clips_value, dict) else list(clips_value)
    if not rows:
        raise ValueError("cannot render a reference grid from an empty clip manifest")
    chosen = np.linspace(0, len(rows) - 1, min(count, len(rows)), dtype=int)
    cells = []
    for index in chosen:
        row = rows[int(index)]
        frame_index = int(row["start"]) + max(0, int(row["n"]) // 2)
        rgba = np.asarray(frames[frame_index]).astype(np.float32) / 255.0
        rgb = rgba[..., :3] * rgba[..., 3:4] + 1.0 - rgba[..., 3:4]
        cells.append(Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)))
    side = int(np.ceil(np.sqrt(len(cells))))
    size = cells[0].width
    label_height = 22
    canvas = Image.new("RGB", (side * size, label_height + side * size), "white")
    ImageDraw.Draw(canvas).text((4, 4), f"deterministic cache preflight: {len(cells)} clips", fill="black")
    for i, cell in enumerate(cells):
        canvas.paste(cell, ((i % side) * size, label_height + (i // side) * size))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILES))
    parser.add_argument("--out", required=True, help="JSON record, written on pass or failure")
    parser.add_argument("--grid", default="", help="optional deterministic real-data PNG grid")
    parser.add_argument("--grid-count", type=int, default=64)
    args = parser.parse_args()

    profiles = json.loads(Path(args.profiles).read_text())
    if args.profile not in profiles:
        raise SystemExit(f"unknown dataset profile {args.profile!r}")
    expected = profiles[args.profile]
    expected_identity = {key: value for key, value in expected.items() if key != "description"}
    actual = fingerprint_cache(args.cache)
    record = {
        "protocol": "dataset_preflight_v1",
        "cache": str(Path(args.cache).resolve()),
        "profile": args.profile,
        "expected": expected,
        "actual": actual,
        "valid": False,
        "errors": [],
    }
    try:
        validate_fingerprint(actual, expected_identity)
    except ValueError as error:
        record["errors"] = str(error).splitlines()[1:]
        _atomic_json(Path(args.out), record)
        raise SystemExit(str(error))
    record["valid"] = True
    _atomic_json(Path(args.out), record)
    if args.grid:
        render_reference_grid(args.cache, args.grid, args.grid_count)
    print(json.dumps({"profile": args.profile, "valid": True, "out": args.out, "grid": args.grid}))


if __name__ == "__main__":
    main()
