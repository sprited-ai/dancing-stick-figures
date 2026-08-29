"""Validate colour-derived oracle masks against released segmentation labels.

The final dataset stores both 64-pixel straight RGBA frames and 27-part
segmentation PNGs.  This script selects one deterministic frame per rendered
clip, maps the 27 part ids into the evaluator's nine colour classes, and
reports pixel classification and threshold-sensitivity statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from eval.oracle import COLS, NAMES as ORACLE_NAMES
from generator.render import INK, PALETTE
from generator.skeleton import NAMES as SEGMENT_NAMES


def segment_class_lookup() -> np.ndarray:
    """Return segment-id -> oracle-class index, with background mapped to -1."""
    palette_to_class = {tuple(map(int, colour)): index for index, colour in enumerate(COLS)}
    lookup = np.full(len(SEGMENT_NAMES) + 1, -1, np.int16)
    for segment_id, name in enumerate(SEGMENT_NAMES, start=1):
        colour = tuple(PALETTE.get(name, INK))
        lookup[segment_id] = palette_to_class[colour]
    return lookup


SEGMENT_CLASS = segment_class_lookup()


def classify_rgba(rgba: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply the evaluator's alpha and nearest-palette rules at a chosen RGB radius."""
    foreground = rgba[..., 3] > 127
    rgb = rgba[..., :3].astype(np.float32)
    distance = np.linalg.norm(rgb[..., None, :] - COLS[None, None, :, :], axis=-1)
    labels = distance.argmin(axis=-1).astype(np.int16)
    labels = np.where(foreground & (distance.min(axis=-1) < tau), labels, -1)
    return labels, foreground


def segmentation_classes(segmentation: np.ndarray) -> np.ndarray:
    segmentation = np.asarray(segmentation, dtype=np.int64)
    if segmentation.min(initial=0) < 0 or segmentation.max(initial=0) >= len(SEGMENT_CLASS):
        raise ValueError("segmentation id outside the released 0..27 range")
    return SEGMENT_CLASS[segmentation]


def selected_frame(clip_id: str, n_frames: int = 120) -> int:
    digest = hashlib.sha256(clip_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_frames


def iter_selected_rows(data: Path, limit_clips: int = 0):
    """Yield one hash-selected frame per clip without loading all image bytes at once."""
    seen: set[str] = set()
    yielded = 0
    for path in sorted(data.glob("*.parquet")):
        metadata = pq.read_table(path, columns=["clip_id", "frame_idx"]).to_pydict()
        indices = [
            index for index, (clip_id, frame_idx) in enumerate(zip(metadata["clip_id"], metadata["frame_idx"]))
            if clip_id not in seen and int(frame_idx) == selected_frame(clip_id)
        ]
        if not indices:
            continue
        payload = pq.read_table(path, columns=["clip_id", "frame_idx", "color", "seg"]).take(indices).to_pylist()
        for row in payload:
            clip_id = row["clip_id"]
            if clip_id in seen:
                continue
            seen.add(clip_id)
            rgba = np.asarray(Image.open(io.BytesIO(row["color"]["bytes"])).convert("RGBA"))
            seg = np.asarray(Image.open(io.BytesIO(row["seg"]["bytes"])).convert("L"))
            yield clip_id, int(row["frame_idx"]), rgba, seg
            yielded += 1
            if limit_clips and yielded >= limit_clips:
                return


def empty_counts() -> dict[str, np.ndarray | int]:
    classes = len(ORACLE_NAMES)
    return {
        "confusion": np.zeros((classes, classes), np.int64),
        "unassigned_by_truth": np.zeros(classes, np.int64),
        "foreground": 0,
        "segmentation_foreground": 0,
        "segmentation_excluded_by_alpha": 0,
    }


def update_counts(counts: dict, rgba: np.ndarray, segmentation: np.ndarray, tau: float) -> None:
    predicted, foreground = classify_rgba(rgba, tau)
    truth = segmentation_classes(segmentation)
    valid_truth = truth >= 0
    counts["segmentation_foreground"] += int(valid_truth.sum())
    counts["segmentation_excluded_by_alpha"] += int((valid_truth & ~foreground).sum())
    evaluated = foreground & valid_truth
    counts["foreground"] += int(evaluated.sum())
    true_values = truth[evaluated]
    predicted_values = predicted[evaluated]
    assigned = predicted_values >= 0
    np.add.at(counts["confusion"], (true_values[assigned], predicted_values[assigned]), 1)
    np.add.at(counts["unassigned_by_truth"], true_values[~assigned], 1)


def summarise(counts: dict) -> dict:
    confusion = counts["confusion"]
    unassigned = counts["unassigned_by_truth"]
    support = confusion.sum(axis=1) + unassigned
    predicted = confusion.sum(axis=0)
    tp = np.diag(confusion)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support > 0)
    union = support + predicted - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp, dtype=float), where=union > 0)
    foreground = int(counts["foreground"])
    assigned = int(confusion.sum())
    correct = int(tp.sum())
    present = support > 0
    per_class = {
        name: {
            "support_pixels": int(support[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "iou": float(iou[index]),
            "unassigned_pixels": int(unassigned[index]),
        }
        for index, name in enumerate(ORACLE_NAMES)
    }
    return {
        "evaluated_foreground_pixels": foreground,
        "assigned_fraction": float(assigned / foreground) if foreground else 0.0,
        "unassigned_fraction": float(1.0 - assigned / foreground) if foreground else 1.0,
        "pixel_accuracy": float(correct / foreground) if foreground else 0.0,
        "accuracy_given_assigned": float(correct / assigned) if assigned else 0.0,
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "macro_iou": float(iou[present].mean()) if present.any() else 0.0,
        "segmentation_foreground_pixels": int(counts["segmentation_foreground"]),
        "segmentation_pixels_excluded_by_alpha_fraction": float(
            counts["segmentation_excluded_by_alpha"] / counts["segmentation_foreground"]
        ) if counts["segmentation_foreground"] else 0.0,
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }


def validate(data: Path, thresholds: list[float], limit_clips: int = 0) -> dict:
    accumulators = {threshold: empty_counts() for threshold in thresholds}
    selected = []
    for clip_id, frame_idx, rgba, segmentation in iter_selected_rows(data, limit_clips=limit_clips):
        selected.append((clip_id, frame_idx))
        for threshold, counts in accumulators.items():
            update_counts(counts, rgba, segmentation, threshold)
    if not selected:
        raise ValueError(f"no selected frames found under {data}")
    return {
        "data": str(data),
        "selection": "one frame per rendered clip; frame_idx = first-8-byte SHA256(clip_id) mod 120",
        "n_frames": len(selected),
        "n_unique_clips": len({clip for clip, _ in selected}),
        "thresholds": {
            str(int(t) if float(t).is_integer() else t): summarise(accumulators[t]) for t in thresholds
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thresholds", default="20,30,40,50,60,70,80,100")
    parser.add_argument("--limit-clips", type=int, default=0)
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    result = validate(args.data, thresholds, limit_clips=args.limit_clips)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "n_frames": result["n_frames"],
        "threshold_60": result["thresholds"].get("60"),
    }, indent=2))


if __name__ == "__main__":
    main()
