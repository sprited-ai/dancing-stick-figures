"""Aligned data utilities for the S0 text-to-motion baseline.

The video task uses 50 frames at 10 fps.  This module turns the released ARDY
motion representation into the same deterministic time grid without depending
on the pixel-video loader:

* ``joints``: hips-centred cskel27 positions in the frame-0 figure basis [50,27,3]
* ``root``: hips displacement from frame 0 in that basis [50,3]
* ``heading``: root yaw as (cos, sin), relative to frame 0 [50,2]
* ``contacts``: the four released foot-contact flags [50,4], when available

The local pose and root trajectory are deliberately separate.  Together they
retain the information needed by the exact renderer while making bone and
contact losses straightforward.  Normalisation statistics are fit on the
training split only and saved in the cache.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np

from generator.ardy_adapter import _frame0_basis
from generator.skeleton import IDX, NAMES


TARGET_FRAMES = 50
TARGET_FPS = 10.0
CONTACT_DIM = 4


@dataclass(frozen=True)
class MotionClip:
    clip_id: str
    split: str
    text: str
    joints: np.ndarray
    root: np.ndarray
    heading: np.ndarray
    contacts: np.ndarray | None = None
    group: str = ""
    seed: int = 0
    source_fps: float = 20.0


def _linear_resample(x: np.ndarray, source_fps: float, target_times: np.ndarray) -> np.ndarray:
    """Linear interpolation along time, preserving all trailing dimensions."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim < 1 or len(x) < 2:
        raise ValueError("a motion array needs at least two source frames")
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"invalid source fps: {source_fps}")
    source_times = np.arange(len(x), dtype=np.float64) / float(source_fps)
    if target_times[-1] > source_times[-1] + 1e-8:
        raise ValueError(
            f"clip is too short: {source_times[-1]:.3f}s available, "
            f"{target_times[-1]:.3f}s required"
        )
    flat = x.reshape(len(x), -1)
    out = np.empty((len(target_times), flat.shape[1]), dtype=np.float32)
    for column in range(flat.shape[1]):
        out[:, column] = np.interp(target_times, source_times, flat[:, column])
    return out.reshape((len(target_times),) + x.shape[1:])


def _nearest_resample(x: np.ndarray, source_fps: float, target_times: np.ndarray) -> np.ndarray:
    """Nearest-frame resampling for discrete labels such as foot contacts."""
    indices = np.floor(target_times * float(source_fps) + 0.5).astype(np.int64)
    return np.asarray(x)[np.clip(indices, 0, len(x) - 1)]


def canonical_components(world_joints: np.ndarray, basis: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split world joints into local pose, root trajectory and relative yaw.

    ``basis`` rows are x/left, y/up, z/forward axes in world coordinates.  If
    omitted, it is recovered from frame 0 using the renderer's canonical rule.
    """
    p = np.asarray(world_joints, dtype=np.float32)
    if p.ndim != 3 or p.shape[1:] != (len(NAMES), 3):
        raise ValueError(f"expected [T,{len(NAMES)},3] posed_joints, got {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("posed_joints contains NaN or infinity")
    r = np.asarray(_frame0_basis(p.astype(np.float64))[0] if basis is None else basis, dtype=np.float32)
    if r.shape != (3, 3):
        raise ValueError(f"expected frame0_basis [3,3], got {r.shape}")

    hips = p[:, IDX["Hips"]]
    local = (p - hips[:, None, :]) @ r.T
    root = (hips - hips[0]) @ r.T

    left = p[:, IDX["LeftUpLeg"]] - p[:, IDX["RightUpLeg"]]
    left_local = left @ r.T
    norm = np.linalg.norm(left_local[:, (0, 2)], axis=1, keepdims=True)
    left_xz = left_local[:, (0, 2)] / np.maximum(norm, 1e-8)
    yaw = np.arctan2(left_xz[:, 1], left_xz[:, 0])
    heading = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)
    return local.astype(np.float32), root.astype(np.float32), heading.astype(np.float32)


def align_clip(clip: MotionClip, frames: int = TARGET_FRAMES, fps: float = TARGET_FPS, start_s: float = 0.0) -> MotionClip:
    """Resample one canonical clip onto a fixed, video-aligned time grid."""
    if frames <= 0 or fps <= 0 or start_s < 0:
        raise ValueError("frames/fps must be positive and start_s non-negative")
    times = start_s + np.arange(frames, dtype=np.float64) / float(fps)
    joints = _linear_resample(clip.joints, clip.source_fps, times)
    root = _linear_resample(clip.root, clip.source_fps, times)
    heading = _linear_resample(clip.heading, clip.source_fps, times)
    heading /= np.maximum(np.linalg.norm(heading, axis=1, keepdims=True), 1e-8)
    contacts = None
    if clip.contacts is not None:
        contacts = _nearest_resample(np.asarray(clip.contacts, dtype=np.bool_), clip.source_fps, times).astype(np.bool_)
    return MotionClip(clip.clip_id, clip.split, clip.text, joints, root, heading, contacts,
                      clip.group, clip.seed, float(fps))


def load_ardy_npz(path: str | os.PathLike, *, split: str, clip_id: str | None = None,
                  group: str = "", seed: int = 0) -> MotionClip:
    """Read one raw ARDY NPZ.  Split is explicit to prevent accidental leakage."""
    with np.load(path, allow_pickle=False) as d:
        world = np.asarray(d["posed_joints"], dtype=np.float32)
        source_fps = float(d["fps"]) if "fps" in d else 20.0
        text = str(d["text"]) if "text" in d else ""
        contacts = np.asarray(d["foot_contacts"], dtype=np.bool_) if "foot_contacts" in d else None
    joints, root, heading = canonical_components(world)
    cid = clip_id or Path(path).stem
    return MotionClip(cid, split, text, joints, root, heading, contacts, group, seed, source_fps)


def iter_motion_parquet(paths: Iterable[str | os.PathLike]) -> Iterator[MotionClip]:
    """Stream the released one-row-per-motion parquet format."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency message is the useful behaviour
        raise RuntimeError("pyarrow is required to read released motion parquet shards") from exc
    columns = ["clip_id", "split", "text", "group", "seed", "fps", "n_frames",
               "frame0_basis", "posed_joints", "foot_contacts"]
    for path in paths:
        table = pq.read_table(path, columns=columns)
        for row in table.to_pylist():
            t = int(row["n_frames"])
            world = np.frombuffer(row["posed_joints"], dtype="<f4").reshape(t, len(NAMES), 3)
            basis = np.frombuffer(row["frame0_basis"], dtype="<f4").reshape(3, 3)
            contacts = None if row["foot_contacts"] is None else np.frombuffer(row["foot_contacts"], dtype=np.bool_).reshape(t, CONTACT_DIM)
            joints, root, heading = canonical_components(world, basis)
            yield MotionClip(row["clip_id"], row["split"], row["text"], joints, root, heading,
                             contacts, row["group"], int(row["seed"]), float(row["fps"]))


def frame_metadata_index(paths: Iterable[str | os.PathLike]) -> dict[str, dict[str, object]]:
    """Recover leak-free motion metadata from released frame shards.

    Frame clip ids end in ``/c0``..``/c2``; stripping that suffix joins them
    to raw ARDY NPZ paths. Only metadata columns are read.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to index released frame shards") from exc
    result: dict[str, dict[str, object]] = {}
    for path in paths:
        table = pq.read_table(path, columns=["clip_id", "split", "text", "group", "seed"])
        for row in table.to_pylist():
            base = row["clip_id"].rsplit("/c", 1)[0]
            metadata = {name: row[name] for name in ("split", "text", "group", "seed")}
            previous = result.setdefault(base, metadata)
            if previous != metadata:
                raise ValueError(f"inconsistent camera metadata for {base}")
    return result


def iter_indexed_ardy_npz(npz_root: str | os.PathLike,
                          metadata: Mapping[str, Mapping[str, object]]) -> Iterator[MotionClip]:
    """Read a raw NPZ tree using an explicit released-frame split index."""
    root = Path(npz_root)
    for path in sorted(root.glob("*/*.npz")):
        clip_id = path.relative_to(root).with_suffix("").as_posix()
        if clip_id not in metadata:
            raise KeyError(f"raw motion {clip_id!r} is absent from the frame split index")
        row = metadata[clip_id]
        clip = load_ardy_npz(path, split=str(row["split"]), clip_id=clip_id,
                             group=str(row["group"]), seed=int(row["seed"]))
        if clip.text and clip.text != str(row["text"]):
            raise ValueError(f"prompt mismatch for {clip_id}")
        yield clip


def _fit_stats(clips: list[MotionClip]) -> dict[str, np.ndarray]:
    train = [c for c in clips if c.split == "train"]
    if not train:
        raise ValueError("normalisation requires at least one training clip")
    stats: dict[str, np.ndarray] = {}
    for name in ("joints", "root"):
        x = np.concatenate([getattr(c, name) for c in train], axis=0).astype(np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        stats[f"{name}_mean"] = mean.astype(np.float32)
        # A 0.1 mm floor avoids magnifying float32 round-off in effectively
        # constant coordinates while remaining negligible at human scale.
        stats[f"{name}_std"] = np.maximum(std, 1e-4).astype(np.float32)
    return stats


def write_cache(clips: Iterable[MotionClip], out_dir: str | os.PathLike,
                *, frames: int = TARGET_FRAMES, fps: float = TARGET_FPS) -> None:
    """Write aligned arrays and train-only normalisation statistics."""
    aligned = [align_clip(c, frames, fps) for c in clips]
    if not aligned:
        raise ValueError("cannot build an empty motion cache")
    aligned.sort(key=lambda c: c.clip_id)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "joints.npy", np.stack([c.joints for c in aligned]).astype(np.float32))
    np.save(out / "root.npy", np.stack([c.root for c in aligned]).astype(np.float32))
    np.save(out / "heading.npy", np.stack([c.heading for c in aligned]).astype(np.float32))
    available = np.asarray([c.contacts is not None for c in aligned], dtype=np.bool_)
    contacts = np.stack([c.contacts if c.contacts is not None else np.zeros((frames, CONTACT_DIM), dtype=np.bool_) for c in aligned])
    np.save(out / "contacts.npy", contacts)
    np.save(out / "contacts_available.npy", available)
    np.savez(out / "stats.npz", **_fit_stats(aligned))
    records = [dict(clip_id=c.clip_id, split=c.split, text=c.text, group=c.group, seed=c.seed) for c in aligned]
    (out / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(dict(version=1, frames=frames, fps=fps, joints=NAMES,
        representation="hips-centred local joints + separate root displacement + relative heading",
        normalization="mean/std fitted on split=train only"), indent=2), encoding="utf-8")


class MotionCache:
    """Small NumPy dataset view over a cache produced by :func:`write_cache`."""
    def __init__(self, path: str | os.PathLike, split: str, normalize: bool = True):
        path = Path(path)
        records = json.loads((path / "records.json").read_text(encoding="utf-8"))
        self.indices = [i for i, record in enumerate(records) if record["split"] == split]
        self.records = records
        self.joints = np.load(path / "joints.npy", mmap_mode="r")
        self.root = np.load(path / "root.npy", mmap_mode="r")
        self.heading = np.load(path / "heading.npy", mmap_mode="r")
        self.contacts = np.load(path / "contacts.npy", mmap_mode="r")
        self.contacts_available = np.load(path / "contacts_available.npy", mmap_mode="r")
        self.normalize = normalize
        with np.load(path / "stats.npz") as stats:
            self.stats = {name: stats[name].copy() for name in stats.files}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Mapping[str, object]:
        i = self.indices[item]
        joints = np.asarray(self.joints[i], dtype=np.float32).copy()
        root = np.asarray(self.root[i], dtype=np.float32).copy()
        if self.normalize:
            joints = (joints - self.stats["joints_mean"]) / self.stats["joints_std"]
            root = (root - self.stats["root_mean"]) / self.stats["root_std"]
        return dict(joints=joints, root=root, heading=np.asarray(self.heading[i]).copy(),
                    contacts=np.asarray(self.contacts[i]).copy(),
                    contacts_available=bool(self.contacts_available[i]), metadata=self.records[i])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--motion-parquet", help="glob for released motion parquet shards")
    source.add_argument("--npz-root", help="raw ARDY NPZ tree (requires --frame-parquet)")
    parser.add_argument("--frame-parquet", help="frame-shard glob used as leak-free split index for --npz-root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.motion_parquet:
        paths = sorted(glob.glob(args.motion_parquet))
        if not paths:
            raise SystemExit(f"no motion parquet files match {args.motion_parquet!r}")
        clips = iter_motion_parquet(paths)
    else:
        frame_paths = sorted(glob.glob(args.frame_parquet or ""))
        if not frame_paths:
            raise SystemExit("--npz-root requires a --frame-parquet glob with matching shards")
        clips = iter_indexed_ardy_npz(args.npz_root, frame_metadata_index(frame_paths))
    write_cache(clips, args.out)
    print(f"motion cache -> {args.out}")


if __name__ == "__main__":
    main()
