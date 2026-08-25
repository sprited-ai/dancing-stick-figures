#!/usr/bin/env python3
"""Create a shorter dataset horizon without regenerating motion or pixels."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SPLITS = ("train", "val", "test")


def _set_n_frames(table: pa.Table, value: int) -> pa.Table:
    index = table.schema.get_field_index("n_frames")
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array([value] * len(table), type=field.type))


def _truncate_motion_row(row: dict, arrays: dict, max_frames: int) -> dict:
    source_frames = int(row["n_frames"])
    if source_frames < max_frames:
        raise ValueError(f"clip {row.get('clip_id', '')} has only {source_frames} frames")
    row = dict(row)
    for name, spec in arrays.items():
        shape = list(spec["shape"])
        if not shape or shape[0] != "T" or name not in row:
            continue
        source_shape = [source_frames, *map(int, shape[1:])]
        value = np.frombuffer(row[name], dtype=np.dtype(spec["dtype"])).reshape(source_shape)
        row[name] = np.ascontiguousarray(value[:max_frames]).tobytes()
    row["n_frames"] = max_frames
    return row


def truncate_parquet(
    source: Path | str,
    output: Path | str,
    *,
    max_frames: int = 40,
    rows_per_shard: int = 2000,
    batch_size: int = 256,
) -> dict:
    source, output = Path(source), Path(output)
    files = sorted(source.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet shards under {source}")
    if max_frames < 1 or rows_per_shard < 1 or batch_size < 1:
        raise ValueError("frame and batch sizes must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    meta_path = source / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    schema = pq.ParquetFile(files[0]).schema_arrow
    mode = "frames" if "frame_idx" in schema.names else "motion"
    if mode == "motion" and "posed_joints" not in schema.names:
        raise ValueError("could not identify frame or motion configuration")

    buffers: dict[str, list[pa.Table]] = {split: [] for split in SPLITS}
    buffered = Counter()
    shards = Counter()
    split_rows = Counter()

    def flush(split: str, force: bool = False) -> None:
        while buffered[split] >= rows_per_shard or (force and buffered[split]):
            joined = pa.concat_tables(buffers[split])
            count = min(rows_per_shard, len(joined))
            chunk, remainder = joined.slice(0, count), joined.slice(count)
            pq.write_table(chunk, output / f"{split}-{shards[split]:05d}.parquet", compression="zstd")
            shards[split] += 1
            buffers[split] = [remainder] if len(remainder) else []
            buffered[split] -= count

    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            if mode == "frames":
                table = table.filter(pc.less(table.column("frame_idx"), max_frames))
                if not len(table):
                    continue
                table = _set_n_frames(table, max_frames)
            else:
                rows = [_truncate_motion_row(row, meta.get("arrays", {}), max_frames) for row in table.to_pylist()]
                table = pa.Table.from_pylist(rows, schema=table.schema)
            split_values = table.column("split").to_pylist()
            for split in SPLITS:
                indices = [index for index, value in enumerate(split_values) if value == split]
                if not indices:
                    continue
                part = table.take(pa.array(indices, type=pa.int64()))
                buffers[split].append(part)
                buffered[split] += len(part)
                split_rows[split] += len(part)
                flush(split)

    for split in SPLITS:
        flush(split, force=True)

    fps = int(meta.get("fps", 20))
    meta["version"] = "0.2.0"
    meta["shards"] = {split: shards[split] for split in SPLITS}
    meta["clip_frames"] = max_frames
    meta["clip_seconds"] = max_frames / fps
    if mode == "frames":
        meta["frames"] = sum(split_rows.values())
    meta["horizon_contract"] = "first 40 native-cadence ARDY frames" if max_frames == 40 and fps == 20 else f"first {max_frames} frames"
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    report = {
        "protocol": "truncate_horizon_v1",
        "mode": mode,
        "source": str(source.resolve()),
        "max_frames": max_frames,
        "fps": fps,
        "clip_seconds": max_frames / fps,
        "rows": sum(split_rows.values()),
        "split_rows": {split: split_rows[split] for split in SPLITS},
        "shards": {split: shards[split] for split in SPLITS},
        "changed_columns": ["n_frames"],
    }
    (output / "truncate.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--rows-per-shard", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(truncate_parquet(
        args.source, args.out, max_frames=args.frames,
        rows_per_shard=args.rows_per_shard, batch_size=args.batch_size,
    ), indent=2))


if __name__ == "__main__":
    main()
