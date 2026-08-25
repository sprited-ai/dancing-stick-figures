#!/usr/bin/env python3
"""Repartition released Parquet rows by ARDY seed without rerendering data.

Seeds 0--7 become train, seed 8 validation (``val``), and seed 9 test.
Every column other than ``split`` and ``held_out`` is copied unchanged.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from generator.build import split_for_seed


SPLITS = ("train", "val", "test")


def _replace_contract_columns(table: pa.Table, split: str) -> pa.Table:
    split_index = table.schema.get_field_index("split")
    held_index = table.schema.get_field_index("held_out")
    if split_index < 0 or held_index < 0:
        raise ValueError("Parquet schema must contain split and held_out columns")
    table = table.set_column(
        split_index,
        table.schema.field(split_index),
        pa.array([split] * len(table), type=table.schema.field(split_index).type),
    )
    return table.set_column(
        held_index,
        table.schema.field(held_index),
        pa.array([False] * len(table), type=table.schema.field(held_index).type),
    )


def repartition_parquet(
    source: Path | str,
    output: Path | str,
    *,
    rows_per_shard: int = 2000,
    batch_size: int = 256,
) -> dict:
    source, output = Path(source), Path(output)
    files = sorted(source.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet shards under {source}")
    if rows_per_shard < 1 or batch_size < 1:
        raise ValueError("rows_per_shard and batch_size must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    buffers: dict[str, list[pa.Table]] = {split: [] for split in SPLITS}
    buffered = Counter()
    shard_counts = Counter()
    split_rows = Counter()
    input_rows = 0

    def flush(split: str, force: bool = False) -> None:
        while buffered[split] >= rows_per_shard or (force and buffered[split]):
            joined = pa.concat_tables(buffers[split])
            count = min(rows_per_shard, len(joined))
            chunk = joined.slice(0, count)
            remainder = joined.slice(count)
            destination = output / f"{split}-{shard_counts[split]:05d}.parquet"
            pq.write_table(chunk, destination, compression="zstd")
            shard_counts[split] += 1
            buffers[split] = [remainder] if len(remainder) else []
            buffered[split] -= count

    for path in files:
        parquet = pq.ParquetFile(path)
        if "seed" not in parquet.schema_arrow.names:
            raise ValueError(f"{path} has no seed column")
        for batch in parquet.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            seeds = table.column("seed").to_pylist()
            input_rows += len(seeds)
            indices = {split: [] for split in SPLITS}
            for index, seed in enumerate(seeds):
                indices[split_for_seed(int(seed))].append(index)
            for split, selected in indices.items():
                if not selected:
                    continue
                part = table.take(pa.array(selected, type=pa.int64()))
                part = _replace_contract_columns(part, split)
                buffers[split].append(part)
                buffered[split] += len(part)
                split_rows[split] += len(part)
                flush(split)

    for split in SPLITS:
        flush(split, force=True)
    report = {
        "protocol": "seed_split_v1",
        "source": str(source.resolve()),
        "rows": input_rows,
        "split_rows": {split: split_rows[split] for split in SPLITS},
        "shards": {split: shard_counts[split] for split in SPLITS},
        "contract": {"train": "seeds 0-7", "val": "seed 8", "test": "seed 9"},
        "changed_columns": ["split", "held_out"],
    }
    source_meta = source / "meta.json"
    if source_meta.is_file():
        meta = json.loads(source_meta.read_text())
        meta["shards"] = {split: shard_counts[split] for split in SPLITS}
        meta["split_contract"] = report["contract"]
        meta["held_out_groups"] = []
        (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (output / "repartition.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows-per-shard", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    report = repartition_parquet(
        args.source,
        args.out,
        rows_per_shard=args.rows_per_shard,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
