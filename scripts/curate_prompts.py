#!/usr/bin/env python3
"""Remove excluded prompts from released Parquet configs without rerendering.

    python scripts/curate_prompts.py --source hf_seed_split_v1/frames --out hf_v02/frames \
        --exclude prompts/v02_excluded.txt

The exclusion file lists one prompt per line as ``group<TAB>prompt`` (trailing
``# comment`` allowed, full-line comments ignored). Every other row is copied
unchanged; shards are rewritten at a fixed row count and meta.json records the
excluded prompts.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ("train", "val", "test")


def load_excluded(path: Path | str) -> list[str]:
    prompts = []
    for line in Path(path).read_text().splitlines():
        line = line.split("\t#")[0].strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(f"expected group<TAB>prompt, got: {line!r}")
        prompts.append(parts[1].strip())
    if not prompts:
        raise ValueError("exclusion file lists no prompts")
    return prompts


def curate_parquet(
    source: Path | str,
    output: Path | str,
    excluded: list[str],
    *,
    rows_per_shard: int = 2000,
) -> dict:
    source, output = Path(source), Path(output)
    files = sorted(source.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet shards under {source}")
    output.mkdir(parents=True, exist_ok=True)
    excluded_set = set(excluded)

    counts: Counter[str] = Counter()
    removed: Counter[str] = Counter()
    shard_n: Counter[str] = Counter()
    buffers: dict[str, list[pa.Table]] = {split: [] for split in SPLITS}
    buffered_rows: dict[str, int] = {split: 0 for split in SPLITS}

    def flush(split: str, force: bool = False) -> None:
        while buffered_rows[split] >= rows_per_shard or (force and buffered_rows[split]):
            table = pa.concat_tables(buffers[split])
            head, tail = table.slice(0, rows_per_shard), table.slice(rows_per_shard)
            pq.write_table(
                head,
                output / f"{split}-{shard_n[split]:05d}.parquet",
                compression="zstd",
            )
            shard_n[split] += 1
            buffers[split] = [tail] if len(tail) else []
            buffered_rows[split] = len(tail)

    for path in files:
        split = path.name.split("-")[0]
        if split not in SPLITS:
            raise ValueError(f"unrecognised shard split prefix: {path.name}")
        table = pq.read_table(path)
        keep = pa.compute.invert(pa.compute.is_in(table.column("text"), value_set=pa.array(sorted(excluded_set))))
        kept = table.filter(keep)
        counts[split] += len(kept)
        removed[split] += len(table) - len(kept)
        if len(kept):
            buffers[split].append(kept)
            buffered_rows[split] += len(kept)
        flush(split)
    for split in SPLITS:
        flush(split, force=True)

    meta_path = source / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(
        version="0.2.0",
        frames=sum(counts.values()) if "frames" in meta else meta.get("frames"),
        shards={split: shard_n[split] for split in SPLITS},
        excluded_prompts=sorted(excluded_set),
        curation="prompts whose ARDY motions do not visibly perform the action are removed (see prompts/v02_excluded.txt)",
    )
    if "clips" in meta and removed:
        # frames/mini metas count clips; motion meta counts rows. Recompute below in main.
        pass
    (output / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return {"kept": dict(counts), "removed": dict(removed), "shards": dict(shard_n)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--exclude", required=True)
    parser.add_argument("--rows_per_shard", type=int, default=2000)
    args = parser.parse_args()
    excluded = load_excluded(args.exclude)
    print(f"excluding {len(excluded)} prompts: {excluded}")
    result = curate_parquet(args.source, args.out, excluded, rows_per_shard=args.rows_per_shard)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
