import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.curate_prompts import curate_parquet, load_excluded


def _write_shard(path, split, texts):
    table = pa.table(
        {
            "text": pa.array(texts),
            "split": pa.array([split] * len(texts)),
            "value": pa.array(list(range(len(texts)))),
        }
    )
    pq.write_table(table, path)


def test_load_excluded_parses_group_tab_prompt(tmp_path):
    f = tmp_path / "ex.txt"
    f.write_text(
        "# comment\n"
        "dance\tA person sways to slow music.\t# reason\n"
        "gesture\tA person shakes their head no.\n"
    )
    assert load_excluded(f) == [
        "A person sways to slow music.",
        "A person shakes their head no.",
    ]


def test_curate_drops_only_excluded_prompts_and_reshards(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _write_shard(src / "train-00000.parquet", "train", ["keep A", "drop me", "keep B"])
    _write_shard(src / "train-00001.parquet", "train", ["drop me", "keep C"])
    _write_shard(src / "val-00000.parquet", "val", ["keep A", "drop me"])
    _write_shard(src / "test-00000.parquet", "test", ["keep A"])
    (src / "meta.json").write_text(json.dumps({"version": "0.1.0", "frames": 8}))

    out = tmp_path / "out"
    result = curate_parquet(src, out, ["drop me"], rows_per_shard=2)

    assert result["kept"] == {"train": 3, "val": 1, "test": 1}
    assert result["removed"] == {"train": 2, "val": 1, "test": 0}
    train = pa.concat_tables(
        [pq.read_table(p) for p in sorted(out.glob("train-*.parquet"))]
    )
    assert train.column("text").to_pylist() == ["keep A", "keep B", "keep C"]
    assert result["shards"]["train"] == 2  # 3 rows at 2 rows/shard
    meta = json.loads((out / "meta.json").read_text())
    assert meta["excluded_prompts"] == ["drop me"]
    assert meta["version"] == "0.2.0"
