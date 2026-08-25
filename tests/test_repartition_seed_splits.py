import pyarrow as pa
import pyarrow.parquet as pq
import json

from scripts.repartition_seed_splits import repartition_parquet


def test_repartition_changes_only_split_contract(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    rows = [
        {"clip_id": "p0_s0/c0", "seed": 0, "split": "test", "held_out": True, "payload": b"zero"},
        {"clip_id": "p0_s8/c0", "seed": 8, "split": "train", "held_out": True, "payload": b"eight"},
        {"clip_id": "p0_s9/c0", "seed": 9, "split": "val", "held_out": True, "payload": b"nine"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), source / "legacy-00000.parquet")
    (source / "meta.json").write_text(json.dumps({"config": "tiny", "clips": 3, "shards": {"legacy": 1}}))

    report = repartition_parquet(source, output, rows_per_shard=1)

    assert report["rows"] == 3
    assert report["split_rows"] == {"train": 1, "val": 1, "test": 1}
    found = {}
    for path in output.glob("*.parquet"):
        for row in pq.read_table(path).to_pylist():
            found[row["seed"]] = row
    assert {seed: row["split"] for seed, row in found.items()} == {0: "train", 8: "val", 9: "test"}
    assert all(row["held_out"] is False for row in found.values())
    assert {seed: row["payload"] for seed, row in found.items()} == {0: b"zero", 8: b"eight", 9: b"nine"}
    meta = json.loads((output / "meta.json").read_text())
    assert meta["config"] == "tiny"
    assert meta["clips"] == 3
    assert meta["shards"] == {"train": 1, "val": 1, "test": 1}
    assert meta["split_contract"] == {"train": "seeds 0-7", "val": "seed 8", "test": "seed 9"}
