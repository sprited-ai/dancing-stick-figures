import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.truncate_dataset_horizon import truncate_parquet


def test_truncates_frame_rows_and_updates_n_frames(tmp_path):
    source, output = tmp_path / "frames", tmp_path / "out"
    source.mkdir()
    rows = [
        {"sample_id": f"c/f{i:03d}", "frame_idx": i, "n_frames": 6, "split": "train", "payload": bytes([i])}
        for i in range(6)
    ]
    pq.write_table(pa.Table.from_pylist(rows), source / "train-00000.parquet")
    (source / "meta.json").write_text(json.dumps({"frames": 6, "fps": 2, "version": "0.1.0"}))

    report = truncate_parquet(source, output, max_frames=4, rows_per_shard=2)

    tables = [pq.read_table(path) for path in sorted(output.glob("*.parquet"))]
    found = pa.concat_tables(tables).to_pylist()
    assert [row["frame_idx"] for row in found] == [0, 1, 2, 3]
    assert [row["n_frames"] for row in found] == [4, 4, 4, 4]
    assert [row["payload"] for row in found] == [b"\0", b"\1", b"\2", b"\3"]
    assert report["rows"] == 4
    meta = json.loads((output / "meta.json").read_text())
    assert meta["frames"] == 4
    assert meta["clip_frames"] == 4
    assert meta["clip_seconds"] == 2.0


def test_truncates_every_time_dependent_motion_array(tmp_path):
    source, output = tmp_path / "motion", tmp_path / "out"
    source.mkdir()
    posed = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    contacts = (np.arange(6) % 2 == 0).reshape(6, 1)
    row = {
        "clip_id": "c", "n_frames": 6, "fps": 2, "split": "test",
        "posed_joints": posed.tobytes(), "foot_contacts": contacts.tobytes(), "fixed": b"keep",
    }
    pq.write_table(pa.Table.from_pylist([row]), source / "test-00000.parquet")
    meta = {
        "config": "motion", "clips": 1,
        "arrays": {
            "posed_joints": {"dtype": "float32", "shape": ["T", 2]},
            "foot_contacts": {"dtype": "bool", "shape": ["T", 1]},
        },
    }
    (source / "meta.json").write_text(json.dumps(meta))

    truncate_parquet(source, output, max_frames=4, rows_per_shard=2)

    found = pq.read_table(next(output.glob("*.parquet"))).to_pylist()[0]
    assert found["n_frames"] == 4
    np.testing.assert_array_equal(np.frombuffer(found["posed_joints"], np.float32).reshape(4, 2), posed[:4])
    np.testing.assert_array_equal(np.frombuffer(found["foot_contacts"], bool).reshape(4, 1), contacts[:4])
    assert found["fixed"] == b"keep"

