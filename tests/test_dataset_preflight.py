import hashlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.dataset_preflight import fingerprint_cache, validate_fingerprint


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    frames = np.zeros((6, 8, 8, 4), dtype=np.uint8)
    frames[:, 2:6, 3:5, :3] = (40, 110, 230)
    frames[:, 2:6, 3:5, 3] = 255
    np.save(cache / "frames.npy", frames)
    clips = {
        "a": {"start": 0, "n": 3, "split": "train", "text": "wave"},
        "b": {"start": 3, "n": 3, "split": "test", "text": "jump"},
    }
    (cache / "clips.json").write_text(json.dumps(clips))
    (cache / "meta.json").write_text(json.dumps({"size": 8, "frames": 6, "clips": 2}))
    return cache


def test_fingerprint_records_content_shape_and_splits(tmp_path):
    cache = _cache(tmp_path)
    result = fingerprint_cache(cache)
    assert result["files"]["frames.npy"]["sha256"] == _sha256(cache / "frames.npy")
    assert result["array"] == {"shape": [6, 8, 8, 4], "dtype": "uint8"}
    assert result["clips"] == {"count": 2, "splits": {"test": 1, "train": 1}}


def test_validation_rejects_same_shape_but_different_renderer_bytes(tmp_path):
    cache = _cache(tmp_path)
    actual = fingerprint_cache(cache)
    expected = json.loads(json.dumps(actual))
    frames = np.load(cache / "frames.npy")
    frames[0, 0, 0, 0] = 1
    np.save(cache / "frames.npy", frames)
    changed = fingerprint_cache(cache)
    assert changed["array"] == actual["array"]
    with pytest.raises(ValueError, match="frames.npy.sha256"):
        validate_fingerprint(changed, expected)


def test_validation_reports_all_identity_mismatches():
    actual = {
        "files": {"frames.npy": {"sha256": "bad"}},
        "array": {"shape": [10, 64, 64, 4], "dtype": "uint8"},
        "clips": {"count": 3, "splits": {"train": 3}},
    }
    expected = {
        "files": {"frames.npy": {"sha256": "good"}},
        "array": {"shape": [20, 64, 64, 4], "dtype": "uint8"},
        "clips": {"count": 4, "splits": {"train": 3, "test": 1}},
    }
    with pytest.raises(ValueError) as error:
        validate_fingerprint(actual, expected)
    message = str(error.value)
    assert "frames.npy.sha256" in message
    assert "array.shape" in message
    assert "clips.count" in message
    assert "clips.splits" in message
