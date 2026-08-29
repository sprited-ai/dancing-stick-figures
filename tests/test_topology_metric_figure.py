import numpy as np

from paper.make_topology_metric_figure import examples, scores


def test_each_controlled_example_isolates_one_metric():
    measured = scores(examples())

    assert measured["clean"]["tvr"] == 0.0
    assert measured["clean"]["lie"] == 0.0
    assert measured["clean"]["cpe"] < 0.01

    assert measured["extra_component"]["tvr"] == 1 / 8
    assert measured["extra_component"]["lie"] == 0.0
    assert measured["extra_component"]["cpe"] < 0.01

    assert measured["broken_adjacency"]["tvr"] == 0.0
    assert measured["broken_adjacency"]["lie"] == 1 / 8
    assert measured["broken_adjacency"]["cpe"] < 0.01

    assert measured["off_palette"]["tvr"] == 0.0
    assert measured["off_palette"]["lie"] == 0.0
    assert measured["off_palette"]["cpe"] > 0.09


def test_examples_are_deterministic_rgba_frames():
    first = examples()
    second = examples()
    assert set(first) == set(second)
    for key in first:
        assert first[key].shape == (64, 64, 4)
        assert first[key].dtype == np.uint8
        np.testing.assert_array_equal(first[key], second[key])
