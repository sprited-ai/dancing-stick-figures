import numpy as np

from eval.keyframes import select_representative_frames


def test_keyframes_stop_at_end_of_active_motion_instead_of_idle_tail():
    joints = np.zeros((10, 2, 3), dtype=np.float32)
    joints[:5, 1, 0] = [0, 1, 2, 3, 4]
    joints[5:, 1, 0] = 4
    indices = select_representative_frames(joints, count=4)
    assert indices[0] == 0
    assert indices[-1] <= 5
    assert indices == sorted(set(indices))


def test_static_motion_falls_back_to_even_temporal_coverage():
    joints = np.zeros((10, 2, 3), dtype=np.float32)
    assert select_representative_frames(joints, count=4) == [0, 3, 6, 9]

