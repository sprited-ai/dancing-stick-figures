import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from generator.ardy_adapter import apply_bone_scale
from generator.skeleton import NAMES, IDX, PARENT, NEUTRAL


def _len(Q, n): return np.linalg.norm(Q[0, IDX[n]] - Q[0, IDX[PARENT[n]]])


def test_bone_scale_lengths_and_rigidity():
    Q = np.array([[NEUTRAL[n] for n in NAMES]], dtype=float)
    bs = {"LeftForeArm": 1.5, "LeftLeg": 0.7, "Spine1": 1.2}
    Q2 = apply_bone_scale(Q, bs)
    for n, s in bs.items(): assert abs(_len(Q2, n) / _len(Q, n) - s) < 1e-9
    for n in NAMES:
        if PARENT[n] and n not in bs: assert abs(_len(Q2, n) - _len(Q, n)) < 1e-9   # children move rigidly
    assert np.allclose(apply_bone_scale(Q, {}), Q) and np.allclose(apply_bone_scale(Q, {n: 1.0 for n in NAMES}), Q)


if __name__ == "__main__":
    test_bone_scale_lengths_and_rigidity(); print("ok")
