import numpy as np

from eval.part_motion_localization import CHAINS, freeze_chain
from generator.skeleton import IDX, PARENT


def test_freeze_chain_preserves_lengths_and_unaffected_joints():
    rng = np.random.default_rng(4)
    joints = rng.normal(size=(6, 27, 3)).astype(np.float32)
    # Give every left-arm bone a stable, nonzero length and changing direction.
    for frame in range(len(joints)):
        for offset, child_name in enumerate(CHAINS["L"], start=1):
            parent_name = PARENT[child_name]
            parent = joints[frame, IDX[parent_name]]
            joints[frame, IDX[child_name]] = parent + np.array(
                [1.0, 0.2 * frame, 0.1 * offset], np.float32,
            )

    frozen = freeze_chain(joints, "L", 1.0)

    affected = {IDX[name] for name in CHAINS["L"]}
    unaffected = [index for index in range(27) if index not in affected]
    np.testing.assert_allclose(frozen[:, unaffected], joints[:, unaffected])
    for child_name in CHAINS["L"]:
        child, parent = IDX[child_name], IDX[PARENT[child_name]]
        clean_length = np.linalg.norm(joints[:, child] - joints[:, parent], axis=1)
        frozen_length = np.linalg.norm(frozen[:, child] - frozen[:, parent], axis=1)
        np.testing.assert_allclose(frozen_length, clean_length, rtol=1e-5, atol=1e-5)
        frozen_direction = frozen[:, child] - frozen[:, parent]
        frozen_direction /= np.linalg.norm(frozen_direction, axis=1, keepdims=True)
        np.testing.assert_allclose(
            frozen_direction,
            np.repeat(frozen_direction[:1], len(frozen_direction), axis=0),
            rtol=1e-5,
            atol=1e-5,
        )


def test_zero_severity_is_identity():
    joints = np.random.default_rng(8).normal(size=(3, 27, 3)).astype(np.float32)
    np.testing.assert_allclose(freeze_chain(joints, "R", 0.0), joints)
