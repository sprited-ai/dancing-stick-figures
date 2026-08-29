import numpy as np

from eval.fvd_prompt_seed_validation import clip_inventory, grouped_prompt_halves, indices_for


def _clips():
    rows = {}
    start = 0
    for group, prompts in {"gesture": ["wave", "clap"], "locomotion": ["run", "walk"]}.items():
        for prompt in prompts:
            for seed in range(10):
                for camera in (0, 1):
                    clip_id = f"{group}/{prompt}_s{seed}/c{camera}"
                    rows[clip_id] = {"start": start, "n": 120, "text": prompt, "group": group}
                    start += 120
    return rows


def test_inventory_keeps_one_camera_and_all_seeds():
    rows = clip_inventory(_clips(), camera=0)
    assert len(rows) == 40
    assert {row["seed"] for row in rows} == set(range(10))
    assert all(row["clip_id"].endswith("/c0") for row in rows)


def test_grouped_prompt_halves_are_disjoint_and_balanced():
    rows = clip_inventory(_clips(), camera=0)
    left, right, dropped = grouped_prompt_halves(rows, np.random.default_rng(3))
    assert not (left & right)
    assert not dropped
    assert len(left) == len(right) == 2
    assert len(indices_for(rows, prompts=left)) == len(indices_for(rows, prompts=right)) == 20
    assert len(indices_for(rows, seeds={0, 2, 4, 6, 8})) == 20
