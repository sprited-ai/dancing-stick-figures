from pathlib import Path

import pytest

from generator.build import load_prompts, split_for_seed


PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "v1.txt"


def test_v01_uses_seed_only_split_contract():
    assert [split_for_seed(seed) for seed in range(10)] == [
        "train", "train", "train", "train", "train",
        "train", "train", "train", "val", "test",
    ]
    with pytest.raises(ValueError):
        split_for_seed(-1)
    with pytest.raises(ValueError):
        split_for_seed(10)


def test_every_prompt_appears_in_all_splits():
    prompts, held_groups = load_prompts(PROMPTS)
    assert len(prompts) == 143
    assert held_groups == set()
    expected = {"train", "val", "test"}
    for prompt in prompts:
        assert {split_for_seed(seed) for seed in range(10)} == expected, prompt
