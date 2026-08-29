import numpy as np

from eval.i3d_embedding_validation import prompt_id, validate_embeddings


def _row(prompt, seed, group="gesture"):
    return {"motion_id": f"{group}/{prompt}_s{seed}"}


def test_prompt_id_removes_only_terminal_seed():
    assert prompt_id(_row("person_turns_180", 7)) == "gesture/person_turns_180"


def test_validation_separates_prompts_and_reports_paired_corruption():
    # Two prompts with three seeds each. Same-prompt points form tight clusters;
    # the corruption moves every B point far from its clean source.
    a = np.asarray([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]])
    b = np.asarray([[0.0, 0.1], [10.1, 0.0], [10.0, 0.1]])
    manifest = {
        "statistical_unit": "source_motion",
        "reference_a": [_row("wave", 0), _row("wave", 1), _row("jump", 0)],
        "reference_b": [_row("wave", 2), _row("jump", 1), _row("jump", 2)],
    }
    result = validate_embeddings(
        {
            "real_reference_a": a,
            "real_reference_b": b,
            "repeat_first": b + np.asarray([20.0, 20.0]),
        },
        manifest,
        seed=3,
        bootstrap_draws=50,
    )
    clean = result["clean_pair_distance"]
    assert clean["same_prompt_different_seed"]["median"] < clean["different_prompt_same_group"]["median"]
    assert clean["probability_same_prompt_is_closer"] == 1.0
    assert clean["nearest_neighbour_prompt_accuracy"] == 1.0
    assert clean["random_neighbour_prompt_accuracy"] == 0.4
    assert result["paired_clean_to_corruption_distance"]["repeat_first"]["fraction_above_clean_same_prompt_median"] == 1.0
