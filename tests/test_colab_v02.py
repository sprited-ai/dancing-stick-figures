import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "dancing_stick_figures_colab_v0_2.ipynb"


def notebook_text():
    document = json.loads(NOTEBOOK.read_text())
    return document, "\n".join(
        "".join(cell.get("source", [])) for cell in document["cells"]
    )


def test_v02_exposes_the_reference_backbone_without_an_option_matrix():
    document, text = notebook_text()
    assert document["nbformat"] == 4
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
    assert "The fixed reference backbone" in text
    assert "ResBlock, Attn, UNet3D, initialize_video_input" in text
    assert 'IMAGE_SIZE = "64" #@param ["32", "64"]' in text
    assert "BASE_CHANNELS, NEW_FRAMES, CONTEXT_FRAMES = 64, 8, 8" in text


def test_v02_trains_both_stages_then_scores_the_typed_prompt_rollout():
    _, text = notebook_text()
    assert "--frames 1" in text and "--cond text" in text
    assert "--ar_ctx $CONTEXT_FRAMES" in text
    assert '--prompt "$PROMPT"' in text
    assert "--score --cache data/cache" in text
    assert "mean-pooled and added through the time-conditioning path" in text
    assert "IMAGE_WALL_SECONDS" in text and "VIDEO_WALL_SECONDS" in text
    assert "V02_STARTED" in text and "V02_TOTAL_WALL_SECONDS" in text
    assert "V02_IMAGE_PEAK_GB" in text and "V02_VIDEO_PEAK_GB" in text
    assert "V02_COMPLETE=1" in text
    assert "centroid speed, acceleration, motion fraction, and angular jerk" in text
    assert "--sample_every $VSTEPS --val_every 400" in text
    assert "Sampling is intentionally deferred until training finishes" in text


def test_v02_dataset_setup_verifies_download_and_contact_sheet_is_shard_agnostic():
    document, text = notebook_text()
    assert "DATA_DOWNLOAD_ATTEMPTS = 3" in text
    assert "download_cmds" in text
    assert 'for pattern in ("mini/*", "motion/val-*")' in text
    assert "all(result.returncode == 0 for result in results)" in text
    assert "Dataset download did not produce mini parquet files" in text
    assert "random.sample(files, 16)" not in text
    assert "frame_candidates" in text
    assert "random.sample(frame_candidates, 32)" in text
    cells = ["".join(cell.get("source", [])) for cell in document["cells"]]
    score_index = next(i for i, source in enumerate(cells) if "Measure visible structure" in source)
    completion_index = next(i for i, source in enumerate(cells) if "Verification record" in source)
    next_steps_index = next(i for i, source in enumerate(cells) if "Where to go next" in source)
    assert score_index < completion_index < next_steps_index
