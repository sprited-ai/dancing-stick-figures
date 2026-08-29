import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "dancing_stick_figures_colab_v0_3.ipynb"


def notebook_text():
    document = json.loads(NOTEBOOK.read_text())
    return document, "\n".join(
        "".join(cell.get("source", [])) for cell in document["cells"]
    )


def test_v03_is_a_clean_dit_first_notebook():
    document, text = notebook_text()
    assert document["nbformat"] == 4
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
    assert "factorised video DiT" in text
    assert "Attention, TextCrossAttention, Block, VideoDiT, prepare_warmstart_state" in text
    assert 'IMAGE_SIZE = "64" #@param ["32", "64"]' in text
    assert "VIDEO_FRAMES, FRAME_STRIDE = 64, 1" in text
    assert "UNet3D" not in text


def test_v03_separates_public_clips_from_the_first_64_frame_protocol():
    _, text = notebook_text()
    assert "120 frames at 20 fps" in text
    assert "first 64 frames" in text and "native 20 fps" in text
    assert "first-64-frame window" in text
    assert "--arch dit" in text or '"--arch", "dit"' in text
    assert "--frames 1" in text or '\"--frames\", \"1\"' in text
    assert "--frames $VIDEO_FRAMES --stride $FRAME_STRIDE" in text or '"--frames", str(VIDEO_FRAMES)' in text
    assert "--init $INIT_CKPT" in text or '"--init", INIT_CKPT' in text
    assert "--cond text" in text or '"--cond", "text"' in text
    assert "--i2v_frac 0.2" in text or '"--i2v_frac", "0.2"' in text
    assert "eval.post_eval_t2v" in text
    assert "V03_COMPLETE=1" in text


def test_v03_keeps_setup_and_diagnostics_in_order():
    document, text = notebook_text()
    assert "DATA_DOWNLOAD_ATTEMPTS = 3" in text
    assert "Dataset download did not produce mini parquet files" in text
    cells = ["".join(cell.get("source", [])) for cell in document["cells"]]
    image_index = next(i for i, source in enumerate(cells) if "Train the image DiT" in source)
    video_index = next(i for i, source in enumerate(cells) if "Train the 64-frame video DiT" in source)
    score_index = next(i for i, source in enumerate(cells) if "Measure visible structure" in source)
    completion_index = next(i for i, source in enumerate(cells) if "Verification record" in source)
    assert image_index < video_index < score_index < completion_index


def test_v03_reports_the_current_budget_and_propagates_training_failures():
    _, text = notebook_text()
    assert "STEPS = 5000" in text
    assert "default 5,000-step exercise" in text
    assert "Two thousand steps" not in text
    assert "raise subprocess.CalledProcessError(return_code, cmd)" in text
    assert "raise subprocess.CalledProcessError(video_return_code, video_cmd)" in text
