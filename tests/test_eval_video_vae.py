from eval.eval_video_vae import resolve_evaluation_frames


def test_common_eval_window_overrides_checkpoint_training_window():
    assert resolve_evaluation_frames({"frames": 80}, 20) == 20
    assert resolve_evaluation_frames({"frames": 80}, 0) == 80
