import numpy as np

from eval.oracle import NAMES, PAL
from eval.palette_mask_validation import (
    SEGMENT_CLASS,
    classify_rgba,
    segmentation_classes,
    selected_frame,
    summarise,
    empty_counts,
    update_counts,
)
from generator.skeleton import NAMES as SEGMENT_NAMES


def test_palette_and_alpha_thresholds_are_strict():
    rgba = np.zeros((1, 4, 4), np.uint8)
    rgba[0, :, :3] = PAL["ink"]
    rgba[0, :, 3] = (128, 128, 127, 128)
    rgba[0, 0, 0] += 59
    rgba[0, 1, 0] += 60

    labels, foreground = classify_rgba(rgba, tau=60)
    assert labels[0, 0] == NAMES.index("ink")
    assert labels[0, 1] == -1
    assert labels[0, 2] == -1
    assert labels[0, 3] == NAMES.index("ink")
    assert foreground.tolist() == [[True, True, False, True]]


def test_segment_ids_map_to_renderer_palette_classes():
    ids = np.array([[0, SEGMENT_NAMES.index("LeftForeArm") + 1,
                     SEGMENT_NAMES.index("LeftHand") + 1,
                     SEGMENT_NAMES.index("Spine") + 1]], np.uint8)
    classes = segmentation_classes(ids)
    assert classes.tolist() == [[-1, NAMES.index("arm_L"), NAMES.index("fore_L"), NAMES.index("ink")]]
    assert SEGMENT_CLASS.shape == (28,)


def test_summary_counts_unassigned_pixels_as_false_negatives():
    rgba = np.zeros((2, 2, 4), np.uint8)
    rgba[..., :3] = PAL["arm_L"]
    rgba[..., 3] = 255
    rgba[0, 0, :3] = (150, 150, 150)
    seg_id = SEGMENT_NAMES.index("LeftForeArm") + 1
    segmentation = np.full((2, 2), seg_id, np.uint8)
    counts = empty_counts()
    update_counts(counts, rgba, segmentation, tau=60)
    result = summarise(counts)
    assert result["assigned_fraction"] == 0.75
    assert result["pixel_accuracy"] == 0.75
    assert result["accuracy_given_assigned"] == 1.0
    assert result["per_class"]["arm_L"]["recall"] == 0.75


def test_selected_frame_is_stable_and_bounded():
    assert selected_frame("gesture/example_s3/c0") == selected_frame("gesture/example_s3/c0")
    assert 0 <= selected_frame("gesture/example_s3/c0") < 120
