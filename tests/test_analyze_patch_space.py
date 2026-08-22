import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.analyze_patch_space import patchify, resize_seg_majority, summarize


def test_patchify_and_summary_detect_blank_and_thin_patches():
    image = np.zeros((4, 4, 4), dtype=np.float32)
    image[0, 0] = 1
    features = patchify(image, 2).reshape(4, -1)
    occupancy = patchify((image[..., 3:] > 0).astype(np.float32), 2)[..., 0].mean(1)
    seg = np.zeros((4, 4, 1), dtype=np.uint8)
    seg[0, 0, 0] = 1
    seg_patches = patchify(seg, 2)[..., 0]
    result = summarize(features, occupancy, seg_patches, 2)
    assert result["blank_patch_fraction"] == 0.75
    assert result["thin_nonblank_fraction_le_25pct_alpha_pixels"] == 1.0
    assert result["single_seg_pixel_fraction_nonblank"] == 1.0


def test_seg_majority_resize_includes_background():
    seg = np.zeros((4, 4), dtype=np.uint8)
    seg[0, 0] = 3
    small = resize_seg_majority(seg, 2)
    assert small.shape == (2, 2)
    assert not small.any()
