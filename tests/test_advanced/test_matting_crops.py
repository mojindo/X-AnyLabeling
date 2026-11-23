import numpy as np

from anylabeling.views.labeling.advanced.batch_annotation_check.cropper import (
    dilate_mask_alpha,
)
from anylabeling.views.labeling.advanced.matting.crops import crop_mode_for_shape_type
from anylabeling.views.labeling.advanced.batch_annotation_check.cropper import (
    CROP_MODE_HBB,
    CROP_MODE_OBB,
    CROP_MODE_SEG,
)


def test_dilate_mask_alpha_expands():
    alpha = np.zeros((20, 20), dtype=np.uint8)
    alpha[8:12, 8:12] = 255
    out = dilate_mask_alpha(alpha, 2)
    assert int(np.sum(out > 0)) > int(np.sum(alpha > 0))


def test_crop_mode_mapping():
    assert crop_mode_for_shape_type("rectangle") == CROP_MODE_HBB
    assert crop_mode_for_shape_type("rotation") == CROP_MODE_OBB
    assert crop_mode_for_shape_type("polygon") == CROP_MODE_SEG