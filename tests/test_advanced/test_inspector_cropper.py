import numpy as np

from anylabeling.views.labeling.advanced.batch_annotation_check.cropper import (
    CROP_MODE_HBB,
    CROP_MODE_OBB,
    CROP_MODE_SEG,
    crop_output_extension,
    crop_rotation_rectified_bgra,
    crop_shape_to_array,
    shape_eligible_for_crop_mode,
)


def test_shape_eligible_modes():
    assert shape_eligible_for_crop_mode("rectangle", CROP_MODE_HBB)
    assert shape_eligible_for_crop_mode("rotation", CROP_MODE_OBB)
    assert not shape_eligible_for_crop_mode("rectangle", CROP_MODE_OBB)
    assert shape_eligible_for_crop_mode("polygon", CROP_MODE_SEG)


def test_crop_extensions():
    assert crop_output_extension(CROP_MODE_HBB) == ".jpg"
    assert crop_output_extension(CROP_MODE_OBB) == ".png"
    assert crop_output_extension(CROP_MODE_SEG) == ".png"


def test_obb_masked_png_has_alpha():
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[:, :] = (40, 80, 120)
    quad = [[20, 20], [60, 25], [55, 55], [15, 50]]
    out = crop_shape_to_array(
        img,
        shape_type="rotation",
        points=quad,
        img_w=80,
        img_h=80,
        crop_mode=CROP_MODE_OBB,
        obb_rectify=False,
    )
    assert out is not None
    assert out.shape[2] == 4
    assert out[:, :, 3].max() > 0


def test_obb_rectify_uses_adjacent_edges_not_long_short():
    """Width/height follow point order (edge 0→1, 1→2), not long vs short."""
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    cx, cy = 60.0, 60.0
    half_w, half_h = 30.0, 10.0
    theta = np.pi / 6.0
    c, s = np.cos(theta), np.sin(theta)
    corners = [
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h],
    ]
    quad = []
    for x, y in corners:
        quad.append([cx + x * c - y * s, cy + x * s + y * c])
    edge_w = 2 * half_w
    edge_h = 2 * half_h
    out = crop_rotation_rectified_bgra(img, quad)
    assert out is not None
    assert abs(out.shape[1] - round(edge_w)) <= 2
    assert abs(out.shape[0] - round(edge_h)) <= 2


def test_seg_polygon_mask():
    img = np.full((50, 50, 3), 200, dtype=np.uint8)
    poly = [[10, 10], [40, 10], [40, 40], [10, 40]]
    out = crop_shape_to_array(
        img,
        shape_type="polygon",
        points=poly,
        img_w=50,
        img_h=50,
        crop_mode=CROP_MODE_SEG,
    )
    assert out is not None and out.shape[2] == 4