import numpy as np

from anylabeling.views.labeling.advanced.desensitize.mosaic import (
    apply_mosaic_bgr,
    apply_solid_gray_bgr,
)


def test_mosaic_only_on_mask():
    xx, yy = np.meshgrid(np.arange(40), np.arange(40))
    img = np.stack(
        [
            (xx * 7).astype(np.uint8),
            (yy * 5).astype(np.uint8),
            ((xx + yy) * 3).astype(np.uint8),
        ],
        axis=2,
    )
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    out = apply_mosaic_bgr(img, mask, block_size=4)
    assert np.array_equal(out[0:8, :], img[0:8, :])
    assert np.array_equal(out[25:40, 25:40], img[25:40, 25:40])
    assert not np.array_equal(out[12:18, 12:18], img[12:18, 12:18])


def test_solid_gray_only_on_mask():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[:] = (100, 50, 200)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    out = apply_solid_gray_bgr(img, mask, gray=42)
    assert np.array_equal(out[0:4, 0:4], img[0:4, 0:4])
    assert np.all(out[6:14, 6:14] == (42, 42, 42))