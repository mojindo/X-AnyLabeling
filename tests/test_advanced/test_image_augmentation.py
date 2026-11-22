import json
import os.path as osp

import cv2
import numpy as np

from anylabeling.views.labeling.advanced.augment.engine import (
    AugmentStep,
    _apply_one,
    _motion_kernel,
    apply_one_method_dataset,
    assign_one_method,
)


def test_each_image_gets_one_method_in_turn():
    fog = AugmentStep("random_fog", True, 1.0)
    rain = AugmentStep("random_rain", True, 1.0)
    assert assign_one_method(0, [fog, rain]).op_id == "random_fog"
    assert assign_one_method(1, [fog, rain]).op_id == "random_rain"
    assert assign_one_method(2, [fog, rain]).op_id == "random_fog"


def test_output_contains_only_augmented_image_and_json(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "out"
    src.mkdir()
    image = src / "a.jpg"
    bgr = np.zeros((32, 32, 3), dtype=np.uint8)
    bgr[:, :] = (20, 40, 60)
    cv2.imwrite(str(image), bgr)
    label = {"imagePath": "a.jpg", "imageWidth": 32, "imageHeight": 32, "shapes": []}
    (src / "a.json").write_text(json.dumps(label), encoding="utf-8")
    total, ok = apply_one_method_dataset(
        image_paths=[str(image)],
        steps=[AugmentStep("brightness_contrast", True, 1.0, {"brightness_limit": 0.4, "contrast_limit": 0.2})],
        overwrite=False,
        output_dir=str(dst),
    )
    assert (total, ok) == (1, 1)
    written = list(dst.iterdir())
    names = sorted(path.name for path in written)
    assert len(names) == 2
    assert names[0].startswith("a_brightness_contrast_")
    assert names[0].endswith(".jpg")
    assert names[1] == osp.splitext(names[0])[0] + ".json"
    saved = json.loads((dst / names[1]).read_text(encoding="utf-8"))
    assert saved["imagePath"] == names[0]
    assert not (dst / "a.jpg").exists()


def test_motion_direction_changes_the_streak():
    horizontal = _motion_kernel(9, 0)
    vertical = _motion_kernel(9, 90)
    assert horizontal[4, :].sum() > horizontal[:, 4].sum()
    assert vertical[:, 4].sum() > vertical[4, :].sum()
    image = np.zeros((31, 31, 3), dtype=np.uint8)
    image[15, :, :] = 255
    rng = np.random.default_rng(1)
    along, _ = _apply_one(image, AugmentStep("motion_blur", True, 1.0, {"blur_limit": 9, "angle_lower": 0, "angle_upper": 0}), rng)
    across, _ = _apply_one(image, AugmentStep("motion_blur", True, 1.0, {"blur_limit": 9, "angle_lower": 90, "angle_upper": 90}), rng)
    assert across[13, 15, 0] > along[13, 15, 0]


def test_grid_distortion_moves_rectangle_points():
    image = np.full((80, 100, 3), 30, dtype=np.uint8)
    out, mapper = _apply_one(
        image,
        AugmentStep("grid_distortion", True, 1.0, {"num_steps": 4, "distort_limit": 0.6}),
        np.random.default_rng(2),
    )
    assert mapper is not None
    assert out.shape == image.shape
    source = np.array([[20.0, 20.0], [70.0, 20.0], [70.0, 60.0], [20.0, 60.0]])
    moved = mapper(source)
    assert not np.allclose(moved, source, atol=0.5)


def test_added_photometric_ops_keep_annotations_still():
    image = np.zeros((40, 48, 3), dtype=np.uint8)
    image[5:30, 8:40] = (20, 80, 160)
    cases = [
        ("iso_noise", {"color_shift_lo": 0.05, "color_shift_hi": 0.05, "intensity_lo": 0.8, "intensity_hi": 0.8}),
        ("rgb_shift", {"shift_limit": 30}),
        ("random_gamma", {"gamma_lo": 60, "gamma_hi": 60}),
        ("clahe", {"clip_lo": 4, "clip_hi": 4, "tile_grid": 4}),
        ("blur", {"blur_limit": (5, 5)}),
        ("median_blur", {"blur_limit": (5, 5)}),
        ("coarse_dropout", {"holes_lo": 3, "holes_hi": 3, "height_lo": 0.2, "height_hi": 0.2, "width_lo": 0.2, "width_hi": 0.2, "fill_value": 0}),
    ]
    for op_id, params in cases:
        out, mapper = _apply_one(image, AugmentStep(op_id, True, 1.0, params), np.random.default_rng(3))
        assert mapper is None
        assert out.shape == image.shape
        assert not np.array_equal(out, image)


def test_optical_and_elastic_return_a_point_mapper():
    image = np.full((64, 64, 3), 40, dtype=np.uint8)
    image[16:48, 16:48] = 200
    for op_id, params in (
        ("optical_distortion", {"distort_limit": 0.2, "shift_limit": 0.0}),
        ("elastic_transform", {"alpha": 20, "sigma": 8, "alpha_affine": 0}),
    ):
        out, mapper = _apply_one(image, AugmentStep(op_id, True, 1.0, params), np.random.default_rng(4))
        assert mapper is not None
        assert out.shape == image.shape
        moved = mapper(np.array([[32.0, 32.0]]))
        assert moved.shape == (1, 2)

