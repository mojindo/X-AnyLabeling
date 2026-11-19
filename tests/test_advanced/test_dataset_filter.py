import json
import os
import tempfile
import unittest

from anylabeling.views.labeling.advanced.compare.annotation_compare import (
    COMPARE_STATUS,
    STATUS_MODEL_FALSE,
    STATUS_SUSPECT_FP,
    STATUS_SUSPECT_FN,
)
from anylabeling.views.labeling.advanced.ops.dataset_filter import (
    SUSPECT_ATTR_MODEL_FALSE,
    SUSPECT_ATTR_NORMAL,
    SUSPECT_ATTR_SUSPECT_FN,
    image_matches_filter,
    is_strict_empty_annotation,
)


class TestDatasetFilter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.image_path = os.path.join(self.root, "a.jpg")
        open(self.image_path, "wb").close()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_json(self, shapes):
        json_path = os.path.join(self.root, "a.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": "1.0.0",
                    "imagePath": "a.jpg",
                    "imageWidth": 100,
                    "imageHeight": 100,
                    "shapes": shapes,
                },
                f,
            )

    def test_strict_empty_ignores_suspect_only(self):
        self._write_json(
            [
                {
                    "label": "pred",
                    "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "shape_type": "rectangle",
                    "attributes": {COMPARE_STATUS: STATUS_SUSPECT_FN},
                }
            ]
        )
        self.assertFalse(
            is_strict_empty_annotation(self.image_path, output_dir=self.root)
        )
        self._write_json([])
        self.assertTrue(
            is_strict_empty_annotation(self.image_path, output_dir=self.root)
        )

    def test_suspect_or_within_group(self):
        self._write_json(
            [
                {
                    "label": "car",
                    "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "shape_type": "rectangle",
                    "attributes": {COMPARE_STATUS: STATUS_MODEL_FALSE},
                }
            ]
        )
        self.assertTrue(
            image_matches_filter(
                self.image_path,
                labels=[],
                include_empty=False,
                suspect_attrs=[SUSPECT_ATTR_MODEL_FALSE],
                output_dir=self.root,
            )
        )

    def test_same_shape_and_for_label_and_suspect(self):
        self._write_json(
            [
                {
                    "label": "person",
                    "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "shape_type": "rectangle",
                    "attributes": {COMPARE_STATUS: STATUS_MODEL_FALSE},
                },
                {
                    "label": "car",
                    "points": [[20, 0], [30, 0], [30, 10], [20, 10]],
                    "shape_type": "rectangle",
                    "attributes": {COMPARE_STATUS: STATUS_MODEL_FALSE},
                },
            ]
        )
        self.assertTrue(
            image_matches_filter(
                self.image_path,
                labels=["person"],
                include_empty=False,
                suspect_attrs=[SUSPECT_ATTR_MODEL_FALSE],
                output_dir=self.root,
            )
        )
        self.assertFalse(
            image_matches_filter(
                self.image_path,
                labels=["person"],
                include_empty=False,
                suspect_attrs=[SUSPECT_ATTR_SUSPECT_FN],
                output_dir=self.root,
            )
        )

    def test_normal_suspect_attr_with_label(self):
        self._write_json(
            [
                {
                    "label": "person",
                    "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "shape_type": "rectangle",
                    "attributes": {},
                }
            ]
        )
        self.assertTrue(
            image_matches_filter(
                self.image_path,
                labels=["person"],
                include_empty=False,
                suspect_attrs=[SUSPECT_ATTR_NORMAL],
                output_dir=self.root,
            )
        )

    def test_normal_is_image_level_not_per_shape(self):
        self._write_json(
            [
                {
                    "label": "person",
                    "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "shape_type": "rectangle",
                    "attributes": {},
                },
                {
                    "label": "car",
                    "points": [[20, 0], [30, 0], [30, 10], [20, 10]],
                    "shape_type": "rectangle",
                    "attributes": {COMPARE_STATUS: STATUS_SUSPECT_FN},
                },
            ]
        )
        self.assertFalse(
            image_matches_filter(
                self.image_path,
                labels=["person"],
                include_empty=False,
                suspect_attrs=[SUSPECT_ATTR_NORMAL],
                output_dir=self.root,
            )
        )
        self.assertFalse(
            image_matches_filter(
                self.image_path,
                labels=[],
                include_empty=False,
                suspect_attrs=[SUSPECT_ATTR_NORMAL],
                output_dir=self.root,
            )
        )

    def test_empty_or_content_union(self):
        self._write_json(
            [
                {
                    "label": "person",
                    "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "shape_type": "rectangle",
                }
            ]
        )
        self.assertTrue(
            image_matches_filter(
                self.image_path,
                labels=["person"],
                include_empty=True,
                suspect_attrs=[],
                output_dir=self.root,
            )
        )
        self._write_json([])
        self.assertTrue(
            image_matches_filter(
                self.image_path,
                labels=["person"],
                include_empty=True,
                suspect_attrs=[],
                output_dir=self.root,
            )
        )
        self._write_json(
            [
                {
                    "label": "person",
                    "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "shape_type": "rectangle",
                }
            ]
        )
        self.assertFalse(
            image_matches_filter(
                self.image_path,
                labels=["car"],
                include_empty=True,
                suspect_attrs=[],
                output_dir=self.root,
            )
        )


if __name__ == "__main__":
    unittest.main()