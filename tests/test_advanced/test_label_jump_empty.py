import json
import os
import tempfile
import unittest

from anylabeling.views.labeling.advanced.compare.annotation_compare import (
    COMPARE_STATUS,
    STATUS_SUSPECT_FN,
)
from anylabeling.views.labeling.advanced.ops.label_jump import (
    EMPTY_IMAGE_JUMP_KEY,
    count_strict_empty_images,
    find_image_with_labels,
    image_matches_jump_targets,
)


class TestLabelJumpEmpty(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _img(self, name: str) -> str:
        path = os.path.join(self.root, name)
        open(path, "wb").close()
        return path

    def _json(self, stem: str, shapes):
        with open(os.path.join(self.root, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": "1.0.0",
                    "imagePath": stem + ".jpg",
                    "shapes": shapes,
                },
                f,
            )

    def test_empty_jump_among_strict_empty_only(self) -> None:
        a = self._img("a.jpg")
        b = self._img("b.jpg")
        c = self._img("c.jpg")
        self._json("a", [])
        self._json("b", [])
        self._json(
            "c",
            [
                {
                    "label": "person",
                    "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "shape_type": "rectangle",
                }
            ],
        )
        self.assertEqual(count_strict_empty_images([a, b, c], output_dir=self.root), 2)
        self.assertTrue(
            image_matches_jump_targets(a, [EMPTY_IMAGE_JUMP_KEY], output_dir=self.root)
        )
        self.assertFalse(
            image_matches_jump_targets(c, [EMPTY_IMAGE_JUMP_KEY], output_dir=self.root)
        )
        nxt = find_image_with_labels(
            [a, b, c],
            current_path=a,
            labels=[EMPTY_IMAGE_JUMP_KEY],
            output_dir=self.root,
            direction=1,
        )
        self.assertEqual(nxt, b)

    def test_suspect_only_counts_as_empty_for_jump(self) -> None:
        path = self._img("x.jpg")
        self._json(
            "x",
            [
                {
                    "label": "p",
                    "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "shape_type": "rectangle",
                    "attributes": {COMPARE_STATUS: STATUS_SUSPECT_FN},
                }
            ],
        )
        self.assertTrue(
            image_matches_jump_targets(
                path, [EMPTY_IMAGE_JUMP_KEY], output_dir=self.root
            )
        )


if __name__ == "__main__":
    unittest.main()