import unittest

import numpy as np

from anylabeling.views.labeling.advanced.dedup.config import Config
from anylabeling.views.labeling.advanced.dedup.instance import judge_instance
from anylabeling.views.labeling.advanced.dedup.io_utils import Box, ImageRecord
from anylabeling.views.labeling.advanced.dedup.verify import VerifyResult


def _rec(idx: int, boxes: list[Box]) -> ImageRecord:
    return ImageRecord(index=idx, image_path=f"{idx}.jpg", json_path=None, boxes=boxes)


def _box(label: str = "person") -> Box:
    return Box(label=label, score=1.0, xyxy=(0.1, 0.1, 0.5, 0.5))


class TestDedupG3Boxes(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()
        self.cfg.feature.box_label_filter = ("person",)
        self.rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        self.verify = VerifyResult(a=0, b=1, passed=True)

    def test_both_without_boxes_passes_g3(self) -> None:
        r = judge_instance(
            _rec(0, []), _rec(1, []), self.verify, self.rgb, self.rgb, self.cfg
        )
        self.assertTrue(r.passed)
        self.assertEqual(r.reason, "no_boxes_both_pass")

    def test_one_side_with_boxes_rejected(self) -> None:
        r = judge_instance(
            _rec(0, []), _rec(1, [_box()]), self.verify, self.rgb, self.rgb, self.cfg
        )
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "instance_presence_mismatch")

    def test_filtered_boxes_only_count_selected_labels(self) -> None:
        r = judge_instance(
            _rec(0, [_box("car")]),
            _rec(1, [_box("car")]),
            self.verify,
            self.rgb,
            self.rgb,
            self.cfg,
        )
        self.assertTrue(r.passed)
        self.assertEqual(r.reason, "no_boxes_both_pass")


if __name__ == "__main__":
    unittest.main()