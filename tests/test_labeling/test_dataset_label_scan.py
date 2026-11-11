import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets

from anylabeling.views.labeling.label_widget import LABEL_OPACITY, LabelingWidget
from anylabeling.views.labeling.widgets.unique_label_qlist_widget import (
    UniqueLabelQListWidget,
)


class _Host(QtWidgets.QWidget):
    def __init__(self, image_files, output_dir=None, config_labels=None):
        super().__init__()
        self.file_list_widget = QtWidgets.QListWidget()
        for path in image_files:
            self.file_list_widget.addItem(path)
        self.unique_label_list = UniqueLabelQListWidget()
        self.label_info = {"old": {"delete": False, "value": None}}
        self.output_dir = output_dir
        self._config = {"labels": config_labels or []}
        self.label_dialog = Mock()
        self.label_dialog.label_list = QtWidgets.QListWidget()
        self.label_dialog._last_label = "old"
        self.label_dialog.add_label_history = Mock()
        self._get_rgb_by_label = Mock(return_value=(1, 2, 3))
        self.refresh_panel_controls = Mock()
        self.apply_label_visibility = Mock()

        self.reset_session_labels = (
            LabelingWidget.reset_session_labels.__get__(self)
        )
        self.collect_labels_from_annotations = (
            LabelingWidget.collect_labels_from_annotations.__get__(self)
        )
        self._register_session_label = (
            LabelingWidget._register_session_label.__get__(self)
        )
        self.load_labels = LabelingWidget.load_labels.__get__(self)
        self.load_dataset_labels = (
            LabelingWidget.load_dataset_labels.__get__(self)
        )

    @property
    def image_list(self):
        return [
            self.file_list_widget.item(i).text()
            for i in range(self.file_list_widget.count())
        ]


class TestDatasetLabelScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.image_files = []
        for i, labels in enumerate(
            (["cat", "dog"], ["dog", "bird"], ["cat"])
        ):
            image = Path(self.temp.name) / f"{i}.jpg"
            image.write_bytes(b"")
            self.image_files.append(str(image))
            image.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "shapes": [
                            {"label": label, "points": [[0, 0], [1, 1]]}
                            for label in labels
                        ]
                    }
                ),
                encoding="utf-8",
            )

    def tearDown(self):
        self.temp.cleanup()

    def test_load_dataset_labels_resets_and_scans_all(self):
        host = _Host(self.image_files, config_labels=["person", "preset"])
        # Pretend previous dataset / config labels still exist.
        for name in ("old", "person"):
            item = host.unique_label_list.create_item_from_label(name)
            host.unique_label_list.addItem(item)
        host.unique_label_list.lock_states["old"] = True
        host.label_info["person"] = {"delete": False, "value": None}

        host.load_dataset_labels(self.image_files)

        labels = host.unique_label_list.all_labels()
        # Only labels found in current annotations; config presets are ignored.
        self.assertEqual(labels, ["bird", "cat", "dog"])
        self.assertNotIn("old", labels)
        self.assertNotIn("person", labels)
        self.assertNotIn("preset", labels)
        self.assertNotIn("old", host.label_info)
        self.assertNotIn("person", host.label_info)
        self.assertEqual(host.unique_label_list.lock_states, {})
        for name in ("bird", "cat", "dog"):
            self.assertIn(name, host.label_info)
        host.refresh_panel_controls.assert_called()
        host.apply_label_visibility.assert_called()

    def test_collect_labels_uses_output_dir(self):
        out = Path(self.temp.name) / "labels"
        out.mkdir()
        image = Path(self.temp.name) / "only.png"
        image.write_bytes(b"")
        (out / "only.json").write_text(
            json.dumps({"shapes": [{"label": "truck"}]}),
            encoding="utf-8",
        )
        host = _Host([str(image)], output_dir=str(out))
        self.assertEqual(
            host.collect_labels_from_annotations([str(image)]),
            ["truck"],
        )


if __name__ == "__main__":
    unittest.main()
