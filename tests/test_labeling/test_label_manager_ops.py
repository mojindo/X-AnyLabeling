import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from anylabeling.views.labeling.label_widget import LABEL_OPACITY, LabelingWidget
from anylabeling.views.labeling.widgets.label_dialog import LabelModifyDialog
from anylabeling.views.labeling.widgets.unique_label_qlist_widget import (
    UniqueLabelQListWidget,
)


class _FakeLabelingParent(QtWidgets.QWidget):
    def __init__(self, image_files, output_dir=None):
        super().__init__()
        self.file_list_widget = QtWidgets.QListWidget()
        for path in image_files:
            self.file_list_widget.addItem(path)
        self.unique_label_list = UniqueLabelQListWidget()
        self.label_info = {}
        self.output_dir = output_dir
        self.filename = None
        self.label_dialog = Mock()
        self._get_rgb_by_label = Mock(return_value=(0, 255, 0))
        self.apply_label_visibility = Mock()
        self.refresh_panel_controls = Mock()
        self.load_file = Mock()

        # Bind the real helpers under test onto this lightweight host.
        self._ensure_label_info_entry = (
            LabelingWidget._ensure_label_info_entry.__get__(self)
        )
        self._image_file_list = LabelingWidget._image_file_list.__get__(self)
        self._reset_label_info_ops = (
            LabelingWidget._reset_label_info_ops.__get__(self)
        )
        self.apply_unique_label_dataset_op = (
            LabelingWidget.apply_unique_label_dataset_op.__get__(self)
        )
        self.delete_unique_label = (
            LabelingWidget.delete_unique_label.__get__(self)
        )
        self.merge_unique_label = (
            LabelingWidget.merge_unique_label.__get__(self)
        )
        self.rename_unique_label = (
            LabelingWidget.rename_unique_label.__get__(self)
        )
        self.tr = lambda text: text


class TestLabelsPanelOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.image_files = []
        self.paths = []
        for i in range(2):
            image = Path(self.temp.name) / f"{i}.png"
            image.write_bytes(b"")
            self.image_files.append(str(image))
            path = image.with_suffix(".json")
            path.write_text(
                json.dumps(
                    {
                        "shapes": [
                            {"label": "cat", "points": [[0, 0], [1, 1]]},
                            {"label": "dog", "points": [[2, 2], [3, 3]]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.paths.append(path)

        self.parent = _FakeLabelingParent(self.image_files)
        for label in ("cat", "dog"):
            item = self.parent.unique_label_list.create_item_from_label(label)
            self.parent.unique_label_list.addItem(item)
            self.parent.unique_label_list.set_item_label(
                item, label, (0, 255, 0), LABEL_OPACITY
            )
            self.parent._ensure_label_info_entry(label)

    def tearDown(self):
        self.parent.close()
        self.temp.cleanup()

    def _labels_in_files(self):
        labels = []
        for path in self.paths:
            shapes = json.loads(path.read_text(encoding="utf-8"))["shapes"]
            labels.append([shape["label"] for shape in shapes])
        return labels

    def test_context_menu_actions_emit_signals(self):
        received = {"delete": [], "merge": [], "rename": [], "lock": []}
        widget = self.parent.unique_label_list
        widget.label_delete_requested.connect(received["delete"].append)
        widget.label_merge_requested.connect(received["merge"].append)
        widget.label_rename_requested.connect(received["rename"].append)
        widget.items_lock_requested.connect(
            lambda items: received["lock"].append(
                [i.data(Qt.ItemDataRole.UserRole) for i in items]
            )
        )
        item = widget.find_items_by_label("cat")[0]
        chosen_text = {"value": None}

        def choose_exec(menu, *_args, **_kwargs):
            for action in menu.actions():
                if action.text() == chosen_text["value"]:
                    return action
            return None

        with patch.object(
            widget, "itemAt", return_value=item
        ), patch.object(
            widget, "selectedItems", return_value=[item]
        ), patch.object(QtWidgets.QMenu, "exec", choose_exec):
            for text, key in (
                (widget.tr("Delete Label…"), "delete"),
                (widget.tr("Merge into Existing Label…"), "merge"),
                (widget.tr("Rename Label…"), "rename"),
                (widget.tr("Toggle Lock"), "lock"),
            ):
                chosen_text["value"] = text
                widget._show_context_menu(QtCore.QPoint(5, 5))
                if key == "lock":
                    self.assertEqual(received[key][-1], ["cat"])
                else:
                    self.assertEqual(received[key][-1], "cat")

    def test_delete_label_removes_annotations(self):
        with patch(
            "anylabeling.views.labeling.label_widget.Popup"
        ), patch.object(
            QMessageBox,
            "exec",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            self.parent.delete_unique_label("cat")
        self.assertEqual(self._labels_in_files(), [["dog"], ["dog"]])
        self.assertNotIn("cat", self.parent.label_info)
        self.assertFalse(
            self.parent.unique_label_list.find_items_by_label("cat")
        )

    def test_merge_label_into_existing(self):
        with patch(
            "anylabeling.views.labeling.label_widget.Popup"
        ), patch(
            "PyQt6.QtWidgets.QInputDialog.getItem",
            return_value=("dog", True),
        ), patch.object(
            QMessageBox,
            "exec",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            self.parent.merge_unique_label("cat")
        self.assertEqual(
            self._labels_in_files(),
            [["dog", "dog"], ["dog", "dog"]],
        )
        self.assertNotIn("cat", self.parent.label_info)
        self.assertIn("dog", self.parent.label_info)

    def test_rename_label_updates_annotations(self):
        with patch(
            "anylabeling.views.labeling.label_widget.Popup"
        ), patch(
            "PyQt6.QtWidgets.QInputDialog.getText",
            return_value=("kitten", True),
        ):
            self.parent.rename_unique_label("cat")
        self.assertEqual(
            self._labels_in_files(),
            [["kitten", "dog"], ["kitten", "dog"]],
        )
        self.assertNotIn("cat", self.parent.label_info)
        self.assertIn("kitten", self.parent.label_info)
        self.assertTrue(
            self.parent.unique_label_list.find_items_by_label("kitten")
        )

    def test_rename_to_existing_label_offers_merge(self):
        with patch(
            "anylabeling.views.labeling.label_widget.Popup"
        ), patch(
            "PyQt6.QtWidgets.QInputDialog.getText",
            return_value=("dog", True),
        ), patch.object(
            QMessageBox,
            "exec",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            self.parent.rename_unique_label("cat")
        self.assertEqual(
            self._labels_in_files(),
            [["dog", "dog"], ["dog", "dog"]],
        )
        self.assertNotIn("cat", self.parent.label_info)

    def test_rewrite_helper_shared_with_label_manager(self):
        label_info = {
            "cat": {
                "delete": False,
                "value": "bird",
                "color": [0, 255, 0],
                "opacity": 128,
                "visible": True,
            },
            "dog": {
                "delete": False,
                "value": None,
                "color": [0, 255, 0],
                "opacity": 128,
                "visible": True,
            },
        }
        ok = LabelModifyDialog.rewrite_annotation_labels(
            self.image_files,
            None,
            label_info,
            {},
        )
        self.assertTrue(ok)
        self.assertEqual(
            self._labels_in_files(),
            [["bird", "dog"], ["bird", "dog"]],
        )


if __name__ == "__main__":
    unittest.main()
