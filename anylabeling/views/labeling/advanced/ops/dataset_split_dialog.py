"""Dialog: split image filenames into train / val / test list files."""

from __future__ import annotations

import os
import os.path as osp
import random

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from anylabeling.views.labeling import utils
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_ok_btn_style,
    get_spinbox_style,
)


def _split_sizes(n: int, w_train: int, w_val: int, w_test: int) -> tuple[int, int, int]:
    total_w = w_train + w_val + w_test
    if total_w <= 0:
        raise ValueError("invalid weights")
    n_train = n * w_train // total_w
    n_val = n * w_val // total_w
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def open_dataset_split_dialog(parent):
    """Open dataset split dialog; `parent` is LabelingWidget."""
    dlg = DatasetSplitDialog(parent)
    dlg.exec()


def _default_image_folder(parent) -> str:
    fn = getattr(parent, "filename", None)
    if fn and osp.isfile(fn):
        return osp.dirname(osp.abspath(fn))
    lod = getattr(parent, "last_open_dir", None)
    if lod and osp.isdir(lod):
        return lod
    return ""


class DatasetSplitDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Dataset Split"))
        self.setModal(True)
        self.setMinimumWidth(520)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        )
        self.setStyleSheet(get_dialog_style())

        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setPlaceholderText(
            self.tr("Folder containing images (scanned recursively)")
        )
        browse_btn = QtWidgets.QPushButton(self.tr("Browse…"))
        browse_btn.clicked.connect(self._browse_folder)

        row_folder = QtWidgets.QHBoxLayout()
        row_folder.addWidget(self.folder_edit, 1)
        row_folder.addWidget(browse_btn)

        self.train_spin = QtWidgets.QSpinBox()
        self.val_spin = QtWidgets.QSpinBox()
        self.test_spin = QtWidgets.QSpinBox()
        for sb in (self.train_spin, self.val_spin, self.test_spin):
            sb.setRange(0, 1_000_000)
            sb.setStyleSheet(get_spinbox_style())
        self.train_spin.setValue(7)
        self.val_spin.setValue(2)
        self.test_spin.setValue(1)

        ratio_form = QtWidgets.QFormLayout()
        ratio_form.addRow(self.tr("Train weight:"), self.train_spin)
        ratio_form.addRow(self.tr("Val weight:"), self.val_spin)
        ratio_form.addRow(self.tr("Test weight:"), self.test_spin)

        self.shuffle_cb = QtWidgets.QCheckBox(self.tr("Shuffle before split"))
        self.shuffle_cb.setChecked(True)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(42)
        self.seed_spin.setStyleSheet(get_spinbox_style())
        self.seed_spin.setEnabled(self.shuffle_cb.isChecked())
        self.shuffle_cb.toggled.connect(self.seed_spin.setEnabled)

        seed_row = QtWidgets.QHBoxLayout()
        seed_row.addWidget(self.shuffle_cb)
        seed_row.addStretch(1)
        seed_row.addWidget(QtWidgets.QLabel(self.tr("Random seed:")))
        seed_row.addWidget(self.seed_spin)

        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        btn_ok = QtWidgets.QPushButton(self.tr("Generate"))
        btn_ok.setStyleSheet(get_ok_btn_style())
        btn_ok.clicked.connect(self._on_generate)
        btn_cancel = QtWidgets.QPushButton(self.tr("Cancel"))
        btn_cancel.setStyleSheet(get_cancel_btn_style())
        btn_cancel.clicked.connect(self.reject)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)

        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(12)
        main.addWidget(QtWidgets.QLabel(self.tr("Image folder:")))
        main.addLayout(row_folder)
        main.addLayout(ratio_form)
        main.addLayout(seed_row)
        main.addWidget(self.preview_label)
        main.addLayout(btn_row)

        init_dir = _default_image_folder(parent) if parent else ""
        if init_dir:
            self.folder_edit.setText(init_dir)
        self._refresh_preview()

        self.folder_edit.textChanged.connect(lambda _: self._refresh_preview())
        self.train_spin.valueChanged.connect(lambda _: self._refresh_preview())
        self.val_spin.valueChanged.connect(lambda _: self._refresh_preview())
        self.test_spin.valueChanged.connect(lambda _: self._refresh_preview())

    def _browse_folder(self):
        start = self.folder_edit.text().strip()
        if not start or not osp.isdir(start):
            start = self.folder_edit.text() or "."
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Image Folder"),
            start,
            QtWidgets.QFileDialog.Option.ShowDirsOnly
            | QtWidgets.QFileDialog.Option.DontResolveSymlinks,
        )
        if path:
            self.folder_edit.setText(path)

    def _refresh_preview(self):
        folder = self.folder_edit.text().strip()
        if not folder:
            self.preview_label.setText(
                self.tr("List files will be written next to this folder (parent directory).")
            )
            return
        parent_dir = osp.dirname(osp.normpath(osp.abspath(folder)))
        self.preview_label.setText(
            self.tr("Will write train.txt, val.txt, test.txt, all.txt to:\n%s")
            % parent_dir
        )

    def _on_generate(self):
        folder = self.folder_edit.text().strip()
        if not folder or not osp.isdir(folder):
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Dataset Split"),
                self.tr("Please choose a valid image folder."),
            )
            return

        w_tr = int(self.train_spin.value())
        w_v = int(self.val_spin.value())
        w_te = int(self.test_spin.value())
        if w_tr + w_v + w_te <= 0:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Dataset Split"),
                self.tr("The sum of train / val / test weights must be greater than zero."),
            )
            return

        paths = utils.scan_all_images(folder)
        if not paths:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Dataset Split"),
                self.tr("No images found under this folder."),
            )
            return

        if self.shuffle_cb.isChecked():
            rng = random.Random(int(self.seed_spin.value()))
            rng.shuffle(paths)

        n_tr, n_v, n_te = _split_sizes(len(paths), w_tr, w_v, w_te)
        i0 = 0
        part_train = paths[i0 : i0 + n_tr]
        i0 += n_tr
        part_val = paths[i0 : i0 + n_v]
        i0 += n_v
        part_test = paths[i0 : i0 + n_te]

        out_dir = osp.dirname(osp.normpath(osp.abspath(folder)))
        targets = {
            "train.txt": part_train,
            "val.txt": part_val,
            "test.txt": part_test,
            "all.txt": paths,
        }
        existing = [name for name in targets if osp.isfile(osp.join(out_dir, name))]
        if existing:
            ret = QtWidgets.QMessageBox.question(
                self,
                self.tr("Overwrite?"),
                self.tr("These files already exist and will be replaced:\n%s")
                % "\n".join(existing),
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if ret != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        try:
            for fname, rows in targets.items():
                fpath = osp.join(out_dir, fname)
                with open(fpath, "w", encoding="utf-8", newline="\n") as fp:
                    for p in rows:
                        fp.write(osp.basename(p) + "\n")
        except OSError as e:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Dataset Split"),
                self.tr("Failed to write list files:\n%s") % e,
            )
            return

        QtWidgets.QMessageBox.information(
            self,
            self.tr("Dataset Split"),
            self.tr(
                "Done. %d images → train %d, val %d, test %d; all.txt has %d.\nOutput: %s"
            )
            % (
                len(paths),
                len(part_train),
                len(part_val),
                len(part_test),
                len(paths),
                out_dir,
            ),
        )
        self.accept()
