"""Dialog: split image filenames into train / val / test list files."""

from __future__ import annotations

import os
import os.path as osp
from typing import List, Optional, Tuple

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from anylabeling.views.labeling import utils
from anylabeling.views.labeling.advanced.ops.dataset_split_engine import (
    compute_split_stats,
    move_test_set_to_directory,
    paths_for_all_list,
    run_split,
)
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_ok_btn_style,
    get_spinbox_style,
)


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


def _label_output_dir(parent) -> Optional[str]:
    if parent is None:
        return None
    out = getattr(parent, "output_dir", None)
    if out and osp.isdir(str(out)):
        return str(out)
    return None


class _ClassStatsTable(QtWidgets.QTableWidget):
    def __init__(
        self,
        tr_func,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._tr = tr_func
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(
            [
                self._tr("Class"),
                self._tr("Images"),
                self._tr("Boxes"),
            ]
        )
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.setAlternatingRowColors(True)

    def fill(self, *, image_count: int, unlabeled: int, rows: List[Tuple[str, int, int]]) -> None:
        self.setRowCount(len(rows) + (1 if unlabeled > 0 else 0) + 1)
        summary = QtWidgets.QTableWidgetItem(
            self._tr("Total images: %d (unlabeled: %d)") % (image_count, unlabeled)
        )
        summary.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.setItem(0, 0, summary)
        self.setSpan(0, 0, 1, 3)
        row_idx = 1
        if unlabeled > 0:
            self._set_row(
                row_idx,
                self._tr("(no labels)"),
                unlabeled,
                0,
            )
            row_idx += 1
        for name, img_n, box_n in rows:
            self._set_row(row_idx, name, img_n, box_n)
            row_idx += 1

    def _set_row(self, row: int, name: str, images: int, boxes: int) -> None:
        self.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
        self.setItem(row, 1, QtWidgets.QTableWidgetItem(str(images)))
        self.setItem(row, 2, QtWidgets.QTableWidgetItem(str(boxes)))


class DatasetSplitDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self._label_dir = _label_output_dir(parent)
        self._cached_parts: Optional[Tuple[List[str], List[str], List[str]]] = None

        self.setWindowTitle(self.tr("Dataset Split"))
        self.setModal(True)
        self.setMinimumSize(620, 520)
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

        self.stratify_cb = QtWidgets.QCheckBox(self.tr("Stratify by labels"))
        self.stratify_cb.setChecked(True)
        self.stratify_cb.setToolTip(
            self.tr(
                "Balance classes across splits: each image is assigned once, "
                "but every class on that image is counted when balancing. "
                "Images without any label (no shapes/flags) are stratified "
                "with the same train/val/test ratio in a separate pass."
            )
        )

        self.shuffle_cb = QtWidgets.QCheckBox(self.tr("Shuffle before split"))
        self.shuffle_cb.setChecked(True)
        self.shuffle_cb.setToolTip(
            self.tr("Used for non-stratified splits, or as tie-break order when stratifying.")
        )
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(42)
        self.seed_spin.setStyleSheet(get_spinbox_style())

        opt_row = QtWidgets.QHBoxLayout()
        opt_row.addWidget(self.stratify_cb)
        opt_row.addWidget(self.shuffle_cb)
        opt_row.addStretch(1)
        opt_row.addWidget(QtWidgets.QLabel(self.tr("Random seed:")))
        opt_row.addWidget(self.seed_spin)

        self._on_stratify_toggled(self.stratify_cb.isChecked())
        self.stratify_cb.toggled.connect(self._on_stratify_toggled)

        self.isolate_test_cb = QtWidgets.QCheckBox(
            self.tr("Isolate test set (move to separate folder)")
        )
        self.isolate_test_cb.setChecked(False)
        self.isolate_test_cb.setToolTip(
            self.tr(
                "When enabled, images assigned to test are moved with their "
                "LabelMe JSON into the folder below. all.txt lists train+val only."
            )
        )

        self.isolate_test_hint = QtWidgets.QLabel(
            self.tr(
                "Test isolation: after the split, every test image and its "
                "annotation JSON are moved to the output folder you choose. "
                "train.txt / val.txt / test.txt are still written next to the "
                "image folder's parent; test.txt is empty (test files live in "
                "the output folder). all.txt contains only train and val images. "
                "Use this when the test set must not appear in training or "
                "full-dataset lists."
            )
        )
        self.isolate_test_hint.setWordWrap(True)
        self.isolate_test_hint.setStyleSheet("color: #666; font-size: 11px;")
        self.isolate_test_hint.setVisible(False)

        self.test_out_edit = QtWidgets.QLineEdit()
        self.test_out_edit.setPlaceholderText(
            self.tr("Folder to move test images and JSON into")
        )
        self.test_out_edit.setEnabled(False)
        test_out_browse = QtWidgets.QPushButton(self.tr("Browse…"))
        test_out_browse.setEnabled(False)
        test_out_browse.clicked.connect(self._browse_test_out)

        row_test_out = QtWidgets.QHBoxLayout()
        row_test_out.addWidget(QtWidgets.QLabel(self.tr("Test output folder:")))
        row_test_out.addWidget(self.test_out_edit, 1)
        row_test_out.addWidget(test_out_browse)
        self._test_out_browse_btn = test_out_browse

        self.isolate_test_cb.toggled.connect(self._on_isolate_test_toggled)

        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self._stats_tabs = QtWidgets.QTabWidget()
        self._table_train = _ClassStatsTable(self.tr, self)
        self._table_val = _ClassStatsTable(self.tr, self)
        self._table_test = _ClassStatsTable(self.tr, self)
        self._stats_tabs.addTab(self._table_train, self.tr("Train"))
        self._stats_tabs.addTab(self._table_val, self.tr("Val"))
        self._stats_tabs.addTab(self._table_test, self.tr("Test"))

        self._stats_hint = QtWidgets.QLabel(
            self.tr("Click Preview split to compute class counts per list (labels from JSON).")
        )
        self._stats_hint.setWordWrap(True)
        self._stats_hint.setStyleSheet("color: #666; font-size: 11px;")

        btn_preview = QtWidgets.QPushButton(self.tr("Preview split"))
        btn_preview.clicked.connect(self._on_preview)
        btn_ok = QtWidgets.QPushButton(self.tr("Generate"))
        btn_ok.setStyleSheet(get_ok_btn_style())
        btn_ok.clicked.connect(self._on_generate)
        btn_cancel = QtWidgets.QPushButton(self.tr("Cancel"))
        btn_cancel.setStyleSheet(get_cancel_btn_style())
        btn_cancel.clicked.connect(self.reject)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(btn_preview)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)

        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(12)
        main.addWidget(QtWidgets.QLabel(self.tr("Image folder:")))
        main.addLayout(row_folder)
        main.addLayout(ratio_form)
        main.addLayout(opt_row)
        main.addWidget(self.isolate_test_cb)
        main.addWidget(self.isolate_test_hint)
        main.addLayout(row_test_out)
        main.addWidget(self.preview_label)
        main.addWidget(QtWidgets.QLabel(self.tr("Class distribution:")))
        main.addWidget(self._stats_hint)
        main.addWidget(self._stats_tabs, 1)
        main.addLayout(btn_row)

        init_dir = _default_image_folder(parent) if parent else ""
        if init_dir:
            self.folder_edit.setText(init_dir)
        self._refresh_output_hint()

        self.folder_edit.textChanged.connect(lambda _: self._refresh_output_hint())
        self.train_spin.valueChanged.connect(lambda _: self._invalidate_preview())
        self.val_spin.valueChanged.connect(lambda _: self._invalidate_preview())
        self.test_spin.valueChanged.connect(lambda _: self._invalidate_preview())
        self.stratify_cb.toggled.connect(lambda _: self._invalidate_preview())
        self.shuffle_cb.toggled.connect(lambda _: self._invalidate_preview())
        self.seed_spin.valueChanged.connect(lambda _: self._invalidate_preview())
        self.isolate_test_cb.toggled.connect(lambda _: self._invalidate_preview())
        self.test_out_edit.textChanged.connect(lambda _: self._refresh_output_hint())

    def _on_isolate_test_toggled(self, on: bool) -> None:
        self.isolate_test_hint.setVisible(bool(on))
        self.test_out_edit.setEnabled(bool(on))
        self._test_out_browse_btn.setEnabled(bool(on))
        self._refresh_output_hint()

    def _browse_test_out(self) -> None:
        start = self.test_out_edit.text().strip()
        if not start or not osp.isdir(start):
            start = self.folder_edit.text().strip() or "."
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select test set output folder"),
            start,
            QtWidgets.QFileDialog.Option.ShowDirsOnly
            | QtWidgets.QFileDialog.Option.DontResolveSymlinks,
        )
        if path:
            self.test_out_edit.setText(path)

    def _on_stratify_toggled(self, on: bool) -> None:
        self.shuffle_cb.setEnabled(not on)
        if on:
            self.shuffle_cb.setToolTip(
                self.tr("Stratified mode uses the random seed for assignment order only.")
            )
        else:
            self.shuffle_cb.setToolTip(self.tr("Shuffle the image list before proportional split."))

    def _invalidate_preview(self) -> None:
        self._cached_parts = None

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

    def _refresh_output_hint(self):
        folder = self.folder_edit.text().strip()
        if not folder:
            self.preview_label.setText(
                self.tr("List files will be written next to this folder (parent directory).")
            )
            return
        parent_dir = osp.dirname(osp.normpath(osp.abspath(folder)))
        if self.isolate_test_cb.isChecked():
            test_dest = self.test_out_edit.text().strip() or self.tr("(not set)")
            self.preview_label.setText(
                self.tr(
                    "Will write train.txt, val.txt, test.txt, all.txt to:\n%s\n\n"
                    "Test isolation on: test files move to %s; all.txt = train + val only."
                )
                % (parent_dir, test_dest)
            )
            return
        self.preview_label.setText(
            self.tr("Will write train.txt, val.txt, test.txt, all.txt to:\n%s")
            % parent_dir
        )

    def _test_isolation_enabled(self) -> bool:
        return bool(self.isolate_test_cb.isChecked())

    def _validate_test_output_dir(self) -> Optional[str]:
        if not self._test_isolation_enabled():
            return None
        dest = self.test_out_edit.text().strip()
        if not dest:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Dataset Split"),
                self.tr("Please choose a folder for the isolated test set."),
            )
            return None
        folder = self.folder_edit.text().strip()
        try:
            dest_abs = osp.normpath(osp.abspath(dest))
            folder_abs = osp.normpath(osp.abspath(folder)) if folder else ""
        except Exception:
            dest_abs, folder_abs = dest, folder
        if folder_abs and dest_abs == folder_abs:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Dataset Split"),
                self.tr(
                    "The test output folder must not be the same as the image folder."
                ),
            )
            return None
        try:
            os.makedirs(dest_abs, exist_ok=True)
        except OSError as e:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Dataset Split"),
                self.tr("Cannot create test output folder:\n%s") % e,
            )
            return None
        return dest_abs

    def _weights(self) -> Tuple[int, int, int] | None:
        w_tr = int(self.train_spin.value())
        w_v = int(self.val_spin.value())
        w_te = int(self.test_spin.value())
        if w_tr + w_v + w_te <= 0:
            return None
        return w_tr, w_v, w_te

    def _collect_paths(self) -> List[str]:
        folder = self.folder_edit.text().strip()
        if not folder or not osp.isdir(folder):
            return []
        return list(utils.scan_all_images(folder))

    def _compute_parts(self) -> Optional[Tuple[List[str], List[str], List[str], List[str]]]:
        weights = self._weights()
        if weights is None:
            return None
        paths = self._collect_paths()
        if not paths:
            return None
        w_tr, w_v, w_te = weights
        part_train, part_val, part_test = run_split(
            paths,
            w_train=w_tr,
            w_val=w_v,
            w_test=w_te,
            stratify=bool(self.stratify_cb.isChecked()),
            shuffle=bool(self.shuffle_cb.isChecked()),
            seed=int(self.seed_spin.value()),
            output_dir=self._label_dir,
        )
        return part_train, part_val, part_test, paths

    def _update_stats_tables(
        self,
        part_train: List[str],
        part_val: List[str],
        part_test: List[str],
    ) -> None:
        out = self._label_dir

        def _fill(table: _ClassStatsTable, paths: List[str]) -> None:
            st = compute_split_stats(paths, output_dir=out)
            rows = st.sorted_rows()
            table.fill(
                image_count=st.image_count,
                unlabeled=st.unlabeled_images,
                rows=rows,
            )

        _fill(self._table_train, part_train)
        _fill(self._table_val, part_val)
        _fill(self._table_test, part_test)

    def _on_preview(self) -> None:
        folder = self.folder_edit.text().strip()
        if not folder or not osp.isdir(folder):
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Dataset Split"),
                self.tr("Please choose a valid image folder."),
            )
            return
        weights = self._weights()
        if weights is None:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Dataset Split"),
                self.tr("The sum of train / val / test weights must be greater than zero."),
            )
            return
        paths = self._collect_paths()
        if not paths:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Dataset Split"),
                self.tr("No images found under this folder."),
            )
            return
        computed = self._compute_parts()
        if computed is None:
            return
        part_train, part_val, part_test, all_paths = computed
        self._cached_parts = (part_train, part_val, part_test)
        self._update_stats_tables(part_train, part_val, part_test)
        all_n = len(
            paths_for_all_list(
                all_paths,
                part_train=part_train,
                part_val=part_val,
                part_test=part_test,
                exclude_test_from_all=self._test_isolation_enabled(),
            )
        )
        extra = ""
        if self._test_isolation_enabled():
            extra = " " + self.tr("(all.txt would list %d train+val images.)") % all_n
        self._stats_hint.setText(
            (
                self.tr(
                    "Preview: %d images → train %d, val %d, test %d. "
                    "Images = files containing the class; Boxes = shape count."
                )
                % (
                    len(all_paths),
                    len(part_train),
                    len(part_val),
                    len(part_test),
                )
            )
            + extra
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

        computed = self._compute_parts()
        if computed is None:
            weights = self._weights()
            if weights is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Dataset Split"),
                    self.tr("The sum of train / val / test weights must be greater than zero."),
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Dataset Split"),
                    self.tr("No images found under this folder."),
                )
            return

        part_train, part_val, part_test, paths = computed
        self._cached_parts = (part_train, part_val, part_test)
        self._update_stats_tables(part_train, part_val, part_test)

        test_dest = self._validate_test_output_dir()
        if self._test_isolation_enabled() and test_dest is None:
            return

        isolate = self._test_isolation_enabled()
        if isolate and part_test:
            ret = QtWidgets.QMessageBox.question(
                self,
                self.tr("Move test set?"),
                self.tr(
                    "Move %d test image(s) and their JSON labels to:\n%s\n\nContinue?"
                )
                % (len(part_test), test_dest),
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if ret != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        moved_test = 0
        move_errors: List[str] = []
        if isolate and part_test and test_dest:
            moved_test, move_errors = move_test_set_to_directory(
                part_test,
                test_dest,
                label_output_dir=self._label_dir,
            )
            if move_errors and moved_test == 0:
                QtWidgets.QMessageBox.critical(
                    self,
                    self.tr("Dataset Split"),
                    self.tr("Failed to move test set:\n%s")
                    % "\n".join(move_errors[:20]),
                )
                return

        all_list_paths = paths_for_all_list(
            paths,
            part_train=part_train,
            part_val=part_val,
            part_test=part_test,
            exclude_test_from_all=isolate,
        )

        out_dir = osp.dirname(osp.normpath(osp.abspath(folder)))
        test_txt_paths: List[str] = [] if isolate else list(part_test)
        targets = {
            "train.txt": part_train,
            "val.txt": part_val,
            "test.txt": test_txt_paths,
            "all.txt": all_list_paths,
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

        if isolate:
            done_msg = self.tr(
                "Done. %d images → train %d, val %d, test %d; all.txt has %d "
                "(train+val only).\nMoved %d test file(s) to:\n%s\nList output: %s"
            ) % (
                len(paths),
                len(part_train),
                len(part_val),
                len(part_test),
                len(all_list_paths),
                moved_test,
                test_dest or "",
                out_dir,
            )
        else:
            done_msg = self.tr(
                "Done. %d images → train %d, val %d, test %d; all.txt has %d.\nOutput: %s"
            ) % (
                len(paths),
                len(part_train),
                len(part_val),
                len(part_test),
                len(all_list_paths),
                out_dir,
            )
        if move_errors:
            done_msg += "\n\n" + self.tr("Some test files could not be moved:") + "\n"
            done_msg += "\n".join(move_errors[:15])
            if len(move_errors) > 15:
                done_msg += "\n…"
        QtWidgets.QMessageBox.information(
            self,
            self.tr("Dataset Split"),
            done_msg,
        )
        # Keep dialog open so user can read class stats tabs
        self._stats_hint.setText(
            self.tr("List files written. Review class distribution in the tabs above.")
        )