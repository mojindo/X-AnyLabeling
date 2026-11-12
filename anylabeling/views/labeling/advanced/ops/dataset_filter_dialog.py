"""Advanced dialog: filter dataset by labels / empty images, then copy or move."""

from __future__ import annotations

import os
import os.path as osp
from typing import List, Optional, Sequence

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from anylabeling.views.labeling.advanced.ops.dataset_filter import (
    filter_matching_images,
    transfer_image_and_label,
)
from anylabeling.views.labeling.advanced.ops.label_jump import (
    collect_available_labels_with_counts,
    fuzzy_match_label,
)
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_lineedit_style,
    get_ok_btn_style,
    get_progress_dialog_style,
)


class _DatasetFilterWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(bool, str, int, int, list)

    def __init__(
        self,
        *,
        image_paths: Sequence[str],
        target_dir: str,
        mode: str,
        output_dir: Optional[str],
    ) -> None:
        super().__init__()
        self._image_paths = [str(p) for p in (image_paths or []) if p]
        self._target_dir = str(target_dir or "")
        self._mode = str(mode or "copy")
        self._output_dir = output_dir
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        ok_count = 0
        fail_count = 0
        transferred: List[str] = []
        try:
            total = len(self._image_paths)
            for i, img_path in enumerate(self._image_paths):
                if self._cancel:
                    break
                self.progress.emit(i + 1, total, img_path)
                success, _msg = transfer_image_and_label(
                    img_path,
                    target_dir=self._target_dir,
                    mode=self._mode,
                    output_dir=self._output_dir,
                )
                if success:
                    ok_count += 1
                    transferred.append(img_path)
                else:
                    fail_count += 1
            self.finished.emit(True, "", ok_count, fail_count, transferred)
        except Exception as e:
            self.finished.emit(False, str(e), ok_count, fail_count, transferred)


class DatasetFilterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Dataset Filter"))
        self.setMinimumWidth(560)
        self.setMinimumHeight(520)
        self.setStyleSheet(get_dialog_style())

        self._parent = parent
        self._worker: Optional[_DatasetFilterWorker] = None
        self._progress: Optional[QtWidgets.QProgressDialog] = None

        self._build_ui()
        self.refresh_labels()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        tip = QLabel(
            self.tr(
                "Select labels and/or empty images. Matching images and their "
                "annotation JSON will be copied or moved to the output folder."
            )
        )
        tip.setWordWrap(True)
        root.addWidget(tip)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel(self.tr("Search:")))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(
            self.tr("Fuzzy filter labels…")
        )
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setStyleSheet(get_lineedit_style())
        self.search_edit.textChanged.connect(self._apply_label_filter)
        search_row.addWidget(self.search_edit, 1)
        root.addLayout(search_row)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.list_widget.setUniformItemSizes(True)
        root.addWidget(self.list_widget, 1)

        select_row = QHBoxLayout()
        self.select_all_btn = QPushButton(self.tr("Select All"))
        self.select_all_btn.setStyleSheet(get_cancel_btn_style())
        self.select_all_btn.clicked.connect(self._select_all)
        self.clear_btn = QPushButton(self.tr("Clear"))
        self.clear_btn.setStyleSheet(get_cancel_btn_style())
        self.clear_btn.clicked.connect(self._clear_selection)
        self.refresh_btn = QPushButton(self.tr("Refresh Labels"))
        self.refresh_btn.setStyleSheet(get_cancel_btn_style())
        self.refresh_btn.clicked.connect(self.refresh_labels)
        select_row.addWidget(self.select_all_btn)
        select_row.addWidget(self.clear_btn)
        select_row.addWidget(self.refresh_btn)
        select_row.addStretch(1)
        root.addLayout(select_row)

        self.cb_empty = QCheckBox(self.tr("Include empty images (no annotations)"))
        root.addWidget(self.cb_empty)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel(self.tr("Output folder:")))
        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText(
            self.tr("Select a folder to copy/move files into")
        )
        self.ed_out.setClearButtonEnabled(True)
        self.ed_out.setStyleSheet(get_lineedit_style())
        btn_browse = QPushButton(self.tr("Browse…"))
        btn_browse.setStyleSheet(get_cancel_btn_style())
        btn_browse.clicked.connect(self._browse_out)
        out_row.addWidget(self.ed_out, 1)
        out_row.addWidget(btn_browse)
        root.addLayout(out_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel(self.tr("Mode:")))
        self.rb_copy = QRadioButton(self.tr("Copy"))
        self.rb_move = QRadioButton(self.tr("Move"))
        self.rb_copy.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.rb_copy)
        self._mode_group.addButton(self.rb_move)
        mode_row.addWidget(self.rb_copy)
        mode_row.addWidget(self.rb_move)
        mode_row.addStretch(1)
        root.addLayout(mode_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_close = QPushButton(self.tr("Close"))
        self.btn_close.setStyleSheet(get_cancel_btn_style())
        self.btn_close.clicked.connect(self.reject)
        self.btn_run = QPushButton(self.tr("Run"))
        self.btn_run.setStyleSheet(get_ok_btn_style())
        self.btn_run.clicked.connect(self._run)
        btns.addWidget(self.btn_close)
        btns.addWidget(self.btn_run)
        root.addLayout(btns)

    def _image_list(self) -> List[str]:
        parent = self._parent
        if parent is None:
            return []
        return [str(p) for p in (getattr(parent, "image_list", None) or []) if p]

    def _output_dir(self) -> Optional[str]:
        parent = self._parent
        if parent is None:
            return None
        value = getattr(parent, "output_dir", None)
        return str(value) if value else None

    def refresh_labels(self) -> None:
        seed = []
        parent = self._parent
        if parent is not None:
            seed = list(getattr(parent, "label_hist", None) or [])
            uniq = getattr(parent, "unique_label_list", None)
            if uniq is not None and hasattr(uniq, "get_labels"):
                try:
                    seed = list(uniq.get_labels()) or seed
                except Exception:
                    pass
        items = collect_available_labels_with_counts(
            self._image_list(),
            output_dir=self._output_dir(),
            seed_labels=seed,
        )
        self.list_widget.clear()
        for name, count in items:
            text = f"{name} ({count})" if count else name
            item = QListWidgetItem(text)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.list_widget.addItem(item)
        self._apply_label_filter()
        self.status_label.setText(
            self.tr("Loaded %d label(s) from %d image(s).")
            % (self.list_widget.count(), len(self._image_list()))
        )

    def _apply_label_filter(self, _text: str = "") -> None:
        query = self.search_edit.text()
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            name = item.data(Qt.ItemDataRole.UserRole) or item.text()
            item.setHidden(not fuzzy_match_label(str(name), query))

    def selected_labels(self) -> List[str]:
        out: List[str] = []
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                name = item.data(Qt.ItemDataRole.UserRole) or item.text()
                text = str(name).strip()
                if text:
                    out.append(text)
        return out

    def _select_all(self) -> None:
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if not item.isHidden():
                item.setCheckState(Qt.CheckState.Checked)

    def _clear_selection(self) -> None:
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if not item.isHidden():
                item.setCheckState(Qt.CheckState.Unchecked)

    def _browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select output folder"),
            self.ed_out.text() or "",
            QFileDialog.Option.ShowDirsOnly
            | QFileDialog.Option.DontResolveSymlinks,
        )
        if path:
            self.ed_out.setText(path)

    def _run(self) -> None:
        images = self._image_list()
        if not images:
            QMessageBox.information(
                self,
                self.tr("Dataset Filter"),
                self.tr("No images loaded."),
            )
            return

        labels = self.selected_labels()
        include_empty = bool(self.cb_empty.isChecked())
        if not labels and not include_empty:
            QMessageBox.warning(
                self,
                self.tr("Dataset Filter"),
                self.tr(
                    "Please select at least one label, or enable empty images."
                ),
            )
            return

        out_dir = (self.ed_out.text() or "").strip()
        if not out_dir:
            QMessageBox.warning(
                self,
                self.tr("Dataset Filter"),
                self.tr("Please select an output folder."),
            )
            return
        if not osp.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    self.tr("Dataset Filter"),
                    self.tr("Cannot create output folder: %s") % str(e),
                )
                return

        matched = filter_matching_images(
            images,
            labels=labels,
            include_empty=include_empty,
            output_dir=self._output_dir(),
        )
        if not matched:
            QMessageBox.information(
                self,
                self.tr("Dataset Filter"),
                self.tr("No matching images found."),
            )
            return

        mode = "move" if self.rb_move.isChecked() else "copy"
        if mode == "move":
            confirm = self.tr(
                "About to move %d image(s) (with annotation files when present) "
                "to:\n%s\n\nContinue?"
            ) % (len(matched), out_dir)
        else:
            confirm = self.tr(
                "About to copy %d image(s) (with annotation files when present) "
                "to:\n%s\n\nContinue?"
            ) % (len(matched), out_dir)
        reply = QMessageBox.question(
            self,
            self.tr("Dataset Filter"),
            confirm,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.btn_run.setEnabled(False)
        self.btn_close.setEnabled(False)

        self._progress = QtWidgets.QProgressDialog(self)
        self._progress.setWindowTitle(self.tr("Dataset Filter"))
        self._progress.setLabelText(self.tr("Preparing…"))
        self._progress.setCancelButtonText(self.tr("Cancel"))
        self._progress.setStyleSheet(get_progress_dialog_style())
        self._progress.setRange(0, max(1, len(matched)))
        self._progress.setMinimumDuration(0)
        self._progress.canceled.connect(self._cancel_worker)
        self._progress.show()

        self._worker = _DatasetFilterWorker(
            image_paths=matched,
            target_dir=out_dir,
            mode=mode,
            output_dir=self._output_dir(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel_worker(self) -> None:
        if self._worker is not None:
            try:
                self._worker.request_cancel()
            except Exception:
                pass

    def _on_progress(self, i: int, n: int, path: str) -> None:
        if self._progress is None:
            return
        self._progress.setRange(0, max(1, int(n)))
        self._progress.setValue(int(i))
        self._progress.setLabelText(
            self.tr("Processing: %s") % osp.basename(path or "")
        )

    def _on_finished(
        self,
        ok: bool,
        err: str,
        ok_count: int,
        fail_count: int,
        transferred: list,
    ) -> None:
        if self._progress is not None:
            try:
                self._progress.close()
            except Exception:
                pass
            self._progress = None
        self.btn_run.setEnabled(True)
        self.btn_close.setEnabled(True)
        mode = "move" if self.rb_move.isChecked() else "copy"
        worker = self._worker
        self._worker = None

        if mode == "move" and transferred:
            self._remove_transferred_from_parent(transferred)

        if not ok:
            QMessageBox.warning(
                self,
                self.tr("Dataset Filter"),
                self.tr("Failed: %s") % str(err),
            )
            return

        verb = self.tr("Moved") if mode == "move" else self.tr("Copied")
        msg = self.tr("%s %d image(s).") % (verb, int(ok_count))
        if fail_count:
            msg += " " + self.tr("Failed: %d.") % int(fail_count)
        QMessageBox.information(self, self.tr("Dataset Filter"), msg)
        self.status_label.setText(msg)

    def _remove_transferred_from_parent(self, transferred: Sequence[str]) -> None:
        parent = self._parent
        if parent is None:
            return
        moved = {osp.normcase(osp.normpath(str(p))) for p in transferred}
        if not moved:
            return

        current = str(getattr(parent, "filename", "") or "")
        current_moved = bool(
            current and osp.normcase(osp.normpath(current)) in moved
        )

        file_list = getattr(parent, "file_list_widget", None)
        fn_to_index = getattr(parent, "fn_to_index", None)
        if file_list is None:
            return

        # Remove from bottom to top so indices stay valid.
        for i in range(file_list.count() - 1, -1, -1):
            item = file_list.item(i)
            if item is None:
                continue
            path = osp.normcase(osp.normpath(str(item.text())))
            if path in moved:
                file_list.takeItem(i)

        if isinstance(fn_to_index, dict):
            fn_to_index.clear()
            for i in range(file_list.count()):
                item = file_list.item(i)
                if item is not None:
                    fn_to_index[str(item.text())] = i

        remaining = list(getattr(parent, "image_list", None) or [])
        if not current_moved:
            return
        if remaining:
            try:
                if hasattr(parent, "may_continue") and not parent.may_continue():
                    return
                parent.load_file(remaining[0])
            except Exception:
                pass
        else:
            try:
                if hasattr(parent, "reset_state"):
                    parent.reset_state()
            except Exception:
                pass


def open_dataset_filter_dialog(parent) -> DatasetFilterDialog:
    dlg = DatasetFilterDialog(parent)
    dlg.exec()
    return dlg
