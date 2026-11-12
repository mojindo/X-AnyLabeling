"""Dialog for batch dataset rename (Advanced Features, no license gate)."""

from __future__ import annotations

import os.path as osp
from typing import List, Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.advanced.ops.dataset_rename import (
    DatasetRenameResult,
    build_rename_plan,
    default_prefix_from_image_paths,
    rename_dataset,
)
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_ok_btn_style,
    get_progress_dialog_style,
    get_spinbox_style,
)


class _DatasetRenameWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(bool, str, object)

    def __init__(
        self,
        *,
        image_paths: List[str],
        prefix: str,
        output_dir: Optional[str],
        start_index: int,
        num_digits: int,
    ):
        super().__init__()
        self._image_paths = list(image_paths)
        self._prefix = prefix
        self._output_dir = output_dir
        self._start_index = int(start_index)
        self._num_digits = int(num_digits)

    def run(self) -> None:
        total = len(self._image_paths)
        try:
            self.progress.emit(0, total, self.tr("Preparing..."))
            result = rename_dataset(
                self._image_paths,
                self._prefix,
                self._output_dir,
                start_index=self._start_index,
                num_digits=self._num_digits,
            )
            summary = (
                f"Renamed {result.renamed_images} image(s), "
                f"{result.renamed_labels} label file(s); "
                f"updated {result.updated_json} JSON imagePath field(s)."
            )
            if result.skipped_same_name:
                summary += f" Skipped {result.skipped_same_name} unchanged name(s)."
            if result.errors:
                summary += f" {len(result.errors)} warning(s)."
            self.finished.emit(True, summary, result)
        except Exception as e:
            logger.error("Dataset rename failed: %s", e)
            self.finished.emit(False, str(e), None)


class DatasetRenameDialog(QDialog):
    def __init__(self, parent_widget):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self._worker: Optional[_DatasetRenameWorker] = None
        self._progress: Optional[QtWidgets.QProgressDialog] = None

        self.setWindowTitle(self.tr("Dataset Rename"))
        self.setMinimumWidth(560)
        self.setStyleSheet(get_dialog_style())
        self._build_ui()
        self._load_defaults()

    def _image_paths(self) -> List[str]:
        return list(getattr(self.parent_widget, "image_list", []) or [])

    def _load_defaults(self) -> None:
        paths = self._image_paths()
        if paths:
            self.prefix_edit.setText(default_prefix_from_image_paths(paths))
        self._update_preview()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        intro = QLabel(
            self.tr(
                "Batch-rename all images in the current dataset. "
                "Format: {prefix}_{6-digit index}.ext (e.g. base_0_0_000001.jpg). "
                "Matching JSON files are renamed and their imagePath field is updated."
            )
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("base_0_0")
        self.prefix_edit.textChanged.connect(self._update_preview)
        form.addRow(self.tr("Name prefix:"), self.prefix_edit)

        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 999999999)
        self.start_spin.setValue(1)
        self.start_spin.setStyleSheet(get_spinbox_style())
        self.start_spin.valueChanged.connect(self._update_preview)
        form.addRow(self.tr("Start index:"), self.start_spin)

        self.digits_spin = QSpinBox()
        self.digits_spin.setRange(1, 12)
        self.digits_spin.setValue(6)
        self.digits_spin.setStyleSheet(get_spinbox_style())
        self.digits_spin.valueChanged.connect(self._update_preview)
        form.addRow(self.tr("Index digits:"), self.digits_spin)

        root.addLayout(form)

        self.preview_label = QLabel()
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("color: #666;")
        root.addWidget(self.preview_label)

        btns = QHBoxLayout()
        btns.addStretch(1)

        self.btn_close = QPushButton(self.tr("Close"))
        self.btn_close.setStyleSheet(get_cancel_btn_style())
        self.btn_close.clicked.connect(self.reject)

        self.btn_run = QPushButton(self.tr("Rename"))
        self.btn_run.setStyleSheet(get_ok_btn_style())
        self.btn_run.clicked.connect(self._run_rename)

        btns.addWidget(self.btn_close)
        btns.addWidget(self.btn_run)
        root.addLayout(btns)

    def _update_preview(self) -> None:
        paths = self._image_paths()
        prefix = self.prefix_edit.text().strip()
        if not paths or not prefix:
            self.preview_label.setText(self.tr("Preview: —"))
            return
        try:
            plan = build_rename_plan(
                paths[:3],
                prefix,
                start_index=int(self.start_spin.value()),
                num_digits=int(self.digits_spin.value()),
            )
            lines = [osp.basename(n) for _o, n in plan]
            if len(paths) > 3:
                lines.append("…")
            self.preview_label.setText(
                self.tr("Preview: %s") % ", ".join(lines)
            )
        except Exception as e:
            self.preview_label.setText(self.tr("Preview error: %s") % str(e))

    def _set_running(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        self.prefix_edit.setEnabled(not running)
        self.start_spin.setEnabled(not running)
        self.digits_spin.setEnabled(not running)

    def _run_rename(self) -> None:
        parent = self.parent_widget
        image_list = self._image_paths()
        if not image_list:
            QMessageBox.information(
                self,
                self.tr("Dataset Rename"),
                self.tr("No images loaded."),
            )
            return

        prefix = self.prefix_edit.text().strip()
        if not prefix:
            QMessageBox.warning(
                self,
                self.tr("Dataset Rename"),
                self.tr("Please enter a name prefix."),
            )
            return

        try:
            plan = build_rename_plan(
                image_list,
                prefix,
                start_index=int(self.start_spin.value()),
                num_digits=int(self.digits_spin.value()),
            )
        except Exception as e:
            QMessageBox.warning(
                self,
                self.tr("Dataset Rename"),
                self.tr("Invalid rename plan: %s") % str(e),
            )
            return

        sample = osp.basename(plan[0][1]) if plan else ""
        reply = QMessageBox.question(
            parent,
            self.tr("Dataset Rename"),
            self.tr(
                "Rename %d image(s) under prefix %r?\nExample: %s\n"
                "This cannot be undone automatically."
            )
            % (len(plan), prefix, sample),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._set_running(True)
        self._progress = QtWidgets.QProgressDialog(
            self.tr("Renaming..."),
            self.tr("Cancel"),
            0,
            len(image_list),
            parent,
        )
        self._progress.setWindowTitle(self.tr("Dataset Rename"))
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setStyleSheet(
            get_progress_dialog_style(color="#1d1d1f", height=20)
        )
        self._progress.setMinimumDuration(0)
        self._progress.canceled.connect(self._cancel_worker)
        self._progress.show()

        self._worker = _DatasetRenameWorker(
            image_paths=image_list,
            prefix=prefix,
            output_dir=getattr(parent, "output_dir", None),
            start_index=int(self.start_spin.value()),
            num_digits=int(self.digits_spin.value()),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel_worker(self) -> None:
        if self._worker is not None:
            self._worker.requestInterruption()

    def _on_progress(self, i: int, n: int, name: str) -> None:
        if self._progress is None:
            return
        self._progress.setMaximum(max(1, n))
        self._progress.setValue(min(i, n))
        self._progress.setLabelText(name)

    def _reload_dataset(self, result: DatasetRenameResult) -> None:
        parent = self.parent_widget
        old_current = getattr(parent, "filename", None)
        new_current = None
        if old_current and old_current in result.path_map:
            new_current = result.path_map[old_current]

        image_list = self._image_paths()
        if not image_list:
            return
        image_dir = osp.dirname(osp.normpath(image_list[0]))
        if not osp.isdir(image_dir):
            return

        try:
            parent.import_image_folder(image_dir, load=False)
        except Exception as e:
            logger.warning("Reload image folder after rename failed: %s", e)
            return

        if new_current and osp.isfile(new_current) and hasattr(parent, "load_file"):
            try:
                parent.load_file(new_current)
            except Exception as e:
                logger.warning("Reload current image after rename failed: %s", e)

    def _on_finished(self, ok: bool, msg: str, result: object) -> None:
        if self._progress is not None:
            try:
                self._progress.close()
            except Exception:
                pass
            self._progress = None
        self._set_running(False)
        self._worker = None

        parent = self.parent_widget
        if not ok:
            QMessageBox.warning(
                parent,
                self.tr("Dataset Rename"),
                self.tr("Failed: %s") % msg,
            )
            return

        if isinstance(result, DatasetRenameResult):
            self._reload_dataset(result)

        if isinstance(result, DatasetRenameResult) and result.errors:
            detail = "\n".join(result.errors[:8])
            QMessageBox.warning(
                parent,
                self.tr("Dataset Rename"),
                self.tr("%s\n\nWarnings:\n%s") % (msg, detail),
            )
        else:
            QMessageBox.information(
                parent,
                self.tr("Dataset Rename"),
                msg,
            )

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            event.ignore()
            QMessageBox.information(
                self,
                self.tr("Dataset Rename"),
                self.tr(
                    "Rename is still running. "
                    "Cancel from the progress dialog or wait until it finishes."
                ),
            )
            return
        super().closeEvent(event)


def open_dataset_rename_dialog(parent) -> None:
    dlg = DatasetRenameDialog(parent)
    dlg.exec()
