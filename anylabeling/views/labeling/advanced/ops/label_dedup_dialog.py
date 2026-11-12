"""Dialog for duplicate label removal (Advanced Features)."""

from __future__ import annotations

import os.path as osp
from typing import List, Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from anylabeling.views.labeling.classifier.utils import get_label_file_path
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.advanced.ops.label_dedup import dedup_label_file, dedup_shapes
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_ok_btn_style,
    get_progress_dialog_style,
    get_spinbox_style,
)

_SETTINGS_NAME = "label_dedup_settings.json"


def _settings_path() -> str:
    from anylabeling.config import get_work_directory

    return osp.join(get_work_directory(), _SETTINGS_NAME)


def _load_iou_default() -> float:
    path = _settings_path()
    if not osp.isfile(path):
        return 0.9
    try:
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("iou_threshold", 0.9))
    except Exception:
        return 0.9


def _save_iou_default(value: float) -> None:
    import json
    import os

    path = _settings_path()
    try:
        os.makedirs(osp.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"iou_threshold": float(value)}, f, indent=2)
    except Exception as e:
        logger.warning("Failed to save label dedup settings: %s", e)


class _LabelDedupBatchWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(bool, str, int, int)

    def __init__(
        self,
        *,
        image_paths: List[str],
        output_dir: Optional[str],
        iou_threshold: float,
    ):
        super().__init__()
        self._image_paths = list(image_paths)
        self._output_dir = output_dir
        self._iou_threshold = float(iou_threshold)

    def run(self) -> None:
        total_removed = 0
        files_changed = 0
        total = len(self._image_paths)
        try:
            for i, image_path in enumerate(self._image_paths):
                if self.isInterruptionRequested():
                    break
                base = osp.basename(image_path)
                self.progress.emit(i, total, base)
                label_path = get_label_file_path(image_path, self._output_dir)
                try:
                    removed, _remaining, written = dedup_label_file(
                        label_path, self._iou_threshold
                    )
                    if written:
                        files_changed += 1
                        total_removed += removed
                except Exception as e:
                    logger.error(
                        "Label dedup failed for %s: %s", label_path, e
                    )
            summary = (
                f"Removed {total_removed} duplicate shape(s) "
                f"across {files_changed} file(s)."
            )
            self.finished.emit(True, summary, total_removed, files_changed)
        except Exception as e:
            self.finished.emit(False, str(e), 0, 0)


class LabelDedupDialog(QDialog):
    def __init__(self, parent_widget):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self._worker: Optional[_LabelDedupBatchWorker] = None
        self._progress: Optional[QtWidgets.QProgressDialog] = None

        self.setWindowTitle(self.tr("Label Deduplication"))
        self.setMinimumWidth(520)
        self.setStyleSheet(get_dialog_style())
        self._build_ui()
        self.iou_spin.setValue(_load_iou_default())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        intro = QLabel(
            self.tr(
                "Treat two boxes as duplicates when they share the same class "
                "label and IoU exceeds the threshold. The largest box is kept; "
                "smaller overlapping boxes are removed (including groups of 3+)."
            )
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(0.9)
        self.iou_spin.setStyleSheet(get_spinbox_style())
        form.addRow(self.tr("IoU threshold:"), self.iou_spin)
        root.addLayout(form)

        btns = QHBoxLayout()
        btns.addStretch(1)

        self.btn_close = QPushButton(self.tr("Close"))
        self.btn_close.setStyleSheet(get_cancel_btn_style())
        self.btn_close.clicked.connect(self.reject)

        self.btn_batch = QPushButton(self.tr("Batch deduplicate"))
        self.btn_batch.setStyleSheet(get_cancel_btn_style())
        self.btn_batch.clicked.connect(self._run_batch)

        self.btn_current = QPushButton(self.tr("Deduplicate current image"))
        self.btn_current.setStyleSheet(get_ok_btn_style())
        self.btn_current.clicked.connect(self._run_current)

        btns.addWidget(self.btn_close)
        btns.addWidget(self.btn_batch)
        btns.addWidget(self.btn_current)
        root.addLayout(btns)

    def _iou_threshold(self) -> float:
        return float(self.iou_spin.value())

    def _save_settings(self) -> None:
        _save_iou_default(self._iou_threshold())

    def _set_running(self, running: bool) -> None:
        self.btn_current.setEnabled(not running)
        self.btn_batch.setEnabled(not running)

    def _collect_current_shape_dicts(self) -> List[dict]:
        parent = self.parent_widget
        shapes = []
        for item in parent.label_list:
            shape = item.shape()
            if not shape or not shape.label:
                continue
            shapes.append(shape.to_dict())
        return shapes

    def _apply_shapes_to_canvas(self, shape_dicts: List[dict]) -> None:
        parent = self.parent_widget
        parent.load_shapes([], replace=True)
        parent.label_list.clear()
        loaded = []
        for d in shape_dicts:
            try:
                loaded.append(Shape().load_from_dict(d, close=False))
            except Exception as e:
                logger.warning("Skip invalid shape after dedup: %s", e)
        if loaded:
            parent.load_shapes(loaded, replace=True)
        parent.set_dirty()

    def _run_current(self) -> None:
        parent = self.parent_widget
        if not getattr(parent, "filename", None):
            QMessageBox.information(
                self,
                self.tr("Label Deduplication"),
                self.tr("No image loaded."),
            )
            return

        self._save_settings()
        shapes = self._collect_current_shape_dicts()
        if not shapes:
            QMessageBox.information(
                self,
                self.tr("Label Deduplication"),
                self.tr("No annotations on the current image."),
            )
            return

        new_shapes, removed = dedup_shapes(shapes, self._iou_threshold())
        if removed <= 0:
            QMessageBox.information(
                self,
                self.tr("Label Deduplication"),
                self.tr("No duplicate annotations found."),
            )
            return

        self._apply_shapes_to_canvas(new_shapes)
        QMessageBox.information(
            self,
            self.tr("Label Deduplication"),
            self.tr("Removed %d duplicate annotation(s).") % removed,
        )

    def _run_batch(self) -> None:
        parent = self.parent_widget
        image_list = list(getattr(parent, "image_list", []) or [])
        if not image_list:
            QMessageBox.information(
                self,
                self.tr("Label Deduplication"),
                self.tr("No images loaded."),
            )
            return

        self._save_settings()
        self._set_running(True)

        self._progress = QtWidgets.QProgressDialog(
            self.tr("Processing..."),
            self.tr("Cancel"),
            0,
            len(image_list),
            parent,
        )
        self._progress.setWindowTitle(self.tr("Label Deduplication"))
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setStyleSheet(
            get_progress_dialog_style(color="#1d1d1f", height=20)
        )
        self._progress.setMinimumDuration(0)
        self._progress.canceled.connect(self._cancel_batch)
        self._progress.show()

        self._worker = _LabelDedupBatchWorker(
            image_paths=image_list,
            output_dir=getattr(parent, "output_dir", None),
            iou_threshold=self._iou_threshold(),
        )
        self._worker.progress.connect(self._on_batch_progress)
        self._worker.finished.connect(self._on_batch_finished)
        self._worker.start()

    def _cancel_batch(self) -> None:
        if self._worker is not None:
            self._worker.requestInterruption()

    def _on_batch_progress(self, i: int, n: int, name: str) -> None:
        if self._progress is None:
            return
        self._progress.setMaximum(max(1, n))
        self._progress.setValue(i)
        self._progress.setLabelText(
            self.tr("Processing: %s (%d / %d)") % (name, i + 1, n)
        )

    def _on_batch_finished(
        self, ok: bool, msg: str, removed: int, files_changed: int
    ) -> None:
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
                self.tr("Label Deduplication"),
                self.tr("Failed: %s") % msg,
            )
            return

        current = getattr(parent, "filename", None)
        if current and hasattr(parent, "load_file"):
            try:
                parent.load_file(current)
            except Exception:
                pass

        QMessageBox.information(
            parent,
            self.tr("Label Deduplication"),
            msg if removed > 0 or files_changed > 0 else self.tr("No duplicates found."),
        )

    def closeEvent(self, event) -> None:
        self._save_settings()
        if self._worker is not None and self._worker.isRunning():
            event.ignore()
            QMessageBox.information(
                self,
                self.tr("Label Deduplication"),
                self.tr(
                    "Batch deduplication is still running. "
                    "Cancel from the progress dialog or wait until it finishes."
                ),
            )
            return
        super().closeEvent(event)


def open_label_dedup_dialog(parent) -> None:
    dlg = LabelDedupDialog(parent)
    dlg.exec()
