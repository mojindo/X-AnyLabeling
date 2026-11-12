"""Dialog to run batch model evaluation (GT JSON vs YOLO predictions).

Reuses annotation-compare matching / IO; marks 漏报 (miss) and 误报 (false alarm).
"""

from __future__ import annotations

import os.path as osp
from typing import List, Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.advanced.compare.annotation_compare import (
    REVIEW_MODE_MODEL_EVAL,
    CompareSettings,
)
from anylabeling.views.labeling.advanced.compare.annotation_compare_io import (
    run_batch_compare,
)
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_ok_btn_style,
    get_progress_dialog_style,
    get_spinbox_style,
)


class _ModelEvalBatchWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(bool, str, object)

    def __init__(
        self,
        *,
        image_paths: List[str],
        output_dir: Optional[str],
        pred_dir: str,
        classes_path: str,
        settings: CompareSettings,
    ):
        super().__init__()
        self._image_paths = list(image_paths)
        self._output_dir = output_dir
        self._pred_dir = pred_dir
        self._classes_path = classes_path
        self._settings = settings

    def run(self) -> None:
        try:
            result = run_batch_compare(
                image_paths=self._image_paths,
                output_dir=self._output_dir,
                pred_dir=self._pred_dir,
                classes_path=self._classes_path,
                settings=self._settings,
                progress_cb=lambda i, n, name: self.progress.emit(i, n, name),
            )
            st = result.stats_total
            summary = (
                f"Processed {result.images_processed} image(s).\n"
                f"Matched: {st.matched}, miss (漏报): {st.suspect_fp}, "
                f"false alarm (误报): {st.suspect_fn}.\n"
                f"Backup: {result.backup_dir}"
            )
            val_metrics = getattr(result, "val_metrics", None)
            if val_metrics is not None:
                summary += "\n\n" + val_metrics.format_summary()
            if result.errors:
                summary += f"\n\n{len(result.errors)} warning(s)."
            self.finished.emit(True, summary, result)
        except Exception as e:
            logger.error("Model evaluation batch failed: %s", e)
            self.finished.emit(False, str(e), None)


class ModelEvalDialog(QDialog):
    def __init__(self, parent_widget):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self._worker: Optional[_ModelEvalBatchWorker] = None
        self._progress: Optional[QtWidgets.QProgressDialog] = None

        self.setWindowTitle(self.tr("Model Evaluation"))
        self.setMinimumWidth(560)
        self.setStyleSheet(get_dialog_style())
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        intro = QLabel(
            self.tr(
                "Evaluate model predictions against dataset JSON (ground truth). "
                "Only labels that appear in both GT and classes.txt are compared. "
                "Yellow = false alarm (误报); Red = miss (漏报)."
            )
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.pred_dir_edit = QLineEdit()
        pred_row = QHBoxLayout()
        pred_row.addWidget(self.pred_dir_edit, 1)
        pred_btn = QPushButton(self.tr("Browse…"))
        pred_btn.clicked.connect(self._browse_pred_dir)
        pred_row.addWidget(pred_btn)
        form.addRow(self.tr("Prediction labels:"), pred_row)

        self.classes_edit = QLineEdit()
        cls_row = QHBoxLayout()
        cls_row.addWidget(self.classes_edit, 1)
        cls_btn = QPushButton(self.tr("Browse…"))
        cls_btn.clicked.connect(self._browse_classes)
        cls_row.addWidget(cls_btn)
        form.addRow(self.tr("classes.txt:"), cls_row)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("ios (IoMin)", "ios")
        self.mode_combo.addItem("iou", "iou")
        form.addRow(self.tr("Match mode:"), self.mode_combo)

        self.match_spin = QDoubleSpinBox()
        self.match_spin.setRange(0.01, 1.0)
        self.match_spin.setSingleStep(0.05)
        self.match_spin.setValue(0.3)
        self.match_spin.setDecimals(2)
        self.match_spin.setStyleSheet(get_spinbox_style())
        form.addRow(self.tr("Match threshold:"), self.match_spin)

        self.dedup_spin = QDoubleSpinBox()
        self.dedup_spin.setRange(0.01, 1.0)
        self.dedup_spin.setSingleStep(0.05)
        self.dedup_spin.setValue(0.85)
        self.dedup_spin.setDecimals(2)
        self.dedup_spin.setStyleSheet(get_spinbox_style())
        form.addRow(self.tr("IOS dedup threshold:"), self.dedup_spin)

        check_row = QHBoxLayout()
        self.check_miss_cb = QCheckBox(self.tr("Miss check (漏报)"))
        self.check_miss_cb.setChecked(True)
        self.check_miss_cb.setToolTip(
            self.tr(
                "Mark unmatched GT boxes as model misses (red). "
                "When unchecked, those boxes are not flagged."
            )
        )
        self.check_false_cb = QCheckBox(self.tr("False-alarm check (误报)"))
        self.check_false_cb.setChecked(True)
        self.check_false_cb.setToolTip(
            self.tr(
                "Add unmatched prediction boxes as false alarms (yellow). "
                "When unchecked, they are not written to the label file."
            )
        )
        check_row.addWidget(self.check_miss_cb)
        check_row.addWidget(self.check_false_cb)
        check_row.addStretch(1)
        form.addRow(self.tr("Checks:"), check_row)

        root.addLayout(form)

        hint = QLabel(
            self.tr(
                "Use Advanced → Clear Compare Results to clear both annotation "
                "compare marks and model evaluation marks."
            )
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        btns = QHBoxLayout()
        btns.addStretch(1)
        close_btn = QPushButton(self.tr("Close"))
        close_btn.setStyleSheet(get_cancel_btn_style())
        close_btn.clicked.connect(self.reject)
        run_btn = QPushButton(self.tr("Run evaluation"))
        run_btn.setStyleSheet(get_ok_btn_style())
        run_btn.clicked.connect(self._run_eval)
        btns.addWidget(close_btn)
        btns.addWidget(run_btn)
        root.addLayout(btns)

    def _browse_pred_dir(self) -> None:
        start = self.pred_dir_edit.text().strip() or "."
        path = QFileDialog.getExistingDirectory(
            self, self.tr("Select YOLO label folder"), start
        )
        if path:
            self.pred_dir_edit.setText(path)

    def _browse_classes(self) -> None:
        start = self.classes_edit.text().strip() or "."
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select classes.txt"),
            start,
            self.tr("Text files (*.txt);;All files (*)"),
        )
        if path:
            self.classes_edit.setText(path)

    def _settings(self) -> CompareSettings:
        mode = self.mode_combo.currentData() or "ios"
        return CompareSettings(
            match_mode=str(mode),
            match_thresh=float(self.match_spin.value()),
            dedup_ios_thresh=float(self.dedup_spin.value()),
            # check_fp gates unmatched GT → 漏报; check_fn gates unmatched pred → 误报
            check_fp=bool(self.check_miss_cb.isChecked()),
            check_fn=bool(self.check_false_cb.isChecked()),
            review_mode=REVIEW_MODE_MODEL_EVAL,
        )

    def _run_eval(self) -> None:
        parent = self.parent_widget
        image_list = list(getattr(parent, "image_list", []) or [])
        if not image_list:
            QMessageBox.information(
                self, self.tr("Model Evaluation"), self.tr("No images loaded.")
            )
            return

        if not self.check_miss_cb.isChecked() and not self.check_false_cb.isChecked():
            QMessageBox.warning(
                self,
                self.tr("Model Evaluation"),
                self.tr(
                    "Please enable at least one of: miss check (漏报) "
                    "or false-alarm check (误报)."
                ),
            )
            return

        pred_dir = self.pred_dir_edit.text().strip()
        classes_path = self.classes_edit.text().strip()
        if not pred_dir or not osp.isdir(pred_dir):
            QMessageBox.warning(
                self,
                self.tr("Model Evaluation"),
                self.tr("Please select a valid prediction label folder."),
            )
            return
        if not classes_path or not osp.isfile(classes_path):
            QMessageBox.warning(
                self,
                self.tr("Model Evaluation"),
                self.tr("Please select a valid classes.txt file."),
            )
            return

        reply = QMessageBox.question(
            parent,
            self.tr("Model Evaluation"),
            self.tr(
                "Run model evaluation on %d image(s)? "
                "Original JSON will be backed up first."
            )
            % len(image_list),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._progress = QtWidgets.QProgressDialog(
            self.tr("Evaluating..."),
            self.tr("Cancel"),
            0,
            len(image_list),
            parent,
        )
        self._progress.setWindowTitle(self.tr("Model Evaluation"))
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setStyleSheet(
            get_progress_dialog_style(color="#1d1d1f", height=20)
        )
        self._progress.setMinimumDuration(0)
        self._progress.canceled.connect(self._cancel_worker)
        self._progress.show()

        self._worker = _ModelEvalBatchWorker(
            image_paths=image_list,
            output_dir=getattr(parent, "output_dir", None),
            pred_dir=pred_dir,
            classes_path=classes_path,
            settings=self._settings(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel_worker(self) -> None:
        if self._worker is not None:
            self._worker.requestInterruption()

    def _close_progress_safely(self) -> None:
        prog = self._progress
        self._progress = None
        if prog is None:
            return
        try:
            prog.canceled.disconnect()
        except Exception:
            pass
        try:
            prog.close()
        except Exception:
            pass

    def _on_progress(self, i: int, n: int, name: str) -> None:
        prog = self._progress
        if prog is None:
            return
        try:
            prog.setMaximum(max(1, n))
            prog.setValue(i)
            prog.setLabelText(
                self.tr("Evaluating: %s (%d / %d)") % (name, i + 1, n)
            )
        except RuntimeError:
            pass

    def _on_finished(self, ok: bool, msg: str, result: object) -> None:
        if self._worker is not None:
            try:
                self._worker.progress.disconnect()
            except Exception:
                pass
        self._close_progress_safely()
        self._worker = None

        parent = self.parent_widget
        if not ok:
            QMessageBox.warning(
                parent, self.tr("Model Evaluation"), self.tr("Failed: %s") % msg
            )
            return

        parent._annotation_compare_active = True
        parent._review_marks_mode = REVIEW_MODE_MODEL_EVAL
        if hasattr(parent, "_sync_compare_issue_nav_enabled"):
            parent._sync_compare_issue_nav_enabled()
        parent._annotation_compare_classes = []
        classes_path = self.classes_edit.text().strip()
        if classes_path and osp.isfile(classes_path):
            try:
                from anylabeling.views.labeling.advanced.compare.annotation_compare import (
                    load_classes_file,
                )

                parent._annotation_compare_classes = load_classes_file(classes_path)
            except Exception:
                pass

        current = getattr(parent, "filename", None)
        if current and hasattr(parent, "load_file"):
            try:
                parent.load_file(current)
            except Exception:
                pass

        if result is not None and getattr(result, "errors", None):
            detail = "\n".join(result.errors[:8])
            QMessageBox.warning(
                parent,
                self.tr("Model Evaluation"),
                self.tr("%s\n\nWarnings:\n%s") % (msg, detail),
            )
        else:
            QMessageBox.information(
                parent, self.tr("Model Evaluation"), msg
            )
        self.accept()

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            event.ignore()
            QMessageBox.information(
                self,
                self.tr("Model Evaluation"),
                self.tr(
                    "Evaluation is still running. Cancel from the progress dialog."
                ),
            )
            return
        super().closeEvent(event)


def open_model_eval_dialog(parent) -> None:
    dlg = ModelEvalDialog(parent)
    dlg.exec()
