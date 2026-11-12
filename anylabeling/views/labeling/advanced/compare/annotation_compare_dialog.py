"""Dialog to run batch annotation compare (GT JSON vs YOLO predictions)."""

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
from anylabeling.views.labeling.advanced.compare.annotation_compare import CompareSettings
from anylabeling.views.labeling.advanced.compare.annotation_compare_io import (
    ClearCompareResult,
    run_batch_clear_compare_results,
    run_batch_compare,
)
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_ok_btn_style,
    get_progress_dialog_style,
    get_spinbox_style,
)


class _ClearCompareWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(bool, str, object)

    def __init__(
        self,
        *,
        image_paths: List[str],
        output_dir: Optional[str],
    ):
        super().__init__()
        self._image_paths = list(image_paths)
        self._output_dir = output_dir

    def run(self) -> None:
        try:
            result = run_batch_clear_compare_results(
                image_paths=self._image_paths,
                output_dir=self._output_dir,
                progress_cb=lambda i, n, name: self.progress.emit(i, n, name),
            )
            self.finished.emit(True, "", result)
        except Exception as e:
            logger.exception("Clear compare results failed")
            self.finished.emit(False, str(e), None)


class _CompareBatchWorker(QtCore.QThread):
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
                f"Matched: {st.matched}, suspect missed (漏标): {st.suspect_fn}, "
                f"suspect false label (误标): {st.suspect_fp}.\n"
                f"Backup: {result.backup_dir}"
            )
            val_metrics = getattr(result, "val_metrics", None)
            if val_metrics is not None:
                summary += "\n\n" + val_metrics.format_summary()
            if result.errors:
                summary += f"\n\n{len(result.errors)} warning(s)."
            self.finished.emit(True, summary, result)
        except Exception as e:
            logger.error("Annotation compare batch failed: %s", e)
            self.finished.emit(False, str(e), None)


class AnnotationCompareDialog(QDialog):
    def __init__(self, parent_widget):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self._worker: Optional[_CompareBatchWorker] = None
        self._clear_worker: Optional[_ClearCompareWorker] = None
        self._progress: Optional[QtWidgets.QProgressDialog] = None

        self.setWindowTitle(self.tr("Annotation Compare"))
        self.setMinimumWidth(560)
        self.setStyleSheet(get_dialog_style())
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        intro = QLabel(
            self.tr(
                "Compare current dataset JSON (ground truth) with uploaded YOLO "
                "predictions. Only labels that appear in both GT and classes.txt "
                "are compared. Yellow = suspect missed label; Red = suspect false label."
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
        self.check_fp_cb = QCheckBox(self.tr("False-label check (误标)"))
        self.check_fp_cb.setChecked(True)
        self.check_fp_cb.setToolTip(
            self.tr(
                "Mark unmatched GT boxes as suspect false labels (red). "
                "When unchecked, those boxes are not flagged in the label file."
            )
        )
        self.check_fn_cb = QCheckBox(self.tr("Missed-label check (漏标)"))
        self.check_fn_cb.setChecked(True)
        self.check_fn_cb.setToolTip(
            self.tr(
                "Add unmatched prediction boxes as suspect missed labels (yellow). "
                "When unchecked, they are not written to the label file."
            )
        )
        check_row.addWidget(self.check_fp_cb)
        check_row.addWidget(self.check_fn_cb)
        check_row.addStretch(1)
        form.addRow(self.tr("Checks:"), check_row)

        root.addLayout(form)

        clear_hint = QLabel(
            self.tr(
                "One-click clear: delete all suspect missed-label boxes (yellow); "
                "keep suspect false-label boxes (red) and remove their compare flags."
            )
        )
        clear_hint.setWordWrap(True)
        root.addWidget(clear_hint)

        clear_btn = QPushButton(self.tr("Clear compare results"))
        clear_btn.setStyleSheet(get_cancel_btn_style())
        clear_btn.clicked.connect(self._run_clear_compare)
        root.addWidget(clear_btn)

        btns = QHBoxLayout()
        btns.addStretch(1)
        close_btn = QPushButton(self.tr("Close"))
        close_btn.setStyleSheet(get_cancel_btn_style())
        close_btn.clicked.connect(self.reject)
        run_btn = QPushButton(self.tr("Run compare"))
        run_btn.setStyleSheet(get_ok_btn_style())
        run_btn.clicked.connect(self._run_compare)
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
            check_fp=bool(self.check_fp_cb.isChecked()),
            check_fn=bool(self.check_fn_cb.isChecked()),
        )

    def _run_compare(self) -> None:
        parent = self.parent_widget
        image_list = list(getattr(parent, "image_list", []) or [])
        if not image_list:
            QMessageBox.information(
                self, self.tr("Annotation Compare"), self.tr("No images loaded.")
            )
            return

        if not self.check_fp_cb.isChecked() and not self.check_fn_cb.isChecked():
            QMessageBox.warning(
                self,
                self.tr("Annotation Compare"),
                self.tr(
                    "Please enable at least one of: false-label check (误标) "
                    "or missed-label check (漏标)."
                ),
            )
            return

        pred_dir = self.pred_dir_edit.text().strip()
        classes_path = self.classes_edit.text().strip()
        if not pred_dir or not osp.isdir(pred_dir):
            QMessageBox.warning(
                self,
                self.tr("Annotation Compare"),
                self.tr("Please select a valid prediction label folder."),
            )
            return
        if not classes_path or not osp.isfile(classes_path):
            QMessageBox.warning(
                self,
                self.tr("Annotation Compare"),
                self.tr("Please select a valid classes.txt file."),
            )
            return

        reply = QMessageBox.question(
            parent,
            self.tr("Annotation Compare"),
            self.tr(
                "Run compare on %d image(s)? Original JSON will be backed up first."
            )
            % len(image_list),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._progress = QtWidgets.QProgressDialog(
            self.tr("Comparing..."),
            self.tr("Cancel"),
            0,
            len(image_list),
            parent,
        )
        self._progress.setWindowTitle(self.tr("Annotation Compare"))
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setStyleSheet(
            get_progress_dialog_style(color="#1d1d1f", height=20)
        )
        self._progress.setMinimumDuration(0)
        self._progress.canceled.connect(self._cancel_worker)
        self._progress.show()

        self._worker = _CompareBatchWorker(
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
        if self._clear_worker is not None:
            self._clear_worker.requestInterruption()

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

    def _update_progress_label(
        self, i: int, n: int, name: str, template: str
    ) -> None:
        prog = self._progress
        if prog is None:
            return
        try:
            prog.setMaximum(max(1, n))
            prog.setValue(i)
            prog.setLabelText(template % (name, i + 1, n))
        except RuntimeError:
            pass

    def _run_clear_compare(self) -> None:
        parent = self.parent_widget
        image_list = list(getattr(parent, "image_list", []) or [])
        if not image_list:
            QMessageBox.information(
                self,
                self.tr("Annotation Compare"),
                self.tr("No images loaded."),
            )
            return

        reply = QMessageBox.question(
            parent,
            self.tr("Clear Compare Results"),
            self.tr(
                "Clear compare / model-eval results on %d image(s)?\n\n"
                "• Delete all suspect missed-label / false-alarm boxes\n"
                "• Keep false-label / miss boxes and remove their flags\n\n"
                "Original JSON will be backed up first."
            )
            % len(image_list),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._progress = QtWidgets.QProgressDialog(
            self.tr("Clearing..."),
            self.tr("Cancel"),
            0,
            len(image_list),
            parent,
        )
        self._progress.setWindowTitle(self.tr("Clear Compare Results"))
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setStyleSheet(
            get_progress_dialog_style(color="#1d1d1f", height=20)
        )
        self._progress.setMinimumDuration(0)
        self._progress.canceled.connect(self._cancel_worker)
        self._progress.show()

        self._clear_worker = _ClearCompareWorker(
            image_paths=image_list,
            output_dir=getattr(parent, "output_dir", None),
        )
        self._clear_worker.progress.connect(self._on_clear_progress)
        self._clear_worker.finished.connect(self._on_clear_finished)
        self._clear_worker.start()

    def _on_clear_progress(self, i: int, n: int, name: str) -> None:
        self._update_progress_label(
            i, n, name, self.tr("Clearing: %s (%d / %d)")
        )

    def _on_clear_finished(self, ok: bool, msg: str, result: object) -> None:
        if self._clear_worker is not None:
            try:
                self._clear_worker.progress.disconnect()
            except Exception:
                pass
        self._close_progress_safely()
        self._clear_worker = None

        parent = self.parent_widget
        if not ok:
            QMessageBox.warning(
                parent,
                self.tr("Clear Compare Results"),
                self.tr("Failed: %s") % msg,
            )
            return

        _refresh_after_clear_compare(parent, result)

        if isinstance(result, ClearCompareResult):
            msg = self.tr(
                "Cleared compare / model-eval results on %d file(s).\n"
                "Removed pred-side marks (漏标/误报): %d\n"
                "Cleared GT-side flags (误标/漏报): %d\n"
                "Backup: %s"
            ) % (
                result.files_updated,
                result.fn_removed,
                result.fp_cleared,
                result.backup_dir,
            )
        else:
            msg = msg or self.tr("Done.")

        if result is not None and getattr(result, "errors", None):
            detail = "\n".join(result.errors[:8])
            QMessageBox.warning(
                parent,
                self.tr("Clear Compare Results"),
                self.tr("%s\n\nWarnings:\n%s") % (msg, detail),
            )
        else:
            QMessageBox.information(
                parent, self.tr("Clear Compare Results"), msg
            )

    def _on_progress(self, i: int, n: int, name: str) -> None:
        self._update_progress_label(
            i, n, name, self.tr("Comparing: %s (%d / %d)")
        )

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
                parent, self.tr("Annotation Compare"), self.tr("Failed: %s") % msg
            )
            return

        from anylabeling.views.labeling.advanced.compare.annotation_compare import (
            REVIEW_MODE_ANNOTATION,
            load_classes_file,
        )

        parent._annotation_compare_active = True
        parent._review_marks_mode = REVIEW_MODE_ANNOTATION
        if hasattr(parent, "_sync_compare_issue_nav_enabled"):
            parent._sync_compare_issue_nav_enabled()
        parent._annotation_compare_classes = []
        classes_path = self.classes_edit.text().strip()
        if classes_path and osp.isfile(classes_path):
            try:
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
                self.tr("Annotation Compare"),
                self.tr("%s\n\nWarnings:\n%s") % (msg, detail),
            )
        else:
            QMessageBox.information(
                parent, self.tr("Annotation Compare"), msg
            )
        self.accept()

    def closeEvent(self, event) -> None:
        if self._clear_worker is not None and self._clear_worker.isRunning():
            event.ignore()
            QMessageBox.information(
                self,
                self.tr("Clear Compare Results"),
                self.tr("Clear is still running. Cancel from the progress dialog."),
            )
            return
        if self._worker is not None and self._worker.isRunning():
            event.ignore()
            QMessageBox.information(
                self,
                self.tr("Annotation Compare"),
                self.tr("Compare is still running. Cancel from the progress dialog."),
            )
            return
        super().closeEvent(event)


def _refresh_after_clear_compare(parent, result: object) -> None:
    from anylabeling.views.labeling.advanced.compare.annotation_compare_io import (
        label_json_has_pending_compare,
    )
    from anylabeling.views.labeling.classifier.utils import get_label_file_path

    image_list = list(getattr(parent, "image_list", []) or [])
    output_dir = getattr(parent, "output_dir", None)
    any_pending = False
    for image_path in image_list:
        lp = get_label_file_path(image_path, output_dir)
        if label_json_has_pending_compare(lp):
            any_pending = True
            break
    if not any_pending:
        parent._annotation_compare_active = False
        if hasattr(parent, "_sync_compare_issue_nav_enabled"):
            parent._sync_compare_issue_nav_enabled()

    current = getattr(parent, "filename", None)
    if current and hasattr(parent, "load_file"):
        try:
            parent.load_file(current)
        except Exception:
            pass


def run_clear_compare_results_ui(parent) -> None:
    """Confirm and batch-clear compare marks (menu entry without opening compare dialog)."""
    image_list = list(getattr(parent, "image_list", []) or [])
    if not image_list:
        QMessageBox.information(
            parent,
            parent.tr("Clear Compare Results"),
            parent.tr("No images loaded."),
        )
        return

    reply = QMessageBox.question(
        parent,
        parent.tr("Clear Compare Results"),
        parent.tr(
            "Clear compare / model-eval results on %d image(s)?\n\n"
            "• Delete all suspect missed-label / false-alarm boxes\n"
            "• Keep false-label / miss boxes and remove their flags\n\n"
            "Original JSON will be backed up first."
        )
        % len(image_list),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if reply != QMessageBox.StandardButton.Yes:
        return

    progress = QtWidgets.QProgressDialog(
        parent.tr("Clearing..."),
        parent.tr("Cancel"),
        0,
        len(image_list),
        parent,
    )
    progress.setWindowTitle(parent.tr("Clear Compare Results"))
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setStyleSheet(get_progress_dialog_style(color="#1d1d1f", height=20))
    progress.setMinimumDuration(0)
    progress.show()
    QtWidgets.QApplication.processEvents()

    def on_progress(i: int, n: int, name: str) -> None:
        progress.setMaximum(max(1, n))
        progress.setValue(i)
        progress.setLabelText(
            parent.tr("Clearing: %s (%d / %d)") % (name, i + 1, n)
        )
        QtWidgets.QApplication.processEvents()

    try:
        result = run_batch_clear_compare_results(
            image_paths=image_list,
            output_dir=getattr(parent, "output_dir", None),
            progress_cb=on_progress,
        )
        msg = parent.tr(
            "Cleared compare / model-eval results on %d file(s).\n"
            "Removed pred-side marks (漏标/误报): %d\n"
            "Cleared GT-side flags (误标/漏报): %d\n"
            "Backup: %s"
        ) % (
            result.files_updated,
            result.fn_removed,
            result.fp_cleared,
            result.backup_dir,
        )
        _refresh_after_clear_compare(parent, result)
        if result.errors:
            detail = "\n".join(result.errors[:8])
            QMessageBox.warning(
                parent,
                parent.tr("Clear Compare Results"),
                parent.tr("%s\n\nWarnings:\n%s") % (msg, detail),
            )
        else:
            QMessageBox.information(
                parent, parent.tr("Clear Compare Results"), msg
            )
    except Exception as e:
        logger.exception("Clear compare results failed")
        QMessageBox.warning(
            parent,
            parent.tr("Clear Compare Results"),
            parent.tr("Failed: %s") % str(e),
        )
    finally:
        try:
            progress.close()
        except Exception:
            pass


def open_annotation_compare_dialog(parent) -> None:
    dlg = AnnotationCompareDialog(parent)
    dlg.exec()
