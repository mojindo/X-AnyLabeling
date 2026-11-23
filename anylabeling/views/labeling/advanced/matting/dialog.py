"""Refined matting dialog (fine / fast modes)."""
from __future__ import annotations

import os
import os.path as osp
from typing import List, Optional, Set

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt

from anylabeling.views.labeling.advanced.matting.backends.registry import (
    MATTING_MODELS,
    MODEL_BY_KEY,
    default_model_key,
)
from anylabeling.views.labeling.advanced.matting.backends.base import MattingBackend
from anylabeling.views.labeling.advanced.matting.pipeline import MODE_FAST, MODE_FINE
from anylabeling.views.labeling.advanced.ops.label_jump import (
    collect_available_labels_with_counts,
)
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_lineedit_style,
    get_ok_btn_style,
    get_spinbox_style,
)


def _project_image_root(parent) -> str:
    """Folder the user opened for the current dataset (not annotation output_dir)."""
    if parent is None:
        return ""
    lod = getattr(parent, "last_open_dir", None)
    if lod and osp.isdir(str(lod)):
        return osp.normpath(osp.abspath(str(lod)))
    fn = getattr(parent, "filename", None)
    if fn and osp.isfile(str(fn)):
        return osp.normpath(osp.dirname(osp.abspath(str(fn))))
    paths = getattr(parent, "image_list", None) or []
    for p in paths:
        if p and osp.isfile(str(p)):
            return osp.normpath(osp.dirname(osp.abspath(str(p))))
    return ""


def _default_matting_output_dir(parent) -> str:
    root = _project_image_root(parent)
    if not root:
        return ""
    return osp.join(root, "refined-matting")


class _MattingWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    finished_ok = QtCore.pyqtSignal(int, int, int, bool)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        backend: MattingBackend,
        image_paths: List[str],
        mode: str,
        output_dir: str,
        label_output_dir: Optional[str],
        allowed_labels: Optional[Set[str]],
        padding_px: int,
        obb_rectify: bool,
        mask_dilate_px: int,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._paths = list(image_paths)
        self._mode = mode
        self._output_dir = output_dir
        self._label_output_dir = label_output_dir
        self._allowed_labels = allowed_labels
        self._padding = padding_px
        self._obb_rectify = obb_rectify
        self._mask_dilate_px = int(mask_dilate_px)
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            from anylabeling.views.labeling.advanced.matting.pipeline import (
                run_matting_dataset,
            )

            if self._cancel:
                self.finished_ok.emit(0, 0, 0, True)
                return
            n_img, n_shapes, n_out, cancelled = run_matting_dataset(
                self._paths,
                self._backend,
                mode=self._mode,
                output_dir=self._output_dir,
                output_label_dir=self._label_output_dir,
                allowed_labels=self._allowed_labels,
                padding_px=self._padding,
                obb_rectify=self._obb_rectify,
                mask_dilate_px=self._mask_dilate_px,
                on_progress=lambda i, n, name: self.progress.emit(i, n, name),
                should_cancel=lambda: self._cancel,
            )
            self.finished_ok.emit(
                n_img, n_shapes, n_out, bool(cancelled or self._cancel)
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class RefinedMattingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._host = parent
        self._worker: Optional[_MattingWorker] = None
        self._progress: Optional[QtWidgets.QProgressDialog] = None
        self._backend: Optional[MattingBackend] = None
        self._user_cancelled = False

        self.setWindowTitle(self.tr("Refined Matting"))
        self.setMinimumSize(520, 560)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setStyleSheet(get_dialog_style())
        self._build_ui()
        self._on_mode_changed()
        self._reload_labels()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        hint = QtWidgets.QLabel(
            self.tr(
                "Export RGBA crops per annotation. Fine mode: crop then RMBG refine. "
                "Fast mode: full-image RMBG once, then crop by box."
            )
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        layout.addWidget(hint)

        mode_box = QtWidgets.QGroupBox(self.tr("Mode"))
        mode_lay = QtWidgets.QVBoxLayout(mode_box)
        self._radio_fine = QtWidgets.QRadioButton(self.tr("Fine (crop then matting)"))
        self._radio_fast = QtWidgets.QRadioButton(self.tr("Fast (full image matting, then crop)"))
        self._radio_fine.setChecked(True)
        mode_lay.addWidget(self._radio_fine)
        mode_lay.addWidget(self._radio_fast)
        self._radio_fine.toggled.connect(self._on_mode_changed)
        layout.addWidget(mode_box)

        model_form = QtWidgets.QFormLayout()
        self._model_combo = QtWidgets.QComboBox()
        for spec in MATTING_MODELS:
            self._model_combo.addItem(spec.display_name, spec.key)
        model_form.addRow(self.tr("Matting model:"), self._model_combo)
        layout.addLayout(model_form)

        scope_box = QtWidgets.QGroupBox(self.tr("Scope"))
        scope_lay = QtWidgets.QVBoxLayout(scope_box)
        self._scope_current = QtWidgets.QRadioButton(self.tr("Current image only"))
        self._scope_dataset = QtWidgets.QRadioButton(self.tr("Entire open dataset"))
        self._scope_dataset.setChecked(True)
        scope_lay.addWidget(self._scope_current)
        scope_lay.addWidget(self._scope_dataset)
        layout.addWidget(scope_box)

        out_row = QtWidgets.QHBoxLayout()
        self._out_edit = QtWidgets.QLineEdit()
        self._out_edit.setStyleSheet(get_lineedit_style())
        self._out_edit.setPlaceholderText(self.tr("Output folder for PNG materials"))
        btn_browse = QtWidgets.QPushButton(self.tr("Browse…"))
        btn_browse.clicked.connect(self._pick_output)
        out_row.addWidget(self._out_edit, 1)
        out_row.addWidget(btn_browse)
        layout.addWidget(QtWidgets.QLabel(self.tr("Output directory:")))
        layout.addLayout(out_row)

        opt_form = QtWidgets.QFormLayout()
        self._padding_spin = QtWidgets.QSpinBox()
        self._padding_spin.setRange(0, 256)
        self._padding_spin.setValue(4)
        self._padding_spin.setStyleSheet(get_spinbox_style())
        self._obb_rectify = QtWidgets.QCheckBox(self.tr("Rectify oriented boxes (OBB)"))
        self._obb_rectify.setChecked(True)
        self._mask_dilate_spin = QtWidgets.QSpinBox()
        self._mask_dilate_spin.setRange(0, 128)
        self._mask_dilate_spin.setValue(0)
        self._mask_dilate_spin.setStyleSheet(get_spinbox_style())
        self._mask_dilate_spin.setToolTip(
            self.tr(
                "For polygon (segmentation) shapes only: expand the filled mask "
                "before cropping (elliptical dilation). Does not move vertices; "
                "use together with crop padding if the dilated mask hits the crop edge."
            )
        )
        opt_form.addRow(self.tr("Crop padding (px):"), self._padding_spin)
        opt_form.addRow(self.tr("Segmentation mask dilation (px):"), self._mask_dilate_spin)
        opt_form.addRow("", self._obb_rectify)
        layout.addLayout(opt_form)

        labels_box = QtWidgets.QGroupBox(self.tr("Labels to export"))
        labels_lay = QtWidgets.QVBoxLayout(labels_box)
        btn_row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton(self.tr("Select all"))
        btn_none = QtWidgets.QPushButton(self.tr("Select none"))
        btn_reload = QtWidgets.QPushButton(self.tr("Reload labels"))
        btn_all.clicked.connect(lambda: self._set_all_labels(True))
        btn_none.clicked.connect(lambda: self._set_all_labels(False))
        btn_reload.clicked.connect(self._reload_labels)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addWidget(btn_reload)
        btn_row.addStretch(1)
        labels_lay.addLayout(btn_row)
        self._label_list = QtWidgets.QListWidget()
        self._label_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        labels_lay.addWidget(self._label_list, 1)
        layout.addWidget(labels_box, 1)

        btn_run = QtWidgets.QPushButton(self.tr("Run"))
        btn_run.setStyleSheet(get_ok_btn_style())
        btn_run.clicked.connect(self._on_run)
        btn_cancel = QtWidgets.QPushButton(self.tr("Close"))
        btn_cancel.setStyleSheet(get_cancel_btn_style())
        btn_cancel.clicked.connect(self.reject)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(btn_cancel)
        bottom.addWidget(btn_run)
        layout.addLayout(bottom)

        default_out = _default_matting_output_dir(self._host)
        if default_out:
            self._out_edit.setText(default_out)

    def _on_mode_changed(self) -> None:
        fine = self._radio_fine.isChecked()
        key = default_model_key(fine=fine)
        idx = self._model_combo.findData(key)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)

    def _pick_output(self) -> None:
        start = self._out_edit.text().strip()
        if not start or not osp.isdir(start):
            start = _project_image_root(self._host) or "."
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select output folder"),
            start,
        )
        if path:
            self._out_edit.setText(path)

    def _reload_labels(self) -> None:
        self._label_list.clear()
        if self._host is None:
            return
        paths = list(getattr(self._host, "image_list", None) or [])
        output_dir = getattr(self._host, "output_dir", None)
        items = collect_available_labels_with_counts(
            paths, output_dir=output_dir, seed_labels=getattr(self._host, "label_list", None)
        )
        for name, count in items:
            text = f"{name}  ({count})"
            item = QtWidgets.QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self._label_list.addItem(item)

    def _set_all_labels(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self._label_list.count()):
            self._label_list.item(i).setCheckState(state)

    def _selected_labels(self) -> Optional[Set[str]]:
        labels: Set[str] = set()
        for i in range(self._label_list.count()):
            it = self._label_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                labels.add(str(it.data(Qt.ItemDataRole.UserRole)))
        if not labels:
            return None
        return labels

    def _image_paths(self) -> List[str]:
        if self._host is None:
            return []
        if self._scope_current.isChecked():
            fn = getattr(self._host, "filename", None)
            return [str(fn)] if fn and osp.isfile(fn) else []
        return list(getattr(self._host, "image_list", None) or [])

    def _on_run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Refined Matting"),
                self.tr("Matting is still running."),
            )
            return
        out_dir = self._out_edit.text().strip()
        if not out_dir:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Refined Matting"),
                self.tr("Please choose an output directory."),
            )
            return
        paths = self._image_paths()
        if not paths:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Refined Matting"),
                self.tr("No images to process."),
            )
            return
        labels = self._selected_labels()
        if labels is None:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Refined Matting"),
                self.tr("Select at least one label."),
            )
            return
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Refined Matting"),
                self.tr("Cannot create output directory:\n%s") % exc,
            )
            return

        mode = MODE_FINE if self._radio_fine.isChecked() else MODE_FAST
        model_key = str(self._model_combo.currentData() or default_model_key(fine=mode == MODE_FINE))
        if model_key not in MODEL_BY_KEY:
            model_key = default_model_key(fine=mode == MODE_FINE)

        self._user_cancelled = False
        self._progress = QtWidgets.QProgressDialog(
            self.tr("Loading model…"),
            self.tr("Cancel"),
            0,
            max(1, len(paths)),
            self,
        )
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.setAutoClose(False)
        self._progress.setAutoReset(False)
        self._progress.canceled.connect(self._on_progress_cancelled)
        self._progress.rejected.connect(self._on_progress_cancelled)
        self._progress.show()
        QtWidgets.QApplication.processEvents()

        if self._user_cancelled:
            self._close_progress()
            return

        from anylabeling.views.labeling.advanced.matting.backends.rmbg_backend import (
            create_matting_backend,
        )

        try:
            self._backend = create_matting_backend(model_key)
        except Exception as exc:
            self._close_progress()
            self._unload_backend()
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Refined Matting"),
                self.tr("Failed to load matting model:\n%s") % exc,
            )
            return

        if self._user_cancelled:
            self._unload_backend()
            self._close_progress()
            return

        self._progress.setLabelText(self.tr("Preparing…"))
        QtWidgets.QApplication.processEvents()

        self._worker = _MattingWorker(
            backend=self._backend,
            image_paths=paths,
            mode=mode,
            output_dir=out_dir,
            label_output_dir=getattr(self._host, "output_dir", None),
            allowed_labels=labels,
            padding_px=int(self._padding_spin.value()),
            obb_rectify=bool(self._obb_rectify.isChecked()),
            mask_dilate_px=int(self._mask_dilate_spin.value()),
        )
        _direct = Qt.ConnectionType.DirectConnection
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_thread_finished)
        self._progress.canceled.connect(self._worker.request_cancel, _direct)
        self._progress.rejected.connect(self._worker.request_cancel, _direct)
        self._backend = None
        self._worker.start()

    def _on_progress_cancelled(self) -> None:
        self._user_cancelled = True
        if self._worker is not None:
            try:
                self._worker.request_cancel()
            except Exception:
                pass

    def _close_progress(self) -> None:
        if self._progress is None:
            return
        try:
            self._progress.canceled.disconnect()
            self._progress.rejected.disconnect()
        except Exception:
            pass
        try:
            self._progress.close()
        except Exception:
            pass
        self._progress = None

    def _unload_backend(self) -> None:
        backend = self._backend
        self._backend = None
        if backend is None:
            return
        try:
            backend.unload()
        except Exception:
            pass

    def _on_worker_thread_finished(self) -> None:
        self._worker = None
        self._unload_backend()

    def _on_progress(self, current: int, total: int, name: str) -> None:
        if self._progress is None:
            return
        self._progress.setMaximum(max(1, total))
        self._progress.setValue(current)
        self._progress.setLabelText(
            self.tr("Processing %d / %d: %s") % (current, total, name)
        )

    def _on_finished(
        self, n_img: int, n_shapes: int, n_written: int, cancelled: bool
    ) -> None:
        self._close_progress()
        if cancelled or self._user_cancelled:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Refined Matting"),
                self.tr("Matting cancelled."),
            )
            return
        QtWidgets.QMessageBox.information(
            self,
            self.tr("Refined Matting"),
            self.tr(
                "Done. Processed %d image(s); %d shape(s) matched; wrote %d PNG crop(s).\n%s"
            )
            % (n_img, n_shapes, n_written, self._out_edit.text().strip()),
        )

    def _on_failed(self, msg: str) -> None:
        self._close_progress()
        QtWidgets.QMessageBox.critical(
            self,
            self.tr("Refined Matting"),
            self.tr("Matting failed:\n%s") % msg,
        )

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._user_cancelled = True
            try:
                self._worker.request_cancel()
                self._worker.wait(3000)
            except Exception:
                pass
        self._close_progress()
        self._unload_backend()
        super().closeEvent(event)


def open_refined_matting_dialog(parent) -> None:
    if parent is None:
        return
    if not getattr(parent, "image_list", None):
        QtWidgets.QMessageBox.information(
            parent,
            parent.tr("Refined Matting"),
            parent.tr("No images loaded."),
        )
        return
    dlg = RefinedMattingDialog(parent)
    dlg.exec()