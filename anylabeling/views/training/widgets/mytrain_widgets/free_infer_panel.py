"""Free Inference panel UI for Ultralytics training dialog."""

from __future__ import annotations

import os
import re
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from anylabeling.services.auto_training.mytrain.free_infer import (
    VIDEO_EXTS,
    WeightEntry,
    describe_infer_project,
)
from anylabeling.services.auto_training.mytrain.style import (
    get_log_display_style,
    get_progress_bar_style,
    get_ultralytics_dialog_style,
)
from anylabeling.views.labeling.utils.theme import get_theme
from anylabeling.services.auto_training.mytrain.predict_options import (
    PredictOptions,
)
from anylabeling.views.training.widgets.mytrain_widgets.custom_widgets import (
    CustomComboBox,
    CustomLineEdit,
    PrimaryButton,
    SecondaryButton,
)
from anylabeling.views.training.widgets.mytrain_widgets.predict_params_widget import (
    PredictParamsWidget,
)


class FreeInferPanel(QWidget):
    """Independent free-inference workspace (not tied to Data/Config/Train)."""

    request_scan_weights = pyqtSignal()
    request_start = pyqtSignal()
    request_download = pyqtSignal()
    request_cleanup = pyqtSignal()
    request_open_project = pyqtSignal()
    mode_changed = pyqtSignal(str)  # local | interpreter | remote

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.setStyleSheet(get_ultralytics_dialog_style())
        self._weight_entries: List[WeightEntry] = []
        self._uploaded_weight: Optional[WeightEntry] = None
        self._material_path = ""
        self._last_job_dir = ""
        self._last_source_dir = ""
        self._last_labels_dir = ""
        self._build_ui()

    def _build_ui(self) -> None:
        t = get_theme()
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        tip = QLabel(
            self.tr(
                "Free Inference is independent of Data / Config / Train. "
                "Pick an environment, weight, and material (image / folder / video), "
                "then run batch-style YOLO predict and download images + labels."
            )
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {t['text_secondary']};")
        root.addWidget(tip)

        # Environment (settings on Data tab; details shown below)
        env_group = QGroupBox(self.tr("Environment"))
        env_outer = QVBoxLayout(env_group)
        env_row = QHBoxLayout()
        self.env_combo = CustomComboBox()
        self.env_combo.addItem(self.tr("Local"), "local")
        self.env_combo.addItem(self.tr("Interpreter"), "interpreter")
        self.env_combo.addItem(self.tr("Remote"), "remote")
        self.env_combo.currentIndexChanged.connect(self._on_env_changed)
        env_row.addWidget(QLabel(self.tr("Mode:")))
        env_row.addWidget(self.env_combo, 1)
        env_outer.addLayout(env_row)
        env_hint = QLabel(
            self.tr(
                "Same Local / Interpreter / Remote as Training Mode on the Data tab."
            )
        )
        env_hint.setWordWrap(True)
        env_hint.setStyleSheet(f"color: {t['text_secondary']}; font-size: 12px;")
        env_outer.addWidget(env_hint)
        self.env_detail_label = QLabel("—")
        self.env_detail_label.setWordWrap(True)
        self.env_detail_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.env_detail_label.setStyleSheet(
            f"color: {t['text']}; font-size: 12px; padding: 4px 0;"
        )
        env_outer.addWidget(self.env_detail_label)
        root.addWidget(env_group)

        # Weights
        w_group = QGroupBox(self.tr("Weights"))
        w_form = QFormLayout(w_group)
        weight_row = QHBoxLayout()
        self.weight_combo = CustomComboBox()
        self.weight_combo.setMinimumWidth(120)
        weight_row.addWidget(self.weight_combo, 1)
        self.scan_btn = SecondaryButton(self.tr("Scan Projects"))
        self.scan_btn.clicked.connect(self.request_scan_weights.emit)
        weight_row.addWidget(self.scan_btn)
        self.upload_weight_btn = SecondaryButton(self.tr("Upload Local Weight"))
        self.upload_weight_btn.clicked.connect(self._browse_weight)
        weight_row.addWidget(self.upload_weight_btn)
        w_form.addRow(self.tr("Weight:"), weight_row)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        w_form.addRow(self.tr("Image size:"), self.imgsz_spin)

        # Material
        m_group = QGroupBox(self.tr("Inference Material"))
        m_layout = QVBoxLayout(m_group)
        kind_row = QHBoxLayout()
        self.kind_group = QButtonGroup(self)
        self.kind_image = QRadioButton(self.tr("Single image"))
        self.kind_folder = QRadioButton(self.tr("Image folder"))
        self.kind_video = QRadioButton(self.tr("Video"))
        self.kind_folder.setChecked(True)
        for i, btn in enumerate(
            (self.kind_image, self.kind_folder, self.kind_video)
        ):
            self.kind_group.addButton(btn, i)
            kind_row.addWidget(btn)
        kind_row.addStretch()
        m_layout.addLayout(kind_row)

        path_row = QHBoxLayout()
        self.material_edit = CustomLineEdit()
        self.material_edit.setReadOnly(True)
        self.material_edit.setPlaceholderText(
            self.tr("Select image / folder / video…")
        )
        browse_mat = SecondaryButton(self.tr("Browse"))
        browse_mat.clicked.connect(self._browse_material)
        path_row.addWidget(self.material_edit, 1)
        path_row.addWidget(browse_mat)
        m_layout.addLayout(path_row)

        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel(self.tr("Video frame interval:")))
        self.video_interval = QSpinBox()
        self.video_interval.setRange(1, 1000)
        self.video_interval.setValue(1)
        self.video_interval.setToolTip(
            self.tr("Keep 1 frame every N frames when extracting video.")
        )
        interval_row.addWidget(self.video_interval)
        interval_row.addStretch()
        m_layout.addLayout(interval_row)

        w_group.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        m_group.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        weights_material_row = QHBoxLayout()
        weights_material_row.setSpacing(12)
        weights_material_row.addWidget(w_group, 1)
        weights_material_row.addWidget(m_group, 1)
        root.addLayout(weights_material_row)

        self.predict_params = PredictParamsWidget(
            self, preset="default", show_output_options=True
        )
        root.addWidget(self.predict_params, 0)

        # Actions
        action_row = QHBoxLayout()
        self.start_btn = PrimaryButton(self.tr("Start Inference"))
        self.start_btn.clicked.connect(self.request_start.emit)
        self.download_btn = SecondaryButton(
            self.tr("Download Images + Labels")
        )
        self.download_btn.clicked.connect(self.request_download.emit)
        self.download_btn.setEnabled(False)
        self.cleanup_btn = SecondaryButton(self.tr("Cleanup Projects…"))
        self.cleanup_btn.clicked.connect(self.request_cleanup.emit)
        self.open_project_btn = SecondaryButton(self.tr("Open Project…"))
        self.open_project_btn.clicked.connect(self.request_open_project.emit)
        action_row.addWidget(self.start_btn)
        action_row.addWidget(self.download_btn)
        action_row.addWidget(self.cleanup_btn)
        action_row.addWidget(self.open_project_btn)
        action_row.addStretch()
        root.addLayout(action_row)

        self.status_label = QLabel(self.tr("Ready."))
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {t['text_secondary']};")
        root.addWidget(self.status_label)

        log_group = QGroupBox(self.tr("Inference log"))
        log_layout = QVBoxLayout(log_group)
        self.infer_progress = QProgressBar()
        self.infer_progress.setRange(0, 100)
        self.infer_progress.setValue(0)
        self.infer_progress.setFormat(self.tr("Idle"))
        self.infer_progress.setStyleSheet(get_progress_bar_style())
        log_layout.addWidget(self.infer_progress)
        self.infer_log = QTextEdit()
        self.infer_log.setReadOnly(True)
        self.infer_log.setMinimumHeight(140)
        self.infer_log.setStyleSheet(get_log_display_style())
        log_layout.addWidget(self.infer_log, 1)
        root.addWidget(log_group, 1)

    def _on_env_changed(self, _index: int = 0) -> None:
        mode = self.env_combo.currentData() or "local"
        self.mode_changed.emit(str(mode))

    def set_environment_detail(self, text: str) -> None:
        self.env_detail_label.setText((text or "").strip() or "—")

    def sync_env_mode(self, mode: str) -> None:
        mapping = {"local": 0, "interpreter": 1, "remote": 2}
        idx = mapping.get((mode or "local").lower(), 0)
        if self.env_combo.currentIndex() == idx:
            return
        self.env_combo.blockSignals(True)
        self.env_combo.setCurrentIndex(idx)
        self.env_combo.blockSignals(False)

    def current_env_mode(self) -> str:
        return str(self.env_combo.currentData() or "local")

    def set_status(self, text: str) -> None:
        self.status_label.setText(text or "")

    def clear_infer_log(self) -> None:
        self.infer_log.clear()
        self.infer_progress.setRange(0, 100)
        self.infer_progress.setValue(0)
        self.infer_progress.setFormat(self.tr("Idle"))

    def append_infer_log(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            self.infer_log.append(line)
            self._note_infer_progress(line)
        bar = self.infer_log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def mark_infer_finished(self, ok: bool) -> None:
        if ok and self.infer_progress.maximum() > 0:
            self.infer_progress.setValue(self.infer_progress.maximum())
        self.infer_progress.setFormat(
            self.tr("Done") if ok else self.tr("Failed")
        )

    def _note_infer_progress(self, line: str) -> None:
        total_match = re.match(r"PREDICT_TOTAL\s+(\d+)", line)
        if total_match:
            total = int(total_match.group(1))
            if total > 0:
                self.infer_progress.setRange(0, total)
                self.infer_progress.setValue(0)
                self.infer_progress.setFormat(f"0/{total}")
            return
        step = re.match(
            r"(?:PREDICT|Uploading)\s+(\d+)\s*/\s*(\d+)",
            line,
        )
        if not step:
            return
        current = int(step.group(1))
        total = int(step.group(2))
        if total <= 0:
            return
        self.infer_progress.setRange(0, total)
        self.infer_progress.setValue(min(current, total))
        self.infer_progress.setFormat(f"{current}/{total}")

    def set_weight_entries(self, entries: List[WeightEntry]) -> None:
        self._weight_entries = list(entries or [])
        self.weight_combo.clear()
        for entry in self._weight_entries:
            self.weight_combo.addItem(entry.display, entry.display)
        if self._uploaded_weight is not None:
            self.weight_combo.insertItem(
                0,
                self.tr("[Upload] ") + self._uploaded_weight.display,
                "__upload__",
            )
            self.weight_combo.setCurrentIndex(0)
        elif self.weight_combo.count() > 0:
            self.weight_combo.setCurrentIndex(0)

    def current_weight(self) -> Optional[WeightEntry]:
        data = self.weight_combo.currentData()
        if data == "__upload__":
            return self._uploaded_weight
        display = self.weight_combo.currentText()
        for entry in self._weight_entries:
            if entry.display == display or entry.display == data:
                return entry
        return None

    def imgsz(self) -> int:
        return int(self.imgsz_spin.value())

    def predict_options(self) -> PredictOptions:
        return self.predict_params.options_with_imgsz(self.imgsz())

    def video_frame_interval(self) -> int:
        return int(self.video_interval.value())

    def material_path(self) -> str:
        return (self._material_path or self.material_edit.text() or "").strip()

    def material_kind_hint(self) -> str:
        if self.kind_image.isChecked():
            return "image"
        if self.kind_video.isChecked():
            return "video"
        return "folder"

    def set_last_result(
        self, job_dir: str, source_dir: str, labels_dir: str
    ) -> None:
        self._last_job_dir = job_dir or ""
        self._last_source_dir = source_dir or ""
        self._last_labels_dir = labels_dir or ""
        self.download_btn.setEnabled(bool(source_dir and labels_dir))

    def last_source_dir(self) -> str:
        return self._last_source_dir

    def last_labels_dir(self) -> str:
        return self._last_labels_dir

    def last_job_dir(self) -> str:
        return self._last_job_dir

    def set_busy(self, busy: bool) -> None:
        self.start_btn.setEnabled(not busy)
        self.scan_btn.setEnabled(not busy)
        self.upload_weight_btn.setEnabled(not busy)
        self.cleanup_btn.setEnabled(not busy)
        self.open_project_btn.setEnabled(not busy)
        if not busy and self._last_source_dir and self._last_labels_dir:
            self.download_btn.setEnabled(True)
        elif busy:
            self.download_btn.setEnabled(False)

    def _browse_weight(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select weight file"),
            "",
            self.tr("Weights (*.pt *.onnx);;All files (*)"),
        )
        if not path:
            return
        name = os.path.basename(path)
        self._uploaded_weight = WeightEntry(
            display=name,
            project_id="upload",
            run_dir="",
            weight_name=name,
            weight_path=path,
            source="upload",
            remote=False,
        )
        self.set_weight_entries(self._weight_entries)
        self.set_status(
            self.tr("Uploaded weight selected: {path}").format(path=path)
        )

    def _browse_material(self) -> None:
        kind = self.material_kind_hint()
        if kind == "image":
            path, _ = QFileDialog.getOpenFileName(
                self,
                self.tr("Select image"),
                "",
                self.tr(
                    "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;"
                    "All files (*)"
                ),
            )
        elif kind == "video":
            filters = " ".join(f"*{e}" for e in sorted(VIDEO_EXTS))
            path, _ = QFileDialog.getOpenFileName(
                self,
                self.tr("Select video"),
                "",
                self.tr("Videos ({filters});;All files (*)").format(
                    filters=filters
                ),
            )
        else:
            path = QFileDialog.getExistingDirectory(
                self, self.tr("Select image folder"), ""
            )
        if path:
            self._material_path = path
            self.material_edit.setText(path)


class CleanupProjectsDialog(QDialog):
    """Pick training or free-infer projects and delete selected ones."""

    def __init__(
        self,
        parent=None,
        *,
        train_projects: Optional[List[tuple]] = None,
        infer_projects: Optional[List[tuple]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Cleanup Projects"))
        self.setMinimumSize(560, 420)
        self.setModal(True)
        self.setStyleSheet(get_ultralytics_dialog_style())
        self._selected: List[tuple] = []

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                self.tr(
                    "Select projects to delete. Training and free-inference "
                    "projects are listed separately. Nothing is deleted until "
                    "you confirm."
                )
            )
        )

        self.scope_combo = CustomComboBox()
        self.scope_combo.addItem(self.tr("Training projects"), "train")
        self.scope_combo.addItem(self.tr("Free inference projects"), "infer")
        self.scope_combo.currentIndexChanged.connect(self._reload_list)
        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr("Scope:")))
        row.addWidget(self.scope_combo, 1)
        layout.addLayout(row)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )
        layout.addWidget(self.list_widget, 1)

        self._train = list(train_projects or [])
        self._infer = list(infer_projects or [])
        self._reload_list()

        btns = QHBoxLayout()
        btns.addStretch()
        cancel = SecondaryButton(self.tr("Cancel"))
        cancel.clicked.connect(self.reject)
        ok = PrimaryButton(self.tr("Delete Selected"))
        ok.clicked.connect(self._on_accept)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        layout.addLayout(btns)

    def _reload_list(self) -> None:
        self.list_widget.clear()
        scope = self.scope_combo.currentData() or "train"
        items = self._train if scope == "train" else self._infer
        for project_id, path in items:
            item = QListWidgetItem(f"{project_id}\n{path}")
            item.setData(Qt.ItemDataRole.UserRole, (project_id, path))
            self.list_widget.addItem(item)

    def _on_accept(self) -> None:
        selected = self.list_widget.selectedItems()
        if not selected:
            QMessageBox.warning(
                self,
                self.tr("Cleanup"),
                self.tr("Please select at least one project."),
            )
            return
        self._selected = [
            item.data(Qt.ItemDataRole.UserRole) for item in selected
        ]
        names = "\n".join(p[0] for p in self._selected)
        reply = QMessageBox.question(
            self,
            self.tr("Confirm delete"),
            self.tr(
                "Permanently delete the following project(s)?\n\n{names}"
            ).format(names=names),
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.accept()

    def selected_projects(self) -> List[tuple]:
        return list(self._selected)


class OpenInferProjectDialog(QDialog):
    """Pick one free-inference project to view."""

    def __init__(
        self,
        parent=None,
        *,
        infer_projects: Optional[List[tuple]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Open Inference Project"))
        self.setMinimumSize(560, 420)
        self.setModal(True)
        self.setStyleSheet(get_ultralytics_dialog_style())
        self._selected: Optional[tuple] = None

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                self.tr(
                    "Select a free-inference project to view its images and "
                    "detection results."
                )
            )
        )
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection
        )
        self.list_widget.itemDoubleClicked.connect(self._on_accept)
        for project_id, path in infer_projects or []:
            item = QListWidgetItem(f"{project_id}\n{path}")
            item.setData(Qt.ItemDataRole.UserRole, (project_id, path))
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget, 1)

        btns = QHBoxLayout()
        btns.addStretch()
        cancel = SecondaryButton(self.tr("Cancel"))
        cancel.clicked.connect(self.reject)
        ok = PrimaryButton(self.tr("Open"))
        ok.clicked.connect(self._on_accept)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        layout.addLayout(btns)

    def _on_accept(self) -> None:
        items = self.list_widget.selectedItems()
        if not items:
            QMessageBox.warning(
                self,
                self.tr("Open Project"),
                self.tr("Please select a project."),
            )
            return
        self._selected = items[0].data(Qt.ItemDataRole.UserRole)
        self.accept()

    def selected_project(self) -> Optional[tuple]:
        return self._selected


def _read_yolo_boxes(label_path: str) -> List[tuple]:
    boxes = []
    if not label_path or not os.path.isfile(label_path):
        return boxes
    try:
        text = open(label_path, "r", encoding="utf-8").read()
    except OSError:
        return boxes
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(parts[i]) for i in range(1, 5))
        except ValueError:
            continue
        boxes.append((cid, cx, cy, w, h))
    return boxes


class InferResultViewer(QDialog):
    """Browse one inference project's images, with YOLO boxes when needed."""

    def __init__(
        self,
        parent=None,
        *,
        project_id: str,
        project_path: str,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Inference results — {name}").format(name=project_id))
        self.setMinimumSize(960, 640)
        self.setStyleSheet(get_ultralytics_dialog_style())
        info = describe_infer_project(project_path)
        self._images: List[str] = list(info.get("images") or [])
        self._labels_dir = str(info.get("labels_dir") or "")
        self._annotated = bool(info.get("annotated"))
        self._index = 0

        layout = QVBoxLayout(self)
        self.summary = QLabel("")
        layout.addWidget(self.summary)
        body = QHBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.setMinimumWidth(220)
        for path in self._images:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.list_widget.addItem(item)
        self.list_widget.currentRowChanged.connect(self._show_index)
        body.addWidget(self.list_widget)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        body.addWidget(self.image_label, 1)
        layout.addLayout(body, 1)

        nav = QHBoxLayout()
        prev_btn = SecondaryButton(self.tr("Previous"))
        next_btn = SecondaryButton(self.tr("Next"))
        prev_btn.clicked.connect(lambda: self._step(-1))
        next_btn.clicked.connect(lambda: self._step(1))
        close_btn = PrimaryButton(self.tr("Close"))
        close_btn.clicked.connect(self.accept)
        nav.addWidget(prev_btn)
        nav.addWidget(next_btn)
        nav.addStretch()
        nav.addWidget(close_btn)
        layout.addLayout(nav)
        if self._images:
            self.list_widget.setCurrentRow(0)
        else:
            self._refresh_summary()

    def _step(self, delta: int) -> None:
        if not self._images:
            return
        self._show_index((self._index + delta) % len(self._images))
        self.list_widget.setCurrentRow(self._index)

    def _show_index(self, index: int) -> None:
        if index < 0 or index >= len(self._images):
            return
        self._index = index
        self._refresh_summary()
        path = self._images[index]
        pix = QPixmap(path)
        if pix.isNull():
            self.image_label.setText(self.tr("Cannot display this image."))
            return
        if not self._annotated:
            pix = self._draw_boxes(pix, path)
        target = self.image_label.size()
        if target.width() > 40 and target.height() > 40:
            pix = pix.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.image_label.setPixmap(pix)

    def _refresh_summary(self) -> None:
        total = len(self._images)
        current = self._index + 1 if total else 0
        name = os.path.basename(self._images[self._index]) if total else "—"
        mode = (
            self.tr("annotated images")
            if self._annotated
            else self.tr("source images with label boxes")
        )
        self.summary.setText(
            self.tr("{current}/{total} · {name} · {mode}").format(
                current=current,
                total=total,
                name=name,
                mode=mode,
            )
        )

    def _draw_boxes(self, pix: QPixmap, image_path: str) -> QPixmap:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        boxes = _read_yolo_boxes(os.path.join(self._labels_dir, stem + ".txt"))
        if not boxes:
            return pix
        canvas = QPixmap(pix)
        painter = QPainter(canvas)
        pen = QPen(QColor(0, 220, 80))
        pen.setWidth(max(2, canvas.width() // 400))
        painter.setPen(pen)
        w, h = canvas.width(), canvas.height()
        for cid, cx, cy, bw, bh in boxes:
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            painter.drawRect(int(x1), int(y1), int(bw * w), int(bh * h))
            painter.drawText(int(x1), max(12, int(y1) - 4), str(cid))
        painter.end()
        return canvas

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._images:
            self._show_index(self._index)
