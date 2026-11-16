"""Free Inference panel UI for Ultralytics training dialog."""

from __future__ import annotations

import os
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
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
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from anylabeling.services.auto_training.mytrain.free_infer import (
    VIDEO_EXTS,
    WeightEntry,
)
from anylabeling.services.auto_training.mytrain.style import (
    get_ultralytics_dialog_style,
)
from anylabeling.views.labeling.utils.theme import get_theme
from anylabeling.views.training.widgets.mytrain_widgets.custom_widgets import (
    CustomComboBox,
    CustomLineEdit,
    PrimaryButton,
    SecondaryButton,
)


class FreeInferPanel(QWidget):
    """Independent free-inference workspace (not tied to Data/Config/Train)."""

    request_scan_weights = pyqtSignal()
    request_start = pyqtSignal()
    request_download = pyqtSignal()
    request_cleanup = pyqtSignal()
    mode_changed = pyqtSignal(str)  # local | interpreter | remote

    def __init__(self, parent=None):
        super().__init__(parent)
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

        # Environment
        env_group = QGroupBox(self.tr("Environment"))
        env_layout = QHBoxLayout(env_group)
        self.env_combo = CustomComboBox()
        self.env_combo.addItem(self.tr("Local"), "local")
        self.env_combo.addItem(self.tr("Interpreter"), "interpreter")
        self.env_combo.addItem(self.tr("Remote"), "remote")
        self.env_combo.currentIndexChanged.connect(self._on_env_changed)
        env_layout.addWidget(QLabel(self.tr("Mode:")))
        env_layout.addWidget(self.env_combo, 1)
        env_hint = QLabel(
            self.tr(
                "Uses the same Local / Interpreter / Remote settings as Training Mode "
                "(interpreter list & SSH profiles on the Data tab)."
            )
        )
        env_hint.setWordWrap(True)
        env_layout.addWidget(env_hint, 2)
        root.addWidget(env_group)

        # Weights
        w_group = QGroupBox(self.tr("Weights"))
        w_form = QFormLayout(w_group)
        weight_row = QHBoxLayout()
        self.weight_combo = CustomComboBox()
        self.weight_combo.setMinimumWidth(360)
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
        root.addWidget(w_group)

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
        root.addWidget(m_group)

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
        action_row.addWidget(self.start_btn)
        action_row.addWidget(self.download_btn)
        action_row.addWidget(self.cleanup_btn)
        action_row.addStretch()
        root.addLayout(action_row)

        self.status_label = QLabel(self.tr("Ready."))
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {t['text_secondary']};")
        root.addWidget(self.status_label)
        root.addStretch(1)

    def _on_env_changed(self, _index: int = 0) -> None:
        mode = self.env_combo.currentData() or "local"
        self.mode_changed.emit(str(mode))

    def current_env_mode(self) -> str:
        return str(self.env_combo.currentData() or "local")

    def set_status(self, text: str) -> None:
        self.status_label.setText(text or "")

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
        self.set_status(self.tr(f"Uploaded weight selected: {path}"))

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
                self.tr(f"Videos ({filters});;All files (*)"),
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
