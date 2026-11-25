from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
)
from typing import List, Optional
import os

from .custom_widgets import (
    CustomComboBox,
    CustomLineEdit,
    PrimaryButton,
    SecondaryButton,
)
from anylabeling.services.auto_training.mytrain.export_naming import (
    build_onnx_export_name,
)
from anylabeling.services.auto_training.mytrain.predict_options import (
    PredictOptions,
)
from anylabeling.views.training.widgets.mytrain_widgets.predict_params_widget import (
    PredictParamsWidget,
)
from anylabeling.services.auto_training.mytrain.style import (
    get_ultralytics_dialog_style,
)
from anylabeling.views.labeling.utils.theme import get_theme


class ExportFormatDialog(QDialog):
    """Export settings: weight source, ONNX naming fields, optional format."""

    def __init__(
        self,
        parent=None,
        *,
        remote: bool = False,
        model_stem: str = "model",
        imgsz: int = 640,
        num_classes: int = 0,
        project: str = "project",
        subdir: str = "data",
        version: str = "v1",
    ):
        super().__init__(parent)
        self._remote = remote
        self._model_stem = model_stem or "model"
        self._imgsz = int(imgsz) if imgsz else 640
        self._num_classes = int(num_classes) if num_classes else 0
        self.setWindowTitle(self.tr("Export Settings"))
        self.setMinimumWidth(480)
        self.setModal(True)
        self.setStyleSheet(get_ultralytics_dialog_style())

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(24, 24, 24, 24)

        t = get_theme()
        desc = QLabel(
            self.tr(
                "Convert weights to ONNX and save under the run's weights/ folder."
            )
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"color: {t['text_secondary']}; margin-bottom: 8px;"
        )
        layout.addWidget(desc)

        form = QFormLayout()
        form.setSpacing(8)

        self.weight_combo = CustomComboBox()
        self.weight_combo.addItem("best.pt", "best.pt")
        self.weight_combo.addItem("last.pt", "last.pt")
        form.addRow(self.tr("Weights:"), self.weight_combo)

        self.format_combo = CustomComboBox()
        if remote:
            self.format_combo.addItem("ONNX", "onnx")
        else:
            formats = [
                ("ONNX", "onnx"),
                ("TorchScript", "torchscript"),
                ("OpenVINO", "openvino"),
                ("TensorRT", "engine"),
                ("CoreML", "coreml"),
                ("TensorFlow SavedModel", "saved_model"),
                ("TensorFlow Lite", "tflite"),
                ("TensorFlow Edge TPU", "edgetpu"),
                ("TensorFlow.js", "tfjs"),
                ("PaddlePaddle", "paddle"),
                ("MNN", "mnn"),
                ("NCNN", "ncnn"),
                ("IMX500", "imx"),
                ("RKNN", "rknn"),
            ]
            for display_name, format_code in formats:
                self.format_combo.addItem(display_name, format_code)
        self.format_combo.setCurrentIndex(0)
        self.format_combo.currentIndexChanged.connect(self._update_preview)
        form.addRow(self.tr("Format:"), self.format_combo)

        # Auto from Data/Config page — display only
        self._project = project or "project"
        self._subdir = subdir or "data"
        self.project_label = QLabel(self._project)
        self.project_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow(self.tr("Project name:"), self.project_label)

        self.subdir_label = QLabel(self._subdir)
        self.subdir_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow(self.tr("Sub-dataset version:"), self.subdir_label)

        self.version_edit = CustomLineEdit()
        self.version_edit.setText(version or "v1")
        self.version_edit.setPlaceholderText("v1")
        self.version_edit.textChanged.connect(self._update_preview)
        form.addRow(self.tr("Model version:"), self.version_edit)

        layout.addLayout(form)

        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.preview_label.setStyleSheet(
            f"color: {t.get('text_primary', t['text_secondary'])}; "
            f"font-size: 12px; margin-top: 4px;"
        )
        layout.addWidget(self.preview_label)

        info_label = QLabel(
            self.tr(
                "ONNX will be written to: <run>/weights/<name>.onnx"
            )
        )
        info_label.setStyleSheet(f"""
            color: {t['warning']};
            font-size: 12px;
            margin-top: 4px;
        """)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.cancel_btn = SecondaryButton(self.tr("Cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        self.ok_btn = PrimaryButton(self.tr("Export"))
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        self._update_preview()

    def _update_preview(self) -> None:
        fmt = self.format_combo.currentData() or "onnx"
        if fmt != "onnx":
            self.preview_label.setText(
                self.tr("Preview: default export name for selected format")
            )
            return
        name = build_onnx_export_name(
            self._model_stem,
            self._imgsz,
            self._num_classes,
            self._project,
            self._subdir,
            self.version_edit.text(),
        )
        self.preview_label.setText(
            self.tr("Preview: {name}").format(name=name)
        )

    def get_selected_format(self):
        return self.format_combo.currentData() or "onnx"

    def get_weights_file(self) -> str:
        return self.weight_combo.currentData() or "best.pt"

    def get_naming_fields(self) -> dict:
        return {
            "model_stem": self._model_stem,
            "imgsz": self._imgsz,
            "num_classes": self._num_classes,
            "project": self._project,
            "subdir": self._subdir,
            "version": self.version_edit.text().strip() or "v1",
        }

    def get_onnx_filename(self) -> str:
        fields = self.get_naming_fields()
        return build_onnx_export_name(
            fields["model_stem"],
            fields["imgsz"],
            fields["num_classes"],
            fields["project"],
            fields["subdir"],
            fields["version"],
        )


class RemotePredictDialog(QDialog):
    """Choose weights for dataset-wide YOLO predict (txt only, no images)."""

    def __init__(
        self,
        parent=None,
        *,
        weight_choices: Optional[List[str]] = None,
        source_all_txt: str = "",
        output_labels_dir: str = "",
        imgsz: int = 640,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Remote Infer"))
        self.setMinimumWidth(520)
        self.setMinimumHeight(640)
        self.setModal(True)
        self.setStyleSheet(get_ultralytics_dialog_style())
        self._imgsz = int(imgsz) if imgsz else 640

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(24, 24, 24, 24)

        t = get_theme()
        desc = QLabel(
            self.tr(
                "Run YOLO predict on the full dataset (all.txt). "
                "Only YOLO-format txt labels are saved under the training "
                "folder (no annotated images)."
            )
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"color: {t['text_secondary']}; margin-bottom: 8px;"
        )
        layout.addWidget(desc)

        form = QFormLayout()
        form.setSpacing(8)

        self.weight_combo = CustomComboBox()
        choices = weight_choices or ["best.pt", "last.pt"]
        for name in choices:
            self.weight_combo.addItem(name, name)
        form.addRow(self.tr("Weights:"), self.weight_combo)

        self.source_label = QLabel(source_all_txt or "-")
        self.source_label.setWordWrap(True)
        self.source_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow(self.tr("Source (all.txt):"), self.source_label)

        self.output_label = QLabel(output_labels_dir or "-")
        self.output_label.setWordWrap(True)
        self.output_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow(self.tr("Labels output:"), self.output_label)

        self.imgsz_label = QLabel(str(self._imgsz))
        form.addRow(self.tr("Image size:"), self.imgsz_label)

        layout.addLayout(form)
        self.predict_params = PredictParamsWidget(
            self, preset="labels_only", show_output_options=True
        )
        layout.addWidget(self.predict_params)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = SecondaryButton(self.tr("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        ok_btn = PrimaryButton(self.tr("Start Infer"))
        ok_btn.clicked.connect(self.accept)
        buttons.addWidget(ok_btn)
        layout.addLayout(buttons)
        self.setLayout(layout)

    def get_weights_file(self) -> str:
        return self.weight_combo.currentData() or "best.pt"

    def get_imgsz(self) -> int:
        return self._imgsz

    def get_predict_options(self) -> PredictOptions:
        return self.predict_params.options_with_imgsz(self._imgsz)


class BatchInferDialog(QDialog):
    """Pick a local image folder and weights for batch YOLO predict."""

    def __init__(
        self,
        parent=None,
        *,
        weight_choices: Optional[List[str]] = None,
        output_labels_dir: str = "",
        imgsz: int = 640,
        remote: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Batch Infer"))
        self.setMinimumWidth(560)
        self.setMinimumHeight(640)
        self.setModal(True)
        self.setStyleSheet(get_ultralytics_dialog_style())
        self._imgsz = int(imgsz) if imgsz else 640
        self._remote = bool(remote)
        self._folder = ""

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(24, 24, 24, 24)

        t = get_theme()
        desc = QLabel(
            self.tr(
                "Select a local folder of images. YOLO will write one txt "
                "label per image (same stem) under the results folder. "
                "Annotated images are not saved."
            )
            + (
                self.tr(" Remote: folder is uploaded first, then inferred.")
                if self._remote
                else ""
            )
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"color: {t['text_secondary']}; margin-bottom: 8px;"
        )
        layout.addWidget(desc)

        form = QFormLayout()
        form.setSpacing(8)

        self.weight_combo = CustomComboBox()
        choices = weight_choices or ["best.pt", "last.pt"]
        for name in choices:
            self.weight_combo.addItem(name, name)
        form.addRow(self.tr("Weights:"), self.weight_combo)

        folder_row = QHBoxLayout()
        self.folder_edit = CustomLineEdit()
        self.folder_edit.setPlaceholderText(
            self.tr("Local image folder…")
        )
        self.folder_edit.setReadOnly(True)
        browse_btn = SecondaryButton(self.tr("Browse"))
        browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(browse_btn)
        form.addRow(self.tr("Image folder:"), folder_row)

        self.output_label = QLabel(output_labels_dir or "-")
        self.output_label.setWordWrap(True)
        self.output_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow(self.tr("Labels output:"), self.output_label)

        self.imgsz_label = QLabel(str(self._imgsz))
        form.addRow(self.tr("Image size:"), self.imgsz_label)

        layout.addLayout(form)
        self.predict_params = PredictParamsWidget(
            self, preset="labels_only", show_output_options=True
        )
        layout.addWidget(self.predict_params)

        hint = QLabel(
            self.tr(
                'After finish, use "Download Batch Results" to save the labels folder.'
            )
        )
        hint.setStyleSheet(f"color: {t['text_secondary']}; font-size: 12px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = SecondaryButton(self.tr("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        self.ok_btn = PrimaryButton(self.tr("Start Batch Infer"))
        self.ok_btn.clicked.connect(self._on_accept)
        buttons.addWidget(self.ok_btn)
        layout.addLayout(buttons)
        self.setLayout(layout)

    def _browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select image folder"),
            self.folder_edit.text().strip() or "",
        )
        if path:
            self._folder = path
            self.folder_edit.setText(path)

    def _on_accept(self) -> None:
        folder = (self.folder_edit.text() or self._folder or "").strip()
        if not folder or not os.path.isdir(folder):
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                self.tr("Batch Infer"),
                self.tr("Please select a valid local image folder."),
            )
            return
        self._folder = folder
        self.accept()

    def get_weights_file(self) -> str:
        return self.weight_combo.currentData() or "best.pt"

    def get_image_folder(self) -> str:
        return (self._folder or self.folder_edit.text() or "").strip()

    def get_predict_options(self) -> PredictOptions:
        return self.predict_params.options_with_imgsz(self._imgsz)

    def get_imgsz(self) -> int:
        return self._imgsz


class RemoteDirBrowserDialog(QDialog):
    """Browse remote directories over SSH (single selection for exec cwd)."""

    def __init__(self, parent=None, ssh_client=None, start_path: str = ""):
        super().__init__(parent)
        self._client = ssh_client
        self._cwd = (start_path or "").strip().replace("\\", "/")
        if not self._cwd:
            self._cwd = getattr(ssh_client, "home", "") or "/"
        self._selected = self._cwd
        self.setWindowTitle(self.tr("Remote directory"))
        self.setMinimumSize(520, 420)
        self.setModal(True)
        self.setStyleSheet(get_ultralytics_dialog_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        self.path_label = QLabel(self._cwd)
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        self.dir_list = QListWidget()
        self.dir_list.itemDoubleClicked.connect(self._enter_dir)
        layout.addWidget(self.dir_list)

        btn_row = QHBoxLayout()
        up_btn = SecondaryButton(self.tr("Up"))
        up_btn.clicked.connect(self._go_up)
        btn_row.addWidget(up_btn)
        btn_row.addStretch()
        cancel_btn = SecondaryButton(self.tr("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        ok_btn = PrimaryButton(self.tr("Select"))
        ok_btn.clicked.connect(self._accept_selection)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        self._refresh_listing()

    def _refresh_listing(self) -> None:
        self.path_label.setText(self._cwd)
        self.dir_list.clear()
        client = self._client
        if client is None or not getattr(client, "connected", False):
            return
        parent = posix_dirname(self._cwd)
        if parent and parent != self._cwd:
            item = QListWidgetItem(
                "..  ({parent})".format(parent=self.tr("parent"))
            )
            item.setData(256, parent)
            self.dir_list.addItem(item)
        try:
            names = client.listdir(self._cwd)
        except Exception:
            names = []
        for name in sorted(names):
            if name in (".", ".."):
                continue
            child = self._cwd.rstrip("/") + "/" + name
            is_dir = (
                client.is_dir(child)
                if hasattr(client, "is_dir")
                else client.exists(child)
            )
            if is_dir:
                item = QListWidgetItem(name + "/")
                item.setData(256, child)
                self.dir_list.addItem(item)

    def _enter_dir(self, item: QListWidgetItem) -> None:
        path = item.data(256)
        if path:
            self._cwd = str(path).replace("\\", "/")
            self._selected = self._cwd
            self._refresh_listing()

    def _go_up(self) -> None:
        parent = posix_dirname(self._cwd)
        if parent and parent != self._cwd:
            self._cwd = parent
            self._selected = self._cwd
            self._refresh_listing()

    def _accept_selection(self) -> None:
        cur = self.dir_list.currentItem()
        if cur is not None:
            path = cur.data(256)
            parent_tag = "..  ({parent})".format(parent=self.tr("parent"))
            if path and cur.text() != parent_tag:
                self._selected = str(path).replace("\\", "/")
            else:
                self._selected = self._cwd
        else:
            self._selected = self._cwd
        self.accept()

    def selected_path(self) -> str:
        return self._selected or self._cwd


def posix_dirname(path: str) -> str:
    path = (path or "/").replace("\\", "/").rstrip("/") or "/"
    if path == "/":
        return "/"
    idx = path.rfind("/")
    if idx <= 0:
        return "/"
    return path[:idx] or "/"