"""Dialog to run batch dataset image analysis and write results to label JSON."""

from __future__ import annotations

import json
import os.path as osp
import traceback

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from anylabeling.views.labeling.classifier.utils import get_label_file_path
from anylabeling.views.labeling.advanced.dataset_analysis.aggregate import (
    aggregate_numeric_metrics,
    build_diversity_summary,
    build_optimization_suggestions,
    collect_resolution_counts,
    write_summary_file,
)
from anylabeling.views.labeling.advanced.dataset_analysis.analysis_store import (
    ANALYSIS_KEY,
)
from anylabeling.views.labeling.advanced.dataset_analysis.annotation_stats import (
    collect_annotation_stats,
)
from anylabeling.views.labeling.advanced.dataset_analysis.single_image import (
    analyze_one_image,
    load_analysis_block,
)
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.style import (
    get_dialog_style,
    get_plain_text_edit_style,
    get_progress_dialog_style,
)
from anylabeling.views.labeling.advanced.dataset_analysis.dataset_analysis_report_dialog import (
    DatasetAnalysisReportDialog,
)
from anylabeling.views.labeling.advanced.dataset_analysis.single_image_analysis_report_dialog import (
    SingleImageAnalysisReportDialog,
)


DEFAULT_VLM_FULL_PROMPT = """任务：对输入图片做图像属性结构化解析，严格按照字段要求提取信息，最终只返回标准JSON格式，无任何多余中文说明。
JSON字段定义：
{
    "scene_category": "场景大类，可选：城郊道路/城市市政道路/乡村土路/山林边坡/室内场地/河道水岸/工地厂区/农田耕地等等",
    "shoot_view": "拍摄视角，只能从【高空俯视/斜向俯拍/平视/仰拍】四选一",
    "main_objects": "画面核心物体，用英文逗号分隔，如：沥青公路,人行道,绿植,施工人员",
    "image_quality": "画质问题，多选逗号分隔：无异常/明显模糊/画面过曝/压缩伪影，无异常填无异常",
    "light_weather": "光线天气时段，格式参考范例：白天户外阳光充足/阴天弱光/黄昏逆光/夜晚室内灯光；线索不足则末尾标注(推断)"
}

约束规则：
1. scene_category优先结合地貌、道路、植被、建筑精准归类，无法分类填其他；
2. main_objects只填写画面里大面积/主体可见物体，或者视觉显著的物体；
3. light_weather依据光影、环境特征判断，没有明确线索统一加后缀(推断)。"""

DEFAULT_VLM_PROMPT = DEFAULT_VLM_FULL_PROMPT


def _load_chatbot_vlm_defaults():
    """Pre-fill API fields from Chatbot models.json + providers (plan: reuse config)."""
    import os

    from anylabeling.views.labeling.chatbot.config import (
        DEFAULT_PROVIDERS_DATA,
        get_models_config_path,
    )
    from anylabeling.views.labeling.chatbot.provider import get_providers_data
    from anylabeling.views.labeling.chatbot.utils import load_json

    base_url, api_key, model = "", "", ""
    path = get_models_config_path()
    if not os.path.isfile(path):
        return base_url, api_key, model
    cfg = load_json(path)
    settings = cfg.get("settings") or {}
    prov = settings.get("provider") or "openai"
    mid = settings.get("model_id")
    if isinstance(mid, str) and mid.strip():
        model = mid.split(" (", 1)[0].strip()
    pdata = get_providers_data().get(prov) or DEFAULT_PROVIDERS_DATA.get(
        prov
    ) or {}
    base_url = (pdata.get("api_address") or "").strip()
    ak = pdata.get("api_key")
    if isinstance(ak, str):
        api_key = ak.strip()
    return base_url, api_key, model


class DatasetAnalysisWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal()
    single_done = pyqtSignal(str, dict, str)  # image_path, analysis_block, label_path
    failed = pyqtSignal(str)

    def __init__(
        self,
        parent_widget,
        *,
        image_paths: list,
        output_dir: str | None,
        max_side: int,
        skip_existing: bool,
        run_metrics: bool,
        run_vlm: bool,
        vlm_overwrite_existing: bool,
        api_base: str,
        api_key: str,
        model: str,
        vlm_prompt: str,
        vlm_max_tokens: int,
        single_only: bool = False,
        collect_annot_stats: bool = False,
    ):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.max_side = max_side
        self.skip_existing = skip_existing
        self.run_metrics = run_metrics
        self.run_vlm = run_vlm
        self.vlm_overwrite_existing = bool(vlm_overwrite_existing)
        self.api_base = api_base.strip()
        self.api_key = api_key
        self.model = model.strip()
        self.vlm_prompt = vlm_prompt.strip() or DEFAULT_VLM_PROMPT
        self.vlm_max_tokens = int(vlm_max_tokens)
        self.single_only = bool(single_only)
        self.collect_annot_stats = bool(collect_annot_stats)

    def _emit_single_done(self, image_path: str, block: dict, label_path: str):
        annotation_stats = None
        if self.collect_annot_stats:
            try:
                annotation_stats = collect_annotation_stats(
                    [image_path], output_dir=self.output_dir
                ).to_dict()
            except Exception as e:
                logger.error("single image annotation stats failed: %s", e)
                annotation_stats = {"error": str(e)}
        self.single_done.emit(
            image_path,
            {"block": block, "annotation_stats": annotation_stats},
            label_path,
        )

    def run(self):
        total = len(self.image_paths)
        try:
            for i, image_path in enumerate(self.image_paths):
                if self.isInterruptionRequested():
                    break
                base = osp.basename(image_path)
                self.progress.emit(i, total, base)
                label_path = get_label_file_path(image_path, self.output_dir)

                if self.skip_existing and osp.isfile(label_path):
                    try:
                        with open(label_path, "r", encoding="utf-8") as f:
                            existing = json.load(f)
                        if existing.get(ANALYSIS_KEY):
                            if self.single_only:
                                block = load_analysis_block(label_path)
                                self._emit_single_done(image_path, block, label_path)
                                self.finished_ok.emit()
                                return
                            continue
                    except (OSError, json.JSONDecodeError):
                        pass

                block = analyze_one_image(
                    image_path,
                    self.output_dir,
                    max_side=self.max_side,
                    run_metrics=self.run_metrics,
                    run_vlm=self.run_vlm,
                    vlm_overwrite_existing=self.vlm_overwrite_existing,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    model=self.model,
                    vlm_prompt=self.vlm_prompt,
                    vlm_max_tokens=self.vlm_max_tokens,
                )

                if self.single_only:
                    self._emit_single_done(image_path, block, label_path)
                    self.finished_ok.emit()
                    return

            self.progress.emit(total, total, "")
            self.finished_ok.emit()
        except Exception as e:
            logger.error(traceback.format_exc())
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class DatasetAnalysisDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_widget = parent
        self.setWindowTitle(self.tr("Dataset Image Analysis"))
        self.setMinimumWidth(520)
        self.setStyleSheet(get_dialog_style())

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.max_side_spin = QSpinBox()
        self.max_side_spin.setRange(256, 4096)
        self.max_side_spin.setValue(1024)
        self.max_side_spin.setToolTip(
            self.tr("Downscale long edge for faster metrics / smaller VLM payload.")
        )
        form.addRow(self.tr("Max analysis side (px):"), self.max_side_spin)

        self.skip_existing_cb = QCheckBox(self.tr("Skip images that already have analysis"))
        self.skip_existing_cb.setChecked(False)
        form.addRow(self.skip_existing_cb)

        self.metrics_cb = QCheckBox(self.tr("Compute local metrics (sharpness, brightness, ...)"))
        self.metrics_cb.setChecked(True)
        form.addRow(self.metrics_cb)

        self.vlm_cb = QCheckBox(self.tr("Call multimodal model for Chinese description"))
        self.vlm_cb.setChecked(False)
        form.addRow(self.vlm_cb)

        self.vlm_overwrite_cb = QCheckBox(
            self.tr("Overwrite existing VLM captions")
        )
        self.vlm_overwrite_cb.setChecked(False)
        self.vlm_overwrite_cb.setToolTip(
            self.tr(
                "If unchecked, existing non-empty captions in JSON will be kept "
                "and VLM will only run for empty captions."
            )
        )
        form.addRow(self.vlm_overwrite_cb)

        self.annot_stats_cb = QCheckBox(self.tr("Annotation stats (requires imported labels)"))
        self.annot_stats_cb.setChecked(False)
        self.annot_stats_cb.setToolTip(
            self.tr(
                "Compute dataset-level annotation statistics from existing JSON labels. "
                "You must import labels (YOLO/VOC/COCO/...) first."
            )
        )
        form.addRow(self.annot_stats_cb)

        self.suggest_cb = QCheckBox(self.tr("Optimization suggestions (heuristic)"))
        self.suggest_cb.setChecked(True)
        self.suggest_cb.setToolTip(
            self.tr(
                "Generate dataset improvement suggestions based on metric distributions "
                "and caption diversity (heuristic)."
            )
        )
        form.addRow(self.suggest_cb)

        self.vlm_max_tokens_spin = QSpinBox()
        self.vlm_max_tokens_spin.setRange(256, 8192)
        self.vlm_max_tokens_spin.setValue(2048)
        self.vlm_max_tokens_spin.setToolTip(
            self.tr(
                "Max new tokens per image for the VLM reply "
                "(each request is independent; higher reduces truncation)."
            )
        )
        form.addRow(self.tr("VLM max output tokens:"), self.vlm_max_tokens_spin)

        self.api_base_edit = QLineEdit()
        self.api_base_edit.setPlaceholderText("https://api.openai.com/v1")
        form.addRow(self.tr("API base URL:"), self.api_base_edit)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText(self.tr("API key (optional if using env)"))
        form.addRow(self.tr("API key:"), self.api_key_edit)

        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("gpt-4o-mini")
        form.addRow(self.tr("Model:"), self.model_edit)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlainText(DEFAULT_VLM_FULL_PROMPT)
        self.prompt_edit.setMinimumHeight(120)
        self.prompt_edit.setToolTip(
            self.tr(
                "The full prompt sent to the VLM for each image. "
                "Edit it directly if you want a new template."
            )
        )
        self.prompt_edit.setStyleSheet(get_plain_text_edit_style())
        form.addRow(self.tr("VLM prompt:"), self.prompt_edit)

        bu, ak, md = _load_chatbot_vlm_defaults()
        if bu:
            self.api_base_edit.setText(bu)
        if ak:
            self.api_key_edit.setText(ak)
        if md:
            self.model_edit.setText(md)

        def _sync_vlm_controls(checked: bool):
            for w in (
                self.api_base_edit,
                self.api_key_edit,
                self.model_edit,
                self.prompt_edit,
                self.vlm_max_tokens_spin,
                self.vlm_overwrite_cb,
            ):
                w.setEnabled(checked)

        self.vlm_cb.toggled.connect(_sync_vlm_controls)
        _sync_vlm_controls(self.vlm_cb.isChecked())

        hint = QLabel(
            self.tr(
                "Distortion and viewpoint scores are heuristic; BRISQUE may be "
                "unavailable depending on OpenCV build."
            )
        )
        hint.setWordWrap(True)
        layout.addLayout(form)
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        self.current_btn = QPushButton(self.tr("Analyze current image"))
        self.current_btn.setToolTip(
            self.tr("Run analysis on the image currently open in the editor.")
        )
        ok_btn = QPushButton(self.tr("Run batch"))
        cancel_btn = QPushButton(self.tr("Cancel"))
        self.current_btn.clicked.connect(self._on_run_current)
        ok_btn.clicked.connect(self._on_run)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.current_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        self._worker: DatasetAnalysisWorker | None = None
        self._progress: QProgressDialog | None = None

    def _validate_common(self) -> bool:
        if not self.metrics_cb.isChecked() and not self.vlm_cb.isChecked():
            QMessageBox.warning(
                self,
                self.tr("Dataset Analysis"),
                self.tr("Please enable at least metrics or VLM."),
            )
            return False
        if self.vlm_cb.isChecked() and not self.model_edit.text().strip():
            QMessageBox.warning(
                self,
                self.tr("Dataset Analysis"),
                self.tr("Please set a model name for VLM."),
            )
            return False
        return True

    def _on_run_current(self):
        parent = self.parent_widget
        image_path = getattr(parent, "filename", None)
        if not image_path:
            QMessageBox.information(
                self,
                self.tr("Dataset Analysis"),
                self.tr("No image loaded."),
            )
            return
        if not self._validate_common():
            return
        output_dir = getattr(parent, "output_dir", None)
        if self.annot_stats_cb.isChecked() and not self._has_any_annotations(
            [image_path], output_dir
        ):
            QMessageBox.warning(
                self,
                self.tr("Annotation stats"),
                self.tr(
                    "No annotations found for the current image. Please import "
                    "labels first (YOLO/VOC/COCO/...) so JSON label files contain shapes."
                ),
            )
            return
        self._start_worker([image_path], single_only=True, close_dialog=False)

    def _on_run(self):
        if not self._validate_common():
            return
        if self.annot_stats_cb.isChecked():
            parent = self.parent_widget
            image_paths = list(getattr(parent, "image_list", []) or [])
            output_dir = getattr(parent, "output_dir", None)
            if not self._has_any_annotations(image_paths, output_dir):
                QMessageBox.warning(
                    self,
                    self.tr("Annotation stats"),
                    self.tr(
                        "No annotations found. Please import/upload labels first "
                        "(YOLO/VOC/COCO/...) so JSON label files contain shapes."
                    ),
                )
                return
        self.accept()
        parent = self.parent_widget
        image_paths = list(getattr(parent, "image_list", []) or [])
        self._start_worker(image_paths, single_only=False, close_dialog=True)

    def _has_any_annotations(self, image_paths, output_dir) -> bool:
        for p in image_paths:
            label_path = get_label_file_path(p, output_dir)
            if not osp.isfile(label_path):
                continue
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                shapes = data.get("shapes") or []
                if isinstance(shapes, list) and len(shapes) > 0:
                    return True
            except Exception:
                continue
        return False

    def _start_worker(
        self,
        image_paths: list,
        *,
        single_only: bool,
        close_dialog: bool,
    ):
        parent = self.parent_widget
        output_dir = getattr(parent, "output_dir", None)

        vlm_full_prompt = (
            self.prompt_edit.toPlainText().strip() or DEFAULT_VLM_FULL_PROMPT
        )

        worker = DatasetAnalysisWorker(
            parent,
            image_paths=image_paths,
            output_dir=output_dir,
            max_side=int(self.max_side_spin.value()),
            skip_existing=self.skip_existing_cb.isChecked(),
            run_metrics=self.metrics_cb.isChecked(),
            run_vlm=self.vlm_cb.isChecked(),
            vlm_overwrite_existing=self.vlm_overwrite_cb.isChecked(),
            api_base=self.api_base_edit.text(),
            api_key=self.api_key_edit.text(),
            model=self.model_edit.text(),
            vlm_prompt=vlm_full_prompt,
            vlm_max_tokens=int(self.vlm_max_tokens_spin.value()),
            single_only=single_only,
            collect_annot_stats=self.annot_stats_cb.isChecked(),
        )
        self._worker = worker

        title = (
            self.tr("Analyzing current image...")
            if single_only
            else self.tr("Analyzing images...")
        )
        prog = QProgressDialog(title, self.tr("Cancel"), 0, len(image_paths), parent)
        self._progress = prog
        prog.setWindowModality(parent.windowModality())
        prog.setMinimumWidth(500)
        prog.setStyleSheet(get_progress_dialog_style(color="#1d1d1f", height=20))

        def on_prog(cur, total, name):
            prog.setMaximum(max(1, total))
            prog.setValue(cur)
            if single_only:
                prog.setLabelText(self.tr("Analyzing: %s") % (name or "—"))
            else:
                prog.setLabelText(
                    self.tr("Analyzing: %s (%d / %d)") % (name or "—", cur + 1, total)
                )

        worker.progress.connect(on_prog)
        prog.canceled.connect(worker.requestInterruption)

        def on_fail(msg):
            prog.close()
            self._progress = None
            self.current_btn.setEnabled(True)
            QMessageBox.critical(parent, self.tr("Dataset Analysis Failed"), msg)

        def on_single_done(image_path, payload, label_path):
            block = payload.get("block") if isinstance(payload, dict) else payload
            annotation_stats = (
                payload.get("annotation_stats") if isinstance(payload, dict) else None
            )
            dlg = SingleImageAnalysisReportDialog(
                parent,
                image_path=image_path,
                label_path=label_path,
                analysis_block=block or {},
                annotation_stats=annotation_stats,
            )
            dlg.exec()

        def on_done():
            prog.close()
            self._progress = None
            self.current_btn.setEnabled(True)
            if single_only:
                worker.deleteLater()
                return

            label_paths = [
                get_label_file_path(p, output_dir) for p in image_paths
            ]
            series, n = aggregate_numeric_metrics(label_paths)
            diversity = build_diversity_summary(label_paths)
            resolution_counts = collect_resolution_counts(label_paths)
            annotation_stats = None
            if self.annot_stats_cb.isChecked():
                try:
                    annotation_stats = collect_annotation_stats(
                        image_paths, output_dir=output_dir
                    ).to_dict()
                except Exception as e:
                    logger.error("annotation stats failed: %s", e)
                    annotation_stats = {"error": str(e)}
            optimization_suggestions = None
            if self.suggest_cb.isChecked():
                try:
                    optimization_suggestions = build_optimization_suggestions(
                        numeric_series=series,
                        resolution_counts=resolution_counts,
                        diversity=diversity,
                    )
                except Exception as e:
                    logger.error("optimization suggestions failed: %s", e)
                    optimization_suggestions = {"error": str(e)}
            out_dir = output_dir or osp.dirname(image_paths[0])
            try:
                summary_path = write_summary_file(
                    out_dir,
                    {
                        "per_image_numeric_series": {
                            k: v for k, v in series.items() if v
                        },
                        "images_with_metrics": n,
                        "diversity": diversity,
                        "resolution_counts": resolution_counts,
                        "annotation_stats": annotation_stats,
                        "optimization_suggestions": optimization_suggestions,
                    },
                )
            except Exception as e:
                logger.error("summary write failed: %s", e)
                summary_path = ""

            dlg = DatasetAnalysisReportDialog(
                parent,
                numeric_series=series,
                diversity=diversity,
                summary_path=summary_path,
                annotation_stats=annotation_stats,
                resolution_counts=resolution_counts,
                optimization_suggestions=optimization_suggestions,
                image_paths=image_paths,
            )
            dlg.exec()
            worker.deleteLater()

        self.current_btn.setEnabled(False)
        worker.failed.connect(on_fail)
        if single_only:
            worker.single_done.connect(on_single_done)
        worker.finished_ok.connect(on_done)
        worker.start()
        prog.show()
        if close_dialog:
            self.hide()


def open_dataset_analysis_dialog(parent):
    """Entry point from LabelingWidget."""
    if not getattr(parent, "image_list", None):
        parent.error_message(
            parent.tr("No images loaded"),
            parent.tr("Please open an image folder first."),
        )
        return
    d = DatasetAnalysisDialog(parent)
    d.exec()
