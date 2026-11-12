"""Report dialog for single-image dataset analysis results."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from anylabeling.views.labeling.utils.style import (
    get_dialog_style,
    get_plain_text_edit_style,
)

_METRIC_LABELS: Dict[str, str] = {
    "original_width": "Original width (px)",
    "original_height": "Original height (px)",
    "megapixels": "Megapixels",
    "aspect_ratio": "Aspect ratio (W/H)",
    "analysis_max_side": "Analysis max side (px)",
    "analysis_scale": "Downscale factor",
    "sharpness_laplacian_var_log1p": "Sharpness (log1p Laplacian var)",
    "brightness_l_mean": "Brightness (LAB L mean)",
    "contrast_rms_gray": "Contrast (gray RMS)",
    "hue_mean_deg": "Hue mean (deg)",
    "saturation_mean": "Saturation mean",
    "value_mean": "Value mean (HSV V)",
    "noise_mad_residual": "Noise (MAD residual)",
    "artifact_heuristic_score": "Artifact score (heuristic)",
    "artifact_method": "Artifact method",
    "brisque": "BRISQUE score",
    "brisque_error": "BRISQUE error",
    "dominant_line_roll_deg": "Dominant line roll (deg)",
    "methods_note": "Methods note",
    "image_path_resolved": "Image file",
}


def _format_metric_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _format_vlm_caption(caption: str) -> str:
    text = (caption or "").strip()
    if not text:
        return ""
    if text.startswith("{") or text.startswith("["):
        try:
            return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
    return text


class SingleImageAnalysisReportDialog(QDialog):
    def __init__(
        self,
        parent,
        *,
        image_path: str,
        label_path: str,
        analysis_block: Dict[str, Any],
        annotation_stats: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(parent)
        self._image_path = image_path
        self._label_path = label_path
        self._block = analysis_block or {}
        self._annotation_stats = annotation_stats

        self.setWindowTitle(self.tr("Single Image Analysis Report"))
        self.setMinimumSize(640, 520)
        self.setStyleSheet(get_dialog_style())

        root = QVBoxLayout(self)
        tabs = QTabWidget()

        tabs.addTab(self._build_summary_tab(), self.tr("Summary"))
        tabs.addTab(self._build_metrics_tab(), self.tr("Image metrics"))
        tabs.addTab(self._build_vlm_tab(), self.tr("VLM"))
        tabs.addTab(self._build_annotations_tab(), self.tr("Annotations"))

        root.addWidget(tabs)

        close_btn = QPushButton(self.tr("Close"))
        close_btn.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(close_btn)
        root.addLayout(row)

    def _build_summary_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        metrics = self._block.get("metrics") if isinstance(self._block.get("metrics"), dict) else {}
        vlm = self._block.get("vlm") if isinstance(self._block.get("vlm"), dict) else {}

        rows: List[Tuple[str, str]] = [
            (self.tr("Image path"), self._image_path),
            (self.tr("Label JSON"), self._label_path),
            (self.tr("Generated at"), str(self._block.get("generated_at") or "—")),
            (self.tr("Schema version"), str(self._block.get("schema_version") or "—")),
            (
                self.tr("Resolution"),
                f"{metrics.get('original_width', '—')} × {metrics.get('original_height', '—')}",
            ),
            (self.tr("VLM model"), str(vlm.get("model") or "—")),
            (self.tr("Analysis error"), str(self._block.get("error") or "—")),
        ]
        for label, value in rows:
            lab = QLabel(str(value))
            lab.setWordWrap(True)
            lab.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            form.addRow(label, lab)
        return page

    def _build_metrics_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        metrics = self._block.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            layout.addWidget(
                QLabel(self.tr("No local metrics were computed for this image."))
            )
            return page

        table = QTableWidget(len(metrics), 2)
        table.setHorizontalHeaderLabels(
            [self.tr("Parameter"), self.tr("Value")]
        )
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        for row, (key, value) in enumerate(sorted(metrics.items())):
            name = self.tr(_METRIC_LABELS.get(key, key))
            table.setItem(row, 0, QTableWidgetItem(name))
            val_item = QTableWidgetItem(_format_metric_value(value))
            val_item.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            table.setItem(row, 1, val_item)
        table.resizeColumnsToContents()
        layout.addWidget(table)
        return page

    def _build_vlm_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        vlm = self._block.get("vlm")
        if not isinstance(vlm, dict):
            layout.addWidget(QLabel(self.tr("VLM analysis was not run.")))
            return page

        meta = QFormLayout()
        meta.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for key in ("model", "base_url", "prompt_version", "max_tokens"):
            if key in vlm:
                meta.addRow(key, QLabel(str(vlm.get(key))))
        layout.addLayout(meta)

        caption = _format_vlm_caption(str(vlm.get("caption") or ""))
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(caption or self.tr("(empty caption)"))
        editor.setStyleSheet(get_plain_text_edit_style())
        layout.addWidget(QLabel(self.tr("VLM response:")))
        layout.addWidget(editor, 1)

        prompt = str(vlm.get("prompt") or "").strip()
        if prompt:
            prompt_edit = QPlainTextEdit()
            prompt_edit.setReadOnly(True)
            prompt_edit.setPlainText(prompt)
            prompt_edit.setMaximumHeight(120)
            prompt_edit.setStyleSheet(get_plain_text_edit_style())
            layout.addWidget(QLabel(self.tr("Prompt:")))
            layout.addWidget(prompt_edit)
        return page

    def _build_annotations_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        stats = self._annotation_stats
        if not stats:
            layout.addWidget(
                QLabel(
                    self.tr(
                        "No annotation statistics (enable annotation stats or "
                        "import labels first)."
                    )
                )
            )
            return page

        if stats.get("error"):
            layout.addWidget(QLabel(str(stats.get("error"))))
            return page

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.addRow(
            self.tr("Images with boxes"),
            QLabel(str(stats.get("images_with_any_box", 0))),
        )
        form.addRow(
            self.tr("Images without boxes"),
            QLabel(str(stats.get("images_with_zero_box", 0))),
        )

        per_class = stats.get("per_class_box_count") or {}
        if isinstance(per_class, dict) and per_class:
            lines = [f"{k}: {v}" for k, v in sorted(per_class.items())]
            lab = QLabel("\n".join(lines))
            lab.setWordWrap(True)
            form.addRow(self.tr("Boxes per class"), lab)

        per_image = stats.get("per_image_class_counts") or []
        if isinstance(per_image, list) and per_image:
            first = per_image[0] if isinstance(per_image[0], dict) else {}
            if first:
                lines = [f"{k}: {v}" for k, v in sorted(first.items())]
                lab = QLabel("\n".join(lines))
                form.addRow(self.tr("This image (class counts)"), lab)

        area_ratio = stats.get("per_box_area_ratio") or {}
        if isinstance(area_ratio, dict) and area_ratio:
            lines = []
            for cls, ratios in sorted(area_ratio.items()):
                if not isinstance(ratios, list) or not ratios:
                    continue
                avg = sum(float(x) for x in ratios) / len(ratios)
                lines.append(f"{cls}: avg {avg:.4f} ({len(ratios)} box(es))")
            if lines:
                lab = QLabel("\n".join(lines))
                lab.setWordWrap(True)
                form.addRow(self.tr("Box area ratio"), lab)

        layout.addLayout(form)
        layout.addStretch()
        return page
