"""YOLO predict parameter form for MyTrain inference dialogs."""

from __future__ import annotations

from dataclasses import replace

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QResizeEvent, QShowEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from anylabeling.services.auto_training.mytrain.predict_options import (
    PredictOptions,
    parse_classes_value,
)
from anylabeling.services.auto_training.mytrain.style import (
    get_ultralytics_dialog_style,
)
from anylabeling.views.labeling.utils.theme import get_theme
from anylabeling.views.training.widgets.mytrain_widgets.custom_widgets import (
    CustomLineEdit,
)


class PredictParamsWidget(QWidget):
    """Form fields + optional custom ``key=value`` CLI overrides."""

    def __init__(
        self,
        parent=None,
        *,
        preset: str = "default",
        show_output_options: bool = True,
    ):
        super().__init__(parent)
        self.setStyleSheet(get_ultralytics_dialog_style())
        self._preset = preset
        self._show_output = show_output_options
        self._param_grid: QGridLayout | None = None
        self._params_group: QGroupBox | None = None
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self._build_ui(preset)

    def _expand_horizontally(self, widget: QWidget) -> None:
        widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

    def _spin_size_policy(self, widget: QWidget) -> None:
        widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        widget.setMinimumWidth(88)

    def _labeled_field(
        self, label: str, control: QWidget, *, t: dict
    ) -> QWidget:
        cell = QWidget()
        lay = QVBoxLayout(cell)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        caption = QLabel(label)
        caption.setStyleSheet(
            f"color: {t['text_secondary']}; font-size: 12px;"
        )
        lay.addWidget(caption)
        lay.addWidget(control)
        self._expand_horizontally(cell)
        self._expand_horizontally(control)
        return cell

    def _build_ui(self, preset: str) -> None:
        t = get_theme()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        group = QGroupBox(self.tr("Predict parameters"))
        self._params_group = group
        self._expand_horizontally(group)
        grid = QGridLayout(group)
        self._param_grid = grid
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setToolTip(
            self.tr("Confidence threshold; higher = fewer boxes.")
        )
        self._spin_size_policy(self.conf_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(0.7)
        self.iou_spin.setToolTip(self.tr("NMS IoU threshold."))
        self._spin_size_policy(self.iou_spin)

        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 10000)
        self.max_det_spin.setValue(300)
        self._spin_size_policy(self.max_det_spin)

        self.classes_edit = CustomLineEdit()
        self.classes_edit.setPlaceholderText(
            self.tr("All classes if empty, e.g. 0 or 0,1")
        )
        self.classes_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        grid.addWidget(
            self._labeled_field(self.tr("conf:"), self.conf_spin, t=t), 0, 0
        )
        grid.addWidget(
            self._labeled_field(self.tr("iou:"), self.iou_spin, t=t), 0, 1
        )
        grid.addWidget(
            self._labeled_field(self.tr("max_det:"), self.max_det_spin, t=t),
            1,
            0,
        )
        grid.addWidget(
            self._labeled_field(self.tr("classes:"), self.classes_edit, t=t),
            1,
            1,
        )

        self.agnostic_cb = QCheckBox(self.tr("agnostic_nms"))
        self.agnostic_cb.setToolTip(
            self.tr("When enabled, NMS ignores class ids.")
        )
        grid.addWidget(self.agnostic_cb, 2, 0, 1, 2)

        if self._show_output:
            self.save_cb = QCheckBox(self.tr("save"))
            self.save_txt_cb = QCheckBox(self.tr("save_txt"))
            self.save_conf_cb = QCheckBox(self.tr("save_conf"))
            self.save_crop_cb = QCheckBox(self.tr("save_crop"))
            out_wrap = QWidget()
            out_grid = QGridLayout(out_wrap)
            out_grid.setContentsMargins(0, 0, 0, 0)
            out_grid.setHorizontalSpacing(12)
            out_grid.setVerticalSpacing(6)
            out_grid.setColumnStretch(0, 1)
            out_grid.setColumnStretch(1, 1)
            out_grid.addWidget(self.save_cb, 0, 0)
            out_grid.addWidget(self.save_txt_cb, 0, 1)
            out_grid.addWidget(self.save_conf_cb, 1, 0)
            out_grid.addWidget(self.save_crop_cb, 1, 1)
            self._expand_horizontally(out_wrap)

            out_cell = QWidget()
            self._expand_horizontally(out_cell)
            out_lay = QVBoxLayout(out_cell)
            out_lay.setContentsMargins(0, 0, 0, 0)
            out_lay.setSpacing(6)
            out_title = QLabel(self.tr("Output:"))
            out_title.setStyleSheet(
                f"color: {t['text_secondary']}; font-size: 12px;"
            )
            out_lay.addWidget(out_title)
            out_lay.addWidget(out_wrap)
            grid.addWidget(out_cell, 3, 0, 1, 2)

        root.addWidget(group, 0)

        cli_group = QGroupBox(self.tr("Custom predict CLI (optional)"))
        self._expand_horizontally(cli_group)
        cli_layout = QVBoxLayout(cli_group)
        hint = QLabel(
            self.tr(
                "Extra or overriding arguments, space-separated, "
                "e.g. conf=0.3 iou=0.5 classes=0 save_txt=True"
            )
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {t['text_secondary']}; font-size: 12px;")
        cli_layout.addWidget(hint)
        self.cli_edit = QPlainTextEdit()
        self.cli_edit.setMaximumHeight(72)
        self.cli_edit.setPlaceholderText("conf=0.25 iou=0.7 max_det=300")
        cli_layout.addWidget(self.cli_edit)
        root.addWidget(cli_group)

        if preset == "labels_only":
            self.apply_options(PredictOptions.for_labels_only())
        else:
            self.apply_options(PredictOptions())
        QTimer.singleShot(0, self.relayout_parameters)

    def _layout_width_hint(self) -> int:
        side_margins = 56
        win = self.window()
        candidates = []
        if win is not None:
            candidates.append(win.width() - side_margins)
        probe = self._params_group or self
        candidates.append(probe.width())
        parent = self.parentWidget()
        while parent is not None:
            candidates.append(parent.width() - side_margins)
            parent = parent.parentWidget()
        for w in candidates:
            if w >= 200:
                return w
        return max(candidates) if candidates else 640

    def relayout_parameters(self) -> None:
        grid = self._param_grid
        if grid is None:
            return
        avail = self._layout_width_hint()
        if avail < 200:
            return
        inner = max(0, avail - grid.contentsMargins().left() - grid.contentsMargins().right())
        half = max(140, (inner - grid.horizontalSpacing()) // 2)
        grid.setColumnMinimumWidth(0, half)
        grid.setColumnMinimumWidth(1, half)
        grid.invalidate()
        root = self.layout()
        if root is not None:
            root.activate()
        self.updateGeometry()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self.relayout_parameters)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout_parameters()

    def apply_options(self, opts: PredictOptions) -> None:
        self.conf_spin.setValue(float(opts.conf))
        self.iou_spin.setValue(float(opts.iou))
        self.max_det_spin.setValue(int(opts.max_det))
        if opts.classes:
            self.classes_edit.setText(
                ",".join(str(c) for c in opts.classes)
            )
        else:
            self.classes_edit.clear()
        self.agnostic_cb.setChecked(bool(opts.agnostic_nms))
        if self._show_output:
            self.save_cb.setChecked(bool(opts.save))
            self.save_txt_cb.setChecked(bool(opts.save_txt))
            self.save_conf_cb.setChecked(bool(opts.save_conf))
            self.save_crop_cb.setChecked(bool(opts.save_crop))
        self.cli_edit.setPlainText(opts.custom_cli or "")

    def options(self) -> PredictOptions:
        if self._preset == "labels_only":
            base = PredictOptions.for_labels_only()
        else:
            base = PredictOptions()
        save = base.save
        save_txt = base.save_txt
        save_conf = base.save_conf
        save_crop = base.save_crop
        if self._show_output:
            save = self.save_cb.isChecked()
            save_txt = self.save_txt_cb.isChecked()
            save_conf = self.save_conf_cb.isChecked()
            save_crop = self.save_crop_cb.isChecked()
        opts = PredictOptions(
            conf=self.conf_spin.value(),
            iou=self.iou_spin.value(),
            max_det=self.max_det_spin.value(),
            classes=parse_classes_value(self.classes_edit.text()),
            agnostic_nms=self.agnostic_cb.isChecked(),
            save=save,
            save_txt=save_txt,
            save_conf=save_conf,
            save_crop=save_crop,
            custom_cli=self.cli_edit.toPlainText().strip(),
        )
        return opts.with_cli_overrides()

    def options_with_imgsz(self, imgsz: int) -> PredictOptions:
        return replace(self.options(), imgsz=int(imgsz))