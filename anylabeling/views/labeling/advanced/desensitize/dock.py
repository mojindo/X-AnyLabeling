from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt


class DesensitizeDock(QtWidgets.QDockWidget):
    applyRequested = QtCore.pyqtSignal()
    clearMaskRequested = QtCore.pyqtSignal()
    closeModeRequested = QtCore.pyqtSignal()
    toolChanged = QtCore.pyqtSignal(str)
    brushRadiusChanged = QtCore.pyqtSignal(int)
    blockSizeChanged = QtCore.pyqtSignal(int)
    maskToolsEnabledChanged = QtCore.pyqtSignal(bool)
    effectModeChanged = QtCore.pyqtSignal(str)
    grayLevelChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("DesensitizeDock")
        self.setWindowTitle(self.tr("Dataset Desensitization"))
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.setMinimumWidth(280)

        root = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        hint = QtWidgets.QLabel(
            self.tr(
                "Mark regions to pixelate on the current image. "
                "Changes are written to the image file when you apply."
            )
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        layout.addWidget(hint)

        self._enable_cb = QtWidgets.QCheckBox(
            self.tr("Enable mask tools")
        )
        self._enable_cb.setChecked(True)
        self._enable_cb.toggled.connect(self._on_enable_toggled)
        layout.addWidget(self._enable_cb)

        self._enable_hint = QtWidgets.QLabel(
            self.tr(
                "When enabled: draw masks only (no label edit or canvas pan). "
                "When disabled: normal labeling."
            )
        )
        self._enable_hint.setWordWrap(True)
        self._enable_hint.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self._enable_hint)

        tool_box = QtWidgets.QGroupBox(self.tr("Tool"))
        tool_layout = QtWidgets.QVBoxLayout(tool_box)
        self._btn_brush = QtWidgets.QRadioButton(self.tr("Brush"))
        self._btn_rect = QtWidgets.QRadioButton(self.tr("Rectangle"))
        self._btn_poly = QtWidgets.QRadioButton(self.tr("Polygon"))
        self._btn_brush.setChecked(True)
        for rb in (self._btn_brush, self._btn_rect, self._btn_poly):
            tool_layout.addWidget(rb)
        layout.addWidget(tool_box)

        self._tool_group = QtWidgets.QButtonGroup(self)
        self._tool_group.addButton(self._btn_brush, 0)
        self._tool_group.addButton(self._btn_rect, 1)
        self._tool_group.addButton(self._btn_poly, 2)
        self._tool_group.idToggled.connect(self._on_tool_toggled)

        brush_row = QtWidgets.QHBoxLayout()
        self._brush_label = QtWidgets.QLabel(self.tr("Brush radius:"))
        brush_row.addWidget(self._brush_label)
        self._brush_spin = QtWidgets.QSpinBox()
        self._brush_spin.setRange(1, 200)
        self._brush_spin.setValue(24)
        self._brush_spin.setToolTip(
            self.tr(
                "Brush size in image pixels. Alt + mouse wheel on canvas to adjust."
            )
        )
        self._brush_spin.valueChanged.connect(self.brushRadiusChanged.emit)
        brush_row.addWidget(self._brush_spin)
        layout.addLayout(brush_row)

        effect_box = QtWidgets.QGroupBox(self.tr("Effect"))
        effect_layout = QtWidgets.QVBoxLayout(effect_box)
        self._btn_mosaic = QtWidgets.QRadioButton(self.tr("Mosaic"))
        self._btn_solid = QtWidgets.QRadioButton(self.tr("Solid gray fill"))
        self._btn_mosaic.setChecked(True)
        effect_layout.addWidget(self._btn_mosaic)
        effect_layout.addWidget(self._btn_solid)
        self._effect_group = QtWidgets.QButtonGroup(self)
        self._effect_group.addButton(self._btn_mosaic, 0)
        self._effect_group.addButton(self._btn_solid, 1)
        self._effect_group.idToggled.connect(self._on_effect_toggled)
        layout.addWidget(effect_box)

        block_row = QtWidgets.QHBoxLayout()
        self._block_label = QtWidgets.QLabel(self.tr("Mosaic block:"))
        block_row.addWidget(self._block_label)
        self._block_spin = QtWidgets.QSpinBox()
        self._block_spin.setRange(2, 64)
        self._block_spin.setValue(12)
        block_row.addWidget(self._block_spin)
        self._block_spin.valueChanged.connect(self.blockSizeChanged.emit)
        layout.addLayout(block_row)

        gray_row = QtWidgets.QHBoxLayout()
        self._gray_label = QtWidgets.QLabel(self.tr("Gray level (0–255):"))
        gray_row.addWidget(self._gray_label)
        self._gray_spin = QtWidgets.QSpinBox()
        self._gray_spin.setRange(0, 255)
        self._gray_spin.setValue(128)
        self._gray_spin.valueChanged.connect(self.grayLevelChanged.emit)
        gray_row.addWidget(self._gray_spin)
        layout.addLayout(gray_row)

        poly_hint = QtWidgets.QLabel(
            self.tr(
                "Polygon: left-click vertices, right-click or Enter to finish. "
                "Esc cancels the current shape."
            )
        )
        poly_hint.setWordWrap(True)
        poly_hint.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(poly_hint)

        btn_row = QtWidgets.QHBoxLayout()
        self._apply_btn = QtWidgets.QPushButton(self.tr("Apply to image…"))
        self._clear_btn = QtWidgets.QPushButton(self.tr("Clear mask"))
        self._close_btn = QtWidgets.QPushButton(self.tr("Close"))
        self._apply_btn.clicked.connect(self.applyRequested.emit)
        self._clear_btn.clicked.connect(self.clearMaskRequested.emit)
        self._close_btn.clicked.connect(self.closeModeRequested.emit)
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(self._clear_btn)
        btn_row.addWidget(self._close_btn)
        layout.addLayout(btn_row)

        layout.addStretch(1)
        self.setWidget(root)
        self._hint = hint
        self._tool_box = tool_box
        self._effect_box = effect_box
        self._poly_hint = poly_hint
        self._sync_mask_tool_widgets(True)
        self._sync_effect_option_widgets()

    def _on_enable_toggled(self, checked: bool) -> None:
        self._sync_mask_tool_widgets(checked)
        self.maskToolsEnabledChanged.emit(checked)

    def _sync_mask_tool_widgets(self, mask_on: bool) -> None:
        self._tool_box.setEnabled(mask_on)
        self._effect_box.setEnabled(mask_on)
        self._brush_spin.setEnabled(mask_on)
        self._poly_hint.setEnabled(mask_on)
        self._sync_effect_option_widgets()

    def _on_effect_toggled(self, button_id: int, checked: bool) -> None:
        if not checked:
            return
        self._sync_effect_option_widgets()
        mode = "mosaic" if button_id == 0 else "solid"
        self.effectModeChanged.emit(mode)

    def _sync_effect_option_widgets(self) -> None:
        mosaic = self._btn_mosaic.isChecked()
        mask_on = self._enable_cb.isChecked()
        self._block_spin.setEnabled(mask_on and mosaic)
        self._block_label.setEnabled(mask_on and mosaic)
        self._gray_spin.setEnabled(mask_on and not mosaic)
        self._gray_label.setEnabled(mask_on and not mosaic)

    def effect_mode(self) -> str:
        return "mosaic" if self._btn_mosaic.isChecked() else "solid"

    def gray_level(self) -> int:
        return int(self._gray_spin.value())

    def set_mask_tools_enabled(self, enabled: bool) -> None:
        self._enable_cb.blockSignals(True)
        self._enable_cb.setChecked(bool(enabled))
        self._enable_cb.blockSignals(False)
        self._sync_mask_tool_widgets(bool(enabled))

    def mask_tools_enabled(self) -> bool:
        return self._enable_cb.isChecked()

    def retranslate_ui(self) -> None:
        self.setWindowTitle(self.tr("Dataset Desensitization"))
        self._hint.setText(
            self.tr(
                "Mark regions to pixelate on the current image. "
                "Changes are written to the image file when you apply."
            )
        )
        self._enable_cb.setText(self.tr("Enable mask tools"))
        self._enable_hint.setText(
            self.tr(
                "When enabled: draw masks only (no label edit or canvas pan). "
                "When disabled: normal labeling."
            )
        )
        self._tool_box.setTitle(self.tr("Tool"))
        self._btn_brush.setText(self.tr("Brush"))
        self._btn_rect.setText(self.tr("Rectangle"))
        self._btn_poly.setText(self.tr("Polygon"))
        self._brush_label.setText(self.tr("Brush radius:"))
        self._brush_spin.setToolTip(
            self.tr(
                "Brush size in image pixels. Alt + mouse wheel on canvas to adjust."
            )
        )
        self._effect_box.setTitle(self.tr("Effect"))
        self._btn_mosaic.setText(self.tr("Mosaic"))
        self._btn_solid.setText(self.tr("Solid gray fill"))
        self._block_label.setText(self.tr("Mosaic block:"))
        self._gray_label.setText(self.tr("Gray level (0–255):"))
        self._poly_hint.setText(
            self.tr(
                "Polygon: left-click vertices, right-click or Enter to finish. "
                "Esc cancels the current shape."
            )
        )
        self._apply_btn.setText(self.tr("Apply to image…"))
        self._clear_btn.setText(self.tr("Clear mask"))
        self._close_btn.setText(self.tr("Close"))

    def changeEvent(self, event: QtCore.QEvent) -> None:
        if event.type() == QtCore.QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(event)

    def _on_tool_toggled(self, button_id: int, checked: bool) -> None:
        if not checked:
            return
        tool = {0: "brush", 1: "rectangle", 2: "polygon"}.get(button_id, "brush")
        self.toolChanged.emit(tool)

    def current_tool(self) -> str:
        button_id = int(self._tool_group.checkedId())
        return {0: "brush", 1: "rectangle", 2: "polygon"}.get(
            button_id, "brush"
        )

    def brush_radius(self) -> int:
        return int(self._brush_spin.value())

    def block_size(self) -> int:
        return int(self._block_spin.value())

    def closeEvent(self, event) -> None:
        self.closeModeRequested.emit()
        super().closeEvent(event)