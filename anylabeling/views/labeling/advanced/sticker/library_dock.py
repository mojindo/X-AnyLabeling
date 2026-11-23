from __future__ import annotations

import os
import os.path as osp
from dataclasses import dataclass
from typing import List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from collections import OrderedDict

from anylabeling.views.labeling.logger import logger


@dataclass(frozen=True)
class MaterialItem:
    path: str
    label: str


class StickerLibraryDock(QtWidgets.QDockWidget):
    materialSelected = QtCore.pyqtSignal(str, str)  # path, label
    commitRequested = QtCore.pyqtSignal(str, str, str, bool)  # blend, anno, warp, appearance
    warpModeChanged = QtCore.pyqtSignal(str)  # "4" or "8"
    cancelRequested = QtCore.pyqtSignal()
    rotateRequested = QtCore.pyqtSignal(float)  # delta degrees
    moveRequested = QtCore.pyqtSignal(float, float)  # dx, dy in image coords (pixels)
    quickApplyToggled = QtCore.pyqtSignal(bool)
    settingsChanged = QtCore.pyqtSignal(str, str, str, bool)  # blend, anno, warp, appearance
    closeModeRequested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("StickerLibraryDock")
        self.setWindowTitle(self.tr("Sticker"))
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        root = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Controls
        ctrl = QtWidgets.QHBoxLayout()
        self._btn_import = QtWidgets.QPushButton(self.tr("Import materials folder"))
        self._btn_import.clicked.connect(self._on_import_folder)
        ctrl.addWidget(self._btn_import)

        ctrl.addStretch(1)
        layout.addLayout(ctrl)

        opts = QtWidgets.QHBoxLayout()
        self._warp_combo = QtWidgets.QComboBox()
        self._warp_combo.addItem(self.tr("4-point (Perspective)"), "4")
        self._warp_combo.addItem(self.tr("8-point (Warp)"), "8")
        self._warp_combo.currentIndexChanged.connect(self._on_warp_changed)
        opts.addWidget(QtWidgets.QLabel(self.tr("Warp:")))
        opts.addWidget(self._warp_combo, 1)

        blend_lbl = QtWidgets.QLabel(self.tr("Blend:"))
        blend_lbl.setToolTip(
            self.tr("How the warped sticker is combined with the base image.")
        )
        self._blend_combo = QtWidgets.QComboBox()
        for mode in ("alpha", "multiply", "screen", "overlay", "poisson"):
            self._blend_combo.addItem(mode)
        self._blend_combo.currentIndexChanged.connect(self._on_blend_changed)
        self._apply_blend_item_tooltips()
        opts.addWidget(blend_lbl)
        opts.addWidget(self._blend_combo, 1)

        self._anno_combo = QtWidgets.QComboBox()
        self._anno_combo.addItem(self.tr("Detection (HBB)"), "hbb")
        self._anno_combo.addItem(self.tr("Detection (OBB)"), "obb")
        self._anno_combo.addItem(self.tr("Segmentation (Polygon)"), "seg")
        self._anno_combo.currentIndexChanged.connect(self._emit_settings_changed)
        opts.addWidget(QtWidgets.QLabel(self.tr("Annotation:")))
        opts.addWidget(self._anno_combo, 1)
        layout.addLayout(opts)

        self._blend_hint = QtWidgets.QLabel("")
        self._blend_hint.setWordWrap(True)
        self._blend_hint.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self._blend_hint)
        self._refresh_blend_hint()

        quick = QtWidgets.QHBoxLayout()
        self._quick_apply = QtWidgets.QCheckBox(self.tr("Quick apply"))
        self._quick_apply.setToolTip(self.tr("Click on image to apply sticker at that position"))
        self._quick_apply.toggled.connect(self._on_quick_apply_toggled)
        quick.addWidget(self._quick_apply)
        quick.addStretch(1)
        layout.addLayout(quick)

        appear_row = QtWidgets.QHBoxLayout()
        self._appearance_match = QtWidgets.QCheckBox(
            self.tr("Match surrounding appearance")
        )
        self._appearance_match.setToolTip(
            self.tr(
                "Adjust sticker colors to match the background ring around the placement quad before blending."
            )
        )
        self._appearance_match.setChecked(True)
        self._appearance_match.toggled.connect(self._on_appearance_toggled)
        appear_row.addWidget(self._appearance_match)
        appear_row.addStretch(1)
        layout.addLayout(appear_row)

        rot = QtWidgets.QHBoxLayout()
        rot.addWidget(QtWidgets.QLabel(self.tr("Rotate:")))
        self._btn_rot_l = QtWidgets.QPushButton(self.tr("-5°"))
        self._btn_rot_r = QtWidgets.QPushButton(self.tr("+5°"))
        self._btn_rot_l.clicked.connect(lambda: self.rotateRequested.emit(-5.0))
        self._btn_rot_r.clicked.connect(lambda: self.rotateRequested.emit(5.0))
        rot.addWidget(self._btn_rot_l)
        rot.addWidget(self._btn_rot_r)
        rot.addStretch(1)
        layout.addLayout(rot)

        move = QtWidgets.QHBoxLayout()
        move.addWidget(QtWidgets.QLabel(self.tr("Move:")))
        self._move_step = QtWidgets.QSpinBox()
        self._move_step.setRange(1, 500)
        self._move_step.setValue(5)
        self._move_step.setSingleStep(1)
        self._move_step.setFixedWidth(70)
        move.addWidget(self._move_step)
        move.addWidget(QtWidgets.QLabel(self.tr("px")))

        self._btn_move_up = QtWidgets.QPushButton(self.tr("↑"))
        self._btn_move_down = QtWidgets.QPushButton(self.tr("↓"))
        self._btn_move_left = QtWidgets.QPushButton(self.tr("←"))
        self._btn_move_right = QtWidgets.QPushButton(self.tr("→"))
        for b in (self._btn_move_up, self._btn_move_down, self._btn_move_left, self._btn_move_right):
            b.setFixedWidth(36)

        self._btn_move_up.clicked.connect(lambda: self._emit_move(0.0, -float(self._move_step.value())))
        self._btn_move_down.clicked.connect(lambda: self._emit_move(0.0, float(self._move_step.value())))
        self._btn_move_left.clicked.connect(lambda: self._emit_move(-float(self._move_step.value()), 0.0))
        self._btn_move_right.clicked.connect(lambda: self._emit_move(float(self._move_step.value()), 0.0))

        move.addWidget(self._btn_move_left)
        move.addWidget(self._btn_move_up)
        move.addWidget(self._btn_move_down)
        move.addWidget(self._btn_move_right)
        move.addStretch(1)
        layout.addLayout(move)

        btns = QtWidgets.QHBoxLayout()
        self._btn_cancel = QtWidgets.QPushButton(self.tr("Cancel"))
        self._btn_cancel.clicked.connect(lambda: self.cancelRequested.emit())
        self._btn_commit = QtWidgets.QPushButton(self.tr("Apply"))
        self._btn_commit.clicked.connect(self._on_commit)
        btns.addStretch(1)
        btns.addWidget(self._btn_cancel)
        btns.addWidget(self._btn_commit)
        layout.addLayout(btns)

        # Material strip
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._strip = QtWidgets.QWidget()
        self._strip_layout = QtWidgets.QHBoxLayout(self._strip)
        self._strip_layout.setContentsMargins(0, 0, 0, 0)
        self._strip_layout.setSpacing(6)
        self._strip_layout.addStretch(1)
        self._scroll.setWidget(self._strip)
        layout.addWidget(self._scroll, 1)

        self.setWidget(root)

        self._items: List[MaterialItem] = []
        self._buttons: List[QtWidgets.QToolButton] = []
        self._icon_cache: "OrderedDict[str, QtGui.QIcon]" = OrderedDict()
        self._icon_cache_max = 400

    def _icon_for_path(self, path: str) -> Optional[QtGui.QIcon]:
        if not path or not osp.isfile(path):
            return None
        icon = self._icon_cache.get(path)
        if icon is not None:
            self._icon_cache.move_to_end(path)
            return icon
        pm = QtGui.QPixmap(path)
        if pm.isNull():
            return None
        pm = pm.scaled(
            96,
            96,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        icon = QtGui.QIcon(pm)
        self._icon_cache[path] = icon
        while len(self._icon_cache) > int(self._icon_cache_max):
            self._icon_cache.popitem(last=False)
        return icon

    def _clear(self) -> None:
        for btn in self._buttons:
            btn.setParent(None)
            btn.deleteLater()
        self._buttons = []
        while self._strip_layout.count() > 0:
            it = self._strip_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._strip_layout.addStretch(1)

    def _render(self) -> None:
        self._clear()
        for it in self._items:
            btn = QtWidgets.QToolButton()
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            btn.setAutoRaise(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setIconSize(QtCore.QSize(96, 96))
            btn.setFixedWidth(120)
            btn.setText(it.label)
            ic = self._icon_for_path(it.path)
            if ic is not None:
                btn.setIcon(ic)
            btn.clicked.connect(
                lambda _=False, p=it.path, l=it.label: self._emit_material_selected(p, l)
            )
            self._strip_layout.insertWidget(self._strip_layout.count() - 1, btn)
            self._buttons.append(btn)

    def _emit_material_selected(self, path: str, label: str) -> None:
        # If user is in quick-apply mode and selects a new material,
        # automatically exit quick-apply so they can edit the new material first.
        try:
            if getattr(self, "_quick_apply", None) is not None and self._quick_apply.isChecked():
                self._quick_apply.setChecked(False)
        except Exception:
            pass
        try:
            logger.info(f"[Sticker] material clicked: label={label} path={path}")
        except Exception:
            pass
        self.materialSelected.emit(str(path), str(label))

    def _on_import_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select materials folder"), ""
        )
        if not folder:
            return
        exts = (".png", ".jpg", ".jpeg")
        items: List[MaterialItem] = []
        for root, _dirs, files in os.walk(folder):
            for fn in files:
                if not fn.lower().endswith(exts):
                    continue
                path = osp.join(root, fn)
                label = osp.basename(osp.dirname(path)) or "object"
                items.append(MaterialItem(path=path, label=label))
        # stable sort by label then filename
        items.sort(key=lambda x: (x.label, osp.basename(x.path)))
        self._items = items
        self._render()

    def _on_commit(self) -> None:
        blend = self._blend_combo.currentText().strip() or "alpha"
        anno = self._anno_combo.currentData()
        anno = str(anno or "hbb")
        warp = str(self._warp_combo.currentData() or "4")
        appearance = self._appearance_match.isChecked()
        self.commitRequested.emit(blend, anno, warp, appearance)

    def _on_appearance_toggled(self, _enabled: bool) -> None:
        self._emit_settings_changed()

    def _blend_mode_hints(self) -> dict:
        return {
            "alpha": self.tr(
                "Default: standard transparency overlay. Best for typical PNG stickers with clean alpha."
            ),
            "multiply": self.tr(
                "Darkens the background under the sticker. Good for shadows, stains, or dark printed marks."
            ),
            "screen": self.tr(
                "Lightens the background under the sticker. Good for highlights, glow, or light markings."
            ),
            "overlay": self.tr(
                "Increases local contrast (dark stays darker, bright stays brighter). "
                "Often looks more like ink on textured surfaces."
            ),
            "poisson": self.tr(
                "Seamless clone: blends gradients at the boundary for a softer join. "
                "Can look more natural but may ignore fine alpha; falls back to alpha if it fails."
            ),
        }

    def _apply_blend_item_tooltips(self) -> None:
        hints = self._blend_mode_hints()
        for i in range(self._blend_combo.count()):
            mode = self._blend_combo.itemText(i).strip()
            tip = hints.get(mode, "")
            if tip:
                self._blend_combo.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)

    def _refresh_blend_hint(self) -> None:
        mode = self._blend_combo.currentText().strip() or "alpha"
        text = self._blend_mode_hints().get(mode, "")
        self._blend_hint.setText(text)

    def _on_blend_changed(self, _idx: int = 0) -> None:
        self._refresh_blend_hint()
        self._emit_settings_changed()

    def _on_warp_changed(self, _idx: int) -> None:
        try:
            warp = str(self._warp_combo.currentData() or "4")
        except Exception:
            warp = "4"
        self.warpModeChanged.emit(warp)
        self._emit_settings_changed()

    def _emit_move(self, dx: float, dy: float) -> None:
        try:
            self.moveRequested.emit(float(dx), float(dy))
        except Exception:
            pass

    def _on_quick_apply_toggled(self, enabled: bool) -> None:
        try:
            self.quickApplyToggled.emit(bool(enabled))
        except Exception:
            pass
        self._emit_settings_changed()

    def _emit_settings_changed(self, _idx: int = 0) -> None:
        try:
            blend = self._blend_combo.currentText().strip() or "alpha"
            anno = str(self._anno_combo.currentData() or "hbb")
            warp = str(self._warp_combo.currentData() or "4")
            appearance = self._appearance_match.isChecked()
            self.settingsChanged.emit(blend, anno, warp, appearance)
        except Exception:
            pass

    def set_quick_apply_checked(self, checked: bool) -> None:
        """Update quick-apply checkbox without emitting quickApplyToggled."""
        cb = getattr(self, "_quick_apply", None)
        if cb is None:
            return
        try:
            cb.blockSignals(True)
            cb.setChecked(bool(checked))
        finally:
            cb.blockSignals(False)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.closeModeRequested.emit()
        super().closeEvent(event)

