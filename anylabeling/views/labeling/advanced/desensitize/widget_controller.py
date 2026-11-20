"""Dataset desensitization — bound onto LabelingWidget by advanced.bootstrap."""

from __future__ import annotations

import types

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from anylabeling.views.labeling.advanced.desensitize.dock import DesensitizeDock
from anylabeling.views.labeling.advanced.desensitize.mosaic import (
    apply_desensitize_to_file,
)
from anylabeling.views.labeling.logger import logger


class _DesensitizeWidgetMethods:
    def open_desensitize_mode(self) -> None:
        if not self.filename:
            QMessageBox.information(
                self,
                self.tr("Dataset Desensitization"),
                self.tr("No image loaded."),
            )
            return
        try:
            from anylabeling.views.labeling.advanced.bootstrap import (
                install_canvas_hooks,
            )
            from anylabeling.views.labeling.advanced.desensitize.canvas_controller import (
                install_desensitize_support,
            )

            install_canvas_hooks(self.canvas)
            install_desensitize_support(self.canvas)
        except Exception as e:
            QMessageBox.warning(
                self,
                self.tr("Dataset Desensitization"),
                self.tr("Failed to open desensitization: %s") % e,
            )
            return

        dock = getattr(self, "_desensitize_dock", None)
        if dock is None:
            dock = DesensitizeDock(self)
            dock.applyRequested.connect(self._on_desensitize_apply)
            dock.clearMaskRequested.connect(self._on_desensitize_clear_mask)
            dock.closeModeRequested.connect(self._close_desensitize_mode)
            dock.toolChanged.connect(self._on_desensitize_tool_changed)
            dock.brushRadiusChanged.connect(self._on_desensitize_brush_changed)
            dock.blockSizeChanged.connect(self._on_desensitize_block_changed)
            dock.maskToolsEnabledChanged.connect(
                self._on_desensitize_mask_tools_enabled
            )
            self._desensitize_dock = dock
            host = self._find_desensitize_dock_host()
            if host is not None:
                host.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            else:
                dock.setFloating(True)
        dock.show()
        dock.raise_()
        try:
            dock.retranslate_ui()
        except Exception:
            pass

        was_active = getattr(self, "_desensitize_active", False)
        self._sync_desensitize_dock_to_canvas(dock)
        if not was_active:
            self.canvas.desensitize_reset_mask()
            dock.set_mask_tools_enabled(True)
        self.canvas.set_desensitize_session_active(True)
        self.canvas.set_desensitize_enabled(dock.mask_tools_enabled())
        self._desensitize_active = True

    def _sync_desensitize_dock_to_canvas(self, dock: DesensitizeDock) -> None:
        """Apply dock UI state to canvas (tool + sliders must match)."""
        self.canvas.desensitize_set_brush_radius(dock.brush_radius())
        self.canvas.desensitize_set_block_size(dock.block_size())
        self.canvas.desensitize_set_tool(dock.current_tool())

    def _find_desensitize_dock_host(self):
        w = self
        while w is not None:
            if hasattr(w, "addDockWidget") and hasattr(w, "removeDockWidget"):
                return w
            try:
                w = w.parent()
            except Exception:
                w = getattr(w, "parent", None)
        return None

    def _close_desensitize_mode(self) -> None:
        self._desensitize_active = False
        try:
            self.canvas.set_desensitize_session_active(False)
            self.canvas.set_desensitize_enabled(False)
        except Exception:
            pass
        dock = getattr(self, "_desensitize_dock", None)
        if dock is not None:
            try:
                dock.hide()
            except Exception:
                pass

    def _on_desensitize_mask_tools_enabled(self, enabled: bool) -> None:
        if not getattr(self, "_desensitize_active", False):
            return
        try:
            self.canvas.set_desensitize_enabled(enabled)
        except Exception:
            pass

    def _on_desensitize_tool_changed(self, tool: str) -> None:
        if getattr(self, "_desensitize_active", False):
            self.canvas.desensitize_set_tool(tool)

    def _on_desensitize_brush_changed(self, radius: int) -> None:
        if getattr(self, "_desensitize_active", False):
            self.canvas.desensitize_set_brush_radius(radius)

    def _on_desensitize_block_changed(self, block_size: int) -> None:
        if getattr(self, "_desensitize_active", False):
            self.canvas.desensitize_set_block_size(block_size)

    def _on_desensitize_clear_mask(self) -> None:
        if getattr(self, "_desensitize_active", False):
            self.canvas.desensitize_clear_mask()

    def _on_desensitize_apply(self) -> None:
        if not self.filename:
            return
        mask = None
        try:
            mask = self.canvas.desensitize_mask_array()
        except Exception:
            pass
        if mask is None or not (mask > 0).any():
            QMessageBox.information(
                self,
                self.tr("Dataset Desensitization"),
                self.tr("Draw a mask region before applying."),
            )
            return

        block_size = 12
        effect = "mosaic"
        gray_level = 128
        dock = getattr(self, "_desensitize_dock", None)
        if dock is not None:
            block_size = dock.block_size()
            effect = dock.effect_mode()
            gray_level = dock.gray_level()

        if effect == "solid":
            confirm = self.tr(
                "Apply solid gray fill (level {level}) to the marked regions "
                "and overwrite the current image file?\n{path}"
            ).format(level=gray_level, path=self.filename)
        else:
            confirm = self.tr(
                "Apply mosaic to the marked regions and overwrite the "
                "current image file?\n{path}"
            ).format(path=self.filename)

        reply = QMessageBox.question(
            self,
            self.tr("Dataset Desensitization"),
            confirm,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            apply_desensitize_to_file(
                str(self.filename),
                mask,
                effect=effect,
                block_size=block_size,
                gray_level=gray_level,
            )
        except Exception as e:
            logger.exception("Desensitize apply failed")
            QMessageBox.warning(
                self,
                self.tr("Dataset Desensitization"),
                self.tr("Failed to apply mosaic: %s") % e,
            )
            return

        self.canvas.desensitize_clear_mask()
        try:
            self.load_file(self.filename)
        except Exception as e:
            QMessageBox.information(
                self,
                self.tr("Dataset Desensitization"),
                self.tr("Mosaic saved, but reload failed: %s") % e,
            )
            return
        QMessageBox.information(
            self,
            self.tr("Dataset Desensitization"),
            self.tr("Mosaic applied to the current image."),
        )


def install(widget) -> None:
    for name in dir(_DesensitizeWidgetMethods):
        if name.startswith("__"):
            continue
        attr = getattr(_DesensitizeWidgetMethods, name, None)
        if callable(attr):
            setattr(widget, name, types.MethodType(attr, widget))
    widget._desensitize_dock = None
    widget._desensitize_active = False

    if not getattr(widget, "_desensitize_load_wrapped", False):
        original_load = widget.load_file

        def _load_file_reset_desensitize_mask(self_w, *args, **kwargs):
            ok = original_load(*args, **kwargs)
            if getattr(self_w, "_desensitize_active", False):
                try:
                    self_w.canvas.desensitize_reset_mask()
                    self_w.canvas._desensitize_cancel_drawing()
                except Exception:
                    pass
            return ok

        widget.load_file = types.MethodType(_load_file_reset_desensitize_mask, widget)
        widget._desensitize_load_wrapped = True