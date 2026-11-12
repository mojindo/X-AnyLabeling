"""Sticker dock wiring handlers — bound onto LabelingWidget by advanced.bootstrap."""
from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.shape import Shape


class _StickerWidgetMethods:
    """Mixin-style methods; install() attaches them to a LabelingWidget instance."""

    def open_sticker_mode(self):
        """Open Sticker (material overlay) dock under Advanced menu."""
        if not self.filename:
            QMessageBox.information(
                self,
                self.tr("Sticker"),
                self.tr("No image loaded."),
            )
            return
        try:
            from anylabeling.views.labeling.advanced.bootstrap import (
                install_canvas_hooks,
            )
            from anylabeling.views.labeling.advanced.sticker.library_dock import (
                StickerLibraryDock,
            )

            install_canvas_hooks(self.canvas)
        except Exception as e:
            QMessageBox.warning(
                self,
                self.tr("Sticker"),
                self.tr("Failed to open Sticker dock: %s") % str(e),
            )
            return

        if getattr(self, "_sticker_dock", None) is None:
            self._sticker_dock = StickerLibraryDock(self)
            self._sticker_dock.materialSelected.connect(self._on_sticker_material_selected)
            self._sticker_dock.commitRequested.connect(self._on_sticker_commit_requested)
            self._sticker_dock.cancelRequested.connect(self._on_sticker_cancel_requested)
            try:
                self._sticker_dock.rotateRequested.connect(self._on_sticker_rotate_requested)
            except Exception:
                pass
            try:
                self._sticker_dock.moveRequested.connect(self._on_sticker_move_requested)
            except Exception:
                pass
            try:
                self._sticker_dock.warpModeChanged.connect(self._on_sticker_warp_mode_changed)
            except Exception:
                pass
            try:
                self._sticker_dock.quickApplyToggled.connect(self._on_sticker_quick_apply_toggled)
                self._sticker_dock.settingsChanged.connect(self._on_sticker_settings_changed)
            except Exception:
                pass
            self._sticker_dock.setFloating(True)
        self._sticker_dock.show()
        self._sticker_dock.raise_()
        self._sticker_dock.activateWindow()
        try:
            self.canvas.set_sticker_enabled(True)
            # Ensure user can manipulate sticker immediately.
            self.set_edit_mode()
        except Exception:
            pass


    def _on_sticker_material_selected(self, path: str, label: str) -> None:
        try:
            logger.info(f"[Sticker] received materialSelected: label={label} path={path}")
        except Exception:
            pass
        # Ensure sticker mode is enabled before setting material, so that
        # clicking a material always shows a live preview immediately.
        try:
            self.canvas.set_sticker_enabled(True)
            self.set_edit_mode()
        except Exception:
            pass
        try:
            self.canvas.sticker_set_material(path, label)
        except Exception as e:
            QMessageBox.warning(
                self,
                self.tr("Sticker"),
                self.tr("Failed to preview material: %s") % str(e),
            )


    def _on_sticker_commit_requested(self, blend_mode: str, anno_mode: str, warp_mode: str) -> None:
        try:
            self.canvas.sticker_commit(
                blend_mode=str(blend_mode or "alpha"),
                anno_mode=str(anno_mode or "hbb"),
                output_dir=self.output_dir,
                warp_mode=str(warp_mode or "4"),
            )
            # reload current image after overwrite
            if self.filename:
                self.load_file(self.filename)
        except Exception as e:
            QMessageBox.warning(
                self,
                self.tr("Sticker"),
                self.tr("Failed to apply sticker: %s") % str(e),
            )


    def _on_sticker_warp_mode_changed(self, warp_mode: str) -> None:
        try:
            self.canvas.sticker_set_warp_mode(str(warp_mode or "4"))
        except Exception:
            pass


    def _on_sticker_quick_apply_toggled(self, enabled: bool) -> None:
        try:
            # Quick-apply is meant to work immediately on canvas clicks.
            self.canvas.set_sticker_enabled(True)
            self.set_edit_mode()
            self.canvas.sticker_set_quick_apply_enabled(bool(enabled), output_dir=self.output_dir)
        except Exception:
            pass


    def _on_sticker_settings_changed(self, blend: str, anno: str, warp: str) -> None:
        try:
            self.canvas.sticker_set_quick_apply_settings(
                blend_mode=str(blend or "alpha"),
                anno_mode=str(anno or "hbb"),
                warp_mode=str(warp or "4"),
                output_dir=self.output_dir,
            )
        except Exception:
            pass


    def _on_sticker_cancel_requested(self) -> None:
        try:
            self.canvas.sticker_cancel()
        except Exception:
            pass


    def _on_sticker_rotate_requested(self, deg_delta: float) -> None:
        try:
            self.canvas.sticker_rotate(float(deg_delta))
        except Exception:
            pass


    def _on_sticker_move_requested(self, dx: float, dy: float) -> None:
        try:
            self.canvas.sticker_nudge(float(dx), float(dy))
        except Exception:
            pass

    class _InspectorBuildWorker(QtCore.QObject):
        progress = QtCore.pyqtSignal(int, int, str)
        finished = QtCore.pyqtSignal(bool, str)

        def __init__(
            self,
            *,
            cache: InspectorCache,
            image_paths: list,
            output_dir: Optional[str],
            labels_filter: list,
        ) -> None:
            super().__init__()
            self._cache = cache
            self._image_paths = image_paths
            self._output_dir = output_dir
            self._labels_filter = labels_filter
            self._cancel = False

        def request_cancel(self) -> None:
            self._cancel = True

        def _should_cancel(self) -> bool:
            return bool(self._cancel)

        @QtCore.pyqtSlot()
        def run(self) -> None:
            try:
                self._cache.build_index(
                    self._image_paths,
                    output_dir=self._output_dir,
                    labels_filter=self._labels_filter,
                    force=True,
                    on_progress=lambda i, n, p: self.progress.emit(i, n, p),
                    should_cancel=self._should_cancel,
                )
                self.finished.emit(True, "")
            except Exception as e:
                self.finished.emit(False, str(e))




def install(widget) -> None:
    import types
    for name in dir(_StickerWidgetMethods):
        if name.startswith("_") and name not in (
            # keep private helpers
        ):
            pass
        attr = getattr(_StickerWidgetMethods, name, None)
        if name.startswith("__"):
            continue
        if callable(attr):
            setattr(widget, name, types.MethodType(attr, widget))
