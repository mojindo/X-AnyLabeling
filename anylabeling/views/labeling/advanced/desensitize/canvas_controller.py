"""Canvas interaction for dataset desensitization (mask + preview)."""

from __future__ import annotations

import types
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import Qt


class _DesensitizeMethods:
    def set_desensitize_session_active(self, active: bool) -> None:
        """Dock open: show mask overlay; does not capture canvas input."""
        self._desensitize_session_active = bool(active)
        if not active:
            self._desensitize_enabled = False
            self._desensitize_cancel_drawing()
            self._desensitize_mask = None
        self._desensitize_overlay_dirty = True
        self.update()

    def set_desensitize_enabled(self, enabled: bool) -> None:
        """When True: only mask tools; block label edit and canvas pan."""
        self._desensitize_enabled = bool(enabled)
        if not enabled:
            self._desensitize_cancel_drawing()
        self._desensitize_overlay_dirty = True
        self.update()

    def desensitize_set_tool(self, tool: str) -> None:
        t = str(tool or "brush").lower()
        if t not in ("brush", "rectangle", "polygon"):
            t = "brush"
        if self._desensitize_tool != t:
            self._desensitize_cancel_drawing()
        self._desensitize_tool = t
        self.update()

    def desensitize_set_brush_radius(self, radius: int) -> None:
        self._desensitize_brush_radius = max(1, min(200, int(radius)))

    def desensitize_set_block_size(self, block_size: int) -> None:
        self._desensitize_block_size = max(2, min(64, int(block_size)))

    def desensitize_clear_mask(self) -> None:
        self._desensitize_reset_mask()
        self._desensitize_cancel_drawing()
        self.update()

    def desensitize_mask_array(self) -> Optional[np.ndarray]:
        return self._desensitize_mask

    def desensitize_reset_mask(self) -> None:
        self._desensitize_reset_mask()

    def _desensitize_reset_mask(self) -> None:
        if self.pixmap is None or self.pixmap.isNull():
            self._desensitize_mask = None
            return
        h, w = int(self.pixmap.height()), int(self.pixmap.width())
        self._desensitize_mask = np.zeros((h, w), dtype=np.uint8)
        self._desensitize_overlay_dirty = True

    def _desensitize_ensure_mask(self) -> bool:
        if self.pixmap is None or self.pixmap.isNull():
            return False
        h, w = int(self.pixmap.height()), int(self.pixmap.width())
        if (
            self._desensitize_mask is None
            or self._desensitize_mask.shape[0] != h
            or self._desensitize_mask.shape[1] != w
        ):
            self._desensitize_reset_mask()
        return self._desensitize_mask is not None

    def _desensitize_cancel_drawing(self) -> None:
        self._desensitize_drag_active = False
        self._desensitize_rect_start = None
        self._desensitize_rect_current = None
        self._desensitize_poly_points = []

    def _desensitize_stamp_brush(self, x: float, y: float, *, erase: bool = False) -> None:
        if not self._desensitize_ensure_mask():
            return
        Canvas = type(self)
        Canvas._apply_brush_to_mask(
            self._desensitize_mask,
            x,
            y,
            max(1, int(self._desensitize_brush_radius)),
            add=not erase,
        )
        self._desensitize_overlay_dirty = True

    def _desensitize_fill_rect(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> None:
        if not self._desensitize_ensure_mask():
            return
        xmin = int(round(min(x1, x2)))
        xmax = int(round(max(x1, x2)))
        ymin = int(round(min(y1, y2)))
        ymax = int(round(max(y1, y2)))
        h, w = self._desensitize_mask.shape[:2]
        xmin = max(0, min(xmin, w - 1))
        xmax = max(0, min(xmax, w - 1))
        ymin = max(0, min(ymin, h - 1))
        ymax = max(0, min(ymax, h - 1))
        if xmax <= xmin or ymax <= ymin:
            return
        cv2.rectangle(
            self._desensitize_mask,
            (xmin, ymin),
            (xmax, ymax),
            255,
            thickness=-1,
        )
        self._desensitize_overlay_dirty = True

    def _desensitize_fill_polygon(self, points: List[Tuple[float, float]]) -> None:
        if not self._desensitize_ensure_mask() or len(points) < 3:
            return
        pts = np.array(
            [[float(x), float(y)] for x, y in points],
            dtype=np.float32,
        )
        h, w = self._desensitize_mask.shape[:2]
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        poly = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(self._desensitize_mask, [poly], 255)
        self._desensitize_overlay_dirty = True

    def _desensitize_paint_overlay(self, painter: QtGui.QPainter) -> None:
        if not getattr(self, "_desensitize_session_active", False):
            return
        if not self._desensitize_ensure_mask():
            return
        if not np.any(self._desensitize_mask):
            pass
        else:
            if getattr(self, "_desensitize_overlay_dirty", True):
                self._desensitize_rebuild_overlay_qimage()
            qimg = getattr(self, "_desensitize_overlay_qimage", None)
            if qimg is not None and not qimg.isNull():
                painter.save()
                painter.setOpacity(0.45)
                painter.drawImage(0, 0, qimg)
                painter.restore()

        if self._desensitize_tool == "rectangle" and self._desensitize_rect_start:
            a = self._desensitize_rect_start
            b = self._desensitize_rect_current or a
            pen = QtGui.QPen(QtGui.QColor(255, 80, 80, 220), max(1.0, 2.0 / Shape.scale))
            painter.setPen(pen)
            painter.setBrush(QtGui.QColor(255, 0, 0, 40))
            x1, y1 = float(a.x()), float(a.y())
            x2, y2 = float(b.x()), float(b.y())
            painter.drawRect(
                QtCore.QRectF(
                    min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
                )
            )

        if self._desensitize_tool == "polygon" and self._desensitize_poly_points:
            pen = QtGui.QPen(QtGui.QColor(255, 120, 0, 220), max(1.0, 2.0 / Shape.scale))
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            pts = [
                QtCore.QPointF(float(x), float(y))
                for x, y in self._desensitize_poly_points
            ]
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    painter.drawLine(pts[i], pts[i + 1])
            for pt in pts:
                r = max(2.0, 4.0 / float(max(Shape.scale, 1e-6)))
                painter.drawEllipse(pt, r, r)

    def _desensitize_rebuild_overlay_qimage(self) -> None:
        if self._desensitize_mask is None:
            self._desensitize_overlay_qimage = None
            self._desensitize_overlay_dirty = False
            return
        h, w = self._desensitize_mask.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        sel = self._desensitize_mask > 0
        rgba[sel, 0] = 255
        rgba[sel, 1] = 60
        rgba[sel, 2] = 60
        rgba[sel, 3] = 140
        qimg = QtGui.QImage(
            rgba.data,
            w,
            h,
            rgba.strides[0],
            QtGui.QImage.Format.Format_RGBA8888,
        ).copy()
        self._desensitize_overlay_qimage = qimg
        self._desensitize_overlay_dirty = False

    def _desensitize_handle_mouse_press(self, ev, *, pos: QtCore.QPointF) -> bool:
        if not getattr(self, "_desensitize_enabled", False):
            return False
        if not self._has_valid_pixmap():
            return False
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            if self._desensitize_tool == "polygon" and len(self._desensitize_poly_points) >= 3:
                self._desensitize_fill_polygon(self._desensitize_poly_points)
                self._desensitize_poly_points = []
                self.update()
                return True
            return False

        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return False

        tool = self._desensitize_tool
        x, y = float(pos.x()), float(pos.y())

        if tool == "brush":
            erase = bool(
                ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
            )
            self._desensitize_stamp_brush(x, y, erase=erase)
            self._desensitize_drag_active = True
            self.update()
            return True

        if tool == "rectangle":
            self._desensitize_rect_start = QtCore.QPointF(pos)
            self._desensitize_rect_current = QtCore.QPointF(pos)
            self._desensitize_drag_active = True
            self.update()
            return True

        if tool == "polygon":
            self._desensitize_poly_points.append((x, y))
            self.update()
            return True

        return False

    def _desensitize_block_canvas_navigation(self, ev) -> bool:
        if not getattr(self, "_desensitize_enabled", False):
            return False
        if getattr(self, "_space_panning", False) or getattr(
            self, "_space_pressed", False
        ):
            return True
        if getattr(self, "_right_panning", False) or getattr(
            self, "_right_pan_start", None
        ) is not None:
            return True
        button = ev.button() if hasattr(ev, "button") and callable(ev.button) else None
        if button in (
            QtCore.Qt.MouseButton.MiddleButton,
            QtCore.Qt.MouseButton.RightButton,
        ):
            return not (
                self._desensitize_tool == "polygon"
                and button == QtCore.Qt.MouseButton.RightButton
                and len(self._desensitize_poly_points) >= 3
            )
        buttons = ev.buttons() if hasattr(ev, "buttons") else QtCore.Qt.MouseButton.NoButton
        if buttons & (
            QtCore.Qt.MouseButton.LeftButton
            | QtCore.Qt.MouseButton.MiddleButton
        ):
            return True
        return False

    def _desensitize_handle_mouse_move(self, ev, *, pos: QtCore.QPointF) -> bool:
        if not getattr(self, "_desensitize_enabled", False):
            return False
        x, y = float(pos.x()), float(pos.y())

        if self._desensitize_tool == "brush" and self._desensitize_drag_active:
            if ev.buttons() & QtCore.Qt.MouseButton.LeftButton:
                erase = bool(
                    ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
                )
                self._desensitize_stamp_brush(x, y, erase=erase)
                self.update()
                return True

        if (
            self._desensitize_tool == "rectangle"
            and self._desensitize_drag_active
            and self._desensitize_rect_start is not None
        ):
            self._desensitize_rect_current = QtCore.QPointF(pos)
            self.update()
            return True

        if self._desensitize_tool == "polygon" and (
            ev.buttons() & QtCore.Qt.MouseButton.LeftButton
        ):
            return True

        if self._desensitize_block_canvas_navigation(ev):
            return True

        return False

    def _desensitize_handle_mouse_release(self, ev) -> bool:
        if not getattr(self, "_desensitize_enabled", False):
            return False
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return False

        if self._desensitize_tool == "brush":
            self._desensitize_drag_active = False
            return True

        if self._desensitize_tool == "rectangle" and self._desensitize_rect_start:
            a = self._desensitize_rect_start
            b = self._desensitize_rect_current or a
            self._desensitize_fill_rect(a.x(), a.y(), b.x(), b.y())
            self._desensitize_cancel_drawing()
            self.update()
            return True

        if self._desensitize_tool == "polygon":
            return True

        return False

    def _desensitize_handle_wheel(self, ev: QtGui.QWheelEvent) -> bool:
        if not getattr(self, "_desensitize_enabled", False):
            return False
        if not (ev.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier):
            return False
        if self._desensitize_tool != "brush":
            return False
        delta = ev.angleDelta().y()
        if delta == 0:
            return False
        step = 1 if delta > 0 else -1
        self._desensitize_brush_radius = max(
            1, min(200, int(self._desensitize_brush_radius) + step)
        )
        parent = self.parent()
        dock = getattr(parent, "_desensitize_dock", None) if parent else None
        if dock is not None and hasattr(dock, "_brush_spin"):
            dock._brush_spin.blockSignals(True)
            dock._brush_spin.setValue(self._desensitize_brush_radius)
            dock._brush_spin.blockSignals(False)
        self.update()
        return True

    def _desensitize_handle_key(self, ev) -> bool:
        if not getattr(self, "_desensitize_enabled", False):
            return False
        if ev.key() == QtCore.Qt.Key.Key_Escape:
            self._desensitize_cancel_drawing()
            self.update()
            return True
        if ev.key() in (
            QtCore.Qt.Key.Key_Return,
            QtCore.Qt.Key.Key_Enter,
        ):
            if (
                self._desensitize_tool == "polygon"
                and len(self._desensitize_poly_points) >= 3
            ):
                self._desensitize_fill_polygon(self._desensitize_poly_points)
                self._desensitize_poly_points = []
                self.update()
                return True
        return False


from anylabeling.views.labeling.shape import Shape  # noqa: E402  — after class for scale ref


def install_desensitize_support(canvas) -> None:
    if getattr(canvas, "_desensitize_support_installed", False):
        return

    for name in dir(_DesensitizeMethods):
        if name.startswith("__"):
            continue
        attr = getattr(_DesensitizeMethods, name, None)
        if callable(attr) and (
            name.startswith("desensitize")
            or name.startswith("_desensitize")
            or name == "set_desensitize_enabled"
            or name == "set_desensitize_session_active"
        ):
            setattr(canvas, name, types.MethodType(attr, canvas))

    canvas._desensitize_session_active = False
    canvas._desensitize_enabled = False
    canvas._desensitize_mask = None
    canvas._desensitize_tool = "brush"
    canvas._desensitize_brush_radius = 24
    canvas._desensitize_block_size = 12
    canvas._desensitize_drag_active = False
    canvas._desensitize_rect_start = None
    canvas._desensitize_rect_current = None
    canvas._desensitize_poly_points = []
    canvas._desensitize_overlay_dirty = True
    canvas._desensitize_overlay_qimage = None

    _wrap_event(canvas, "mousePressEvent", _mouse_press_before)
    _wrap_event(canvas, "mouseMoveEvent", _mouse_move_before)
    _wrap_event(canvas, "mouseReleaseEvent", _mouse_release_before)
    _wrap_event(canvas, "wheelEvent", _wheel_before)
    _wrap_event(canvas, "keyPressEvent", _key_before)
    _wrap_event(canvas, "keyReleaseEvent", _key_release_before)

    canvas._desensitize_support_installed = True


def _wrap_event(canvas, name: str, wrapper_factory):
    original = getattr(canvas, name)

    def bound_original(*args, **kwargs):
        return original(*args, **kwargs)

    wrapped = wrapper_factory(canvas, bound_original)
    setattr(canvas, name, wrapped)


def _mouse_press_before(canvas, original):
    def mousePressEvent(ev):
        try:
            if getattr(canvas, "_desensitize_enabled", False):
                pos = (
                    canvas.transform_pos(ev.position())
                    if hasattr(canvas, "transform_pos")
                    else ev.position()
                )
                handler = getattr(canvas, "_desensitize_handle_mouse_press", None)
                if callable(handler) and handler(ev, pos=pos):
                    ev.accept()
                    return
                block = getattr(canvas, "_desensitize_block_canvas_navigation", None)
                if callable(block) and block(ev):
                    ev.accept()
                    return
        except Exception:
            pass
        return original(ev)

    return mousePressEvent


def _mouse_move_before(canvas, original):
    def mouseMoveEvent(ev):
        try:
            if getattr(canvas, "_desensitize_enabled", False):
                pos = (
                    canvas.transform_pos(ev.position())
                    if hasattr(canvas, "transform_pos")
                    else ev.position()
                )
                handler = getattr(canvas, "_desensitize_handle_mouse_move", None)
                if callable(handler) and handler(ev, pos=pos):
                    ev.accept()
                    return
                block = getattr(canvas, "_desensitize_block_canvas_navigation", None)
                if callable(block) and block(ev):
                    ev.accept()
                    return
        except Exception:
            pass
        return original(ev)

    return mouseMoveEvent


def _mouse_release_before(canvas, original):
    def mouseReleaseEvent(ev):
        try:
            if getattr(canvas, "_desensitize_enabled", False):
                handler = getattr(canvas, "_desensitize_handle_mouse_release", None)
                if callable(handler) and handler(ev):
                    ev.accept()
                    return
                block = getattr(canvas, "_desensitize_block_canvas_navigation", None)
                if callable(block) and block(ev):
                    ev.accept()
                    return
        except Exception:
            pass
        return original(ev)

    return mouseReleaseEvent


def _wheel_before(canvas, original):
    def wheelEvent(ev):
        try:
            if getattr(canvas, "_desensitize_enabled", False):
                handler = getattr(canvas, "_desensitize_handle_wheel", None)
                if callable(handler) and handler(ev):
                    ev.accept()
                    return
                if ev.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
                    ev.accept()
                    return
        except Exception:
            pass
        return original(ev)

    return wheelEvent


def _key_before(canvas, original):
    def keyPressEvent(ev):
        try:
            if getattr(canvas, "_desensitize_enabled", False):
                handler = getattr(canvas, "_desensitize_handle_key", None)
                if callable(handler) and handler(ev):
                    ev.accept()
                    return
                if ev.key() == QtCore.Qt.Key.Key_Space and not ev.isAutoRepeat():
                    ev.accept()
                    return
        except Exception:
            pass
        return original(ev)

    return keyPressEvent


def _key_release_before(canvas, original):
    def keyReleaseEvent(ev):
        try:
            if getattr(canvas, "_desensitize_enabled", False):
                if ev.key() == QtCore.Qt.Key.Key_Space and not ev.isAutoRepeat():
                    ev.accept()
                    return
        except Exception:
            pass
        return original(ev)

    return keyReleaseEvent