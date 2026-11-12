"""Compare-review + inspector overlay helpers installed onto Canvas."""

from __future__ import annotations

import types
from typing import Optional

from PyQt6 import QtCore, QtGui

from anylabeling.views.labeling.shape import Shape


def _annotation_compare_review_active(self) -> bool:
    # Prefer explicit weakref set by advanced.bootstrap
    ref = getattr(self, "_advanced_labeling_widget_ref", None)
    if ref is not None:
        widget = ref() if callable(ref) else None
        if widget is not None:
            return bool(getattr(widget, "_annotation_compare_active", False))
    parent = self.parent()
    while parent is not None:
        if hasattr(parent, "_annotation_compare_active"):
            return bool(getattr(parent, "_annotation_compare_active", False))
        parent = parent.parent() if hasattr(parent, "parent") else None
    return False


def _compare_status_label_rect(
    self, shape, font: QtGui.QFont, fm: QtGui.QFontMetrics
) -> Optional[QtCore.QRect]:
    from anylabeling.views.labeling.advanced.compare.annotation_compare import (
        PENDING_REVIEW_STATUSES,
        compare_status_display_text,
        get_compare_status,
    )

    if not self._annotation_compare_review_active():
        return None
    status = get_compare_status(shape)
    if status not in PENDING_REVIEW_STATUSES:
        return None
    label = compare_status_display_text(status)
    if not label:
        return None

    text_rect = fm.tightBoundingRect(label)
    padding_x = 8
    padding_y = 2
    rect_width = text_rect.width() + 2 * padding_x
    rect_height = fm.height() + 2 * padding_y
    d_react = shape.point_size / shape.scale

    if shape.shape_type in [
        "rectangle",
        "polygon",
        "rotation",
        "quadrilateral",
        "cuboid",
    ]:
        try:
            bbox = shape.bounding_rect()
        except IndexError:
            return None
        return QtCore.QRect(
            int(bbox.x()),
            int(bbox.y() + bbox.height() + 1),
            rect_width,
            rect_height,
        )
    if shape.shape_type in ["circle", "line", "linestrip", "point"]:
        points = shape.points
        if not points:
            return None
        point = points[0]
        return QtCore.QRect(
            int(point.x() + d_react),
            int(point.y() + 1),
            rect_width,
            rect_height,
        )
    return None


def _hit_compare_review_label(self, pos: QtCore.QPointF):
    if not self._annotation_compare_review_active():
        return None
    font_size = int(max(8.0, int(round(10.0 / Shape.scale))))
    font = QtGui.QFont("Arial", font_size, QtGui.QFont.Weight.Bold)
    fm = QtGui.QFontMetrics(font)
    hit_pad = 6
    for shape in reversed(self.shapes):
        if not shape.visible:
            continue
        rect = self._compare_status_label_rect(shape, font, fm)
        if rect is None:
            continue
        hit_rect = rect.adjusted(-hit_pad, -hit_pad, hit_pad, hit_pad)
        if hit_rect.contains(int(pos.x()), int(pos.y())):
            return shape
    return None


def set_inspector_overlays_from_bbox01(self, bbox01) -> None:
    try:
        if not bbox01 or len(bbox01) != 4:
            self._inspector_overlays = []
            self.update()
            return
        x0, y0, x1, y1 = [float(v) for v in bbox01]
        w = float(self.pixmap.width()) if self.pixmap and not self.pixmap.isNull() else 0
        h = float(self.pixmap.height()) if self.pixmap and not self.pixmap.isNull() else 0
        if w <= 0 or h <= 0:
            self._inspector_overlays = []
            self.update()
            return
        rect = QtCore.QRectF(x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h)
        self._inspector_overlays = [rect]
        self.update()
    except Exception:
        self._inspector_overlays = []
        self.update()


def _advanced_paint_overlays(self, painter: QtGui.QPainter) -> None:
    """Draw sticker preview, inspector overlays, and compare status chips."""
    # Sticker
    try:
        if getattr(self, "_sticker_enabled", False) and getattr(self, "_sticker", None):
            st = self._sticker
            from anylabeling.views.labeling.advanced.sticker.canvas_controller import (
                _sticker_corners,
            )

            corners = _sticker_corners(st)
            poly = QtGui.QPolygonF(list(corners))
            painter.save()
            painter.setRenderHint(
                QtGui.QPainter.RenderHint.SmoothPixmapTransform, True
            )

            # Ensure warped preview is scheduled when dirty.
            if getattr(st, "preview_dirty", False):
                mark = getattr(self, "_sticker_mark_dirty", None)
                if callable(mark):
                    mark()

            # preview_pixmap is a full-frame warp of the base image (possibly downscaled).
            if st.preview_pixmap is not None and not st.preview_pixmap.isNull():
                target = QtCore.QRect(
                    0,
                    0,
                    int(self.pixmap.width()),
                    int(self.pixmap.height()),
                )
                painter.drawPixmap(target, st.preview_pixmap)
            elif st.pixmap is not None and not st.pixmap.isNull():
                # Interim fallback until warped preview is ready.
                br = poly.boundingRect()
                painter.drawPixmap(br.toRect(), st.pixmap)

            # Always draw control polygon + handles so the overlay is editable
            # even before / without a warped preview.
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 200, 255, 220), 2))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPolygon(poly)
            r = max(3.0, 6.0 / float(max(Shape.scale, 1e-6)))
            painter.setBrush(QtGui.QColor(0, 200, 255, 200))
            pts = (
                st.points
                if str(getattr(st, "warp_mode", "4") or "4") == "8"
                else corners
            )
            for pt in pts:
                painter.drawEllipse(pt, r, r)
            painter.restore()
    except Exception:
        pass

    # Inspector FN overlays
    if getattr(self, "_inspector_overlays", None):
        try:
            pen = QtGui.QPen(QtGui.QColor(255, 215, 0, 220))
            pen.setWidth(max(1, int(round(4.0 / Shape.scale))))
            painter.setPen(pen)
            painter.setBrush(QtGui.QColor(255, 255, 0, 40))
            for r in self._inspector_overlays:
                if isinstance(r, QtCore.QRectF):
                    painter.drawRect(r)
        except Exception:
            pass

    # Compare status labels
    try:
        if self._annotation_compare_review_active():
            from anylabeling.views.labeling.advanced.compare.annotation_compare import (
                PENDING_REVIEW_STATUSES,
                compare_status_display_text,
                get_compare_status,
            )

            font_size = int(max(8.0, int(round(10.0 / Shape.scale))))
            font = QtGui.QFont("Arial", font_size, QtGui.QFont.Weight.Bold)
            painter.setFont(font)
            fm = QtGui.QFontMetrics(font)
            for shape in self.shapes:
                if not shape.visible:
                    continue
                status = get_compare_status(shape)
                if status not in PENDING_REVIEW_STATUSES:
                    continue
                label = compare_status_display_text(status)
                if not label:
                    continue
                rect = self._compare_status_label_rect(shape, font, fm)
                if rect is None:
                    continue
                painter.fillRect(rect, QtGui.QColor(0, 0, 0, 160))
                painter.setPen(QtGui.QColor(255, 255, 0))
                painter.drawText(
                    rect.adjusted(8, 0, 0, 0),
                    QtCore.Qt.AlignmentFlag.AlignVCenter
                    | QtCore.Qt.AlignmentFlag.AlignLeft,
                    label,
                )
    except Exception:
        pass


def install_compare_canvas_hooks(canvas) -> None:
    if getattr(canvas, "_advanced_compare_hooks_installed", False):
        return
    canvas._inspector_overlays = []
    for name, fn in (
        ("_annotation_compare_review_active", _annotation_compare_review_active),
        ("_compare_status_label_rect", _compare_status_label_rect),
        ("_hit_compare_review_label", _hit_compare_review_label),
        ("set_inspector_overlays_from_bbox01", set_inspector_overlays_from_bbox01),
        ("_advanced_paint_overlays", _advanced_paint_overlays),
    ):
        setattr(canvas, name, types.MethodType(fn, canvas))

    canvas._advanced_compare_hooks_installed = True
