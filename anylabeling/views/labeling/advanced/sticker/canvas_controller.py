"""Sticker canvas controller extracted from B-AnyLabeling (modular hook)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent

import numpy as np

from anylabeling.views.labeling.logger import logger

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class _StickerOverlay:
    path: str
    label: str
    pixmap: QtGui.QPixmap
    enabled: bool = True
    # Control points in image coordinates (pixmap space).
    # 8-point warp: 0..3 are corners (tl,tr,br,bl), 4..7 are edge midpoints
    # (top,right,bottom,left). Move handle uses selected_handle==8.
    points: Tuple[QtCore.QPointF, ...] = ()
    warp_mode: str = "4"  # "4" fast perspective, "8" mesh warp
    # Midpoint locks: when True, edge midpoint follows its two corners.
    # Order: top,right,bottom,left (points[4:8]).
    mid_locks: Tuple[bool, bool, bool, bool] = (True, True, True, True)
    # Interaction state
    selected_handle: int = -1  # 0..7 points, 8=move whole
    last_pos: Optional[QtCore.QPointF] = None
    debug_logged: bool = False
    # Preview cache (avoid Qt quadToQuad inconsistencies across bindings)
    rgba: Optional[np.ndarray] = None  # HxWx4 BGRA
    preview_pixmap: Optional[QtGui.QPixmap] = None
    preview_dirty: bool = True
    preview_size: Tuple[int, int] = (0, 0)
    preview_timer_pending: bool = False


def _sticker_build_preview_bgra(
    *,
    material_bgra: np.ndarray,
    base_w: int,
    base_h: int,
    preview_max: int,
    points8: Tuple[QtCore.QPointF, ...],
    warp_mode: str,
) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
    """
    Build a BGRA preview image (ph,pw,4) for sticker overlay.
    Heavy OpenCV work must not run inside paintEvent.
    """
    if material_bgra is None or material_bgra.size == 0:
        return None, (0, 0)
    if base_w <= 0 or base_h <= 0:
        return None, (0, 0)
    if not points8 or len(points8) < 4:
        return None, (0, 0)

    max_dim = max(1, int(max(base_w, base_h)))
    target_max = max(64, int(preview_max))
    s = 1.0 if max_dim <= target_max else (float(target_max) / float(max_dim))
    pw = max(1, int(round(base_w * s)))
    ph = max(1, int(round(base_h * s)))

    mh, mw = material_bgra.shape[:2]
    mode = str(warp_mode or "4")
    corners = list(points8)[0:4]
    sx = float(pw) / float(base_w)
    sy = float(ph) / float(base_h)

    if mode != "8":
        src = np.array(
            [
                [0.0, 0.0],
                [float(mw - 1), 0.0],
                [float(mw - 1), float(mh - 1)],
                [0.0, float(mh - 1)],
            ],
            dtype=np.float32,
        )
        dst = np.array(
            [[float(p.x()) * sx, float(p.y()) * sy] for p in corners],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            material_bgra,
            M,
            (pw, ph),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        return warped, (pw, ph)

    grid_full = _sticker_points_to_grid3x3(tuple(points8[0:8]))  # type: ignore
    if not grid_full or len(grid_full) != 9:
        return None, (pw, ph)
    grid = tuple(QtCore.QPointF(p.x() * sx, p.y() * sy) for p in grid_full)
    if not grid or len(grid) != 9:
        return None, (pw, ph)

    x0, x1, x2 = 0.0, float(mw - 1) / 2.0, float(mw - 1)
    y0, y1, y2 = 0.0, float(mh - 1) / 2.0, float(mh - 1)
    xs = [x0, x1, x2]
    ys = [y0, y1, y2]

    out = np.zeros((ph, pw, 4), dtype=np.uint8)

    def _alpha_compose(dst_bgra: np.ndarray, src_bgra: np.ndarray) -> np.ndarray:
        sa = (src_bgra[:, :, 3].astype(np.float32) / 255.0)[..., None]
        da = (dst_bgra[:, :, 3].astype(np.float32) / 255.0)[..., None]
        out_a = sa + da * (1.0 - sa)
        src_rgb = src_bgra[:, :, :3].astype(np.float32)
        dst_rgb = dst_bgra[:, :, :3].astype(np.float32)
        denom = np.clip(out_a, 1e-6, 1.0)
        out_rgb = (src_rgb * sa + dst_rgb * da * (1.0 - sa)) / denom
        out2 = dst_bgra.copy()
        out2[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        out2[:, :, 3] = np.clip(out_a[:, :, 0] * 255.0, 0, 255).astype(np.uint8)
        return out2

    for gy in range(2):
        for gx in range(2):
            sx0, sx1 = xs[gx], xs[gx + 1]
            sy0, sy1 = ys[gy], ys[gy + 1]
            ix0, ix1 = int(round(sx0)), int(round(sx1))
            iy0, iy1 = int(round(sy0)), int(round(sy1))
            ix1 = max(ix1, ix0 + 1)
            iy1 = max(iy1, iy0 + 1)
            cell = material_bgra[iy0:iy1, ix0:ix1]
            ch, cw = cell.shape[:2]
            if ch <= 1 or cw <= 1:
                continue
            src = np.array(
                [
                    [0.0, 0.0],
                    [float(cw - 1), 0.0],
                    [float(cw - 1), float(ch - 1)],
                    [0.0, float(ch - 1)],
                ],
                dtype=np.float32,
            )
            p00 = grid[gy * 3 + gx]
            p10 = grid[gy * 3 + (gx + 1)]
            p11 = grid[(gy + 1) * 3 + (gx + 1)]
            p01 = grid[(gy + 1) * 3 + gx]
            dst = np.array(
                [
                    [float(p00.x()), float(p00.y())],
                    [float(p10.x()), float(p10.y())],
                    [float(p11.x()), float(p11.y())],
                    [float(p01.x()), float(p01.y())],
                ],
                dtype=np.float32,
            )
            M = cv2.getPerspectiveTransform(src, dst)
            warped_cell = cv2.warpPerspective(
                cell,
                M,
                (pw, ph),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            out = _alpha_compose(out, warped_cell)

    return out, (pw, ph)


def _rotate_points_any(
    pts: Tuple[QtCore.QPointF, ...],
    deg: float,
) -> Tuple[QtCore.QPointF, ...]:
    if not pts:
        return pts
    ang = float(deg) * math.pi / 180.0
    c, s = math.cos(ang), math.sin(ang)
    n = float(len(pts))
    cx = sum([p.x() for p in pts]) / n
    cy = sum([p.y() for p in pts]) / n
    out = []
    for p in pts:
        dx = p.x() - cx
        dy = p.y() - cy
        out.append(QtCore.QPointF(cx + dx * c - dy * s, cy + dx * s + dy * c))
    return tuple(out)  # type: ignore


def _sticker_points_to_grid3x3(pts8: Tuple[QtCore.QPointF, ...]) -> Tuple[QtCore.QPointF, ...]:
    """
    Map 8 control points to a 3x3 grid (row-major) by inferring center point.
    Order for pts8:
      0 tl, 1 tr, 2 br, 3 bl, 4 top-mid, 5 right-mid, 6 bottom-mid, 7 left-mid
    Returns 9 points: [p00,p10,p20, p01,p11,p21, p02,p12,p22]
    """
    if not pts8 or len(pts8) != 8:
        return ()
    p0, p1, p2, p3, mt, mr, mb, ml = pts8
    # Center: average of edge midpoints (more stable if corners are skewed)
    cx = (mt.x() + mr.x() + mb.x() + ml.x()) / 4.0
    cy = (mt.y() + mr.y() + mb.y() + ml.y()) / 4.0
    c = QtCore.QPointF(cx, cy)
    return (p0, mt, p1, ml, c, mr, p3, mb, p2)


def _sticker_recalc_locked_midpoints(st: _StickerOverlay) -> None:
    """Recalculate locked edge midpoints based on current corners."""
    try:
        pts = list(st.points)
        if len(pts) != 8:
            return
        p0, p1, p2, p3 = pts[0], pts[1], pts[2], pts[3]
        locks = st.mid_locks or (True, True, True, True)
        if locks[0]:
            pts[4] = QtCore.QPointF((p0.x() + p1.x()) / 2.0, (p0.y() + p1.y()) / 2.0)
        if locks[1]:
            pts[5] = QtCore.QPointF((p1.x() + p2.x()) / 2.0, (p1.y() + p2.y()) / 2.0)
        if locks[2]:
            pts[6] = QtCore.QPointF((p2.x() + p3.x()) / 2.0, (p2.y() + p3.y()) / 2.0)
        if locks[3]:
            pts[7] = QtCore.QPointF((p3.x() + p0.x()) / 2.0, (p3.y() + p0.y()) / 2.0)
        st.points = tuple(pts)  # type: ignore
    except Exception:
        return


def _sticker_corners(st: _StickerOverlay) -> Tuple[QtCore.QPointF, QtCore.QPointF, QtCore.QPointF, QtCore.QPointF]:
    pts = getattr(st, "points", None) or ()
    if len(pts) >= 4:
        return (pts[0], pts[1], pts[2], pts[3])
    return ()




class _StickerMethods:
    def set_sticker_enabled(self, enabled: bool) -> None:
        self._sticker_enabled = bool(enabled)
        if not self._sticker_enabled:
            self._sticker = None
            self._sticker_quick_apply_enabled = False
        self.update()

    def sticker_set_material(self, path: str, label: str) -> None:
        """Create/update sticker overlay with default quad centered on current image."""
        if not self._sticker_enabled:
            return
        pm = QtGui.QPixmap(path)
        if pm.isNull():
            # Fallback for Unicode paths on Windows: use OpenCV imdecode + fromImage.
            try:
                data = np.fromfile(path, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f"Failed to load material: {path}")
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                # BGRA -> RGBA for Qt
                rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                h, w = rgba.shape[:2]
                qimg = QtGui.QImage(
                    rgba.data,
                    int(w),
                    int(h),
                    int(rgba.strides[0]),
                    QtGui.QImage.Format.Format_RGBA8888,
                ).copy()
                pm = QtGui.QPixmap.fromImage(qimg)
            except Exception:
                pm = QtGui.QPixmap()
        if pm.isNull():
            raise ValueError(f"Failed to load material: {path}")
        # Default size: fit within 1/3 of image width
        iw = float(self.pixmap.width())
        ih = float(self.pixmap.height())
        if iw <= 0 or ih <= 0:
            raise ValueError("No base image loaded.")
        w = float(pm.width())
        h = float(pm.height())
        if w <= 0 or h <= 0:
            raise ValueError("Invalid material image.")
        scale = min((iw / 3.0) / w, (ih / 3.0) / h, 1.0)
        ww = w * scale
        hh = h * scale
        cx = iw / 2.0
        cy = ih / 2.0
        p0 = QtCore.QPointF(cx - ww / 2.0, cy - hh / 2.0)
        p1 = QtCore.QPointF(cx + ww / 2.0, cy - hh / 2.0)
        p2 = QtCore.QPointF(cx + ww / 2.0, cy + hh / 2.0)
        p3 = QtCore.QPointF(cx - ww / 2.0, cy + hh / 2.0)
        mt = QtCore.QPointF((p0.x() + p1.x()) / 2.0, (p0.y() + p1.y()) / 2.0)
        mr = QtCore.QPointF((p1.x() + p2.x()) / 2.0, (p1.y() + p2.y()) / 2.0)
        mb = QtCore.QPointF((p2.x() + p3.x()) / 2.0, (p2.y() + p3.y()) / 2.0)
        ml = QtCore.QPointF((p3.x() + p0.x()) / 2.0, (p3.y() + p0.y()) / 2.0)

        # Keep a BGRA copy for OpenCV warp preview (Unicode safe).
        rgba_bgra = None
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                rgba_bgra = img
        except Exception:
            rgba_bgra = None

        self._sticker = _StickerOverlay(
            path=str(path),
            label=str(label or ""),
            pixmap=pm,
            points=(p0, p1, p2, p3, mt, mr, mb, ml),
            mid_locks=(True, True, True, True),
            warp_mode=getattr(getattr(self, "_sticker", None), "warp_mode", "4") or "4",
            rgba=rgba_bgra,
            preview_pixmap=None,
            preview_dirty=True,
            preview_size=(0, 0),
            preview_timer_pending=False,
        )
        # Kick off warped preview so the overlay appears for interactive edit.
        self._sticker_mark_dirty()
        self.update()

    def sticker_set_warp_mode(self, mode: str) -> None:
        """Switch sticker warp mode: '4' (fast perspective) or '8' (mesh warp)."""
        st = getattr(self, "_sticker", None)
        if st is None:
            return
        m = str(mode or "4").strip()
        if m not in ("4", "8"):
            m = "4"
        st.warp_mode = m
        if m == "4":
            # In 4-point mode, keep midpoints locked and glued to edges.
            st.mid_locks = (True, True, True, True)
            _sticker_recalc_locked_midpoints(st)
        self._sticker_mark_dirty()
        self.update()

    def sticker_set_quick_apply_enabled(self, enabled: bool, *, output_dir: Optional[str]) -> None:
        self._sticker_quick_apply_enabled = bool(enabled)
        self._sticker_quick_output_dir = output_dir
        self.update()

    def sticker_set_quick_apply_settings(
        self,
        *,
        blend_mode: str,
        anno_mode: str,
        warp_mode: str,
        output_dir: Optional[str],
    ) -> None:
        self._sticker_quick_blend_mode = str(blend_mode or "alpha")
        self._sticker_quick_anno_mode = str(anno_mode or "hbb")
        self._sticker_quick_warp_mode = str(warp_mode or "4")
        self._sticker_quick_output_dir = output_dir
        self.update()

    def _sticker_translate_to(self, pos: QtCore.QPointF) -> None:
        st = getattr(self, "_sticker", None)
        if st is None:
            return
        pts = list(st.points)
        if len(pts) != 8:
            return
        corners = pts[0:4]
        cx = sum([p.x() for p in corners]) / 4.0
        cy = sum([p.y() for p in corners]) / 4.0
        dx = float(pos.x()) - cx
        dy = float(pos.y()) - cy
        pts2 = [QtCore.QPointF(p.x() + dx, p.y() + dy) for p in pts]
        iw = float(self.pixmap.width())
        ih = float(self.pixmap.height())
        pts3 = []
        for p in pts2:
            pts3.append(
                QtCore.QPointF(
                    min(max(p.x(), 0.0), max(iw - 1.0, 0.0)),
                    min(max(p.y(), 0.0), max(ih - 1.0, 0.0)),
                )
            )
        st.points = tuple(pts3)  # type: ignore
        _sticker_recalc_locked_midpoints(st)
        self._sticker_mark_dirty()

    def sticker_nudge(self, dx: float, dy: float) -> None:
        """Nudge sticker by (dx,dy) in image coordinates."""
        st = getattr(self, "_sticker", None)
        if st is None:
            return
        try:
            pts = list(st.points)
            if len(pts) != 8:
                return
            pts2 = [QtCore.QPointF(p.x() + float(dx), p.y() + float(dy)) for p in pts]
            iw = float(self.pixmap.width())
            ih = float(self.pixmap.height())
            pts3 = []
            for p in pts2:
                pts3.append(
                    QtCore.QPointF(
                        min(max(p.x(), 0.0), max(iw - 1.0, 0.0)),
                        min(max(p.y(), 0.0), max(ih - 1.0, 0.0)),
                    )
                )
            st.points = tuple(pts3)  # type: ignore
            _sticker_recalc_locked_midpoints(st)
            self._sticker_mark_dirty()
            self.update()
        except Exception:
            pass

    def sticker_cancel(self) -> None:
        if self._sticker is not None:
            self._sticker = None
            self.update()

    def sticker_rotate(self, deg_delta: float) -> None:
        """Rotate current sticker control points in-place (so render and interaction match)."""
        if self._sticker is None:
            return
        try:
            self._sticker.points = _rotate_points_any(self._sticker.points, float(deg_delta))  # type: ignore
            # Rotating keeps midpoints on their edges (locked behavior).
            _sticker_recalc_locked_midpoints(self._sticker)
            self._sticker_mark_dirty()
            self.update()
        except Exception:
            logger.exception("[Sticker] rotate failed")

    def _sticker_mark_dirty(self) -> None:
        try:
            if self._sticker is not None:
                self._sticker.preview_dirty = True
                # Allow one-time error logging again after state changes.
                self._sticker.debug_logged = False
                # Debounce preview rebuild to keep UI responsive while dragging.
                self._sticker.preview_timer_pending = True
                if not self._sticker_preview_timer.isActive():
                    self._sticker_preview_timer.start(30)
        except Exception:
            pass

    def _sticker_rebuild_preview_if_needed(self) -> None:
        st = getattr(self, "_sticker", None)
        if st is None:
            return
        try:
            if not st.preview_dirty:
                st.preview_timer_pending = False
                return
            if st.rgba is None:
                st.preview_pixmap = None
                st.preview_dirty = False
                st.preview_timer_pending = False
                self.update()
                return

            base_w = int(self.pixmap.width())
            base_h = int(self.pixmap.height())
            warped, (pw, ph) = _sticker_build_preview_bgra(
                material_bgra=st.rgba,
                base_w=base_w,
                base_h=base_h,
                preview_max=1200,
                points8=st.points,
                warp_mode=str(getattr(st, "warp_mode", "4") or "4"),
            )
            st.preview_size = (pw, ph)
            if warped is None:
                st.preview_pixmap = None
                st.preview_dirty = False
                st.preview_timer_pending = False
                self.update()
                return

            rgba = cv2.cvtColor(warped, cv2.COLOR_BGRA2RGBA)
            qimg = QtGui.QImage(
                rgba.data,
                int(pw),
                int(ph),
                int(rgba.strides[0]),
                QtGui.QImage.Format.Format_RGBA8888,
            ).copy()
            st.preview_pixmap = QtGui.QPixmap.fromImage(qimg)
            st.preview_dirty = False
            st.preview_timer_pending = False
            self.update()
        except Exception:
            try:
                if not getattr(st, "debug_logged", False):
                    st.debug_logged = True
                    logger.exception("[Sticker] preview rebuild failed")
            except Exception:
                pass
            try:
                st.preview_pixmap = None
                st.preview_dirty = False
                st.preview_timer_pending = False
            except Exception:
                pass
            self.update()

    def sticker_commit(
        self,
        *,
        blend_mode: str,
        anno_mode: str,
        output_dir: Optional[str],
        warp_mode: str = "4",
        keep_overlay: bool = False,
        reload_after: bool = False,
    ) -> None:
        """
        Apply sticker onto the base image file (overwrite) and generate annotation.
        Actual blending/annotation is implemented in sticker modules.
        """
        if self._sticker is None:
            return
        if not getattr(self.parent, "filename", None):
            return
        from anylabeling.views.labeling.advanced.sticker.blend import apply_sticker_overwrite
        from anylabeling.views.labeling.advanced.sticker.annotation import write_annotation_for_sticker

        base_image_path = str(self.parent.filename)
        # Choose fast 4-point or 8-point mesh warp
        mode = str(warp_mode or getattr(self._sticker, "warp_mode", "4") or "4")
        pts8 = self._sticker.points
        if mode == "8":
            pts_for_apply = pts8
        else:
            pts_for_apply = _sticker_corners(self._sticker)
        mat_path = self._sticker.path
        label = self._sticker.label
        blend_mode = str(blend_mode or "alpha")
        anno_mode = str(anno_mode or "hbb")

        mask = apply_sticker_overwrite(
            base_image_path=base_image_path,
            material_path=mat_path,
            quad=pts_for_apply,
            blend_mode=blend_mode,
        )
        write_annotation_for_sticker(
            base_image_path=base_image_path,
            output_dir=output_dir,
            label=label,
            anno_mode=anno_mode,
            mask=mask,
        )
        # Clear overlay after commit unless quick-apply needs to keep template.
        if not keep_overlay:
            self._sticker = None
        if reload_after:
            try:
                if getattr(self.parent, "filename", None):
                    self.parent.load_file(self.parent.filename)
            except Exception:
                pass
        self.update()

    def _sticker_handle_mouse_press(self, ev, *, pos: QtCore.QPointF) -> bool:
        """Handle sticker selection/move and quick-apply. Returns True if consumed."""
        if not (
            self._sticker_enabled
            and self._sticker is not None
            and ev.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            return False
        # Sticker interaction: select handle or move
        # Allow even in drawing mode, and prioritize sticker interaction
        # when user clicks on the sticker quad/handles.
        try:
            st = self._sticker
            pts = list(st.points)
            if len(pts) == 8:
                corners = pts[0:4]
                xs = [p.x() for p in corners]
                ys = [p.y() for p in corners]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                bw = max(0.0, xmax - xmin)
                bh = max(0.0, ymax - ymin)
                min_dim = max(1.0, min(bw, bh))
                r = max(3.5, min(10.0, 0.10 * min_dim))
                handle_pts = pts if getattr(st, "warp_mode", "4") == "8" else corners
                for i, p in enumerate(handle_pts):
                    if (pos.x() - p.x()) ** 2 + (pos.y() - p.y()) ** 2 <= r * r:
                        st.selected_handle = i
                        st.last_pos = pos
                        # If user grabs an edge midpoint, unlock that midpoint so it can be warped.
                        if getattr(st, "warp_mode", "4") == "8" and 4 <= i <= 7:
                            locks = list(st.mid_locks or (True, True, True, True))
                            locks[i - 4] = False
                            st.mid_locks = tuple(locks)  # type: ignore
                        ev.accept()
                        return True
                # Use bbox hit-test for robust "move whole sticker" selection.
                # In quick-apply mode, do NOT consume clicks as "move" unless on a handle;
                # otherwise user can never trigger quick apply.
                if not getattr(self, "_sticker_quick_apply_enabled", False):
                    pad = max(10.0, 0.25 * min_dim)
                    if (xmin - pad) <= pos.x() <= (xmax + pad) and (ymin - pad) <= pos.y() <= (ymax + pad):
                        st.selected_handle = 8
                        st.last_pos = pos
                        ev.accept()
                        return True
        except Exception:
            return False

        # Sticker quick-apply: click to place+apply (bake) at position.
        if (
            self._sticker_enabled
            and self._sticker is not None
            and self._sticker_quick_apply_enabled
        ):
            try:
                st = self._sticker
                self._sticker_translate_to(pos)
                self.sticker_commit(
                    blend_mode=self._sticker_quick_blend_mode,
                    anno_mode=self._sticker_quick_anno_mode,
                    output_dir=self._sticker_quick_output_dir,
                    warp_mode=self._sticker_quick_warp_mode,
                    keep_overlay=True,
                    reload_after=True,
                )
                try:
                    st.selected_handle = -1
                    st.last_pos = None
                except Exception:
                    pass
                ev.accept()
                return True
            except Exception:
                try:
                    logger.exception("[Sticker] quick apply failed")
                except Exception:
                    pass
        return False

    def _sticker_handle_mouse_release(self, ev) -> bool:
        """Clear sticker drag state on left-button release."""
        if not (
            self._sticker_enabled
            and self._sticker is not None
            and ev.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            return False
        st = self._sticker
        if st.selected_handle == -1 and st.last_pos is None:
            return False
        st.selected_handle = -1
        st.last_pos = None
        ev.accept()
        return True

    def _sticker_handle_mouse_move(self, ev, *, pos: QtCore.QPointF) -> bool:
        """Handle sticker dragging (handles / move). Returns True if consumed."""
        if not (
            self._sticker_enabled
            and self._sticker is not None
            and self._sticker.selected_handle != -1
            and bool(ev.buttons() & QtCore.Qt.MouseButton.LeftButton)
        ):
            return False
        try:
            st = self._sticker
            if st.last_pos is None:
                st.last_pos = pos
                return True
            dx = pos.x() - st.last_pos.x()
            dy = pos.y() - st.last_pos.y()
            pts = list(st.points)
            if len(pts) == 8:
                if 0 <= st.selected_handle <= 7:
                    # In 4-point mode, only corners are draggable.
                    if getattr(st, "warp_mode", "4") != "8" and st.selected_handle > 3:
                        st.selected_handle = 8
                    p0 = pts[st.selected_handle]
                    pts[st.selected_handle] = QtCore.QPointF(p0.x() + dx, p0.y() + dy)
                elif st.selected_handle == 8:
                    pts = [QtCore.QPointF(p.x() + dx, p.y() + dy) for p in pts]
                iw = float(self.pixmap.width())
                ih = float(self.pixmap.height())
                pts2 = []
                for p in pts:
                    pts2.append(
                        QtCore.QPointF(
                            min(max(p.x(), 0.0), max(iw - 1.0, 0.0)),
                            min(max(p.y(), 0.0), max(ih - 1.0, 0.0)),
                        )
                    )
                st.points = tuple(pts2)  # type: ignore
                # Keep locked edge midpoints glued to edges when corners move / translate.
                if st.selected_handle in (0, 1, 2, 3, 8):
                    _sticker_recalc_locked_midpoints(st)
                st.last_pos = pos
                self._sticker_mark_dirty()
                self.update()
                return True
        except Exception:
            return False
        return False

    def _sticker_handle_wheel(self, ev: QWheelEvent) -> bool:
        """Handle sticker scaling with mouse wheel. Returns True if consumed."""
        delta = ev.angleDelta()
        try:
            if self._sticker_enabled and self._sticker is not None and self.editing():
                factor = 1.05 if delta.y() > 0 else 0.95
                st = self._sticker
                pts = list(st.points)
                if len(pts) == 8:
                    cx = sum([pt.x() for pt in pts]) / 8.0
                    cy = sum([pt.y() for pt in pts]) / 8.0
                    pts2 = []
                    for pt in pts:
                        pts2.append(
                            QtCore.QPointF(
                                cx + (pt.x() - cx) * factor,
                                cy + (pt.y() - cy) * factor,
                            )
                        )
                    st.points = tuple(pts2)  # type: ignore
                    _sticker_recalc_locked_midpoints(st)
                    self._sticker_mark_dirty()
                    self.update()
                    ev.accept()
                    return True
        except Exception:
            return False
        return False




def install_sticker_support(canvas) -> None:
    """Attach sticker API onto Canvas; wrap move/wheel/key once."""
    if getattr(canvas, "_sticker_support_installed", False):
        return
    import types

    for name in dir(_StickerMethods):
        if name.startswith("__"):
            continue
        attr = getattr(_StickerMethods, name, None)
        if callable(attr) and (
            name.startswith("sticker")
            or name.startswith("_sticker")
            or name == "set_sticker_enabled"
        ):
            setattr(canvas, name, types.MethodType(attr, canvas))

    canvas._sticker_enabled = False
    canvas._sticker = None
    canvas._sticker_quick_apply_enabled = False
    canvas._sticker_quick_blend_mode = "alpha"
    canvas._sticker_quick_anno_mode = "hbb"
    canvas._sticker_quick_warp_mode = "4"
    canvas._sticker_quick_output_dir = None
    canvas._sticker_preview_timer = QtCore.QTimer(canvas)
    canvas._sticker_preview_timer.setSingleShot(True)
    if hasattr(canvas, "_sticker_rebuild_preview_if_needed"):
        canvas._sticker_preview_timer.timeout.connect(
            canvas._sticker_rebuild_preview_if_needed
        )

    _wrap_event(canvas, "mouseMoveEvent", _mouse_move_before)
    _wrap_event(canvas, "mouseReleaseEvent", _mouse_release_before)
    _wrap_event(canvas, "wheelEvent", _wheel_before)
    _wrap_event(canvas, "keyPressEvent", _key_before)

    canvas._sticker_support_installed = True



def _wrap_event(canvas, name: str, wrapper_factory):
    original = getattr(canvas, name)
    # bound method -> unwrap to function via __func__ if needed
    def bound_original(*args, **kwargs):
        return original(*args, **kwargs)

    wrapped = wrapper_factory(canvas, bound_original)
    setattr(canvas, name, wrapped)


def _paint_after(canvas, original):
    return original




def _mouse_press_before(canvas, original):
    def mousePressEvent(ev):
        try:
            if getattr(canvas, "_sticker_enabled", False) and getattr(canvas, "_sticker", None):
                pos = canvas.transform_pos(ev.position()) if hasattr(canvas, "transform_pos") else ev.position()
                handler = getattr(canvas, "_sticker_handle_mouse_press", None)
                if callable(handler) and handler(ev, pos=pos):
                    return
        except Exception:
            pass
        return original(ev)
    return mousePressEvent


def _mouse_move_before(canvas, original):
    def mouseMoveEvent(ev):
        try:
            if getattr(canvas, "_sticker_enabled", False) and getattr(canvas, "_sticker", None):
                pos = canvas.transform_pos(ev.position()) if hasattr(canvas, "transform_pos") else ev.position()
                handler = getattr(canvas, "_sticker_handle_mouse_move", None)
                if callable(handler) and handler(ev, pos=pos):
                    return
        except Exception:
            pass
        return original(ev)
    return mouseMoveEvent


def _mouse_release_before(canvas, original):
    def mouseReleaseEvent(ev):
        try:
            handler = getattr(canvas, "_sticker_handle_mouse_release", None)
            if callable(handler) and handler(ev):
                return
        except Exception:
            pass
        return original(ev)
    return mouseReleaseEvent


def _wheel_before(canvas, original):
    def wheelEvent(ev):
        try:
            handler = getattr(canvas, "_sticker_handle_wheel", None)
            if callable(handler) and handler(ev):
                return
        except Exception:
            pass
        return original(ev)
    return wheelEvent


def _key_before(canvas, original):
    def keyPressEvent(ev):
        try:
            handler = getattr(canvas, "_sticker_handle_key", None)
            if callable(handler) and handler(ev):
                return
        except Exception:
            pass
        return original(ev)
    return keyPressEvent
