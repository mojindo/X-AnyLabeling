from __future__ import annotations

import json
import os
import os.path as osp
import shutil
from dataclasses import dataclass
from typing import Optional, Tuple

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_ok_btn_style,
    get_progress_dialog_style,
    get_spinbox_style,
)


@dataclass(frozen=True)
class _Constraints:
    min_w: Optional[int]
    max_w: Optional[int]
    min_h: Optional[int]
    max_h: Optional[int]

    def any_set(self) -> bool:
        return any(v is not None for v in (self.min_w, self.max_w, self.min_h, self.max_h))


def _shape_bbox_wh(shape: dict) -> Optional[Tuple[float, float]]:
    pts = shape.get("points")
    if not isinstance(pts, list) or not pts:
        return None
    xs = []
    ys = []
    for p in pts:
        if not (isinstance(p, (list, tuple)) and len(p) >= 2):
            continue
        try:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
        except Exception:
            continue
    if not xs or not ys:
        return None
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if w < 0 or h < 0:
        return None
    return (w, h)


def _match_wh(w: float, h: float, c: _Constraints) -> bool:
    if c.min_w is not None and w < c.min_w:
        return False
    if c.max_w is not None and w > c.max_w:
        return False
    if c.min_h is not None and h < c.min_h:
        return False
    if c.max_h is not None and h > c.max_h:
        return False
    return True


def _dedupe_path(path: str) -> str:
    if not osp.exists(path):
        return path
    base, ext = osp.splitext(path)
    k = 1
    while True:
        cand = f"{base}_{k}{ext}"
        if not osp.exists(cand):
            return cand
        k += 1


class _AnnoFilterWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(bool, str, int)

    def __init__(
        self,
        *,
        image_paths: list[str],
        constraints: _Constraints,
        invert: bool,
        target_dir: str,
    ) -> None:
        super().__init__()
        self._image_paths = list(image_paths or [])
        self._c = constraints
        self._invert = bool(invert)
        self._target_dir = str(target_dir or "")
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    def _should_cancel(self) -> bool:
        return bool(self._cancel)

    def run(self) -> None:
        moved = 0
        try:
            os.makedirs(self._target_dir, exist_ok=True)
            total = len(self._image_paths)
            for i, img_path in enumerate(self._image_paths):
                if self._should_cancel():
                    break
                self.progress.emit(i + 1, total, img_path)
                if not isinstance(img_path, str) or not img_path:
                    continue

                img_dir = osp.dirname(img_path)
                stem = osp.splitext(osp.basename(img_path))[0]
                json_path = osp.join(img_dir, stem + ".json")
                if not osp.isfile(json_path):
                    continue

                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                shapes = data.get("shapes", [])
                if not isinstance(shapes, list):
                    shapes = []

                hit = False
                for shp in shapes:
                    if not isinstance(shp, dict):
                        continue
                    wh = _shape_bbox_wh(shp)
                    if wh is None:
                        continue
                    if _match_wh(wh[0], wh[1], self._c):
                        hit = True
                        break

                selected = (not hit) if self._invert else hit
                if not selected:
                    continue

                out_img = _dedupe_path(osp.join(self._target_dir, osp.basename(img_path)))
                out_json = osp.splitext(out_img)[0] + ".json"
                try:
                    shutil.move(img_path, out_img)
                except Exception:
                    continue
                try:
                    shutil.move(json_path, out_json)
                except Exception:
                    try:
                        shutil.move(out_img, img_path)
                    except Exception:
                        pass
                    continue

                moved += 1
            self.finished.emit(True, "", moved)
        except Exception as e:
            self.finished.emit(False, str(e), moved)


class AnnotationFilterDialog(QDialog):
    def __init__(self, parent, *, image_paths: list[str]):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Annotation Filter"))
        self.setMinimumWidth(560)
        self.setStyleSheet(get_dialog_style())

        self._image_paths = list(image_paths or [])
        self._worker: Optional[_AnnoFilterWorker] = None
        self._progress: Optional[QtWidgets.QProgressDialog] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        intro = QLabel(
            self.tr(
                "Move images (and their LabelMe JSON) if any shape's bbox width/height "
                "matches the constraints."
            )
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        gb = QGroupBox(self.tr("Constraints (pixels)"))
        form = QFormLayout(gb)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.sp_min_w = QSpinBox()
        self.sp_min_w.setRange(0, 100000)
        self.sp_min_w.setValue(0)
        self.sp_min_w.setStyleSheet(get_spinbox_style())

        self.sp_max_w = QSpinBox()
        self.sp_max_w.setRange(0, 100000)
        self.sp_max_w.setValue(0)
        self.sp_max_w.setStyleSheet(get_spinbox_style())

        self.sp_min_h = QSpinBox()
        self.sp_min_h.setRange(0, 100000)
        self.sp_min_h.setValue(0)
        self.sp_min_h.setStyleSheet(get_spinbox_style())

        self.sp_max_h = QSpinBox()
        self.sp_max_h.setRange(0, 100000)
        self.sp_max_h.setValue(0)
        self.sp_max_h.setStyleSheet(get_spinbox_style())

        form.addRow(self.tr("Min width (>=):"), self.sp_min_w)
        form.addRow(self.tr("Max width (<=):"), self.sp_max_w)
        form.addRow(self.tr("Min height (>=):"), self.sp_min_h)
        form.addRow(self.tr("Max height (<=):"), self.sp_max_h)
        root.addWidget(gb)

        self.cb_invert = QCheckBox(self.tr("Invert selection (move NON-matching images)"))
        root.addWidget(self.cb_invert)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel(self.tr("Target folder:")))
        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText(self.tr("Select a folder to move files into"))
        btn_browse = QPushButton(self.tr("Browse"))
        btn_browse.setStyleSheet(get_cancel_btn_style())
        btn_browse.clicked.connect(self._browse_out)
        out_row.addWidget(self.ed_out, 1)
        out_row.addWidget(btn_browse)
        root.addLayout(out_row)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_close = QPushButton(self.tr("Close"))
        self.btn_close.setStyleSheet(get_cancel_btn_style())
        self.btn_close.clicked.connect(self.reject)
        self.btn_run = QPushButton(self.tr("Run"))
        self.btn_run.setStyleSheet(get_ok_btn_style())
        self.btn_run.clicked.connect(self._run)
        btns.addWidget(self.btn_close)
        btns.addWidget(self.btn_run)
        root.addLayout(btns)

    def _browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select target folder"),
            self.ed_out.text() or "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if path:
            self.ed_out.setText(path)

    def _constraints(self) -> _Constraints:
        min_w = int(self.sp_min_w.value()) or 0
        max_w = int(self.sp_max_w.value()) or 0
        min_h = int(self.sp_min_h.value()) or 0
        max_h = int(self.sp_max_h.value()) or 0
        return _Constraints(
            min_w=min_w if min_w > 0 else None,
            max_w=max_w if max_w > 0 else None,
            min_h=min_h if min_h > 0 else None,
            max_h=max_h if max_h > 0 else None,
        )

    def _run(self) -> None:
        if not self._image_paths:
            QMessageBox.information(self, self.tr("Annotation Filter"), self.tr("No images loaded."))
            return
        out_dir = (self.ed_out.text() or "").strip()
        if not out_dir:
            QMessageBox.warning(self, self.tr("Annotation Filter"), self.tr("Please select a target folder."))
            return
        c = self._constraints()
        if not c.any_set():
            QMessageBox.warning(
                self,
                self.tr("Annotation Filter"),
                self.tr("Please set at least one constraint (min/max width/height)."),
            )
            return

        self.btn_run.setEnabled(False)
        self.btn_close.setEnabled(False)

        self._progress = QtWidgets.QProgressDialog(self)
        self._progress.setWindowTitle(self.tr("Annotation Filter"))
        self._progress.setLabelText(self.tr("Preparing..."))
        self._progress.setCancelButtonText(self.tr("Cancel"))
        self._progress.setStyleSheet(get_progress_dialog_style())
        self._progress.setRange(0, 0)
        self._progress.setMinimumDuration(0)
        self._progress.canceled.connect(self._cancel_worker)
        self._progress.show()

        self._worker = _AnnoFilterWorker(
            image_paths=self._image_paths,
            constraints=c,
            invert=bool(self.cb_invert.isChecked()),
            target_dir=out_dir,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel_worker(self) -> None:
        if self._worker is not None:
            try:
                self._worker.request_cancel()
            except Exception:
                pass

    def _on_progress(self, i: int, n: int, path: str) -> None:
        if self._progress is None:
            return
        self._progress.setRange(0, max(1, int(n)))
        self._progress.setValue(int(i))
        self._progress.setLabelText(self.tr("Processing: %s") % osp.basename(path or ""))

    def _on_finished(self, ok: bool, err: str, moved: int) -> None:
        if self._progress is not None:
            try:
                self._progress.close()
            except Exception:
                pass
            self._progress = None
        self.btn_run.setEnabled(True)
        self.btn_close.setEnabled(True)
        self._worker = None

        if not ok:
            QMessageBox.warning(self, self.tr("Annotation Filter"), self.tr("Failed: %s") % str(err))
            return
        QMessageBox.information(
            self,
            self.tr("Annotation Filter"),
            self.tr("Done. Moved %d image(s).") % int(moved),
        )

