from __future__ import annotations

import os.path as osp
from typing import Iterable, List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from collections import OrderedDict


class AnnotationInspectorDock(QtWidgets.QDockWidget):
    inspectorItemClicked = QtCore.pyqtSignal(str, str, int)
    dockAboutToClose = QtCore.pyqtSignal()
    runFalsePositiveClipRequested = QtCore.pyqtSignal(float)
    fnItemClicked = QtCore.pyqtSignal(str, str, object)  # (image_path, label, bbox01)

    _ROLE_FP_CLICK = Qt.ItemDataRole.UserRole
    _ROLE_FN_CLICK = Qt.ItemDataRole.UserRole + 1

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("BatchAnnotationCheckDock")
        self.setWindowTitle(self.tr("Batch Annotation Check"))
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.setMinimumWidth(280)

        root = QtWidgets.QWidget(self)
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        self._hint = QtWidgets.QLabel(
            self.tr("Click a crop to open its image and highlight the annotation.")
        )
        self._hint.setStyleSheet("color: #666;")
        self._hint.setWordWrap(True)
        root_layout.addWidget(self._hint)

        tools = QtWidgets.QHBoxLayout()
        self._fp_thresh = QtWidgets.QDoubleSpinBox()
        self._fp_thresh.setRange(-1.0, 1.0)
        self._fp_thresh.setSingleStep(0.01)
        self._fp_thresh.setValue(0.23)
        self._fp_thresh.setToolTip(self.tr("CLIP cosine similarity threshold"))
        tools.addWidget(self._fp_thresh)

        self._run_fp_clip_btn = QtWidgets.QPushButton(self.tr("Run FP check (CLIP)"))
        self._run_fp_clip_btn.clicked.connect(
            lambda: self.runFalsePositiveClipRequested.emit(
                float(self._fp_thresh.value())
            )
        )
        tools.addWidget(self._run_fp_clip_btn)
        tools.addStretch(1)
        root_layout.addLayout(tools)

        self._fp_section = QtWidgets.QLabel(self.tr("All crops (dataset)"))
        self._fp_section.setStyleSheet("font-weight: 600;")

        self._fp_list = QtWidgets.QListWidget()
        self._fp_list.setViewMode(QtWidgets.QListWidget.ViewMode.IconMode)
        self._fp_list.setFlow(QtWidgets.QListWidget.Flow.LeftToRight)
        self._fp_list.setWrapping(True)
        self._fp_list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self._fp_list.setMovement(QtWidgets.QListWidget.Movement.Static)
        self._fp_list.setSpacing(8)
        self._fp_list.setIconSize(QtCore.QSize(96, 96))
        self._fp_list.setUniformItemSizes(True)
        self._fp_list.setWordWrap(True)
        self._fp_list.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self._fp_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._fp_list.setMinimumHeight(120)
        self._fp_list.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._fp_list.itemClicked.connect(self._on_fp_item_clicked)

        fp_block = QtWidgets.QWidget()
        fp_block_layout = QtWidgets.QVBoxLayout(fp_block)
        fp_block_layout.setContentsMargins(0, 0, 0, 0)
        fp_block_layout.setSpacing(6)
        fp_block_layout.addWidget(self._fp_section)
        fp_block_layout.addWidget(self._fp_list, 1)

        self._fn_hint = QtWidgets.QLabel(
            self.tr("FN candidates (yellow): click to show overlay on the image.")
        )
        self._fn_hint.setStyleSheet("color: #666;")
        self._fn_hint.setWordWrap(True)

        self._fn_list = QtWidgets.QListWidget()
        self._fn_list.setViewMode(QtWidgets.QListWidget.ViewMode.IconMode)
        self._fn_list.setFlow(QtWidgets.QListWidget.Flow.LeftToRight)
        self._fn_list.setWrapping(True)
        self._fn_list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self._fn_list.setMovement(QtWidgets.QListWidget.Movement.Static)
        self._fn_list.setSpacing(8)
        self._fn_list.setIconSize(QtCore.QSize(96, 96))
        self._fn_list.setUniformItemSizes(True)
        self._fn_list.setWordWrap(True)
        self._fn_list.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self._fn_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._fn_list.setMinimumHeight(80)
        self._fn_list.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._fn_list.itemClicked.connect(self._on_fn_item_clicked)

        fn_block = QtWidgets.QWidget()
        fn_block_layout = QtWidgets.QVBoxLayout(fn_block)
        fn_block_layout.setContentsMargins(0, 0, 0, 0)
        fn_block_layout.setSpacing(6)
        fn_block_layout.addWidget(self._fn_hint)
        fn_block_layout.addWidget(self._fn_list, 1)

        self._lists_splitter = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        self._lists_splitter.setChildrenCollapsible(False)
        self._lists_splitter.setHandleWidth(6)
        self._lists_splitter.addWidget(fp_block)
        self._lists_splitter.addWidget(fn_block)
        self._lists_splitter.setStretchFactor(0, 1)
        self._lists_splitter.setStretchFactor(1, 1)
        self._lists_splitter.setSizes([340, 260])

        root_layout.addWidget(self._lists_splitter, 1)

        self.setWidget(root)

        # LRU cache for scaled QIcons to avoid disk decode on large lists.
        self._icon_cache: "OrderedDict[str, QtGui.QIcon]" = OrderedDict()
        self._icon_cache_max = 600
        self._icon_job_token = 0
        self._fp_icon_jobs: List[tuple[int, QtWidgets.QListWidgetItem, str]] = []
        self._fn_icon_jobs: List[tuple[int, QtWidgets.QListWidgetItem, str]] = []
        self._icon_loader_timer: Optional[QtCore.QTimer] = None
        self._fp_highlight_item: Optional[QtWidgets.QListWidgetItem] = None
        self._fp_highlight_saved_bg: Optional[QtGui.QBrush] = None

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

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.dockAboutToClose.emit()
        super().closeEvent(event)

    def _stop_icon_loader(self) -> None:
        if self._icon_loader_timer is not None:
            try:
                self._icon_loader_timer.stop()
            except Exception:
                pass
            self._icon_loader_timer = None

    def clear_items(self) -> None:
        self.clear_fp_highlight()
        self._icon_job_token += 1
        self._fp_icon_jobs = []
        self._fn_icon_jobs = []
        self._stop_icon_loader()

        self._fp_list.clear()
        self._fn_list.clear()

    def _drain_icon_jobs(self) -> None:
        token = self._icon_job_token
        batch = 24
        drained = 0
        while drained < batch and self._fp_icon_jobs:
            t, item, path = self._fp_icon_jobs.pop(0)
            if t != token:
                drained += 1
                continue
            try:
                if item.listWidget() is not self._fp_list:
                    drained += 1
                    continue
            except Exception:
                drained += 1
                continue
            ic = self._icon_for_path(path)
            if ic is not None:
                item.setIcon(ic)
            drained += 1

        while drained < batch and self._fn_icon_jobs:
            t, item, path = self._fn_icon_jobs.pop(0)
            if t != token:
                drained += 1
                continue
            try:
                if item.listWidget() is not self._fn_list:
                    drained += 1
                    continue
            except Exception:
                drained += 1
                continue
            ic = self._icon_for_path(path)
            if ic is not None:
                item.setIcon(ic)
            drained += 1

        if not self._fp_icon_jobs and not self._fn_icon_jobs:
            self._stop_icon_loader()

    def _ensure_icon_timer(self) -> None:
        if self._icon_loader_timer is None:
            self._icon_loader_timer = QtCore.QTimer(self)
            self._icon_loader_timer.setInterval(0)
            self._icon_loader_timer.timeout.connect(self._drain_icon_jobs)
            self._icon_loader_timer.start()

    def _on_fp_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        data = item.data(self._ROLE_FP_CLICK)
        if not isinstance(data, tuple) or len(data) != 3:
            return
        image_path, label_json_path, shape_index = data
        self.focus_fp_crop(str(image_path), int(shape_index))
        self.inspectorItemClicked.emit(str(image_path), str(label_json_path), int(shape_index))

    def _on_fn_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        data = item.data(self._ROLE_FN_CLICK)
        if not isinstance(data, tuple) or len(data) != 3:
            return
        image_path, label, bbox01 = data[0], data[1], data[2]
        self.fnItemClicked.emit(str(image_path), str(label), bbox01)

    def _schedule_restore_list_scroll(
        self, list_widget: QtWidgets.QListWidget, y: int
    ) -> None:
        """Keep viewport after refresh (e.g. delete shape/crop rebuild)."""

        def _apply() -> None:
            sb = list_widget.verticalScrollBar()
            sb.setValue(min(max(0, int(y)), sb.maximum()))

        QtCore.QTimer.singleShot(0, _apply)

    def _paths_same_file(self, a: str, b: str) -> bool:
        try:
            return osp.normcase(osp.normpath(osp.abspath(a))) == osp.normcase(
                osp.normpath(osp.abspath(b))
            )
        except Exception:
            return str(a) == str(b)

    def clear_fp_highlight(self) -> None:
        """Remove selection highlight from the crop grid (restore prior row background)."""
        if self._fp_highlight_item is not None:
            try:
                if self._fp_highlight_item.listWidget() == self._fp_list:
                    if self._fp_highlight_saved_bg is not None:
                        self._fp_highlight_item.setBackground(
                            self._fp_highlight_saved_bg
                        )
                    else:
                        self._fp_highlight_item.setBackground(QtGui.QBrush())
            except Exception:
                pass
        self._fp_highlight_item = None
        self._fp_highlight_saved_bg = None

    def focus_fp_crop(self, image_path: str, shape_index: int) -> None:
        """Scroll to the matching thumbnail and highlight it."""

        if not image_path or int(shape_index) < 0:
            return

        def _do() -> None:
            target_si = int(shape_index)
            self.clear_fp_highlight()
            for row in range(self._fp_list.count()):
                lw_item = self._fp_list.item(row)
                if lw_item is None:
                    continue
                data = lw_item.data(self._ROLE_FP_CLICK)
                if not isinstance(data, tuple) or len(data) != 3:
                    continue
                ip, _lj, si = data[0], data[1], data[2]
                if int(si) != target_si:
                    continue
                if not self._paths_same_file(str(ip), str(image_path)):
                    continue
                self._fp_highlight_saved_bg = lw_item.background()
                self._fp_highlight_item = lw_item
                lw_item.setBackground(
                    QtGui.QBrush(QtGui.QColor(70, 150, 255, 140))
                )
                self._fp_list.setCurrentItem(lw_item)
                self._fp_list.scrollToItem(
                    lw_item,
                    QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter,
                )
                return

        QtCore.QTimer.singleShot(0, _do)

    def set_items(self, items: Iterable[object]) -> None:
        """
        items should provide fields:
          - image_path, label_json_path, shape_index, crop_path, label (optional)
        """
        self.clear_fp_highlight()
        fp_scroll_y = self._fp_list.verticalScrollBar().value()
        self._fp_list.clear()
        self._fp_icon_jobs = []
        self._stop_icon_loader()
        token = self._icon_job_token

        for it in items:
            try:
                image_path = getattr(it, "image_path")
                label_json_path = getattr(it, "label_json_path")
                shape_index = int(getattr(it, "shape_index"))
                crop_path = getattr(it, "crop_path")
                label = str(getattr(it, "label", ""))
            except Exception:
                continue

            base = osp.basename(str(image_path))
            short = base if len(base) <= 18 else base[:15] + "…"
            text = f"{label}\n{short}" if label else short

            lw_item = QtWidgets.QListWidgetItem(text)
            lw_item.setTextAlignment(
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            )
            lw_item.setSizeHint(QtCore.QSize(110, 124))
            lw_item.setData(
                self._ROLE_FP_CLICK,
                (str(image_path), str(label_json_path), int(shape_index)),
            )
            lw_item.setToolTip(f"{image_path}\n{crop_path}")

            try:
                fp_is_match = getattr(it, "fp_is_match", None)
                fp_err = getattr(it, "fp_error", None)
                if fp_err:
                    lw_item.setBackground(
                        QtGui.QBrush(QtGui.QColor(255, 165, 0, 80))
                    )
                elif fp_is_match is False:
                    lw_item.setBackground(
                        QtGui.QBrush(QtGui.QColor(255, 0, 0, 80))
                    )
            except Exception:
                pass

            if crop_path:
                self._fp_icon_jobs.append((token, lw_item, str(crop_path)))
            self._fp_list.addItem(lw_item)

        self._schedule_restore_list_scroll(self._fp_list, fp_scroll_y)

        if self._fp_icon_jobs:
            self._ensure_icon_timer()

    def set_fn_items(self, items: Iterable[object]) -> None:
        fn_scroll_y = self._fn_list.verticalScrollBar().value()
        self._fn_list.clear()
        self._fn_icon_jobs = []
        # FP queue may still run; only stop timer if both empty after drain
        token = self._icon_job_token

        for it in items:
            try:
                image_path = getattr(it, "fn_source_image_path")
                crop_path = getattr(it, "crop_path")
                label = str(getattr(it, "fn_label", ""))
                bbox01 = getattr(it, "fn_bbox01", None)
            except Exception:
                continue

            base = osp.basename(str(image_path))
            short = base if len(base) <= 18 else base[:15] + "…"
            text = f"{label}\n{short}" if label else short

            lw_item = QtWidgets.QListWidgetItem(text)
            lw_item.setTextAlignment(
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            )
            lw_item.setSizeHint(QtCore.QSize(110, 124))
            lw_item.setData(
                self._ROLE_FN_CLICK,
                (str(image_path), label, bbox01),
            )
            lw_item.setBackground(QtGui.QBrush(QtGui.QColor(255, 255, 0, 90)))
            lw_item.setToolTip(str(image_path))

            if crop_path:
                self._fn_icon_jobs.append((token, lw_item, str(crop_path)))
            self._fn_list.addItem(lw_item)

        self._schedule_restore_list_scroll(self._fn_list, fn_scroll_y)

        if self._fn_icon_jobs or self._fp_icon_jobs:
            self._ensure_icon_timer()
