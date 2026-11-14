"""Batch annotation check handlers — bound onto LabelingWidget by advanced.bootstrap."""
from __future__ import annotations

import os.path as osp
from typing import Any, Dict, List, Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from anylabeling.views.labeling.advanced.batch_annotation_check.cache import (
    InspectorCache,
    new_session_cache_dir,
)
from anylabeling.views.labeling.advanced.batch_annotation_check.dock import (
    AnnotationInspectorDock,
)
from anylabeling.views.labeling.logger import logger


class InspectorBuildWorker(QtCore.QObject):
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


class InspectorFpWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int, str)
    status = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(bool, str, object)  # ok, err, results

    def __init__(
        self,
        *,
        items: list,
        threshold: float,
        label_prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        self._items = list(items or [])
        self._threshold = float(threshold)
        self._label_prompts = dict(label_prompts or {})
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            from anylabeling.views.labeling.advanced.batch_annotation_check.checks.clip_fp import (
                run_fp_check_on_crops,
            )

            results = run_fp_check_on_crops(
                self._items,
                threshold=self._threshold,
                label_prompts=self._label_prompts,
                on_progress=lambda i, n, p: self.progress.emit(i, n, p),
                should_cancel=lambda: bool(self._cancel),
                on_message=lambda m: self.status.emit(str(m)),
            )
            self.finished.emit(True, "", results)
        except Exception as e:  # noqa: BLE001
            self.finished.emit(False, str(e), [])


def _apply_suspect_fp_marks(results: List[Dict[str, Any]]) -> int:
    """Write/clear ``suspect_fp`` on label JSON shapes. Returns updated file count."""
    if not results:
        return 0
    from anylabeling.views.labeling.advanced.batch_annotation_check.cropper import (
        make_shape_key,
    )
    from anylabeling.views.labeling.advanced.compare.annotation_compare import (
        COMPARE_STATUS,
        STATUS_SUSPECT_FP,
    )
    from anylabeling.views.labeling.advanced.compare.annotation_compare_io import (
        read_label_json,
        write_label_json,
    )

    by_json: Dict[str, List[Dict[str, Any]]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        if row.get("error"):
            continue
        if row.get("passed") is None:
            continue
        path = str(row.get("label_json_path") or "")
        if not path:
            continue
        by_json.setdefault(path, []).append(row)

    updated_files = 0
    for path, rows in by_json.items():
        if not osp.isfile(path):
            continue
        data = read_label_json(path)
        shapes = data.get("shapes")
        if not isinstance(shapes, list):
            continue
        changed = False
        for row in rows:
            key = str(row.get("shape_key") or "").strip()
            try:
                shape_index = int(row.get("shape_index", -1))
            except Exception:
                shape_index = -1
            is_suspect = row.get("passed") is False
            target = None
            if key:
                for shape in shapes:
                    if not isinstance(shape, dict):
                        continue
                    sk = make_shape_key(
                        label=shape.get("label", ""),
                        shape_type=shape.get("shape_type", ""),
                        points=shape.get("points"),
                    )
                    if sk == key:
                        target = shape
                        break
            if target is None and 0 <= shape_index < len(shapes):
                cand = shapes[shape_index]
                if isinstance(cand, dict):
                    target = cand
            if target is None:
                continue
            attrs = target.get("attributes")
            if not isinstance(attrs, dict):
                attrs = {}
                target["attributes"] = attrs
            cur = attrs.get(COMPARE_STATUS)
            if is_suspect:
                if cur != STATUS_SUSPECT_FP:
                    attrs[COMPARE_STATUS] = STATUS_SUSPECT_FP
                    changed = True
            elif cur == STATUS_SUSPECT_FP:
                del attrs[COMPARE_STATUS]
                changed = True
        if changed:
            write_label_json(path, data)
            updated_files += 1
    return updated_files


class _BatchCheckWidgetMethods:
    """Mixin-style methods; install() attaches them to a LabelingWidget instance."""

    def toggle_annotation_inspector(self, checked: bool) -> None:
        if checked:
            self._enable_annotation_inspector()
        else:
            self._disable_annotation_inspector()


    def _enable_annotation_inspector(self) -> None:
        if self._inspector_enabled:
            return
        if not self.image_list:
            QMessageBox.information(
                self,
                self.tr("No images loaded"),
                self.tr(
                    "Please load an image folder before using Batch Annotation Check."
                ),
            )
            self.actions.batch_annotation_check.setChecked(False)
            return
        labels = self._prompt_select_inspector_labels()
        if not labels:
            self.actions.batch_annotation_check.setChecked(False)
            return

        # Choose: import existing cache dir or build a new session cache.
        choice = QMessageBox(self)
        choice.setWindowTitle(self.tr("Batch Annotation Check"))
        choice.setIcon(QMessageBox.Icon.Question)
        choice.setText(
            self.tr(
                "Use an existing crop cache, or build a new cache for the selected labels?"
            )
        )
        btn_import = choice.addButton(
            self.tr("Import cache…"), QMessageBox.ButtonRole.ActionRole
        )
        btn_build = choice.addButton(
            self.tr("Build new"), QMessageBox.ButtonRole.AcceptRole
        )
        btn_cancel = choice.addButton(
            self.tr("Cancel"), QMessageBox.ButtonRole.RejectRole
        )
        choice.setDefaultButton(btn_build)
        choice.exec()

        clicked = choice.clickedButton()
        if clicked is btn_cancel:
            self.actions.batch_annotation_check.setChecked(False)
            return

        if clicked is btn_import:
            cache_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("Select Batch Annotation Check cache directory"),
                "",
                QtWidgets.QFileDialog.Option.ShowDirsOnly
                | QtWidgets.QFileDialog.Option.DontResolveSymlinks,
            )
            if not cache_dir:
                self.actions.batch_annotation_check.setChecked(False)
                return
            cache = InspectorCache(cache_dir)
            if not cache.is_valid_cache_dir():
                QMessageBox.warning(
                    self,
                    self.tr("Batch Annotation Check"),
                    self.tr(
                        "Invalid cache directory. Expect index.json and crops/ under the folder."
                    ),
                )
                self.actions.batch_annotation_check.setChecked(False)
                return
            idx = cache.load_index()
            lf = idx.get("labels_filter") if isinstance(idx, dict) else None
            self._inspector_cache = cache
            self._inspector_labels_filter = (
                [x for x in lf if isinstance(x, str) and x] if isinstance(lf, list) else list(labels)
            )
            self._inspector_enabled = True
            dock = self._ensure_inspector_dock()
            dock.show()
            self._refresh_inspector_for_current_image()
            return

        # Build new cache
        cache_dir = new_session_cache_dir()
        self._inspector_cache = InspectorCache(cache_dir)
        self._inspector_labels_filter = list(labels)
        self._start_inspector_preprocess()


    def _disable_annotation_inspector(self) -> None:
        self._inspector_enabled = False
        if self._inspector_dock is not None:
            host = self._find_dock_host()
            if host is not None:
                try:
                    host.removeDockWidget(self._inspector_dock)
                except Exception:
                    pass
            try:
                self._inspector_dock.deleteLater()
            except Exception:
                pass
            self._inspector_dock = None
        if self._inspector_progress is not None:
            try:
                self._inspector_progress.close()
            except Exception:
                pass
            self._inspector_progress = None
        if self._inspector_thread is not None:
            try:
                self._inspector_thread.quit()
                self._inspector_thread.wait(500)
            except Exception:
                pass
            self._inspector_thread = None
        self._inspector_worker = None
        if self._inspector_cache is not None:
            # keep cache for close/save prompt; disable only hides UI
            pass


    def _prompt_select_inspector_labels(self) -> list:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(self.tr("Select labels"))
        dlg.setModal(True)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(10, 10, 10, 10)

        hint = QtWidgets.QLabel(
            self.tr("Select one or more labels to inspect.")
        )
        layout.addWidget(hint)

        lw = QtWidgets.QListWidget()
        lw.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        labels = []
        try:
            for i in range(self.unique_label_list.count()):
                it = self.unique_label_list.item(i)
                if it is None:
                    continue
                # UniqueLabelQListWidget stores label in UserRole.
                v = it.data(Qt.ItemDataRole.UserRole)
                if isinstance(v, str) and v:
                    labels.append(v)
        except Exception:
            labels = []
        for t in labels:
            lw.addItem(QtWidgets.QListWidgetItem(t))
        layout.addWidget(lw, 1)

        btn_row = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton(self.tr("OK"))
        cancel = QtWidgets.QPushButton(self.tr("Cancel"))
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        btn_row.addStretch(1)
        btn_row.addWidget(ok)
        btn_row.addWidget(cancel)
        layout.addLayout(btn_row)

        dlg.setMinimumSize(420, 360)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return []
        selected = [it.text() for it in lw.selectedItems()]
        selected = [s for s in selected if s]
        if not selected:
            QMessageBox.information(
                self,
                self.tr("Batch Annotation Check"),
                self.tr("No labels selected."),
            )
            return []
        return selected


    def _ensure_inspector_dock(self) -> AnnotationInspectorDock:
        if self._inspector_dock is not None:
            return self._inspector_dock
        dock = AnnotationInspectorDock(self)
        dock.inspectorItemClicked.connect(self._on_inspector_item_clicked)
        dock.dockAboutToClose.connect(self._on_inspector_dock_close)
        dock.fnItemClicked.connect(self._on_inspector_fn_item_clicked)
        # Chinese-CLIP FP UI is hidden for now; handlers kept for a later VLM phase.
        host = self._find_dock_host()
        if host is not None:
            host.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        else:
            dock.setFloating(True)
        self._inspector_dock = dock
        return dock

    def _collect_inspector_labels(self) -> list:
        labels = []
        seen = set()
        cache = getattr(self, "_inspector_cache", None)
        if cache is None:
            return labels
        try:
            for it in cache.get_all_crop_items():
                lab = str(getattr(it, "label", "") or "").strip()
                if lab and lab not in seen:
                    seen.add(lab)
                    labels.append(lab)
        except Exception:
            pass
        if not labels:
            for lab in list(getattr(self, "_inspector_labels_filter", None) or []):
                s = str(lab or "").strip()
                if s and s not in seen:
                    seen.add(s)
                    labels.append(s)
        return labels

    def _prompt_fp_label_phrases(
        self, *, require_confirm: bool = True
    ) -> Optional[Dict[str, str]]:
        """Show label→phrase dialog. Returns mapping, or None if cancelled."""
        if self._inspector_cache is None:
            return None
        from anylabeling.views.labeling.advanced.batch_annotation_check.fp_prompt_dialog import (
            FpLabelPromptDialog,
        )

        labels = self._collect_inspector_labels()
        if not labels:
            QMessageBox.information(
                self,
                self.tr("Batch Annotation Check"),
                self.tr("No labels available for phrase mapping."),
            )
            return None if require_confirm else {}
        existing = {}
        try:
            existing = self._inspector_cache.get_fp_label_prompts()
        except Exception:
            existing = {}
        dlg = FpLabelPromptDialog(
            self, labels=labels, existing=existing
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        mapping = dlg.mapping()
        try:
            self._inspector_cache.set_fp_label_prompts(mapping)
        except Exception as e:
            logger.warning("Failed to save FP label prompts: %s", e)
        return mapping

    def _on_edit_fp_label_prompts(self) -> None:
        if not self._inspector_enabled or self._inspector_cache is None:
            return
        self._prompt_fp_label_phrases(require_confirm=True)

    def _on_run_fp_clip(self, threshold: float) -> None:
        if getattr(self, "_inspector_fp_running", False):
            return
        if not self._inspector_enabled or self._inspector_cache is None:
            return
        items = self._inspector_cache.get_all_crop_items()
        if not items:
            QMessageBox.information(
                self,
                self.tr("Batch Annotation Check"),
                self.tr("No crops available for FP check."),
            )
            return

        label_prompts = self._prompt_fp_label_phrases(require_confirm=True)
        if label_prompts is None:
            return

        progress = QtWidgets.QProgressDialog(
            self.tr("Running Chinese-CLIP FP check (CPU)..."),
            self.tr("Cancel"),
            0,
            max(1, len(items)),
            self,
        )
        progress.setWindowTitle(self.tr("Batch Annotation Check"))
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        worker = InspectorFpWorker(
            items=items,
            threshold=float(threshold),
            label_prompts=label_prompts,
        )
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        self._inspector_fp_worker = worker
        self._inspector_fp_thread = thread
        self._inspector_fp_running = True
        self._inspector_fp_progress = progress

        def _on_progress(i: int, n: int, _name: str) -> None:
            progress.setMaximum(max(1, n))
            progress.setValue(min(i, n))

        def _on_status(msg: str) -> None:
            if msg:
                progress.setLabelText(str(msg))

        def _on_cancel() -> None:
            try:
                worker.request_cancel()
            except Exception:
                pass

        def _cleanup() -> None:
            self._inspector_fp_running = False
            self._inspector_fp_worker = None
            self._inspector_fp_thread = None
            self._inspector_fp_progress = None
            try:
                progress.close()
            except Exception:
                pass

        def _on_finished(ok: bool, err: str, results: object) -> None:
            rows = results if isinstance(results, list) else []
            try:
                thread.quit()
            except Exception:
                pass
            if not ok:
                _cleanup()
                QMessageBox.warning(
                    self,
                    self.tr("Batch Annotation Check"),
                    self.tr("FP check failed: %s") % (err or "unknown"),
                )
                return
            try:
                self._inspector_cache.apply_fp_results(rows)
            except Exception as e:
                logger.warning("Failed to save FP results to cache: %s", e)
            marked_files = 0
            try:
                marked_files = _apply_suspect_fp_marks(rows)
            except Exception as e:
                logger.warning("Failed to write suspect_fp marks: %s", e)
            try:
                self._sync_canvas_suspect_fp_from_results(rows)
            except Exception:
                pass
            try:
                self._refresh_inspector_for_current_image()
            except Exception:
                pass
            _cleanup()

            n_fail = sum(1 for r in rows if isinstance(r, dict) and r.get("passed") is False)
            n_err = sum(1 for r in rows if isinstance(r, dict) and r.get("error"))
            n_ok = sum(1 for r in rows if isinstance(r, dict) and r.get("passed") is True)
            QMessageBox.information(
                self,
                self.tr("Batch Annotation Check"),
                self.tr(
                    "Chinese-CLIP FP check finished.\n"
                    "Match: {ok}\n"
                    "Suspect FP: {fail}\n"
                    "Errors: {err}\n"
                    "Label files updated: {files}"
                ).format(ok=n_ok, fail=n_fail, err=n_err, files=marked_files),
            )

        progress.canceled.connect(_on_cancel)
        worker.progress.connect(_on_progress)
        worker.status.connect(_on_status)
        worker.finished.connect(_on_finished)
        thread.started.connect(worker.run)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _sync_canvas_suspect_fp_from_results(self, results: List[Dict[str, Any]]) -> None:
        """Update in-memory canvas shapes for the currently open image."""
        if not results or not getattr(self, "filename", None):
            return
        from anylabeling.views.labeling.advanced.batch_annotation_check.cropper import (
            shape_key_from_shape,
        )
        from anylabeling.views.labeling.advanced.compare.annotation_compare import (
            COMPARE_STATUS,
            STATUS_SUSPECT_FP,
        )

        cur = osp.normcase(osp.normpath(osp.abspath(str(self.filename))))
        relevant = []
        for row in results:
            if not isinstance(row, dict):
                continue
            ip = str(row.get("image_path") or "")
            if not ip:
                continue
            try:
                if osp.normcase(osp.normpath(osp.abspath(ip))) != cur:
                    continue
            except Exception:
                continue
            relevant.append(row)
        if not relevant:
            return

        shapes = list(getattr(self.canvas, "shapes", None) or [])
        changed = False
        for row in relevant:
            if row.get("error") or row.get("passed") is None:
                continue
            key = str(row.get("shape_key") or "").strip()
            is_suspect = row.get("passed") is False
            target = None
            if key:
                for shape in shapes:
                    try:
                        if shape_key_from_shape(shape) == key:
                            target = shape
                            break
                    except Exception:
                        continue
            if target is None:
                try:
                    si = int(row.get("shape_index", -1))
                except Exception:
                    si = -1
                if 0 <= si < len(shapes):
                    target = shapes[si]
            if target is None:
                continue
            attrs = getattr(target, "attributes", None)
            if not isinstance(attrs, dict):
                attrs = {}
            cur_status = attrs.get(COMPARE_STATUS)
            if is_suspect:
                if cur_status != STATUS_SUSPECT_FP:
                    attrs = dict(attrs)
                    attrs[COMPARE_STATUS] = STATUS_SUSPECT_FP
                    target.attributes = attrs
                    changed = True
            elif cur_status == STATUS_SUSPECT_FP:
                attrs = dict(attrs)
                del attrs[COMPARE_STATUS]
                target.attributes = attrs
                changed = True
        if changed:
            try:
                self.set_dirty()
            except Exception:
                pass
            try:
                self.canvas.update()
            except Exception:
                pass


    def _on_inspector_fn_item_clicked(
        self, image_path: str, _label: str, bbox01: object
    ) -> None:
        if image_path and not self._paths_refer_same_image(
            getattr(self, "filename", "") or "", image_path
        ):
            if not self._load_file_for_inspector_crop_nav(image_path):
                return
        try:
            self.canvas.set_inspector_overlays_from_bbox01(bbox01)
        except Exception:
            pass



    def _find_dock_host(self):
        w = self
        while w is not None:
            if hasattr(w, "addDockWidget") and hasattr(w, "removeDockWidget"):
                return w
            try:
                w = w.parent()
            except Exception:
                w = getattr(w, "parent", None)
        return None


    def _abort_inspector_build_return_to_labels(self) -> None:
        """Discard partial build cache and run label + import/build flow again."""
        self._inspector_worker = None
        self._inspector_thread = None
        if self._inspector_cache is not None:
            try:
                self._inspector_cache.clear()
            except Exception:
                pass
            self._inspector_cache = None
        self._inspector_labels_filter = []
        self._inspector_enabled = False
        try:
            self.actions.batch_annotation_check.setChecked(True)
        except Exception:
            pass
        QtCore.QTimer.singleShot(0, self._enable_annotation_inspector)


    def _start_inspector_preprocess(self) -> None:
        if self._inspector_cache is None:
            return

        progress = QtWidgets.QProgressDialog(
            self.tr("Building Batch Annotation Check cache..."),
            self.tr("Cancel"),
            0,
            max(1, len(self.image_list)),
            self,
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        self._inspector_progress = progress

        worker = InspectorBuildWorker(
            cache=self._inspector_cache,
            image_paths=list(self.image_list),
            output_dir=self.output_dir,
            labels_filter=list(self._inspector_labels_filter),
        )
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_inspector_progress)
        worker.finished.connect(self._on_inspector_preprocess_done)
        # DirectConnection: worker lives on the worker thread but run() blocks its event
        # loop, so QueuedConnection would never deliver request_cancel until build finishes.
        _direct = Qt.ConnectionType.DirectConnection
        progress.canceled.connect(worker.request_cancel, _direct)
        progress.rejected.connect(worker.request_cancel, _direct)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._inspector_thread = thread
        self._inspector_worker = worker
        thread.start()


    def _on_inspector_progress(self, i: int, n: int, path: str) -> None:
        if self._inspector_progress is None:
            return
        self._inspector_progress.setMaximum(max(1, int(n)))
        self._inspector_progress.setValue(max(0, int(i)))
        self._inspector_progress.setLabelText(
            self.tr("Building cache: %d/%d\n%s") % (i, n, path)
        )


    def _on_inspector_preprocess_done(self, ok: bool, err: str) -> None:
        if self._inspector_progress is not None:
            try:
                self._inspector_progress.close()
            except Exception:
                pass
            self._inspector_progress = None

        if not ok:
            err_norm = (err or "").strip().lower()
            if err_norm in ("canceled", "cancelled"):
                self._abort_inspector_build_return_to_labels()
                return
            QMessageBox.warning(
                self,
                self.tr("Batch Annotation Check"),
                self.tr("Failed to build cache: %s") % err,
            )
            self.actions.batch_annotation_check.setChecked(False)
            return

        self._inspector_enabled = True
        dock = self._ensure_inspector_dock()
        dock.show()
        self._refresh_inspector_for_current_image()


    def _refresh_inspector_for_current_image(self) -> None:
        if not self._inspector_enabled or self._inspector_cache is None:
            return
        dock = self._ensure_inspector_dock()
        items = self._inspector_cache.get_all_crop_items()
        dock.set_items(items)
        try:
            fn_items = self._inspector_cache.get_all_fn_items()
            dock.set_fn_items(fn_items)
        except Exception:
            pass


    def _inspector_live_rebuild_current_image(self) -> None:
        """Rebuild inspector crops for current image from in-memory shapes."""
        if not getattr(self, "_inspector_enabled", False) or not self._inspector_cache:
            return
        if not self.filename or not hasattr(self, "image") or self.image.isNull():
            return
        try:
            # IMPORTANT: Use canvas.shapes order so that shape_index aligns with
            # click-to-highlight logic (canvas.shapes[shape_index]).
            shapes_live = []
            for s in getattr(self.canvas, "shapes", []) or []:
                try:
                    if not getattr(s, "label", None):
                        continue
                    if s.label in (
                        AutoLabelingMode.OBJECT,
                        AutoLabelingMode.ADD,
                        AutoLabelingMode.REMOVE,
                    ):
                        continue
                    shapes_live.append(s.to_dict())
                except Exception:
                    continue
            self._inspector_cache.update_for_image_live(
                self.filename,
                output_dir=self.output_dir,
                labels_filter=list(self._inspector_labels_filter),
                shapes=shapes_live,
                image_w=int(self.image.width()),
                image_h=int(self.image.height()),
            )
            self._refresh_inspector_for_current_image()
        except Exception:
            pass


    def _paths_refer_same_image(self, a: str, b: str) -> bool:
        try:
            return osp.normcase(osp.normpath(osp.abspath(str(a)))) == osp.normcase(
                osp.normpath(osp.abspath(str(b)))
            )
        except Exception:
            return str(a) == str(b)


    def _load_file_for_inspector_crop_nav(self, image_path: str) -> bool:
        """Switch main image for crop click without rebuilding thumbnail lists.

        ``load_file`` may return False after deferring to the file-list selection
        change (which triggers a nested ``load_file``). Keep the suppress flag
        for that nested call; clear leftovers afterward.
        """
        self._inspector_suppress_list_refresh_on_load = True
        try:
            ok = bool(self.load_file(image_path))
            if ok:
                return True
            return self._paths_refer_same_image(
                getattr(self, "filename", "") or "", image_path
            )
        finally:
            self._inspector_suppress_list_refresh_on_load = False


    def _find_shape_for_inspector_crop(
        self, *, shape_key: str, shape_index: int
    ):
        """Resolve crop → live shape by stable shape_key (not list index)."""
        from anylabeling.views.labeling.advanced.batch_annotation_check.cropper import (
            shape_bbox_key_from_shape,
            shape_key_from_shape,
        )

        shapes = list(getattr(self.canvas, "shapes", None) or [])
        key = str(shape_key or "").strip()
        if key:
            for shape in shapes:
                try:
                    if shape_key_from_shape(shape) == key:
                        return shape
                except Exception:
                    continue
            # Old cache rows may only have padded-bbox keys.
            if "\x1fb:" in key:
                try:
                    iw = int(self.image.width()) if hasattr(self, "image") else 0
                    ih = int(self.image.height()) if hasattr(self, "image") else 0
                except Exception:
                    iw, ih = 0, 0
                if iw > 0 and ih > 0:
                    for shape in shapes:
                        try:
                            if shape_bbox_key_from_shape(shape, iw, ih) == key:
                                return shape
                        except Exception:
                            continue
            return None
        # Legacy caches without shape_key: index is best-effort only.
        idx = int(shape_index)
        if 0 <= idx < len(shapes):
            return shapes[idx]
        return None

    def _on_inspector_item_clicked(
        self,
        image_path: str,
        _label_json_path: str,
        shape_index: int,
        shape_key: str = "",
    ) -> None:
        if not image_path:
            return
        if not self._paths_refer_same_image(
            getattr(self, "filename", "") or "", image_path
        ):
            if not self._load_file_for_inspector_crop_nav(image_path):
                return

        shape = self._find_shape_for_inspector_crop(
            shape_key=shape_key, shape_index=shape_index
        )
        if shape is None:
            return
        self.canvas.select_shapes([shape])
        if getattr(self.canvas, "show_masks", False):
            try:
                self.canvas.set_inspector_mask_suppressed(shape)
            except Exception:
                pass

    def _inspector_after_shapes_deleted(self, deleted_keys: list) -> None:
        """Keep crop grid as-is; only mark deleted bindings orphaned."""
        if not getattr(self, "_inspector_enabled", False):
            return
        dock = getattr(self, "_inspector_dock", None)
        if dock is None or not deleted_keys:
            return
        try:
            dock.mark_fp_orphans_by_keys(deleted_keys)
        except Exception:
            pass
        # Intentionally do NOT call _refresh_inspector_for_current_image /
        # _inspector_live_rebuild_current_image — list rebuild is too costly.


    def _on_inspector_dock_close(self) -> None:
        if self._inspector_cache is not None and osp.isdir(
            self._inspector_cache.root_dir
        ):
            msg = QMessageBox(self)
            msg.setWindowTitle(self.tr("Batch Annotation Check"))
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setText(self.tr("Save Annotation Inspector cache directory?"))
            btn_save = msg.addButton(
                self.tr("Save As..."), QMessageBox.ButtonRole.AcceptRole
            )
            btn_discard = msg.addButton(
                self.tr("Don't Save"), QMessageBox.ButtonRole.DestructiveRole
            )
            btn_cancel = msg.addButton(
                self.tr("Cancel"), QMessageBox.ButtonRole.RejectRole
            )
            msg.setDefaultButton(btn_save)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked is btn_cancel:
                # reopen dock and keep enabled
                if self._inspector_dock is not None:
                    self._inspector_dock.show()
                try:
                    self.actions.batch_annotation_check.setChecked(True)
                except Exception:
                    pass
                return
            if clicked is btn_save:
                dst_parent = QtWidgets.QFileDialog.getExistingDirectory(
                    self,
                    self.tr("Select a directory to save cache"),
                    "",
                )
                if not dst_parent:
                    if self._inspector_dock is not None:
                        self._inspector_dock.show()
                    try:
                        self.actions.batch_annotation_check.setChecked(True)
                    except Exception:
                        pass
                    return
                base = osp.basename(self._inspector_cache.root_dir.rstrip("/\\"))
                dst_dir = osp.join(dst_parent, base)
                if osp.exists(dst_dir):
                    k = 1
                    while osp.exists(f"{dst_dir}_{k}"):
                        k += 1
                    dst_dir = f"{dst_dir}_{k}"
                try:
                    self._inspector_cache.export_to(dst_dir)
                except Exception:
                    if self._inspector_dock is not None:
                        self._inspector_dock.show()
                    try:
                        self.actions.batch_annotation_check.setChecked(True)
                    except Exception:
                        pass
                    return
                try:
                    self._inspector_cache.clear()
                except Exception:
                    pass
            else:
                try:
                    self._inspector_cache.clear()
                except Exception:
                    pass

        self._inspector_cache = None
        self._inspector_enabled = False
        try:
            self.actions.batch_annotation_check.setChecked(False)
        except Exception:
            pass

    # Help



def install(widget) -> None:
    import types

    for name in dir(_BatchCheckWidgetMethods):
        if name.startswith("_") and name not in (
            # keep private helpers
        ):
            pass
        attr = getattr(_BatchCheckWidgetMethods, name, None)
        if name.startswith("__"):
            continue
        if callable(attr):
            setattr(widget, name, types.MethodType(attr, widget))

    # Wrap delete so crop↔shape binding stays stable without refreshing the grid.
    if not getattr(widget, "_inspector_delete_wrapped", False):
        original_delete = widget.delete_selected_shape

        def _delete_selected_shape_with_inspector(self_w):
            from anylabeling.views.labeling.advanced.batch_annotation_check.cropper import (
                shape_bbox_key_from_shape,
                shape_key_from_shape,
            )

            deleted_keys = []
            try:
                shapes_to_delete = list(
                    getattr(self_w.canvas, "selected_shapes", None) or []
                )
                group_fn = getattr(
                    self_w.canvas, "_active_group_shapes", None
                )
                if callable(group_fn):
                    group = list(group_fn() or [])
                    if group:
                        shapes_to_delete = group
                iw = ih = 0
                try:
                    iw = int(self_w.image.width())
                    ih = int(self_w.image.height())
                except Exception:
                    pass
                for shape in shapes_to_delete:
                    deleted_keys.append(shape_key_from_shape(shape))
                    if iw > 0 and ih > 0:
                        deleted_keys.append(
                            shape_bbox_key_from_shape(shape, iw, ih)
                        )
            except Exception:
                deleted_keys = []
            original_delete()
            try:
                self_w._inspector_after_shapes_deleted(deleted_keys)
            except Exception:
                pass

        widget.delete_selected_shape = types.MethodType(
            _delete_selected_shape_with_inspector, widget
        )
        widget._inspector_delete_wrapped = True
