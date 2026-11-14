"""Annotation compare / model-eval review handlers — bound onto LabelingWidget by advanced.bootstrap."""
from __future__ import annotations

import html
import os.path as osp
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QMessageBox

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.shape import Shape

try:
    from anylabeling.views.labeling.label_widget import LABEL_OPACITY
except Exception:  # pragma: no cover
    LABEL_OPACITY = 128


class _CompareWidgetMethods:
    """Mixin-style methods; install() attaches them to a LabelingWidget instance."""

    def get_labeling_instruction(self):
        """Extend mode instruction text while compare/model-eval review is active."""
        if not getattr(self, "_annotation_compare_active", False):
            return self._advanced_original_get_labeling_instruction()

        text_mode = self.tr("Mode:")
        text_shortcuts = self.tr("Shortcuts:")
        text_settings = self.tr("Settings")
        text_previous = self.tr("Previous")
        text_next = self.tr("Next")
        shortcuts = self._config.get("shortcuts", {})
        if self._review_marks_mode_is_model_eval():
            text_compare = self.tr("Model Evaluation Review")
            text_next_fn = self.tr("Next false alarm")
            text_next_fp = self.tr("Next miss")
        else:
            text_compare = self.tr("Annotation Compare Review")
            text_next_fn = self.tr("Next missed")
            text_next_fp = self.tr("Next false-label")
        browse_on = bool(getattr(self, "_compare_browse_mode", False))
        text_browse = (
            self.tr("Browse mode on")
            if browse_on
            else self.tr("Browse mode off")
        )
        fn_sc = shortcuts.get("open_next_compare_fn") or shortcuts.get(
            "open_next_compare_issue"
        )
        return (
            f"<b>{text_mode}</b> {text_compare} | "
            f"<b>{text_browse}</b> | "
            f"<b>{text_next_fn}</b>"
            f"({self._format_instruction_shortcut(fn_sc)}), "
            f"<b>{text_next_fp}</b>"
            f"({self._format_instruction_shortcut(shortcuts.get('open_next_compare_fp'))}) | "
            f"<b>{text_shortcuts}</b>"
            f" {text_previous}({self._format_instruction_shortcut(shortcuts.get('open_prev'))}),"
            f" {text_next}({self._format_instruction_shortcut(shortcuts.get('open_next'))}),"
            f" {text_settings}({self._format_instruction_shortcut(shortcuts.get('open_settings'))})"
        )

    def _on_compare_browse_mode_toggled(self, checked: bool) -> None:
        self._compare_browse_mode = bool(checked)
        if hasattr(self, "label_instruction"):
            self.label_instruction.setText(self.get_labeling_instruction())


    def _review_marks_mode_is_model_eval(self) -> bool:
        from anylabeling.views.labeling.advanced.compare.annotation_compare import (
            REVIEW_MODE_MODEL_EVAL,
        )

        return (
            getattr(self, "_review_marks_mode", "annotation")
            == REVIEW_MODE_MODEL_EVAL
        )


    def _sync_compare_issue_nav_enabled(self) -> None:
        enabled = bool(
            getattr(self, "_annotation_compare_active", False)
            and getattr(self, "image_list", None)
        )
        model_mode = self._review_marks_mode_is_model_eval() if enabled else False
        pending = 0
        if enabled:
            from anylabeling.views.labeling.advanced.compare.annotation_compare import (
                count_pending_review,
            )

            try:
                pending = count_pending_review(self.canvas.shapes)
            except Exception:
                pending = 0

        def _set_fn_btn(btn, *, prev: bool) -> None:
            if btn is None:
                return
            btn.setVisible(enabled)
            btn.setEnabled(enabled)
            if not enabled:
                return
            if model_mode:
                btn.setText(
                    self.tr("Prev False Alarm") if prev else self.tr("Next False Alarm")
                )
                btn.setToolTip(
                    self.tr(
                        "Open previous image with model false alarms (误报)"
                        if prev
                        else "Open next image with model false alarms (误报)"
                    )
                )
            else:
                btn.setText(
                    self.tr("Prev Missed") if prev else self.tr("Next Missed")
                )
                btn.setToolTip(
                    self.tr(
                        "Open previous image with suspect missed labels"
                        if prev
                        else "Open next image with suspect missed labels"
                    )
                )

        def _set_fp_btn(btn, *, prev: bool) -> None:
            if btn is None:
                return
            btn.setVisible(enabled)
            btn.setEnabled(enabled)
            if not enabled:
                return
            if model_mode:
                btn.setText(
                    self.tr("Prev Miss") if prev else self.tr("Next Miss")
                )
                btn.setToolTip(
                    self.tr(
                        "Open previous image with model misses (漏报)"
                        if prev
                        else "Open next image with model misses (漏报)"
                    )
                )
            else:
                btn.setText(
                    self.tr("Prev False-Label")
                    if prev
                    else self.tr("Next False-Label")
                )
                btn.setToolTip(
                    self.tr(
                        "Open previous image with suspect false labels"
                        if prev
                        else "Open next image with suspect false labels"
                    )
                )

        _set_fn_btn(getattr(self, "_compare_fn_prev_btn", None), prev=True)
        _set_fn_btn(getattr(self, "_compare_fn_nav_btn", None), prev=False)
        _set_fp_btn(getattr(self, "_compare_fp_prev_btn", None), prev=True)
        _set_fp_btn(getattr(self, "_compare_fp_nav_btn", None), prev=False)

        browse_cb = getattr(self, "_compare_browse_mode_cb", None)
        if browse_cb is not None:
            browse_cb.setVisible(enabled)
            browse_cb.setEnabled(enabled)
            if enabled:
                browse_cb.blockSignals(True)
                browse_cb.setChecked(
                    bool(getattr(self, "_compare_browse_mode", False))
                )
                browse_cb.blockSignals(False)

        for btn_name in (
            "_compare_accept_all_btn",
            "_compare_clear_image_btn",
        ):
            btn = getattr(self, btn_name, None)
            if btn is None:
                continue
            btn.setVisible(enabled)
            btn.setEnabled(enabled and pending > 0)

        clear_ds = getattr(self, "_compare_clear_dataset_btn", None)
        if clear_ds is not None:
            clear_ds.setVisible(enabled)
            clear_ds.setEnabled(enabled)

        for action_name in (
            "open_next_compare_fn_image",
            "open_next_compare_fp_image",
            "open_prev_compare_fn_image",
            "open_prev_compare_fp_image",
        ):
            action = getattr(self.actions, action_name, None)
            if action is not None:
                action.setEnabled(enabled)
        try:
            from anylabeling.views.labeling.advanced.compare.ui_bar import (
                sync_compare_review_bar_visibility,
            )

            sync_compare_review_bar_visibility(self)
        except Exception:
            pass
        if hasattr(self, "label_instruction"):
            self.label_instruction.setText(self.get_labeling_instruction())
            if enabled:
                self.label_instruction.show()


    def accept_compare_results_current_image(self, _value=False) -> bool:
        """Accept all pending compare marks on the current canvas and save."""
        if not getattr(self, "_annotation_compare_active", False):
            return False
        if not self.filename:
            QMessageBox.information(
                self,
                self.tr("Annotation Compare Review"),
                self.tr("No image loaded."),
            )
            return False

        from anylabeling.views.labeling.advanced.compare.annotation_compare import (
            PENDING_GT_STATUSES,
            PENDING_PRED_STATUSES,
            count_pending_review,
            get_compare_status,
        )

        if count_pending_review(self.canvas.shapes) <= 0:
            self.status(
                self.tr("No unreviewed compare marks on this image."),
                3000,
            )
            return False

        fn_count = 0
        fp_count = 0
        for shape in self.canvas.shapes:
            status = get_compare_status(shape)
            if status in PENDING_PRED_STATUSES:
                fn_count += 1
            elif status in PENDING_GT_STATUSES:
                fp_count += 1

        self._accept_all_pending_compare_on_canvas()
        self.set_dirty()
        try:
            self.save_file()
        except Exception:
            pass
        self._sync_compare_issue_nav_enabled()
        if self._review_marks_mode_is_model_eval():
            self.status(
                self.tr(
                    "Accepted model-eval results: %d false alarm(s) kept, "
                    "%d miss flag(s) cleared."
                )
                % (fn_count, fp_count),
                6000,
            )
        else:
            self.status(
                self.tr(
                    "Accepted compare results: %d missed label(s) kept, "
                    "%d false-label flag(s) cleared."
                )
                % (fn_count, fp_count),
                6000,
            )
        return True


    def clear_compare_results_current_image(self, _value=False) -> bool:
        """Clear all pending compare marks on the current canvas and save."""
        if not getattr(self, "_annotation_compare_active", False):
            return False
        if not self.filename:
            QMessageBox.information(
                self,
                self.tr("Annotation Compare Review"),
                self.tr("No image loaded."),
            )
            return False

        from anylabeling.views.labeling.advanced.compare.annotation_compare import (
            PENDING_GT_STATUSES,
            PENDING_PRED_STATUSES,
            count_pending_review,
            get_compare_status,
        )

        if count_pending_review(self.canvas.shapes) <= 0:
            self.status(
                self.tr("No unreviewed compare marks on this image."),
                3000,
            )
            return False

        fn_count = 0
        fp_count = 0
        for shape in self.canvas.shapes:
            status = get_compare_status(shape)
            if status in PENDING_PRED_STATUSES:
                fn_count += 1
            elif status in PENDING_GT_STATUSES:
                fp_count += 1

        reply = QMessageBox.question(
            self,
            self.tr("Clear This Check"),
            self.tr(
                "Clear all suspect marks on this image?\n\n"
                "• Delete pred-side marks (漏标/误报): %d\n"
                "• Clear GT-side flags (误标/漏报): %d"
            )
            % (fn_count, fp_count),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return False

        self._clear_all_pending_compare_on_canvas()
        self.set_dirty()
        try:
            self.save_file()
        except Exception:
            pass
        self._sync_compare_issue_nav_enabled()
        self.status(
            self.tr(
                "Cleared this image: removed %d pred-side mark(s), "
                "cleared %d GT-side flag(s)."
            )
            % (fn_count, fp_count),
            6000,
        )
        return True


    def clear_compare_results_dataset(self, _value=False) -> None:
        """Reuse Advanced → Clear Compare Results for the whole dataset."""
        from anylabeling.views.labeling.advanced.compare.annotation_compare_dialog import (
            run_clear_compare_results_ui,
        )

        run_clear_compare_results_ui(self)
        try:
            self._sync_compare_issue_nav_enabled()
        except Exception:
            pass


    def open_next_compare_fn_image(self, _value=False) -> bool:
        """Jump to next image with pred-side marks (漏标 or 误报)."""
        return self._open_compare_fn_image(direction=1)


    def open_prev_compare_fn_image(self, _value=False) -> bool:
        """Jump to previous image with pred-side marks (漏标 or 误报)."""
        return self._open_compare_fn_image(direction=-1)


    def open_next_compare_fp_image(self, _value=False) -> bool:
        """Jump to next image with GT-side marks (误标 or 漏报)."""
        return self._open_compare_fp_image(direction=1)


    def open_prev_compare_fp_image(self, _value=False) -> bool:
        """Jump to previous image with GT-side marks (误标 or 漏报)."""
        return self._open_compare_fp_image(direction=-1)


    def _open_compare_fn_image(self, *, direction: int) -> bool:
        from anylabeling.views.labeling.advanced.compare.annotation_compare import (
            STATUS_MODEL_FALSE,
            STATUS_SUSPECT_FN,
        )

        going_prev = int(direction) < 0
        if self._review_marks_mode_is_model_eval():
            return self._open_compare_issue_image(
                direction=direction,
                statuses=(STATUS_MODEL_FALSE,),
                empty_msg=self.tr(
                    "No more images with false alarms in this dataset."
                ),
                same_msg=self.tr(
                    "Already on the first image with false alarms in this pass."
                    if going_prev
                    else "Already on the last image with false alarms in this pass."
                ),
            )
        return self._open_compare_issue_image(
            direction=direction,
            statuses=(STATUS_SUSPECT_FN,),
            empty_msg=self.tr(
                "No more images with suspect missed labels in this dataset."
            ),
            same_msg=self.tr(
                "Already on the first image with suspect missed labels in this pass."
                if going_prev
                else "Already on the last image with suspect missed labels in this pass."
            ),
        )


    def _open_compare_fp_image(self, *, direction: int) -> bool:
        from anylabeling.views.labeling.advanced.compare.annotation_compare import (
            STATUS_MODEL_MISS,
            STATUS_SUSPECT_FP,
        )

        going_prev = int(direction) < 0
        if self._review_marks_mode_is_model_eval():
            return self._open_compare_issue_image(
                direction=direction,
                statuses=(STATUS_MODEL_MISS,),
                empty_msg=self.tr(
                    "No more images with misses in this dataset."
                ),
                same_msg=self.tr(
                    "Already on the first image with misses in this pass."
                    if going_prev
                    else "Already on the last image with misses in this pass."
                ),
            )
        return self._open_compare_issue_image(
            direction=direction,
            statuses=(STATUS_SUSPECT_FP,),
            empty_msg=self.tr(
                "No more images with suspect false labels in this dataset."
            ),
            same_msg=self.tr(
                "Already on the first image with suspect false labels in this pass."
                if going_prev
                else "Already on the last image with suspect false labels in this pass."
            ),
        )


    def open_next_compare_issue_image(self, _value=False) -> bool:
        """Backward-compatible: jump to next image with any suspect FN/FP."""
        return self._open_compare_issue_image(direction=1)


    def _open_next_compare_issue_image(
        self,
        *,
        statuses=None,
        empty_msg: Optional[str] = None,
        same_msg: Optional[str] = None,
    ) -> bool:
        """Backward-compatible alias for next-direction navigation."""
        return self._open_compare_issue_image(
            direction=1,
            statuses=statuses,
            empty_msg=empty_msg,
            same_msg=same_msg,
        )


    def _open_compare_issue_image(
        self,
        *,
        direction: int = 1,
        statuses=None,
        empty_msg: Optional[str] = None,
        same_msg: Optional[str] = None,
    ) -> bool:
        """Jump to next/previous image with matching suspect marks."""
        if not getattr(self, "_annotation_compare_active", False):
            return False
        image_list = list(getattr(self, "image_list", []) or [])
        if not image_list:
            return False
        from anylabeling.views.labeling.advanced.compare.annotation_compare_io import (
            find_next_pending_compare_image,
            find_prev_pending_compare_image,
        )

        finder = (
            find_prev_pending_compare_image
            if int(direction) < 0
            else find_next_pending_compare_image
        )
        next_path = finder(
            image_list,
            current_path=self.filename,
            output_dir=getattr(self, "output_dir", None),
            statuses=statuses,
        )
        if not next_path:
            self.status(
                empty_msg
                or self.tr(
                    "No more images with suspect missed/false labels in this dataset."
                ),
                4000,
            )
            return False
        cur = osp.normpath(str(self.filename)) if self.filename else ""
        if osp.normpath(str(next_path)) == cur:
            self.status(
                same_msg
                or self.tr(
                    "Already on the last image with compare issues in this pass."
                ),
                3000,
            )
            return False
        if not self._ensure_can_leave_current_image(next_path):
            return False
        if not self.may_continue():
            return False
        self.load_file(next_path)
        return True


    def _ensure_can_leave_current_image(self, _next_path=None) -> bool:
        """
        Gate image switches while compare/model-eval review is active.

        Browse mode skips the prompt. Otherwise, if this image still has
        unreviewed marks, ask: keep (allow leave), accept (apply + stay),
        or cancel.
        """
        if not getattr(self, "_annotation_compare_active", False):
            return True
        if bool(getattr(self, "_compare_browse_mode", False)):
            return True

        from anylabeling.views.labeling.advanced.compare.annotation_compare import (
            count_pending_review,
        )

        if count_pending_review(self.canvas.shapes) <= 0:
            return True

        choice = self._prompt_unreviewed_compare_on_leave()
        if choice == "keep":
            return True
        if choice == "accept":
            self._accept_all_pending_compare_on_canvas()
            self.set_dirty()
            try:
                self.save_file()
            except Exception:
                pass
            self._sync_compare_issue_nav_enabled()
            # Stay on this image so the user can verify accepted results.
            return False
        return False


    def _prompt_unreviewed_compare_on_leave(self) -> Optional[str]:
        """Return 'keep', 'accept', or None (cancel navigation)."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("Unreviewed Compare Marks"))
        msg_box.setText(
            self.tr(
                "This image still has unreviewed suspect missed/false labels."
            )
        )
        msg_box.setInformativeText(
            self.tr(
                "Choose an action before switching images:\n"
                "• Keep markers — save and switch to the next image now\n"
                "• Accept results — apply to this image and stay here; "
                "switch manually after you verify"
            )
        )
        keep_btn = msg_box.addButton(
            self.tr("Keep markers"),
            QMessageBox.ButtonRole.YesRole,
        )
        accept_btn = msg_box.addButton(
            self.tr("Accept results"),
            QMessageBox.ButtonRole.NoRole,
        )
        cancel_btn = msg_box.addButton(
            self.tr("Cancel"),
            QMessageBox.ButtonRole.RejectRole,
        )
        msg_box.exec()
        clicked = msg_box.clickedButton()
        if clicked == keep_btn:
            return "keep"
        if clicked == accept_btn:
            return "accept"
        return None


    def _apply_pending_compare_actions_on_canvas(
        self, *, pred_action: str, gt_action: str
    ) -> None:
        from anylabeling.views.labeling.shape import Shape
        from anylabeling.views.labeling.advanced.compare.annotation_compare import (
            PENDING_GT_STATUSES,
            PENDING_PRED_STATUSES,
            apply_review_action,
            get_compare_status,
        )

        for idx in range(len(self.canvas.shapes) - 1, -1, -1):
            shape = self.canvas.shapes[idx]
            status = get_compare_status(shape)
            if status in PENDING_PRED_STATUSES:
                action = pred_action
            elif status in PENDING_GT_STATUSES:
                action = gt_action
            else:
                continue
            result = apply_review_action(shape, action)
            if result is None:
                self.canvas.shapes.pop(idx)
                item = self.label_list.find_item_by_shape(shape)
                if item is not None:
                    self.label_list.remove_item(item)
                continue
            new_shape = Shape().load_from_dict(result, close=False)
            self.canvas.shapes[idx] = new_shape
            item = self.label_list.find_item_by_shape(shape)
            if item is not None:
                item.set_shape(new_shape)
                if new_shape.group_id is None:
                    item.setText(html.escape(new_shape.label))
                else:
                    item.setText(f"{new_shape.label} ({new_shape.group_id})")
                color = new_shape.fill_color.getRgb()[:3]
                item.setBackground(QtGui.QColor(*color, LABEL_OPACITY))
            self._update_shape_color(new_shape)

        self.canvas.selected_shapes = []
        self.label_list.clearSelection()
        self.apply_compare_review_colors()
        self.canvas.update()
        self._refresh_shape_filters()
        self._sync_compare_issue_nav_enabled()


    def _accept_all_pending_compare_on_canvas(self) -> None:
        from anylabeling.views.labeling.advanced.compare.annotation_compare import (
            FN_ACCEPT_LABEL,
            FP_KEEP,
        )

        self._apply_pending_compare_actions_on_canvas(
            pred_action=FN_ACCEPT_LABEL,
            gt_action=FP_KEEP,
        )


    def _clear_all_pending_compare_on_canvas(self) -> None:
        from anylabeling.views.labeling.advanced.compare.annotation_compare import (
            FN_REJECT,
            FP_KEEP,
        )

        # Same semantics as batch clear: drop pred-side boxes, strip GT flags.
        self._apply_pending_compare_actions_on_canvas(
            pred_action=FN_REJECT,
            gt_action=FP_KEEP,
        )


    def _maybe_resume_compare_review_on_load(self) -> None:
        """Resume compare/model-eval review UI when JSON still has pending markers."""
        if getattr(self, "_annotation_compare_active", False):
            return
        from anylabeling.views.labeling.advanced.compare.annotation_compare import (
            REVIEW_MODE_ANNOTATION,
            REVIEW_MODE_MODEL_EVAL,
            STATUS_MODEL_FALSE,
            STATUS_MODEL_MISS,
            count_pending_review,
            get_compare_status,
        )

        if count_pending_review(self.canvas.shapes) <= 0:
            return
        statuses = {get_compare_status(s) for s in self.canvas.shapes}
        has_model = bool(statuses & {STATUS_MODEL_FALSE, STATUS_MODEL_MISS})
        has_ann = bool(statuses & {"suspect_fn", "suspect_fp"})
        # Prefer model-eval UI only when no annotation-compare marks remain.
        if has_model and not has_ann:
            self._review_marks_mode = REVIEW_MODE_MODEL_EVAL
        else:
            self._review_marks_mode = REVIEW_MODE_ANNOTATION
        self._annotation_compare_active = True
        self._sync_compare_issue_nav_enabled()
        self.apply_compare_review_colors()


    def apply_compare_review_colors(self) -> None:
        """Highlight pending review boxes; matched use label colors."""
        if not getattr(self, "_annotation_compare_active", False):
            return
        from anylabeling.views.labeling.advanced.compare.annotation_compare import (
            compare_review_rgb,
            get_compare_status,
        )

        for shape in self.canvas.shapes:
            st = get_compare_status(shape)
            rgb = compare_review_rgb(st)
            if rgb is None:
                self._update_shape_color(shape)
                continue
            r, g, b = rgb
            shape.line_color = QtGui.QColor(r, g, b)
            shape.vertex_fill_color = QtGui.QColor(r, g, b)
            shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
            shape.fill_color = QtGui.QColor(r, g, b, 100)
            shape.select_line_color = QtGui.QColor(255, 255, 255)
            shape.select_fill_color = QtGui.QColor(r, g, b, 140)
        self.canvas.update()


    def _on_compare_review_label_clicked(self, shape: Shape) -> None:
        if not getattr(self, "_annotation_compare_active", False):
            return
        self._open_compare_review_for_shape(shape)


    def _open_compare_review_for_shape(self, shape: Shape) -> None:
        from anylabeling.views.labeling.advanced.compare.annotation_compare import (
            PENDING_REVIEW_STATUSES,
            get_compare_status,
        )
        from anylabeling.views.labeling.advanced.compare.annotation_compare_review_dialog import (
            AnnotationCompareReviewDialog,
        )

        status = get_compare_status(shape)
        if status not in PENDING_REVIEW_STATUSES:
            return
        if self._annotation_compare_dialog_open:
            return

        self._annotation_compare_dialog_open = True
        try:
            dlg = AnnotationCompareReviewDialog(
                self,
                status=status,
                label=str(shape.label or ""),
                class_names=getattr(self, "_annotation_compare_classes", None),
                default_action=AnnotationCompareReviewDialog.last_action_for(
                    status
                ),
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            action = dlg.action()
            if not action:
                return
            AnnotationCompareReviewDialog.remember_action(status, action)
            self._apply_compare_review_for_shape(shape, action, dlg.new_label())
        finally:
            self._annotation_compare_dialog_open = False


    def _apply_compare_review_for_shape(
        self,
        shape: Shape,
        action: str,
        new_label: Optional[str],
    ) -> None:
        from anylabeling.views.labeling.advanced.compare.annotation_compare import (
            apply_review_action,
        )

        try:
            idx = self.canvas.shapes.index(shape)
        except ValueError:
            return

        result = apply_review_action(shape, action, new_label=new_label)
        if result is None:
            self.canvas.shapes.pop(idx)
            item = self.label_list.find_item_by_shape(shape)
            if item is not None:
                self.label_list.remove_item(item)
        else:
            new_shape = Shape().load_from_dict(result, close=False)
            self.canvas.shapes[idx] = new_shape
            item = self.label_list.find_item_by_shape(shape)
            if item is not None:
                item.set_shape(new_shape)
                if new_shape.group_id is None:
                    item.setText(html.escape(new_shape.label))
                else:
                    item.setText(f"{new_shape.label} ({new_shape.group_id})")
                color = new_shape.fill_color.getRgb()[:3]
                item.setBackground(QtGui.QColor(*color, LABEL_OPACITY))
            self._update_shape_color(new_shape)

        self.canvas.selected_shapes = []
        self.label_list.clearSelection()
        self.set_dirty()
        try:
            self.save_file()
        except Exception:
            pass
        self.apply_compare_review_colors()
        self.canvas.update()
        self._refresh_shape_filters()
        self._sync_compare_issue_nav_enabled()




def install(widget) -> None:
    import types

    # Preserve upstream instruction text builder for non-compare modes.
    if not hasattr(widget, "_advanced_original_get_labeling_instruction"):
        widget._advanced_original_get_labeling_instruction = (
            widget.get_labeling_instruction
        )
    widget._annotation_compare_dialog_open = False

    for name in dir(_CompareWidgetMethods):
        attr = getattr(_CompareWidgetMethods, name, None)
        if name.startswith("__"):
            continue
        if callable(attr):
            setattr(widget, name, types.MethodType(attr, widget))
