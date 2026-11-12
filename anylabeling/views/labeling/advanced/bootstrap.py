"""Register Advanced menu and install canvas/widget hooks (no license)."""

from __future__ import annotations

import functools
from typing import Any

from PyQt6.QtWidgets import QMessageBox

from anylabeling.views.labeling import utils


def install_canvas_hooks(canvas) -> None:
    """Install sticker + compare/inspector canvas helpers (idempotent)."""
    if getattr(canvas, "_advanced_hooks_installed", False):
        return
    try:
        from anylabeling.views.labeling.advanced.sticker.canvas_controller import (
            install_sticker_support,
        )

        install_sticker_support(canvas)
    except Exception:
        pass
    try:
        from anylabeling.views.labeling.advanced.compare.canvas_hooks import (
            install_compare_canvas_hooks,
        )

        install_compare_canvas_hooks(canvas)
    except Exception:
        pass
    canvas._advanced_hooks_installed = True


def _install_widget_controllers(widget) -> None:
    import weakref

    from anylabeling.views.labeling.advanced.batch_annotation_check.widget_controller import (
        install as install_batch,
    )
    from anylabeling.views.labeling.advanced.compare.ui_bar import (
        install_compare_ui_bar,
    )
    from anylabeling.views.labeling.advanced.compare.widget_controller import (
        install as install_compare,
    )
    from anylabeling.views.labeling.advanced.sticker.widget_controller import (
        install as install_sticker,
    )

    install_compare(widget)
    install_batch(widget)
    install_sticker(widget)

    widget._inspector_enabled = False
    widget._inspector_dock = None
    widget._inspector_cache = None
    widget._inspector_labels_filter = []
    widget._inspector_progress = None
    widget._inspector_thread = None
    widget._inspector_worker = None
    widget._sticker_dock = None
    widget._label_jump_dialog = None
    widget._annotation_compare_active = False
    widget._annotation_compare_classes = []
    widget._review_marks_mode = "annotation"
    widget._compare_browse_mode = False
    widget._annotation_compare_dialog_open = False

    # Let canvas hooks resolve compare-active without fragile parent walks.
    try:
        widget.canvas._advanced_labeling_widget_ref = weakref.ref(widget)
    except Exception:
        pass

    # Dialog openers (not extracted into controllers)
    widget.open_annotation_compare = lambda *_a, **_k: _open_annotation_compare(
        widget
    )
    widget.open_model_eval = lambda *_a, **_k: _open_model_eval(widget)
    widget.open_clear_annotation_compare_results = (
        lambda *_a, **_k: _open_clear_compare(widget)
    )

    # Mode-row buttons: browse / accept / next FN / next FP
    try:
        install_compare_ui_bar(widget)
    except Exception:
        pass


def _open_dataset_statistics(widget) -> None:
    if not getattr(widget, "image_list", None):
        QMessageBox.information(
            widget,
            widget.tr("Dataset Statistics"),
            widget.tr("No images loaded."),
        )
        return
    try:
        from anylabeling.views.labeling.advanced.dataset_analysis.dataset_analysis_dialog import (
            open_dataset_analysis_dialog,
        )
    except Exception as e:
        QMessageBox.warning(
            widget,
            widget.tr("Dataset Statistics"),
            widget.tr("Failed to open dialog: %s") % str(e),
        )
        return
    open_dataset_analysis_dialog(widget)


def _guarded(widget, title: str, opener, need_images: bool = True) -> None:
    if need_images and not getattr(widget, "image_list", None):
        QMessageBox.information(
            widget,
            widget.tr(title),
            widget.tr("Please open an image folder first."),
        )
        return
    try:
        opener(widget)
    except Exception as e:
        QMessageBox.warning(
            widget,
            widget.tr(title),
            widget.tr("Failed to open: %s") % str(e),
        )


def _open_annotation_compare(widget) -> None:
    if not getattr(widget, "image_list", None):
        QMessageBox.information(
            widget,
            widget.tr("Annotation Compare"),
            widget.tr("No images loaded."),
        )
        return
    from anylabeling.views.labeling.advanced.compare.annotation_compare_dialog import (
        open_annotation_compare_dialog,
    )

    open_annotation_compare_dialog(widget)


def _open_model_eval(widget) -> None:
    if not getattr(widget, "image_list", None):
        QMessageBox.information(
            widget,
            widget.tr("Model Evaluation"),
            widget.tr("No images loaded."),
        )
        return
    from anylabeling.views.labeling.advanced.compare.model_eval_dialog import (
        open_model_eval_dialog,
    )

    open_model_eval_dialog(widget)


def _open_clear_compare(widget) -> None:
    if not widget.filename:
        QMessageBox.information(
            widget,
            widget.tr("Clear Compare Results"),
            widget.tr("Please load an image folder before proceeding!"),
        )
        return
    from anylabeling.views.labeling.advanced.compare.annotation_compare_dialog import (
        run_clear_compare_results_ui,
    )

    run_clear_compare_results_ui(widget)


def _open_label_jump(widget) -> None:
    if not getattr(widget, "image_list", None):
        QMessageBox.information(
            widget,
            widget.tr("Jump by Labels"),
            widget.tr("Please open an image folder first."),
        )
        return
    from anylabeling.views.labeling.advanced.ops.label_jump_dialog import (
        open_label_jump_dialog,
    )

    open_label_jump_dialog(widget)


def register_advanced_menu(widget: Any) -> None:
    """
    Create Advanced menu actions on `widget` (LabelingWidget).
    Expects `widget.menus.advanced` to already exist.
    """
    install_canvas_hooks(widget.canvas)
    _install_widget_controllers(widget)

    try:
        widget.canvas.compare_review_label_clicked.connect(
            widget._on_compare_review_label_clicked
        )
    except Exception:
        pass

    action = functools.partial(utils.new_action, widget)

    open_stats = action(
        widget.tr("Dataset Statistics"),
        lambda: _open_dataset_statistics(widget),
        tip=widget.tr("Batch image quality metrics and annotation distribution"),
    )
    dataset_split = action(
        widget.tr("Dataset Split…"),
        lambda: _guarded(
            widget,
            "Dataset Split",
            lambda w: __import__(
                "anylabeling.views.labeling.advanced.ops.dataset_split_dialog",
                fromlist=["open_dataset_split_dialog"],
            ).open_dataset_split_dialog(w),
            need_images=False,
        ),
        tip=widget.tr("Write train/val/test/all.txt lists by ratio"),
    )
    batch_check = action(
        widget.tr("Batch Annotation Check"),
        widget.toggle_annotation_inspector,
        tip=widget.tr(
            "Build/browse crop cache and highlight annotations (batch QA)"
        ),
        checkable=True,
        enabled=True,
    )
    label_dedup = action(
        widget.tr("Label Deduplication…"),
        lambda: _guarded(
            widget,
            "Label Deduplication",
            lambda w: __import__(
                "anylabeling.views.labeling.advanced.ops.label_dedup_dialog",
                fromlist=["open_label_dedup_dialog"],
            ).open_label_dedup_dialog(w),
        ),
        tip=widget.tr("Remove same-class high-IoU duplicate boxes"),
    )
    dataset_rename = action(
        widget.tr("Dataset Rename…"),
        lambda: _guarded(
            widget,
            "Dataset Rename",
            lambda w: __import__(
                "anylabeling.views.labeling.advanced.ops.dataset_rename_dialog",
                fromlist=["open_dataset_rename_dialog"],
            ).open_dataset_rename_dialog(w),
        ),
        tip=widget.tr("Batch-rename images and sync JSON / imagePath"),
    )
    annotation_compare = action(
        widget.tr("Annotation Compare…"),
        widget.open_annotation_compare,
        tip=widget.tr("Compare GT JSON with YOLO predictions"),
    )
    model_eval = action(
        widget.tr("Model Evaluation…"),
        widget.open_model_eval,
        tip=widget.tr("Mark false negatives / false positives vs GT"),
    )
    clear_compare = action(
        widget.tr("Clear Compare Results…"),
        widget.open_clear_annotation_compare_results,
        tip=widget.tr("Clear annotation-compare and model-eval marks"),
    )
    sticker = action(
        widget.tr("Sticker"),
        widget.open_sticker_mode,
        tip=widget.tr("Auto overlay materials to address insufficient samples"),
    )
    anno_filter = action(
        widget.tr("Annotation Filter…"),
        lambda: _guarded(
            widget,
            "Annotation Filter",
            lambda w: __import__(
                "anylabeling.views.labeling.advanced.ops.annotation_filter_dialog",
                fromlist=["AnnotationFilterDialog"],
            ).AnnotationFilterDialog(w, image_paths=list(w.image_list)).exec(),
        ),
        tip=widget.tr("Move images/JSON by bounding-box size thresholds"),
    )
    label_jump = action(
        widget.tr("Jump by Labels…"),
        lambda: _open_label_jump(widget),
        tip=widget.tr(
            "Select labels and jump to previous/next image containing any of them"
        ),
    )

    try:
        widget.actions.batch_annotation_check = batch_check
    except Exception:
        pass

    utils.add_actions(
        widget.menus.advanced,
        (
            open_stats,
            dataset_split,
            batch_check,
            label_dedup,
            dataset_rename,
            annotation_compare,
            model_eval,
            clear_compare,
            sticker,
            anno_filter,
            label_jump,
        ),
    )
