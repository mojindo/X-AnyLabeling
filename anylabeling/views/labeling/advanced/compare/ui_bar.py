"""Compare / model-eval review toolbar inserted into LabelingWidget layout."""

from __future__ import annotations

from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QWidget

from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_checkbox_indicator_style,
    get_ok_btn_style,
)


def _nav_btn(widget, text: str, tip: str, slot) -> QPushButton:
    btn = QPushButton(widget.tr(text))
    btn.setToolTip(widget.tr(tip))
    btn.setStyleSheet(get_cancel_btn_style())
    btn.clicked.connect(slot)
    btn.hide()
    return btn


def install_compare_ui_bar(widget) -> None:
    """
    Insert browse / accept / clear / prev-next FN/FP / clear-dataset controls
    above the canvas. Idempotent.
    """
    if getattr(widget, "_compare_ui_bar_installed", False):
        return

    # Ensure instruction label exists (created by LabelingWidget, often hidden).
    if not hasattr(widget, "label_instruction") or widget.label_instruction is None:
        widget.label_instruction = QLabel(widget.get_labeling_instruction(), widget)
        widget.label_instruction.setContentsMargins(0, 0, 0, 0)

    widget._compare_browse_mode_cb = QCheckBox(widget.tr("Browse Mode"))
    widget._compare_browse_mode_cb.setToolTip(
        widget.tr(
            "Browse mode: keep unreviewed suspect marks and switch images "
            "without confirmation. Uncheck to require confirmation before switching."
        )
    )
    try:
        widget._compare_browse_mode_cb.setStyleSheet(
            get_checkbox_indicator_style()
        )
    except Exception:
        pass
    widget._compare_browse_mode_cb.setChecked(
        bool(getattr(widget, "_compare_browse_mode", False))
    )
    widget._compare_browse_mode_cb.toggled.connect(
        widget._on_compare_browse_mode_toggled
    )
    widget._compare_browse_mode_cb.hide()

    widget._compare_accept_all_btn = QPushButton(widget.tr("Accept This Check"))
    widget._compare_accept_all_btn.setToolTip(
        widget.tr(
            "Accept all suspect marks on this image: keep missed-label boxes "
            "as annotations; keep false-label boxes and clear their flags."
        )
    )
    widget._compare_accept_all_btn.setStyleSheet(get_ok_btn_style())
    widget._compare_accept_all_btn.clicked.connect(
        widget.accept_compare_results_current_image
    )
    widget._compare_accept_all_btn.hide()

    widget._compare_clear_image_btn = QPushButton(widget.tr("Clear This Check"))
    widget._compare_clear_image_btn.setToolTip(
        widget.tr(
            "Clear all suspect marks on this image: delete missed-label / "
            "false-alarm boxes; keep false-label / miss boxes and remove flags."
        )
    )
    widget._compare_clear_image_btn.setStyleSheet(get_cancel_btn_style())
    widget._compare_clear_image_btn.clicked.connect(
        widget.clear_compare_results_current_image
    )
    widget._compare_clear_image_btn.hide()

    widget._compare_fn_prev_btn = _nav_btn(
        widget,
        "Prev Missed",
        "Open previous image with suspect missed labels (annotation compare)",
        widget.open_prev_compare_fn_image,
    )
    widget._compare_fn_nav_btn = _nav_btn(
        widget,
        "Next Missed",
        "Open next image with suspect missed labels (annotation compare)",
        widget.open_next_compare_fn_image,
    )
    widget._compare_fp_prev_btn = _nav_btn(
        widget,
        "Prev False-Label",
        "Open previous image with suspect false labels (annotation compare)",
        widget.open_prev_compare_fp_image,
    )
    widget._compare_fp_nav_btn = _nav_btn(
        widget,
        "Next False-Label",
        "Open next image with suspect false labels (annotation compare)",
        widget.open_next_compare_fp_image,
    )

    widget._compare_clear_dataset_btn = QPushButton(
        widget.tr("Clear Dataset Checks")
    )
    widget._compare_clear_dataset_btn.setToolTip(
        widget.tr(
            "Clear compare / model-eval marks on the whole dataset "
            "(same as Advanced → Clear Compare Results)."
        )
    )
    widget._compare_clear_dataset_btn.setStyleSheet(get_cancel_btn_style())
    widget._compare_clear_dataset_btn.clicked.connect(
        widget.clear_compare_results_dataset
    )
    widget._compare_clear_dataset_btn.hide()

    row = QHBoxLayout()
    row.setContentsMargins(8, 2, 8, 2)
    row.setSpacing(6)
    row.addWidget(widget.label_instruction, 1)
    row.addWidget(widget._compare_browse_mode_cb)
    row.addWidget(widget._compare_accept_all_btn)
    row.addWidget(widget._compare_clear_image_btn)
    row.addWidget(widget._compare_fn_prev_btn)
    row.addWidget(widget._compare_fn_nav_btn)
    row.addWidget(widget._compare_fp_prev_btn)
    row.addWidget(widget._compare_fp_nav_btn)
    row.addWidget(widget._compare_clear_dataset_btn)

    bar = QWidget()
    bar.setObjectName("AdvancedCompareReviewBar")
    bar.setLayout(row)
    bar.hide()
    widget._compare_review_bar = bar

    # Insert above auto_labeling_widget in the central column.
    # Never create/replace LabelingWidget's own layout — that wipes the toolbar.
    inserted = False
    try:
        auto = getattr(widget, "auto_labeling_widget", None)
        parent = auto.parentWidget() if auto is not None else None
        layout = parent.layout() if parent is not None else None
        if layout is not None and auto is not None:
            idx = layout.indexOf(auto)
            if idx >= 0:
                layout.insertWidget(idx, bar)
            else:
                layout.addWidget(bar)
            inserted = True
    except Exception:
        inserted = False

    if not inserted:
        # Do not mark installed — allow a later retry once layout exists.
        bar.setParent(None)
        from anylabeling.views.labeling.logger import logger

        logger.warning(
            "Compare review bar was not inserted: central layout not ready"
        )
        return

    widget._compare_ui_bar_installed = True


def sync_compare_review_bar_visibility(widget) -> None:
    """Show/hide the whole review bar with compare-active state."""
    bar = getattr(widget, "_compare_review_bar", None)
    active = bool(getattr(widget, "_annotation_compare_active", False))
    if bar is not None:
        bar.setVisible(active)
    instr = getattr(widget, "label_instruction", None)
    if instr is not None and active:
        instr.show()
    elif instr is not None and not active:
        # Keep hidden when not reviewing (matches upstream default).
        instr.hide()
