"""Optional export guard: block export while unreviewed compare/eval marks exist."""

from __future__ import annotations

from typing import Any


def check_export_no_compare_marks(labeling_widget: Any) -> bool:
    """
    Return True if export may proceed.
    When advanced compare marks remain, show a message and return False.
    """
    try:
        from anylabeling.views.labeling.advanced.compare.annotation_compare_io import (
            collect_images_with_compare_marks,
        )
    except Exception:
        return True

    image_list = getattr(labeling_widget, "image_list", None) or []
    if not image_list:
        return True
    try:
        marked = collect_images_with_compare_marks(
            image_list,
            output_dir=getattr(labeling_widget, "output_dir", None),
        )
    except Exception:
        return True
    if not marked:
        return True

    from PyQt6.QtWidgets import QMessageBox

    QMessageBox.warning(
        labeling_widget,
        labeling_widget.tr("Export"),
        labeling_widget.tr(
            "Cannot export: %d image(s) still have unreviewed compare / "
            "model-eval marks. Finish review or use "
            "Advanced → Clear Compare Results first."
        )
        % len(marked),
    )
    return False
