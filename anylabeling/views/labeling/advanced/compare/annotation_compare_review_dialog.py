"""Review dialog for suspect FN/FP shapes after annotation compare."""

from __future__ import annotations

from typing import List, Optional

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

from anylabeling.views.labeling.advanced.compare.annotation_compare import (
    FN_ACCEPT_LABEL,
    FN_ACCEPT_NEW_LABEL,
    FN_REJECT,
    FP_CONFIRM_DELETE,
    FP_CONFIRM_RELABEL,
    FP_KEEP,
    PENDING_GT_STATUSES,
    PENDING_PRED_STATUSES,
    STATUS_MODEL_FALSE,
    STATUS_MODEL_MISS,
    STATUS_SUSPECT_FN,
    STATUS_SUSPECT_FP,
)
from anylabeling.views.labeling.utils.style import get_dialog_style


class AnnotationCompareReviewDialog(QDialog):
    """Modal choices for one suspect shape."""

    _last_action_by_status: dict = {}

    @classmethod
    def last_action_for(cls, status: str) -> Optional[str]:
        return cls._last_action_by_status.get(status)

    @classmethod
    def remember_action(cls, status: str, action: str) -> None:
        if action:
            cls._last_action_by_status[status] = action

    def __init__(
        self,
        parent,
        *,
        status: str,
        label: str,
        class_names: Optional[List[str]] = None,
        default_action: Optional[str] = None,
    ):
        super().__init__(parent)
        self._status = status
        self._label = label
        self._class_names = list(class_names or [])
        self._default_action = default_action
        self._action: Optional[str] = None
        self._new_label: Optional[str] = None

        self.setWindowTitle(self.tr("Annotation Compare Review"))
        self.setMinimumWidth(420)
        self.setStyleSheet(get_dialog_style())
        self._build_ui()

    def action(self) -> Optional[str]:
        return self._action

    def new_label(self) -> Optional[str]:
        return self._new_label

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        if self._status in PENDING_PRED_STATUSES:
            if self._status == STATUS_MODEL_FALSE:
                title = self.tr("False alarm (误报)")
            else:
                title = self.tr("Suspect missed label (疑是漏标)")
            layout.addWidget(QLabel(f"<b>{title}</b> — {self._label}"))
            self._group = QButtonGroup(self)
            opts = [
                (self.tr("Accept with current label"), FN_ACCEPT_LABEL),
                (self.tr("Accept with a different label"), FN_ACCEPT_NEW_LABEL),
                (self.tr("Reject (delete this box)"), FN_REJECT),
            ]
        else:
            if self._status == STATUS_MODEL_MISS:
                title = self.tr("Miss (漏报)")
            else:
                title = self.tr("Suspect false label (疑是误标)")
            layout.addWidget(QLabel(f"<b>{title}</b> — {self._label}"))
            self._group = QButtonGroup(self)
            opts = [
                (self.tr("Confirm false label — delete box"), FP_CONFIRM_DELETE),
                (self.tr("Confirm false label — change label"), FP_CONFIRM_RELABEL),
                (self.tr("Not false label — keep as-is"), FP_KEEP),
            ]

        self._relabel_action = (
            FN_ACCEPT_NEW_LABEL
            if self._status in PENDING_PRED_STATUSES
            else FP_CONFIRM_RELABEL
        )
        for i, (text, code) in enumerate(opts):
            rb = QRadioButton(text)
            rb.setProperty("action_code", code)
            self._group.addButton(rb, i)
            layout.addWidget(rb)

        self._label_combo: Optional[QComboBox] = QComboBox()
        self._label_combo.setEditable(True)
        if self._class_names:
            self._label_combo.addItems(self._class_names)
        self._label_combo.setCurrentText(self._label)
        layout.addWidget(QLabel(self.tr("New label (when changing label):")))
        layout.addWidget(self._label_combo)

        default_action = self._default_action
        default_set = False
        if default_action:
            for btn in self._group.buttons():
                if btn.property("action_code") == default_action:
                    btn.setChecked(True)
                    default_set = True
                    break
        if not default_set and self._group.buttons():
            self._group.buttons()[0].setChecked(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        btn = self._group.checkedButton()
        if btn is None:
            return
        self._action = btn.property("action_code")
        if self._action in (FN_ACCEPT_NEW_LABEL, FP_CONFIRM_RELABEL):
            self._new_label = self._label_combo.currentText().strip()
            if not self._new_label:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Annotation Compare"),
                    self.tr("Please enter a label name."),
                )
                return
        self.accept()
