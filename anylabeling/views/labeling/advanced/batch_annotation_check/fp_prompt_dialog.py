"""Dialog: map annotation labels to Chinese-CLIP text phrases."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from PyQt6 import QtCore, QtWidgets

from anylabeling.views.labeling.advanced.batch_annotation_check.checks.clip_fp import (
    DEFAULT_LABEL_PROMPTS,
)


class FpLabelPromptDialog(QtWidgets.QDialog):
    """Edit label -> Chinese phrase mapping used by FP check."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        labels: Iterable[str],
        existing: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Chinese-CLIP label phrases"))
        self.setMinimumSize(480, 360)

        existing = dict(existing or {})
        labels_sorted = sorted(
            {str(x).strip() for x in labels if str(x).strip()},
            key=lambda s: s.lower(),
        )

        root = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            self.tr(
                "Map each annotation label to a short Chinese phrase for "
                "Chinese-CLIP. Example: person → 人. Leave blank to use the "
                "label text as-is."
            )
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        root.addWidget(hint)

        self._table = QtWidgets.QTableWidget(len(labels_sorted), 2)
        self._table.setHorizontalHeaderLabels(
            [self.tr("Label"), self.tr("Chinese phrase")]
        )
        self._table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )

        for row, label in enumerate(labels_sorted):
            label_item = QtWidgets.QTableWidgetItem(label)
            label_item.setFlags(
                label_item.flags()
                & ~QtCore.Qt.ItemFlag.ItemIsEditable
            )
            self._table.setItem(row, 0, label_item)

            phrase = existing.get(label) or existing.get(label.lower())
            if not phrase:
                phrase = DEFAULT_LABEL_PROMPTS.get(label.lower(), "")
            phrase_item = QtWidgets.QTableWidgetItem(str(phrase or ""))
            self._table.setItem(row, 1, phrase_item)

        root.addWidget(self._table, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def mapping(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for row in range(self._table.rowCount()):
            label_item = self._table.item(row, 0)
            phrase_item = self._table.item(row, 1)
            if label_item is None:
                continue
            label = label_item.text().strip()
            phrase = (phrase_item.text() if phrase_item else "").strip()
            if label and phrase:
                out[label] = phrase
        return out
