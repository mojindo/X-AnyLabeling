# -*- encoding: utf-8 -*-

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt

from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_ok_btn_style,
)


class LabelFilterDialog(QtWidgets.QDialog):
    """Pick label categories to keep; unselected labels are removed dataset-wide."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_widget = parent
        self._checkboxes = []
        self.setWindowTitle(self.tr("Filter Labels"))
        self.setMinimumSize(420, 360)
        self.setStyleSheet(get_dialog_style())
        self._build_ui()
        self._populate_labels()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            self.tr(
                "Checked labels will be kept. Unchecked labels and all of "
                "their annotations will be removed from the entire dataset."
            )
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._list_host = QtWidgets.QWidget()
        self._list_layout = QtWidgets.QVBoxLayout(self._list_host)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.addStretch(1)
        self._scroll.setWidget(self._list_host)
        layout.addWidget(self._scroll, 1)

        select_row = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton(self.tr("Select All"))
        select_all_btn.setStyleSheet(get_ok_btn_style())
        select_all_btn.clicked.connect(self._select_all)
        deselect_all_btn = QtWidgets.QPushButton(self.tr("Deselect All"))
        deselect_all_btn.setStyleSheet(get_cancel_btn_style())
        deselect_all_btn.clicked.connect(self._deselect_all)
        select_row.addWidget(select_all_btn)
        select_row.addWidget(deselect_all_btn)
        select_row.addStretch(1)
        layout.addLayout(select_row)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        cancel_btn = QtWidgets.QPushButton(self.tr("Cancel"))
        cancel_btn.setStyleSheet(get_cancel_btn_style())
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QtWidgets.QPushButton(self.tr("OK"))
        ok_btn.setStyleSheet(get_ok_btn_style())
        ok_btn.clicked.connect(self._on_accept)
        button_row.addWidget(cancel_btn)
        button_row.addWidget(ok_btn)
        layout.addLayout(button_row)

    def _populate_labels(self):
        labels = self.parent_widget.collect_label_manager_label_names()
        for label in labels:
            row = QtWidgets.QHBoxLayout()
            checkbox = QtWidgets.QCheckBox(label)
            checkbox.setChecked(True)
            if self.parent_widget.unique_label_list.lock_states.get(
                label, False
            ):
                checkbox.setToolTip(
                    self.tr(
                        "This label is locked; its annotations cannot be "
                        "removed until the lock is cleared."
                    )
                )
            row.addWidget(checkbox)
            self._list_layout.insertLayout(
                self._list_layout.count() - 1, row
            )
            self._checkboxes.append((label, checkbox))

    def _select_all(self):
        for _, checkbox in self._checkboxes:
            checkbox.setChecked(True)

    def _deselect_all(self):
        for _, checkbox in self._checkboxes:
            checkbox.setChecked(False)

    def _on_accept(self):
        if not self.selected_labels():
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Filter Labels"),
                self.tr("Select at least one label to keep."),
            )
            return
        self.accept()

    def selected_labels(self):
        return {
            label
            for label, checkbox in self._checkboxes
            if checkbox.isChecked()
        }

    def unchecked_labels(self):
        return {
            label
            for label, checkbox in self._checkboxes
            if not checkbox.isChecked()
        }