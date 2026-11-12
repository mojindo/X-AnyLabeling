"""Advanced dialog: jump to prev/next image by selected labels."""

from __future__ import annotations

import os
import shutil
from typing import List, Sequence

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from anylabeling.views.labeling.classifier.utils import get_label_file_path
from anylabeling.views.labeling.advanced.ops.label_jump import (
    collect_available_labels_with_counts,
    find_image_with_labels,
    fuzzy_match_label,
)
from anylabeling.views.labeling.utils.style import (
    get_dialog_style,
    get_ok_btn_style,
    get_cancel_btn_style,
    get_lineedit_style,
)


class LabelJumpDialog(QDialog):
    """Non-modal panel to jump images by multi-selected labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Jump by Labels"))
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setModal(False)
        self.setMinimumSize(0, 0)
        self.setStyleSheet(get_dialog_style())

        self._parent = parent
        self._build_ui()
        self.refresh_labels()

    def closeEvent(self, event) -> None:
        self._clear_jump_highlight()
        super().closeEvent(event)

    def _apply_jump_highlight(self) -> None:
        """Select matching shapes so they show the same semi-transparent fill as selection."""
        parent = self._parent
        if parent is None:
            return
        canvas = getattr(parent, "canvas", None)
        if canvas is None:
            return
        labels = set(self.selected_labels())
        matching = [
            s
            for s in (getattr(canvas, "shapes", None) or [])
            if str(getattr(s, "label", "") or "").strip() in labels
        ]
        parent._label_jump_highlight_active = True
        canvas.select_shapes(matching)

    def _clear_jump_highlight(self) -> None:
        """Restore shapes by clearing canvas selection from label-jump highlight."""
        parent = self._parent
        if parent is None:
            return
        if not getattr(parent, "_label_jump_highlight_active", False):
            return
        parent._label_jump_highlight_active = False
        canvas = getattr(parent, "canvas", None)
        if canvas is not None:
            canvas.deselect_shape()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(
            QtWidgets.QLayout.SizeConstraint.SetNoConstraint
        )
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        tip = QLabel(
            self.tr(
                "Select one or more labels. Next / Previous jumps to an image "
                "that contains any of the selected labels."
            )
        )
        tip.setWordWrap(True)
        layout.addWidget(tip)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel(self.tr("Search:")))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(
            self.tr("Fuzzy filter labels (e.g. car, c a r, red car)…")
        )
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setStyleSheet(get_lineedit_style())
        self.search_edit.textChanged.connect(self._apply_label_filter)
        search_row.addWidget(self.search_edit, 1)
        layout.addLayout(search_row)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.list_widget.setUniformItemSizes(True)
        layout.addWidget(self.list_widget, 1)

        select_row = QHBoxLayout()
        self.select_all_btn = QPushButton(self.tr("Select All"))
        self.select_all_btn.setStyleSheet(get_cancel_btn_style())
        self.select_all_btn.clicked.connect(self._select_all)
        self.clear_btn = QPushButton(self.tr("Clear"))
        self.clear_btn.setStyleSheet(get_cancel_btn_style())
        self.clear_btn.clicked.connect(self._clear_selection)
        self.refresh_btn = QPushButton(self.tr("Refresh Labels"))
        self.refresh_btn.setStyleSheet(get_cancel_btn_style())
        self.refresh_btn.setToolTip(
            self.tr(
                "Reload labels from the dataset and jump the main view "
                "to the first image."
            )
        )
        self.refresh_btn.clicked.connect(self._on_refresh_clicked)
        select_row.addWidget(self.select_all_btn)
        select_row.addWidget(self.clear_btn)
        select_row.addWidget(self.refresh_btn)
        select_row.addStretch()
        layout.addLayout(select_row)

        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton(self.tr("Previous"))
        self.prev_btn.setStyleSheet(get_ok_btn_style())
        self.prev_btn.clicked.connect(lambda: self._jump(-1))
        self.next_btn = QPushButton(self.tr("Next"))
        self.next_btn.setStyleSheet(get_ok_btn_style())
        self.next_btn.clicked.connect(lambda: self._jump(1))
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.next_btn)
        layout.addLayout(nav_row)

        copy_row = QHBoxLayout()
        copy_row.addWidget(QLabel(self.tr("Copy to:")))
        self.copy_dir_edit = QLineEdit()
        self.copy_dir_edit.setPlaceholderText(
            self.tr("Output folder for copied image and label…")
        )
        self.copy_dir_edit.setClearButtonEnabled(True)
        self.copy_dir_edit.setStyleSheet(get_lineedit_style())
        saved_dir = ""
        if self._parent is not None:
            saved_dir = str(
                getattr(self._parent, "_label_jump_copy_dir", "") or ""
            )
        if saved_dir:
            self.copy_dir_edit.setText(saved_dir)
        self.browse_copy_dir_btn = QPushButton(self.tr("Browse…"))
        self.browse_copy_dir_btn.setStyleSheet(get_cancel_btn_style())
        self.browse_copy_dir_btn.clicked.connect(self._browse_copy_dir)
        copy_row.addWidget(self.copy_dir_edit, 1)
        copy_row.addWidget(self.browse_copy_dir_btn)
        layout.addLayout(copy_row)

        copy_btn_row = QHBoxLayout()
        self.copy_btn = QPushButton(self.tr("Copy Current"))
        self.copy_btn.setStyleSheet(get_ok_btn_style())
        self.copy_btn.setToolTip(
            self.tr(
                "Copy the current image and its annotation JSON to the "
                "selected output folder."
            )
        )
        self.copy_btn.clicked.connect(self._copy_current)
        copy_btn_row.addWidget(self.copy_btn)
        copy_btn_row.addStretch()
        layout.addLayout(copy_btn_row)

        self.status_label = QLabel(self.tr("Select at least one label."))
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _on_refresh_clicked(self) -> None:
        self.refresh_labels(relocate_to_first=True)

    def refresh_labels(self, *, relocate_to_first: bool = False) -> None:
        previous = set(self.selected_labels())
        parent = self._parent
        seed: List[str] = []
        if parent is not None and hasattr(parent, "unique_label_list"):
            ulist = parent.unique_label_list
            for row in range(ulist.count()):
                item = ulist.item(row)
                label = item.data(Qt.ItemDataRole.UserRole)
                if label is not None and str(label).strip():
                    seed.append(str(label).strip())

        image_list = list(getattr(parent, "image_list", []) or [])
        output_dir = getattr(parent, "output_dir", None)
        labels_with_counts = collect_available_labels_with_counts(
            image_list, output_dir=output_dir, seed_labels=seed
        )

        self.list_widget.clear()
        for name, count in labels_with_counts:
            item = QListWidgetItem(f"{name} ({count})")
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(
                Qt.CheckState.Checked
                if name in previous
                else Qt.CheckState.Unchecked
            )
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.list_widget.addItem(item)

        relocated = False
        if relocate_to_first and parent is not None and image_list:
            relocated = self._relocate_to_first_image(image_list)

        self._apply_label_filter()
        visible = self._visible_count()
        n_labels = len(labels_with_counts)
        if relocated:
            self.status_label.setText(
                self.tr(
                    "Loaded {n} label(s), showing {m}. Relocated to the first image."
                ).format(n=n_labels, m=visible)
            )
        else:
            self.status_label.setText(
                self.tr(
                    "Loaded {n} label(s), showing {m}. Select at least one."
                ).format(n=n_labels, m=visible)
            )

    def _apply_label_filter(self, _text: str = "") -> None:
        query = (self.search_edit.text() if hasattr(self, "search_edit") else "") or ""
        visible = 0
        total = self.list_widget.count()
        for row in range(total):
            item = self.list_widget.item(row)
            name = str(item.data(Qt.ItemDataRole.UserRole) or item.text())
            matched = fuzzy_match_label(name, query)
            item.setHidden(not matched)
            if matched:
                visible += 1
        if query.strip():
            self.status_label.setText(
                self.tr("Filter “{q}”: {m}/{n} label(s).").format(
                    q=query.strip(), m=visible, n=total
                )
            )
        elif total > 0:
            self.status_label.setText(
                self.tr(
                    "Loaded {n} label(s), showing {m}. Select at least one."
                ).format(n=total, m=visible)
            )

    def _visible_count(self) -> int:
        n = 0
        for row in range(self.list_widget.count()):
            if not self.list_widget.item(row).isHidden():
                n += 1
        return n

    def _relocate_to_first_image(self, image_list: Sequence[str]) -> bool:
        """Jump the main window to the first image in the dataset."""
        parent = self._parent
        if parent is None or not image_list:
            return False
        first = str(image_list[0])
        current = str(getattr(parent, "filename", "") or "")
        if current and os.path.normpath(current) == os.path.normpath(first):
            return True

        ensure = getattr(parent, "_ensure_can_leave_current_image", None)
        if callable(ensure) and not ensure(first):
            return False
        may_continue = getattr(parent, "may_continue", None)
        if callable(may_continue) and not may_continue():
            return False

        parent.load_file(first)
        return True

    def selected_labels(self) -> List[str]:
        out: List[str] = []
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                name = item.data(Qt.ItemDataRole.UserRole) or item.text()
                text = str(name).strip()
                if text:
                    out.append(text)
        return out

    def _select_all(self) -> None:
        """Check all currently visible (filter-matched) labels."""
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if not item.isHidden():
                item.setCheckState(Qt.CheckState.Checked)

    def _clear_selection(self) -> None:
        """Uncheck all currently visible (filter-matched) labels."""
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if not item.isHidden():
                item.setCheckState(Qt.CheckState.Unchecked)

    def _jump(self, direction: int) -> None:
        labels = self.selected_labels()
        if not labels:
            QMessageBox.warning(
                self,
                self.tr("Jump by Labels"),
                self.tr("Please select at least one label."),
            )
            return

        parent = self._parent
        if parent is None:
            return
        image_list = list(getattr(parent, "image_list", []) or [])
        if not image_list:
            QMessageBox.information(
                self,
                self.tr("Jump by Labels"),
                self.tr("Please open an image folder first."),
            )
            return

        target = find_image_with_labels(
            image_list,
            current_path=getattr(parent, "filename", None),
            labels=labels,
            output_dir=getattr(parent, "output_dir", None),
            direction=direction,
        )
        if not target:
            msg = (
                self.tr("No later image contains the selected label(s).")
                if direction >= 0
                else self.tr(
                    "No earlier image contains the selected label(s)."
                )
            )
            self.status_label.setText(msg)
            QMessageBox.information(self, self.tr("Jump by Labels"), msg)
            return

        ensure = getattr(parent, "_ensure_can_leave_current_image", None)
        if callable(ensure) and not ensure(target):
            return
        may_continue = getattr(parent, "may_continue", None)
        if callable(may_continue) and not may_continue():
            return

        # load_file resets selection; clear flag so exit won't deselect a stale set
        if parent is not None:
            parent._label_jump_highlight_active = False
        parent.load_file(target)
        self._apply_jump_highlight()
        self.status_label.setText(
            self.tr("Jumped to: {name}").format(
                name=os.path.basename(target)
            )
        )

    def _browse_copy_dir(self) -> None:
        start = (self.copy_dir_edit.text() or "").strip()
        if not start or not os.path.isdir(start):
            parent = self._parent
            filename = getattr(parent, "filename", None) if parent else None
            if filename:
                start = os.path.dirname(str(filename))
            else:
                start = os.getcwd()
        path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Copy Output Folder"),
            start,
            QFileDialog.Option.ShowDirsOnly
            | QFileDialog.Option.DontResolveSymlinks,
        )
        if path:
            self.copy_dir_edit.setText(path)
            if self._parent is not None:
                self._parent._label_jump_copy_dir = path

    def _copy_current(self) -> None:
        dest_dir = (self.copy_dir_edit.text() or "").strip()
        if not dest_dir:
            QMessageBox.warning(
                self,
                self.tr("Jump by Labels"),
                self.tr("Please select an output folder first."),
            )
            return
        if not os.path.isdir(dest_dir):
            QMessageBox.warning(
                self,
                self.tr("Jump by Labels"),
                self.tr("Output folder does not exist:\n{path}").format(
                    path=dest_dir
                ),
            )
            return

        parent = self._parent
        if parent is None:
            return
        image_path = str(getattr(parent, "filename", "") or "")
        if not image_path or not os.path.isfile(image_path):
            QMessageBox.information(
                self,
                self.tr("Jump by Labels"),
                self.tr("No current image to copy."),
            )
            return

        # Persist latest choice
        parent._label_jump_copy_dir = dest_dir

        # Flush unsaved annotations so the JSON on disk matches the canvas
        dirty = bool(getattr(parent, "dirty", False))
        if dirty:
            save_file = getattr(parent, "save_file", None)
            if callable(save_file):
                try:
                    save_file()
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        self.tr("Jump by Labels"),
                        self.tr(
                            "Failed to save current annotations before copy:\n{err}"
                        ).format(err=str(e)),
                    )
                    return

        label_path = get_label_file_path(
            image_path, getattr(parent, "output_dir", None)
        )
        img_name = os.path.basename(image_path)
        label_name = os.path.basename(label_path)
        img_dst = os.path.join(dest_dir, img_name)
        label_dst = os.path.join(dest_dir, label_name)

        try:
            shutil.copy2(image_path, img_dst)
            label_copied = False
            if os.path.isfile(label_path):
                shutil.copy2(label_path, label_dst)
                label_copied = True
        except OSError as e:
            QMessageBox.critical(
                self,
                self.tr("Jump by Labels"),
                self.tr("Copy failed:\n{err}").format(err=str(e)),
            )
            return

        if label_copied:
            msg = self.tr(
                "Copied image and label to:\n{dir}\n• {img}\n• {label}"
            ).format(dir=dest_dir, img=img_name, label=label_name)
        else:
            msg = self.tr(
                "Copied image to:\n{dir}\n• {img}\n"
                "(No annotation JSON found for this image.)"
            ).format(dir=dest_dir, img=img_name)
        self.status_label.setText(msg)


def open_label_jump_dialog(parent) -> LabelJumpDialog:
    """Open or raise a singleton non-modal LabelJumpDialog on parent."""
    existing = getattr(parent, "_label_jump_dialog", None)
    if existing is not None:
        try:
            if existing.isVisible():
                existing.raise_()
                existing.activateWindow()
                saved_dir = str(
                    getattr(parent, "_label_jump_copy_dir", "") or ""
                )
                if saved_dir and hasattr(existing, "copy_dir_edit"):
                    existing.copy_dir_edit.setText(saved_dir)
                existing.refresh_labels()
                return existing
        except RuntimeError:
            pass

    dlg = LabelJumpDialog(parent)
    parent._label_jump_dialog = dlg
    dlg.show()
    return dlg
