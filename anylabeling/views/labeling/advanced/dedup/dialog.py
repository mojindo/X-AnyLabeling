"""Scan the open dataset for near-duplicates, then delete or review."""

from __future__ import annotations

import os
import os.path as osp
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from anylabeling.views.labeling.advanced.dedup.config import Config
from anylabeling.views.labeling.advanced.dedup.io_utils import records_from_paths
from anylabeling.views.labeling.advanced.dedup.pipeline import ScanResult, run_records
from anylabeling.views.labeling.utils.style import (
    get_dialog_style,
    get_ok_btn_style,
    get_progress_dialog_style,
    get_spinbox_style,
)


# G4 metric keys aligned with QualityConfig defaults (rel_tol as fraction).
_G4_METRIC_UI: tuple[tuple[str, str, float], ...] = (
    ("sharpness_laplacian_var_log1p", "Sharpness", 0.20),
    ("brightness_l_mean", "Brightness (LAB L)", 0.20),
    ("contrast_rms_gray", "Contrast", 0.20),
    ("saturation_mean", "Saturation", 0.25),
    ("value_mean", "Value (HSV V)", 0.20),
    ("noise_mad_residual", "Noise", 0.20),
)


def _label_path(parent, image_path: str) -> Optional[str]:
    side = osp.splitext(image_path)[0] + ".json"
    output_dir = getattr(parent, "output_dir", None)
    if output_dir:
        alt = osp.join(output_dir, osp.basename(side))
        if osp.isfile(alt):
            return alt
    if osp.isfile(side):
        return side
    return None


def _annotation_paths(parent, image_path: str) -> list[str]:
    side = osp.splitext(image_path)[0] + ".json"
    found: list[str] = []
    if osp.isfile(side):
        found.append(side)
    output_dir = getattr(parent, "output_dir", None)
    if output_dir:
        alt = osp.join(output_dir, osp.basename(side))
        if osp.isfile(alt) and osp.normcase(alt) not in {osp.normcase(p) for p in found}:
            found.append(alt)
    return found


class _ScanWorker(QtCore.QThread):
    progressed = QtCore.pyqtSignal(str)
    succeeded = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, cfg: Config, image_paths: list[str], parent_widget) -> None:
        super().__init__()
        self._cfg = cfg
        self._image_paths = image_paths
        self._parent_widget = parent_widget

    def run(self) -> None:
        try:
            recs = records_from_paths(
                self._image_paths, lambda p: _label_path(self._parent_widget, p)
            )
            result = run_records(self._cfg, recs, progress=self.progressed.emit)
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class _GroupCard(QGroupBox):
    def __init__(self, title: str, paths: list[str], parent=None) -> None:
        super().__init__(title, parent)
        self.paths = list(paths)
        self.checks: list[QCheckBox] = []
        row = QHBoxLayout(self)
        row.setSpacing(8)
        for index, path in enumerate(self.paths):
            cell = QWidget()
            lay = QVBoxLayout(cell)
            lay.setContentsMargins(0, 0, 0, 0)
            image = QLabel()
            pix = QtGui.QPixmap(path)
            if not pix.isNull():
                image.setPixmap(
                    pix.scaled(140, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                )
            image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(image)
            box = QCheckBox(osp.basename(path))
            box.setChecked(index == 0)
            box.setToolTip(path)
            self.checks.append(box)
            lay.addWidget(box)
            row.addWidget(cell)
        row.addStretch(1)

    def checked_paths(self) -> list[str]:
        return [path for path, box in zip(self.paths, self.checks) if box.isChecked()]


class DatasetDedupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Dataset Dedup"))
        self.setMinimumSize(880, 640)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setStyleSheet(get_dialog_style())
        self._host = parent
        self._worker: Optional[_ScanWorker] = None
        self._progress: Optional[QtWidgets.QProgressDialog] = None
        self._result: Optional[ScanResult] = None
        self._cards: list[_GroupCard] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(8)

        hint = QLabel(
            self.tr(
                "Scan the images currently open in this folder. "
                "A duplicate group keeps the pictures you choose and deletes the rest, with their JSON files."
            )
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        settings_bar = QHBoxLayout()
        self._settings_toggle = QPushButton(self.tr("Collapse settings"))
        self._settings_toggle.setCheckable(True)
        self._settings_toggle.setChecked(True)
        self._settings_toggle.toggled.connect(self._on_settings_toggled)
        settings_bar.addWidget(self._settings_toggle)
        settings_bar.addStretch(1)
        root.addLayout(settings_bar)

        self._settings_panel = QWidget()
        settings_layout = QVBoxLayout(self._settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(8)

        cond = QGroupBox(self.tr("Conditions"))
        cond_layout = QVBoxLayout(cond)
        cond_intro = QLabel(
            self.tr(
                "Scan runs in stages: scene match (overlap), then camera motion, then instances, then optional quality. "
                "Loosening a limit keeps more pairs as duplicates. Tightening drops more pairs. "
                "Scene motion limits apply only after a pair already passes scene matching."
            )
        )
        cond_intro.setWordWrap(True)
        cond_intro.setStyleSheet("color: #666;")
        cond_layout.addWidget(cond_intro)
        form = QtWidgets.QFormLayout()
        self._shift = self._spin(0.02)
        self._scale = self._spin(0.02)
        self._rotation = self._spin(0.03)
        self._true_shift = self._spin(0.03)
        self._box_scale = self._spin(0.20)
        self._pose = self._spin(0.20)
        self._overlap = self._spin(0.35, maximum=1.0)
        self._add_condition_row(
            form,
            self.tr("Max scene shift:"),
            self._shift,
            self.tr("of shorter image side"),
            self.tr(
                "Gate 2. After scene match, center displacement from the background homography, divided by the shorter image side. "
                "Raise to allow more camera pan as still duplicate (default 0.02 = 2% of min width/height). "
                "Lower to treat small pans as diversity."
            ),
        )
        self._add_condition_row(
            form,
            self.tr("Max scene scale change:"),
            self._scale,
            self.tr("absolute scale delta"),
            self.tr(
                "Gate 2. Absolute zoom change between views (|scale factor − 1|). "
                "Default 0.02 ≈ ±2% zoom. Raise to merge different zoom levels; lower to keep zoom variants."
            ),
        )
        self._add_condition_row(
            form,
            self.tr("Max scene rotation (rad):"),
            self._rotation,
            self.tr("radians"),
            self.tr(
                "Gate 2. Absolute rotation from the homography, in radians (0.03 ≈ 1.7°). "
                "Raise to allow rotated duplicates; lower to split rotated views."
            ),
        )
        self._add_condition_row(
            form,
            self.tr("Min scene overlap:"),
            self._overlap,
            self.tr("area ratio 0–1"),
            self.tr(
                "Gate 1 (scene match). Overlap of warped view vs the other image: intersection area divided by the smaller image area. "
                "Default 0.35 = 35%. Lower admits more crop/partial views; higher requires almost the same framing. "
                "Not the same as camera motion limits above."
            ),
        )
        self._add_condition_row(
            form,
            self.tr("Max instance shift:"),
            self._true_shift,
            self.tr("of shorter image side"),
            self.tr(
                "Gate 3. After compensating scene motion, max box-center residual between matched instances, divided by the shorter side. "
                "Default 0.03 = 3%. Raise if subjects may move slightly; lower to require nearly fixed objects."
            ),
        )
        self._add_condition_row(
            form,
            self.tr("Max box scale change:"),
            self._box_scale,
            self.tr("|ln(area ratio)|"),
            self.tr(
                "Gate 3. Log ratio of matched box areas. 0.20 ≈ 22% area change (e^0.2). "
                "Raise to allow size changes; lower to require same box size."
            ),
        )
        self._add_condition_row(
            form,
            self.tr("Max pose change:"),
            self._pose,
            self.tr("|ln(aspect ratio)|"),
            self.tr(
                "Gate 3. Log ratio of matched box aspect ratios (shape proxy). "
                "Default 0.20. Raise to allow pose/shape drift; lower to require similar box shape."
            ),
        )
        cond_layout.addLayout(form)
        settings_layout.addWidget(cond)

        quality_box = QGroupBox(self.tr("Quality gate (G4)"))
        quality_layout = QVBoxLayout(quality_box)
        quality_intro = QLabel(
            self.tr(
                "After instance match, optional photometric checks on thumbnails. "
                "Each enabled metric compares two values; pass when |a−b| / max(|a|,|b|) is at or below the relative limit. "
                "Raise limits to allow more blur, exposure, or color change as still duplicate."
            )
        )
        quality_intro.setWordWrap(True)
        quality_intro.setStyleSheet("color: #666;")
        quality_layout.addWidget(quality_intro)
        self._quality_enable = QCheckBox(self.tr("Enable quality gate"))
        self._quality_enable.setChecked(True)
        self._quality_enable.toggled.connect(self._on_quality_gate_toggled)
        quality_layout.addWidget(self._quality_enable)
        g4_form = QtWidgets.QFormLayout()
        self._quality_rows: list[tuple[str, QCheckBox, QDoubleSpinBox]] = []
        metric_notes = {
            "sharpness_laplacian_var_log1p": self.tr(
                "Laplacian variance (log1p) on thumbnail. Lower limit rejects blur vs sharp pairs."
            ),
            "brightness_l_mean": self.tr(
                "Mean LAB L (0–255). Relative limit is vs the brighter of the two images."
            ),
            "contrast_rms_gray": self.tr("Gray RMS contrast. Catches flat vs punchy tone mapping."),
            "saturation_mean": self.tr("Mean HSV saturation. Default relative limit 25%."),
            "value_mean": self.tr("Mean HSV value (overall lightness in HSV)."),
            "noise_mad_residual": self.tr("Median absolute deviation of denoise residual. High noise vs clean shots."),
        }
        for key, title_key, default_rel in _G4_METRIC_UI:
            cb = QCheckBox(self.tr(title_key))
            cb.setChecked(True)
            tol = self._percent_spin(default_rel * 100.0)
            note = metric_notes.get(key, "")
            cb.setToolTip(note)
            tol.setToolTip(
                self.tr(
                    "Max relative difference allowed for this metric. "
                    "Example: 20 means up to 20% vs the larger of the two values."
                )
                + (" " + note if note else "")
            )
            unit = QLabel(self.tr("% vs larger value"))
            unit.setStyleSheet("color: #666;")
            unit.setToolTip(tol.toolTip())
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(tol)
            lay.addWidget(unit)
            lay.addStretch(1)
            g4_form.addRow(cb, row)
            self._quality_rows.append((key, cb, tol))
        quality_layout.addLayout(g4_form)
        settings_layout.addWidget(quality_box)
        self._on_quality_gate_toggled(self._quality_enable.isChecked())

        labels_box = QGroupBox(self.tr("Labels"))
        labels_layout = QVBoxLayout(labels_box)
        labels_hint = QLabel(
            self.tr(
                "Loaded from Label Manager. Checked labels take part in the instance check."
            )
        )
        labels_hint.setWordWrap(True)
        labels_layout.addWidget(labels_hint)
        self._label_checks: list[tuple[str, QCheckBox]] = []
        self._label_host = QWidget()
        self._label_list = QVBoxLayout(self._label_host)
        self._label_list.setContentsMargins(0, 0, 0, 0)
        label_scroll = QScrollArea()
        label_scroll.setWidgetResizable(True)
        label_scroll.setMaximumHeight(140)
        label_scroll.setWidget(self._label_host)
        labels_layout.addWidget(label_scroll)
        label_buttons = QHBoxLayout()
        select_all = QPushButton(self.tr("Select All"))
        select_all.clicked.connect(lambda: self._set_all_labels(True))
        clear_all = QPushButton(self.tr("Deselect All"))
        clear_all.clicked.connect(lambda: self._set_all_labels(False))
        label_buttons.addWidget(select_all)
        label_buttons.addWidget(clear_all)
        label_buttons.addStretch(1)
        labels_layout.addLayout(label_buttons)
        settings_layout.addWidget(labels_box)
        root.addWidget(self._settings_panel)
        self._reload_labels()

        actions = QHBoxLayout()
        self._scan_btn = QPushButton(self.tr("Scan"))
        self._scan_btn.setStyleSheet(get_ok_btn_style())
        self._scan_btn.clicked.connect(self._start_scan)
        self._default_btn = QPushButton(self.tr("Default dedup"))
        self._default_btn.setEnabled(False)
        self._default_btn.clicked.connect(self._default_dedup)
        self._confirm_btn = QPushButton(self.tr("Confirm reviewed dedup"))
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.clicked.connect(self._confirm_review)
        actions.addWidget(self._scan_btn)
        actions.addWidget(self._default_btn)
        actions.addWidget(self._confirm_btn)
        actions.addStretch(1)
        root.addLayout(actions)

        self._summary = QLabel(self.tr("No scan yet."))
        self._summary.setWordWrap(True)
        root.addWidget(self._summary)

        self._review_host = QWidget()
        self._review_layout = QVBoxLayout(self._review_host)
        self._review_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._review_host)
        root.addWidget(scroll, 1)

    def _on_settings_toggled(self, expanded: bool) -> None:
        self._settings_panel.setVisible(expanded)
        self._settings_toggle.setText(
            self.tr("Collapse settings") if expanded else self.tr("Expand settings")
        )

    def _spin(self, value: float, maximum: float = 1.0) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(0.0, maximum)
        box.setDecimals(3)
        box.setSingleStep(0.01)
        box.setValue(value)
        box.setStyleSheet(get_spinbox_style())
        return box

    def _percent_spin(self, percent: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(1.0, 100.0)
        box.setDecimals(0)
        box.setSingleStep(5.0)
        box.setValue(percent)
        box.setStyleSheet(get_spinbox_style())
        return box

    def _on_quality_gate_toggled(self, enabled: bool) -> None:
        for _key, cb, spin in self._quality_rows:
            cb.setEnabled(enabled)
            spin.setEnabled(enabled)

    def _add_condition_row(
        self,
        form: QtWidgets.QFormLayout,
        title: str,
        spin: QDoubleSpinBox,
        unit: str,
        tooltip: str,
    ) -> None:
        spin.setToolTip(tooltip)
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(spin)
        unit_label = QLabel(unit)
        unit_label.setStyleSheet("color: #666;")
        unit_label.setToolTip(tooltip)
        lay.addWidget(unit_label)
        lay.addStretch(1)
        form.addRow(title, row)

    def _config_from_ui(self) -> Config:
        cfg = Config()
        cfg.strict.gate2.max_translation = float(self._shift.value())
        cfg.strict.gate2.max_scale = float(self._scale.value())
        cfg.strict.gate2.max_rotation_rad = float(self._rotation.value())
        cfg.strict.instance.max_true_shift = float(self._true_shift.value())
        cfg.strict.instance.max_scale_diff = float(self._box_scale.value())
        cfg.strict.instance.max_pose_diff = float(self._pose.value())
        cfg.verify.min_overlap = float(self._overlap.value())
        gate_on = bool(self._quality_enable.isChecked())
        cfg.strict.require_quality = gate_on
        cfg.quality.enabled = gate_on
        by_key = {key: (cb, spin) for key, cb, spin in self._quality_rows}
        for metric in cfg.quality.metrics:
            row = by_key.get(metric.key)
            if row is None:
                continue
            cb, spin = row
            metric.enabled = gate_on and cb.isChecked()
            metric.rel_tol = float(spin.value()) / 100.0
        cfg.feature.box_label_filter = tuple(
            name for name, box in self._label_checks if box.isChecked()
        )
        return cfg

    def _reload_labels(self) -> None:
        while self._label_list.count():
            item = self._label_list.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._label_checks = []
        names = []
        collector = getattr(self._host, "collect_label_manager_label_names", None)
        if callable(collector):
            names = list(collector())
        if not names:
            self._label_list.addWidget(QLabel(self.tr("No labels in Label Manager.")))
            return
        for name in names:
            box = QCheckBox(name)
            box.setChecked(True)
            self._label_checks.append((name, box))
            self._label_list.addWidget(box)

    def _set_all_labels(self, checked: bool) -> None:
        for _name, box in self._label_checks:
            box.setChecked(checked)

    def _start_scan(self) -> None:
        images = [str(p) for p in (getattr(self._host, "image_list", None) or []) if p]
        if self._label_checks and not any(box.isChecked() for _name, box in self._label_checks):
            QMessageBox.information(
                self,
                self.tr("Dataset Dedup"),
                self.tr("Select at least one label."),
            )
            return
        if len(images) < 2:
            QMessageBox.information(
                self,
                self.tr("Dataset Dedup"),
                self.tr("Open a folder with at least two images first."),
            )
            return
        self._scan_btn.setEnabled(False)
        self._default_btn.setEnabled(False)
        self._confirm_btn.setEnabled(False)
        self._progress = QtWidgets.QProgressDialog(self)
        self._progress.setWindowTitle(self.tr("Dataset Dedup"))
        self._progress.setLabelText(self.tr("Scanning…"))
        self._progress.setRange(0, 0)
        self._progress.setCancelButton(None)
        self._progress.setStyleSheet(get_progress_dialog_style())
        self._progress.show()
        self._worker = _ScanWorker(self._config_from_ui(), images, self._host)
        self._worker.progressed.connect(self._on_progress)
        self._worker.succeeded.connect(self._on_scanned)
        self._worker.failed.connect(self._on_scan_failed)
        self._worker.start()

    def _on_progress(self, text: str) -> None:
        if self._progress is not None:
            self._progress.setLabelText(text)

    def _on_scan_failed(self, message: str) -> None:
        if self._progress is not None:
            self._progress.close()
        self._scan_btn.setEnabled(True)
        QMessageBox.warning(self, self.tr("Dataset Dedup"), message)

    def _on_scanned(self, result: ScanResult) -> None:
        if self._progress is not None:
            self._progress.close()
        self._scan_btn.setEnabled(True)
        self._result = result
        stats = result.gate_stats
        self._summary.setText(
            self.tr(
                "Images: {images}. Exact groups: {exact}. "
                "Candidates: {candidates}. Scene matches: {scene}. "
                "Small camera motion: {view}. Same instance: {instance}. "
                "Pairs without boxes (kept): {nobox}. "
                "Groups to review: {groups}."
            ).format(
                images=stats.get("images", 0),
                exact=stats.get("L0_exact_groups", 0),
                candidates=stats.get("L1_candidates", 0),
                scene=stats.get("G1_scene_match", 0),
                view=stats.get("G2_small_view_change", 0),
                instance=stats.get("G3_instance_same", 0),
                nobox=stats.get("G3_no_boxes", 0),
                groups=stats.get("groups", 0),
            )
        )
        self._fill_review(result.group_paths())
        has_groups = bool(result.groups)
        self._default_btn.setEnabled(has_groups)
        self._confirm_btn.setEnabled(has_groups)

    def _fill_review(self, groups: list[list[str]]) -> None:
        while self._review_layout.count():
            item = self._review_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._cards = []
        if not groups:
            self._review_layout.addWidget(QLabel(self.tr("No duplicate groups with 2 or more images.")))
            return
        for index, paths in enumerate(groups, start=1):
            card = _GroupCard(
                self.tr("Group {index} ({count})").format(index=index, count=len(paths)),
                paths,
            )
            self._cards.append(card)
            self._review_layout.addWidget(card)
        self._review_layout.addStretch(1)

    def _default_dedup(self) -> None:
        if self._result is None:
            return
        doomed: list[str] = []
        removed: set[str] = set()
        for group in self._result.groups:
            alive = []
            for index in group:
                path = self._result.recs[index].image_path
                if osp.normcase(path) not in removed:
                    alive.append(path)
            if len(alive) < 2:
                continue
            for path in alive[1:]:
                doomed.append(path)
                removed.add(osp.normcase(path))
        self._delete_paths(doomed)

    def _confirm_review(self) -> None:
        if not self._cards:
            return
        for card in self._cards:
            if not card.checked_paths():
                QMessageBox.information(
                    self,
                    self.tr("Dataset Dedup"),
                    self.tr("Each group must keep at least one checked image."),
                )
                return
        votes: dict[str, list[bool]] = {}
        for card in self._cards:
            for path, box in zip(card.paths, card.checks):
                votes.setdefault(osp.normcase(path), []).append(box.isChecked())
        doomed = []
        seen: set[str] = set()
        for card in self._cards:
            for path in card.paths:
                key = osp.normcase(path)
                if key in seen:
                    continue
                seen.add(key)
                if not any(votes.get(key, [])):
                    doomed.append(path)
        self._delete_paths(doomed)

    def _delete_paths(self, image_paths: list[str]) -> None:
        unique: list[str] = []
        seen: set[str] = set()
        for path in image_paths:
            key = osp.normcase(path)
            if key in seen or not path:
                continue
            seen.add(key)
            unique.append(path)
        if not unique:
            QMessageBox.information(
                self,
                self.tr("Dataset Dedup"),
                self.tr("Nothing to delete."),
            )
            return
        answer = QMessageBox.question(
            self,
            self.tr("Dataset Dedup"),
            self.tr("Delete {count} image(s) and their annotation files? This cannot be undone.").format(
                count=len(unique)
            ),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        errors: list[str] = []
        deleted: list[str] = []
        for path in unique:
            try:
                for label in _annotation_paths(self._host, path):
                    os.remove(label)
                if osp.isfile(path):
                    os.remove(path)
                deleted.append(path)
            except OSError as exc:
                errors.append(f"{osp.basename(path)}: {exc}")
        self._reload_open_folder()
        if errors:
            QMessageBox.warning(
                self,
                self.tr("Dataset Dedup"),
                self.tr("Deleted {count}. Some files could not be removed:\n{errors}").format(
                    count=len(deleted), errors="\n".join(errors[:8])
                ),
            )
        else:
            QMessageBox.information(
                self,
                self.tr("Dataset Dedup"),
                self.tr("Deleted {count} image(s).").format(count=len(deleted)),
            )
        self._result = None
        self._cards = []
        self._default_btn.setEnabled(False)
        self._confirm_btn.setEnabled(False)
        self._summary.setText(self.tr("Dedup finished. Scan again if the folder changed."))

    def _reload_open_folder(self) -> None:
        folder = getattr(self._host, "last_open_dir", None)
        if folder and hasattr(self._host, "import_image_folder"):
            self._host.import_image_folder(folder)


def open_dataset_dedup_dialog(parent) -> None:
    DatasetDedupDialog(parent).exec()