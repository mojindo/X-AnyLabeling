"""Show charts of aggregated dataset analysis metrics and annotations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QComboBox,
    QPlainTextEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from anylabeling.views.labeling.advanced.dataset_analysis.chart_export import (
    export_dataset_analysis_charts,
    export_optimization_suggestions_txt,
)
from anylabeling.views.labeling.advanced.dataset_analysis.scene_export import (
    export_scene_cluster_images,
)
from anylabeling.views.labeling.advanced.dataset_analysis.metrics import numeric_keys_for_charts
from anylabeling.views.labeling.utils.style import get_dialog_style
from anylabeling.views.labeling.utils.qt import new_icon_path

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    _MATPLOTLIB_OK = True
except ImportError:
    _MATPLOTLIB_OK = False
    FigureCanvasQTAgg = None  # type: ignore
    Figure = None  # type: ignore


class DatasetAnalysisReportDialog(QDialog):
    def __init__(
        self,
        parent,
        *,
        numeric_series: Dict[str, List[float]],
        diversity: Dict[str, Any],
        summary_path: str,
        annotation_stats: Optional[Dict[str, Any]] = None,
        resolution_counts: Optional[Dict[str, int]] = None,
        optimization_suggestions: Optional[Dict[str, Any]] = None,
        image_paths: Optional[List[str]] = None,
    ):
        super().__init__(parent)
        self._numeric_series = numeric_series
        self._diversity = diversity
        self._annotation_stats = annotation_stats
        self._resolution_counts = resolution_counts or {}
        self._optimization_suggestions = optimization_suggestions or {}
        self._image_paths = list(image_paths or [])
        self.setWindowTitle(self.tr("Dataset Analysis Report"))
        self.setMinimumSize(720, 520)
        self.setStyleSheet(get_dialog_style())

        root = QVBoxLayout(self)
        tabs = QTabWidget()
        self._tabs = tabs

        nav_row = QHBoxLayout()
        prev_btn = QPushButton()
        prev_btn.setToolTip(self.tr("Previous report tab"))
        prev_btn.setIcon(QIcon(new_icon_path("arrow-left", "svg")))
        prev_btn.clicked.connect(self._goto_prev_tab)

        next_btn = QPushButton()
        next_btn.setToolTip(self.tr("Next report tab"))
        next_btn.setIcon(QIcon(new_icon_path("arrow-right", "svg")))
        next_btn.clicked.connect(self._goto_next_tab)

        title = QLabel(self.tr("Dataset Analysis Report"))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        nav_row.addWidget(prev_btn)
        nav_row.addWidget(next_btn)
        nav_row.addStretch()
        nav_row.addWidget(title)
        nav_row.addStretch()
        root.addLayout(nav_row)

        if not _MATPLOTLIB_OK:
            lab = QLabel(
                self.tr(
                    "matplotlib is not installed. Install optional extra:\n"
                    "pip install \"x-anylabeling-cvhub[advance]\""
                )
            )
            lab.setWordWrap(True)
            root.addWidget(lab)
        else:
            # Resolution: show as a single WxH count bar chart (not 4 separate histograms)
            self._append_resolution_tab(tabs, self._resolution_counts)

            skip_keys = {
                "original_width",
                "original_height",
                "megapixels",
                "aspect_ratio",
            }
            order = [
                k
                for k in numeric_keys_for_charts()
                if k not in skip_keys and numeric_series.get(k)
            ]
            for key in order:
                values = numeric_series[key]
                if not values:
                    continue
                page = QWidget()
                lay = QVBoxLayout(page)
                fig = Figure(figsize=(6, 3.5), dpi=100)
                ax = fig.add_subplot(111)
                ax.hist(values, bins=min(30, max(10, len(values) // 5)), color="#0066FF", alpha=0.85)
                ax.set_title(key)
                ax.set_xlabel(self.tr("value"))
                ax.set_ylabel(self.tr("count"))
                fig.tight_layout()
                canvas = FigureCanvasQTAgg(fig)
                lay.addWidget(canvas)
                tabs.addTab(page, key[:32])

            if isinstance(annotation_stats, dict) and annotation_stats:
                self._append_annotation_tabs(tabs, annotation_stats)

            cap_lens = diversity.get("caption_lengths") or []
            if (
                cap_lens
                and isinstance(cap_lens, list)
                and all(isinstance(x, (int, float)) for x in cap_lens)
            ):
                lens = [float(x) for x in cap_lens]
                page = QWidget()
                lay = QVBoxLayout(page)
                fig = Figure(figsize=(6, 3.5), dpi=100)
                ax = fig.add_subplot(111)
                ax.hist(
                    lens,
                    bins=min(25, max(8, len(lens) // 4)),
                    color="#00A86B",
                    alpha=0.85,
                )
                ax.set_title(self.tr("caption character length"))
                ax.set_xlabel(self.tr("length"))
                ax.set_ylabel(self.tr("count"))
                fig.tight_layout()
                canvas = FigureCanvasQTAgg(fig)
                lay.addWidget(canvas)
                tabs.addTab(page, self.tr("Caption len"))

        div_page = QWidget()
        div_layout = QVBoxLayout(div_page)
        div_text = QLabel()
        div_text.setWordWrap(True)
        div_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        summary = diversity.get("summary_zh", "")
        div_text.setText(summary or self.tr("No diversity summary."))
        div_layout.addWidget(div_text)
        tabs.addTab(div_page, self.tr("Diversity"))

        # Scene cluster counts + export list
        try:
            scene_clusters = diversity.get("scene_clusters") or []
        except Exception:
            scene_clusters = []
        if isinstance(scene_clusters, list) and scene_clusters:
            page = QWidget()
            lay = QVBoxLayout(page)

            top = QHBoxLayout()
            combo = QComboBox()
            combo.setMinimumWidth(260)
            # Map combo index -> cluster dict
            clusters_ok = []
            for c in scene_clusters:
                if not isinstance(c, dict):
                    continue
                cid = c.get("cluster_id")
                cnt = c.get("count")
                try:
                    cid_i = int(cid)
                    cnt_i = int(cnt)
                except Exception:
                    continue
                center = c.get("center_caption") or ""
                center = str(center).replace("\n", " ").strip()
                if len(center) > 30:
                    center = center[:30] + "…"
                combo.addItem(
                    self.tr("Cluster %d (%d images) - %s")
                    % (cid_i, cnt_i, center),
                    userData=c,
                )
                clusters_ok.append(c)

            export_btn = QPushButton(self.tr("Export image list…"))
            top.addWidget(combo)
            top.addStretch()
            top.addWidget(export_btn)
            lay.addLayout(top)

            sample_row = QHBoxLayout()
            sample_row.addWidget(
                QLabel(self.tr("Images per scene (all clusters):"))
            )
            sample_count_spin = QSpinBox()
            sample_count_spin.setRange(1, 10000)
            sample_count_spin.setValue(5)
            sample_count_spin.setToolTip(
                self.tr(
                    "Randomly copy this many images from each scene cluster "
                    "to the export folder."
                )
            )
            sample_row.addWidget(sample_count_spin)
            sample_row.addStretch()
            export_images_btn = QPushButton(self.tr("Export scene images…"))
            sample_row.addWidget(export_images_btn)
            lay.addLayout(sample_row)

            txt = QPlainTextEdit()
            txt.setReadOnly(True)
            lay.addWidget(txt, 1)

            def _render_current():
                c = combo.currentData()
                if not isinstance(c, dict):
                    txt.setPlainText("")
                    return
                imgs = c.get("images") or []
                lines = []
                for it in imgs:
                    if not isinstance(it, dict):
                        continue
                    p = it.get("imagePath") or ""
                    p = str(p).strip()
                    if p:
                        lines.append(p)
                txt.setPlainText("\n".join(lines))

            def _export_current():
                c = combo.currentData()
                if not isinstance(c, dict):
                    return
                cid = c.get("cluster_id", 0)
                try:
                    cid_i = int(cid)
                except Exception:
                    cid_i = 0
                path, _flt = QFileDialog.getSaveFileName(
                    self,
                    self.tr("Save image list for cluster %d") % cid_i,
                    f"cluster_{cid_i}_images.txt",
                    "Text (*.txt);;JSON (*.json)",
                )
                if not path:
                    return
                imgs = c.get("images") or []
                lines = []
                for it in imgs:
                    if not isinstance(it, dict):
                        continue
                    p = it.get("imagePath") or ""
                    p = str(p).strip()
                    if p:
                        lines.append(p)
                try:
                    if path.lower().endswith(".json"):
                        import json

                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(
                                {"cluster_id": cid_i, "images": lines},
                                f,
                                indent=2,
                                ensure_ascii=False,
                            )
                    else:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write("\n".join(lines))
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        self.tr("Export failed"),
                        self.tr("Could not save file: %s") % str(e),
                    )

            def _export_scene_images():
                if not self._image_paths:
                    QMessageBox.warning(
                        self,
                        self.tr("Export failed"),
                        self.tr("No source image list available."),
                    )
                    return
                export_dir = QFileDialog.getExistingDirectory(
                    self,
                    self.tr("Select folder for scene image export"),
                    "",
                    QFileDialog.Option.ShowDirsOnly
                    | QFileDialog.Option.DontResolveSymlinks,
                )
                if not export_dir:
                    return
                per_n = int(sample_count_spin.value())
                result = export_scene_cluster_images(
                    export_dir,
                    clusters_ok,
                    self._image_paths,
                    per_n,
                )
                if result.images_copied <= 0 and not result.clusters_processed:
                    detail = self.tr("No images were copied.")
                    if result.errors:
                        detail += "\n\n" + "\n".join(result.errors[:8])
                    QMessageBox.warning(
                        self,
                        self.tr("Export failed"),
                        detail,
                    )
                    return
                msg = self.tr(
                    "Copied %d image(s) from %d scene cluster(s) to:\n%s"
                ) % (
                    result.images_copied,
                    result.clusters_processed,
                    export_dir,
                )
                if result.errors:
                    msg += "\n\n" + self.tr("Warnings:\n%s") % "\n".join(
                        result.errors[:8]
                    )
                    QMessageBox.warning(
                        self,
                        self.tr("Export partially failed"),
                        msg,
                    )
                else:
                    QMessageBox.information(
                        self,
                        self.tr("Export complete"),
                        msg,
                    )

            combo.currentIndexChanged.connect(_render_current)
            export_btn.clicked.connect(_export_current)
            export_images_btn.clicked.connect(_export_scene_images)
            _render_current()
            tabs.addTab(page, self.tr("Scenes"))

        # Optimization suggestions
        sug = self._optimization_suggestions or {}
        if isinstance(sug, dict):
            sugg_list = sug.get("suggestions_zh")
            if isinstance(sugg_list, list) and sugg_list:
                page = QWidget()
                lay = QVBoxLayout(page)
                lab = QLabel()
                lab.setWordWrap(True)
                lab.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                )
                lines = []
                for s in sugg_list:
                    if isinstance(s, str) and s.strip():
                        lines.append(f"- {s.strip()}")
                lab.setText("\n".join(lines) if lines else self.tr("No suggestions."))
                lay.addWidget(lab)
                tabs.addTab(page, self.tr("Suggestions"))

        root.addWidget(tabs)

        if summary_path:
            fp = QLabel(self.tr("Summary file: %s") % summary_path)
            fp.setWordWrap(True)
            root.addWidget(fp)

        btn_row = QHBoxLayout()
        export_btn = QPushButton(self.tr("Export charts to folder…"))
        export_btn.setToolTip(
            self.tr(
                "Choose a separate folder for PNG exports; "
                "this is not the project label output directory."
            )
        )
        export_btn.clicked.connect(self._on_export_charts)
        btn_row.addWidget(export_btn)
        btn_row.addStretch()
        close_btn = QPushButton(self.tr("Close"))
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _goto_prev_tab(self) -> None:
        tabs = getattr(self, "_tabs", None)
        if tabs is None:
            return
        n = tabs.count()
        if n <= 0:
            return
        idx = tabs.currentIndex()
        tabs.setCurrentIndex((idx - 1) % n)

    def _goto_next_tab(self) -> None:
        tabs = getattr(self, "_tabs", None)
        if tabs is None:
            return
        n = tabs.count()
        if n <= 0:
            return
        idx = tabs.currentIndex()
        tabs.setCurrentIndex((idx + 1) % n)

    def _on_export_charts(self):
        path = QFileDialog.getExistingDirectory(
            self,
            self.tr(
                "Select folder for chart export (independent of label output)"
            ),
            "",
            QFileDialog.Option.ShowDirsOnly
            | QFileDialog.Option.DontResolveSymlinks,
        )
        if not path:
            return
        saved, err = export_dataset_analysis_charts(
            path,
            self._numeric_series,
            self._diversity,
            annotation_stats=self._annotation_stats,
            resolution_counts=self._resolution_counts,
            title_caption_len=self.tr("caption character length"),
        )
        sugg_path, sugg_err = export_optimization_suggestions_txt(
            path, self._optimization_suggestions
        )
        if err and not saved:
            QMessageBox.warning(
                self,
                self.tr("Export failed"),
                self.tr("Could not export charts: %s") % err,
            )
            return
        if err:
            QMessageBox.warning(
                self,
                self.tr("Export partially failed"),
                self.tr("Some charts may be missing: %s") % err,
            )
        if sugg_err and not sugg_path:
            QMessageBox.warning(
                self,
                self.tr("Export partially failed"),
                self.tr("Charts saved but suggestions file failed: %s") % sugg_err,
            )
        detail = self.tr("Saved %d image(s) to:\n%s") % (len(saved), path)
        if sugg_path:
            detail += "\n\n" + self.tr("Suggestions saved to:\n%s") % sugg_path
        QMessageBox.information(
            self,
            self.tr("Export complete"),
            detail,
        )

    def _append_annotation_tabs(self, tabs: QTabWidget, stats: Dict[str, Any]) -> None:
        """Add annotation statistics charts."""
        try:
            import numpy as np
        except Exception:
            return

        class_names = stats.get("class_names") or []
        per_class_box = stats.get("per_class_box_count") or {}
        images_total = int(stats.get("image_total") or 0)
        images_with_any = int(stats.get("images_with_any_box") or 0)
        images_zero = int(stats.get("images_with_zero_box") or 0)

        # Summary: image counts + per-class box counts
        page = QWidget()
        lay = QVBoxLayout(page)
        fig = Figure(figsize=(7, 4.2), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        bars1 = ax1.bar(
            ["total", "with_box", "zero_box"],
            [images_total, images_with_any, images_zero],
            color=["#6C757D", "#0066FF", "#FF6B6B"],
            width=0.6,
        )
        ax1.bar_label(bars1, padding=2, fontsize=9)
        ax1.set_title(self.tr("Images"))
        ax1.set_ylabel(self.tr("count"))

        if class_names:
            y = [int(per_class_box.get(c, 0)) for c in class_names]
            xs = np.arange(len(class_names), dtype=np.float32)
            bars2 = ax2.bar(xs, y, color="#00A86B", width=0.6)
            ax2.set_xticks(xs)
            ax2.set_xticklabels(class_names)
            ax2.bar_label(bars2, padding=2, fontsize=8)
            ax2.set_title(self.tr("Boxes per class"))
            ax2.tick_params(axis="x", labelrotation=60)
        else:
            ax2.text(0.5, 0.5, self.tr("No classes"), ha="center", va="center")
            ax2.set_axis_off()

        fig.tight_layout()
        lay.addWidget(FigureCanvasQTAgg(fig))
        tabs.addTab(page, self.tr("Annot summary"))

        # Distribution: how many images contain k boxes of class c (single chart)
        per_image_counts = stats.get("per_image_class_counts") or []
        if class_names and isinstance(per_image_counts, list) and per_image_counts:
            class_totals = {
                c: int(per_class_box.get(c, 0)) for c in class_names
            }
            # Limit classes to keep chart readable; fixed bar width requirement
            max_classes = 6
            top_classes = [
                c
                for c, _v in sorted(
                    class_totals.items(), key=lambda kv: kv[1], reverse=True
                )
                if class_totals.get(c, 0) > 0
            ][:max_classes]
            if not top_classes:
                top_classes = class_names[:1]

            # Build histogram over k=0..maxk for each class
            max_k = 0
            per_class_k_counts: Dict[str, Dict[int, int]] = {}
            for c in top_classes:
                kk: Dict[int, int] = {}
                for d in per_image_counts:
                    v = 0
                    if isinstance(d, dict):
                        v = int(d.get(c, 0) or 0)
                    kk[v] = kk.get(v, 0) + 1
                    max_k = max(max_k, v)
                per_class_k_counts[c] = kk

            ks = list(range(0, max_k + 1))
            x = np.arange(len(ks), dtype=np.float32)
            width = 0.12  # fixed bar width
            page = QWidget()
            lay = QVBoxLayout(page)
            fig = Figure(figsize=(7.6, 4.6), dpi=100)
            ax = fig.add_subplot(111)
            palette = ["#0066FF", "#00A86B", "#845EF7", "#FF6B6B", "#F59F00", "#12B886"]
            for idx, c in enumerate(top_classes):
                offs = (idx - (len(top_classes) - 1) / 2.0) * width
                ys = [int(per_class_k_counts[c].get(k, 0)) for k in ks]
                bars = ax.bar(
                    x + offs,
                    ys,
                    width=width,
                    color=palette[idx % len(palette)],
                    label=c,
                )
                ax.bar_label(bars, padding=1, fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([str(k) for k in ks])
            ax.set_xlabel(self.tr("boxes per image (k)"))
            ax.set_ylabel(self.tr("image count"))
            ax.set_title(self.tr("Images distribution by boxes-per-image (per class)"))
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            lay.addWidget(FigureCanvasQTAgg(fig))
            tabs.addTab(page, self.tr("Boxes per image"))

        # Area ratio distribution bins (per class, one chart with colors)
        ratios = stats.get("per_box_area_ratio") or {}
        if class_names and isinstance(ratios, dict):
            # choose top classes similar to boxes-per-image chart
            class_totals = {
                c: int(per_class_box.get(c, 0)) for c in class_names
            }
            max_classes = 6
            top_classes = [
                c
                for c, _v in sorted(
                    class_totals.items(), key=lambda kv: kv[1], reverse=True
                )
                if class_totals.get(c, 0) > 0
            ][:max_classes]
            if top_classes:
                all_vals = []
                for c in top_classes:
                    all_vals.extend(
                        [
                            float(x)
                            for x in (ratios.get(c) or [])
                            if isinstance(x, (int, float))
                        ]
                    )
                all_vals = [v for v in all_vals if 0.0 <= v <= 1.0]
                if len(all_vals) >= 5:
                    qs = np.quantile(np.array(all_vals), [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
                    edges = sorted(set([0.0, *[float(x) for x in qs], 1.0]))
                else:
                    edges = [0.0, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
                # ensure monotonic and at least 2 bins
                if len(edges) < 3:
                    edges = [0.0, 0.1, 1.0]
                bin_labels = [
                    f"{edges[i]*100:.2f}%–{edges[i+1]*100:.2f}%"
                    for i in range(len(edges) - 1)
                ]
                x = np.arange(len(bin_labels), dtype=np.float32)
                width = 0.12
                page = QWidget()
                lay = QVBoxLayout(page)
                fig = Figure(figsize=(7.6, 4.6), dpi=100)
                ax = fig.add_subplot(111)
                palette = ["#0066FF", "#00A86B", "#845EF7", "#FF6B6B", "#F59F00", "#12B886"]
                for idx, c in enumerate(top_classes):
                    vals = [
                        float(v)
                        for v in (ratios.get(c) or [])
                        if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0
                    ]
                    hist, _ = np.histogram(np.array(vals, dtype=np.float64), bins=np.array(edges, dtype=np.float64))
                    offs = (idx - (len(top_classes) - 1) / 2.0) * width
                    bars = ax.bar(
                        x + offs,
                        hist.tolist(),
                        width=width,
                        color=palette[idx % len(palette)],
                        label=c,
                    )
                    ax.bar_label(bars, padding=1, fontsize=8)
                ax.set_xticks(x)
                ax.set_xticklabels(bin_labels, rotation=35, ha="right")
                ax.set_xlabel(self.tr("box area ratio bins"))
                ax.set_ylabel(self.tr("box count"))
                ax.set_title(self.tr("Box area ratio distribution (per class)"))
                ax.legend(loc="best", fontsize=8)
                fig.tight_layout()
                lay.addWidget(FigureCanvasQTAgg(fig))
                tabs.addTab(page, self.tr("Area ratio bins"))

        # Centers: one tab per class (limit count)
        centers = stats.get("per_class_centers") or {}
        if isinstance(centers, dict) and class_names:
            max_class_tabs = 20
            shown = 0
            for c in class_names:
                pts = centers.get(c) or []
                if not isinstance(pts, list) or not pts:
                    continue
                xs, ys = [], []
                for p in pts:
                    if isinstance(p, dict):
                        x, y = p.get("x"), p.get("y")
                    else:
                        try:
                            x, y = p
                        except Exception:
                            continue
                    if isinstance(x, (int, float)) and isinstance(
                        y, (int, float)
                    ):
                        xs.append(float(x))
                        ys.append(float(y))
                if not xs:
                    continue
                page = QWidget()
                lay = QVBoxLayout(page)
                fig = Figure(figsize=(6.5, 4.5), dpi=100)
                ax = fig.add_subplot(111)
                hb = ax.hexbin(
                    xs,
                    ys,
                    gridsize=35,
                    extent=(0, 1, 0, 1),
                    cmap="viridis",
                )
                ax.set_title(self.tr("Center distribution: %s") % c)
                ax.set_xlabel("cx / W")
                ax.set_ylabel("cy / H")
                fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.02)
                fig.tight_layout()
                lay.addWidget(FigureCanvasQTAgg(fig))
                tabs.addTab(page, (self.tr("Center ") + c)[:32])
                shown += 1
                if shown >= max_class_tabs:
                    break

    def _append_resolution_tab(self, tabs: QTabWidget, counts: Dict[str, int]) -> None:
        """Resolution: one bar chart of WxH counts."""
        if not counts:
            return
        import numpy as np

        # show top N resolutions for readability
        items = list(counts.items())
        max_n = 30
        shown = items[:max_n]
        other = sum(v for _, v in items[max_n:])
        if other:
            shown.append((self.tr("other"), other))
        labels = [k for k, _v in shown]
        vals = [int(v) for _k, v in shown]
        x = np.arange(len(labels), dtype=np.float32)
        page = QWidget()
        lay = QVBoxLayout(page)
        fig = Figure(figsize=(7.6, 4.6), dpi=100)
        ax = fig.add_subplot(111)
        bars = ax.bar(x, vals, width=0.6, color="#0066FF")
        ax.bar_label(bars, padding=2, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(self.tr("image count"))
        ax.set_title(self.tr("Resolution counts (WxH)"))
        fig.tight_layout()
        lay.addWidget(FigureCanvasQTAgg(fig))
        tabs.addTab(page, self.tr("Resolution"))
