"""Image augmentation for the open dataset. One method per source image."""

from __future__ import annotations

import os.path as osp
from typing import List, Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
)

from anylabeling.views.labeling.advanced.augment.engine import (
    AugmentStep,
    apply_one_method_dataset,
)
from anylabeling.views.labeling.utils.style import (
    get_dialog_style,
    get_lineedit_style,
    get_ok_btn_style,
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


class _AugmentWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    finished_ok = QtCore.pyqtSignal(int, int)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, image_paths, steps, overwrite, output_dir, label_for) -> None:
        super().__init__()
        self._image_paths = list(image_paths)
        self._steps = list(steps)
        self._overwrite = overwrite
        self._output_dir = output_dir
        self._label_for = label_for
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            total, ok = apply_one_method_dataset(
                image_paths=self._image_paths,
                steps=self._steps,
                overwrite=self._overwrite,
                output_dir=self._output_dir,
                label_for=self._label_for,
                on_progress=lambda i, n, p: self.progress.emit(i, n, p),
                should_cancel=lambda: self._cancel,
            )
            self.finished_ok.emit(total, ok)
        except Exception as exc:
            self.failed.emit(str(exc))


class ImageAugmentationDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Image Augmentation"))
        self.setMinimumSize(1000, 640)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setStyleSheet(get_dialog_style())
        self._host = parent
        self._worker: Optional[_AugmentWorker] = None
        self._progress: Optional[QtWidgets.QProgressDialog] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(8)
        hint = QLabel(
            self.tr(
                "Each source image receives exactly one enabled method, chosen in turn. "
                "Methods are not stacked. The output folder receives only the augmented image and its JSON."
            )
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        out_box = QGroupBox(self.tr("Augmentation output"))
        out_lay = QVBoxLayout(out_box)
        self.rb_overwrite = QRadioButton(self.tr("Overwrite original images"))
        self.rb_copy = QRadioButton(self.tr("Write augmented images and JSON to an output folder"))
        self.rb_copy.setChecked(True)
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText(self.tr("Select output folder..."))
        self.out_dir_edit.setStyleSheet(get_lineedit_style())
        browse = QPushButton(self.tr("Browse..."))
        browse.clicked.connect(self._pick_output_dir)
        row = QHBoxLayout()
        row.addWidget(self.out_dir_edit, 1)
        row.addWidget(browse)
        out_lay.addWidget(self.rb_overwrite)
        out_lay.addWidget(self.rb_copy)
        out_lay.addLayout(row)
        out_lay.addWidget(QLabel(self.tr("Original images are not copied. A missed probability also writes nothing.")))
        root.addWidget(out_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QtWidgets.QWidget()
        self._method_grid_cols = 2
        self._method_grid_next = 0
        ops_lay = QGridLayout(host)
        ops_lay.setContentsMargins(0, 0, 0, 0)
        ops_lay.setHorizontalSpacing(12)
        ops_lay.setVerticalSpacing(12)
        for col in range(self._method_grid_cols):
            ops_lay.setColumnStretch(col, 1)
        self._method_grid = ops_lay
        self.ops = []
        self.ops.append(self._mk_random_fog_row())
        self.ops.append(self._mk_random_rain_row())
        self.ops.append(self._mk_random_snow_row())
        self.ops.append(self._mk_random_sun_flare_row())
        self.ops.append(self._mk_random_shadow_row())
        self.ops.append(self._mk_brightness_contrast_row())
        self.ops.append(self._mk_hsv_row())
        self.ops.append(self._mk_motion_blur_row())
        self.ops.append(self._mk_gaussian_blur_row())
        self.ops.append(self._mk_defocus_row())
        self.ops.append(self._mk_gauss_noise_row())
        self.ops.append(self._mk_image_compression_row())
        self.ops.append(self._mk_iso_noise_row())
        self.ops.append(self._mk_rgb_shift_row())
        self.ops.append(self._mk_gamma_row())
        self.ops.append(self._mk_clahe_row())
        self.ops.append(self._mk_blur_row())
        self.ops.append(self._mk_median_blur_row())
        self.ops.append(self._mk_coarse_dropout_row())
        self.ops.append(self._mk_elastic_row())
        self.ops.append(self._mk_grid_row())
        self.ops.append(self._mk_optical_row())
        last_row = (len(self.ops) + self._method_grid_cols - 1) // self._method_grid_cols
        self._method_grid.setRowStretch(last_row, 1)
        scroll.setWidget(host)
        root.addWidget(scroll, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        close_btn = QPushButton(self.tr("Close"))
        close_btn.clicked.connect(self.reject)
        self.btn_run = QPushButton(self.tr("Start augmentation"))
        self.btn_run.setStyleSheet(get_ok_btn_style())
        self.btn_run.clicked.connect(self._run)
        buttons.addWidget(close_btn)
        buttons.addWidget(self.btn_run)
        root.addLayout(buttons)

    def _note(self, layout: QVBoxLayout, text: str) -> None:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #666;")
        layout.addWidget(label)

    def _place_method_box(self, box: QGroupBox) -> None:
        idx = self._method_grid_next
        self._method_grid_next += 1
        row = idx // self._method_grid_cols
        col = idx % self._method_grid_cols
        box.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self._method_grid.addWidget(box, row, col)

    def _mk_common_box(self, *, op_id: str, title: str, note: str):
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        self._note(lay, note)
        top = QHBoxLayout()
        cb = QCheckBox(self.tr("Enable"))
        probability = QDoubleSpinBox()
        probability.setRange(0.0, 1.0)
        probability.setSingleStep(0.05)
        probability.setDecimals(2)
        probability.setValue(1.0)
        probability.setToolTip(
            self.tr("Chance this method runs when it is the one assigned to an image. Below 1, some assigned images are skipped and not copied.")
        )
        top.addWidget(cb)
        top.addStretch(1)
        top.addWidget(QLabel(self.tr("Apply chance")))
        top.addWidget(probability)
        lay.addLayout(top)
        form = QFormLayout()
        lay.addLayout(form)
        self._place_method_box(box)
        return {"op_id": op_id, "cb": cb, "p": probability, "form": form}

    def _dspin(self, lo, hi, step, value, decimals, tip):
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setSingleStep(step)
        box.setDecimals(decimals)
        box.setValue(value)
        box.setToolTip(tip)
        return box

    def _ispin(self, lo, hi, step, value, tip):
        box = QSpinBox()
        box.setRange(lo, hi)
        box.setSingleStep(step)
        box.setValue(value)
        box.setToolTip(tip)
        return box

    def _mk_random_fog_row(self):
        item = self._mk_common_box( op_id="random_fog", title=self.tr("Fog"), note=self.tr("Blends the image toward white. Higher fog and alpha make the scene milkier. Annotations stay in place."))
        form = item["form"]
        fog_lo = self._dspin(0, 1, 0.05, 0.3, 2, self.tr("Lower bound of fog strength, 0 to 1. Must be less than or equal to the upper bound."))
        fog_hi = self._dspin(0, 1, 0.05, 1.0, 2, self.tr("Upper bound of fog strength, 0 to 1."))
        alpha = self._dspin(0, 1, 0.02, 0.08, 2, self.tr("How much of that fog is mixed in. 0 keeps the original colors; 1 is a strong white veil."))
        form.addRow(self.tr("Fog low"), fog_lo)
        form.addRow(self.tr("Fog high"), fog_hi)
        form.addRow(self.tr("Mix amount"), alpha)
        item["params_widgets"] = {"fog_lo": fog_lo, "fog_hi": fog_hi, "alpha": alpha}
        return item

    def _mk_random_rain_row(self):
        item = self._mk_common_box( op_id="random_rain", title=self.tr("Rain"), note=self.tr("Draws slanted streaks, then darkens the frame. Negative slant leans one way, positive the other."))
        form = item["form"]
        sl_lo = self._ispin(-60, 0, 1, -10, self.tr("Most leftward streak slant. Keep this less than or equal to slant high."))
        sl_hi = self._ispin(0, 60, 1, 10, self.tr("Most rightward streak slant."))
        drop_len = self._ispin(1, 100, 1, 20, self.tr("Streak length in pixels."))
        drop_w = self._ispin(1, 10, 1, 1, self.tr("Streak thickness in pixels."))
        blur_v = self._ispin(0, 15, 1, 3, self.tr("Extra blur on the streaks. 0 leaves them sharp."))
        bright = self._dspin(0.1, 1.0, 0.05, 0.9, 2, self.tr("Brightness kept after rain. 1 is unchanged; smaller values look darker."))
        form.addRow(self.tr("Slant low"), sl_lo)
        form.addRow(self.tr("Slant high"), sl_hi)
        form.addRow(self.tr("Drop length"), drop_len)
        form.addRow(self.tr("Drop width"), drop_w)
        form.addRow(self.tr("Streak blur"), blur_v)
        form.addRow(self.tr("Brightness kept"), bright)
        item["params_widgets"] = {"sl_lo": sl_lo, "sl_hi": sl_hi, "drop_len": drop_len, "drop_w": drop_w, "blur_v": blur_v, "bright": bright}
        return item

    def _mk_random_snow_row(self):
        item = self._mk_common_box( op_id="random_snow", title=self.tr("Snow"), note=self.tr("Sprinkles white points and lifts brightness. A higher snow range means more flakes."))
        form = item["form"]
        sp_lo = self._dspin(0, 1, 0.05, 0.1, 2, self.tr("Lower snow density, 0 to 1."))
        sp_hi = self._dspin(0, 1, 0.05, 0.3, 2, self.tr("Upper snow density, 0 to 1."))
        bc = self._dspin(0.1, 5, 0.1, 2.0, 2, self.tr("How much the snowy frame is brightened. 1 is mild."))
        form.addRow(self.tr("Density low"), sp_lo)
        form.addRow(self.tr("Density high"), sp_hi)
        form.addRow(self.tr("Brightness gain"), bc)
        item["params_widgets"] = {"sp_lo": sp_lo, "sp_hi": sp_hi, "bc": bc}
        return item

    def _mk_random_shadow_row(self):
        item = self._mk_common_box( op_id="random_shadow", title=self.tr("Shadow"), note=self.tr("Darkens a few random patches. Intensity 0 is invisible; 1 replaces the patch with black."))
        form = item["form"]
        ns_lo = self._ispin(0, 20, 1, 1, self.tr("Fewest shadow patches."))
        ns_hi = self._ispin(0, 20, 1, 3, self.tr("Most shadow patches."))
        i_lo = self._dspin(0, 1, 0.05, 0.2, 2, self.tr("Lightest shadow, 0 to 1."))
        i_hi = self._dspin(0, 1, 0.05, 0.6, 2, self.tr("Darkest shadow, 0 to 1."))
        form.addRow(self.tr("Count low"), ns_lo)
        form.addRow(self.tr("Count high"), ns_hi)
        form.addRow(self.tr("Intensity low"), i_lo)
        form.addRow(self.tr("Intensity high"), i_hi)
        item["params_widgets"] = {"ns_lo": ns_lo, "ns_hi": ns_hi, "i_lo": i_lo, "i_hi": i_hi}
        return item

    def _mk_random_sun_flare_row(self):
        item = self._mk_common_box( op_id="random_sun_flare", title=self.tr("Sun flare"), note=self.tr("Places a short chain of bright circles inside a region of the frame. Coordinates are fractions of width and height."))
        form = item["form"]
        x1 = self._dspin(0, 1, 0.05, 0.2, 2, self.tr("Left edge of the flare region, 0 to 1."))
        y1 = self._dspin(0, 1, 0.05, 0.0, 2, self.tr("Top edge of the flare region, 0 to 1."))
        x2 = self._dspin(0, 1, 0.05, 0.8, 2, self.tr("Right edge of the flare region, 0 to 1."))
        y2 = self._dspin(0, 1, 0.05, 0.5, 2, self.tr("Bottom edge of the flare region, 0 to 1."))
        n_lo = self._ispin(1, 20, 1, 6, self.tr("Fewest flare circles."))
        n_hi = self._ispin(1, 20, 1, 10, self.tr("Most flare circles."))
        src_r = self._ispin(4, 400, 4, 80, self.tr("Spacing and size of the flare circles, in pixels."))
        form.addRow(self.tr("Region left"), x1)
        form.addRow(self.tr("Region top"), y1)
        form.addRow(self.tr("Region right"), x2)
        form.addRow(self.tr("Region bottom"), y2)
        form.addRow(self.tr("Circles low"), n_lo)
        form.addRow(self.tr("Circles high"), n_hi)
        form.addRow(self.tr("Flare size"), src_r)
        item["params_widgets"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "n_lo": n_lo, "n_hi": n_hi, "src_r": src_r}
        return item

    def _mk_brightness_contrast_row(self):
        item = self._mk_common_box( op_id="brightness_contrast", title=self.tr("Brightness and contrast"), note=self.tr("Shifts brightness and contrast by a random amount inside the limits. 0 means no change; 1 is a strong swing."))
        form = item["form"]
        b = self._dspin(0, 1, 0.05, 0.2, 2, self.tr("Largest brightness change, as a fraction of full white."))
        c = self._dspin(0, 1, 0.05, 0.2, 2, self.tr("Largest contrast change. 0.2 allows about ±20%."))
        form.addRow(self.tr("Brightness limit"), b)
        form.addRow(self.tr("Contrast limit"), c)
        item["params_widgets"] = {"b": b, "c": c}
        return item

    def _mk_hsv_row(self):
        item = self._mk_common_box( op_id="hsv", title=self.tr("Color shift"), note=self.tr("Moves hue, saturation, and brightness by a random amount within each limit. Boxes are not moved."))
        form = item["form"]
        h = self._ispin(0, 90, 1, 10, self.tr("Largest hue shift. Hue wraps around the color wheel."))
        s = self._ispin(0, 255, 1, 15, self.tr("Largest saturation shift. 0 is gray, 255 is fully saturated."))
        v = self._ispin(0, 255, 1, 10, self.tr("Largest brightness shift in the color model."))
        form.addRow(self.tr("Hue limit"), h)
        form.addRow(self.tr("Saturation limit"), s)
        form.addRow(self.tr("Value limit"), v)
        item["params_widgets"] = {"h": h, "s": s, "v": v}
        return item

    def _mk_motion_blur_row(self):
        item = self._mk_common_box(
            op_id="motion_blur",
            title=self.tr("Motion blur"),
            note=self.tr(
                "Blurs along a straight streak. Direction is degrees: 0 is horizontal, 90 is vertical, positive angles turn counterclockwise. The low and high bounds pick a random direction; set them equal to keep one direction. Kernel size must be odd."
            ),
        )
        form = item["form"]
        k = self._ispin(3, 51, 2, 7, self.tr("Blur length in pixels. Larger values look like faster motion. Even values are rounded up."))
        angle_lo = self._dspin(-180, 180, 5, 0, 0, self.tr("Smallest motion direction in degrees. 0 is horizontal."))
        angle_hi = self._dspin(-180, 180, 5, 0, 0, self.tr("Largest motion direction in degrees. Keep this greater than or equal to the low value."))
        form.addRow(self.tr("Blur length"), k)
        form.addRow(self.tr("Direction low"), angle_lo)
        form.addRow(self.tr("Direction high"), angle_hi)
        item["params_widgets"] = {"k": k, "angle_lo": angle_lo, "angle_hi": angle_hi}
        return item

    def _mk_gaussian_blur_row(self):
        item = self._mk_common_box( op_id="gaussian_blur", title=self.tr("Gaussian blur"), note=self.tr("Soft focus. A random odd kernel between the two limits is used."))
        form = item["form"]
        kmin = self._ispin(3, 51, 2, 3, self.tr("Smallest blur kernel. Use an odd number."))
        kmax = self._ispin(3, 51, 2, 7, self.tr("Largest blur kernel. Keep this greater than or equal to the minimum."))
        form.addRow(self.tr("Kernel min"), kmin)
        form.addRow(self.tr("Kernel max"), kmax)
        item["params_widgets"] = {"kmin": kmin, "kmax": kmax}
        return item

    def _mk_defocus_row(self):
        item = self._mk_common_box( op_id="defocus", title=self.tr("Defocus"), note=self.tr("Out-of-focus blur. Radius is in pixels; a larger radius spreads each point farther."))
        form = item["form"]
        rmin = self._ispin(1, 30, 1, 2, self.tr("Smallest defocus radius."))
        rmax = self._ispin(1, 30, 1, 4, self.tr("Largest defocus radius."))
        form.addRow(self.tr("Radius min"), rmin)
        form.addRow(self.tr("Radius max"), rmax)
        item["params_widgets"] = {"rmin": rmin, "rmax": rmax}
        return item

    def _mk_gauss_noise_row(self):
        item = self._mk_common_box( op_id="gauss_noise", title=self.tr("Gaussian noise"), note=self.tr("Adds random grain. Variance is the spread of the noise; a higher value looks grainier."))
        form = item["form"]
        vmin = self._dspin(0, 2000, 5, 10, 1, self.tr("Smallest noise variance."))
        vmax = self._dspin(0, 2000, 5, 50, 1, self.tr("Largest noise variance."))
        form.addRow(self.tr("Variance min"), vmin)
        form.addRow(self.tr("Variance max"), vmax)
        item["params_widgets"] = {"vmin": vmin, "vmax": vmax}
        return item

    def _mk_image_compression_row(self):
        item = self._mk_common_box( op_id="image_compression", title=self.tr("JPEG compression"), note=self.tr("Re-saves the image as JPEG. Quality 100 is clean; lower values add block artifacts."))
        form = item["form"]
        qlo = self._ispin(1, 100, 1, 30, self.tr("Lowest JPEG quality. Smaller means stronger artifacts."))
        qhi = self._ispin(1, 100, 1, 95, self.tr("Highest JPEG quality."))
        form.addRow(self.tr("Quality low"), qlo)
        form.addRow(self.tr("Quality high"), qhi)
        item["params_widgets"] = {"qlo": qlo, "qhi": qhi}
        return item

    def _mk_iso_noise_row(self):
        item = self._mk_common_box( op_id="iso_noise", title=self.tr("ISO noise"), note=self.tr("Camera-sensor noise from Albumentations ISONoise. Color shift is a fraction of the hue wheel. Intensity scales the grain. Annotations stay in place."))
        form = item["form"]
        c_lo = self._dspin(0, 0.5, 0.01, 0.01, 2, self.tr("Smallest hue shift, as a fraction of the color wheel."))
        c_hi = self._dspin(0, 0.5, 0.01, 0.05, 2, self.tr("Largest hue shift. Keep this greater than or equal to the low value."))
        i_lo = self._dspin(0, 2, 0.05, 0.1, 2, self.tr("Smallest noise intensity. About 0.5 is clearly visible."))
        i_hi = self._dspin(0, 2, 0.05, 0.5, 2, self.tr("Largest noise intensity."))
        form.addRow(self.tr("Color shift low"), c_lo)
        form.addRow(self.tr("Color shift high"), c_hi)
        form.addRow(self.tr("Intensity low"), i_lo)
        form.addRow(self.tr("Intensity high"), i_hi)
        item["params_widgets"] = {"c_lo": c_lo, "c_hi": c_hi, "i_lo": i_lo, "i_hi": i_hi}
        return item

    def _mk_rgb_shift_row(self):
        item = self._mk_common_box( op_id="rgb_shift", title=self.tr("RGB shift"), note=self.tr("Albumentations RGBShift. Each channel moves by a random amount inside ±limit. Boxes do not move."))
        form = item["form"]
        limit = self._ispin(0, 255, 1, 20, self.tr("Maximum channel shift. 0 leaves the colors unchanged."))
        form.addRow(self.tr("Shift limit"), limit)
        item["params_widgets"] = {"limit": limit}
        return item

    def _mk_gamma_row(self):
        item = self._mk_common_box( op_id="random_gamma", title=self.tr("Random gamma"), note=self.tr("Albumentations RandomGamma. Values are percentages: 100 is unchanged, below 100 brightens midtones, above 100 darkens them."))
        form = item["form"]
        g_lo = self._ispin(20, 300, 5, 80, self.tr("Lowest gamma, in percent. 80 means gamma 0.8."))
        g_hi = self._ispin(20, 300, 5, 120, self.tr("Highest gamma, in percent. Keep this greater than or equal to the low value."))
        form.addRow(self.tr("Gamma low"), g_lo)
        form.addRow(self.tr("Gamma high"), g_hi)
        item["params_widgets"] = {"g_lo": g_lo, "g_hi": g_hi}
        return item

    def _mk_clahe_row(self):
        item = self._mk_common_box( op_id="clahe", title=self.tr("CLAHE"), note=self.tr("Albumentations CLAHE. Contrast is equalized on the lightness channel. A higher clip limit gives stronger local contrast. Tile size is the grid in pixels."))
        form = item["form"]
        clip_lo = self._dspin(0.1, 20, 0.5, 1.0, 1, self.tr("Smallest clip limit. 1 is mild."))
        clip_hi = self._dspin(0.1, 20, 0.5, 4.0, 1, self.tr("Largest clip limit."))
        tile = self._ispin(1, 64, 1, 8, self.tr("Tile grid size. 8 means an 8 by 8 grid."))
        form.addRow(self.tr("Clip low"), clip_lo)
        form.addRow(self.tr("Clip high"), clip_hi)
        form.addRow(self.tr("Tile size"), tile)
        item["params_widgets"] = {"clip_lo": clip_lo, "clip_hi": clip_hi, "tile": tile}
        return item

    def _mk_blur_row(self):
        item = self._mk_common_box( op_id="blur", title=self.tr("Blur"), note=self.tr("Albumentations Blur, a plain box blur. This is separate from Gaussian blur. An odd kernel between the two limits is chosen."))
        form = item["form"]
        kmin = self._ispin(3, 51, 2, 3, self.tr("Smallest box kernel. Use an odd number."))
        kmax = self._ispin(3, 51, 2, 7, self.tr("Largest box kernel."))
        form.addRow(self.tr("Kernel min"), kmin)
        form.addRow(self.tr("Kernel max"), kmax)
        item["params_widgets"] = {"kmin": kmin, "kmax": kmax}
        return item

    def _mk_median_blur_row(self):
        item = self._mk_common_box( op_id="median_blur", title=self.tr("Median blur"), note=self.tr("Albumentations MedianBlur. Replaces each pixel with the median of its neighborhood, which removes salt-and-pepper noise."))
        form = item["form"]
        kmin = self._ispin(3, 51, 2, 3, self.tr("Smallest median kernel. Must be odd."))
        kmax = self._ispin(3, 51, 2, 7, self.tr("Largest median kernel."))
        form.addRow(self.tr("Kernel min"), kmin)
        form.addRow(self.tr("Kernel max"), kmax)
        item["params_widgets"] = {"kmin": kmin, "kmax": kmax}
        return item

    def _mk_coarse_dropout_row(self):
        item = self._mk_common_box( op_id="coarse_dropout", title=self.tr("Coarse dropout"), note=self.tr("Albumentations CoarseDropout. Paints random rectangles with a fill value. Hole size is a fraction of the image. Boxes stay where they were."))
        form = item["form"]
        n_lo = self._ispin(0, 50, 1, 1, self.tr("Fewest rectangles to erase."))
        n_hi = self._ispin(0, 50, 1, 8, self.tr("Most rectangles to erase."))
        h_lo = self._dspin(0.01, 1, 0.01, 0.02, 2, self.tr("Smallest hole height, as a fraction of the image height."))
        h_hi = self._dspin(0.01, 1, 0.01, 0.1, 2, self.tr("Largest hole height, as a fraction of the image height."))
        w_lo = self._dspin(0.01, 1, 0.01, 0.02, 2, self.tr("Smallest hole width, as a fraction of the image width."))
        w_hi = self._dspin(0.01, 1, 0.01, 0.1, 2, self.tr("Largest hole width, as a fraction of the image width."))
        fill = self._ispin(0, 255, 1, 0, self.tr("Gray fill written into each hole. 0 is black."))
        form.addRow(self.tr("Holes low"), n_lo)
        form.addRow(self.tr("Holes high"), n_hi)
        form.addRow(self.tr("Height low"), h_lo)
        form.addRow(self.tr("Height high"), h_hi)
        form.addRow(self.tr("Width low"), w_lo)
        form.addRow(self.tr("Width high"), w_hi)
        form.addRow(self.tr("Fill value"), fill)
        item["params_widgets"] = {"n_lo": n_lo, "n_hi": n_hi, "h_lo": h_lo, "h_hi": h_hi, "w_lo": w_lo, "w_hi": w_hi, "fill": fill}
        return item

    def _mk_elastic_row(self):
        item = self._mk_common_box( op_id="elastic_transform", title=self.tr("Elastic transform"), note=self.tr("Albumentations ElasticTransform. Alpha is the pixel shift, sigma smooths that shift, and affine jitter moves three anchor points. Polygon points follow the warp. Rectangles become the axis-aligned box of the warped corners."))
        form = item["form"]
        alpha = self._dspin(0, 400, 1, 40, 0, self.tr("Displacement magnitude in pixels. 1 is a very small ripple; 40 is a visible bend."))
        sigma = self._dspin(1, 200, 1, 50, 0, self.tr("Smoothness of the displacement. Larger values make broader waves."))
        affine = self._dspin(0, 200, 1, 50, 0, self.tr("Random affine jitter in pixels. 0 disables that part."))
        form.addRow(self.tr("Alpha"), alpha)
        form.addRow(self.tr("Sigma"), sigma)
        form.addRow(self.tr("Affine jitter"), affine)
        item["params_widgets"] = {"alpha": alpha, "sigma": sigma, "affine": affine}
        return item

    def _mk_grid_row(self):
        item = self._mk_common_box( op_id="grid_distortion", title=self.tr("Grid distortion"), note=self.tr("Albumentations GridDistortion. The image is split into a grid and each step is scaled. Distort limit must stay below 1. Annotation points move with the grid."))
        form = item["form"]
        steps = self._ispin(2, 20, 1, 5, self.tr("Number of grid steps on each axis. More steps make smaller cells."))
        limit = self._dspin(0, 0.9, 0.05, 0.3, 2, self.tr("How far each grid step can shrink or stretch. 0.3 means about ±30%."))
        form.addRow(self.tr("Grid steps"), steps)
        form.addRow(self.tr("Distort limit"), limit)
        item["params_widgets"] = {"steps": steps, "limit": limit}
        return item

    def _mk_optical_row(self):
        item = self._mk_common_box( op_id="optical_distortion", title=self.tr("Optical distortion"), note=self.tr("Albumentations OpticalDistortion. Distort limit is the lens coefficient; shift limit moves the optical center as a fraction of width and height. Annotation points follow the lens."))
        form = item["form"]
        distort = self._dspin(0, 0.5, 0.01, 0.05, 2, self.tr("Lens distortion magnitude. 0 is no bend. The sign is chosen at random."))
        shift = self._dspin(0, 0.5, 0.01, 0.05, 2, self.tr("How far the optical center may move, as a fraction of the image size."))
        form.addRow(self.tr("Lens limit"), distort)
        form.addRow(self.tr("Center shift"), shift)
        item["params_widgets"] = {"distort": distort, "shift": shift}
        return item

    def _pick_output_dir(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("Select output folder"))
        if folder:
            self.out_dir_edit.setText(folder)

    def _collect_steps(self) -> List[AugmentStep]:
        steps: List[AugmentStep] = []
        for item in self.ops:
            op = item["op_id"]
            widgets = item["params_widgets"]
            if op == "random_fog":
                params = {"fog_coef_lo": widgets["fog_lo"].value(), "fog_coef_hi": widgets["fog_hi"].value(), "alpha_coef": widgets["alpha"].value()}
            elif op == "random_rain":
                params = {"slant_lower": widgets["sl_lo"].value(), "slant_upper": widgets["sl_hi"].value(), "drop_length": widgets["drop_len"].value(), "drop_width": widgets["drop_w"].value(), "blur_value": widgets["blur_v"].value(), "brightness_coefficient": widgets["bright"].value()}
            elif op == "random_snow":
                params = {"snow_point_lo": widgets["sp_lo"].value(), "snow_point_hi": widgets["sp_hi"].value(), "brightness_coeff": widgets["bc"].value()}
            elif op == "random_shadow":
                params = {"num_shadows_lower": widgets["ns_lo"].value(), "num_shadows_upper": widgets["ns_hi"].value(), "shadow_intensity_lower": widgets["i_lo"].value(), "shadow_intensity_upper": widgets["i_hi"].value()}
            elif op == "random_sun_flare":
                params = {"flare_roi": (widgets["x1"].value(), widgets["y1"].value(), widgets["x2"].value(), widgets["y2"].value()), "num_circles_lower": widgets["n_lo"].value(), "num_circles_upper": widgets["n_hi"].value(), "src_radius": widgets["src_r"].value(), "angle_lower": 0.0, "angle_upper": 1.0}
            elif op == "brightness_contrast":
                params = {"brightness_limit": widgets["b"].value(), "contrast_limit": widgets["c"].value()}
            elif op == "hsv":
                params = {"hue": widgets["h"].value(), "sat": widgets["s"].value(), "val": widgets["v"].value()}
            elif op == "motion_blur":
                params = {"blur_limit": widgets["k"].value(), "angle_lower": widgets["angle_lo"].value(), "angle_upper": widgets["angle_hi"].value()}
            elif op == "gaussian_blur":
                params = {"blur_limit": (widgets["kmin"].value(), widgets["kmax"].value())}
            elif op == "blur":
                params = {"blur_limit": (widgets["kmin"].value(), widgets["kmax"].value())}
            elif op == "median_blur":
                params = {"blur_limit": (widgets["kmin"].value(), widgets["kmax"].value())}
            elif op == "defocus":
                params = {"radius": (widgets["rmin"].value(), widgets["rmax"].value())}
            elif op == "gauss_noise":
                params = {"var_limit": (widgets["vmin"].value(), widgets["vmax"].value())}
            elif op == "iso_noise":
                params = {"color_shift_lo": widgets["c_lo"].value(), "color_shift_hi": widgets["c_hi"].value(), "intensity_lo": widgets["i_lo"].value(), "intensity_hi": widgets["i_hi"].value()}
            elif op == "rgb_shift":
                params = {"shift_limit": widgets["limit"].value()}
            elif op == "random_gamma":
                params = {"gamma_lo": widgets["g_lo"].value(), "gamma_hi": widgets["g_hi"].value()}
            elif op == "clahe":
                params = {"clip_lo": widgets["clip_lo"].value(), "clip_hi": widgets["clip_hi"].value(), "tile_grid": widgets["tile"].value()}
            elif op == "coarse_dropout":
                params = {"holes_lo": widgets["n_lo"].value(), "holes_hi": widgets["n_hi"].value(), "height_lo": widgets["h_lo"].value(), "height_hi": widgets["h_hi"].value(), "width_lo": widgets["w_lo"].value(), "width_hi": widgets["w_hi"].value(), "fill_value": widgets["fill"].value()}
            elif op == "elastic_transform":
                params = {"alpha": widgets["alpha"].value(), "sigma": widgets["sigma"].value(), "alpha_affine": widgets["affine"].value()}
            elif op == "grid_distortion":
                params = {"num_steps": widgets["steps"].value(), "distort_limit": widgets["limit"].value()}
            elif op == "optical_distortion":
                params = {"distort_limit": widgets["distort"].value(), "shift_limit": widgets["shift"].value()}
            elif op == "image_compression":
                params = {"quality_lower": widgets["qlo"].value(), "quality_upper": widgets["qhi"].value()}
            else:
                params = {}
            steps.append(AugmentStep(op, bool(item["cb"].isChecked()), float(item["p"].value()), params))
        return steps

    def _run(self) -> None:
        images = [str(p) for p in (getattr(self._host, "image_list", None) or []) if p]
        if not images:
            QMessageBox.information(self, self.tr("Image Augmentation"), self.tr("Open an image folder first."))
            return
        overwrite = self.rb_overwrite.isChecked()
        out_dir = self.out_dir_edit.text().strip()
        if not overwrite and not out_dir:
            QMessageBox.warning(self, self.tr("Image Augmentation"), self.tr("Select an output folder, or choose overwrite."))
            return
        steps = self._collect_steps()
        if not any(step.enabled for step in steps):
            QMessageBox.warning(self, self.tr("Image Augmentation"), self.tr("Enable at least one augmentation."))
            return
        self.btn_run.setEnabled(False)
        progress = QtWidgets.QProgressDialog(self.tr("Augmenting images..."), self.tr("Cancel"), 0, len(images), self)
        progress.setWindowTitle(self.tr("Image Augmentation"))
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        self._progress = progress
        worker = _AugmentWorker(images, steps, overwrite, None if overwrite else out_dir, lambda path: _label_path(self._host, path))
        self._worker = worker
        progress.canceled.connect(worker.request_cancel)
        worker.progress.connect(lambda i, n, path: (progress.setMaximum(n), progress.setValue(i), progress.setLabelText(path)))
        worker.failed.connect(self._on_fail)
        worker.finished_ok.connect(self._on_done)
        worker.start()
        progress.show()

    def _on_fail(self, message: str) -> None:
        if self._progress:
            self._progress.close()
        self.btn_run.setEnabled(True)
        QMessageBox.warning(self, self.tr("Image Augmentation"), message)

    def _on_done(self, total: int, ok: int) -> None:
        if self._progress:
            self._progress.close()
        self.btn_run.setEnabled(True)
        QMessageBox.information(
            self,
            self.tr("Image Augmentation"),
            self.tr("Checked {total} images. Wrote {ok} augmented images.").format(total=int(total), ok=int(ok)),
        )


def open_image_augmentation_dialog(parent) -> None:
    ImageAugmentationDialog(parent).exec()
