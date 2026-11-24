"""i18n + path helpers for compare / model-eval (bound on LabelingWidget at runtime)."""

from __future__ import annotations

import os.path as osp
from typing import Any, Optional

from PyQt6.QtCore import QCoreApplication, QObject

_COMPARE_CTX = "_CompareWidgetMethods"


class _CompareWidgetMethods(QObject):
    """Qt translation context for review UI installed on LabelingWidget."""

    def _pylupdate_catalog(self) -> None:
        """Strings for pylupdate6 only (not called at runtime)."""
        self.tr("Browse Mode")
        self.tr(
            "Browse mode: keep unreviewed suspect marks and switch images "
            "without confirmation. Uncheck to require confirmation before switching."
        )
        self.tr("Accept This Check")
        self.tr(
            "Accept all suspect marks on this image: keep missed-label boxes "
            "as annotations; keep false-label boxes and clear their flags."
        )
        self.tr("Clear This Check")
        self.tr(
            "Clear all suspect marks on this image: delete missed-label / "
            "false-alarm boxes; keep false-label / miss boxes and remove flags."
        )
        self.tr("Clear Dataset Checks")
        self.tr(
            "Clear compare / model-eval marks on the whole dataset "
            "(same as Advanced → Clear Compare Results)."
        )
        self.tr("Prev Missed")
        self.tr("Next Missed")
        self.tr("Prev False-Label")
        self.tr("Next False-Label")
        self.tr("Open previous image with suspect missed labels (annotation compare)")
        self.tr("Open next image with suspect missed labels (annotation compare)")
        self.tr("Open previous image with suspect false labels (annotation compare)")
        self.tr("Open next image with suspect false labels (annotation compare)")
        self.tr("Mode:")
        self.tr("Shortcuts:")
        self.tr("Settings")
        self.tr("Previous")
        self.tr("Next")
        self.tr("Model Evaluation Review")
        self.tr("Annotation Compare Review")
        self.tr("Next false alarm")
        self.tr("Next miss")
        self.tr("Next missed")
        self.tr("Next false-label")
        self.tr("Browse mode on")
        self.tr("Browse mode off")
        self.tr("Prev False Alarm")
        self.tr("Next False Alarm")
        self.tr("Prev Miss")
        self.tr("Next Miss")
        self.tr("No image loaded.")
        self.tr("No unreviewed compare marks on this image.")
        self.tr("Unreviewed Compare Marks")
        self.tr("Keep markers")
        self.tr("Accept results")
        self.tr("Cancel")
        self.tr(
            "Open previous image with model false alarms (误报)"
        )
        self.tr("Open next image with model false alarms (误报)")
        self.tr("Open previous image with model misses (漏报)")
        self.tr("Open next image with model misses (漏报)")
        self.tr(
            "Open previous image with suspect missed labels"
        )
        self.tr("Open next image with suspect missed labels")
        self.tr("Open previous image with suspect false labels")
        self.tr("Open next image with suspect false labels")
        self.tr(
            "Accepted model-eval results: %d false alarm(s) kept, "
            "%d miss flag(s) cleared."
        )
        self.tr(
            "Accepted compare results: %d missed label(s) kept, "
            "%d false-label flag(s) cleared."
        )
        self.tr(
            "Clear all suspect marks on this image?\n\n"
            "• Delete pred-side marks (漏标/误报): %d\n"
            "• Clear GT-side flags (误标/漏报): %d"
        )
        self.tr(
            "Cleared this image: removed %d pred-side mark(s), "
            "cleared %d GT-side flag(s)."
        )
        self.tr("No more images with false alarms in this dataset.")
        self.tr("Already on the first image with false alarms in this pass.")
        self.tr("Already on the last image with false alarms in this pass.")
        self.tr("No more images with suspect missed labels in this dataset.")
        self.tr(
            "Already on the first image with suspect missed labels in this pass."
        )
        self.tr(
            "Already on the last image with suspect missed labels in this pass."
        )
        self.tr("No more images with misses in this dataset.")
        self.tr("Already on the first image with misses in this pass.")
        self.tr("Already on the last image with misses in this pass.")
        self.tr("No more images with suspect false labels in this dataset.")
        self.tr(
            "Already on the first image with suspect false labels in this pass."
        )
        self.tr(
            "Already on the last image with suspect false labels in this pass."
        )
        self.tr(
            "No more images with suspect missed/false labels in this dataset."
        )
        self.tr("Already on the last image with compare issues in this pass.")
        self.tr(
            "This image still has unreviewed suspect missed/false labels."
        )
        self.tr(
            "Choose an action before switching images:\n"
            "• Keep markers — save and switch to the next image now\n"
            "• Accept results — apply to this image and stay here; "
            "switch manually after you verify"
        )


def tr_compare(source: str) -> str:
    return QCoreApplication.translate(_COMPARE_CTX, source)


def project_browse_start(parent: Optional[Any], current: str = "") -> str:
    """Initial folder for QFileDialog — current field, then open project paths."""
    cur = (current or "").strip()
    if cur:
        if osp.isfile(cur):
            return osp.dirname(osp.abspath(cur))
        if osp.isdir(cur):
            return cur
    if parent is None:
        return ""
    out = getattr(parent, "output_dir", None)
    if out and osp.isdir(str(out)):
        return str(out)
    fn = getattr(parent, "filename", None)
    if fn and osp.isfile(fn):
        return osp.dirname(osp.abspath(fn))
    lod = getattr(parent, "last_open_dir", None)
    if lod and osp.isdir(lod):
        return str(lod)
    return ""