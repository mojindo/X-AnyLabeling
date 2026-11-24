"""Remote Detect training + TensorBoard helpers over SSH."""

from __future__ import annotations

import json
import os
import shlex
import tarfile
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from anylabeling.services.auto_training.mytrain.dataset_lists import (
    SPLIT_FILES,
    build_remote_merge_script,
)
from anylabeling.services.auto_training.mytrain.predict_options import (
    PredictOptions,
)

from .remote_paths import dataset_remote_dir, runs_remote_dir
from .remote_runtime import (
    format_exec_python,
    http_probe_python,
    tensorboard_launcher_candidates,
)
from .ssh_client import SSHClient, SSHError


_UPLOAD_MANIFEST = ".xal_upload_manifest.json"
_BUNDLE_LABELS_SPLITS = "_xal_labels_splits.tar.gz"
_BUNDLE_IMAGES = "_xal_images_bulk.tar.gz"
_SYNC_BUNDLE_NAMES = (_BUNDLE_LABELS_SPLITS, _BUNDLE_IMAGES)
_BULK_IMAGE_CHANGE_RATIO = 0.6
_BULK_IMAGE_MIN_CHANGES = 50

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


class UploadStopped(Exception):
    """User stopped a dataset upload. A later run must upload again."""


@dataclass
class UploadProgressReporter:
    """Counts discrete sync steps for overall / per-subdir percentages."""

    total: int
    per_subdir: Dict[str, int]
    done: int = 0
    sub_done: Dict[str, int] = field(default_factory=dict)
    callback: Optional[Callable[[float, str, str, float], None]] = None

    def step(self, detail: str, *, sd: str = "", n: int = 1) -> None:
        self.done = min(self.total, self.done + int(n))
        overall = 100.0 * self.done / max(1, self.total)
        sub_pct = -1.0
        if sd and sd in self.per_subdir:
            cur = min(self.per_subdir[sd], self.sub_done.get(sd, 0) + int(n))
            self.sub_done[sd] = cur
            sub_pct = 100.0 * cur / max(1, self.per_subdir[sd])
        if self.callback:
            self.callback(overall, detail, sd or "", sub_pct)

    def notify(
        self,
        detail: str,
        *,
        sd: str = "",
        partial: float = 0.0,
    ) -> None:
        """Update status without advancing steps; ``partial`` is 0–1 into the next step."""
        partial = max(0.0, min(1.0, float(partial)))
        overall = 100.0 * (self.done + partial) / max(1, self.total)
        sub_pct = -1.0
        if sd and sd in self.per_subdir:
            sub_base = self.sub_done.get(sd, 0)
            sub_pct = 100.0 * (sub_base + partial) / max(
                1, self.per_subdir[sd]
            )
        if self.callback:
            self.callback(overall, detail, sd or "", sub_pct)


def _plan_upload_workload(
    root: Path,
    subdirs: Sequence[str],
    entries: Dict[str, Dict[str, int]],
    orphans: Sequence[str],
    manifest_files: dict,
    remote_sizes: Optional[Dict[str, int]],
) -> Tuple[int, Dict[str, int]]:
    total = len(orphans) + 1 + 3
    per: Dict[str, int] = {}
    for sd in subdirs:
        local_sub = root / sd
        sub = 0
        if not local_sub.is_dir():
            per[sd] = 0
            continue
        sub += sum(
            1
            for rel in entries
            if rel.startswith(f"{sd}/") and "/images/" not in rel
        )
        prefix = f"{sd}/images/"
        if _subdir_bulk_images_recommended(
            sd, entries, manifest_files, remote_sizes, orphans
        ):
            sub += 1
        else:
            for rel, meta in entries.items():
                if not rel.startswith(prefix):
                    continue
                if _image_needs_upload(
                    rel, meta, manifest_files, remote_sizes
                ):
                    sub += 1
        per[sd] = sub
        total += sub
    return max(1, total), per


class UploadGate:
    """Pause blocks the next file; stop aborts and cannot be resumed."""

    def __init__(self) -> None:
        self._resume = threading.Event()
        self._resume.set()
        self._stop = threading.Event()

    def pause(self) -> None:
        if not self._stop.is_set():
            self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def checkpoint(self) -> None:
        while not self._resume.is_set():
            if self._stop.is_set():
                raise UploadStopped("Upload stopped")
            self._resume.wait(0.2)
        if self._stop.is_set():
            raise UploadStopped("Upload stopped")


def _unlink_quiet(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.unlink(path)
    except OSError:
        pass


def _file_stat_entry(full: Path) -> Dict[str, int]:
    st = full.stat()
    return {
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
    }


def _local_upload_entries(
    local_root: str, subdirs: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    """Uploadable files: images (no JSON), labels/*.txt, split lists, classes.txt."""
    root = Path(local_root)
    entries: Dict[str, Dict[str, int]] = {}
    classes = root / "classes.txt"
    if classes.is_file():
        entries["classes.txt"] = _file_stat_entry(classes)
    for sd in subdirs:
        sub = root / sd
        if not sub.is_dir():
            continue
        for split in SPLIT_FILES:
            sp = sub / split
            if sp.is_file():
                entries[f"{sd}/{split}"] = _file_stat_entry(sp)
        labels_dir = sub / "labels"
        if labels_dir.is_dir():
            for txt in sorted(labels_dir.glob("*.txt")):
                if txt.is_file():
                    rel = f"{sd}/labels/{txt.name}"
                    entries[rel] = _file_stat_entry(txt)
        images_dir = sub / "images"
        if images_dir.is_dir():
            for dirpath, dirnames, filenames in os.walk(images_dir):
                dirnames[:] = [d for d in dirnames if d and not d.startswith(".")]
                for name in filenames:
                    if not name or name.startswith("."):
                        continue
                    if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                        continue
                    full = Path(dirpath) / name
                    if full.is_file():
                        entries[full.relative_to(root).as_posix()] = _file_stat_entry(
                            full
                        )
    return entries


def _image_needs_upload(
    rel: str,
    meta: Dict[str, int],
    manifest_files: dict,
    remote_sizes: Optional[Dict[str, int]],
) -> bool:
    if not remote_sizes or rel not in remote_sizes:
        return True
    if int(remote_sizes[rel]) != int(meta["size"]):
        return True
    prev = manifest_files.get(rel) if isinstance(manifest_files, dict) else None
    if not isinstance(prev, dict):
        return True
    return int(prev.get("mtime_ns", -1)) != int(meta["mtime_ns"])


def _local_image_rels(entries: Dict[str, Dict[str, int]]) -> set:
    return {rel for rel in entries if "/images/" in rel}


def _remote_image_rels_to_delete(
    entries: Dict[str, Dict[str, int]],
    remote_sizes: Optional[Dict[str, int]],
    subdirs: Sequence[str],
) -> List[str]:
    """Remote image paths that no longer exist locally."""
    if not remote_sizes:
        return []
    local_images = _local_image_rels(entries)
    prefixes = tuple(f"{sd}/images/" for sd in subdirs)
    out: List[str] = []
    for rel in remote_sizes:
        if rel not in local_images and rel.startswith(prefixes):
            out.append(rel)
    return sorted(out)


def _delete_remote_files(
    client: SSHClient,
    remote_root: str,
    rels: Sequence[str],
    *,
    progress: Optional[Callable[[str], None]] = None,
    tick: Optional[Callable[[], None]] = None,
    reporter: Optional[UploadProgressReporter] = None,
) -> None:
    base = remote_root.rstrip("/")
    for rel in rels:
        if tick is not None:
            tick()
        rp = f"{base}/{rel}"
        sd = rel.split("/", 1)[0] if "/" in rel else ""
        detail = f"deleted {rel}"
        if reporter:
            reporter.step(detail, sd=sd)
        elif progress:
            progress(detail)
        try:
            client.exec(f"rm -f {shlex.quote(rp)}", timeout=60)
        except Exception:
            pass


def _build_labels_splits_tar(local_sub: Path, dest_tgz: str) -> bool:
    """Pack split lists and labels/*.txt for one sub-dataset."""
    added = 0
    with tarfile.open(dest_tgz, "w:gz") as tar:
        for split in SPLIT_FILES:
            p = local_sub / split
            if p.is_file():
                tar.add(p, arcname=split)
                added += 1
        labels = local_sub / "labels"
        if labels.is_dir():
            for txt in sorted(labels.glob("*.txt")):
                if txt.is_file():
                    tar.add(txt, arcname=f"labels/{txt.name}")
                    added += 1
    if added == 0:
        try:
            os.unlink(dest_tgz)
        except OSError:
            pass
        return False
    return True


def _bulk_step_partial(*, pack: float = 0.0, upload: float = 0.0) -> float:
    """Map pack/upload sub-phases (each 0–1) into one step's partial progress."""
    pack = max(0.0, min(1.0, pack))
    upload = max(0.0, min(1.0, upload))
    if upload > 0.0 or pack >= 1.0:
        return min(1.0, 0.18 + 0.78 * upload)
    return 0.18 * pack


def _build_images_tar(
    local_sub: Path,
    dest_tgz: str,
    *,
    total_hint: int = 0,
    on_progress: Optional[Callable[[int, int], None]] = None,
    tick: Optional[Callable[[], None]] = None,
) -> bool:
    """Pack images/ (recognized extensions only) for bulk upload."""
    images = local_sub / "images"
    if not images.is_dir():
        return False
    added = 0
    last_pct = [-1]
    with tarfile.open(dest_tgz, "w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(images):
            dirnames[:] = [d for d in dirnames if d and not d.startswith(".")]
            for name in filenames:
                if tick is not None and added % 40 == 0:
                    tick()
                if not name or name.startswith("."):
                    continue
                if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                    continue
                full = Path(dirpath) / name
                if not full.is_file():
                    continue
                arc = full.relative_to(local_sub).as_posix()
                tar.add(full, arcname=arc)
                added += 1
                if on_progress is not None:
                    total = total_hint if total_hint > 0 else max(added, 1)
                    pct = int(100 * added / total)
                    if pct // 5 > last_pct[0] // 5 or added >= total:
                        last_pct[0] = pct
                        on_progress(added, total)
    if added == 0:
        try:
            os.unlink(dest_tgz)
        except OSError:
            pass
        return False
    return True


def _subdir_bulk_images_recommended(
    sd: str,
    entries: Dict[str, Dict[str, int]],
    manifest_files: dict,
    remote_sizes: Optional[Dict[str, int]],
    orphan_rels: Sequence[str],
) -> bool:
    prefix = f"{sd}/images/"
    local_images = [rel for rel in entries if rel.startswith(prefix)]
    n = len(local_images)
    if n == 0:
        return False
    remote_count = sum(1 for r in (remote_sizes or {}) if r.startswith(prefix))
    if remote_count == 0:
        return True
    to_upload = 0
    for rel in local_images:
        meta = entries.get(rel)
        if meta and _image_needs_upload(
            rel, meta, manifest_files, remote_sizes
        ):
            to_upload += 1
    to_delete = sum(1 for rel in orphan_rels if rel.startswith(prefix))
    changes = to_upload + to_delete
    threshold = max(_BULK_IMAGE_MIN_CHANGES, int(n * _BULK_IMAGE_CHANGE_RATIO))
    return changes >= threshold


def _extract_tarball_on_remote(
    client: SSHClient,
    *,
    remote_archive: str,
    extract_dir: str,
    remove_first: Sequence[str],
    remote_python: str = "",
    tick: Optional[Callable[[], None]] = None,
) -> None:
    """Extract ``.tar.gz`` on the server; remove listed paths under ``extract_dir`` first."""
    dest = extract_dir.rstrip("/")
    archive_q = shlex.quote(remote_archive)
    dest_q = shlex.quote(dest)
    if tick is not None:
        tick()
    for name in remove_first:
        path = f"{dest}/{name}"
        if name.endswith("/") or name in ("labels", "images"):
            cmd = f"rm -rf {shlex.quote(path)}"
        else:
            cmd = f"rm -f {shlex.quote(path)}"
        try:
            client.exec(cmd, timeout=120)
        except Exception:
            pass
    shell_extract = (
        f"set -e; cd {dest_q} && tar -xzf {archive_q} && rm -f {archive_q}"
    )
    code, out, err = client.exec(shell_extract, timeout=3600)
    if code == 0:
        return
    script = """# -*- coding: utf-8 -*-
import json, os, shutil, sys, tarfile
archive, dest, remove = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
for name in remove:
    path = os.path.join(dest, name)
    if name.endswith("/") or os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
    else:
        shutil.rmtree(path, ignore_errors=True)
try:
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest)
finally:
    try:
        os.remove(archive)
    except OSError:
        pass
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_xal_extract.py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        local_script = f.name
    remote_script = f"{extract_dir.rstrip('/')}/_xal_extract_bundle.py"
    try:
        if tick is not None:
            tick()
        client.upload_file(local_script, remote_script)
        rm_json = json.dumps(list(remove_first))
        script_q = shlex.quote(remote_script)
        cmd_tail = (
            f"{script_q} {shlex.quote(remote_archive)} "
            f"{shlex.quote(extract_dir)} {shlex.quote(rm_json)}"
        )
        py = (remote_python or "").strip()
        if py:
            cmd = f"{shlex.quote(py)} {cmd_tail}"
        else:
            cmd = f"python3 {cmd_tail} || python {cmd_tail}"
        code, out, err = client.exec(cmd, timeout=3600)
        client.exec(f"rm -f {shlex.quote(remote_script)}", timeout=60)
        if code != 0:
            try:
                client.exec(f"rm -f {shlex.quote(remote_archive)}", timeout=60)
            except Exception:
                pass
            text = ((out or "") + (err or "")).strip()
            raise SSHError(text or f"Remote extract failed (code {code})")
    finally:
        _unlink_quiet(local_script)


def _purge_remote_sync_bundles(
    client: SSHClient, remote_root: str, subdirs: Sequence[str]
) -> None:
    """Remove leftover transport tarballs under each selected sub-dataset."""
    base = remote_root.rstrip("/")
    for sd in subdirs:
        remote_sub = f"{base}/{sd}"
        for name in _SYNC_BUNDLE_NAMES:
            try:
                client.exec(
                    f"rm -f {shlex.quote(remote_sub + '/' + name)}",
                    timeout=60,
                )
            except Exception:
                pass
        try:
            client.exec(
                f"rm -f {shlex.quote(remote_sub + '/_xal_extract_bundle.py')}",
                timeout=60,
            )
        except Exception:
            pass


def _verify_remote_fingerprint(
    client: SSHClient,
    remote_root: str,
    subdirs: Sequence[str],
    entries: Dict[str, Dict[str, int]],
    *,
    remote_python: str = "",
) -> bool:
    """Re-check remote file sizes match expected upload set."""
    remote_sizes = _remote_selected_sizes(
        client, remote_root, subdirs, remote_python=remote_python
    )
    if remote_sizes is None:
        return False
    local_sizes = {rel: int(meta["size"]) for rel, meta in entries.items()}
    return remote_sizes == local_sizes


def _upload_bundle(
    client: SSHClient,
    *,
    local_tgz: str,
    remote_sub: str,
    remote_name: str,
    extract_dir: str,
    remove_first: Sequence[str],
    remote_python: str,
    tick: Optional[Callable[[], None]] = None,
    progress: Optional[Callable[[str], None]] = None,
    reporter: Optional[UploadProgressReporter] = None,
    reporter_sd: str = "",
    reporter_detail: str = "",
    reporter_detail_upload: str = "",
    bulk_partial: bool = False,
) -> None:
    remote_archive = f"{remote_sub.rstrip('/')}/{remote_name}"
    try:
        if reporter and reporter_detail_upload:
            reporter.notify(
                reporter_detail_upload,
                sd=reporter_sd,
                partial=_bulk_step_partial(upload=0.01) if bulk_partial else 0.0,
            )
        elif progress:
            progress(remote_archive)
        if tick is not None:
            tick()
        last_pct = [-1]

        def _upload_progress(done: int, total: int) -> None:
            if total <= 0 or reporter is None:
                return
            pct = int(100 * done / total)
            if pct // 5 > last_pct[0] // 5 or done >= total:
                last_pct[0] = pct
                upload_frac = done / total
                partial = (
                    _bulk_step_partial(upload=upload_frac)
                    if bulk_partial
                    else 0.0
                )
                reporter.notify(
                    f"{reporter_detail_upload or reporter_detail} ({pct}%)",
                    sd=reporter_sd,
                    partial=partial,
                )

        client.upload_file(
            local_tgz,
            remote_archive,
            progress_callback=_upload_progress
            if reporter is not None
            else None,
            tick=tick,
        )
        if reporter and bulk_partial:
            reporter.notify(
                f"extract {reporter_sd or remote_name}",
                sd=reporter_sd,
                partial=_bulk_step_partial(upload=0.99),
            )
        _extract_tarball_on_remote(
            client,
            remote_archive=remote_archive,
            extract_dir=extract_dir,
            remove_first=remove_first,
            remote_python=remote_python,
            tick=tick,
        )
        if reporter:
            reporter.step(
                reporter_detail or remote_name,
                sd=reporter_sd,
            )
    finally:
        _unlink_quiet(local_tgz)


def _purge_remote_images_json(
    client: SSHClient, remote_root: str, subdirs: Sequence[str]
) -> None:
    for sd in subdirs:
        rs = f"{remote_root.rstrip('/')}/{sd}/images"
        client.exec(
            "find "
            + shlex.quote(rs)
            + " -type f -iname '*.json' -delete 2>/dev/null || true",
            timeout=120,
        )


def _remote_selected_sizes(
    client: SSHClient,
    remote_root: str,
    subdirs: Sequence[str],
    *,
    remote_python: str = "",
) -> Optional[Dict[str, int]]:
    """Size of classes.txt and each selected subdir already on the server."""
    script = """# -*- coding: utf-8 -*-
import json, os, sys
root = sys.argv[1]
subdirs = json.loads(sys.argv[2])
SPLIT_FILES = ("train.txt", "val.txt", "test.txt", "all.txt")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
out = {}
classes = os.path.join(root, "classes.txt")
if os.path.isfile(classes):
    out["classes.txt"] = os.path.getsize(classes)
for sd in subdirs:
    base = os.path.join(root, sd)
    if not os.path.isdir(base):
        continue
    for split in SPLIT_FILES:
        p = os.path.join(base, split)
        if os.path.isfile(p):
            rel = f"{sd}/{split}"
            out[rel] = os.path.getsize(p)
    labels = os.path.join(base, "labels")
    if os.path.isdir(labels):
        for name in os.listdir(labels):
            if not name.endswith(".txt"):
                continue
            full = os.path.join(labels, name)
            if os.path.isfile(full):
                rel = f"{sd}/labels/{name}"
                out[rel] = os.path.getsize(full)
    images = os.path.join(base, "images")
    if os.path.isdir(images):
        for dirpath, dirnames, filenames in os.walk(images):
            dirnames[:] = [d for d in dirnames if d and not d.startswith(".")]
            for name in filenames:
                if not name or name.startswith("."):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext not in IMAGE_EXTS:
                    continue
                full = os.path.join(dirpath, name)
                if not os.path.isfile(full):
                    continue
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                try:
                    out[rel] = os.path.getsize(full)
                except OSError:
                    pass
print(json.dumps(out))
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_xal_sizes.py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        local_script = f.name
    remote_script = f"{remote_root.rstrip('/')}/_xal_sizes.py"
    try:
        client.mkdir_p(remote_root)
        client.upload_file(local_script, remote_script)
        script_q = shlex.quote(remote_script)
        arg_s = (
            shlex.quote(remote_root)
            + " "
            + shlex.quote(json.dumps(list(subdirs)))
        )
        py = (remote_python or "").strip()
        if py:
            cmd = f"{shlex.quote(py)} {script_q} {arg_s}"
        else:
            cmd = (
                f"python3 {script_q} {arg_s} || python {script_q} {arg_s}"
            )
        code, out, err = client.exec(cmd, timeout=300)
        client.exec(f"rm -f {shlex.quote(remote_script)}")
        if code != 0:
            return None
        text = (out or "").strip().splitlines()
        if not text:
            return {}
        data = json.loads(text[-1])
        if not isinstance(data, dict):
            return None
        return {str(k): int(v) for k, v in data.items()}
    except Exception:
        return None
    finally:
        try:
            os.unlink(local_script)
        except OSError:
            pass


def _read_remote_manifest(
    client: SSHClient, remote_root: str
) -> Optional[dict]:
    path = f"{remote_root.rstrip('/')}/{_UPLOAD_MANIFEST}"
    try:
        raw = client.read_text(path, max_bytes=8_000_000)
    except Exception:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _payload_unchanged(
    local_entries: Dict[str, Dict[str, int]],
    remote_sizes: Optional[Dict[str, int]],
    manifest: Optional[dict],
) -> bool:
    if remote_sizes is None:
        return False
    local_sizes = {rel: meta["size"] for rel, meta in local_entries.items()}
    if local_sizes != remote_sizes:
        return False
    if not manifest:
        return True
    old = manifest.get("files") or {}
    if not isinstance(old, dict) or set(old) != set(local_entries):
        return False
    for rel, meta in local_entries.items():
        prev = old.get(rel) or {}
        if not isinstance(prev, dict):
            return False
        if int(prev.get("size", -1)) != int(meta["size"]):
            return False
        if int(prev.get("mtime_ns", -1)) != int(meta["mtime_ns"]):
            return False
    return True


def _write_remote_manifest(
    client: SSHClient,
    remote_root: str,
    subdirs: Sequence[str],
    entries: Dict[str, Dict[str, int]],
) -> None:
    payload = {
        "version": 1,
        "subdirs": list(subdirs),
        "files": entries,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_xal_manifest.json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(payload, f)
        local_path = f.name
    try:
        client.upload_file(
            local_path, f"{remote_root.rstrip('/')}/{_UPLOAD_MANIFEST}"
        )
    finally:
        try:
            os.unlink(local_path)
        except OSError:
            pass


def upload_dataset(
    client: SSHClient,
    local_root: str,
    *,
    subdirs: Sequence[str],
    progress: Optional[Callable[[str], None]] = None,
    progress_state: Optional[Callable[[float, str, str, float], None]] = None,
    remote_base: str = "",
    remote_python: str = "",
    control: Optional[UploadGate] = None,
) -> Tuple[str, bool]:
    """Sync subdirs: bundled labels+splits, incremental or bulk images.

    Labels and split txts ship as one ``.tar.gz`` per sub-dataset; images are
    uploaded one-by-one unless most of the sub-dataset changed (then one tarball).
    Does not upload LabelMe JSON under images/. Returns ``(remote_path, skipped)``.
    """
    remote = dataset_remote_dir(local_root, client.home, remote_base)
    entries = _local_upload_entries(local_root, subdirs)
    if "classes.txt" not in entries:
        raise SSHError(f"classes.txt not found under {local_root}")
    if not any(rel != "classes.txt" for rel in entries):
        raise SSHError("No files found in the selected sub-datasets")

    def _tick() -> None:
        if control is not None:
            control.checkpoint()

    _tick()
    client.mkdir_p(remote)
    remote_sizes = _remote_selected_sizes(
        client, remote, subdirs, remote_python=remote_python
    )
    manifest = _read_remote_manifest(client, remote)
    if _payload_unchanged(entries, remote_sizes, manifest):
        if not manifest:
            _write_remote_manifest(client, remote, subdirs, entries)
        if progress_state:
            progress_state(100.0, f"unchanged {remote}", "", -1.0)
        elif progress:
            progress(f"unchanged {remote}")
        return remote, True

    manifest_files = (
        (manifest.get("files") or {}) if isinstance(manifest, dict) else {}
    )
    root = Path(local_root)
    remote_root = remote.rstrip("/")
    orphans = _remote_image_rels_to_delete(entries, remote_sizes, subdirs)
    total_steps, per_sd = _plan_upload_workload(
        root, subdirs, entries, orphans, manifest_files, remote_sizes
    )
    reporter = UploadProgressReporter(
        total_steps, per_sd, callback=progress_state
    )

    def _put(
        local_path: str,
        remote_path: str,
        *,
        sd: str = "",
        rel_key: str = "",
    ) -> None:
        _tick()
        rel = rel_key
        if not rel and remote_path.startswith(remote_root + "/"):
            rel = remote_path[len(remote_root) + 1 :]
        if not rel:
            rel = remote_path
        sub = sd or (rel.split("/", 1)[0] if "/" in rel else "")
        if reporter.callback:
            reporter.step(rel, sd=sub)
        elif progress:
            progress(remote_path)
        client.upload_file(local_path, remote_path)

    if orphans:
        _delete_remote_files(
            client,
            remote_root,
            orphans,
            progress=progress if not progress_state else None,
            tick=_tick,
            reporter=reporter if progress_state else None,
        )

    classes_path = root / "classes.txt"
    if classes_path.is_file():
        _put(
            str(classes_path),
            f"{remote_root}/classes.txt",
            rel_key="classes.txt",
        )

    local_tmp_bundles: List[str] = []

    def _local_bundle_file() -> str:
        fd, path = tempfile.mkstemp(suffix=".tar.gz")
        os.close(fd)
        local_tmp_bundles.append(path)
        return path

    try:
        for sd in subdirs:
            _tick()
            local_sub = root / sd
            if not local_sub.is_dir():
                continue
            remote_sub = f"{remote_root}/{sd}"
            client.mkdir_p(remote_sub)
            remove_splits = [s for s in SPLIT_FILES if (local_sub / s).is_file()]

            bundle_ls = _local_bundle_file()
            if progress_state:
                reporter.notify(f"packing {sd} labels+splits", sd=sd)
            if _build_labels_splits_tar(local_sub, bundle_ls):
                _upload_bundle(
                    client,
                    local_tgz=bundle_ls,
                    remote_sub=remote_sub,
                    remote_name=_BUNDLE_LABELS_SPLITS,
                    extract_dir=remote_sub,
                    remove_first=["labels"] + remove_splits,
                    remote_python=remote_python,
                    tick=_tick,
                    progress=progress if not progress_state else None,
                    reporter=reporter if progress_state else None,
                    reporter_sd=sd,
                    reporter_detail=f"bundle {sd} labels+splits",
                    reporter_detail_upload=f"upload {sd} labels+splits",
                )
            else:
                _unlink_quiet(bundle_ls)
                if bundle_ls in local_tmp_bundles:
                    local_tmp_bundles.remove(bundle_ls)

            bulk_images = _subdir_bulk_images_recommended(
                sd, entries, manifest_files, remote_sizes, orphans
            )
            images_bulk_done = False
            if bulk_images:
                bundle_im = _local_bundle_file()
                img_prefix = f"{sd}/images/"
                img_total = sum(
                    1 for rel in entries if rel.startswith(img_prefix)
                )

                def _pack_progress(added: int, total: int) -> None:
                    if not progress_state:
                        return
                    pct = int(100 * added / max(1, total))
                    reporter.notify(
                        f"packing {sd} images (bulk) {added}/{total} ({pct}%)",
                        sd=sd,
                        partial=_bulk_step_partial(
                            pack=added / max(1, total)
                        ),
                    )

                if progress_state:
                    reporter.notify(
                        f"packing {sd} images (bulk)",
                        sd=sd,
                        partial=0.0,
                    )
                if _build_images_tar(
                    local_sub,
                    bundle_im,
                    total_hint=img_total,
                    on_progress=_pack_progress if progress_state else None,
                    tick=_tick,
                ):
                    _upload_bundle(
                        client,
                        local_tgz=bundle_im,
                        remote_sub=remote_sub,
                        remote_name=_BUNDLE_IMAGES,
                        extract_dir=remote_sub,
                        remove_first=["images"],
                        remote_python=remote_python,
                        tick=_tick,
                        progress=progress if not progress_state else None,
                        reporter=reporter if progress_state else None,
                        reporter_sd=sd,
                        reporter_detail=f"bundle {sd} images (bulk)",
                        reporter_detail_upload=f"upload {sd} images (bulk)",
                        bulk_partial=bool(progress_state),
                    )
                    images_bulk_done = True
                else:
                    _unlink_quiet(bundle_im)
                    if bundle_im in local_tmp_bundles:
                        local_tmp_bundles.remove(bundle_im)
            if not images_bulk_done:
                client.mkdir_p(f"{remote_sub}/images")
                images_dir = local_sub / "images"
                if images_dir.is_dir():
                    if progress_state:
                        reporter.notify(f"sync {sd} images", sd=sd)
                    for dirpath, dirnames, filenames in os.walk(images_dir):
                        dirnames[:] = [
                            d for d in dirnames if d and not d.startswith(".")
                        ]
                        for name in filenames:
                            if not name or name.startswith("."):
                                continue
                            if (
                                os.path.splitext(name)[1].lower()
                                not in IMAGE_EXTS
                            ):
                                continue
                            full = Path(dirpath) / name
                            if not full.is_file():
                                continue
                            rel = full.relative_to(root).as_posix()
                            meta = entries.get(rel)
                            if meta is None:
                                continue
                            if _image_needs_upload(
                                rel, meta, manifest_files, remote_sizes
                            ):
                                rp = f"{remote_root}/{rel}"
                                _put(str(full), rp, sd=sd)

        _tick()
        _purge_remote_images_json(client, remote_root, subdirs)
        if progress_state:
            reporter.step("purge LabelMe JSON on remote")
        _tick()
        if not _verify_remote_fingerprint(
            client, remote, subdirs, entries, remote_python=remote_python
        ):
            raise SSHError(
                "Remote dataset fingerprint mismatch after sync; manifest not updated."
            )
        if progress_state:
            reporter.step("verify remote fingerprint")
        _purge_remote_sync_bundles(client, remote_root, subdirs)
        _write_remote_manifest(client, remote, subdirs, entries)
        if progress_state:
            reporter.step(f"synced {remote}")
        elif progress:
            progress(f"synced {remote}")
        return remote, False
    finally:
        for path in local_tmp_bundles:
            _unlink_quiet(path)
        try:
            _purge_remote_sync_bundles(client, remote_root, subdirs)
        except Exception:
            pass


def _run_remote_json_script(
    client: SSHClient,
    script_body: str,
    *,
    args: Sequence[str],
    remote_workdir: str,
    remote_python: str = "",
    timeout: int = 180,
) -> Tuple[Optional[dict], str]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_xal_dataset_probe.py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script_body)
        local_script = f.name
    remote_script = f"{remote_workdir.rstrip('/')}/_xal_dataset_probe.py"
    try:
        client.mkdir_p(remote_workdir)
        client.upload_file(local_script, remote_script)
        script_q = shlex.quote(remote_script)
        arg_s = " ".join(shlex.quote(a) for a in args)
        py = (remote_python or "").strip()
        if py:
            cmd = f"{shlex.quote(py)} {script_q} {arg_s}"
        else:
            cmd = f"python3 {script_q} {arg_s} || python {script_q} {arg_s}"
        code, out, err = client.exec(cmd, timeout=timeout)
        client.exec(f"rm -f {shlex.quote(remote_script)}", timeout=60)
        text = (out or "").strip()
        if not text:
            return None, (err or "").strip() or f"probe failed (code {code})"
        try:
            data = json.loads(text.splitlines()[-1])
        except json.JSONDecodeError:
            return None, text or (err or "").strip()
        if not isinstance(data, dict):
            return None, "invalid probe response"
        return data, ""
    finally:
        _unlink_quiet(local_script)


def probe_remote_dataset_subdirs(
    client: SSHClient,
    remote_root: str,
    *,
    remote_python: str = "",
) -> Tuple[bool, str, List[str]]:
    """Return candidate sub-version folder names under a remote dataset ROOT."""
    script = """# -*- coding: utf-8 -*-
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
if not root.is_dir():
    print(json.dumps({"ok": False, "error": "Dataset root does not exist: %s" % root}))
    raise SystemExit(0)
if not (root / "classes.txt").is_file():
    print(json.dumps({"ok": False, "error": "Missing classes.txt: %s" % (root / "classes.txt")}))
    raise SystemExit(0)
SPLIT_FILES = ("train.txt", "val.txt", "test.txt", "all.txt")
names = []
for child in sorted(root.iterdir()):
    if not child.is_dir():
        continue
    if child.name.startswith(".") or child.name.startswith("_"):
        continue
    if (child / "images").is_dir() or any((child / s).is_file() for s in SPLIT_FILES):
        names.append(child.name)
print(json.dumps({"ok": True, "subdirs": names}))
"""
    data, err = _run_remote_json_script(
        client,
        script,
        args=[remote_root],
        remote_workdir=remote_root,
        remote_python=remote_python,
    )
    if data is None:
        return False, err or "Remote probe failed", []
    if not data.get("ok"):
        return False, str(data.get("error") or "Invalid remote dataset root"), []
    raw = data.get("subdirs") or []
    subs = [str(x) for x in raw if isinstance(x, str) and x.strip()]
    return True, "", subs


def validate_remote_dataset_root(
    client: SSHClient,
    remote_root: str,
    subdirs: Sequence[str],
    *,
    remote_python: str = "",
) -> Tuple[bool, str]:
    script = """# -*- coding: utf-8 -*-
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
subdirs = json.loads(sys.argv[2])
SPLIT_FILES = ("train.txt", "val.txt", "test.txt", "all.txt")
if not root.is_dir():
    print(json.dumps({"ok": False, "error": "Dataset root does not exist: %s" % root}))
    raise SystemExit(0)
classes = root / "classes.txt"
if not classes.is_file():
    print(json.dumps({"ok": False, "error": "Missing classes.txt"}))
    raise SystemExit(0)
if not [ln.strip() for ln in classes.read_text(encoding="utf-8").splitlines() if ln.strip()]:
    print(json.dumps({"ok": False, "error": "No valid class names in classes.txt"}))
    raise SystemExit(0)
if not subdirs:
    print(json.dumps({"ok": False, "error": "Please select at least one sub-dataset version"}))
    raise SystemExit(0)
for sd in subdirs:
    sub = root / sd
    if not sub.is_dir():
        print(json.dumps({"ok": False, "error": "Sub-directory does not exist: %s" % sub}))
        raise SystemExit(0)
    for split in SPLIT_FILES:
        p = sub / split
        if not p.is_file():
            print(json.dumps({"ok": False, "error": "Missing %s" % p}))
            raise SystemExit(0)
print(json.dumps({"ok": True}))
"""
    data, err = _run_remote_json_script(
        client,
        script,
        args=[remote_root, json.dumps(list(subdirs))],
        remote_workdir=remote_root,
        remote_python=remote_python,
    )
    if data is None:
        return False, err or "Remote validation failed"
    if not data.get("ok"):
        return False, str(data.get("error") or "Remote dataset invalid")
    return True, ""


def _exec_remote_script_file(
    client: SSHClient,
    remote_script: str,
    *,
    remote_python: Optional[str] = None,
    timeout: int = 600,
) -> Tuple[int, str, str]:
    """Run a remote ``.py`` file; return (exit_code, merged_output, shell_cmd)."""
    script_q = shlex.quote(remote_script)
    py = (remote_python or "").strip()
    if py:
        inner = f"{shlex.quote(py)} -u {script_q}"
    else:
        inner = f"python3 -u {script_q} || python -u {script_q}"
    shell_cmd = (
        f"echo XAL_EXEC_BEGIN; ({inner}) 2>&1; "
        f"ec=$?; echo XAL_EXEC_EXIT=$ec; exit $ec"
    )
    full = f"bash -c {shlex.quote(shell_cmd)}"
    code, out, err = client.exec(full, timeout=timeout, get_pty=True)
    merged = "\n".join(
        part for part in ((out or "").strip(), (err or "").strip()) if part
    )
    return code, merged, full


def _parse_dataset_check_lines(text: str) -> List[Dict[str, object]]:
    """Parse ``DATASET_CHECK|name|images|labels`` lines from a remote merge."""
    issues: List[Dict[str, object]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line.startswith("DATASET_CHECK|"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        try:
            n_images = int(parts[-2])
            n_labels = int(parts[-1])
        except ValueError:
            continue
        name = "|".join(parts[1:-2]).strip()
        if not name:
            continue
        issues.append(
            {"subdir": name, "images": n_images, "labels": n_labels}
        )
    return issues


def _parse_remote_merge_output(
    code: int,
    merged: str,
    *,
    remote_root: str,
    remote_python: Optional[str] = None,
    shell_cmd: str = "",
) -> Tuple[bool, str, List[Dict[str, object]]]:
    text = (merged or "").strip()
    yaml_path = f"{remote_root.rstrip('/')}/data.yaml"
    if "OK yaml=" in text:
        return True, yaml_path, _parse_dataset_check_lines(text)
    if text:
        for line in text.splitlines():
            if line.startswith("MERGE_ERROR:"):
                msg = line[len("MERGE_ERROR:") :].strip() or line
                return False, msg, []
        for line in reversed(text.splitlines()):
            if line.startswith("XAL_EXEC_EXIT="):
                try:
                    ec = int(line.split("=", 1)[1].strip())
                    if ec != 0 and code == 0:
                        code = ec
                except ValueError:
                    pass
                break
        body = "\n".join(
            ln
            for ln in text.splitlines()
            if not ln.startswith("XAL_EXEC_")
        ).strip()
        if body:
            return False, body, []
        return False, text, []
    py_hint = (remote_python or "").strip() or "python3 / python"
    msg = (
        "Remote merge failed (exit {code}). The server returned no output. "
        "Check Data → Remote runtime Python ({py}), SSH connection, and "
        "dataset ROOT: {root}."
    ).format(code=code, py=py_hint, root=remote_root)
    if shell_cmd:
        msg = f"{msg}\n\nRemote command:\n{shell_cmd[:1200]}"
    return False, msg, []


def remote_merge_lists(
    client: SSHClient,
    remote_root: str,
    subdirs: Sequence[str],
    *,
    remote_python: Optional[str] = None,
) -> Tuple[bool, str, List[Dict[str, object]]]:
    """Run merge script on remote; return (ok, message_or_yaml_path, issues)."""
    remote_root = (remote_root or "").strip().replace("\\", "/")
    if not remote_root:
        return False, "Remote dataset ROOT path is empty.", []
    py = (remote_python or "").strip() or "(system python3/python)"
    script = build_remote_merge_script(remote_root, list(subdirs))
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_merge_dataset.py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        local_script = f.name
    try:
        remote_script = f"{remote_root.rstrip('/')}/_xal_merge_dataset.py"
        client.mkdir_p(remote_root)
        client.upload_file(local_script, remote_script)
        code, merged, shell_cmd = _exec_remote_script_file(
            client,
            remote_script,
            remote_python=remote_python,
            timeout=600,
        )
        client.exec(f"rm -f {shlex.quote(remote_script)}", timeout=60)
        ok, msg, issues = _parse_remote_merge_output(
            code,
            merged,
            remote_root=remote_root,
            remote_python=remote_python,
            shell_cmd=shell_cmd,
        )
        if not ok and code != 0 and msg and "(exit" not in msg:
            msg = f"{msg}\n(exit code {code})"
        if not ok:
            msg = f"{msg}\nRemote Python: {py}\nRemote ROOT: {remote_root}"
        return ok, msg, issues
    except SSHError as e:
        return False, str(e), []
    except Exception as e:
        return False, f"Remote merge failed: {e}", []
    finally:
        try:
            os.unlink(local_script)
        except OSError:
            pass


def upload_model_if_local(
    client: SSHClient, model_path: str, remote_project: str
) -> str:
    """If model_path is a local file, upload beside remote project; else return as-is."""
    if model_path and os.path.isfile(model_path):
        name = os.path.basename(model_path)
        remote_model = f"{remote_project.rstrip('/')}/{name}"
        client.upload_file(model_path, remote_model)
        return remote_model
    return model_path


def build_remote_train_script(train_args: Dict) -> str:
    args = dict(train_args)
    model = args.pop("model")
    return f"""# -*- coding: utf-8 -*-
import sys
import traceback

try:
    from ultralytics import YOLO, settings
except Exception as e:
    print("ENV_ERROR: failed to import ultralytics: %s" % e, flush=True)
    traceback.print_exc()
    sys.exit(2)

try:
    # Ultralytics >=8.3.111 disables TensorBoard by default
    settings.update({{"tensorboard": True}})
except Exception as e:
    print("WARN: could not enable tensorboard settings: %s" % e, flush=True)

try:
    model = YOLO({repr(model)})
    train_args = {repr(args)}
    train_args["verbose"] = True
    train_args["show"] = False
    results = model.train(**train_args)
except Exception as e:
    print("TRAIN_ERROR: %s" % e, flush=True)
    traceback.print_exc()
    sys.exit(1)
"""


def _resolve_remote_tensorboard_cmd(
    client: SSHClient, remote_python: Optional[str] = None
) -> str:
    """
    Find how to invoke TensorBoard inside the remote env/container.

    Docker YOLO images often have the Python package but no ``tensorboard``
    shell script on PATH.
    """
    candidates = tensorboard_launcher_candidates(remote_python)
    for launcher in candidates:
        # Run in sh so $? reflects the launcher, not echo.
        code, out, err = client.exec(
            f"sh -c {shlex.quote(launcher + ' --help >/dev/null 2>&1')}; echo __EC:$?",
            timeout=40,
        )
        text = ((out or "") + (err or "")).strip()
        if "__EC:0" in text:
            return launcher
    raise SSHError(
        "TensorBoard is not available in the remote container. "
        "Install it inside the container, e.g.: "
        "pip install tensorboard"
    )


def _remote_tb_ready(
    client: SSHClient,
    port: int,
    tb_log: str,
    remote_python: Optional[str] = None,
) -> Tuple[bool, str]:
    """Return (ready, detail). Prefer log / HTTP probe — ss/netstat often missing in Docker."""
    detail = ""
    try:
        detail = client.read_text(tb_log, max_bytes=4000)
    except Exception:
        detail = ""

    if detail:
        if (
            "No such file or directory" in detail
            or "No module named" in detail
            or "ModuleNotFoundError" in detail
        ):
            return False, detail
        # Success banner, e.g. "TensorBoard 2.21.0 at http://0.0.0.0:6006/"
        if "at http://" in detail.lower() or f":{port}/" in detail:
            return True, detail

    probe = (
        http_probe_python(remote_python, port)
        + f" || curl -sf -o /dev/null http://127.0.0.1:{port}/ && echo OK "
        f"|| wget -q -O /dev/null http://127.0.0.1:{port}/ && echo OK "
        f"|| true"
    )
    _c, out, _e = client.exec(probe, timeout=15)
    if "OK" in (out or ""):
        return True, detail or out

    # Last resort: ss/netstat if present
    _c2, o2, _ = client.exec(
        f"(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) "
        f"| grep -E ':{port}\\s' && echo READY || true"
    )
    if "READY" in (o2 or "") or f":{port}" in (o2 or ""):
        return True, detail or o2

    return False, detail


def start_remote_tensorboard(
    client: SSHClient,
    logdir: str,
    preferred_port: int = 6006,
    *,
    remote_python: Optional[str] = None,
) -> Tuple[int, str]:
    """
    Start tensorboard on remote.

    Returns (remote_port, url) where url is ``http://<ssh-host>:<port>/``
    so the UI "Open TensorBoard" button opens the published Docker port
    (e.g. http://192.168.112.217:6006/).
    """
    import time

    launcher = _resolve_remote_tensorboard_cmd(client, remote_python)

    # Ensure logdir exists (may still be empty before first epoch)
    client.mkdir_p(logdir)

    port = preferred_port
    for candidate in range(preferred_port, preferred_port + 20):
        ready_now, _ = _remote_tb_ready(
            client,
            candidate,
            f"/tmp/xal_tensorboard_{candidate}.log",
            remote_python=remote_python,
        )
        # If something already answers on this port, reuse it after restart
        _c, o, _ = client.exec(
            f"(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null || true) "
            f"| grep -E ':{candidate}\\s' || true"
        )
        if f":{candidate}" not in (o or "") and not ready_now:
            # Also try bind probe via python
            _c3, o3, _ = client.exec(
                f"python3 -c \"import socket;s=socket.socket();r=s.connect_ex(("
                f"'127.0.0.1',{candidate}));print(r);s.close()\" 2>/dev/null || echo 1"
            )
            try:
                if int((o3 or "1").strip().splitlines()[-1]) != 0:
                    port = candidate
                    break
            except Exception:
                port = candidate
                break

    logdir_q = shlex.quote(logdir)
    tb_log = f"/tmp/xal_tensorboard_{port}.log"
    client.exec("pkill -f 'tensorboard' 2>/dev/null || true")
    time.sleep(0.5)

    # reload_interval helps pick up events written after TB starts
    cmd = (
        f"nohup {launcher} --logdir {logdir_q} --host 0.0.0.0 --port {port} "
        f"--reload_interval 5 "
        f"> {shlex.quote(tb_log)} 2>&1 & echo $!"
    )
    code, out, err = client.exec(cmd)
    if code != 0:
        raise SSHError(f"Failed to start TensorBoard: {err or out}")

    ready = False
    last_detail = ""
    for _ in range(40):
        time.sleep(0.5)
        ready, last_detail = _remote_tb_ready(
            client, port, tb_log, remote_python=remote_python
        )
        if ready:
            break
        if last_detail and (
            "No such file or directory" in last_detail
            or "No module named" in last_detail
            or "ModuleNotFoundError" in last_detail
        ):
            raise SSHError(
                f"TensorBoard failed to start via `{launcher}`:\n{last_detail}\n"
                "Install in the container: pip install tensorboard"
            )

    if not ready:
        raise SSHError(
            f"TensorBoard did not become ready on port {port} "
            f"(launcher=`{launcher}`).\n{last_detail}\n"
            "If missing, install inside the container: pip install tensorboard"
        )

    # UI opens host IP directly (e.g. http://192.168.112.217:6006/)
    if getattr(client, "host", None):
        url = f"http://{client.host}:{port}"
    else:
        url = f"http://127.0.0.1:{port}"

    return port, url


def start_local_tensorboard(
    logdir: str,
    preferred_port: int = 6006,
    *,
    python_executable: str = "",
) -> Tuple[object, int, str]:
    """Start local tensorboard subprocess. Returns (Popen, port, url)."""
    import subprocess
    import socket
    import sys

    def _free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) != 0

    port = preferred_port
    for candidate in range(preferred_port, preferred_port + 20):
        if _free(candidate):
            port = candidate
            break

    tb_args = ["--logdir", logdir, "--host", "127.0.0.1", "--port", str(port)]
    py = (python_executable or "").strip() or sys.executable
    candidates = [
        ["tensorboard", *tb_args],
        [py, "-m", "tensorboard.main", *tb_args],
        [sys.executable, "-m", "tensorboard.main", *tb_args],
        [sys.executable, "-m", "tensorboard", *tb_args],
    ]
    last_err: Optional[Exception] = None
    for cmd in candidates:
        exe = cmd[0]
        if exe not in ("tensorboard",) and not os.path.isfile(exe):
            continue
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            url = f"http://127.0.0.1:{port}"
            return proc, port, url
        except (FileNotFoundError, OSError) as e:
            last_err = e
            continue
    raise FileNotFoundError(last_err or "tensorboard not found in PATH")


def fetch_remote_log_tail(
    client: SSHClient, remote_paths: List[str], lines: int = 200
) -> str:
    chunks: List[str] = []
    for path in remote_paths:
        if not path:
            continue
        code, out, err = client.exec(
            f"tail -n {int(lines)} {shlex.quote(path)} 2>/dev/null || true"
        )
        text = (out or "").strip()
        if text:
            chunks.append(f"===== {path} =====\n{text}")
    return "\n\n".join(chunks)


def build_remote_onnx_export_script(
    weights_pt: str,
    output_onnx: str,
    imgsz: int = 640,
) -> str:
    """Python script: export PT → ONNX and save as ``output_onnx``."""
    return f"""# -*- coding: utf-8 -*-
import os
import sys
import traceback

weights_pt = {repr(weights_pt)}
output_onnx = {repr(output_onnx)}
imgsz = {int(imgsz)}

try:
    from ultralytics import YOLO
except Exception as e:
    print("ENV_ERROR: failed to import ultralytics: %s" % e, flush=True)
    traceback.print_exc()
    sys.exit(2)

if not os.path.isfile(weights_pt):
    print("EXPORT_ERROR: weights not found: %s" % weights_pt, flush=True)
    sys.exit(1)

try:
    model = YOLO(weights_pt)
    result = model.export(format="onnx", imgsz=imgsz)
    src = result if isinstance(result, str) else str(result)
    if not src or not os.path.isfile(src):
        guess = os.path.splitext(weights_pt)[0] + ".onnx"
        src = guess if os.path.isfile(guess) else src
    if not src or not os.path.isfile(src):
        print("EXPORT_ERROR: onnx output not found (got %r)" % src, flush=True)
        sys.exit(1)
    out_dir = os.path.dirname(output_onnx) or "."
    os.makedirs(out_dir, exist_ok=True)
    if os.path.abspath(src) != os.path.abspath(output_onnx):
        if os.path.exists(output_onnx):
            os.remove(output_onnx)
        os.replace(src, output_onnx)
    print("EXPORT_OK: %s" % output_onnx, flush=True)
except Exception as e:
    print("EXPORT_ERROR: %s" % e, flush=True)
    traceback.print_exc()
    sys.exit(1)
"""


def remote_export_onnx(
    client: SSHClient,
    run_dir: str,
    *,
    weights_file: str = "best.pt",
    output_name: str,
    imgsz: int = 640,
    remote_python: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Export best/last.pt to ONNX on the remote host into
    ``{run_dir}/weights/<output_name>``. Does not download the file.

    Returns (ok, message_or_remote_onnx_path).
    """
    run_dir = (run_dir or "").rstrip("/").replace("\\", "/")
    weights_file = os.path.basename(weights_file or "best.pt")
    if weights_file not in ("best.pt", "last.pt"):
        weights_file = "best.pt"
    output_name = os.path.basename(output_name or "")
    if not output_name.lower().endswith(".onnx"):
        return False, "output_name must end with .onnx"

    weights_pt = f"{run_dir}/weights/{weights_file}"
    output_onnx = f"{run_dir}/weights/{output_name}"
    script = build_remote_onnx_export_script(weights_pt, output_onnx, imgsz)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_export_onnx.py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        local_script = f.name
    remote_script = f"{run_dir}/_xal_export_onnx.py"
    try:
        try:
            client.upload_file(local_script, remote_script)
        finally:
            try:
                os.unlink(local_script)
            except OSError:
                pass

        code, out, err = client.exec(
            format_exec_python(remote_python, remote_script),
            timeout=3600,
        )
        text = ((out or "") + "\n" + (err or "")).strip()
        if code != 0 or "EXPORT_OK:" not in text:
            return False, text or f"Remote ONNX export failed (code={code})"
        remote_path = output_onnx
        for line in text.splitlines():
            if line.startswith("EXPORT_OK:"):
                remote_path = (
                    line.split("EXPORT_OK:", 1)[-1].strip() or output_onnx
                )
        return True, remote_path
    except SSHError as e:
        return False, str(e)
    finally:
        try:
            client.exec(
                f"rm -f {shlex.quote(remote_script)} 2>/dev/null || true"
            )
        except Exception:
            pass


def build_remote_predict_script(
    model_path: str,
    source_all_txt: str,
    run_dir: str,
    opts: Optional[PredictOptions] = None,
    *,
    imgsz: int = 640,
) -> str:
    """Predict all.txt → YOLO txt under ``{run_dir}/predict/labels``."""
    predict_opts = opts or PredictOptions.for_labels_only(imgsz=int(imgsz))
    predict_call = predict_opts.render_predict_call(
        source_expr="source",
        name_expr="'predict'",
    )
    return f"""# -*- coding: utf-8 -*-
import os
import sys
import traceback

model_path = {repr(model_path)}
source = {repr(source_all_txt)}
run_dir = {repr(run_dir)}
labels_dir = os.path.join(run_dir, "predict", "labels")

try:
    from ultralytics import YOLO
except Exception as e:
    print("ENV_ERROR: failed to import ultralytics: %s" % e, flush=True)
    traceback.print_exc()
    sys.exit(2)

if not os.path.isfile(model_path):
    print("PREDICT_ERROR: model not found: %s" % model_path, flush=True)
    sys.exit(1)
if not os.path.isfile(source):
    print("PREDICT_ERROR: source all.txt not found: %s" % source, flush=True)
    sys.exit(1)

try:
    model = YOLO(model_path)
{predict_call}
    n = 0
    if os.path.isdir(labels_dir):
        n = sum(
            1
            for f in os.listdir(labels_dir)
            if f.lower().endswith(".txt")
        )
    print("PREDICT_OK: %s (%d txt)" % (labels_dir, n), flush=True)
except Exception as e:
    print("PREDICT_ERROR: %s" % e, flush=True)
    traceback.print_exc()
    sys.exit(1)
"""


def remote_predict_yolo(
    client: SSHClient,
    run_dir: str,
    data_yaml: str,
    *,
    weights_file: str = "best.pt",
    imgsz: int = 640,
    remote_python: Optional[str] = None,
    predict_options: Optional[PredictOptions] = None,
) -> Tuple[bool, str]:
    """
    Run YOLO predict on remote ``all.txt``; write txt labels to
    ``{run_dir}/predict/labels`` without saving annotated images.

    Returns (ok, message_or_labels_dir).
    """
    run_dir = (run_dir or "").rstrip("/").replace("\\", "/")
    data_yaml = (data_yaml or "").replace("\\", "/")
    if not run_dir or not data_yaml:
        return False, "run_dir and data_yaml are required"

    dataset_root = os.path.dirname(data_yaml).replace("\\", "/")
    all_txt = f"{dataset_root.rstrip('/')}/all.txt"
    weights_name = os.path.basename(weights_file or "best.pt")
    model_path = f"{run_dir}/weights/{weights_name}"

    if not client.exists(all_txt):
        return False, f"all.txt not found: {all_txt}"
    if not client.exists(model_path):
        return False, f"Model weights not found: {model_path}"

    script = build_remote_predict_script(
        model_path,
        all_txt,
        run_dir,
        predict_options,
        imgsz=imgsz,
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_predict.py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        local_script = f.name
    remote_script = f"{run_dir}/_xal_predict.py"
    labels_dir = f"{run_dir}/predict/labels"
    try:
        try:
            client.upload_file(local_script, remote_script)
        finally:
            try:
                os.unlink(local_script)
            except OSError:
                pass

        code, out, err = client.exec(
            format_exec_python(remote_python, remote_script),
            timeout=7200,
        )
        text = ((out or "") + "\n" + (err or "")).strip()
        if code != 0 or "PREDICT_OK:" not in text:
            return False, text or f"Remote predict failed (code={code})"
        path = labels_dir
        for line in text.splitlines():
            if line.startswith("PREDICT_OK:"):
                rest = line.split("PREDICT_OK:", 1)[-1].strip()
                # "path (N txt)" → path
                if " (" in rest:
                    rest = rest.split(" (", 1)[0].strip()
                path = rest or labels_dir
        return True, path
    except SSHError as e:
        return False, str(e)
    finally:
        try:
            client.exec(
                f"rm -f {shlex.quote(remote_script)} 2>/dev/null || true"
            )
        except Exception:
            pass


def list_remote_weight_choices(client: SSHClient, run_dir: str) -> List[str]:
    """Return best.pt / last.pt / *.onnx that exist under run_dir/weights."""
    run_dir = (run_dir or "").rstrip("/").replace("\\", "/")
    weights_dir = f"{run_dir}/weights"
    choices: List[str] = []
    for name in ("best.pt", "last.pt"):
        try:
            if client.exists(f"{weights_dir}/{name}"):
                choices.append(name)
        except Exception:
            pass
    try:
        for name in sorted(client.listdir(weights_dir)):
            if name.lower().endswith(".onnx") and name not in choices:
                choices.append(name)
    except Exception:
        pass
    return choices or ["best.pt", "last.pt"]


def count_local_images(folder: str) -> int:
    n = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                n += 1
    return n


def build_remote_batch_predict_script(
    model_path: str,
    source_dir: str,
    run_dir: str,
    opts: Optional[PredictOptions] = None,
    *,
    imgsz: int = 640,
) -> str:
    """Predict a folder → YOLO txt under ``{run_dir}/batch_infer/labels``."""
    predict_opts = opts or PredictOptions.for_labels_only(imgsz=int(imgsz))
    predict_call = predict_opts.render_predict_call(
        source_expr="source",
        name_expr="'batch_infer'",
    )
    return f"""# -*- coding: utf-8 -*-
import os
import sys
import traceback

model_path = {repr(model_path)}
source = {repr(source_dir)}
run_dir = {repr(run_dir)}
labels_dir = os.path.join(run_dir, "batch_infer", "labels")

try:
    from ultralytics import YOLO
except Exception as e:
    print("ENV_ERROR: failed to import ultralytics: %s" % e, flush=True)
    traceback.print_exc()
    sys.exit(2)

if not os.path.isfile(model_path):
    print("PREDICT_ERROR: model not found: %s" % model_path, flush=True)
    sys.exit(1)
if not os.path.isdir(source):
    print("PREDICT_ERROR: source folder not found: %s" % source, flush=True)
    sys.exit(1)

try:
    model = YOLO(model_path)
{predict_call}
    n = 0
    if os.path.isdir(labels_dir):
        n = sum(
            1
            for f in os.listdir(labels_dir)
            if f.lower().endswith(".txt")
        )
    print("BATCH_PREDICT_OK: %s (%d txt)" % (labels_dir, n), flush=True)
except Exception as e:
    print("PREDICT_ERROR: %s" % e, flush=True)
    traceback.print_exc()
    sys.exit(1)
"""


def remote_batch_predict_yolo(
    client: SSHClient,
    run_dir: str,
    local_image_folder: str,
    *,
    weights_file: str = "best.pt",
    imgsz: int = 640,
    progress: Optional[Callable[[str], None]] = None,
    model_path: Optional[str] = None,
    local_weight_to_upload: Optional[str] = None,
    remote_python: Optional[str] = None,
    predict_options: Optional[PredictOptions] = None,
) -> Tuple[bool, str]:
    """
    Upload a local image folder and run YOLO predict on the remote server.

    Images → ``{run_dir}/batch_infer/images``
    Labels → ``{run_dir}/batch_infer/labels`` (same stem as images)

    ``model_path`` may override ``{run_dir}/weights/{weights_file}``.
    ``local_weight_to_upload`` copies a local .pt/.onnx into remote weights/.

    Returns (ok, remote_labels_dir_or_error).
    """
    run_dir = (run_dir or "").rstrip("/").replace("\\", "/")
    local_image_folder = os.path.abspath(local_image_folder or "")
    if not run_dir:
        return False, "run_dir is required"
    if not local_image_folder or not os.path.isdir(local_image_folder):
        return False, f"Image folder not found: {local_image_folder}"

    n_img = count_local_images(local_image_folder)
    if n_img <= 0:
        return False, f"No images found under: {local_image_folder}"

    weights_name = os.path.basename(weights_file or "best.pt")
    if local_weight_to_upload:
        local_w = os.path.abspath(local_weight_to_upload)
        if not os.path.isfile(local_w):
            return False, f"Local weight not found: {local_w}"
        weights_name = os.path.basename(local_w)
        remote_weights_dir = f"{run_dir}/weights"
        model_path = f"{remote_weights_dir}/{weights_name}"
        try:
            client.exec(f"mkdir -p {shlex.quote(run_dir)}")
            client.exec(f"mkdir -p {shlex.quote(remote_weights_dir)}")
            if progress:
                progress(f"Uploading weight {weights_name}…")
            client.upload_file(local_w, model_path)
        except SSHError as e:
            return False, str(e)
    elif model_path:
        model_path = model_path.replace("\\", "/")
    else:
        model_path = f"{run_dir}/weights/{weights_name}"
    if not client.exists(model_path):
        return False, f"Model weights not found: {model_path}"

    remote_images = f"{run_dir}/batch_infer/images"
    remote_labels = f"{run_dir}/batch_infer/labels"
    try:
        if progress:
            progress(f"Clearing remote {remote_images}")
        client.exec(
            f"rm -rf {shlex.quote(run_dir + '/batch_infer')} 2>/dev/null || true"
        )
        if progress:
            progress(f"Uploading {n_img} image(s)…")
        client.upload_directory(
            local_image_folder, remote_images, progress=progress
        )

        script = build_remote_batch_predict_script(
            model_path,
            remote_images,
            run_dir,
            predict_options,
            imgsz=imgsz,
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_batch_predict.py",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(script)
            local_script = f.name
        remote_script = f"{run_dir}/_xal_batch_predict.py"
        try:
            client.upload_file(local_script, remote_script)
        finally:
            try:
                os.unlink(local_script)
            except OSError:
                pass

        if progress:
            progress("Running remote batch predict…")
        code, out, err = client.exec(
            format_exec_python(remote_python, remote_script),
            timeout=7200,
        )
        text = ((out or "") + "\n" + (err or "")).strip()
        if code != 0 or "BATCH_PREDICT_OK:" not in text:
            return False, text or f"Remote batch predict failed (code={code})"
        path = remote_labels
        for line in text.splitlines():
            if line.startswith("BATCH_PREDICT_OK:"):
                rest = line.split("BATCH_PREDICT_OK:", 1)[-1].strip()
                if " (" in rest:
                    rest = rest.split(" (", 1)[0].strip()
                path = rest or remote_labels
        return True, path
    except SSHError as e:
        return False, str(e)
    finally:
        try:
            client.exec(
                f"rm -f {shlex.quote(run_dir + '/_xal_batch_predict.py')} "
                f"2>/dev/null || true"
            )
        except Exception:
            pass

