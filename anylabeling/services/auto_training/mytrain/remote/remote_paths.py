"""Remote path conventions for Ultralytics training."""

from __future__ import annotations

import os
from pathlib import PurePosixPath

REMOTE_DATASETS_BASE = "~/xanylabeling_datasets"
REMOTE_RUNS_BASE = "~/xanylabeling_datasets/_runs"


def expand_remote_home(path: str, home: str) -> str:
    """Expand leading ~ on a remote POSIX path using ``home``."""
    home = home.rstrip("/")
    if path == "~":
        return home
    if path.startswith("~/"):
        return f"{home}/{path[2:]}"
    return path


def resolve_remote_base(base: str, home: str) -> str:
    """Remote root for datasets and runs. Empty base keeps the old default."""
    raw = (base or "").strip().replace("\\", "/")
    if not raw:
        return expand_remote_home(REMOTE_DATASETS_BASE, home).rstrip("/")
    if raw == "~" or raw.startswith("~/"):
        return expand_remote_home(raw, home).rstrip("/") or "/"
    if not raw.startswith("/"):
        home = (home or "").rstrip("/")
        return f"{home}/{raw.lstrip('/')}".rstrip("/") or "/"
    return raw.rstrip("/") or "/"


def dataset_remote_dir(
    local_root_name: str, home: str, base: str = ""
) -> str:
    """
    ``<base>/<local_folder_name>`` as an absolute POSIX path.

    ``base`` empty uses ``~/xanylabeling_datasets``.
    """
    name = os.path.basename(str(local_root_name).rstrip("\\/")) or "dataset"
    root = resolve_remote_base(base, home)
    return f"{root}/{name}"


def runs_remote_dir(task: str, home: str, base: str = "") -> str:
    task = (task or "detect").lower()
    root = resolve_remote_base(base, home)
    return f"{root}/_runs/{task}"


def posix_join(*parts: str) -> str:
    p = PurePosixPath(parts[0])
    for part in parts[1:]:
        p = p / part
    return str(p)
