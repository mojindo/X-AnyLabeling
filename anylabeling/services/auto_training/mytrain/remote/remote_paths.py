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


def dataset_remote_dir(local_root_name: str, home: str) -> str:
    """
    ``~/xanylabeling_datasets/<local_folder_name>`` as absolute POSIX path.
    """
    name = os.path.basename(local_root_name.rstrip("\\/"))
    rel = f"{REMOTE_DATASETS_BASE}/{name}"
    return expand_remote_home(rel, home)


def runs_remote_dir(task: str, home: str) -> str:
    task = (task or "detect").lower()
    rel = f"{REMOTE_RUNS_BASE}/{task}"
    return expand_remote_home(rel, home)


def posix_join(*parts: str) -> str:
    p = PurePosixPath(parts[0])
    for part in parts[1:]:
        p = p / part
    return str(p)
