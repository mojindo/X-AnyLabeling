"""Persist / resume local and remote MyTrain job metadata."""

from __future__ import annotations

import json
import os
import shlex
import time
from typing import Any, Dict, Optional, Tuple

from anylabeling.services.auto_training.mytrain.config import (
    get_trainer_root_dir,
)


def remote_job_state_path() -> str:
    return os.path.join(get_trainer_root_dir(), "remote_job.json")


def local_job_state_path() -> str:
    return os.path.join(get_trainer_root_dir(), "local_job.json")


def save_remote_job(state: Dict[str, Any]) -> None:
    path = remote_job_state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(state)
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_remote_job() -> Optional[Dict[str, Any]]:
    path = remote_job_state_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def clear_remote_job() -> None:
    path = remote_job_state_path()
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def job_matches_host(job: Dict[str, Any], host: str, username: str) -> bool:
    if not job:
        return False
    return (
        str(job.get("host", "")).strip() == str(host or "").strip()
        and str(job.get("username", "")).strip() == str(username or "").strip()
    )


def is_remote_pid_alive(client, pid: str) -> bool:
    if not pid:
        return False
    try:
        _c, o, _e = client.exec(
            f"kill -0 {shlex.quote(str(pid))} 2>/dev/null; echo $?"
        )
        alive = (o or "").strip().splitlines()
        return bool(alive and alive[-1].strip() == "0")
    except Exception:
        return False


def save_local_job(state: Dict[str, Any]) -> None:
    path = local_job_state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(state)
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_local_job() -> Optional[Dict[str, Any]]:
    path = local_job_state_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def clear_local_job() -> None:
    path = local_job_state_path()
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def is_local_pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid_int
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid_int, 0)
        return True
    except OSError:
        return False


def probe_local_job(job: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """Return (status, message): none | alive | finished | error."""
    job = job if job is not None else load_local_job()
    if not job:
        return "none", "No saved local training job"
    pid = job.get("pid")
    try:
        if is_local_pid_alive(pid):
            log = job.get("log_path") or ""
            return "alive", f"Local PID {pid} is still running (log: {log})"
        log = job.get("log_path") or ""
        run_dir = job.get("run_dir") or ""
        return "finished", f"PID {pid} not running (run: {run_dir}, log: {log})"
    except Exception as e:
        return "error", str(e)


def probe_remote_job(
    client, job: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]:
    """
    Returns (status, message) where status is:
      none | alive | finished | mismatch | error
    """
    job = job if job is not None else load_remote_job()
    if not job:
        return "none", "No saved remote job"
    if not job_matches_host(
        job, getattr(client, "host", ""), getattr(client, "username", "")
    ):
        return "mismatch", "Saved job belongs to another host/user"
    pid = str(job.get("pid") or "")
    try:
        if is_remote_pid_alive(client, pid):
            return "alive", f"Remote PID {pid} is still running"
        log = job.get("log_path") or ""
        return "finished", f"PID {pid} not running (log: {log})"
    except Exception as e:
        return "error", str(e)
