"""Local training interpreter profiles (Conda / custom Python)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from anylabeling.services.auto_training.mytrain.config import (
    get_default_project_dir,
)


PROFILE_FILE_NAME = "local_interpreters.json"


def profiles_path() -> str:
    return os.path.join(get_default_project_dir(), PROFILE_FILE_NAME)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_profile(*, name: str = "", python: str = "") -> Dict[str, Any]:
    return {
        "id": _new_id(),
        "name": name or "Interpreter",
        "python": python or "",
        "last_ok": None,
        "last_tested_at": "",
        "last_detail": "",
    }


def resolve_python_executable(path: str) -> str:
    """Accept python.exe or a Conda/venv root and return the executable path."""
    p = (path or "").strip().strip('"')
    if not p:
        return ""
    p = os.path.abspath(p)
    if os.path.isfile(p):
        return p
    if os.path.isdir(p):
        candidates = [
            os.path.join(p, "python.exe"),
            os.path.join(p, "Scripts", "python.exe"),
            os.path.join(p, "bin", "python"),
            os.path.join(p, "bin", "python3"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return os.path.abspath(c)
    return p


def _default_name(raw: Dict[str, Any]) -> str:
    name = str(raw.get("name") or "").strip()
    if name:
        return name
    py = str(raw.get("python") or "").strip()
    if not py:
        return "Interpreter"
    # Prefer env folder name: .../envs/yolo/python.exe → yolo
    parts = os.path.normpath(py).split(os.sep)
    lower = [x.lower() for x in parts]
    if "envs" in lower:
        i = lower.index("envs")
        if i + 1 < len(parts):
            return parts[i + 1]
    parent = os.path.basename(os.path.dirname(py))
    return parent or os.path.basename(py) or "Interpreter"


def _normalize_profile(raw: Dict[str, Any]) -> Dict[str, Any]:
    p = empty_profile()
    py = resolve_python_executable(str(raw.get("python") or ""))
    p.update(
        {
            "id": str(raw.get("id") or p["id"]),
            "name": str(raw.get("name") or "").strip() or _default_name(raw),
            "python": py,
            "last_ok": raw.get("last_ok"),
            "last_tested_at": str(raw.get("last_tested_at") or ""),
            "last_detail": str(raw.get("last_detail") or ""),
        }
    )
    return p


def _empty_store() -> Dict[str, Any]:
    return {"version": 1, "active_id": "", "interpreters": []}


def load_interpreter_profiles(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or profiles_path()
    if not os.path.isfile(path):
        return _empty_store()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _empty_store()
        interpreters = [
            _normalize_profile(x)
            for x in (data.get("interpreters") or [])
            if isinstance(x, dict)
        ]
        active_id = str(data.get("active_id") or "")
        if active_id and not any(x["id"] == active_id for x in interpreters):
            active_id = interpreters[0]["id"] if interpreters else ""
        if not active_id and interpreters:
            active_id = interpreters[0]["id"]
        return {
            "version": 1,
            "active_id": active_id,
            "interpreters": interpreters,
        }
    except Exception:
        return _empty_store()


def save_interpreter_profiles(
    store: Dict[str, Any], path: Optional[str] = None
) -> None:
    path = path or profiles_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": 1,
        "active_id": str(store.get("active_id") or ""),
        "interpreters": [
            _normalize_profile(x)
            for x in (store.get("interpreters") or [])
            if isinstance(x, dict)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_active_profile(store: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    active_id = store.get("active_id") or ""
    for x in store.get("interpreters") or []:
        if x.get("id") == active_id:
            return deepcopy(x)
    items = store.get("interpreters") or []
    return deepcopy(items[0]) if items else None


def find_profile(
    store: Dict[str, Any], profile_id: str
) -> Optional[Dict[str, Any]]:
    for x in store.get("interpreters") or []:
        if x.get("id") == profile_id:
            return x
    return None


def upsert_profile(
    store: Dict[str, Any], profile: Dict[str, Any]
) -> Dict[str, Any]:
    profile = _normalize_profile(profile)
    items: List[Dict[str, Any]] = list(store.get("interpreters") or [])
    for i, x in enumerate(items):
        if x.get("id") == profile["id"]:
            items[i] = profile
            store["interpreters"] = items
            return profile
    items.append(profile)
    store["interpreters"] = items
    if not store.get("active_id"):
        store["active_id"] = profile["id"]
    return profile


def delete_profile(store: Dict[str, Any], profile_id: str) -> bool:
    items = [
        x
        for x in (store.get("interpreters") or [])
        if x.get("id") != profile_id
    ]
    if len(items) == len(store.get("interpreters") or []):
        return False
    store["interpreters"] = items
    if store.get("active_id") == profile_id:
        store["active_id"] = items[0]["id"] if items else ""
    return True


def mark_test_result(
    store: Dict[str, Any],
    profile_id: str,
    ok: bool,
    detail: str = "",
) -> Optional[Dict[str, Any]]:
    profile = find_profile(store, profile_id)
    if not profile:
        return None
    profile["last_ok"] = bool(ok)
    profile["last_tested_at"] = _now_iso()
    profile["last_detail"] = (detail or "")[:500]
    return profile


def profile_display_name(profile: Dict[str, Any]) -> str:
    name = str(profile.get("name") or "").strip() or "Interpreter"
    py = str(profile.get("python") or "").strip()
    mark = ""
    if profile.get("last_ok") is True:
        mark = " ✓"
    elif profile.get("last_ok") is False:
        mark = " ✗"
    short = py
    if len(short) > 42:
        short = "…" + short[-40:]
    if short:
        return f"{name} ({short}){mark}"
    return f"{name}{mark}"


def probe_interpreter(
    python_exe: str,
    timeout: float = 45.0,
    *,
    require_ultralytics: bool = True,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Run a small probe in the target interpreter.

    Returns (ok, message, info_dict).
    """
    python_exe = resolve_python_executable(python_exe)
    if not python_exe or not os.path.isfile(python_exe):
        return False, f"Python executable not found: {python_exe}", {}

    code = r"""
import json
info = {"python": "", "ultralytics": False, "torch": "", "cuda": False, "device_count": 0, "error": ""}
try:
    import sys
    info["python"] = sys.version.split()[0]
    try:
        import ultralytics
        info["ultralytics"] = True
        info["ultralytics_version"] = getattr(ultralytics, "__version__", "")
    except Exception as e:
        info["ultralytics"] = False
        info["ultralytics_error"] = str(e)
    try:
        import torch
        info["torch"] = str(torch.__version__)
        info["cuda"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count()) if info["cuda"] else 0
    except Exception as e:
        info["error"] = "torch: %s" % e
        print(json.dumps(info))
        raise SystemExit(3)
    print(json.dumps(info))
except SystemExit:
    raise
except Exception as e:
    info["error"] = str(e)
    print(json.dumps(info))
    raise SystemExit(1)
"""
    try:
        proc = subprocess.run(
            [python_exe, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=os.path.dirname(python_exe) or None,
        )
    except subprocess.TimeoutExpired:
        return False, "Probe timed out", {}
    except Exception as e:
        return False, f"Failed to start interpreter: {e}", {}

    out = (proc.stdout or "").strip().splitlines()
    payload = {}
    if out:
        try:
            payload = json.loads(out[-1])
        except Exception:
            payload = {}
    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or payload.get("error") or out[-1] if out else "probe failed"
        msg = str(err)[:300]
        return False, msg, payload

    ultra = payload.get("ultralytics")
    torch_v = payload.get("torch") or "?"
    if not torch_v or torch_v == "?":
        return False, payload.get("error") or "PyTorch not available", payload
    cuda = payload.get("cuda")
    n = payload.get("device_count") or 0
    py_v = payload.get("python") or "?"
    ultra_v = payload.get("ultralytics_version") or ""
    if ultra:
        ultra_part = f"ultralytics {ultra_v or 'ok'}"
    else:
        ultra_part = "ultralytics not installed"
    msg = f"Python {py_v}, {ultra_part}, torch {torch_v}, cuda={cuda}, GPUs={n}"
    if not ultra:
        if require_ultralytics:
            return False, "ultralytics not importable", payload
        msg = msg + " (ultralytics optional for custom training)"
    return True, msg, payload


def app_python_executable() -> str:
    return sys.executable or ""
