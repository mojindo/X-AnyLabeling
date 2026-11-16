"""Multi-server SSH profile store for remote training."""

from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from anylabeling.services.auto_training.mytrain.config import (
    get_default_project_dir,
)


PROFILE_FILE_NAME = "remote_ssh.json"


def profiles_path() -> str:
    return os.path.join(get_default_project_dir(), PROFILE_FILE_NAME)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_profile(*, name: str = "") -> Dict[str, Any]:
    return {
        "id": _new_id(),
        "name": name or "Server",
        "host": "",
        "port": 22,
        "username": "",
        "password": "",
        "last_ok": None,
        "last_tested_at": "",
        "remote_runtime": "shell",
        "conda_executable": "",
        "conda_env_name": "",
        "remote_python": "",
    }


def _normalize_profile(raw: Dict[str, Any]) -> Dict[str, Any]:
    p = empty_profile()
    p.update(
        {
            "id": str(raw.get("id") or p["id"]),
            "name": str(raw.get("name") or "").strip()
            or _default_name(raw),
            "host": str(raw.get("host") or "").strip(),
            "port": int(raw.get("port") or 22),
            "username": str(raw.get("username") or "").strip(),
            "password": str(raw.get("password") or ""),
            "last_ok": raw.get("last_ok"),
            "last_tested_at": str(raw.get("last_tested_at") or ""),
            "remote_runtime": (
                "conda"
                if str(raw.get("remote_runtime") or "").strip().lower() == "conda"
                else "shell"
            ),
            "conda_executable": str(raw.get("conda_executable") or "").strip(),
            "conda_env_name": str(raw.get("conda_env_name") or "").strip(),
            "remote_python": str(raw.get("remote_python") or "").strip(),
        }
    )
    return p


def _default_name(raw: Dict[str, Any]) -> str:
    host = str(raw.get("host") or "").strip()
    user = str(raw.get("username") or "").strip()
    if host and user:
        return f"{user}@{host}"
    if host:
        return host
    return "Server"


def _empty_store() -> Dict[str, Any]:
    return {"version": 2, "active_id": "", "servers": []}


def _migrate_v1(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert legacy single-server file to multi-server store."""
    if "servers" in data and isinstance(data.get("servers"), list):
        return data
    if not data.get("host"):
        return _empty_store()
    profile = _normalize_profile(
        {
            "name": _default_name(data),
            "host": data.get("host"),
            "port": data.get("port", 22),
            "username": data.get("username"),
            "password": data.get("password"),
        }
    )
    return {
        "version": 2,
        "active_id": profile["id"],
        "servers": [profile],
    }


def load_ssh_profiles(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or profiles_path()
    if not os.path.isfile(path):
        return _empty_store()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _empty_store()
        data = _migrate_v1(data)
        servers = [
            _normalize_profile(s)
            for s in (data.get("servers") or [])
            if isinstance(s, dict)
        ]
        active_id = str(data.get("active_id") or "")
        if active_id and not any(s["id"] == active_id for s in servers):
            active_id = servers[0]["id"] if servers else ""
        if not active_id and servers:
            active_id = servers[0]["id"]
        return {"version": 2, "active_id": active_id, "servers": servers}
    except Exception:
        return _empty_store()


def save_ssh_profiles(store: Dict[str, Any], path: Optional[str] = None) -> None:
    path = path or profiles_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": 2,
        "active_id": str(store.get("active_id") or ""),
        "servers": [
            _normalize_profile(s)
            for s in (store.get("servers") or [])
            if isinstance(s, dict)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_active_profile(store: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    active_id = store.get("active_id") or ""
    for s in store.get("servers") or []:
        if s.get("id") == active_id:
            return deepcopy(s)
    servers = store.get("servers") or []
    return deepcopy(servers[0]) if servers else None


def find_profile(store: Dict[str, Any], profile_id: str) -> Optional[Dict[str, Any]]:
    for s in store.get("servers") or []:
        if s.get("id") == profile_id:
            return s
    return None


def upsert_profile(store: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    profile = _normalize_profile(profile)
    servers: List[Dict[str, Any]] = list(store.get("servers") or [])
    for i, s in enumerate(servers):
        if s.get("id") == profile["id"]:
            servers[i] = profile
            store["servers"] = servers
            return profile
    servers.append(profile)
    store["servers"] = servers
    if not store.get("active_id"):
        store["active_id"] = profile["id"]
    return profile


def delete_profile(store: Dict[str, Any], profile_id: str) -> bool:
    servers = [s for s in (store.get("servers") or []) if s.get("id") != profile_id]
    if len(servers) == len(store.get("servers") or []):
        return False
    store["servers"] = servers
    if store.get("active_id") == profile_id:
        store["active_id"] = servers[0]["id"] if servers else ""
    return True


def mark_test_result(
    store: Dict[str, Any], profile_id: str, ok: bool
) -> Optional[Dict[str, Any]]:
    profile = find_profile(store, profile_id)
    if not profile:
        return None
    profile["last_ok"] = bool(ok)
    profile["last_tested_at"] = _now_iso()
    return profile


def profile_display_name(profile: Dict[str, Any]) -> str:
    name = str(profile.get("name") or "").strip()
    host = str(profile.get("host") or "").strip()
    mark = ""
    if profile.get("last_ok") is True:
        mark = " ✓"
    elif profile.get("last_ok") is False:
        mark = " ✗"
    if name and host and name != host:
        return f"{name} ({host}){mark}"
    return f"{name or host or 'Server'}{mark}"
