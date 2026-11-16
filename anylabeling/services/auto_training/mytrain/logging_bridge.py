"""Unified MyTrain log lines to the training UI callback bridge."""

from __future__ import annotations

from typing import Any, Optional


def emit_train_log(manager: Any, message: str, *, level: str = "info") -> None:
    text = (message or "").strip()
    if not text or manager is None:
        return
    prefix = ""
    if level == "error":
        prefix = "ERROR: "
    elif level == "warn":
        prefix = "WARN: "
    manager.notify_callbacks(
        "training_log",
        {"message": f"{prefix}{text}" if prefix else text},
    )


def ssh_progress_log(manager: Optional[Any], message: str) -> None:
    """SSH upload / remote ops progress hook compatible with ``progress=`` kwarg."""
    if manager is not None:
        emit_train_log(manager, message)