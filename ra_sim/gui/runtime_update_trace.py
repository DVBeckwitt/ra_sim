"""Helpers for persistent GUI runtime update trace logging."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from ra_sim.debug_controls import runtime_update_trace_logging_enabled


def resolve_runtime_update_trace_path(
    downloads_dir: Path | str | None,
    *,
    current_time: datetime | None = None,
    fallback_dir: Path | str | None = None,
) -> Path:
    """Return the daily GUI runtime trace log path."""

    if downloads_dir is not None:
        base_dir = Path(downloads_dir)
    elif fallback_dir is not None:
        base_dir = Path(fallback_dir)
    else:
        base_dir = Path.home() / "Downloads"
    stamp = (current_time or datetime.now()).strftime("%Y%m%d")
    return base_dir / f"runtime_update_trace_{stamp}.log"


def _trace_value_text(value: object) -> str:
    """Format one runtime trace field value."""

    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isnan(numeric):
            return "nan"
        if np.isposinf(numeric):
            return "inf"
        if np.isneginf(numeric):
            return "-inf"
        return f"{numeric:.6f}"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        parts = [
            f"{key}:{_trace_value_text(item)}"
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        ]
        return "{" + ",".join(parts) + "}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        preview = [_trace_value_text(item) for item in items[:8]]
        if len(items) > 8:
            preview.append("...")
        return "[" + ",".join(preview) + "]"
    return str(value).replace("\r", "\\r").replace("\n", "\\n")


def format_runtime_update_trace_line(
    event: str,
    *,
    timestamp: datetime | None = None,
    pid: int | None = None,
    fields: Mapping[str, object] | None = None,
) -> str:
    """Format one append-only GUI runtime trace line."""

    now = timestamp or datetime.now()
    parts = [
        now.isoformat(timespec="milliseconds"),
        f"pid={int(os.getpid() if pid is None else pid)}",
        f"event={str(event)}",
    ]
    for key, value in sorted((fields or {}).items(), key=lambda item: str(item[0])):
        if value is None:
            continue
        parts.append(f"{key}={_trace_value_text(value)}")
    return " ".join(parts)


def append_runtime_update_trace_line(
    path: Path | str,
    event: str,
    **fields: object,
) -> None:
    """Append one line to the GUI runtime trace log."""

    if not runtime_update_trace_logging_enabled():
        return
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(format_runtime_update_trace_line(event, fields=fields) + "\n")
        handle.flush()
