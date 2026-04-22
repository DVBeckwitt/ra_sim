"""Gated high-resolution timing events for GUI performance measurement."""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Iterator

_TRUE_VALUES = {"1", "true", "yes", "on"}
_WRITE_LOCK = threading.RLock()
_UPDATE_COUNTER = count(1)


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in _TRUE_VALUES


def timing_enabled() -> bool:
    """Return whether JSONL timing output is enabled for this process."""

    return _env_flag("RA_SIM_TIMING") and bool(str(os.environ.get("RA_SIM_TIMING_OUT", "")).strip())


def _timing_output_path() -> Path | None:
    raw_path = str(os.environ.get("RA_SIM_TIMING_OUT", "")).strip()
    if not raw_path:
        return None
    return Path(raw_path).expanduser()


def _current_rss_bytes() -> int | None:
    """Return process RSS when it is cheap and available."""

    if os.name == "nt":
        try:

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle,
                ctypes.byref(counters),
                counters.cb,
            )
            if ok:
                return int(counters.WorkingSetSize)
        except Exception:
            return None
        return None

    try:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if rss <= 0:
            return None
        # Linux reports KiB; macOS reports bytes.
        return rss if rss > 10_000_000 else rss * 1024
    except Exception:
        return None


def _string_or_basename(value: str) -> str:
    text = str(value)
    if ("\\" in text or "/" in text) and len(text) > 1:
        name = Path(text).name
        return name or "<path>"
    return text


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _string_or_basename(value)
    if isinstance(value, Path):
        return value.name
    if isinstance(value, bytes | bytearray | memoryview):
        return {"byte_count": len(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe(item) for item in value]
    try:
        return _json_safe(str(value))
    except Exception:
        return f"<{type(value).__name__}>"


def _base_record(event: str, fields: dict[str, object]) -> dict[str, object]:
    update_id = fields.pop("update_id", None)
    phase = fields.pop("phase", None)
    record: dict[str, object] = {
        "event": str(event),
        "perf_counter_ns": time.perf_counter_ns(),
        "wall_time_iso": datetime.now().astimezone().isoformat(timespec="microseconds"),
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "scenario_id": os.environ.get("RA_SIM_TIMING_SCENARIO"),
        "trial_id": os.environ.get("RA_SIM_TIMING_TRIAL"),
        "update_id": update_id,
        "phase": phase,
    }
    rss_bytes = _current_rss_bytes()
    if rss_bytes is not None:
        record["rss_bytes"] = int(rss_bytes)
    record.update({str(key): _json_safe(value) for key, value in fields.items()})
    return record


def timing_event(event: str, **fields: object) -> None:
    """Append one newline-delimited JSON timing event when timing is enabled."""

    if not timing_enabled():
        return
    output_path = _timing_output_path()
    if output_path is None:
        return
    try:
        record = _base_record(event, dict(fields))
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with _WRITE_LOCK:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
    except Exception:
        return


@contextlib.contextmanager
def timing_span(name: str, **fields: object) -> Iterator[None]:
    """Emit ``<name>.start`` and ``<name>.end`` events around a code block."""

    phase = fields.get("phase", name)
    timing_event(f"{name}.start", **{**fields, "phase": phase})
    try:
        yield
    except Exception as exc:
        timing_event(
            f"{name}.error",
            **{**fields, "phase": phase, "exc_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    finally:
        timing_event(f"{name}.end", **{**fields, "phase": phase})


def new_update_id(reason: str) -> int:
    """Return a process-local update id and log why it was allocated."""

    update_id = next(_UPDATE_COUNTER)
    timing_event("update_id.new", update_id=update_id, phase="update", reason=str(reason))
    return int(update_id)
