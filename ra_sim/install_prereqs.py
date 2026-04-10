"""Install-surface prerequisite checks shared by launcher entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any


@dataclass(frozen=True)
class TkinterModules:
    """Imported Tk modules required by RA-SIM GUI entrypoints."""

    tk: Any
    ttk: Any


class MissingPrerequisiteError(RuntimeError):
    """Raised when a local prerequisite needed by one entrypoint is unavailable."""


def build_tkinter_prerequisite_message(entrypoint_label: str) -> str:
    """Return an actionable Tkinter install hint for one GUI entrypoint."""

    label = str(entrypoint_label).strip() or "This RA-SIM entrypoint"
    return (
        f"{label} requires Tkinter, but Python could not import it. "
        "On Linux, install the system Tk package for your interpreter "
        "(for example `python3-tk` or `python3.11-tk`). "
        "Windows and macOS Python distributions usually bundle Tk already. "
        "Headless commands such as `python -m ra_sim simulate` and "
        "`python -m ra_sim hbn-fit` do not require Tk."
    )


def require_tkinter_modules(entrypoint_label: str) -> TkinterModules:
    """Import and return Tkinter modules or raise an actionable error."""

    try:
        tk = importlib.import_module("tkinter")
        ttk = importlib.import_module("tkinter.ttk")
    except Exception as exc:
        raise MissingPrerequisiteError(
            build_tkinter_prerequisite_message(entrypoint_label)
        ) from exc
    return TkinterModules(tk=tk, ttk=ttk)


def require_tkinter(entrypoint_label: str) -> None:
    """Validate that Tkinter is importable for one GUI entrypoint."""

    require_tkinter_modules(entrypoint_label)
