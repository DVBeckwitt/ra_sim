"""RA-SIM GUI package."""

from __future__ import annotations

__all__ = ["CollapsibleFrame"]


def __getattr__(name: str):
    if name == "CollapsibleFrame":
        from .collapsible import CollapsibleFrame

        return CollapsibleFrame
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
