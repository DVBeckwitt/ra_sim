"""Validation helpers for configuration payloads."""

from __future__ import annotations

from typing import Any


def ensure_mapping(value: Any, *, name: str) -> dict[str, Any]:
    """Return *value* as ``dict`` or raise a descriptive ``TypeError``."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value
