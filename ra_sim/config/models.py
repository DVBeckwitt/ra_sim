"""Typed containers for parsed configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConfigBundle:
    """In-memory representation of the project configuration."""

    config_dir: Path
    file_paths: dict[str, Any]
    dir_paths: dict[str, Any]
    materials: dict[str, Any]
    instrument: dict[str, Any]
