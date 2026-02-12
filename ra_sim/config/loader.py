"""Load RA-SIM configuration files from disk."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from .models import ConfigBundle
from .validation import ensure_mapping

ENV_CONFIG_DIR = "RA_SIM_CONFIG_DIR"


def get_config_dir() -> Path:
    """Return the active configuration directory.

    Order of precedence:
    1. ``RA_SIM_CONFIG_DIR`` environment variable when set.
    2. Repository-local ``config/`` directory.
    """

    env_path = os.environ.get(ENV_CONFIG_DIR)
    if env_path:
        return Path(os.path.expanduser(env_path)).resolve()
    return Path(__file__).resolve().parents[2] / "config"


def _read_data_file(path: Path) -> dict[str, Any]:
    """Load a YAML/JSON mapping from *path*.

    Missing files return an empty mapping.
    """

    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if isinstance(data, dict):
        return data

    # Support JSON files with stricter parser error messages when needed.
    if path.suffix.lower() == ".json":
        parsed = json.loads(text)
        return ensure_mapping(parsed, name=str(path))
    raise TypeError(f"{path} must contain a mapping at top level")


def _load_from_dir(config_dir: Path) -> ConfigBundle:
    file_paths = ensure_mapping(
        _read_data_file(config_dir / "file_paths.yaml"),
        name="file_paths.yaml",
    )
    dir_paths = ensure_mapping(
        _read_data_file(config_dir / "dir_paths.yaml"),
        name="dir_paths.yaml",
    )
    materials = ensure_mapping(
        _read_data_file(config_dir / "materials.yaml"),
        name="materials.yaml",
    )
    instrument = ensure_mapping(
        _read_data_file(config_dir / "instrument.yaml"),
        name="instrument.yaml",
    )
    return ConfigBundle(
        config_dir=config_dir,
        file_paths=file_paths,
        dir_paths=dir_paths,
        materials=materials,
        instrument=instrument,
    )


_BUNDLE_CACHE: dict[Path, ConfigBundle] = {}


def clear_config_cache() -> None:
    """Clear cached configuration bundles."""

    _BUNDLE_CACHE.clear()


def get_config_bundle(config_dir: Path | None = None) -> ConfigBundle:
    """Return the active cached configuration bundle."""

    resolved_dir = (config_dir or get_config_dir()).resolve()
    bundle = _BUNDLE_CACHE.get(resolved_dir)
    if bundle is None:
        bundle = _load_from_dir(resolved_dir)
        _BUNDLE_CACHE[resolved_dir] = bundle
    return bundle
