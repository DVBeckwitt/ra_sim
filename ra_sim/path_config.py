"""Compatibility wrappers around the canonical :mod:`ra_sim.config` loader."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

from ra_sim.config import (
    DEFAULT_DIRS,
    clear_config_cache,
    get_config_bundle,
    get_config_dir as _get_config_dir,
    get_dir as _get_dir,
    get_instrument_config as _get_instrument_config,
    get_material_config as _get_material_config,
    get_path as _get_path,
    get_path_first as _get_path_first,
    list_materials as _list_materials,
)

PATHS: dict = {}
DIRS: dict = dict(DEFAULT_DIRS)
_MATERIAL_CONSTANTS: dict = {}
_MATERIALS: dict = {}
_DEFAULT_MATERIAL = None
_instrument_raw: dict = {}
_ACTIVE_CONFIG_DIR: Path | None = None
_TEMP_DIR: Path | None = None


def get_config_dir() -> Path:
    """Return the active configuration directory."""

    return _get_config_dir()


def reload_config_cache() -> None:
    """Reload cached YAML configuration through the canonical loader."""

    global PATHS, DIRS, _MATERIAL_CONSTANTS, _MATERIALS, _DEFAULT_MATERIAL
    global _instrument_raw, _ACTIVE_CONFIG_DIR, _TEMP_DIR

    clear_config_cache()
    bundle = get_config_bundle()
    materials_raw = bundle.materials

    PATHS = copy.deepcopy(bundle.file_paths)
    DIRS = {**DEFAULT_DIRS, **copy.deepcopy(bundle.dir_paths)}
    _MATERIAL_CONSTANTS = copy.deepcopy(materials_raw.get("constants", {}))
    _MATERIALS = copy.deepcopy(materials_raw.get("materials", {}))
    _DEFAULT_MATERIAL = copy.deepcopy(materials_raw.get("default_material"))
    _instrument_raw = copy.deepcopy(bundle.instrument)
    _ACTIVE_CONFIG_DIR = bundle.config_dir
    _TEMP_DIR = None


def _ensure_config_cache_current() -> None:
    if _ACTIVE_CONFIG_DIR != get_config_dir():
        reload_config_cache()


reload_config_cache()


def get_path(key: str):
    """Return the configured path for ``key`` expanding ``~``."""

    _ensure_config_cache_current()
    return _get_path(key)


def get_path_first(*keys: str):
    """Return the first configured path among ``keys``."""

    _ensure_config_cache_current()
    return _get_path_first(*keys)


def get_dir(key: str) -> Path:
    """Return the configured directory for ``key`` and ensure it exists."""

    _ensure_config_cache_current()
    return _get_dir(key)


def get_temp_dir() -> Path:
    """Return a dedicated temporary directory for the current session."""

    global _TEMP_DIR

    _ensure_config_cache_current()
    if _TEMP_DIR is None:
        base_path = get_dir("temp_root")
        base_path.mkdir(parents=True, exist_ok=True)
        _TEMP_DIR = Path(tempfile.mkdtemp(prefix="ra_sim_", dir=str(base_path)))
    return _TEMP_DIR


def get_material_config(material: str | None = None) -> dict:
    """Return the configuration block for ``material``."""

    _ensure_config_cache_current()
    return _get_material_config(material)


def list_materials() -> list[str]:
    """Return the sorted configured material identifiers."""

    _ensure_config_cache_current()
    return _list_materials()


def get_instrument_config() -> dict:
    """Return the parsed instrument configuration."""

    _ensure_config_cache_current()
    return _get_instrument_config()
