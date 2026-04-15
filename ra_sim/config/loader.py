"""Load RA-SIM configuration files from disk."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ra_sim.user_paths import user_cache_root, user_data_root

from .models import ConfigBundle
from .validation import ensure_mapping

ENV_CONFIG_DIR = "RA_SIM_CONFIG_DIR"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
FILE_PATHS_FILENAME = "file_paths.yaml"
FILE_PATHS_EXAMPLE_FILENAME = "file_paths.example.yaml"
def default_dirs() -> dict[str, str]:
    """Return the default directory mapping for the active user."""

    cache_root = user_cache_root()
    return {
        "downloads": str(Path.home() / "Downloads"),
        "overlay_dir": str(cache_root / "overlays"),
        "debug_log_dir": str(cache_root / "logs"),
        "file_dialog_dir": str(user_data_root()),
        "temp_root": str(cache_root),
    }


class _DefaultDirs(Mapping[str, str]):
    """Mapping view over the current per-user default directories."""

    def __getitem__(self, key: str) -> str:
        return default_dirs()[key]

    def __iter__(self):
        return iter(default_dirs())

    def __len__(self) -> int:
        return len(default_dirs())


DEFAULT_DIRS: Mapping[str, str] = _DefaultDirs()


def _path_base_dir(config_dir: Path) -> Path:
    """Return base directory used for relative file-path config values."""

    resolved_dir = Path(config_dir).resolve()
    if resolved_dir == DEFAULT_CONFIG_DIR.resolve():
        return resolved_dir.parent
    return resolved_dir


def _resolve_path_value(value: Any, *, config_dir: Path) -> Any:
    """Resolve relative file-path config values against stable config context."""

    if isinstance(value, str):
        expanded = os.path.expanduser(value)
        if not expanded:
            return expanded
        if expanded.startswith(("/", "\\")):
            return expanded
        path = Path(expanded)
        if path.is_absolute():
            return str(path)
        return str((_path_base_dir(config_dir) / path).resolve())
    if isinstance(value, list):
        return [_resolve_path_value(item, config_dir=config_dir) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_path_value(item, config_dir=config_dir) for item in value)
    return value


def get_config_dir() -> Path:
    """Return the active configuration directory.

    Order of precedence:
    1. ``RA_SIM_CONFIG_DIR`` environment variable when set.
    2. Repository-local ``config/`` directory.
    """

    env_path = os.environ.get(ENV_CONFIG_DIR)
    if env_path:
        return Path(os.path.expanduser(env_path)).resolve()
    return DEFAULT_CONFIG_DIR


def _escape_backslashes_in_double_quoted_yaml(text: str) -> str:
    """Escape ``\\`` in double-quoted YAML scalars for Windows path inputs."""

    out_chars: list[str] = []
    in_double = False
    prev = ""
    for ch in text:
        if ch == '"' and prev != "\\":
            in_double = not in_double
            out_chars.append(ch)
        elif in_double and ch == "\\":
            out_chars.append("\\\\")
        else:
            out_chars.append(ch)
        prev = ch
    return "".join(out_chars)


def _read_data_file(path: Path) -> dict[str, Any]:
    """Load a YAML/JSON mapping from *path*.

    Missing files return an empty mapping.
    """

    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = yaml.safe_load(_escape_backslashes_in_double_quoted_yaml(text))
    if data is None:
        return {}
    if isinstance(data, dict):
        return data

    # Support JSON files with stricter parser error messages when needed.
    if path.suffix.lower() == ".json":
        parsed = json.loads(text)
        return ensure_mapping(parsed, name=str(path))
    raise TypeError(f"{path} must contain a mapping at top level")


def _read_primary_or_example_file(
    config_dir: Path,
    *,
    primary_name: str,
    example_name: str | None = None,
) -> dict[str, Any]:
    """Load a config mapping from *primary_name* or fall back to *example_name*."""

    primary_path = config_dir / primary_name
    if primary_path.exists():
        return _read_data_file(primary_path)
    if example_name:
        example_path = config_dir / example_name
        if example_path.exists():
            return _read_data_file(example_path)
    return {}


def _load_from_dir(config_dir: Path) -> ConfigBundle:
    file_paths = ensure_mapping(
        _read_primary_or_example_file(
            config_dir,
            primary_name=FILE_PATHS_FILENAME,
            example_name=FILE_PATHS_EXAMPLE_FILENAME,
        ),
        name=FILE_PATHS_FILENAME,
    )
    dir_paths = ensure_mapping(
        _read_data_file(config_dir / "dir_paths.yaml"),
        name="dir_paths.yaml",
    )
    debug = ensure_mapping(
        _read_data_file(config_dir / "debug.yaml"),
        name="debug.yaml",
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
        debug=debug,
        materials=materials,
        instrument=instrument,
    )


_BUNDLE_CACHE: dict[Path, ConfigBundle] = {}
_TEMP_DIR_CACHE: dict[Path, Path] = {}


def _register_run_bundle_inputs(value: Any) -> None:
    try:
        from ra_sim.debug_controls import register_run_input_paths
    except Exception:
        return
    register_run_input_paths(value)


def clear_config_cache() -> None:
    """Clear cached configuration bundles."""

    _BUNDLE_CACHE.clear()
    _TEMP_DIR_CACHE.clear()


def get_config_bundle(config_dir: Path | None = None) -> ConfigBundle:
    """Return the active cached configuration bundle."""

    resolved_dir = (config_dir or get_config_dir()).resolve()
    bundle = _BUNDLE_CACHE.get(resolved_dir)
    if bundle is None:
        bundle = _load_from_dir(resolved_dir)
        _BUNDLE_CACHE[resolved_dir] = bundle
    return bundle


def get_path(key: str, *, config_dir: Path | None = None) -> Any:
    """Return the configured file-path value for ``key``."""

    resolved_dir = (config_dir or get_config_dir()).resolve()
    value = get_config_bundle(resolved_dir).file_paths.get(key)
    if value is None:
        raise KeyError(f"No path configured for {key!r}")
    resolved_value = _resolve_path_value(value, config_dir=resolved_dir)
    _register_run_bundle_inputs(resolved_value)
    return resolved_value


def get_path_first(*keys: str, config_dir: Path | None = None) -> Any:
    """Return the first configured file-path value among ``keys``."""

    bundle = get_config_bundle(config_dir)
    for key in keys:
        if key in bundle.file_paths and bundle.file_paths.get(key) is not None:
            return get_path(key, config_dir=config_dir)
    joined = ", ".join(repr(k) for k in keys)
    raise KeyError(f"No path configured for any of: {joined}")


def get_dir(key: str, *, config_dir: Path | None = None) -> Path:
    """Return the configured directory for ``key`` and ensure it exists."""

    bundle = get_config_bundle(config_dir)
    value = {**DEFAULT_DIRS, **bundle.dir_paths}.get(key)
    if value is None:
        raise KeyError(f"No directory configured for {key!r}")
    path = Path(os.path.expanduser(str(value)))
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_temp_dir(*, config_dir: Path | None = None) -> Path:
    """Return a dedicated temporary directory for the active config context."""

    resolved_dir = (config_dir or get_config_dir()).resolve()
    temp_dir = _TEMP_DIR_CACHE.get(resolved_dir)
    if temp_dir is None:
        base_path = get_dir("temp_root", config_dir=resolved_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="ra_sim_", dir=str(base_path)))
        _TEMP_DIR_CACHE[resolved_dir] = temp_dir
    return temp_dir


def get_material_config(
    material: str | None = None,
    *,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """Return the material configuration block used by simulation helpers."""

    materials_raw = get_config_bundle(config_dir).materials
    material_constants = materials_raw.get("constants", {})
    materials = materials_raw.get("materials", {})
    default_material = materials_raw.get("default_material")

    if not materials:
        raise KeyError(
            "No materials configured. Expected materials.yaml to define at least one entry."
        )

    if material is None:
        material = default_material
    if material is None:
        raise KeyError("No default material configured. Provide a material name explicitly.")

    try:
        material_block = materials[material]
    except KeyError as exc:
        available = ", ".join(sorted(materials)) or "<none>"
        raise KeyError(f"Unknown material {material!r}. Available materials: {available}") from exc

    return {
        "name": material,
        "material": material_block,
        "constants": material_constants,
    }


def list_materials(*, config_dir: Path | None = None) -> list[str]:
    """Return the sorted configured material identifiers."""

    materials = get_config_bundle(config_dir).materials.get("materials", {})
    return sorted(str(key) for key in materials)


def get_instrument_config(*, config_dir: Path | None = None) -> dict[str, Any]:
    """Return a defensive copy of the instrument configuration."""

    return copy.deepcopy(get_config_bundle(config_dir).instrument)


def get_debug_config(*, config_dir: Path | None = None) -> dict[str, Any]:
    """Return a defensive copy of the debug configuration."""

    return copy.deepcopy(get_config_bundle(config_dir).debug)
