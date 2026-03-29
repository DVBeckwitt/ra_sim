"""Load RA-SIM configuration files from disk."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .models import ConfigBundle
from .validation import ensure_mapping

ENV_CONFIG_DIR = "RA_SIM_CONFIG_DIR"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
DEFAULT_DIRS: dict[str, str] = {
    "downloads": str(Path.home() / "Downloads"),
    "overlay_dir": str(Path.home() / ".cache" / "ra_sim" / "overlays"),
    "debug_log_dir": str(Path.home() / ".cache" / "ra_sim" / "logs"),
    "file_dialog_dir": str(Path.home() / ".local" / "share" / "ra_sim"),
    "temp_root": str(Path.home() / ".cache" / "ra_sim"),
}


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


def get_path(key: str, *, config_dir: Path | None = None) -> Any:
    """Return the configured file-path value for ``key``."""

    value = get_config_bundle(config_dir).file_paths.get(key)
    if value is None:
        raise KeyError(f"No path configured for {key!r}")
    if isinstance(value, str):
        return os.path.expanduser(value)
    return value


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
        raise KeyError(
            f"Unknown material {material!r}. Available materials: {available}"
        ) from exc

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
