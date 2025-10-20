"""Utility for accessing common file and directory paths."""
from pathlib import Path
import os
import tempfile
import copy
import yaml

_YAML_PATH = Path(__file__).resolve().parents[1] / "file_paths.yaml"
_DIR_YAML_PATH = Path(__file__).resolve().parents[1] / "dir_paths.yaml"
_MATERIALS_YAML_PATH = Path(__file__).resolve().parents[1] / "materials.yaml"
_INSTRUMENT_YAML_PATH = Path(__file__).resolve().parents[1] / "instrument.yaml"

DEFAULT_DIRS = {
    "downloads": str(Path.home() / "Downloads"),
    "overlay_dir": str(Path.home() / ".cache" / "ra_sim" / "overlays"),
    "debug_log_dir": str(Path.home() / ".cache" / "ra_sim" / "logs"),
    "file_dialog_dir": str(Path.home() / ".local" / "share" / "ra_sim"),
    "temp_root": str(Path.home() / ".cache" / "ra_sim"),
}

with open(_YAML_PATH, "r", encoding="utf-8") as fh:
    PATHS = yaml.safe_load(fh)

if _DIR_YAML_PATH.exists():
    with open(_DIR_YAML_PATH, "r", encoding="utf-8") as fh:
        yaml_dirs = yaml.safe_load(fh)
else:
    yaml_dirs = {}

DIRS = {**DEFAULT_DIRS, **yaml_dirs}

if _MATERIALS_YAML_PATH.exists():
    with open(_MATERIALS_YAML_PATH, "r", encoding="utf-8") as fh:
        _materials_raw = yaml.safe_load(fh) or {}
else:  # pragma: no cover - configuration file is optional in tests
    _materials_raw = {}

_MATERIAL_CONSTANTS = _materials_raw.get("constants", {})
_MATERIALS = _materials_raw.get("materials", {})
_DEFAULT_MATERIAL = _materials_raw.get("default_material")

if _INSTRUMENT_YAML_PATH.exists():
    with open(_INSTRUMENT_YAML_PATH, "r", encoding="utf-8") as fh:
        _instrument_raw = yaml.safe_load(fh) or {}
else:  # pragma: no cover - configuration file is optional in tests
    _instrument_raw = {}


def get_path(key: str) -> str:
    """Return the configured path for *key* expanding '~'."""
    value = PATHS.get(key)
    if value is None:
        raise KeyError(f"No path configured for {key!r}")
    if isinstance(value, str):
        return os.path.expanduser(value)
    return value


def get_dir(key: str) -> Path:
    """Return the configured directory for *key*, creating it if needed."""
    value = DIRS.get(key)
    if value is None:
        raise KeyError(f"No directory configured for {key!r}")
    path = Path(os.path.expanduser(value))
    path.mkdir(parents=True, exist_ok=True)
    return path


_TEMP_DIR = None


def get_temp_dir() -> Path:
    """Return a dedicated temporary directory for the current session."""
    global _TEMP_DIR
    if _TEMP_DIR is None:
        base = DIRS.get("temp_root")
        if base:
            base_path = Path(os.path.expanduser(base))
        else:
            base_path = Path(tempfile.gettempdir()) / "ra_sim"
        base_path.mkdir(parents=True, exist_ok=True)
        _TEMP_DIR = Path(tempfile.mkdtemp(prefix="ra_sim_", dir=str(base_path)))
    return _TEMP_DIR


def get_material_config(material: str | None = None) -> dict:
    """Return the configuration block for the requested *material*.

    When *material* is ``None`` the default material defined in
    ``materials.yaml`` is returned.  The resulting dictionary always contains
    the material specific properties under ``"material"`` and any shared
    constants (``"constants"``).
    """

    if not _MATERIALS:
        raise KeyError(
            "No materials configured. Expected materials.yaml to define at least one entry."
        )

    if material is None:
        material = _DEFAULT_MATERIAL
    if material is None:
        raise KeyError("No default material configured. Provide a material name explicitly.")

    try:
        material_block = _MATERIALS[material]
    except KeyError as exc:  # pragma: no cover - defensive programming
        available = ", ".join(sorted(_MATERIALS)) or "<none>"
        raise KeyError(
            f"Unknown material {material!r}. Available materials: {available}"
        ) from exc

    return {
        "name": material,
        "material": material_block,
        "constants": _MATERIAL_CONSTANTS,
    }


def list_materials() -> list[str]:
    """Return the list of available material identifiers."""

    return sorted(_MATERIALS)


def get_instrument_config() -> dict:
    """Return the parsed instrument configuration."""

    return copy.deepcopy(_instrument_raw)
