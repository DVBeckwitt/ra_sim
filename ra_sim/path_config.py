"""Utility for accessing common file and directory paths."""
from pathlib import Path
import os
import tempfile
import copy
import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

_YAML_PATH = _CONFIG_DIR / "file_paths.yaml"
_DIR_YAML_PATH = _CONFIG_DIR / "dir_paths.yaml"
_MATERIALS_YAML_PATH = _CONFIG_DIR / "materials.yaml"
_INSTRUMENT_YAML_PATH = _CONFIG_DIR / "instrument.yaml"

DEFAULT_DIRS = {
    "downloads": str(Path.home() / "Downloads"),
    "overlay_dir": str(Path.home() / ".cache" / "ra_sim" / "overlays"),
    "debug_log_dir": str(Path.home() / ".cache" / "ra_sim" / "logs"),
    "file_dialog_dir": str(Path.home() / ".local" / "share" / "ra_sim"),
    "temp_root": str(Path.home() / ".cache" / "ra_sim"),
}

def _escape_backslashes_in_double_quoted_yaml(text: str) -> str:
    """Escape ``\\`` inside double-quoted YAML scalars.

    This allows Windows paths like:
    ``"C:\\Users\\Kenpo\\...\\file.osc"``
    to be accepted even when users do not manually escape backslashes.
    """

    out_chars = []
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


def _load_yaml_with_windows_path_fallback(path: Path) -> dict:
    """Load YAML, retrying with Windows-path backslash escaping if needed."""

    raw = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        sanitized = _escape_backslashes_in_double_quoted_yaml(raw)
        data = yaml.safe_load(sanitized)
    return data or {}


PATHS = _load_yaml_with_windows_path_fallback(_YAML_PATH)

if _DIR_YAML_PATH.exists():
    yaml_dirs = _load_yaml_with_windows_path_fallback(_DIR_YAML_PATH)
else:
    yaml_dirs = {}

DIRS = {**DEFAULT_DIRS, **yaml_dirs}

if _MATERIALS_YAML_PATH.exists():
    _materials_raw = _load_yaml_with_windows_path_fallback(_MATERIALS_YAML_PATH)
else:  # pragma: no cover - configuration file is optional in tests
    _materials_raw = {}

_MATERIAL_CONSTANTS = _materials_raw.get("constants", {})
_MATERIALS = _materials_raw.get("materials", {})
_DEFAULT_MATERIAL = _materials_raw.get("default_material")

if _INSTRUMENT_YAML_PATH.exists():
    _instrument_raw = _load_yaml_with_windows_path_fallback(_INSTRUMENT_YAML_PATH)
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


def get_path_first(*keys: str):
    """Return the first configured path among *keys*.

    This is useful for smooth key migrations in ``file_paths.yaml`` where a
    newer explicit key name should take priority but legacy keys are still
    accepted.
    """

    for key in keys:
        if key in PATHS and PATHS.get(key) is not None:
            return get_path(key)
    joined = ", ".join(repr(k) for k in keys)
    raise KeyError(f"No path configured for any of: {joined}")


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
