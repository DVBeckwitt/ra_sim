"""Utility for accessing common file and directory paths."""
from pathlib import Path
import os
import tempfile
import copy
import yaml

from ra_sim.config.loader import ENV_CONFIG_DIR

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

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

PATHS: dict = {}
DIRS: dict = dict(DEFAULT_DIRS)
_MATERIAL_CONSTANTS: dict = {}
_MATERIALS: dict = {}
_DEFAULT_MATERIAL = None
_instrument_raw: dict = {}
_ACTIVE_CONFIG_DIR: Path | None = None


def get_config_dir() -> Path:
    """Return the active configuration directory."""

    env_path = os.environ.get(ENV_CONFIG_DIR)
    if env_path:
        return Path(os.path.expanduser(env_path)).resolve()
    return _DEFAULT_CONFIG_DIR


def reload_config_cache() -> None:
    """Reload cached YAML configuration from the active config directory."""

    global PATHS, DIRS, _MATERIAL_CONSTANTS, _MATERIALS, _DEFAULT_MATERIAL
    global _instrument_raw, _TEMP_DIR, _ACTIVE_CONFIG_DIR

    config_dir = get_config_dir()
    yaml_path = config_dir / "file_paths.yaml"
    dir_yaml_path = config_dir / "dir_paths.yaml"
    materials_yaml_path = config_dir / "materials.yaml"
    instrument_yaml_path = config_dir / "instrument.yaml"

    if yaml_path.exists():
        PATHS = _load_yaml_with_windows_path_fallback(yaml_path)
    else:  # pragma: no cover - configuration file is optional in tests
        PATHS = {}

    if dir_yaml_path.exists():
        yaml_dirs = _load_yaml_with_windows_path_fallback(dir_yaml_path)
    else:
        yaml_dirs = {}
    DIRS = {**DEFAULT_DIRS, **yaml_dirs}

    if materials_yaml_path.exists():
        materials_raw = _load_yaml_with_windows_path_fallback(materials_yaml_path)
    else:  # pragma: no cover - configuration file is optional in tests
        materials_raw = {}
    _MATERIAL_CONSTANTS = materials_raw.get("constants", {})
    _MATERIALS = materials_raw.get("materials", {})
    _DEFAULT_MATERIAL = materials_raw.get("default_material")

    if instrument_yaml_path.exists():
        _instrument_raw = _load_yaml_with_windows_path_fallback(instrument_yaml_path)
    else:  # pragma: no cover - configuration file is optional in tests
        _instrument_raw = {}

    _ACTIVE_CONFIG_DIR = config_dir
    _TEMP_DIR = None


def _ensure_config_cache_current() -> None:
    if _ACTIVE_CONFIG_DIR != get_config_dir():
        reload_config_cache()


reload_config_cache()


def get_path(key: str) -> str:
    """Return the configured path for *key* expanding '~'."""
    _ensure_config_cache_current()
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

    _ensure_config_cache_current()
    for key in keys:
        if key in PATHS and PATHS.get(key) is not None:
            return get_path(key)
    joined = ", ".join(repr(k) for k in keys)
    raise KeyError(f"No path configured for any of: {joined}")


def get_dir(key: str) -> Path:
    """Return the configured directory for *key*, creating it if needed."""
    _ensure_config_cache_current()
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
    _ensure_config_cache_current()
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

    _ensure_config_cache_current()
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

    _ensure_config_cache_current()
    return sorted(_MATERIALS)


def get_instrument_config() -> dict:
    """Return the parsed instrument configuration."""

    _ensure_config_cache_current()
    return copy.deepcopy(_instrument_raw)
