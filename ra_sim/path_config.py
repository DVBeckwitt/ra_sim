"""Utility for accessing common file and directory paths."""
from pathlib import Path
import os
import tempfile
import yaml

_YAML_PATH = Path(__file__).resolve().parents[1] / "file_paths.yaml"
_DIR_YAML_PATH = Path(__file__).resolve().parents[1] / "dir_paths.yaml"

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
