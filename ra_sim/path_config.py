"""Utility for accessing common file paths."""
from pathlib import Path
import os
import yaml

_YAML_PATH = Path(__file__).resolve().parents[1] / "file_paths.yaml"

with open(_YAML_PATH, "r", encoding="utf-8") as fh:
    PATHS = yaml.safe_load(fh)


def get_path(key: str) -> str:
    """Return the configured path for *key* expanding '~'."""
    value = PATHS.get(key)
    if value is None:
        raise KeyError(f"No path configured for {key!r}")
    if isinstance(value, str):
        return os.path.expanduser(value)
    return value
