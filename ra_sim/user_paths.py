"""Helpers for stable per-user cache and data directories."""

from __future__ import annotations

from pathlib import Path


def user_cache_root() -> Path:
    """Return the RA-SIM cache root inside the current user's home directory."""

    return Path.home() / ".cache" / "ra_sim"


def user_data_root() -> Path:
    """Return the RA-SIM data root inside the current user's home directory."""

    return Path.home() / ".local" / "share" / "ra_sim"


def dev_cache_dir(name: str) -> Path:
    """Return a named developer-tool cache directory under the RA-SIM cache root."""

    return user_cache_root() / "dev" / name
