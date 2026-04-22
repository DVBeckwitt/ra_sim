"""Top-level package bootstrap for RA-SIM."""

from __future__ import annotations

import os
import sys

from .user_paths import user_cache_root


def _ensure_numba_cache_dir() -> None:
    """Set a stable default ``NUMBA_CACHE_DIR`` when user did not provide one."""

    if "NUMBA_CACHE_DIR" in os.environ:
        return

    cache_tag = getattr(sys.implementation, "cache_tag", None)
    if not cache_tag:
        cache_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    cache_dir = user_cache_root() / "numba" / str(cache_tag)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    try:
        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
    except Exception:
        return


_ensure_numba_cache_dir()
