"""Top-level package bootstrap for RA-SIM."""

from __future__ import annotations

import os
from pathlib import Path


def _ensure_numba_cache_dir() -> None:
    """Set a stable default ``NUMBA_CACHE_DIR`` when user did not provide one."""

    if "NUMBA_CACHE_DIR" in os.environ:
        return

    cache_dir = Path.home() / ".cache" / "ra_sim" / "numba"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    try:
        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
    except Exception:
        return


_ensure_numba_cache_dir()
