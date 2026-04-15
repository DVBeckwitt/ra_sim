"""Route Python bytecode caches into the RA-SIM user cache when importable."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _default_pycache_dir() -> Path:
    return Path.home() / ".cache" / "ra_sim" / "dev" / "pycache"


def _ensure_pycache_prefix() -> None:
    if os.environ.get("PYTHONPYCACHEPREFIX"):
        return
    if getattr(sys, "pycache_prefix", None) is not None:
        return

    cache_dir = _default_pycache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    sys.pycache_prefix = str(cache_dir)


_ensure_pycache_prefix()
