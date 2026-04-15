import importlib
import sys
from pathlib import Path


def test_sitecustomize_sets_default_pycache_prefix(monkeypatch, tmp_path: Path) -> None:
    previous_module = sys.modules.pop("sitecustomize", None)
    previous_ra_sim = sys.modules.pop("ra_sim", None)
    previous_prefix = getattr(sys, "pycache_prefix", None)
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    sys.pycache_prefix = None

    try:
        importlib.import_module("sitecustomize")
        assert sys.pycache_prefix == str(tmp_path / ".cache" / "ra_sim" / "dev" / "pycache")
        assert "ra_sim" not in sys.modules
    finally:
        sys.modules.pop("sitecustomize", None)
        if previous_module is not None:
            sys.modules["sitecustomize"] = previous_module
        if previous_ra_sim is not None:
            sys.modules["ra_sim"] = previous_ra_sim
        sys.pycache_prefix = previous_prefix


def test_sitecustomize_preserves_explicit_pycache_env(monkeypatch, tmp_path: Path) -> None:
    previous_module = sys.modules.pop("sitecustomize", None)
    previous_ra_sim = sys.modules.pop("ra_sim", None)
    previous_prefix = getattr(sys, "pycache_prefix", None)
    explicit = str(tmp_path / "explicit-pyc")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", explicit)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "unused")
    sys.pycache_prefix = explicit

    try:
        importlib.import_module("sitecustomize")
        assert sys.pycache_prefix == explicit
    finally:
        sys.modules.pop("sitecustomize", None)
        if previous_module is not None:
            sys.modules["sitecustomize"] = previous_module
        if previous_ra_sim is not None:
            sys.modules["ra_sim"] = previous_ra_sim
        sys.pycache_prefix = previous_prefix
