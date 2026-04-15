import importlib
import os
import sys
from pathlib import Path
from typing import Any


def _snapshot_ra_sim_modules() -> dict[str, Any]:
    return {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == "ra_sim" or name.startswith("ra_sim.")
    }


def _clear_ra_sim_modules() -> None:
    for name in list(sys.modules):
        if name == "ra_sim" or name.startswith("ra_sim."):
            sys.modules.pop(name, None)


def _restore_modules(modules: dict[str, Any]) -> None:
    for name in list(sys.modules):
        if name == "ra_sim" or name.startswith("ra_sim."):
            sys.modules.pop(name, None)
    sys.modules.update(modules)


def _fresh_import_and_restore(module_name: str) -> None:
    previous = _snapshot_ra_sim_modules()
    try:
        _clear_ra_sim_modules()
        importlib.import_module(module_name)
    finally:
        _restore_modules(previous)


def _default_numba_cache_path(home: str) -> str:
    return str((Path(home) / ".cache" / "ra_sim" / "numba").resolve())


def test_numba_cache_dir_defaults_to_stable_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    _fresh_import_and_restore("ra_sim")

    expected = _default_numba_cache_path(str(tmp_path))
    assert os.environ["NUMBA_CACHE_DIR"] == expected


def test_numba_cache_dir_preserves_existing_env(monkeypatch, tmp_path):
    existing = str(tmp_path / "existing-numba-cache")
    monkeypatch.setenv("NUMBA_CACHE_DIR", existing)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "unused")

    _fresh_import_and_restore("ra_sim")

    assert os.environ["NUMBA_CACHE_DIR"] == existing


def test_numba_cache_dir_bootstraps_before_main_entrypoint_cli_forward(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    previous = _snapshot_ra_sim_modules()

    try:
        _clear_ra_sim_modules()
        importlib.import_module("ra_sim.__main__")
        expected = _default_numba_cache_path(str(tmp_path))
        assert os.environ["NUMBA_CACHE_DIR"] == expected
        assert "ra_sim.cli" not in sys.modules
    finally:
        _restore_modules(previous)


def test_numba_cache_dir_present_before_cli_jit_modules_import(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    previous = _snapshot_ra_sim_modules()

    try:
        _clear_ra_sim_modules()
        importlib.import_module("ra_sim.cli")
        expected = _default_numba_cache_path(str(tmp_path))
        assert os.environ["NUMBA_CACHE_DIR"] == expected
    finally:
        _restore_modules(previous)
