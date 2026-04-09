from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ra_sim.config import loader
from ra_sim.debug_controls import (
    console_debug_enabled,
    diffraction_debug_csv_logging_enabled,
    geometry_fit_extra_sections_enabled,
    geometry_fit_log_files_enabled,
    intersection_cache_logging_enabled,
    is_logging_disabled,
    mosaic_fit_log_files_enabled,
    projection_debug_logging_enabled,
    resolve_intersection_cache_log_root,
    runtime_update_trace_logging_enabled,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _make_config_dir(
    tmp_path: Path,
    *,
    debug: dict | None = None,
    instrument: dict | None = None,
) -> Path:
    cfg = tmp_path / "cfg"
    cfg.mkdir(parents=True)
    _write_yaml(cfg / "file_paths.yaml", {})
    _write_yaml(cfg / "dir_paths.yaml", {"debug_log_dir": str(tmp_path / "logs")})
    _write_yaml(cfg / "materials.yaml", {})
    if debug is not None:
        _write_yaml(cfg / "debug.yaml", debug)
    _write_yaml(cfg / "instrument.yaml", instrument or {})
    return cfg


@pytest.fixture(autouse=True)
def _reset_loader_cache() -> None:
    loader.clear_config_cache()
    yield
    loader.clear_config_cache()


def test_debug_controls_follow_config_values_when_env_overrides_are_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={
            "debug": {
                "console": {"enabled": True},
                "runtime_update_trace": {"enabled": False},
                "geometry_fit": {"log_files": False, "extra_sections": False},
                "mosaic_fit": {"log_files": False},
                "projection_debug": {"enabled": False},
                "diffraction_debug_csv": {"enabled": False},
                "intersection_cache": {
                    "enabled": False,
                    "log_dir": str(tmp_path / "cache-dumps"),
                },
            }
        },
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))

    assert console_debug_enabled()
    assert not runtime_update_trace_logging_enabled()
    assert not geometry_fit_log_files_enabled()
    assert not geometry_fit_extra_sections_enabled()
    assert not mosaic_fit_log_files_enabled()
    assert not projection_debug_logging_enabled()
    assert not diffraction_debug_csv_logging_enabled()
    assert not intersection_cache_logging_enabled()
    assert resolve_intersection_cache_log_root() == tmp_path / "cache-dumps"


def test_env_overrides_take_precedence_over_debug_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={
            "debug": {
                "console": {"enabled": False},
                "projection_debug": {"enabled": True},
                "intersection_cache": {
                    "enabled": False,
                    "log_dir": str(tmp_path / "from-config"),
                },
            }
        },
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))
    monkeypatch.setenv("RA_SIM_DEBUG", "1")
    monkeypatch.setenv("RA_SIM_DISABLE_PROJECTION_DEBUG", "1")
    monkeypatch.setenv("RA_SIM_LOG_INTERSECTION_CACHE", "1")
    monkeypatch.setenv(
        "RA_SIM_INTERSECTION_CACHE_LOG_DIR",
        str(tmp_path / "from-env"),
    )

    assert console_debug_enabled()
    assert not projection_debug_logging_enabled()
    assert intersection_cache_logging_enabled()
    assert resolve_intersection_cache_log_root() == tmp_path / "from-env"


def test_config_global_disable_is_a_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={
            "debug": {
                "global": {"disable_all": True},
                "console": {"enabled": True},
                "runtime_update_trace": {"enabled": True},
                "geometry_fit": {"log_files": True, "extra_sections": True},
                "mosaic_fit": {"log_files": True},
                "projection_debug": {"enabled": True},
                "diffraction_debug_csv": {"enabled": True},
                "intersection_cache": {"enabled": True},
            }
        },
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))
    monkeypatch.setenv("RA_SIM_DEBUG", "1")
    monkeypatch.setenv("RA_SIM_LOG_INTERSECTION_CACHE", "1")

    assert is_logging_disabled()
    assert not console_debug_enabled()
    assert not runtime_update_trace_logging_enabled()
    assert not geometry_fit_log_files_enabled()
    assert not geometry_fit_extra_sections_enabled()
    assert not mosaic_fit_log_files_enabled()
    assert not projection_debug_logging_enabled()
    assert not diffraction_debug_csv_logging_enabled()
    assert not intersection_cache_logging_enabled()


def test_env_global_disable_is_a_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={
            "debug": {
                "console": {"enabled": True},
                "projection_debug": {"enabled": True},
                "intersection_cache": {"enabled": True},
            }
        },
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))
    monkeypatch.setenv("RA_SIM_DISABLE_ALL_LOGGING", "1")
    monkeypatch.setenv("RA_SIM_DEBUG", "1")
    monkeypatch.setenv("RA_SIM_LOG_INTERSECTION_CACHE", "1")

    assert is_logging_disabled()
    assert not console_debug_enabled()
    assert not projection_debug_logging_enabled()
    assert not intersection_cache_logging_enabled()


def test_geometry_fit_extra_sections_fall_back_to_legacy_instrument_key_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        instrument={"instrument": {"fit": {"geometry": {"debug_logging": False}}}},
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))

    assert geometry_fit_log_files_enabled()
    assert not geometry_fit_extra_sections_enabled()


def test_geometry_fit_debug_yaml_overrides_legacy_instrument_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={"debug": {"geometry_fit": {"extra_sections": True}}},
        instrument={"instrument": {"fit": {"geometry": {"debug_logging": False}}}},
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))

    assert geometry_fit_extra_sections_enabled() is True
