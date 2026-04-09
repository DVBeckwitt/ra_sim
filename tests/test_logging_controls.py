from pathlib import Path

import pytest
import yaml

from ra_sim.config import loader
from ra_sim.debug_utils import is_debug_enabled, is_logging_disabled
from ra_sim.gui import geometry_fit, runtime_update_trace
from ra_sim.simulation import diffraction, projection_debug


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
    _write_yaml(
        cfg / "dir_paths.yaml",
        {
            "downloads": str(tmp_path / "downloads"),
            "debug_log_dir": str(tmp_path / "logs"),
        },
    )
    if debug is not None:
        _write_yaml(cfg / "debug.yaml", debug)
    _write_yaml(cfg / "materials.yaml", {})
    _write_yaml(cfg / "instrument.yaml", instrument or {})
    return cfg


@pytest.fixture(autouse=True)
def _reset_loader_cache() -> None:
    loader.clear_config_cache()
    yield
    loader.clear_config_cache()


def test_is_logging_disabled_accepts_preferred_and_legacy_flags() -> None:
    assert is_logging_disabled({"RA_SIM_DISABLE_ALL_LOGGING": "1"})
    assert is_logging_disabled({"RA_SIM_DISABLE_LOGGING": "true"})
    assert not is_logging_disabled({})


def test_global_logging_disable_suppresses_debug_mode(monkeypatch) -> None:
    monkeypatch.setenv("RA_SIM_DEBUG", "1")
    monkeypatch.setenv("RA_SIM_DISABLE_ALL_LOGGING", "1")

    assert not is_debug_enabled()


def test_runtime_update_trace_noops_when_global_logging_is_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RA_SIM_DISABLE_ALL_LOGGING", "1")
    trace_path = tmp_path / "runtime_update_trace_20260409.log"

    runtime_update_trace.append_runtime_update_trace_line(
        trace_path,
        "schedule_update",
        queued=True,
    )

    assert not trace_path.exists()


def test_runtime_update_trace_respects_debug_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={"debug": {"runtime_update_trace": {"enabled": False}}},
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))
    trace_path = tmp_path / "runtime_update_trace_20260409.log"

    runtime_update_trace.append_runtime_update_trace_line(
        trace_path,
        "schedule_update",
        queued=True,
    )

    assert not trace_path.exists()


def test_projection_debug_logging_respects_global_disable_aliases(monkeypatch) -> None:
    monkeypatch.delenv("RA_SIM_DISABLE_ALL_LOGGING", raising=False)
    monkeypatch.delenv("RA_SIM_DISABLE_LOGGING", raising=False)
    monkeypatch.delenv("RA_SIM_DISABLE_PROJECTION_DEBUG", raising=False)
    assert projection_debug.projection_debug_logging_enabled()

    monkeypatch.setenv("RA_SIM_DISABLE_ALL_LOGGING", "1")
    assert not projection_debug.projection_debug_logging_enabled()

    monkeypatch.delenv("RA_SIM_DISABLE_ALL_LOGGING", raising=False)
    monkeypatch.setenv("RA_SIM_DISABLE_LOGGING", "1")
    assert not projection_debug.projection_debug_logging_enabled()


def test_projection_debug_logging_respects_debug_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={"debug": {"projection_debug": {"enabled": False}}},
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))

    assert not projection_debug.projection_debug_logging_enabled()


def test_intersection_cache_logging_respects_global_disable_aliases(monkeypatch) -> None:
    monkeypatch.delenv("RA_SIM_DISABLE_ALL_LOGGING", raising=False)
    monkeypatch.delenv("RA_SIM_DISABLE_LOGGING", raising=False)
    monkeypatch.setenv("RA_SIM_LOG_INTERSECTION_CACHE", "1")
    assert diffraction._should_log_intersection_cache()

    monkeypatch.setenv("RA_SIM_DISABLE_ALL_LOGGING", "1")
    assert not diffraction._should_log_intersection_cache()

    monkeypatch.delenv("RA_SIM_DISABLE_ALL_LOGGING", raising=False)
    monkeypatch.setenv("RA_SIM_DISABLE_LOGGING", "1")
    assert not diffraction._should_log_intersection_cache()


def test_intersection_cache_logging_respects_debug_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={"debug": {"intersection_cache": {"enabled": False}}},
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))

    assert not diffraction._should_log_intersection_cache()


def test_geometry_fit_logging_disable_helper_accepts_legacy_alias() -> None:
    assert geometry_fit.geometry_fit_all_logging_disabled(
        {"RA_SIM_DISABLE_LOGGING": "1"}
    )


def test_geometry_fit_debug_sections_respect_debug_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={"debug": {"geometry_fit": {"log_files": True, "extra_sections": False}}},
        instrument={"instrument": {"fit": {"geometry": {"debug_logging": True}}}},
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))

    assert not geometry_fit.geometry_fit_debug_logging_enabled({"debug_logging": True})
