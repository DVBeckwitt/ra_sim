from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest
import yaml

from ra_sim.config import loader
from ra_sim.debug_controls import (
    cache_retention_mode,
    console_debug_enabled,
    diffraction_debug_csv_logging_enabled,
    finalize_run_bundle,
    geometry_fit_extra_sections_enabled,
    geometry_fit_log_files_enabled,
    intersection_cache_logging_enabled,
    is_logging_disabled,
    mosaic_fit_log_files_enabled,
    projection_debug_logging_enabled,
    register_run_output_path,
    retain_optional_cache,
    reset_run_bundle_state,
    resolve_intersection_cache_log_root,
    resolve_startup_debug_log_path,
    start_run_bundle,
    temporary_startup_debug_override,
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


def test_startup_debug_log_path_is_reused_for_one_process(tmp_path: Path) -> None:
    first = resolve_startup_debug_log_path(
        stamp="20260328_120000",
        log_dir=tmp_path / "logs",
    )
    second = resolve_startup_debug_log_path(
        stamp="20260328_120500",
        log_dir=tmp_path / "other-logs",
    )

    assert first == tmp_path / "logs" / "geometry_fit_log_20260328_120000.txt"
    assert second == first


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


def test_startup_debug_override_can_force_all_debug_on(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={
            "debug": {
                "global": {"disable_all": True},
                "console": {"enabled": False},
                "runtime_update_trace": {"enabled": False},
                "geometry_fit": {"log_files": False, "extra_sections": False},
                "mosaic_fit": {"log_files": False},
                "projection_debug": {"enabled": False},
                "diffraction_debug_csv": {"enabled": False},
                "intersection_cache": {"enabled": False},
            }
        },
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))
    monkeypatch.setenv("RA_SIM_DISABLE_ALL_LOGGING", "1")
    monkeypatch.setenv("RA_SIM_DISABLE_PROJECTION_DEBUG", "1")

    with temporary_startup_debug_override("enable_all"):
        assert not is_logging_disabled()
        assert console_debug_enabled()
        assert runtime_update_trace_logging_enabled()
        assert geometry_fit_log_files_enabled()
        assert geometry_fit_extra_sections_enabled()
        assert mosaic_fit_log_files_enabled()
        assert projection_debug_logging_enabled()
        assert diffraction_debug_csv_logging_enabled()
        assert intersection_cache_logging_enabled()


def test_startup_debug_override_can_force_all_debug_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={
            "debug": {
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

    with temporary_startup_debug_override("disable_all"):
        assert is_logging_disabled()
        assert not console_debug_enabled()
        assert not runtime_update_trace_logging_enabled()
        assert not geometry_fit_log_files_enabled()
        assert not geometry_fit_extra_sections_enabled()
        assert not mosaic_fit_log_files_enabled()
        assert not projection_debug_logging_enabled()
        assert not diffraction_debug_csv_logging_enabled()
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


def test_cache_retention_defaults_and_family_overrides_follow_debug_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={
            "debug": {
                "cache": {
                    "default_retention": "never",
                    "families": {
                        "primary_contribution": "always",
                        "diffraction_last_intersection": "auto",
                    },
                }
            }
        },
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))

    assert cache_retention_mode("primary_contribution") == "always"
    assert cache_retention_mode("caking") == "never"
    assert cache_retention_mode("diffraction_last_intersection") == "auto"
    assert retain_optional_cache("primary_contribution", feature_needed=False) is True
    assert retain_optional_cache("caking", feature_needed=True) is False
    assert retain_optional_cache("diffraction_last_intersection", feature_needed=True) is True


def test_cache_retention_is_separate_from_global_logging_disable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={
            "debug": {
                "global": {"disable_all": True},
                "cache": {
                    "default_retention": "always",
                    "families": {"caking": "never"},
                },
            }
        },
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))

    assert is_logging_disabled()
    assert cache_retention_mode("primary_contribution") == "always"
    assert cache_retention_mode("caking") == "never"
    assert retain_optional_cache("primary_contribution", feature_needed=False) is True
    assert retain_optional_cache("caking", feature_needed=True) is False


def test_run_bundle_zips_run_outputs_and_non_osc_non_cif_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "cfg"
    cfg.mkdir(parents=True)
    inputs_dir = cfg / "inputs"
    inputs_dir.mkdir()
    logs_dir = tmp_path / "logs"
    cache_logs_dir = tmp_path / "cache-logs"
    _write_yaml(
        cfg / "file_paths.yaml",
        {
            "measured_peaks": "./inputs/measured_peaks.npy",
            "geometry_poni": "./inputs/geometry.poni",
            "parameters_file": "./inputs/parameters.npy",
            "simulation_background_osc_files": ["./inputs/background_01.osc"],
            "cif_file": "./inputs/sample.cif",
        },
    )
    _write_yaml(
        cfg / "dir_paths.yaml",
        {
            "debug_log_dir": str(logs_dir),
            "downloads": str(tmp_path / "downloads"),
        },
    )
    _write_yaml(
        cfg / "debug.yaml",
        {
            "debug": {
                "intersection_cache": {
                    "enabled": True,
                    "log_dir": str(cache_logs_dir),
                }
            }
        },
    )
    _write_yaml(cfg / "materials.yaml", {})
    _write_yaml(cfg / "instrument.yaml", {})
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))

    measured_path = inputs_dir / "measured_peaks.npy"
    measured_path.write_text("measured", encoding="utf-8")
    poni_path = inputs_dir / "geometry.poni"
    poni_path.write_text("poni", encoding="utf-8")
    params_path = inputs_dir / "parameters.npy"
    params_path.write_text("params", encoding="utf-8")
    osc_path = inputs_dir / "background_01.osc"
    osc_path.write_text("osc", encoding="utf-8")
    cif_path = inputs_dir / "sample.cif"
    cif_path.write_text("cif", encoding="utf-8")

    start_run_bundle(entrypoint="test:bundle")

    loader.get_path("measured_peaks")
    loader.get_path("geometry_poni")
    loader.get_path("parameters_file")
    loader.get_path("simulation_background_osc_files")
    loader.get_path("cif_file")

    run_log_path = logs_dir / "geometry_fit_log_test.txt"
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    run_log_path.write_text("fit log", encoding="utf-8")

    cache_dump_path = cache_logs_dir / "intersection_cache" / "intersection_cache_test" / "table_0000.npy"
    cache_dump_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dump_path.write_text("cache", encoding="utf-8")

    output_path = tmp_path / "artifacts" / "result.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("png", encoding="utf-8")
    register_run_output_path(output_path)

    bundle_path = finalize_run_bundle()

    assert bundle_path is not None
    assert bundle_path == logs_dir / bundle_path.name
    assert bundle_path.exists()

    with zipfile.ZipFile(bundle_path) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    assert "manifest.json" in names
    assert any(name.endswith("roots/debug_log_dir/geometry_fit_log_test.txt") for name in names)
    assert any(name.endswith("roots/intersection_cache_log_root/intersection_cache/intersection_cache_test/table_0000.npy") for name in names)
    assert any(name.startswith("inputs/") and name.endswith("/measured_peaks.npy") for name in names)
    assert any(name.startswith("inputs/") and name.endswith("/geometry.poni") for name in names)
    assert any(name.startswith("inputs/") and name.endswith("/parameters.npy") for name in names)
    assert any(name.startswith("outputs/") and name.endswith("/result.png") for name in names)
    assert not any(name.endswith(".osc") for name in names)
    assert not any(name.endswith(".cif") for name in names)
    assert str(osc_path.resolve()) in manifest["omitted_inputs"]
    assert str(cif_path.resolve()) in manifest["omitted_inputs"]
    assert manifest["entrypoints"] == ["test:bundle"]

    reset_run_bundle_state()
