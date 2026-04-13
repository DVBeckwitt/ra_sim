from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ra_sim.config import loader


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _make_config_dir(
    tmp_path: Path,
    *,
    file_paths: dict | None = None,
    dir_paths: dict | None = None,
    debug: dict | None = None,
    materials: dict | None = None,
    instrument: dict | None = None,
) -> Path:
    cfg = tmp_path / "cfg"
    cfg.mkdir(parents=True)
    _write_yaml(cfg / "file_paths.yaml", file_paths or {})
    _write_yaml(cfg / "dir_paths.yaml", dir_paths or {})
    if debug is not None:
        _write_yaml(cfg / "debug.yaml", debug)
    _write_yaml(cfg / "materials.yaml", materials or {})
    _write_yaml(cfg / "instrument.yaml", instrument or {})
    return cfg


@pytest.fixture(autouse=True)
def _reset_loader_cache() -> None:
    loader.clear_config_cache()
    yield
    loader.clear_config_cache()


def test_loader_helpers_read_active_config_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        file_paths={
            "cif_file": "~/materials/sample.cif",
            "new_key": "/tmp/new.osc",
            "legacy_key": "/tmp/legacy.osc",
        },
        dir_paths={"downloads": str(tmp_path / "downloads")},
        debug={"debug": {"console": {"enabled": True}}},
        materials={
            "default_material": "MatA",
            "constants": {"k": 1.0},
            "materials": {"MatA": {"density": 1.0}, "MatB": {"density": 2.0}},
        },
        instrument={"instrument": {"detector": {"image_size": 1234}}},
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))

    bundle = loader.get_config_bundle()
    assert bundle.config_dir == cfg.resolve()
    assert Path(loader.get_path("cif_file")) == Path("~/materials/sample.cif").expanduser()
    assert loader.get_path_first("new_key", "legacy_key") == "/tmp/new.osc"
    assert loader.get_dir("downloads") == tmp_path / "downloads"
    assert bundle.debug == {"debug": {"console": {"enabled": True}}}
    assert loader.list_materials() == ["MatA", "MatB"]
    assert loader.get_material_config()["material"]["density"] == 1.0

    instrument = loader.get_instrument_config()
    instrument["instrument"]["detector"]["image_size"] = 42
    assert loader.get_instrument_config()["instrument"]["detector"]["image_size"] == 1234
    debug = loader.get_debug_config()
    debug["debug"]["console"]["enabled"] = False
    assert loader.get_debug_config()["debug"]["console"]["enabled"] is True


def test_loader_supports_windows_paths_in_double_quoted_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "cfg"
    cfg.mkdir(parents=True)
    (cfg / "file_paths.yaml").write_text(
        'osc_file: "C:\\Users\\Kenpo\\data\\sample.osc"\n',
        encoding="utf-8",
    )
    _write_yaml(cfg / "dir_paths.yaml", {})
    _write_yaml(cfg / "materials.yaml", {})
    _write_yaml(cfg / "instrument.yaml", {})
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))

    assert loader.get_path("osc_file") == "C:\\Users\\Kenpo\\data\\sample.osc"


def test_loader_falls_back_to_example_file_paths_when_primary_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "cfg"
    cfg.mkdir(parents=True)
    _write_yaml(cfg / "file_paths.example.yaml", {"cif_file": "/tmp/example.cif"})
    _write_yaml(cfg / "dir_paths.yaml", {})
    _write_yaml(cfg / "materials.yaml", {})
    _write_yaml(cfg / "instrument.yaml", {})
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))

    assert loader.get_path("cif_file") == "/tmp/example.cif"


def test_loader_resolves_relative_file_paths_against_external_config_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        file_paths={
            "cif_file": "./inputs/sample.cif",
            "simulation_background_osc_files": ["./images/a.osc", "./images/b.osc"],
        },
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))

    assert loader.get_path("cif_file") == str((cfg / "inputs" / "sample.cif").resolve())
    assert loader.get_path("simulation_background_osc_files") == [
        str((cfg / "images" / "a.osc").resolve()),
        str((cfg / "images" / "b.osc").resolve()),
    ]


def test_repo_default_config_resolves_relative_paths_against_repo_root() -> None:
    expected = (
        Path(__file__).resolve().parents[1] / "data" / "geometry.poni"
    ).resolve()

    resolved = loader._resolve_path_value(
        "./data/geometry.poni",
        config_dir=loader.DEFAULT_CONFIG_DIR,
    )

    assert Path(str(resolved)).resolve() == expected


def test_loader_cache_can_be_cleared_after_file_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(tmp_path, file_paths={"cif_file": "/tmp/first.cif"})
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))

    assert loader.get_path("cif_file") == "/tmp/first.cif"

    _write_yaml(cfg / "file_paths.yaml", {"cif_file": "/tmp/second.cif"})
    assert loader.get_path("cif_file") == "/tmp/first.cif"

    loader.clear_config_cache()
    assert loader.get_path("cif_file") == "/tmp/second.cif"


def test_loader_switches_config_dirs_when_env_override_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_a = _make_config_dir(tmp_path / "a", file_paths={"cif_file": "/tmp/a.cif"})
    cfg_b = _make_config_dir(tmp_path / "b", file_paths={"cif_file": "/tmp/b.cif"})

    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg_a))
    assert loader.get_path("cif_file") == "/tmp/a.cif"

    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg_b))
    assert loader.get_path("cif_file") == "/tmp/b.cif"


def test_loader_get_path_missing_key_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(tmp_path)
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))

    with pytest.raises(KeyError, match="does_not_exist"):
        loader.get_path("does_not_exist")


def test_loader_get_temp_dir_uses_temp_root_and_caches_per_config_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(tmp_path, dir_paths={"temp_root": str(tmp_path / "scratch")})
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))

    temp_dir_a = loader.get_temp_dir()
    temp_dir_b = loader.get_temp_dir()

    assert temp_dir_a == temp_dir_b
    assert temp_dir_a.parent == tmp_path / "scratch"
    assert temp_dir_a.exists()


def test_repo_instrument_defaults_weight_p0_fully_by_default() -> None:
    instrument_path = Path(__file__).resolve().parents[1] / "config" / "instrument.yaml"
    instrument = yaml.safe_load(instrument_path.read_text(encoding="utf-8"))

    assert instrument["instrument"]["hendricks_teller"]["default_w"] == [
        100.0,
        0.0,
        0.0,
    ]


def test_repo_debug_defaults_disable_console_but_leave_outputs_enabled() -> None:
    debug_path = Path(__file__).resolve().parents[1] / "config" / "debug.yaml"
    debug = yaml.safe_load(debug_path.read_text(encoding="utf-8"))

    assert debug["debug"]["global"]["disable_all"] is False
    assert debug["debug"]["console"]["enabled"] is False
    assert debug["debug"]["runtime_update_trace"]["enabled"] is True
    assert debug["debug"]["geometry_fit"]["log_files"] is True
    assert debug["debug"]["mosaic_fit"]["log_files"] is True
    assert debug["debug"]["cache"]["default_retention"] == "auto"
    assert debug["debug"]["cache"]["families"]["primary_contribution"] == "auto"
    assert debug["debug"]["cache"]["families"]["diffraction_last_intersection"] == "never"
