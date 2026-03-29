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
    materials: dict | None = None,
    instrument: dict | None = None,
) -> Path:
    cfg = tmp_path / "cfg"
    cfg.mkdir(parents=True)
    _write_yaml(cfg / "file_paths.yaml", file_paths or {})
    _write_yaml(cfg / "dir_paths.yaml", dir_paths or {})
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
    assert loader.list_materials() == ["MatA", "MatB"]
    assert loader.get_material_config()["material"]["density"] == 1.0

    instrument = loader.get_instrument_config()
    instrument["instrument"]["detector"]["image_size"] = 42
    assert loader.get_instrument_config()["instrument"]["detector"]["image_size"] == 1234


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
