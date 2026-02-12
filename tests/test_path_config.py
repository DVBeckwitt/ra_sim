from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ra_sim import path_config


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
def _reset_config_cache():
    path_config.reload_config_cache()
    yield
    path_config.reload_config_cache()


def test_get_path_uses_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_a = _make_config_dir(tmp_path / "a", file_paths={"cif_file": "/tmp/a.cif"})
    cfg_b = _make_config_dir(tmp_path / "b", file_paths={"cif_file": "/tmp/b.cif"})

    monkeypatch.setenv("RA_SIM_CONFIG_DIR", str(cfg_a))
    path_config.reload_config_cache()
    assert path_config.get_path("cif_file") == "/tmp/a.cif"

    monkeypatch.setenv("RA_SIM_CONFIG_DIR", str(cfg_b))
    path_config.reload_config_cache()
    assert path_config.get_path("cif_file") == "/tmp/b.cif"


def test_get_path_expands_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _make_config_dir(tmp_path, file_paths={"overlay_output": "~/ra_sim/overlay.png"})
    monkeypatch.setenv("RA_SIM_CONFIG_DIR", str(cfg))
    expanded = Path("~/ra_sim/overlay.png").expanduser()
    assert Path(path_config.get_path("overlay_output")) == expanded


def test_get_path_missing_key_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _make_config_dir(tmp_path)
    monkeypatch.setenv("RA_SIM_CONFIG_DIR", str(cfg))
    with pytest.raises(KeyError):
        path_config.get_path("does_not_exist")


def test_get_dir_resolves_custom_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "artifacts"
    cfg = _make_config_dir(tmp_path, dir_paths={"downloads": str(out_dir)})
    monkeypatch.setenv("RA_SIM_CONFIG_DIR", str(cfg))
    resolved = path_config.get_dir("downloads")
    assert resolved == out_dir
    assert resolved.exists()


def test_material_helpers_backward_compatible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        materials={
            "default_material": "MatA",
            "constants": {"k": 1.0},
            "materials": {"MatA": {"density": 1.0}, "MatB": {"density": 2.0}},
        },
    )
    monkeypatch.setenv("RA_SIM_CONFIG_DIR", str(cfg))

    assert path_config.list_materials() == ["MatA", "MatB"]
    material_cfg = path_config.get_material_config()
    assert material_cfg["name"] == "MatA"
    assert material_cfg["material"]["density"] == 1.0
    assert material_cfg["constants"]["k"] == 1.0


def test_instrument_config_returns_deep_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        instrument={"instrument": {"detector": {"image_size": 1234}}},
    )
    monkeypatch.setenv("RA_SIM_CONFIG_DIR", str(cfg))

    original = path_config.get_instrument_config()
    original["instrument"]["detector"]["image_size"] = 42
    fresh = path_config.get_instrument_config()
    assert fresh["instrument"]["detector"]["image_size"] == 1234
