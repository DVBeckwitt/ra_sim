from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from ra_sim.config import loader
from ra_sim.fitting.optimization import SimulationCache

OPTIMIZATION_SOURCE_PATH = (
    Path(__file__).resolve().parent.parent
    / "ra_sim"
    / "fitting"
    / "optimization.py"
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _make_config_dir(tmp_path: Path, *, debug: dict | None = None) -> Path:
    cfg = tmp_path / "cfg"
    cfg.mkdir(parents=True)
    _write_yaml(cfg / "file_paths.yaml", {})
    _write_yaml(cfg / "dir_paths.yaml", {})
    _write_yaml(cfg / "materials.yaml", {})
    _write_yaml(cfg / "instrument.yaml", {})
    if debug is not None:
        _write_yaml(cfg / "debug.yaml", debug)
    return cfg


@pytest.fixture(autouse=True)
def _reset_loader_cache() -> None:
    loader.clear_config_cache()
    yield
    loader.clear_config_cache()


def test_simulation_cache_retains_entries_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={"debug": {"cache": {"families": {"fit_simulation": "always"}}}},
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))
    cache = SimulationCache(keys=["gamma"])
    params = {"gamma": 1.0}
    image = np.ones((2, 2), dtype=np.float64)
    max_positions = np.ones((1, 7), dtype=np.float64)

    cache.store(params, image, max_positions)

    cached = cache.get(params)
    assert cached is not None
    np.testing.assert_allclose(cached[0], image)
    np.testing.assert_allclose(cached[1], max_positions)


def test_simulation_cache_discards_entries_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={"debug": {"cache": {"families": {"fit_simulation": "never"}}}},
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))
    cache = SimulationCache(keys=["gamma"])
    params = {"gamma": 1.0}

    cache.store(
        params,
        np.ones((2, 2), dtype=np.float64),
        np.ones((1, 7), dtype=np.float64),
    )

    assert cache.get(params) is None
    assert cache.images == {}
    assert cache.max_positions == {}


def test_optimization_source_gates_fit_image_cache_with_retention_policy() -> None:
    source = OPTIMIZATION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "from ra_sim.debug_controls import retain_optional_cache" in source
    assert "def _retain_fit_simulation_cache() -> bool:" in source
    assert "retain_image_cache = _retain_fit_simulation_cache()" in source
    assert "if not retain_image_cache:" in source
