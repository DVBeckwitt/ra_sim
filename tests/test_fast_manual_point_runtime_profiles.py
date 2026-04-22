"""Focused runtime-profile regressions for interactive geometry fitting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from ra_sim.gui import geometry_fit


def _load_new4_ladder_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "debug"
        / "run_new4_geometry_fit_ladder.py"
    )
    spec = importlib.util.spec_from_file_location("new4_ladder_runtime_test", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manual_point_runtime_overrides_use_fast_safe_defaults() -> None:
    changed = geometry_fit.apply_manual_point_geometry_fit_runtime_overrides(
        {
            "solver": {
                "max_nfev": 400,
                "workers": "auto",
                "parallel_mode": "auto",
                "worker_numba_threads": 4,
            },
            "identifiability": {
                "enabled": True,
                "auto_freeze": True,
                "selective_thaw": {"enabled": True},
                "adaptive_regularization": {"enabled": True},
            },
        },
        joint_background_mode=False,
    )

    assert changed["solver"]["max_nfev"] == 30
    assert changed["solver"]["workers"] == 1
    assert changed["solver"]["parallel_mode"] == "off"
    assert changed["solver"]["worker_numba_threads"] == 0
    assert changed["identifiability"]["enabled"] is False
    assert "auto_freeze" not in changed["identifiability"]
    assert "selective_thaw" not in changed["identifiability"]
    assert "adaptive_regularization" not in changed["identifiability"]


def test_manual_point_runtime_overrides_preserve_unsafe_parallel_but_cap_fit() -> None:
    changed = geometry_fit.apply_manual_point_geometry_fit_runtime_overrides(
        {
            "solver": {
                "max_nfev": 400,
                "workers": "auto",
                "parallel_mode": "auto",
                "worker_numba_threads": 0,
            },
            "identifiability": {"enabled": True},
            "allow_unsafe_runtime": True,
        },
        joint_background_mode=False,
    )

    assert changed["solver"]["workers"] == "auto"
    assert changed["solver"]["parallel_mode"] == "auto"
    assert changed["solver"]["worker_numba_threads"] == 0
    assert changed["solver"]["max_nfev"] == 30
    assert changed["identifiability"]["enabled"] is False


def test_manual_point_runtime_overrides_preserve_lower_max_nfev() -> None:
    changed = geometry_fit.apply_manual_point_geometry_fit_runtime_overrides(
        {"solver": {"max_nfev": 10}},
        joint_background_mode=False,
    )

    assert changed["solver"]["max_nfev"] == 10


def test_dynamic_point_runtime_overrides_keep_richer_path() -> None:
    changed = geometry_fit.apply_dynamic_point_geometry_fit_runtime_overrides(
        {
            "solver": {
                "manual_point_fit_mode": True,
                "dynamic_point_geometry_fit": False,
                "workers": "auto",
                "parallel_mode": "auto",
                "worker_numba_threads": 0,
                "loss": "soft_l1",
            },
            "allow_unsafe_runtime": True,
        },
        joint_background_mode=False,
    )

    assert changed["solver"]["dynamic_point_geometry_fit"] is True
    assert "manual_point_fit_mode" not in changed["solver"]
    assert changed["solver"]["workers"] == "auto"
    assert changed["solver"]["parallel_mode"] == "auto"
    assert changed["solver"]["worker_numba_threads"] == 0
    assert changed["solver"]["loss"] == "soft_l1"


def test_new4_ladder_lean_runtime_config_disables_identifiability_by_default() -> None:
    ladder = _load_new4_ladder_module()

    cfg = ladder._lean_runtime_config(
        {
            "solver": {
                "workers": "auto",
                "parallel_mode": "auto",
                "max_nfev": 400,
            },
            "identifiability": {
                "enabled": True,
                "auto_freeze": True,
                "selective_thaw": {"enabled": True},
                "adaptive_regularization": {"enabled": True},
            },
        },
        active_names=["a", "c"],
        max_nfev=20,
    )

    assert cfg["solver"]["max_nfev"] == 20
    assert cfg["solver"]["parallel_mode"] == "off"
    assert cfg["identifiability"]["enabled"] is False
    assert "auto_freeze" not in cfg["identifiability"]
    assert "selective_thaw" not in cfg["identifiability"]
    assert "adaptive_regularization" not in cfg["identifiability"]


def test_new4_ladder_lean_runtime_config_enables_identifiability_feature_run() -> None:
    ladder = _load_new4_ladder_module()

    cfg = ladder._lean_runtime_config(
        {"identifiability": {"enabled": False}},
        active_names=["a"],
        max_nfev=20,
        feature="identifiability_features",
    )

    assert cfg["identifiability"]["enabled"] is True
    assert cfg["identifiability"]["auto_freeze"] is True
    assert cfg["identifiability"]["selective_thaw"] == {"enabled": True}
    assert cfg["identifiability"]["adaptive_regularization"] == {"enabled": True}


def test_new4_ladder_residual_heartbeat_throttle_policy() -> None:
    ladder = _load_new4_ladder_module()

    assert ladder._should_write_residual_heartbeat(
        heartbeat_count=1,
        eval_count=1,
        clean=True,
        now=1.0,
        last_write_s=0.0,
    )
    assert not ladder._should_write_residual_heartbeat(
        heartbeat_count=3,
        eval_count=3,
        clean=True,
        now=1.1,
        last_write_s=1.0,
    )
    assert ladder._should_write_residual_heartbeat(
        heartbeat_count=5,
        eval_count=5,
        clean=True,
        now=1.1,
        last_write_s=1.0,
    )
    assert ladder._should_write_residual_heartbeat(
        heartbeat_count=4,
        eval_count=4,
        clean=False,
        now=1.1,
        last_write_s=1.0,
    )
    assert ladder._should_write_residual_heartbeat(
        heartbeat_count=4,
        eval_count=4,
        clean=True,
        now=1.5,
        last_write_s=1.0,
    )


def test_new4_ladder_solver_rung_resets_stale_heartbeat_trace(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    output_path = tmp_path / "rung_03_one_param_a.json"
    heartbeat_path = output_path.with_suffix(".heartbeat.json")
    ladder._write_json(
        heartbeat_path,
        {
            "status": "old",
            "residual_eval_trace": [{"nfev": 1}],
            "last_residual_eval": {"nfev": 1},
        },
    )

    def _worker(**kwargs):
        assert kwargs["heartbeat_path"] == heartbeat_path
        ladder._heartbeat_write(
            heartbeat_path,
            {"status": "ok", "last_residual_eval": {"nfev": 2}},
        )
        return {"status": "ok", "pass": True}

    monkeypatch.setattr(ladder, "_worker_solve_once", _worker)

    report = ladder._run_solver_rung_with_timeout(
        state_path=state_path,
        background_index=0,
        active_names=["a"],
        output_path=output_path,
        max_nfev=1,
        timeout_seconds=1.0,
        rung=3,
        rung_name="one_param_a",
        use_subprocess=False,
    )

    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert heartbeat["status"] == "ok"
    assert heartbeat["last_residual_eval"] == {"nfev": 2}
    assert "residual_eval_trace" not in heartbeat
