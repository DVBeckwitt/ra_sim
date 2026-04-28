from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/debug/run_geometry_fitter_cache_regression_gate.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location(
        "run_geometry_fitter_cache_regression_gate",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _command_names(commands) -> list[str]:
    return [command.name for command in commands]


def test_gate_local_mode_builds_fast_commands_without_slow_geometry(tmp_path) -> None:
    gate = _load_gate_module()

    commands = gate.build_gate_commands(mode="local", repo_root=tmp_path)

    names = _command_names(commands)
    assert "compile" in names
    assert "fast local pytest gate" in names
    assert "manual identity gate" in names
    assert "workflow slice" in names
    slow_gate = next(command for command in commands if command.name == "slow_geometry gate")
    assert slow_gate.skipped
    assert slow_gate.skip_reason == "skipped local mode"


def test_gate_strict_mode_includes_slow_geometry(tmp_path) -> None:
    gate = _load_gate_module()

    commands = gate.build_gate_commands(mode="strict", repo_root=tmp_path)

    slow_gate = next(command for command in commands if command.name == "slow_geometry gate")
    assert not slow_gate.skipped
    assert "tests/test_manual_geometry_selection_helpers.py" in slow_gate.command
    assert gate.SLOW_GEOMETRY_SLICE in slow_gate.command


def test_gate_skips_new4_when_artifact_absent_by_default(tmp_path) -> None:
    gate = _load_gate_module()

    commands = gate.build_gate_commands(mode="strict", repo_root=tmp_path)

    preflight = next(command for command in commands if command.name == "new4 preflight")
    ladder = next(command for command in commands if command.name == "new4 ladder")
    assert preflight.skipped
    assert ladder.skipped
    assert "missing optional artifact" in preflight.skip_reason
    assert "missing optional artifact" in ladder.skip_reason


def test_gate_require_new4_fails_when_artifact_absent(tmp_path) -> None:
    gate = _load_gate_module()

    with pytest.raises(gate.GateConfigurationError, match="missing optional artifact"):
        gate.build_gate_commands(
            mode="strict",
            repo_root=tmp_path,
            require_new4=True,
        )


def test_gate_uses_sys_executable_for_python_commands(tmp_path) -> None:
    gate = _load_gate_module()
    new4_path = tmp_path / gate.NEW4_STATE_PATH
    new4_path.parent.mkdir(parents=True)
    new4_path.write_text("{}", encoding="utf-8")

    commands = gate.build_gate_commands(mode="strict", repo_root=tmp_path)

    runnable = [command for command in commands if not command.skipped]
    assert runnable
    assert all(command.command[0] == sys.executable for command in runnable)
