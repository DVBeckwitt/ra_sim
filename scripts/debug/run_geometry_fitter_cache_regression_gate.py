"""Run geometry fitter cache regression gates."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


COMPILE_TARGETS = (
    "ra_sim/fitting/optimization.py",
    "ra_sim/fitting/geometry_objective_cache.py",
    "ra_sim/gui/geometry_fit.py",
    "ra_sim/gui/manual_geometry.py",
    "ra_sim/gui/_runtime/runtime_session.py",
    "ra_sim/gui/runtime_invalidation.py",
    "ra_sim/gui/runtime_qr_selector_cache_policy.py",
    "ra_sim/gui/runtime_detector_remap_cache.py",
)

FAST_LOCAL_TESTS = (
    "tests/test_runtime_qr_selector_cache_policy.py",
    "tests/test_gui_runtime_invalidation.py",
    "tests/test_gui_runtime_update_actions.py",
    "tests/test_gui_runtime_optimization_scenarios.py",
    "tests/test_gui_runtime_update_dependencies.py",
    "tests/test_gui_runtime_primary_cache.py",
    "tests/test_gui_runtime_detector_remap_cache.py",
    "tests/test_gui_runtime_update_trace.py",
    "tests/test_fit_cache_controls.py",
    "tests/test_gui_runtime_import_safe.py",
    "tests/test_gui_runtime_geometry_fitter_handoff_fast.py",
    "tests/test_geometry_objective_cache.py",
    "tests/test_gui_runtime_geometry_fitter_cache_handoff.py",
    "tests/test_gui_runtime_mixed_update_regressions.py",
)

WORKFLOW_SLICE = "point_provider or saved_state_without_running_optimizer"

MANUAL_IDENTITY_SLICE = (
    "dynamic_identity_resolves_all_mode_a_branches "
    "or no_partial_qr_objective_allowed "
    "or objective_uses_refined_sim_caked_residual "
    "or fit_pipeline_no_stale_cache_under_trial_params"
)

SLOW_GEOMETRY_SLICE = (
    "caked_refinement_bin_resolution "
    "or observed_trial_caked_recomputed "
    "or sim_trial_caked_recomputed "
    "or refined_objective_theta_phi_decomposition "
    "or full_fit_with_dynamic_refined_center_objective"
)


@dataclass(frozen=True)
class GateCommand:
    name: str
    command: tuple[str, ...] = ()
    skip_reason: str | None = None

    @property
    def skipped(self) -> bool:
        return self.skip_reason is not None


class GateConfigurationError(RuntimeError):
    """Raised when the requested gate cannot be built."""


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def _python_command(*args: str) -> tuple[str, ...]:
    return (sys.executable, *args)


def _pytest_command(
    *args: str,
    pytest_extra_args: Sequence[str] = (),
) -> tuple[str, ...]:
    return _python_command("-m", "pytest", *args, *pytest_extra_args)


def _existing_compile_targets(repo_root: Path) -> tuple[str, ...]:
    return tuple(target for target in COMPILE_TARGETS if (repo_root / target).exists())


def build_gate_commands(
    mode: str = "local",
    *,
    repo_root: Path | str | None = None,
    skip_compile: bool = False,
    skip_workflow_slice: bool = False,
    pytest_extra_args: Sequence[str] = (),
) -> list[GateCommand]:
    """Build the ordered gate commands without running subprocesses."""

    mode_key = str(mode or "local").strip().lower()
    if mode_key not in {"local", "strict"}:
        raise GateConfigurationError(f"unsupported mode: {mode}")
    root = Path(repo_root) if repo_root is not None else repo_root_from_script()
    root = root.resolve()
    extra = tuple(str(arg) for arg in pytest_extra_args)
    commands: list[GateCommand] = []

    if skip_compile:
        commands.append(GateCommand("compile", skip_reason="disabled by --skip-compile"))
    else:
        compile_targets = _existing_compile_targets(root)
        commands.append(
            GateCommand(
                "compile",
                _python_command("-m", "py_compile", *compile_targets),
            )
        )

    commands.append(
        GateCommand(
            "fast local pytest gate",
            _pytest_command(*FAST_LOCAL_TESTS, "-q", pytest_extra_args=extra),
        )
    )

    commands.append(
        GateCommand(
            "manual identity gate",
            _pytest_command(
                "tests/test_manual_geometry_selection_helpers.py",
                "-k",
                MANUAL_IDENTITY_SLICE,
                "-s",
                "-q",
                pytest_extra_args=extra,
            ),
        )
    )

    if skip_workflow_slice:
        commands.append(
            GateCommand(
                "workflow slice",
                skip_reason="disabled by --skip-workflow-slice",
            )
        )
    else:
        commands.append(
            GateCommand(
                "workflow slice",
                _pytest_command(
                    "tests/test_gui_geometry_fit_workflow.py",
                    "-k",
                    WORKFLOW_SLICE,
                    "-vv",
                    pytest_extra_args=extra,
                ),
            )
        )

    if mode_key == "strict":
        commands.append(
            GateCommand(
                "slow_geometry gate",
                _pytest_command(
                    "tests/test_manual_geometry_selection_helpers.py",
                    "-k",
                    SLOW_GEOMETRY_SLICE,
                    "-s",
                    "-q",
                    pytest_extra_args=extra,
                ),
            )
        )
    else:
        commands.append(
            GateCommand("slow_geometry gate", skip_reason="skipped local mode")
        )

    commands.append(
        GateCommand(
            "bi geometry baseline",
            _python_command(
                "scripts/debug/run_geometry_fit_quality_baseline.py",
            ),
        )
    )
    return commands


def _split_pytest_extra_args(values: Iterable[str] | None) -> tuple[str, ...]:
    out: list[str] = []
    for value in values or ():
        out.extend(shlex.split(str(value), posix=os.name != "nt"))
    return tuple(out)


def _format_command(command: Sequence[str]) -> str:
    if hasattr(shlex, "join"):
        return shlex.join([str(part) for part in command])
    return " ".join(str(part) for part in command)


def run_gate(
    *,
    mode: str,
    skip_compile: bool,
    skip_workflow_slice: bool,
    pytest_extra_args: Sequence[str],
    repo_root: Path | None = None,
) -> int:
    root = (repo_root or repo_root_from_script()).resolve()
    try:
        commands = build_gate_commands(
            mode,
            repo_root=root,
            skip_compile=skip_compile,
            skip_workflow_slice=skip_workflow_slice,
            pytest_extra_args=pytest_extra_args,
        )
    except GateConfigurationError as exc:
        print(f"configuration: failed")
        print(f"reason: {exc}")
        print("overall: failed")
        return 2

    summary: list[tuple[str, str]] = []
    for gate in commands:
        if gate.skipped:
            print(f"{gate.name}: skipped ({gate.skip_reason})")
            summary.append((gate.name, f"skipped ({gate.skip_reason})"))
            continue
        print(f"{gate.name}: running")
        print(_format_command(gate.command))
        completed = subprocess.run(gate.command, cwd=root)
        if completed.returncode != 0:
            print(f"{gate.name}: failed")
            print(f"exit code: {completed.returncode}")
            print(f"command: {_format_command(gate.command)}")
            print("summary:")
            for name, status in summary:
                print(f"{name}: {status}")
            print(f"{gate.name}: failed")
            print("overall: failed")
            return int(completed.returncode)
        summary.append((gate.name, "passed"))
        print(f"{gate.name}: passed")

    print("summary:")
    for name, status in summary:
        print(f"{name}: {status}")
    print("overall: passed")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run geometry fitter cache regression gates.",
    )
    parser.add_argument("--mode", choices=("local", "strict"), default="local")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-workflow-slice", action="store_true")
    parser.add_argument(
        "--pytest-extra-args",
        action="append",
        default=[],
        help="Extra pytest args as a quoted string. May be repeated.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_gate(
        mode=args.mode,
        skip_compile=bool(args.skip_compile),
        skip_workflow_slice=bool(args.skip_workflow_slice),
        pytest_extra_args=_split_pytest_extra_args(args.pytest_extra_args),
    )


if __name__ == "__main__":
    raise SystemExit(main())
