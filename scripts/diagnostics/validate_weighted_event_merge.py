"""Run the weighted-event merge diagnostics gate."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON = sys.executable


FOCUSED_TESTS = [
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_diffraction_weighted_events.py::test_solve_q_real_jit_does_not_crash_allocate_sched",
        "-q",
    ],
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_diffraction_weighted_events.py::test_compute_intensity_array_is_serial_njit",
        "-q",
    ],
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_diffraction_weighted_events.py::test_representative_choice_uses_true_mosaic_weight_before_mass",
        "-q",
    ],
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_diffraction_weighted_events.py::test_representative_choice_preserves_mosaic_top_sample_index_in_hit_row",
        "-q",
    ],
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_diffraction_weighted_events.py::test_mosaic_top_representative_survives_even_when_unsampled",
        "-q",
    ],
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_diffraction_weighted_events.py::test_manual_worker_count_one_routes_serial",
        "-q",
    ],
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_diffraction_weighted_events.py::test_manual_worker_count_two_routes_threaded_chunks",
        "-q",
    ],
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_diffraction_weighted_events.py::test_manual_worker_count_four_reports_four_workers_when_enough_samples",
        "-q",
    ],
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_diffraction_weighted_events.py::test_weighted_event_worker_count_config_override",
        "-q",
    ],
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_diffraction_weighted_events.py::test_weighted_events_dispatcher_path_matrix",
        "-q",
    ],
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_diffraction_weighted_events.py::test_weighted_events_parallel_from_bound_matches_serial_controlled_backend",
        "-q",
    ],
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_diffraction_weighted_events.py::test_weighted_events_parallel_from_bound_matches_serial_real_solve_q_small",
        "-q",
    ],
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_diffraction_weighted_events.py::test_weighted_events_original_plan_compliance_matrix",
        "-q",
        "-s",
    ],
]


FOCUSED_SUITES = [
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_diffraction_weighted_events.py",
        "tests/test_source_template_cache.py",
        "tests/test_intersection_cache_schema.py",
        "-q",
    ],
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_gui_geometry_q_group_manager.py::test_qr_selection_uses_weighted_event_mosaic_top_representative",
        "-q",
    ],
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_gui_geometry_q_group_manager.py::test_qr_selection_does_not_use_weighted_sampled_event_when_representative_exists",
        "-q",
    ],
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_gui_geometry_q_group_manager.py::test_qr_selection_preserves_clicked_branch_then_mosaic_top_candidate",
        "-q",
    ],
    [
        PYTHON,
        "-m",
        "pytest",
        "tests/test_gui_peak_selection.py::test_select_peak_by_hkl_prefers_mosaic_top_candidate_over_brighter_duplicate",
        "-q",
    ],
]


STATIC_TARGETS = [
    "ra_sim/simulation/diffraction.py",
    "ra_sim/simulation/intersection_cache_schema.py",
    "ra_sim/utils/parallel.py",
    "ra_sim/cli.py",
    "ra_sim/headless_geometry_fit.py",
    "ra_sim/gui/mosaic_top_selection.py",
    "ra_sim/gui/geometry_q_group_manager.py",
    "tests/test_diffraction_weighted_events.py",
    "tests/test_diffraction_safe_wrapper.py",
    "tests/test_source_template_cache.py",
    "tests/test_intersection_cache_schema.py",
    "tests/test_gui_geometry_q_group_manager.py",
    "tests/test_gui_peak_selection.py",
    "scripts/benchmarks/benchmark_weighted_events_parallel.py",
]


STATIC_CHECKS = [
    [PYTHON, "-m", "ruff", "check", *STATIC_TARGETS],
    [PYTHON, "-m", "py_compile", *STATIC_TARGETS],
]


BENCHMARK = [
    PYTHON,
    "scripts/benchmarks/benchmark_weighted_events_parallel.py",
    "--runs",
    "1",
    "--threads",
    "1",
    "2",
    "4",
    "--n-samp",
    "512",
    "--events",
    "2",
]


FULL_PYTEST = [PYTHON, "-m", "pytest", "tests", "-q", "--durations=20"]


def _display_command(command: list[str]) -> str:
    return " ".join("python" if part == PYTHON else part for part in command)


def _checkout_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
    )
    return env


def _check_checkout_import() -> bool:
    sys.path.insert(0, str(REPO_ROOT))
    import ra_sim

    ra_sim_path = pathlib.Path(ra_sim.__file__).resolve()
    print(f"ra_sim import: {ra_sim_path}")
    if REPO_ROOT not in ra_sim_path.parents:
        print(f"ERROR: imported stale ra_sim outside checkout: {ra_sim_path}", file=sys.stderr)
        return False
    return True


def _run_streamed(
    command: list[str],
    *,
    label: str,
    keep_going: bool,
    require: tuple[str, ...] = (),
    forbid: tuple[str, ...] = ("LLVM ERROR", "allocate_sched"),
    reject_weighted_python: bool = False,
) -> bool:
    print(f"\n== {label} ==")
    print(f"$ {_display_command(command)}")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=_checkout_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_parts: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_parts.append(line)
    return_code = process.wait()
    output = "".join(output_parts)

    checks_ok = return_code == 0
    if return_code != 0:
        print(f"ERROR: {label} exited with {return_code}", file=sys.stderr)

    forbidden = list(forbid)
    if reject_weighted_python:
        forbidden.append("weighted_events_python")
    for marker in forbidden:
        if marker in output:
            print(f"ERROR: {label} emitted forbidden marker: {marker}", file=sys.stderr)
            checks_ok = False
    for marker in require:
        if marker not in output:
            print(f"ERROR: {label} missing required marker: {marker}", file=sys.stderr)
            checks_ok = False

    if not checks_ok and not keep_going:
        raise SystemExit(1)
    return checks_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-full-pytest", action="store_true")
    parser.add_argument("--full-pytest", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    ok = _check_checkout_import()
    if not ok and not args.keep_going:
        return 1

    for command in FOCUSED_TESTS:
        label = command[3] if len(command) > 3 and command[2] == "pytest" else command[-2]
        command_text = " ".join(command)
        require: tuple[str, ...] = ()
        forbid = ("LLVM ERROR", "allocate_sched")
        reject_weighted_python = "test_weighted_events_dispatcher_path_matrix" in command_text
        if "test_weighted_events_original_plan_compliance_matrix" in command_text:
            require = ("original_plan_validation_incomplete=no",)
            forbid = ("LLVM ERROR", "original_plan_validation_incomplete=yes", "untested")
            reject_weighted_python = False
        ok = (
            _run_streamed(
                command,
                label=label,
                keep_going=args.keep_going,
                require=require,
                forbid=forbid,
                reject_weighted_python=reject_weighted_python,
            )
            and ok
        )

    for command in FOCUSED_SUITES:
        ok = (
            _run_streamed(
                command,
                label="focused suite",
                keep_going=args.keep_going,
                forbid=("LLVM ERROR",),
            )
            and ok
        )

    for command in STATIC_CHECKS:
        ok = (
            _run_streamed(command, label=command[2], keep_going=args.keep_going)
            and ok
        )

    ok = (
        _run_streamed(
            BENCHMARK,
            label="benchmark smoke",
            keep_going=args.keep_going,
            require=(
                "threads_1_parallel_backend: fast_serial",
                "threads_2_parallel_backend: threaded_njit_chunks",
                "threads_2_parallel_worker_count: 2",
                "threads_4_parallel_backend: threaded_njit_chunks",
                "threads_4_parallel_worker_count: 4",
                "threads_4_parallel_worker_count_source: explicit",
            ),
            reject_weighted_python=True,
        )
        and ok
    )

    if args.full_pytest and not args.skip_full_pytest:
        ok = (
            _run_streamed(
                FULL_PYTEST,
                label="full pytest",
                keep_going=args.keep_going,
                forbid=("LLVM ERROR",),
            )
            and ok
        )

    if ok:
        print("\nweighted-event merge diagnostics passed")
        return 0
    print("\nweighted-event merge diagnostics failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
