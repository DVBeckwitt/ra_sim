from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from ra_sim.gui import geometry_fit
from ra_sim.gui import geometry_q_group_manager
from ra_sim.simulation import diffraction


RUNTIME_SESSION_PATH = Path("ra_sim/gui/_runtime/runtime_session.py")
GEOMETRY_FIT_JOB_PATH = Path("ra_sim/gui/_runtime/geometry_fit_job.py")
GEOMETRY_FIT_WORKER_PATH = Path("ra_sim/gui/_runtime/geometry_fit_worker.py")


def _imported_modules(module_path: Path) -> set[str]:
    imported_modules: set[str] = set()
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(str(alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(str(node.module))
            imported_modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return imported_modules


def test_imported_modules_records_import_from_aliases(tmp_path: Path) -> None:
    module_path = tmp_path / "worker_import_probe.py"
    module_path.write_text(
        "\n".join(
            (
                "from ra_sim.gui import geometry_fit",
                "from ra_sim.gui import manual_geometry",
                "from ra_sim.gui._runtime import runtime_session",
                "from ra_sim.fitting import optimization",
            )
        ),
        encoding="utf-8",
    )

    imported_modules = _imported_modules(module_path)

    assert "ra_sim.gui" in imported_modules
    assert "ra_sim.gui.geometry_fit" in imported_modules
    assert "ra_sim.gui.manual_geometry" in imported_modules
    assert "ra_sim.gui._runtime" in imported_modules
    assert "ra_sim.gui._runtime.runtime_session" in imported_modules
    assert "ra_sim.fitting" in imported_modules
    assert "ra_sim.fitting.optimization" in imported_modules


def test_geometry_fit_job_helper_has_no_live_runtime_or_tk_imports() -> None:
    imported_modules = _imported_modules(GEOMETRY_FIT_JOB_PATH)

    forbidden_imports = {
        "ra_sim.gui._runtime.runtime_session",
        "tkinter",
        "ra_sim.gui.views",
        "ra_sim.gui.runtime_geometry_fit",
        "ra_sim.gui._runtime.geometry_fit_worker",
    }
    assert not (imported_modules & forbidden_imports)
    assert not any(name.startswith("tkinter.") for name in imported_modules)


def test_geometry_fit_worker_helper_has_no_live_runtime_tk_or_solver_imports() -> None:
    imported_modules = _imported_modules(GEOMETRY_FIT_WORKER_PATH)

    forbidden_imports = {
        "matplotlib",
        "ra_sim.fitting.optimization",
        "ra_sim.gui._runtime.geometry_fit_job",
        "ra_sim.gui._runtime.runtime_session",
        "ra_sim.gui.geometry_fit",
        "ra_sim.gui.manual_geometry",
        "ra_sim.gui.runtime_geometry_fit",
        "ra_sim.gui.views",
        "tkinter",
    }
    assert not (imported_modules & forbidden_imports)
    assert not any(name.startswith("tkinter.") for name in imported_modules)
    assert not any(name.startswith("matplotlib.") for name in imported_modules)


def test_geometry_fit_worker_context_helpers_are_not_duplicated_in_runtime_worker() -> None:
    source = RUNTIME_SESSION_PATH.read_text(encoding="utf-8")
    worker_start = source.index("def _run_async_geometry_fit_worker_job(")
    worker_end = source.index("def _consume_ready_geometry_fit_result(", worker_start)
    tree = ast.parse(source[worker_start:worker_end])
    nested_function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name != "_run_async_geometry_fit_worker_job"
    }

    moved_helper_names = {
        "_advance_source_cache_generation",
        "_caked_projection_payload_status",
        "_current_source_cache_generation",
        "_emit_worker_event",
        "_ensure_worker_caked_projection_payload",
        "_last_worker_simulation_diagnostics",
        "_last_worker_source_snapshot_diagnostics",
        "_load_background_by_index_snapshot",
        "_load_caked_projection_by_index_snapshot",
        "_load_caked_view_by_index_snapshot",
        "_mark_worker_cached_projection_rows",
        "_projection_candidate_state",
        "_project_source_rows_by_row_background",
        "_project_source_rows_for_background",
        "_set_worker_source_snapshot_diagnostics",
        "_source_cache_generation_matches",
        "_store_worker_background_cache_bundle",
        "_worker_cached_projection_rows_match",
        "_bundle_rows",
        "_build_geometry_fit_background_cache_bundle",
    }
    assert not (nested_function_names & moved_helper_names)


def test_geometry_fit_worker_has_moved_only_d3_source_projection_and_bundle_helpers() -> None:
    tree = ast.parse(GEOMETRY_FIT_WORKER_PATH.read_text(encoding="utf-8"))
    worker_function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }

    assert {
        "bundle_rows",
        "build_geometry_fit_background_cache_bundle",
        "mark_worker_cached_projection_rows",
        "project_source_rows_by_row_background",
        "project_source_rows_for_background",
        "store_worker_background_cache_bundle",
        "worker_cached_projection_rows_match",
    } <= worker_function_names

    pending_d3_helper_names = {
        "_prebuild_background_cache_bundle_worker",
        "_prebuild_required_background_caches",
        "prebuild_background_cache_bundle_worker",
        "prebuild_required_background_caches",
    }
    assert not (worker_function_names & pending_d3_helper_names)


def _geometry_fit_param_set() -> dict[str, object]:
    return {
        "a": 3.0,
        "c": 5.0,
        "lambda": 1.54,
        "corto_detector": 100.0,
        "gamma": 1.0,
        "Gamma": 2.0,
        "chi": 3.0,
        "psi": 4.0,
        "psi_z": 5.0,
        "zs": 6.0,
        "zb": 7.0,
        "n2": "n2",
        "debye_x": 0.1,
        "debye_y": 0.2,
        "center": (11.0, 12.0),
        "theta_initial": 8.0,
        "cor_angle": 9.0,
        "optics_mode": diffraction.OPTICS_MODE_EXACT,
        "mosaic_params": {
            "beam_x_array": np.asarray([1.0], dtype=float),
            "beam_y_array": np.asarray([2.0], dtype=float),
            "theta_array": np.asarray([3.0], dtype=float),
            "phi_array": np.asarray([4.0], dtype=float),
            "wavelength_array": np.asarray([1.54], dtype=float),
            "sigma_mosaic_deg": 0.1,
            "gamma_mosaic_deg": 0.2,
            "eta": 0.3,
        },
    }


def test_geometry_fit_simulation_callbacks_can_force_python_runner() -> None:
    prefer_python_runner_seen: list[object] = []

    def fake_process_peaks_parallel(*_args, **kwargs):
        prefer_python_runner_seen.append(kwargs.get("prefer_python_runner"))
        return (
            np.zeros((32, 32), dtype=float),
            [np.asarray([[10.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0]], dtype=float)],
        )

    bundle = geometry_q_group_manager.make_runtime_geometry_fit_simulation_callbacks(
        process_peaks_parallel=fake_process_peaks_parallel,
        hit_tables_to_max_positions=lambda _tables: [[9.0, 1.0, 2.0, 4.0, 6.0, 7.0]],
        native_sim_to_display_coords=lambda col, row, _shape: (col, row),
        default_solve_q_steps=123,
        default_solve_q_rel_tol=2.5e-4,
        default_solve_q_mode=1,
        prefer_safe_python_runner=True,
    )

    miller_array = np.asarray([[1.0, 0.0, 0.0]], dtype=float)
    intensity_array = np.asarray([5.0], dtype=float)
    param_set = _geometry_fit_param_set()

    hit_tables = bundle.simulate_hit_tables(miller_array, intensity_array, 32, param_set)
    peak_centers = bundle.simulate_peak_centers(miller_array, intensity_array, 32, param_set)

    assert len(hit_tables) == 1
    assert peak_centers == [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "sim_col": 1.0,
            "sim_row": 2.0,
            "weight": 5.0,
        }
    ]
    assert prefer_python_runner_seen == [True, True]


def test_geometry_fit_simulation_callbacks_fall_back_to_safe_runner_for_keyword_mismatch(
    monkeypatch,
) -> None:
    safe_runner_calls: list[object] = []

    def raw_process_peaks_parallel(*_args, **_kwargs):
        raise TypeError("some keyword arguments unexpected")

    def safe_process_peaks_parallel(*_args, **kwargs):
        safe_runner_calls.append(kwargs.get("prefer_python_runner"))
        return (
            np.zeros((32, 32), dtype=float),
            [np.asarray([[10.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0]], dtype=float)],
        )

    monkeypatch.setattr(
        geometry_q_group_manager,
        "diffraction_process_peaks_parallel",
        raw_process_peaks_parallel,
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "diffraction_process_peaks_parallel_safe",
        safe_process_peaks_parallel,
    )

    bundle = geometry_q_group_manager.make_runtime_geometry_fit_simulation_callbacks(
        process_peaks_parallel=raw_process_peaks_parallel,
        hit_tables_to_max_positions=lambda _tables: [[9.0, 1.0, 2.0, 4.0, 6.0, 7.0]],
        native_sim_to_display_coords=lambda col, row, _shape: (col, row),
        default_solve_q_steps=123,
        default_solve_q_rel_tol=2.5e-4,
        default_solve_q_mode=1,
        prefer_safe_python_runner=True,
    )

    hit_tables = bundle.simulate_hit_tables(
        np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        np.asarray([5.0], dtype=float),
        32,
        _geometry_fit_param_set(),
    )

    assert len(hit_tables) == 1
    assert safe_runner_calls == [True]


def test_manual_geometry_fit_preflight_allows_small_high_coverage_source_miss() -> None:
    error_text = geometry_fit._manual_geometry_fit_preflight_error(
        [
            {
                "label": "bg0.osc",
                "pair_count": 21,
                "resolved_source_pair_count": 20,
                "orientation_diag": {"pairs": 21},
            },
            {
                "label": "bg1.osc",
                "pair_count": 21,
                "resolved_source_pair_count": 20,
                "orientation_diag": {"pairs": 21},
            },
        ]
    )

    assert error_text is None


def test_manual_geometry_fit_preflight_rejects_low_coverage_source_resolution() -> None:
    error_text = geometry_fit._manual_geometry_fit_preflight_error(
        [
            {
                "label": "bg0.osc",
                "pair_count": 2,
                "resolved_source_pair_count": 1,
                "orientation_diag": {"pairs": 2},
            }
        ]
    )

    assert error_text == (
        "Geometry fit unavailable: some saved manual pairs no longer resolve "
        "to current simulated source rows: bg0.osc (1/2). Refresh the picks "
        "before fitting."
    )
