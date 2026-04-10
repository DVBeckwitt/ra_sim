from __future__ import annotations

import numpy as np

from ra_sim.gui import geometry_fit
from ra_sim.gui import geometry_q_group_manager


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
        "optics_mode": 2,
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
