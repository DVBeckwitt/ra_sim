from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from scripts import geometry_fit_landscape


def test_resolve_candidate_parameter_names_excludes_lattice_params_when_disabled() -> None:
    assert geometry_fit_landscape.resolve_candidate_parameter_names(
        use_shared_theta_offset=False,
        lattice_refinement_enabled=False,
    ) == [
        "zb",
        "zs",
        "theta_initial",
        "psi_z",
        "chi",
        "cor_angle",
        "gamma",
        "Gamma",
        "corto_detector",
        "center_x",
        "center_y",
    ]


def test_resolve_candidate_parameter_names_switches_to_theta_offset_when_shared() -> None:
    assert geometry_fit_landscape.resolve_candidate_parameter_names(
        use_shared_theta_offset=True,
        lattice_refinement_enabled=False,
    ) == [
        "zb",
        "zs",
        "theta_offset",
        "psi_z",
        "chi",
        "cor_angle",
        "gamma",
        "Gamma",
        "corto_detector",
        "center_x",
        "center_y",
    ]


def test_build_argument_parser_exposes_workers_flag() -> None:
    parser = geometry_fit_landscape.build_argument_parser()
    args = parser.parse_args(["--workers", "4"])

    assert args.workers == 4


def test_default_worker_count_uses_ninety_percent_of_available_cores(monkeypatch) -> None:
    monkeypatch.setattr(geometry_fit_landscape.os, "cpu_count", lambda: 32)

    assert geometry_fit_landscape._default_worker_count() == 28


def test_build_argument_parser_defaults_workers_from_available_cores(monkeypatch) -> None:
    monkeypatch.setattr(geometry_fit_landscape.os, "cpu_count", lambda: 10)

    parser = geometry_fit_landscape.build_argument_parser()
    args = parser.parse_args([])

    assert args.workers == 9


def test_resolve_geometry_sweep_range_supports_bounds_modes() -> None:
    bounds_cfg = {
        "gamma": {"mode": "relative", "min": -2.0, "max": 2.0},
        "zb": {"mode": "absolute", "min": -0.002, "max": 0.002},
        "corto_detector": {"mode": "relative_min0", "min": -0.2, "max": 0.1},
    }

    assert geometry_fit_landscape.resolve_geometry_sweep_range(
        "gamma",
        1.5,
        bounds_config=bounds_cfg,
        priors_config={},
    ) == (-0.5, 3.5, "bounds:relative")
    assert geometry_fit_landscape.resolve_geometry_sweep_range(
        "zb",
        0.001,
        bounds_config=bounds_cfg,
        priors_config={},
    ) == (-0.002, 0.002, "bounds:absolute")
    assert geometry_fit_landscape.resolve_geometry_sweep_range(
        "corto_detector",
        0.05,
        bounds_config=bounds_cfg,
        priors_config={},
    ) == (0.0, 0.15000000000000002, "bounds:relative_min0")


def test_resolve_geometry_sweep_range_uses_center_priors_and_fallback() -> None:
    assert geometry_fit_landscape.resolve_geometry_sweep_range(
        "center_x",
        100.0,
        bounds_config={},
        priors_config={"center_x": {"sigma": 20.0}},
    ) == (40.0, 160.0, "center_prior_pm3sigma")
    assert geometry_fit_landscape.resolve_geometry_sweep_range(
        "center_y",
        250.0,
        bounds_config={},
        priors_config={},
    ) == (190.0, 310.0, "center_fallback_pm60")


def test_update_param_set_for_theta_offset_updates_effective_theta() -> None:
    params = {
        "theta_initial": 6.25,
        "theta_offset": 0.25,
        "center": [100.0, 200.0],
        "center_x": 100.0,
        "center_y": 200.0,
    }

    updated = geometry_fit_landscape._update_param_set_for_sweep(
        params,
        param_name="theta_offset",
        value=0.5,
        theta_base_current=6.0,
    )

    assert updated["theta_offset"] == 0.5
    assert updated["theta_initial"] == 6.5


def test_build_sweep_tasks_flattens_specs_in_stable_order() -> None:
    sweep_specs = [
        geometry_fit_landscape.SweepSpec(
            name="gamma",
            baseline=1.0,
            min_value=0.0,
            max_value=2.0,
            values=np.array([0.0, 1.0], dtype=float),
            source="bounds:relative",
        ),
        geometry_fit_landscape.SweepSpec(
            name="Gamma",
            baseline=3.0,
            min_value=2.0,
            max_value=4.0,
            values=np.array([2.0], dtype=float),
            source="bounds:relative",
        ),
    ]

    tasks = geometry_fit_landscape._build_sweep_tasks(sweep_specs)

    assert [(task.parameter, task.sweep_index, task.parameter_value) for task in tasks] == [
        ("gamma", 0, 0.0),
        ("gamma", 1, 1.0),
        ("Gamma", 0, 2.0),
    ]


def test_resolve_worker_count_clamps_to_task_count() -> None:
    assert geometry_fit_landscape._resolve_worker_count(8, 3) == 3
    assert geometry_fit_landscape._resolve_worker_count(2, 10) == 2
    assert geometry_fit_landscape._resolve_worker_count(4, 0) == 1


def test_resolve_worker_count_uses_default_when_unspecified(monkeypatch) -> None:
    monkeypatch.setattr(geometry_fit_landscape.os, "cpu_count", lambda: 20)

    assert geometry_fit_landscape._resolve_worker_count(None, 100) == 18


def test_compute_peak_metrics_returns_finite_defaults_for_empty_input() -> None:
    metrics = geometry_fit_landscape.compute_peak_metrics([])

    assert metrics == {
        "visible_peak_count": 0.0,
        "total_peak_weight": 0.0,
        "centroid_x_px": 0.0,
        "centroid_y_px": 0.0,
        "radius_gyration_px": 0.0,
        "x_span_px": 0.0,
        "y_span_px": 0.0,
        "anisotropy_ratio": 1.0,
    }


def test_compute_peak_metrics_matches_weighted_geometry() -> None:
    peaks = [
        {"sim_col": 0.0, "sim_row": 0.0, "weight": 1.0},
        {"sim_col": 4.0, "sim_row": 0.0, "weight": 3.0},
        {"sim_col": 4.0, "sim_row": 2.0, "weight": 2.0},
    ]

    metrics = geometry_fit_landscape.compute_peak_metrics(peaks)

    assert metrics["visible_peak_count"] == 3.0
    assert metrics["total_peak_weight"] == 6.0
    assert math.isclose(metrics["centroid_x_px"], 20.0 / 6.0)
    assert math.isclose(metrics["centroid_y_px"], 4.0 / 6.0)
    assert math.isclose(metrics["x_span_px"], 4.0)
    assert math.isclose(metrics["y_span_px"], 2.0)
    assert metrics["radius_gyration_px"] > 0.0
    assert metrics["anisotropy_ratio"] >= 1.0


def test_build_correlation_matrix_returns_finite_values_for_constant_metrics() -> None:
    sweep_specs = [
        geometry_fit_landscape.SweepSpec(
            name="gamma",
            baseline=0.0,
            min_value=-1.0,
            max_value=1.0,
            values=np.array([-1.0, 0.0, 1.0], dtype=float),
            source="bounds:relative",
        )
    ]
    rows = [
        {
            "parameter": "gamma",
            "parameter_value": value,
            "visible_peak_count": 5.0,
            "total_peak_weight": 10.0,
            "centroid_x_px": 1.0,
            "centroid_y_px": 2.0,
            "radius_gyration_px": 3.0,
            "x_span_px": 4.0,
            "y_span_px": 5.0,
            "anisotropy_ratio": 1.0,
            "runtime_s": 0.01,
        }
        for value in (-1.0, 0.0, 1.0)
    ]

    matrix = geometry_fit_landscape.build_correlation_matrix(
        rows,
        sweep_specs,
        metric_names=geometry_fit_landscape.SUMMARY_METRIC_NAMES,
    )

    assert matrix.shape == (1, len(geometry_fit_landscape.SUMMARY_METRIC_NAMES))
    assert np.all(np.isfinite(matrix))


def test_render_landscape_figure_writes_with_constant_panel_data(tmp_path: Path) -> None:
    sweep_specs = [
        geometry_fit_landscape.SweepSpec(
            name="gamma",
            baseline=0.0,
            min_value=-1.0,
            max_value=1.0,
            values=np.array([-1.0, 0.0, 1.0], dtype=float),
            source="bounds:relative",
        ),
        geometry_fit_landscape.SweepSpec(
            name="Gamma",
            baseline=0.0,
            min_value=-1.0,
            max_value=1.0,
            values=np.array([-1.0, 0.0, 1.0], dtype=float),
            source="bounds:relative",
        ),
    ]
    rows = []
    for parameter in ("gamma", "Gamma"):
        for value in (-1.0, 0.0, 1.0):
            rows.append(
                {
                    "parameter": parameter,
                    "parameter_value": value,
                    "visible_peak_count": 5.0,
                    "total_peak_weight": 10.0,
                    "centroid_x_px": 1.0,
                    "centroid_y_px": 2.0,
                    "radius_gyration_px": 3.0,
                    "x_span_px": 4.0,
                    "y_span_px": 5.0,
                    "anisotropy_ratio": 1.0,
                    "runtime_s": 0.01,
                    "centroid_shift_px": 0.0,
                }
            )

    output_path = tmp_path / "landscape.png"
    geometry_fit_landscape.render_landscape_figure(
        rows,
        sweep_specs,
        state_path=tmp_path / "state.json",
        output_path=output_path,
        baseline_metrics={
            "visible_peak_count": 5.0,
            "total_peak_weight": 10.0,
            "centroid_x_px": 1.0,
            "centroid_y_px": 2.0,
            "radius_gyration_px": 3.0,
            "x_span_px": 4.0,
            "y_span_px": 5.0,
            "anisotropy_ratio": 1.0,
        },
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
