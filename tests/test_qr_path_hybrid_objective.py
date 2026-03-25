import ast
from pathlib import Path

import numpy as np

GUI_APP_PATH = Path("ra_sim/gui/app.py")


def _load_main_functions(*names: str) -> dict[str, object]:
    source = GUI_APP_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(GUI_APP_PATH))
    extracted: list[str] = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            fn_source = ast.get_source_segment(source, node)
            if fn_source:
                extracted.append(fn_source)
    if len(extracted) != len(names):
        missing = sorted(set(names) - {node.name for node in module.body if isinstance(node, ast.FunctionDef)})
        raise AssertionError(
            f"Failed to extract functions from {GUI_APP_PATH}: {missing}"
        )

    namespace: dict[str, object] = {}
    exec(
        "from __future__ import annotations\n"
        "import math\n"
        "import numpy as np\n"
        "from typing import Sequence\n\n"
        + "\n\n".join(extracted),
        namespace,
    )
    return namespace


def test_qr_path_objective_combines_background_and_simulation_residuals() -> None:
    namespace = _load_main_functions(
        "_rms_distance_px",
        "_background_group_display_points",
        "_build_qr_path_geometry_objective",
    )

    namespace["_intersection_geometry_from_param_set"] = lambda params: dict(params)
    namespace["_point_distances_to_qr_path"] = lambda points, path: [0.0 for _ in points]
    namespace["_project_display_qr_paths"] = (
        lambda qr_value, geometry, **kwargs: [{"branch_sign": 1, "qr": float(qr_value)}]
    )
    namespace["_best_qr_path_match"] = (
        lambda reference_path, candidate_paths, **kwargs: (candidate_paths[0], 4.0, [2.0, 2.0])
    )

    objective = namespace["_build_qr_path_geometry_objective"](
        seed_matches=[
            {
                "group_id": 1,
                "group": {"display_points": [(0.0, 0.0), (1.0, 1.0)]},
                "background_path": {
                    "cols": np.asarray([0.0, 1.0], dtype=float),
                    "rows": np.asarray([0.0, 1.0], dtype=float),
                },
                "branch_sign": 1,
                "sim_source": "primary",
                "sim_m": 3,
                "sim_qr": 0.45,
            }
        ],
        wavelength=1.0,
        n2_value=1.0,
        native_shape=(16, 16),
        cfg={
            "group_normalization": "sqrt_npts",
            "bg_point_weight": 2.0,
            "simulation_trace_weight": 3.0,
            "simulation_trace_samples": 8,
            "missing_path_penalty_px": 25.0,
            "missing_simulation_penalty_px": 30.0,
        },
    )

    residuals, diagnostics, summary = objective({"gamma": 0.0}, True)

    expected_bg = np.zeros(2, dtype=float)
    expected_sim = np.full(2, 3.0 * np.sqrt(2.0), dtype=float)
    assert np.allclose(np.asarray(residuals[:2], dtype=float), expected_bg)
    assert np.allclose(np.asarray(residuals[2:], dtype=float), expected_sim)
    assert diagnostics[0]["status"] == "matched"
    assert diagnostics[0]["bg_rms_px"] == 0.0
    assert diagnostics[0]["sim_rms_px"] == 2.0
    assert summary["matched_group_count"] == 1
    assert summary["matched_bg_group_count"] == 1
    assert summary["matched_simulation_group_count"] == 1
    assert summary["missing_group_count"] == 0
    assert summary["missing_simulation_group_count"] == 0
    assert summary["mean_bg_group_rms_px"] == 0.0
    assert summary["mean_simulation_group_rms_px"] == 2.0


def test_qr_path_objective_penalizes_missing_simulation_trace() -> None:
    namespace = _load_main_functions(
        "_rms_distance_px",
        "_background_group_display_points",
        "_build_qr_path_geometry_objective",
    )

    namespace["_intersection_geometry_from_param_set"] = lambda params: dict(params)
    namespace["_point_distances_to_qr_path"] = lambda points, path: [0.0 for _ in points]
    namespace["_project_display_qr_paths"] = (
        lambda qr_value, geometry, **kwargs: [{"branch_sign": 1, "qr": float(qr_value)}]
    )
    namespace["_best_qr_path_match"] = (
        lambda reference_path, candidate_paths, **kwargs: (None, float("inf"), [])
    )

    objective = namespace["_build_qr_path_geometry_objective"](
        seed_matches=[
            {
                "group_id": 1,
                "group": {"display_points": [(0.0, 0.0), (1.0, 1.0)]},
                "background_path": {
                    "cols": np.asarray([0.0, 1.0], dtype=float),
                    "rows": np.asarray([0.0, 1.0], dtype=float),
                },
                "branch_sign": 1,
                "sim_source": "primary",
                "sim_m": 3,
                "sim_qr": 0.45,
            }
        ],
        wavelength=1.0,
        n2_value=1.0,
        native_shape=(16, 16),
        cfg={
            "group_normalization": "sqrt_npts",
            "bg_point_weight": 1.0,
            "simulation_trace_weight": 2.0,
            "simulation_trace_samples": 4,
            "missing_path_penalty_px": 25.0,
            "missing_simulation_penalty_px": 30.0,
        },
    )

    residuals, diagnostics, summary = objective({"gamma": 0.0}, True)

    assert len(residuals) == 6
    assert diagnostics[0]["status"] == "missing_simulation_trace"
    assert diagnostics[0]["bg_rms_px"] == 0.0
    assert np.isnan(float(diagnostics[0]["sim_rms_px"]))
    assert summary["matched_group_count"] == 0
    assert summary["matched_bg_group_count"] == 1
    assert summary["matched_simulation_group_count"] == 0
    assert summary["missing_group_count"] == 1
    assert summary["missing_simulation_group_count"] == 1


def test_qr_path_seed_matches_use_fixed_background_guides() -> None:
    namespace = _load_main_functions(
        "_rms_distance_px",
        "_build_geometry_fit_qr_path_seed_matches",
    )

    namespace["_background_group_display_points"] = (
        lambda group: [(float(col), float(row)) for col, row in group.get("display_points", ())]
    )
    namespace["_project_qr_cylinder_to_direction_paths"] = (
        lambda **kwargs: [
            {
                "branch_sign": -1 if abs(float(kwargs["qr_value"]) - 0.55) < 1e-9 else 1,
                "dirs": np.asarray(
                    [
                        [1.0, 0.0, 0.0],
                        [0.8, 0.2, 0.0],
                        [0.6, 0.4, 0.0],
                        [0.4, 0.6, 0.0],
                    ],
                    dtype=float,
                ),
                "valid_mask": np.asarray([True, True, True, True]),
            }
        ]
    )
    namespace["_fit_background_group_to_direction_path"] = (
        lambda group, direction_path: {
            "path": {
                "cols": np.asarray([0.0, 1.0], dtype=float),
                "rows": np.asarray([0.0, 1.0], dtype=float),
                "branch_sign": int(direction_path["branch_sign"]),
            },
            "branch_sign": int(direction_path["branch_sign"]),
            "rms_px": 0.0 if int(direction_path["branch_sign"]) == -1 else 5.0,
            "point_rms_px": 0.0,
            "shape_rms_px": 0.0 if int(direction_path["branch_sign"]) == -1 else 5.0,
        }
    )
    namespace["_project_display_qr_paths"] = (
        lambda qr_value, geometry, **kwargs: [
            {
                "branch_sign": -1 if float(qr_value) < 0.6 else 1,
                "qr": float(qr_value),
            }
        ]
    )
    namespace["_best_qr_path_match"] = (
        lambda reference_path, candidate_paths, **kwargs: (
            candidate_paths[0],
            1.0 if abs(float(candidate_paths[0]["qr"]) - 0.55) < 1e-9 else 9.0,
            [1.0, 1.0] if abs(float(candidate_paths[0]["qr"]) - 0.55) < 1e-9 else [3.0, 3.0],
        )
    )
    namespace["_normalize_bragg_qr_source_label"] = lambda label: str(label)

    seed_matches, unique_count = namespace["_build_geometry_fit_qr_path_seed_matches"](
        groups=[
            {
                "id": 7,
                "display_points": [(0.0, 0.0), (1.0, 0.5), (2.0, 1.0), (3.0, 1.5)],
            }
        ],
        qr_entries=[
            {"source": "primary", "m": 3, "qr": 0.55},
            {"source": "secondary", "m": 5, "qr": 0.75},
        ],
        geometry={"gamma": 0.0},
        wavelength=1.0,
        n2_value=1.0,
        native_shape=(16, 16),
        simulation_trace_samples=8,
    )

    assert unique_count == 1
    assert len(seed_matches) == 1
    seed = seed_matches[0]
    assert seed["group_id"] == 7
    assert seed["sim_source"] == "primary"
    assert seed["sim_m"] == 3
    assert seed["sim_qr"] == 0.55
    assert seed["branch_sign"] == -1
    assert seed["seed_rms_px"] == 0.0
    assert seed["guide_point_rms_px"] == 0.0
    assert seed["guide_shape_rms_px"] == 0.0
    assert seed["sim_seed_rms_px"] == 1.0
    assert np.allclose(np.asarray(seed["background_path"]["cols"], dtype=float), [0.0, 1.0])
    assert np.allclose(np.asarray(seed["background_path"]["rows"], dtype=float), [0.0, 1.0])


def test_direction_path_fit_keeps_full_window_sample_distribution() -> None:
    namespace = _load_main_functions(
        "_normalized_path_parameter",
        "_resample_ordered_vectors",
        "_window_parameter_samples",
        "_rms_distance_px",
        "_fit_background_group_to_direction_path",
    )

    namespace["_background_group_display_points"] = (
        lambda group: [(float(col), float(row)) for col, row in group.get("display_points", ())]
    )
    namespace["_background_group_to_detector_path"] = (
        lambda group: {
            "cols": np.asarray([0.0, 1.0], dtype=float),
            "rows": np.asarray([0.0, 1.0], dtype=float),
        }
    )
    namespace["_guide_fit_windows"] = lambda point_count: [(0.25, 0.75)]
    namespace["_fit_direction_homography"] = lambda directions, points: np.eye(3, dtype=float)
    namespace["_apply_direction_homography"] = (
        lambda homography, directions: (
            np.asarray(directions[:, 0], dtype=float),
            np.asarray(directions[:, 1], dtype=float),
            np.ones(directions.shape[0], dtype=bool),
        )
    )
    namespace["_point_distances_to_qr_path"] = lambda points, path: [0.0 for _ in points]
    namespace["_sampled_qr_path_distances_to_qr_path"] = (
        lambda source_path, target_path, sample_count: [0.0]
    )

    fit = namespace["_fit_background_group_to_direction_path"](
        {
            "display_points": [
                (0.0, 0.0),
                (1.0, 0.5),
                (2.0, 1.0),
                (3.0, 1.5),
            ]
        },
        {
            "branch_sign": 1,
            "dirs": np.asarray(
                [
                    [0.0, 0.0, 1.0],
                    [0.2, 0.1, 1.0],
                    [0.4, 0.2, 1.0],
                    [0.6, 0.3, 1.0],
                    [0.8, 0.4, 1.0],
                    [1.0, 0.5, 1.0],
                ],
                dtype=float,
            ),
            "valid_mask": np.asarray([True, True, True, True, True, True]),
        },
    )

    assert fit is not None
    path = fit["path"]
    assert int(path["sample_count"]) == 4
    assert np.allclose(
        np.asarray(path["parameter"], dtype=float),
        np.asarray([0.25, 0.4, 0.6, 0.75], dtype=float),
    )


def test_relative_point_sse_weights_normalize_and_prefer_best() -> None:
    namespace = _load_main_functions("_relative_point_sse_weights")

    weights = namespace["_relative_point_sse_weights"](
        [1.0, 5.0, 13.0],
        sigma_px=2.0,
    )

    assert np.isclose(float(np.sum(weights)), 1.0)
    assert float(weights[0]) > float(weights[1]) > float(weights[2])


def test_background_projection_distribution_builds_weighted_density() -> None:
    namespace = _load_main_functions(
        "_rms_distance_px",
        "_relative_point_sse_weights",
        "_accumulate_weighted_path_density",
        "_build_background_group_projection_distribution",
    )

    namespace["BACKGROUND_PEAK_PICK_FIT_RADIUS_PX"] = 4.0
    namespace["_background_group_display_points"] = (
        lambda group: [(float(col), float(row)) for col, row in group.get("display_points", ())]
    )
    namespace["_background_group_q_samples"] = (
        lambda *args, **kwargs: [
            type("Sample", (), {"qr_value": 0.48})(),
            type("Sample", (), {"qr_value": 0.50})(),
            type("Sample", (), {"qr_value": 0.52})(),
        ]
    )
    namespace["_analytic_qr_fit_bounds"] = lambda seed_qr, qr_values: (0.45, 0.55)
    namespace["_project_display_qr_paths"] = (
        lambda qr_value, geometry, **kwargs: [
            {
                "qr": float(qr_value),
                "branch_sign": -1,
                "cols": np.asarray([2.0, 3.0, 4.0], dtype=float),
                "rows": np.asarray(
                    [4.0 + 10.0 * (float(qr_value) - 0.5), 5.0, 6.0],
                    dtype=float,
                ),
            },
            {
                "qr": float(qr_value),
                "branch_sign": 1,
                "cols": np.asarray([2.0, 3.0, 4.0], dtype=float),
                "rows": np.asarray(
                    [8.0 + 10.0 * (float(qr_value) - 0.5), 9.0, 10.0],
                    dtype=float,
                ),
            },
        ]
    )
    namespace["_point_distances_to_qr_path"] = (
        lambda points, path: [
            (
                abs(float(path.get("qr", 0.0)) - 0.5) * 10.0 + 0.5
                if int(path.get("branch_sign", 0)) == -1
                else abs(float(path.get("qr", 0.0)) - 0.5) * 10.0 + 3.0
            )
            for _ in points
        ]
    )

    distribution = namespace["_build_background_group_projection_distribution"](
        {
            "id": 2,
            "display_points": [(2.0, 4.0), (3.0, 5.0), (4.0, 6.0)],
        },
        geometry={"gamma": 0.0},
        wavelength=1.0,
        n2_value=1.0,
        native_shape=(16, 16),
        display_shape=(12, 12),
        cfg={
            "distribution_qr_samples": 3,
            "distribution_probability_sigma_px": 1.0,
            "distribution_min_weight": 0.0,
        },
    )

    assert distribution is not None
    assert distribution["candidate_count"] == 18
    assert distribution["displayed_candidate_count"] == 18
    assert np.isclose(float(np.sum(np.asarray(distribution["distribution"], dtype=float))), 1.0)
    assert distribution["sim_qr"] == 0.5
    assert distribution["branch_sign"] == -1
    assert distribution["path"]["source"] == "background_projection_distribution"
