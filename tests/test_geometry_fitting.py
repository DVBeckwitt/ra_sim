from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from ra_sim.fitting import optimization as opt


def _base_params(image_size: int, *, optics_mode: int = 0) -> dict:
    return {
        "gamma": 0.0,
        "Gamma": 0.0,
        "corto_detector": 0.1,
        "theta_initial": 0.0,
        "cor_angle": 0.0,
        "zs": 0.0,
        "zb": 0.0,
        "chi": 0.0,
        "a": 4.0,
        "c": 7.0,
        "center": [image_size / 2.0, image_size / 2.0],
        "lambda": 1.0,
        "n2": 1.0,
        "psi": 0.0,
        "psi_z": 0.0,
        "debye_x": 0.0,
        "debye_y": 0.0,
        "optics_mode": int(optics_mode),
        "mosaic_params": {
            "beam_x_array": np.zeros(1, dtype=np.float64),
            "beam_y_array": np.zeros(1, dtype=np.float64),
            "theta_array": np.zeros(1, dtype=np.float64),
            "phi_array": np.zeros(1, dtype=np.float64),
            "sigma_mosaic_deg": 0.2,
            "gamma_mosaic_deg": 0.1,
            "eta": 0.05,
            "wavelength_array": np.ones(1, dtype=np.float64),
        },
    }


def _fake_process_peaks(*args, **kwargs):
    image_size = int(args[2])
    image = np.zeros((image_size, image_size), dtype=np.float64)
    hit_tables = [
        np.array(
            [[1.0, 4.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    ]
    return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []


def _fake_process_peaks_same_hkl_two_hits(*args, **kwargs):
    image_size = int(args[2])
    image = np.zeros((image_size, image_size), dtype=np.float64)
    hit_tables = [
        np.array(
            [
                [1.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                [0.8, 8.0, 8.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]
    return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []


def test_estimate_pixel_size_prefers_positive_sources_in_order():
    params = _base_params(12)
    params["pixel_size"] = 2.5e-4
    params["pixel_size_m"] = 1.5e-4
    params["debye_x"] = 5.0e-5
    assert np.isclose(opt._estimate_pixel_size(params), 2.5e-4)

    params = _base_params(12)
    params["pixel_size_m"] = 1.5e-4
    params["debye_x"] = 5.0e-5
    assert np.isclose(opt._estimate_pixel_size(params), 1.5e-4)

    params = _base_params(12)
    params["debye_x"] = 5.0e-5
    assert np.isclose(opt._estimate_pixel_size(params), 5.0e-5)

    params = _base_params(12)
    params["corto_detector"] = 0.2
    assert np.isclose(opt._estimate_pixel_size(params), 0.2 / 4096.0)


def _fake_process_three_reflections(*args, **kwargs):
    image_size = int(args[2])
    miller_subset = np.asarray(args[0], dtype=np.float64)
    image = np.zeros((image_size, image_size), dtype=np.float64)
    coord_map = {
        (1, 0, 0): (4.0, 4.0),
        (0, 1, 0): (10.0, 10.0),
        (0, 0, 1): (14.0, 14.0),
    }
    hit_tables = []
    for row in miller_subset:
        hkl = tuple(int(round(v)) for v in row)
        col, row_px = coord_map[hkl]
        hit_tables.append(
            np.array(
                [[10.0, col, row_px, 0.0, float(hkl[0]), float(hkl[1]), float(hkl[2])]],
                dtype=np.float64,
            )
        )
    return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []


def _process_wrapper_args(image_size: int):
    params = _base_params(image_size)
    mosaic = params["mosaic_params"]
    return (
        np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        int(image_size),
        float(params["a"]),
        float(params["c"]),
        float(params["lambda"]),
        np.zeros((image_size, image_size), dtype=np.float64),
        float(params["corto_detector"]),
        float(params["gamma"]),
        float(params["Gamma"]),
        float(params["chi"]),
        float(params["psi"]),
        float(params["psi_z"]),
        float(params["zs"]),
        float(params["zb"]),
        params["n2"],
        np.asarray(mosaic["beam_x_array"], dtype=np.float64),
        np.asarray(mosaic["beam_y_array"], dtype=np.float64),
        np.asarray(mosaic["theta_array"], dtype=np.float64),
        np.asarray(mosaic["phi_array"], dtype=np.float64),
        float(mosaic["sigma_mosaic_deg"]),
        float(mosaic["gamma_mosaic_deg"]),
        float(mosaic["eta"]),
        np.asarray(mosaic["wavelength_array"], dtype=np.float64),
        float(params["debye_x"]),
        float(params["debye_y"]),
        np.asarray(params["center"], dtype=np.float64),
        float(params["theta_initial"]),
        float(params["cor_angle"]),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )


def test_build_global_point_matches_uses_global_assignment():
    simulated = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)]
    measured = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]

    matches = opt._build_global_point_matches(simulated, measured)

    assert {
        (int(sim_idx), int(meas_idx))
        for *_pts, sim_idx, meas_idx in matches
    } == {(0, 0), (1, 1), (2, 2)}
    assert np.isclose(
        sum(float(distance) for *_pts, distance, _sim_idx, _meas_idx in matches),
        2.0 * np.sqrt(2.0),
    )


def test_dynamic_point_match_reanchors_measured_anchor_and_reports_motion(
    monkeypatch,
):
    calls: dict[str, object] = {}

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 12.0, 12.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_reanchor(
        measured_entry,
        simulated_detector_point,
        *,
        local_params=None,
        dataset_ctx=None,
    ):
        calls["reanchor"] = {
            "measured_entry": dict(measured_entry),
            "simulated_detector_point": tuple(simulated_detector_point),
            "dataset_index": (
                int(dataset_ctx.dataset_index) if dataset_ctx is not None else None
            ),
        }
        return {
            "x": float(simulated_detector_point[0]),
            "y": float(simulated_detector_point[1]),
            "detector_x": float(simulated_detector_point[0]),
            "detector_y": float(simulated_detector_point[1]),
        }

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    subset = opt.ReflectionSimulationSubset(
        miller=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        measured_entries=[
            {
                "label": "peak-0",
                "hkl": (1, 0, 0),
                "overlay_match_index": 0,
                "source_table_index": 0,
                "source_row_index": 0,
                "detector_x": 6.0,
                "detector_y": 6.0,
                "x": 6.0,
                "y": 6.0,
            }
        ],
        original_indices=np.array([0], dtype=np.int64),
        total_reflection_count=1,
        fixed_source_reflection_count=1,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        subset=subset,
        experimental_image=np.zeros((32, 32), dtype=np.float64),
        dynamic_reanchor_enabled=True,
        dynamic_reanchor_callback=fake_reanchor,
    )
    local = _base_params(32)
    local["pixel_size"] = 1.0
    local["corto_detector"] = 100.0

    residual, diagnostics, summary = (
        opt._evaluate_geometry_fit_dataset_dynamic_point_matches(
            local,
            dataset_ctx,
            image_size=32,
            missing_pair_penalty_deg=5.0,
            theta_value=0.0,
            collect_diagnostics=True,
        )
    )

    assert calls["reanchor"]["simulated_detector_point"] == (12.0, 12.0)
    assert calls["reanchor"]["dataset_index"] == 0
    assert residual.shape == (2,)
    assert np.allclose(residual, 0.0, atol=1.0e-12)
    assert diagnostics[0]["measured_reanchor_attempted"] is True
    assert diagnostics[0]["measured_reanchor_status"] == "updated"
    assert diagnostics[0]["measured_reanchor_motion_px"] > 0.0
    assert summary["matched_pair_count"] == 1
    assert summary["measured_anchor_reanchor_enabled"] is True
    assert summary["measured_anchor_reanchor_attempt_count"] == 1
    assert summary["measured_anchor_reanchor_count"] == 1
    assert summary["measured_anchor_reanchor_fail_count"] == 0
    assert summary["measured_anchor_motion_max_px"] > 0.0


def test_detector_pixels_to_fit_space_matches_zero_tilt_geometry() -> None:
    cols = np.array([10.0, 13.0, 10.0], dtype=np.float64)
    rows = np.array([7.0, 10.0, 13.0], dtype=np.float64)
    center = [10.0, 10.0]
    detector_distance = 5.0
    pixel_size = 2.0

    two_theta, phi = opt._detector_pixels_to_fit_space(
        cols,
        rows,
        center=center,
        detector_distance=detector_distance,
        pixel_size=pixel_size,
    )

    x = (cols - float(center[1])) * pixel_size
    z = (float(center[0]) - rows) * pixel_size
    expected_two_theta = np.degrees(
        np.arctan2(np.hypot(x, z), np.full_like(x, detector_distance))
    )
    expected_phi = np.degrees(np.arctan2(x, z))
    expected_phi = (expected_phi + 180.0) % 360.0 - 180.0

    assert np.allclose(two_theta, expected_two_theta, atol=1.0e-12)
    assert np.allclose(phi, expected_phi, atol=1.0e-12)
    assert phi[-1] == -180.0

    single_two_theta, single_phi = opt._pixel_to_angles(
        float(cols[1]),
        float(rows[1]),
        center,
        detector_distance,
        pixel_size,
    )
    assert single_two_theta == pytest.approx(float(expected_two_theta[1]))
    assert single_phi == pytest.approx(float(expected_phi[1]))


def test_measured_fit_space_anchor_prefers_detector_anchor_over_cached_fit_space() -> None:
    center = [10.0, 10.0]
    entry = {
        "detector_x": 13.0,
        "detector_y": 7.0,
        "background_detector_x": 14.0,
        "background_detector_y": 6.0,
        "background_two_theta_deg": 91.0,
        "background_phi_deg": -42.0,
    }

    anchor, reason, metadata = opt._measured_fit_space_anchor(
        entry,
        center=center,
        detector_distance=5.0,
        pixel_size=2.0,
    )
    expected = opt._pixel_to_angles(13.0, 7.0, center, 5.0, 2.0)

    assert reason == "detector_fit_space_anchor"
    assert anchor == pytest.approx(expected)
    assert metadata["anchor_source"] == "detector_fit_space_anchor"

    fallback_entry = {
        "background_detector_x": 14.0,
        "background_detector_y": 6.0,
        "background_two_theta_deg": 91.0,
        "background_phi_deg": -42.0,
    }
    fallback_anchor, fallback_reason, fallback_metadata = opt._measured_fit_space_anchor(
        fallback_entry,
        center=center,
        detector_distance=5.0,
        pixel_size=2.0,
    )
    expected_fallback = opt._pixel_to_angles(14.0, 6.0, center, 5.0, 2.0)

    assert fallback_reason == "background_detector_fit_space_anchor"
    assert fallback_anchor == pytest.approx(expected_fallback)
    assert fallback_metadata["anchor_source"] == "background_detector_fit_space_anchor"


def test_measured_fit_space_anchor_keeps_cached_fit_space_stable_when_wavelength_changes(
    monkeypatch,
) -> None:
    def fake_theoretical_two_theta(entry, *, a_lattice, c_lattice, wavelength):
        del entry, a_lattice, c_lattice
        if wavelength == pytest.approx(1.0):
            return 21.0
        return 33.0

    monkeypatch.setattr(
        opt,
        "_entry_theoretical_two_theta_deg",
        fake_theoretical_two_theta,
    )

    anchor, reason, metadata = opt._measured_fit_space_anchor(
        {
            "background_two_theta_deg": 20.0,
            "background_phi_deg": 5.0,
            "background_reference_a": 4.0,
            "background_reference_c": 7.0,
            "background_reference_lambda": 1.0,
        },
        center=[10.0, 10.0],
        detector_distance=5.0,
        pixel_size=2.0,
        a_lattice=4.0,
        c_lattice=7.0,
        wavelength=1.3,
    )

    assert reason == "cached_fit_space_anchor"
    assert anchor == pytest.approx((20.0, 5.0))
    assert metadata["two_theta_adjustment_deg"] == 0.0
    assert metadata["reference_two_theta_deg"] == pytest.approx(21.0)
    assert metadata["current_theoretical_two_theta_deg"] == pytest.approx(33.0)


def test_measured_fit_space_anchor_prefers_explicit_fit_space_override_over_detector_anchor() -> None:
    anchor, reason, metadata = opt._measured_fit_space_anchor(
        {
            "detector_x": 13.0,
            "detector_y": 7.0,
            "background_two_theta_deg": 20.0,
            "background_phi_deg": 5.0,
            "fit_space_anchor_override": True,
        },
        center=[10.0, 10.0],
        detector_distance=5.0,
        pixel_size=2.0,
    )

    assert reason == "cached_fit_space_anchor"
    assert anchor == pytest.approx((20.0, 5.0))
    assert metadata["anchor_source"] == "cached_fit_space_anchor"
    assert metadata["cached_two_theta_deg"] == pytest.approx(20.0)
    assert metadata["cached_phi_deg"] == pytest.approx(5.0)


def test_dynamic_point_match_reanchor_does_not_mutate_measured_entries(monkeypatch):
    callback_entries: list[dict[str, object]] = []

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 12.0, 12.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_reanchor(
        measured_entry,
        simulated_detector_point,
        *,
        local_params=None,
        dataset_ctx=None,
    ):
        del local_params, dataset_ctx
        callback_entries.append(dict(measured_entry))
        return {
            "x": float(simulated_detector_point[0]),
            "y": float(simulated_detector_point[1]),
            "detector_x": float(simulated_detector_point[0]),
            "detector_y": float(simulated_detector_point[1]),
            "measured_reanchor_motion_px": 3.0,
        }

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    original_entry = {
        "label": "peak-0",
        "hkl": (1, 0, 0),
        "overlay_match_index": 0,
        "source_table_index": 0,
        "source_row_index": 0,
        "detector_x": 6.0,
        "detector_y": 6.0,
        "background_detector_x": 6.0,
        "background_detector_y": 6.0,
        "x": 6.0,
        "y": 6.0,
    }
    subset = opt.ReflectionSimulationSubset(
        miller=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        measured_entries=[dict(original_entry)],
        original_indices=np.array([0], dtype=np.int64),
        total_reflection_count=1,
        fixed_source_reflection_count=1,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        subset=subset,
        experimental_image=np.zeros((32, 32), dtype=np.float64),
        dynamic_reanchor_enabled=True,
        dynamic_reanchor_callback=fake_reanchor,
    )
    local = _base_params(32)
    local["pixel_size"] = 1.0
    local["corto_detector"] = 100.0

    for _ in range(2):
        residual, diagnostics, summary = (
            opt._evaluate_geometry_fit_dataset_dynamic_point_matches(
                local,
                dataset_ctx,
                image_size=32,
                missing_pair_penalty_deg=5.0,
                theta_value=0.0,
                collect_diagnostics=True,
            )
        )
        assert residual.shape == (2,)
        assert diagnostics[0]["measured_reanchor_status"] == "updated"
        assert summary["matched_pair_count"] == 1
        assert summary["measured_anchor_motion_mean_px"] == pytest.approx(3.0)
        assert summary["measured_anchor_motion_rms_px"] == pytest.approx(3.0)
        assert summary["measured_anchor_motion_max_px"] == pytest.approx(3.0)

    assert len(callback_entries) == 2
    assert callback_entries[0]["detector_x"] == 6.0
    assert callback_entries[1]["detector_x"] == 6.0
    assert subset.measured_entries[0] == original_entry


def test_resolve_parallel_worker_count_auto_reserves_two_threads(monkeypatch):
    monkeypatch.setattr(opt, "_available_parallel_thread_budget", lambda: 12)

    assert opt._resolve_parallel_worker_count("auto", max_tasks=32) == 10
    assert opt._resolve_parallel_worker_count(None, max_tasks=8) == 8


def test_resolve_parallel_worker_count_auto_keeps_one_worker_minimum(monkeypatch):
    monkeypatch.setattr(opt, "_available_parallel_thread_budget", lambda: 2)

    assert opt._resolve_parallel_worker_count("auto", max_tasks=32) == 1


def test_fit_geometry_parameters_cost_fn_uses_updated_psi_z(monkeypatch):
    target = 1.25
    psi_z_seen = []

    def fake_compute(*args, **kwargs):
        psi_z = float(kwargs["psi_z"])
        psi_z_seen.append(psi_z)
        return np.array([psi_z - target], dtype=np.float64)

    monkeypatch.setattr(
        opt, "compute_peak_position_error_geometry_local", fake_compute
    )

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[],
        var_names=["psi_z"],
        experimental_image=None,
    )

    assert result.success
    assert abs(float(result.x[0]) - target) < 1e-8
    assert any(abs(v - target) < 1e-3 for v in psi_z_seen)


def test_fit_geometry_parameters_applies_parameter_priors(monkeypatch):
    target = 1.0

    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        return np.array([gamma - target], dtype=np.float64)

    monkeypatch.setattr(
        opt, "compute_peak_position_error_geometry_local", fake_compute
    )

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[],
        var_names=["gamma"],
        experimental_image=None,
        refinement_config={
            "solver": {"restarts": 0},
            "priors": {"gamma": {"center": 0.0, "sigma": 0.25}},
        },
    )

    assert result.success
    assert 0.0 < float(result.x[0]) < 0.1
    assert isinstance(result.parameter_prior_summary, list)
    assert result.parameter_prior_summary == [
        {"name": "gamma", "center": 0.0, "sigma": 0.25}
    ]


def test_fit_geometry_parameters_pixel_path_forwards_optics_mode(monkeypatch):
    optics_seen = []

    def fake_process(*args, **kwargs):
        optics_seen.append(kwargs.get("optics_mode"))
        return _fake_process_peaks(*args, **kwargs)

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
    )

    assert result.success
    assert optics_seen
    assert all(mode == 1 for mode in optics_seen)


def test_fit_geometry_parameters_pixel_path_uses_central_geometry_ray(monkeypatch):
    process_calls = []

    def fake_process(*args, **kwargs):
        process_calls.append(
            {
                "kwargs": dict(kwargs),
                "wavelength_array": np.asarray(args[5], dtype=np.float64).copy(),
                "beam_x_array": np.asarray(args[16], dtype=np.float64).copy(),
                "beam_y_array": np.asarray(args[17], dtype=np.float64).copy(),
                "theta_array": np.asarray(args[18], dtype=np.float64).copy(),
                "phi_array": np.asarray(args[19], dtype=np.float64).copy(),
            }
        )
        return _fake_process_peaks(*args, **kwargs)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {"restarts": 0},
            "full_beam_polish": {"enabled": False},
        },
    )

    assert result.success
    assert process_calls
    for call in process_calls:
        assert call["beam_x_array"].shape == (1,)
        assert call["beam_y_array"].shape == (1,)
        assert call["theta_array"].shape == (1,)
        assert call["phi_array"].shape == (1,)
        assert call["wavelength_array"].shape == (1,)
        assert np.allclose(call["beam_x_array"], [0.0])
        assert np.allclose(call["beam_y_array"], [0.0])
        assert np.allclose(call["theta_array"], [0.0])
        assert np.allclose(call["phi_array"], [0.0])
        assert np.allclose(call["wavelength_array"], [1.0])
        assert call["kwargs"].get("best_sample_indices_out") is None
        assert call["kwargs"].get("single_sample_indices") is None
    assert isinstance(result.point_match_summary, dict)
    assert bool(result.point_match_summary["central_ray_mode"]) is True
    assert bool(result.point_match_summary["single_ray_enabled"]) is False
    assert int(result.point_match_summary["single_ray_forced_count"]) == 0


def test_fit_geometry_parameters_pixel_path_runs_without_full_ray_polish(
    monkeypatch,
):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 2.0 + 4.0 * gamma, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    solve_calls = {"count": 0}

    def fake_least_squares(residual_fn, x0, **kwargs):
        solve_calls["count"] += 1
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message=f"solve#{solve_calls['count']}",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {"restarts": 0, "stagnation_probe": False},
            "full_beam_polish": {"enabled": False},
        },
    )

    assert result.success
    assert solve_calls["count"] >= 1
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([0.5]))
    assert np.isclose(float(result.cost), 0.0)
    assert isinstance(result.geometry_fit_debug_summary, dict)
    assert str(result.geometry_fit_debug_summary["main_solve_seed"]["seed_kind"]) == "u=0"
    assert isinstance(result.point_match_summary, dict)
    assert bool(result.point_match_summary["central_ray_mode"]) is True
    assert bool(result.point_match_summary["single_ray_enabled"]) is False


def test_process_peaks_wrapper_prefers_python_runner_when_numba_disabled(monkeypatch):
    process_calls = []

    def fake_process(*args, **kwargs):
        process_calls.append(
            {
                "kwargs": dict(kwargs),
                "wavelength_array": np.asarray(args[5], dtype=np.float64).copy(),
                "beam_x_array": np.asarray(args[16], dtype=np.float64).copy(),
                "beam_y_array": np.asarray(args[17], dtype=np.float64).copy(),
                "theta_array": np.asarray(args[18], dtype=np.float64).copy(),
                "phi_array": np.asarray(args[19], dtype=np.float64).copy(),
            }
        )
        return _fake_process_peaks(*args, **kwargs)

    monkeypatch.setattr(opt, "process_peaks_parallel", fake_process)
    monkeypatch.setattr(opt, "_USE_NUMBA_PROCESS_PEAKS", False)

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"].update(
        {
            "beam_x_array": np.array([0.3, -0.2, 0.1], dtype=np.float64),
            "beam_y_array": np.array([0.4, -0.1, 0.2], dtype=np.float64),
            "theta_array": np.array([0.02, -0.03, 0.01], dtype=np.float64),
            "phi_array": np.array([0.04, -0.05, 0.02], dtype=np.float64),
            "wavelength_array": np.array([0.8, 1.2, 1.4], dtype=np.float64),
        }
    )
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {"restarts": 0},
            "single_ray": {"enabled": False},
            "use_numba": False,
            "full_beam_polish": {"enabled": False},
        },
    )

    assert result.success
    assert process_calls
    assert all(call["kwargs"].get("prefer_python_runner") is True for call in process_calls)
    for call in process_calls:
        assert np.allclose(call["beam_x_array"], [0.0])
        assert np.allclose(call["beam_y_array"], [0.0])
        assert np.allclose(call["theta_array"], [0.0])
        assert np.allclose(call["phi_array"], [0.0])
        assert np.allclose(call["wavelength_array"], [1.0])


def test_process_peaks_wrapper_serializes_first_numba_warmup(monkeypatch):
    process_calls = []
    call_gate = threading.Barrier(2)
    state_lock = threading.Lock()
    active_calls = 0
    max_active_calls = 0

    def fake_process(*args, **kwargs):
        nonlocal active_calls
        nonlocal max_active_calls
        with state_lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
            process_calls.append(dict(kwargs))
        try:
            threading.Event().wait(0.05)
            return _fake_process_peaks(*args, **kwargs)
        finally:
            with state_lock:
                active_calls -= 1

    monkeypatch.setattr(opt, "process_peaks_parallel", fake_process)
    monkeypatch.setattr(opt, "_USE_NUMBA_PROCESS_PEAKS", True)
    monkeypatch.setattr(opt, "_NUMBA_PROCESS_PEAKS_WARMED", False)

    process_args = _process_wrapper_args(8)

    def run_once():
        call_gate.wait(timeout=5.0)
        return opt._process_peaks_parallel_safe(*process_args, save_flag=0)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_once) for _ in range(2)]
        for future in futures:
            image, *_ = future.result(timeout=5.0)
            assert image.shape == (8, 8)

    assert len(process_calls) == 2
    assert max_active_calls == 1
    assert opt._NUMBA_PROCESS_PEAKS_WARMED is True


def test_process_peaks_wrapper_disables_numba_after_python_fallback(monkeypatch):
    process_calls = []
    safe_stats = {"used_python_runner": True}

    def fake_safe_wrapper(*args, **kwargs):
        process_calls.append(dict(kwargs))
        return _fake_process_peaks(*args, **kwargs)

    monkeypatch.setattr(opt, "process_peaks_parallel", fake_safe_wrapper)
    monkeypatch.setattr(opt, "_DIFFRACTION_PROCESS_PEAKS_SAFE_WRAPPER", fake_safe_wrapper)
    monkeypatch.setattr(
        opt,
        "get_last_process_peaks_safe_stats",
        lambda: dict(safe_stats),
    )
    monkeypatch.setattr(opt, "_USE_NUMBA_PROCESS_PEAKS", True)
    monkeypatch.setattr(opt, "_NUMBA_PROCESS_PEAKS_WARMED", False)

    process_args = _process_wrapper_args(8)

    opt._process_peaks_parallel_safe(*process_args, save_flag=0)

    assert opt._USE_NUMBA_PROCESS_PEAKS is False
    assert opt._NUMBA_PROCESS_PEAKS_WARMED is False
    assert process_calls[0].get("prefer_python_runner") is None

    opt._process_peaks_parallel_safe(*process_args, save_flag=0)

    assert process_calls[1].get("prefer_python_runner") is True


def test_fit_geometry_parameters_pixel_path_restricts_simulation_to_selected_reflections(
    monkeypatch,
):
    process_calls = []

    def fake_process(*args, **kwargs):
        process_calls.append(
            {
                "miller": np.asarray(args[0], dtype=np.float64).copy(),
                "wavelength_array": np.asarray(args[5], dtype=np.float64).copy(),
                "beam_x_array": np.asarray(args[16], dtype=np.float64).copy(),
                "beam_y_array": np.asarray(args[17], dtype=np.float64).copy(),
                "theta_array": np.asarray(args[18], dtype=np.float64).copy(),
                "phi_array": np.asarray(args[19], dtype=np.float64).copy(),
                "kwargs": dict(kwargs),
            }
        )
        miller_arg = np.asarray(args[0], dtype=np.float64)
        image_size = int(args[2])
        hit_tables = []
        for row in miller_arg:
            hit_tables.append(
                np.array(
                    [[1.0, 5.0, 4.0, 0.0, row[0], row[1], row[2]]],
                    dtype=np.float64,
                )
            )
        best_sample_indices_out = kwargs.get("best_sample_indices_out")
        if isinstance(best_sample_indices_out, np.ndarray):
            best_sample_indices_out[:] = 0
        image = np.zeros((image_size, image_size), dtype=np.float64)
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 5.0,
            "y": 4.0,
            "source_table_index": 1,
            "source_row_index": 0,
        }
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {"restarts": 0},
            "full_beam_polish": {"enabled": False},
        },
    )

    assert result.success
    assert process_calls
    assert all(call["miller"].shape == (1, 3) for call in process_calls)
    assert all(
        np.allclose(call["miller"], np.array([[2.0, 0.0, 0.0]], dtype=np.float64))
        for call in process_calls
    )
    for call in process_calls:
        assert np.allclose(call["beam_x_array"], [0.0])
        assert np.allclose(call["beam_y_array"], [0.0])
        assert np.allclose(call["theta_array"], [0.0])
        assert np.allclose(call["phi_array"], [0.0])
        assert np.allclose(call["wavelength_array"], [1.0])
        assert call["kwargs"].get("best_sample_indices_out") is None
        assert call["kwargs"].get("single_sample_indices") is None
    assert isinstance(result.point_match_summary, dict)
    assert int(result.point_match_summary["simulated_reflection_count"]) == 1
    assert int(result.point_match_summary["total_reflection_count"]) == 3
    assert bool(result.point_match_summary["subset_reduced"]) is True
    assert bool(result.point_match_summary["central_ray_mode"]) is True
    assert bool(result.point_match_summary["single_ray_enabled"]) is False
    assert int(result.point_match_summary["single_ray_forced_count"]) == 0


def test_prepare_reflection_subset_preserves_distinct_reflections_within_one_q_group() -> None:
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    measured = [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 4.0,
            "y": 4.0,
            "q_group_key": ["q", 1],
            "source_table_index": 0,
            "source_reflection_index": 0,
            "source_reflection_is_full": True,
            "source_row_index": 0,
        },
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.2,
            "y": 4.1,
            "q_group_key": ("q", 1),
            "source_table_index": 1,
            "source_reflection_index": 1,
            "source_reflection_is_full": True,
            "source_row_index": 0,
        },
        {
            "hkl": (3, 0, 0),
            "label": "3,0,0",
            "x": 6.0,
            "y": 5.0,
            "q_group_key": ("q", 2),
            "source_table_index": 2,
            "source_reflection_index": 2,
            "source_reflection_is_full": True,
            "source_row_index": 0,
        },
    ]

    subset = opt._prepare_reflection_subset(miller, intensities, measured)

    assert subset.reduced is False
    assert np.allclose(
        subset.miller,
        np.array(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
    )
    assert np.array_equal(subset.original_indices, np.array([0, 1, 2], dtype=np.int64))
    assert len(subset.measured_entries) == 3
    assert [entry["q_group_key"] for entry in subset.measured_entries] == [
        ("q", 1),
        ("q", 1),
        ("q", 2),
    ]
    assert [entry["source_table_index"] for entry in subset.measured_entries] == [
        0,
        1,
        2,
    ]


def test_prepare_reflection_subset_rebinds_stale_source_identity_by_hkl() -> None:
    miller = np.array(
        [
            [5.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([5.0, 2.0, 7.0], dtype=np.float64)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "fit_source_identity_only": True,
        }
    ]

    subset = opt._prepare_reflection_subset(miller, intensities, measured)

    assert subset.reduced is True
    assert subset.fixed_source_reflection_count == 0
    assert subset.fallback_hkl_count == 1
    assert np.array_equal(subset.original_indices, np.array([1], dtype=np.int64))
    assert np.allclose(subset.miller, np.array([[2.0, 0.0, 0.0]], dtype=np.float64))
    for key in (
        "source_table_index",
        "source_reflection_index",
        "resolved_table_index",
        "source_row_index",
        "source_peak_index",
    ):
        assert key not in subset.measured_entries[0]


def test_prepare_reflection_subset_prefers_source_reflection_index_over_stale_table_index() -> None:
    miller = np.array(
        [
            [5.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([5.0, 2.0, 7.0], dtype=np.float64)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_reflection_index": 1,
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "fit_source_identity_only": True,
        }
    ]

    subset = opt._prepare_reflection_subset(miller, intensities, measured)

    assert subset.reduced is True
    assert subset.fixed_source_reflection_count == 1
    assert subset.fallback_hkl_count == 0
    assert np.array_equal(subset.original_indices, np.array([1], dtype=np.int64))
    assert np.allclose(subset.miller, np.array([[2.0, 0.0, 0.0]], dtype=np.float64))
    assert subset.measured_entries[0]["source_table_index"] == 0
    assert subset.measured_entries[0]["source_reflection_index"] == 1
    assert subset.measured_entries[0]["resolved_table_index"] == 0
    assert subset.measured_entries[0]["source_row_index"] == 0


def test_prepare_reflection_subset_preserves_trusted_identity_and_only_remaps_local_lookup_ids() -> None:
    miller = np.array(
        [
            [5.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([5.0, 2.0, 7.0], dtype=np.float64)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_reflection_index": 1,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "resolved_table_index": 99,
            "resolved_peak_index": 99,
            "fit_source_identity_only": True,
        }
    ]

    subset = opt._prepare_reflection_subset(miller, intensities, measured)

    assert subset.reduced is True
    assert subset.fixed_source_reflection_count == 1
    assert subset.fallback_hkl_count == 0
    entry = subset.measured_entries[0]
    assert entry["source_reflection_index"] == 1
    assert entry["source_reflection_namespace"] == "full_reflection"
    assert entry["source_reflection_is_full"] is True
    assert entry["source_branch_index"] == 1
    assert entry["source_peak_index"] == 1
    assert entry["source_table_index"] == 0
    assert entry["source_row_index"] == 0
    assert entry["resolved_table_index"] == 0
    assert entry["resolved_peak_index"] == 1


def test_prepare_reflection_subset_keeps_duplicate_fixed_source_rows_out_of_hkl_fallback() -> None:
    miller = np.array(
        [
            [2.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([2.0, 7.0], dtype=np.float64)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0-a",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 9,
            "source_row_index": 0,
            "fit_source_identity_only": True,
        },
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0-b",
            "x": 5.0,
            "y": 5.0,
            "source_table_index": 9,
            "source_row_index": 1,
            "fit_source_identity_only": True,
        },
    ]

    subset = opt._prepare_reflection_subset(miller, intensities, measured)

    assert subset.fixed_source_reflection_count == 0
    assert subset.fallback_hkl_count == 1
    assert np.array_equal(subset.original_indices, np.array([0], dtype=np.int64))
    for entry in subset.measured_entries:
        for key in (
            "source_table_index",
            "source_reflection_index",
            "resolved_table_index",
            "source_row_index",
            "source_peak_index",
        ):
            assert key not in entry


def test_fit_geometry_parameters_pixel_path_keeps_residual_size_when_pair_status_changes(
    monkeypatch,
):
    captured_residuals = {}

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        sim_col = 4.0 if gamma < 0.5 else 20.0
        sim_row = 4.0 if gamma < 0.5 else 20.0
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, sim_col, sim_row, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x_match = np.asarray(x0, dtype=float).copy()
        x_match[0] = 0.0
        x_missing = np.asarray(x0, dtype=float).copy()
        x_missing[0] = 1.0

        residual_match = np.asarray(residual_fn(x_match), dtype=float)
        residual_missing = np.asarray(residual_fn(x_missing), dtype=float)
        captured_residuals["match"] = residual_match.copy()
        captured_residuals["missing"] = residual_missing.copy()

        assert residual_match.shape == residual_missing.shape == (2,)

        return opt.OptimizeResult(
            x=x_match,
            fun=residual_match,
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x_match, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 24
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"hkl": (1, 0, 0), "label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        pixel_tol=2.0,
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
                "missing_pair_penalty_px": 11.0,
            },
            "single_ray": {"enabled": False},
            "identifiability": {"enabled": False},
            "full_beam_polish": {"enabled": False},
        },
    )

    assert result.success
    assert np.allclose(captured_residuals["match"], [0.0, 0.0])
    assert np.allclose(captured_residuals["missing"], [11.0, 0.0])


def test_fit_geometry_parameters_dynamic_point_path_uses_angular_missing_penalty(
    monkeypatch,
):
    captured_residuals = {}
    phase = {"mode": "match"}

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        if phase["mode"] == "match":
            hit_tables = [
                np.array(
                    [[1.0, 12.0, 12.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                )
            ]
        else:
            hit_tables = [np.empty((0, 7), dtype=np.float64)]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x_match = np.asarray(x0, dtype=float).copy()
        x_match[0] = 0.0
        x_missing = np.asarray(x0, dtype=float).copy()
        x_missing[0] = 1.0

        phase["mode"] = "match"
        residual_match = np.asarray(residual_fn(x_match), dtype=float)
        phase["mode"] = "missing"
        residual_missing = np.asarray(residual_fn(x_missing), dtype=float)
        captured_residuals["match"] = residual_match.copy()
        captured_residuals["missing"] = residual_missing.copy()
        phase["mode"] = "match"

        assert residual_match.shape == residual_missing.shape == (2,)

        return opt.OptimizeResult(
            x=x_match,
            fun=residual_match,
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x_match, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 24
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["debye_x"] = 1.0
    params["debye_y"] = 1.0
    measured = [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 12.0,
            "y": 12.0,
            "detector_x": 12.0,
            "detector_y": 12.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "fit_source_identity_only": True,
        }
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        pixel_tol=2.0,
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "dynamic_point_geometry_fit": True,
                "weighted_matching": False,
                "use_measurement_uncertainty": False,
                "missing_pair_penalty_deg": 7.0,
            },
            "single_ray": {"enabled": False},
            "identifiability": {"enabled": False},
            "full_beam_polish": {"enabled": False},
        },
    )

    assert result.success
    assert result.final_metric_name == "dynamic_angular_point_match"
    assert np.allclose(captured_residuals["match"], [0.0, 0.0])
    assert np.allclose(captured_residuals["missing"], [7.0, 0.0])
    assert int(result.point_match_summary["missing_pair_count"]) == 0
    assert bool(result.geometry_fit_debug_summary["dynamic_point_geometry_fit"]) is True


def test_fit_geometry_parameters_dynamic_point_path_records_fit_space_provenance(
    monkeypatch,
):
    captured: dict[str, np.ndarray] = {}

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 12.0, 12.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
            np.array(
                [[1.0, 14.0, 10.0, 0.0, 0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        captured["residual"] = np.asarray(residual_fn(x), dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=captured["residual"],
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 24
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([1.0, 1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["pixel_size_m"] = 1.0e-4
    params["lambda"] = 1.1
    measured = [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 12.0,
            "y": 12.0,
            "detector_x": 12.0,
            "detector_y": 12.0,
            "background_two_theta_deg": 20.0,
            "background_phi_deg": 5.0,
            "background_reference_a": 4.0,
            "background_reference_c": 7.0,
            "background_reference_lambda": 1.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "fit_source_identity_only": True,
        },
        {
            "hkl": (0, 1, 0),
            "label": "0,1,0",
            "x": 14.0,
            "y": 10.0,
            "detector_x": 14.0,
            "detector_y": 10.0,
            "source_table_index": 1,
            "source_row_index": 0,
            "fit_source_identity_only": True,
        },
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "dynamic_point_geometry_fit": True,
                "weighted_matching": False,
                "use_measurement_uncertainty": False,
                "missing_pair_penalty_deg": 7.0,
            },
            "single_ray": {"enabled": False},
            "identifiability": {"enabled": False},
            "full_beam_polish": {"enabled": False},
        },
    )

    assert result.success
    assert captured["residual"].shape == (4,)
    assert isinstance(result.point_match_summary, dict)
    assert result.point_match_summary["fit_space_pixel_size_source"] == "pixel_size_m"
    assert np.isclose(
        float(result.point_match_summary["fit_space_pixel_size_value"]),
        1.0e-4,
    )
    assert np.isclose(
        float(result.point_match_summary["fit_space_pixel_size_m_raw"]),
        1.0e-4,
    )
    assert np.isclose(
        float(result.point_match_summary["fit_space_debye_x_raw"]),
        0.0,
    )
    assert np.isclose(
        float(result.point_match_summary["fit_space_debye_y_raw"]),
        0.0,
    )
    assert int(result.point_match_summary["fit_space_anchor_count_cached"]) == 0
    assert int(result.point_match_summary["fit_space_anchor_count_detector"]) == 2
    assert result.point_match_summary["fit_space_anchor_source_counts"] == {
        "cached_fit_space_anchor": 0,
        "detector_fit_space_anchor": 2,
    }
    assert int(result.point_match_summary["fit_space_two_theta_adjustment_count"]) == 0
    assert float(
        result.point_match_summary["fit_space_two_theta_adjustment_total_abs_deg"]
    ) == 0.0
    assert np.isnan(
        float(result.point_match_summary["fit_space_two_theta_adjustment_mean_abs_deg"])
    )
    assert np.isnan(
        float(result.point_match_summary["fit_space_two_theta_adjustment_max_abs_deg"])
    )
    assert len(result.point_match_summary["per_dataset"]) == 1
    assert int(
        result.point_match_summary["per_dataset"][0]["fit_space_anchor_count_cached"]
    ) == 0
    assert int(
        result.point_match_summary["per_dataset"][0]["fit_space_anchor_count_detector"]
    ) == 2


def test_simulate_and_compare_hkl_forwards_optics_mode(monkeypatch):
    optics_seen = []

    def fake_process(*args, **kwargs):
        optics_seen.append(kwargs.get("optics_mode"))
        return _fake_process_peaks(*args, **kwargs)

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=2)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]

    distances, *_ = opt.simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        params,
        measured,
    )

    assert distances.size == 2
    assert optics_seen == [2]


def test_simulate_and_compare_hkl_can_force_python_runner(monkeypatch):
    prefer_python_runner_seen = []

    def fake_process(*args, **kwargs):
        prefer_python_runner_seen.append(kwargs.get("prefer_python_runner"))
        return _fake_process_peaks(*args, **kwargs)

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]

    distances, *_ = opt.simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        params,
        measured,
        prefer_python_runner=True,
    )

    assert distances.size == 2
    assert prefer_python_runner_seen == [True]


def test_simulate_and_compare_hkl_restricts_to_measured_hkl_subset(monkeypatch):
    process_millers = []

    def fake_process(*args, **kwargs):
        miller_arg = np.asarray(args[0], dtype=np.float64)
        process_millers.append(miller_arg.copy())
        image_size = int(args[2])
        hit_tables = []
        for row in miller_arg:
            hit_tables.append(
                np.array(
                    [[1.0, 4.0, 4.0, 0.0, row[0], row[1], row[2]]],
                    dtype=np.float64,
                )
            )
        image = np.zeros((image_size, image_size), dtype=np.float64)
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 10
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([1.0, 2.0, 2.5, 3.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "2,0,0", "x": 4.0, "y": 4.0}]

    distances, *_ = opt.simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        params,
        measured,
    )

    assert distances.size > 0
    assert process_millers
    assert process_millers[0].shape == (2, 3)
    assert np.allclose(
        process_millers[0],
        np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
    )


def test_simulate_and_compare_hkl_keeps_hkl_fallback_when_source_indices_are_stale(
    monkeypatch,
):
    process_millers = []

    def fake_process(*args, **kwargs):
        miller_arg = np.asarray(args[0], dtype=np.float64)
        process_millers.append(miller_arg.copy())
        image_size = int(args[2])
        hit_tables = []
        coord_map = {
            (1, 0, 0): (1.0, 1.0),
            (2, 0, 0): (4.0, 4.0),
            (3, 0, 0): (7.0, 7.0),
        }
        for row in miller_arg:
            hkl = tuple(int(round(v)) for v in row)
            col, row_px = coord_map[hkl]
            hit_tables.append(
                np.array(
                    [[1.0, col, row_px, 0.0, row[0], row[1], row[2]]],
                    dtype=np.float64,
                )
            )
        image = np.zeros((image_size, image_size), dtype=np.float64)
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 12
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 1,
        }
    ]

    distances, sim_coords, meas_coords, sim_millers, meas_millers = opt.simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        params,
        measured,
    )

    assert process_millers
    assert process_millers[0].shape == (1, 3)
    assert np.allclose(
        process_millers[0],
        np.array([[2.0, 0.0, 0.0]], dtype=np.float64),
    )
    assert distances.size == 2
    assert sim_coords == [(4.0, 4.0)]
    assert meas_coords == [(4.0, 4.0)]
    assert sim_millers == [(2, 0, 0)]
    assert meas_millers == [(2, 0, 0)]


def test_simulate_and_compare_hkl_falls_back_when_in_range_source_indices_point_to_wrong_hkl(
    monkeypatch,
):
    process_millers = []

    def fake_process(*args, **kwargs):
        miller_arg = np.asarray(args[0], dtype=np.float64)
        process_millers.append(miller_arg.copy())
        image_size = int(args[2])
        hit_tables = []
        coord_map = {
            (1, 0, 0): (1.0, 1.0),
            (2, 0, 0): (4.0, 4.0),
            (3, 0, 0): (7.0, 7.0),
        }
        for row in miller_arg:
            hkl = tuple(int(round(v)) for v in row)
            col, row_px = coord_map[hkl]
            hit_tables.append(
                np.array(
                    [[1.0, col, row_px, 0.0, row[0], row[1], row[2]]],
                    dtype=np.float64,
                )
            )
        image = np.zeros((image_size, image_size), dtype=np.float64)
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 12
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
        }
    ]

    distances, sim_coords, meas_coords, sim_millers, meas_millers = opt.simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        params,
        measured,
    )

    assert process_millers
    assert process_millers[0].shape == (1, 3)
    assert np.allclose(
        process_millers[0],
        np.array([[2.0, 0.0, 0.0]], dtype=np.float64),
    )
    assert distances.size == 2
    assert sim_coords == [(4.0, 4.0)]
    assert meas_coords == [(4.0, 4.0)]
    assert sim_millers == [(2, 0, 0)]
    assert meas_millers == [(2, 0, 0)]


def test_resolve_fixed_source_matches_prefers_source_reflection_index() -> None:
    entry = {
        "hkl": (2, 0, 0),
        "label": "2,0,0",
        "x": 4.0,
        "y": 4.0,
        "source_table_index": 0,
        "source_reflection_index": 1,
        "resolved_table_index": 1,
        "source_row_index": 0,
    }
    hit_tables = [
        np.asarray([[1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64),
        np.asarray([[1.0, 4.0, 4.0, 0.0, 2.0, 0.0, 0.0]], dtype=np.float64),
    ]

    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        [entry],
        hit_tables,
    )

    assert len(resolved) == 1
    assert fallback_entries == []
    assert resolved[0][1] == (4.0, 4.0)
    assert resolved[0][2] == (2, 0, 0)
    assert resolution_lookup[id(entry)]["resolution_kind"] == "fixed_source"
    assert resolution_lookup[id(entry)]["resolution_reason"] == "resolved"


def test_resolve_fixed_source_matches_keeps_distinct_branches(monkeypatch) -> None:
    monkeypatch.setattr(
        opt,
        "hit_tables_to_max_positions",
        lambda hit_tables: np.asarray([[1.0, 2.0, 2.0, 1.0, 8.0, 8.0]], dtype=np.float64),
    )
    entries = [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0-left",
            "x": 2.0,
            "y": 2.0,
            "source_reflection_index": 0,
            "resolved_table_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_row_index": 0,
        },
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0-right",
            "x": 8.0,
            "y": 8.0,
            "source_reflection_index": 0,
            "resolved_table_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "source_row_index": 0,
        },
    ]
    hit_tables = [
        np.asarray(
            [
                [1.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 8.0, 8.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]

    resolved, fallback_entries, _resolution_lookup = opt._resolve_fixed_source_matches(
        entries,
        hit_tables,
    )

    assert fallback_entries == []
    assert [item[1] for item in resolved] == [(2.0, 2.0), (8.0, 8.0)]


def test_geometry_fit_correspondence_simulated_point_prefers_branch_identity() -> None:
    correspondence = {
        "source_reflection_index": 0,
        "resolved_table_index": 0,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    hit_tables = [
        np.asarray(
            [
                [1.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 8.0, 8.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]
    max_positions = np.asarray([[1.0, 2.0, 2.0, 1.0, 8.0, 8.0]], dtype=np.float64)

    point, reason = opt._geometry_fit_correspondence_simulated_point(
        correspondence,
        hit_tables=hit_tables,
        max_positions=max_positions,
    )

    assert point == (8.0, 8.0)
    assert reason == "resolved_source_peak"


def test_prepare_reflection_subset_clears_stale_local_source_ids() -> None:
    miller = np.array(
        [
            [2.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([2.0, 7.0], dtype=np.float64)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 13,
            "source_reflection_index": 13,
            "source_row_index": 0,
            "source_peak_index": 1,
            "fit_source_identity_only": True,
        }
    ]

    subset = opt._prepare_reflection_subset(miller, intensities, measured)

    assert subset.fixed_source_reflection_count == 0
    assert subset.fallback_hkl_count == 1
    assert np.array_equal(subset.original_indices, np.array([0], dtype=np.int64))
    for key in (
        "source_table_index",
        "source_reflection_index",
        "source_row_index",
        "source_peak_index",
        "resolved_table_index",
        "resolved_peak_index",
    ):
        assert key not in subset.measured_entries[0]


def test_geometry_fit_correspondence_simulated_point_ignores_stale_source_table_ids() -> None:
    correspondence = {
        "source_table_index": 13,
        "source_reflection_index": 13,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    hit_tables = [
        np.asarray(
            [
                [1.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 8.0, 8.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]
    max_positions = np.asarray([[1.0, 2.0, 2.0, 1.0, 8.0, 8.0]], dtype=np.float64)

    point, reason = opt._geometry_fit_correspondence_simulated_point(
        correspondence,
        hit_tables=hit_tables,
        max_positions=max_positions,
    )

    assert point is None
    assert reason == "missing_source_table_index"


def test_collect_geometry_fit_simulated_candidates_keeps_deadband_fallback_branchless(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        opt,
        "hit_tables_to_max_positions",
        lambda hit_tables: np.asarray(
            [[1.0, 150.0, 250.0, 0.0, 0.0, 0.0]], dtype=np.float64
        ),
    )

    candidates = opt._collect_geometry_fit_simulated_candidates(
        np.asarray([[1.0, 0.0, 3.0]], dtype=np.float64),
        [
            np.asarray(
                [[50.0, 150.0, 250.0, 5.0e-4, 1.0, 0.0, 3.0]],
                dtype=np.float64,
            )
        ],
        original_indices=np.asarray([7], dtype=np.int64),
    )

    candidate = candidates[(1, 0, 3)][0]
    assert candidate["source_reflection_index"] == 7
    assert "source_branch_index" not in candidate
    assert "source_peak_index" not in candidate
    assert "resolved_peak_index" not in candidate


def test_geometry_fit_correspondence_simulated_point_rejects_legacy_branch_alias_for_trusted_identity() -> None:
    point, reason = opt._geometry_fit_correspondence_simulated_point(
        {
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "resolved_table_index": 0,
            "source_row_index": 0,
            "source_peak_index": 1,
        },
        hit_tables=[
            np.asarray(
                [
                    [1.0, 2.0, 2.0, -22.0, 1.0, 0.0, 3.0],
                    [1.0, 8.0, 8.0, 22.0, 1.0, 0.0, 3.0],
                ],
                dtype=np.float64,
            )
        ],
        max_positions=np.asarray([[1.0, 2.0, 2.0, 1.0, 8.0, 8.0]], dtype=np.float64),
    )

    assert point is None
    assert reason == "missing_source_peak_index"


def test_trusted_identity_survives_fixed_source_bridge_and_seed_correspondence(
    monkeypatch,
) -> None:
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        miller_local = np.asarray(args[0], dtype=np.float64)
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = []
        for row in miller_local:
            h, k, l = (int(round(float(v))) for v in row[:3])
            if (h, k, l) == (1, 0, 0):
                hit_tables.append(
                    np.asarray(
                        [
                            [1.0, 2.0, 2.0, -10.0, 1.0, 0.0, 0.0],
                            [1.0, 8.0, 8.0, 10.0, 1.0, 0.0, 0.0],
                        ],
                        dtype=np.float64,
                    )
                )
            else:
                hit_tables.append(
                    np.asarray(
                        [[1.0, 50.0, 50.0, 0.0, float(h), float(k), float(l)]],
                        dtype=np.float64,
                    )
                )
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array(
        [[5.0, 0.0, 0.0], [1.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    intensities = np.array([5.0, 10.0, 7.0], dtype=np.float64)
    trusted_pair = {
        "pair_id": "bg0:pair0",
        "fit_run_id": "fit-123",
        "label": "1,0,0",
        "hkl": (1, 0, 0),
        "x": 8.0,
        "y": 8.0,
        "source_table_index": 99,
        "source_reflection_index": 1,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "sigma_px": 1.0,
    }

    subset = opt._prepare_reflection_subset(miller, intensities, [dict(trusted_pair)])
    assert subset.reduced is True
    assert len(subset.measured_entries) == 1

    subset_hit_tables = [
        np.asarray(
            [
                [1.0, 2.0, 2.0, -10.0, 1.0, 0.0, 0.0],
                [1.0, 8.0, 8.0, 10.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]
    full_hit_tables = [
        np.asarray([[1.0, 50.0, 50.0, 0.0, 5.0, 0.0, 0.0]], dtype=np.float64),
        np.asarray(
            [
                [1.0, 2.0, 2.0, -10.0, 1.0, 0.0, 0.0],
                [1.0, 8.0, 8.0, 10.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.asarray([[1.0, 50.0, 50.0, 0.0, 7.0, 0.0, 0.0]], dtype=np.float64),
    ]
    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        subset.measured_entries,
        subset_hit_tables,
    )

    assert len(resolved) == 1
    assert fallback_entries == []
    resolved_diag = resolution_lookup[id(subset.measured_entries[0])]

    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"] = {
        "beam_x_array": np.array([0.0, 0.1], dtype=np.float64),
        "beam_y_array": np.zeros(2, dtype=np.float64),
        "theta_array": np.zeros(2, dtype=np.float64),
        "phi_array": np.zeros(2, dtype=np.float64),
        "sigma_mosaic_deg": 0.2,
        "gamma_mosaic_deg": 0.1,
        "eta": 0.05,
        "wavelength_array": np.ones(2, dtype=np.float64),
    }
    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[dict(trusted_pair)],
        var_names=["gamma"],
        experimental_image=np.zeros((image_size, image_size), dtype=np.float64),
        refinement_config={
            "solver": {"restarts": 0, "weighted_matching": False},
            "full_beam_polish": {"enabled": True, "max_nfev": 10},
            "identifiability": {"enabled": False},
        },
    )

    assert result.success
    point_diag = dict(result.point_match_diagnostics[0])
    seed_record = dict(result.full_beam_polish_summary["seed_correspondence_records"][0])

    subset_entry = dict(subset.measured_entries[0])
    for field in (
        "pair_id",
        "fit_run_id",
        "hkl",
        "source_reflection_index",
        "source_reflection_namespace",
        "source_reflection_is_full",
        "source_branch_index",
        "source_peak_index",
    ):
        assert subset_entry.get(field) == trusted_pair.get(field)

    for record in (dict(resolved_diag), point_diag, seed_record):
        for field in (
            "pair_id",
            "fit_run_id",
            "hkl",
            "source_reflection_index",
            "source_reflection_namespace",
            "source_reflection_is_full",
            "source_branch_index",
            "source_peak_index",
        ):
            assert record.get(field) == trusted_pair.get(field)
        assert record.get("source_reflection_index_namespace") == "full_reflection"
        assert record.get("source_table_index_namespace") == "full_hit_table"
        assert record.get("source_row_index_namespace") == "full_hit_table"
        assert record.get("source_peak_index_namespace") == "branch_index"
        assert record.get("source_branch_index_namespace") == "branch_index"

    assert seed_record["frozen_locator_kind"] == "trusted_branch"
    assert seed_record["frozen_table_namespace"] == "full_reflection"


def test_measured_source_peak_index_with_source_forwards_legacy_fallback(
    monkeypatch,
) -> None:
    seen: list[bool] = []

    def fake_resolve(entry, *, allow_legacy_peak_fallback=False):
        seen.append(bool(allow_legacy_peak_fallback))
        return None, None, None

    monkeypatch.setattr(opt, "resolve_canonical_branch", fake_resolve)

    peak_idx, peak_source = opt._measured_source_peak_index_with_source(
        {"source_peak_index": 1},
        allow_legacy_peak_fallback=True,
    )

    assert seen == [True]
    assert peak_idx == 1
    assert peak_source == "source_peak_index"


def test_geometry_fit_correspondence_simulated_point_allows_untrusted_local_row_locator() -> None:
    point, reason = opt._geometry_fit_correspondence_simulated_point(
        {
            "resolved_table_index": 0,
            "source_row_index": 1,
            "hkl": (1, 0, 0),
        },
        hit_tables=[
            np.asarray(
                [
                    [1.0, 2.0, 2.0, -22.0, 1.0, 0.0, 0.0],
                    [1.0, 8.0, 8.0, 22.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        ],
        max_positions=np.asarray([[1.0, 2.0, 2.0, 1.0, 8.0, 8.0]], dtype=np.float64),
    )

    assert point == (8.0, 8.0)
    assert reason == "resolved_source_row"


def test_resolve_geometry_fit_correspondence_rejects_mismatched_local_table_signature() -> None:
    point, payload = opt._resolve_geometry_fit_correspondence(
        {
            "frozen_locator_kind": "local_row",
            "frozen_table_namespace": "current_full_local",
            "frozen_table_index": 0,
            "frozen_row_index": 1,
            "frozen_table_signature": "seed-signature",
        },
        hit_tables=[
            np.asarray(
                [
                    [1.0, 2.0, 2.0, -22.0, 1.0, 0.0, 0.0],
                    [1.0, 8.0, 8.0, 22.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        ],
        max_positions=np.asarray([[1.0, 2.0, 2.0, 1.0, 8.0, 8.0]], dtype=np.float64),
        current_local_table_signature="current-signature",
    )

    assert point is None
    assert payload["resolution_reason"] == "frozen_table_signature_mismatch"


def test_fit_geometry_parameters_pixel_path_falls_back_from_stale_in_range_source_indices(
    monkeypatch,
):
    process_millers = []

    def fake_process(*args, **kwargs):
        miller_arg = np.asarray(args[0], dtype=np.float64)
        process_millers.append(miller_arg.copy())
        image_size = int(args[2])
        hit_tables = []
        coord_map = {
            (1, 0, 0): (1.0, 1.0),
            (2, 0, 0): (4.0, 4.0),
            (3, 0, 0): (7.0, 7.0),
        }
        best_sample_indices_out = kwargs.get("best_sample_indices_out")
        if isinstance(best_sample_indices_out, np.ndarray):
            best_sample_indices_out[:] = 0
        for row in miller_arg:
            hkl = tuple(int(round(v)) for v in row)
            col, row_px = coord_map[hkl]
            hit_tables.append(
                np.array(
                    [[1.0, col, row_px, 0.0, row[0], row[1], row[2]]],
                    dtype=np.float64,
                )
            )
        image = np.zeros((image_size, image_size), dtype=np.float64)
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
        }
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={"solver": {"restarts": 0, "weighted_matching": False}},
    )

    assert result.success
    assert process_millers
    assert np.allclose(process_millers[0], np.array([[2.0, 0.0, 0.0]], dtype=np.float64))
    assert result.fun.size == 2
    assert np.allclose(result.fun, np.zeros(2, dtype=np.float64))
    assert isinstance(result.point_match_summary, dict)
    assert int(result.point_match_summary["fixed_source_resolved_count"]) == 0
    assert int(result.point_match_summary["fixed_source_reflection_count"]) == 0
    assert int(result.point_match_summary["fallback_entry_count"]) == 1
    assert int(result.point_match_summary["matched_pair_count"]) == 1
    assert isinstance(result.point_match_diagnostics, list)
    assert len(result.point_match_diagnostics) == 1
    assert result.point_match_diagnostics[0]["resolution_kind"] == "hkl_fallback"
    assert result.point_match_diagnostics[0]["source_table_index"] is None
    assert result.point_match_diagnostics[0].get("source_reflection_namespace") in (
        None,
        "",
    )
    assert result.point_match_diagnostics[0].get("source_reflection_is_full") in (
        None,
        False,
    )
    assert int(result.point_match_diagnostics[0]["resolved_table_index"]) == 0
    assert result.point_match_diagnostics[0]["match_status"] == "matched"


def test_fit_geometry_parameters_supports_center_component_variables(monkeypatch):
    target_row = 2.5
    target_col = 5.5
    centers_seen = []

    def fake_compute(*args, **kwargs):
        center_row = float(args[10])
        center_col = float(args[11])
        centers_seen.append((center_row, center_col))
        return np.array(
            [center_row - target_row, center_col - target_col],
            dtype=np.float64,
        )

    monkeypatch.setattr(
        opt, "compute_peak_position_error_geometry_local", fake_compute
    )

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[],
        var_names=["center_x", "center_y"],
        experimental_image=None,
    )

    assert result.success
    assert abs(float(result.x[0]) - target_row) < 1e-8
    assert abs(float(result.x[1]) - target_col) < 1e-8
    assert any(
        abs(row - target_row) < 1e-3 and abs(col - target_col) < 1e-3
        for row, col in centers_seen
    )


def test_build_measured_dict_skips_malformed_entries():
    measured = [
        {"label": "1,0,0", "x": 4.0, "y": 5.0},
        {"label": "bad-label", "x": 1.0, "y": 2.0},
        {"hkl": (2, 1, 0), "x": "6.0", "y": "7.0"},
        (3, 0, 0, 8.0, 9.0),
        {"label": "4,0,0", "x": np.nan, "y": 1.0},
        (1, 2),
    ]

    measured_dict = opt.build_measured_dict(measured)

    assert measured_dict == {
        (1, 0, 0): [(4.0, 5.0)],
        (2, 1, 0): [(6.0, 7.0)],
        (3, 0, 0): [(8.0, 9.0)],
    }


def test_fit_geometry_parameters_tolerates_bad_measured_labels(monkeypatch):
    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", _fake_process_peaks)

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {"label": "not,a,peak", "x": 2.0, "y": 2.0},
        {"label": "1,0,0", "x": 4.0, "y": 4.0},
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
    )

    assert result.success


def test_simulate_and_compare_hkl_preserves_fixed_source_row_assignments(monkeypatch):
    monkeypatch.setattr(
        opt,
        "_process_peaks_parallel_safe",
        _fake_process_peaks_same_hkl_two_hits,
    )

    image_size = 12
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 2.0,
            "y": 2.0,
            "overlay_match_index": 7,
            "source_table_index": 0,
            "source_row_index": 1,
            "fit_source_resolution_kind": "source_row",
        },
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 8.0,
            "y": 8.0,
            "overlay_match_index": 3,
            "source_table_index": 0,
            "source_row_index": 0,
            "fit_source_resolution_kind": "source_row",
        },
    ]

    distances, sim_coords, meas_coords, *_ = opt.simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        params,
        measured,
    )

    assert sim_coords == [(8.0, 8.0), (2.0, 2.0)]
    assert meas_coords == [(2.0, 2.0), (8.0, 8.0)]
    assert distances.size == 4
    assert np.max(distances) > 0.0


def test_fit_geometry_parameters_pixel_path_keeps_fixed_source_row_assignments(
    monkeypatch,
):
    monkeypatch.setattr(
        opt,
        "_process_peaks_parallel_safe",
        _fake_process_peaks_same_hkl_two_hits,
    )

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 2.0,
            "y": 2.0,
            "overlay_match_index": 7,
            "source_table_index": 0,
            "source_row_index": 1,
            "fit_source_resolution_kind": "source_row",
        },
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 8.0,
            "y": 8.0,
            "overlay_match_index": 3,
            "source_table_index": 0,
            "source_row_index": 0,
            "fit_source_resolution_kind": "source_row",
        },
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
            }
        },
    )

    assert result.success
    assert result.fun.size == 4
    assert np.max(np.abs(result.fun)) >= 5.0
    assert isinstance(result.point_match_summary, dict)
    assert int(result.point_match_summary["fixed_source_resolved_count"]) == 2
    assert isinstance(result.point_match_diagnostics, list)
    assert len(result.point_match_diagnostics) == 2
    assert all(
        entry["resolution_kind"] == "fixed_source"
        and entry["match_status"] == "matched"
        for entry in result.point_match_diagnostics
    )
    assert {
        str(entry["fit_source_resolution_kind"])
        for entry in result.point_match_diagnostics
    } == {"source_row"}
    assert {
        int(entry["overlay_match_index"]) for entry in result.point_match_diagnostics
    } == {3, 7}


def test_fit_geometry_parameters_pixel_path_probes_out_of_flat_start_region(
    monkeypatch,
):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        sim_col = 6.0 if gamma >= 0.25 else 2.0
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, sim_col, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="flat-local-solver",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 6.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
                "stagnation_probe": True,
                "stagnation_probe_fraction": 0.5,
            }
        },
    )

    assert result.x.shape == (1,)
    assert float(result.x[0]) >= 0.25
    assert np.allclose(np.asarray(result.fun, dtype=float), [0.0, 0.0])
    assert any(
        str(entry.get("seed_kind", "")) in {"axis", "global"}
        for entry in getattr(result, "restart_history", [])
        if str(entry.get("message", "")) == "prescore"
    )


def test_fit_geometry_parameters_multistart_keeps_trusted_parameters_fixed(
    monkeypatch,
):
    solve_starts = []

    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        Gamma = float(args[1])
        return np.array([gamma - 0.2, Gamma - 0.8], dtype=np.float64)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x0_arr = np.asarray(x0, dtype=float)
        solve_starts.append(x0_arr.copy())
        return opt.OptimizeResult(
            x=x0_arr,
            fun=np.asarray(residual_fn(x0_arr), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x0_arr, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "compute_peak_position_error_geometry_local", fake_compute)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[[1.0, 0.0, 0.0, 4.0, 4.0]],
        var_names=["gamma", "Gamma"],
        experimental_image=None,
        refinement_config={
            "bounds": {
                "gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
                "Gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
            },
            "priors": {"gamma": {"sigma": 0.05}},
            "seed_search": {
                "prescore_top_k": 6,
                "n_global": 4,
                "n_jitter": 2,
                "jitter_sigma_u": 0.5,
                "min_seed_separation_u": 0.1,
                "trusted_prior_fraction_of_span": 0.15,
            },
        },
    )

    assert result.success
    assert solve_starts
    assert all(np.isclose(float(start[0]), 0.0) for start in solve_starts)
    assert any(abs(float(start[1])) > 1.0e-9 for start in solve_starts)
    param_entries = result.geometry_fit_debug_summary["parameter_entries"]
    assert str(param_entries[0]["seed_group"]) == "trusted"
    assert str(param_entries[1]["seed_group"]) == "uncertain"


def test_fit_geometry_parameters_pixel_path_broad_restart_seed_escapes_far_coupled_minimum(
    monkeypatch,
):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        Gamma = float(args[9])
        sim_col = 6.0 if gamma >= 0.75 and Gamma >= 0.75 else 2.0
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, sim_col, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="flat-local-solver",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 6.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma", "Gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "bounds": {
                "gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
                "Gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
            },
            "solver": {
                "restarts": 4,
                "restart_jitter": 0.15,
                "weighted_matching": False,
                "stagnation_probe": False,
            },
        },
    )

    assert result.x.shape == (2,)
    assert float(result.x[0]) >= 0.75
    assert float(result.x[1]) >= 0.75
    assert np.allclose(np.asarray(result.fun, dtype=float), [0.0, 0.0])
    assert any(
        str(entry.get("seed_kind", "")) == "global"
        and str(entry.get("message", "")) == "prescore"
        and float(entry.get("cost", np.inf)) <= 1.0e-9
        for entry in getattr(result, "restart_history", [])
    )


def test_fit_geometry_parameters_records_prescore_and_local_seed_history(monkeypatch):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        Gamma = float(args[9])
        sim_col = 6.0 if gamma >= 0.75 and Gamma >= 0.75 else 2.0
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, sim_col, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="flat-local-solver",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 6.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma", "Gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "bounds": {
                "gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
                "Gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
            },
            "solver": {
                "restarts": 4,
                "restart_jitter": 0.15,
                "weighted_matching": False,
                "stagnation_probe": False,
                "workers": 2,
                "parallel_mode": "restarts",
                "worker_numba_threads": 2,
            },
        },
    )

    assert result.x.shape == (2,)
    assert float(result.x[0]) >= 0.75
    assert float(result.x[1]) >= 0.75
    assert any(
        str(entry.get("message", "")) == "prescore"
        for entry in getattr(result, "restart_history", [])
    )
    assert any(
        str(entry.get("message", "")) == "flat-local-solver"
        for entry in getattr(result, "restart_history", [])
    )


def test_fit_geometry_parameters_reports_unweighted_peak_rms(monkeypatch):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[50.0, 14.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": True,
                "f_scale_px": 1.0,
            }
        },
    )

    assert result.success
    raw_distance = float(result.point_match_diagnostics[0]["distance_px"])
    assert np.isclose(float(result.point_match_summary["matched_pair_count"]), 1.0)
    assert np.isclose(float(result.point_match_summary["unweighted_peak_rms_px"]), raw_distance)
    assert np.isclose(float(result.rms_px), raw_distance)
    assert str(result.point_match_summary["peak_weighting_mode"]) == "uniform"
    assert np.isfinite(float(result.weighted_residual_rms_px))
    assert float(result.rms_px) > float(result.weighted_residual_rms_px)


def test_fit_geometry_parameters_seed_status_reports_missing_pair_counts(monkeypatch):
    status_messages = []

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        return image, [], np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={"solver": {"restarts": 0, "stagnation_probe": False}},
        status_callback=status_messages.append,
    )

    assert result.success
    assert any("running normalized-u multistart solve" in msg for msg in status_messages)
    assert any("identity seed" in msg and "cost=" in msg for msg in status_messages)
    assert str(result.geometry_fit_debug_summary["main_solve_seed"]["seed_kind"]) == "u=0"
    assert int(result.point_match_summary["missing_pair_count"]) == 1


def test_fit_geometry_parameters_records_bound_proximity_summary(monkeypatch):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[50.0, 14.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(kwargs["bounds"][1], dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "bounds": {"gamma": {"mode": "absolute", "min": 0.0, "max": 1.0}},
            "solver": {"restarts": 0, "stagnation_probe": False},
        },
    )

    assert result.success
    assert result.bound_hits == ["gamma"]
    assert result.boundary_warning == (
        "Possible identifiability issue: parameters finished near bounds (gamma=upper)."
    )
    assert result.bound_proximity_summary == {
        "threshold_fraction": 0.01,
        "near_bound_parameters": [
            {
                "name": "gamma",
                "side": "upper",
                "value": 1.0,
                "bound": 1.0,
                "gap": 0.0,
                "span": 1.0,
                "gap_fraction": 0.0,
            }
        ],
    }


def test_full_beam_polish_rejects_match_count_regression(monkeypatch):
    solve_calls = []

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        beam_x_array = np.asarray(args[16], dtype=np.float64)
        image = np.zeros((image_size, image_size), dtype=np.float64)

        if beam_x_array.size > 1 and gamma >= 0.5:
            hit_tables = [
                np.array(
                    [[10.0, 4.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                np.empty((0, 7), dtype=np.float64),
            ]
        else:
            hit_tables = [
                np.array(
                    [[10.0, 5.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array(
                    [[10.0, 9.0, 8.0, 0.0, 0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
            ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        call_index = len(solve_calls)
        x = np.array([0.0], dtype=float) if call_index == 0 else np.array([1.0], dtype=float)
        solve_calls.append(x.copy())
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0, 20.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"] = {
        "beam_x_array": np.array([0.0, 0.1], dtype=np.float64),
        "beam_y_array": np.zeros(2, dtype=np.float64),
        "theta_array": np.zeros(2, dtype=np.float64),
        "phi_array": np.zeros(2, dtype=np.float64),
        "sigma_mosaic_deg": 0.2,
        "gamma_mosaic_deg": 0.1,
        "eta": 0.05,
        "wavelength_array": np.ones(2, dtype=np.float64),
    }
    measured = [
        {
            "label": "1,0,0",
            "hkl": (1, 0, 0),
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "sigma_px": 1.0,
        },
        {
            "label": "0,1,0",
            "hkl": (0, 1, 0),
            "x": 8.0,
            "y": 8.0,
            "source_table_index": 1,
            "source_row_index": 0,
            "sigma_px": 1000.0,
        },
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "loss": "linear",
                "weighted_matching": False,
                "use_measurement_uncertainty": True,
                "stagnation_probe": False,
            },
            "full_beam_polish": {"enabled": True, "max_nfev": 10},
            "identifiability": {"enabled": False},
        },
    )

    assert result.success
    assert len(solve_calls) >= 2
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([0.0]))
    assert isinstance(result.full_beam_polish_summary, dict)
    assert bool(result.full_beam_polish_summary["accepted"]) is False
    assert str(result.full_beam_polish_summary["status"]) == "rejected"
    assert "matched_pairs_decreased" in str(result.full_beam_polish_summary["reason"])
    assert float(result.full_beam_polish_summary["candidate_cost"]) < float(
        result.full_beam_polish_summary["start_cost"]
    )
    assert int(result.full_beam_polish_summary["matched_pair_count_before"]) == 2
    assert int(result.full_beam_polish_summary["candidate_matched_pair_count"]) == 1
    assert int(result.point_match_summary["matched_pair_count"]) == 2


def test_full_beam_polish_rejects_unweighted_rms_regression(monkeypatch):
    solve_calls = []

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        beam_x_array = np.asarray(args[16], dtype=np.float64)
        image = np.zeros((image_size, image_size), dtype=np.float64)

        if beam_x_array.size > 1 and gamma >= 0.5:
            hit_tables = [
                np.array(
                    [[10.0, 4.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array(
                    [[10.0, 12.0, 8.0, 0.0, 0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
            ]
        else:
            hit_tables = [
                np.array(
                    [[10.0, 5.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array(
                    [[10.0, 9.0, 8.0, 0.0, 0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
            ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        call_index = len(solve_calls)
        x = np.array([0.0], dtype=float) if call_index == 0 else np.array([1.0], dtype=float)
        solve_calls.append(x.copy())
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0, 20.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"] = {
        "beam_x_array": np.array([0.0, 0.1], dtype=np.float64),
        "beam_y_array": np.zeros(2, dtype=np.float64),
        "theta_array": np.zeros(2, dtype=np.float64),
        "phi_array": np.zeros(2, dtype=np.float64),
        "sigma_mosaic_deg": 0.2,
        "gamma_mosaic_deg": 0.1,
        "eta": 0.05,
        "wavelength_array": np.ones(2, dtype=np.float64),
    }
    measured = [
        {
            "label": "1,0,0",
            "hkl": (1, 0, 0),
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "sigma_px": 1.0,
        },
        {
            "label": "0,1,0",
            "hkl": (0, 1, 0),
            "x": 8.0,
            "y": 8.0,
            "source_table_index": 1,
            "source_row_index": 0,
            "sigma_px": 1000.0,
        },
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "loss": "linear",
                "weighted_matching": False,
                "use_measurement_uncertainty": True,
                "stagnation_probe": False,
            },
            "full_beam_polish": {"enabled": True, "max_nfev": 10},
            "identifiability": {"enabled": False},
        },
    )

    assert result.success
    assert len(solve_calls) >= 2
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([0.0]))
    assert isinstance(result.full_beam_polish_summary, dict)
    assert bool(result.full_beam_polish_summary["accepted"]) is False
    assert str(result.full_beam_polish_summary["status"]) == "rejected"
    assert "point_rms_regressed" in str(result.full_beam_polish_summary["reason"])
    assert "peak_offset_regressed" in str(result.full_beam_polish_summary["reason"])
    assert float(result.full_beam_polish_summary["candidate_cost"]) < float(
        result.full_beam_polish_summary["start_cost"]
    )
    assert np.isclose(float(result.full_beam_polish_summary["start_rms_px"]), 1.0)
    assert np.isclose(float(result.full_beam_polish_summary["candidate_rms_px"]), np.sqrt(8.0))
    assert int(result.point_match_summary["matched_pair_count"]) == 2


def test_full_beam_polish_rejection_preserves_central_point_match_result(monkeypatch):
    def fake_process(*args, **kwargs):
        miller_arg = np.asarray(args[0], dtype=np.float64)
        image_size = int(args[2])
        hit_tables = []
        coord_map = {
            (1, 0, 0): (1.0, 1.0),
            (2, 0, 0): (4.0, 4.0),
            (3, 0, 0): (7.0, 7.0),
        }
        best_sample_indices_out = kwargs.get("best_sample_indices_out")
        if isinstance(best_sample_indices_out, np.ndarray):
            best_sample_indices_out[:] = 0
        for row in miller_arg:
            hkl = tuple(int(round(v)) for v in row)
            col, row_px = coord_map[hkl]
            hit_tables.append(
                np.array(
                    [[1.0, col, row_px, 0.0, row[0], row[1], row[2]]],
                    dtype=np.float64,
                )
            )
        image = np.zeros((image_size, image_size), dtype=np.float64)
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
        }
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    baseline = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {"restarts": 0, "weighted_matching": False},
            "full_beam_polish": {"enabled": False},
            "identifiability": {"enabled": False},
        },
    )
    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {"restarts": 0, "weighted_matching": False},
            "full_beam_polish": {"enabled": True, "max_nfev": 10},
            "identifiability": {"enabled": False},
        },
    )

    assert baseline.success
    assert result.success
    assert result.final_metric_name == baseline.final_metric_name == "central_point_match"
    assert np.allclose(result.fun, baseline.fun)
    assert result.point_match_summary["matched_pair_count"] == baseline.point_match_summary[
        "matched_pair_count"
    ]
    assert np.isclose(
        float(result.point_match_summary["unweighted_peak_rms_px"]),
        float(baseline.point_match_summary["unweighted_peak_rms_px"]),
        equal_nan=True,
    )
    assert len(result.point_match_diagnostics) == len(baseline.point_match_diagnostics)
    for result_entry, baseline_entry in zip(
        result.point_match_diagnostics,
        baseline.point_match_diagnostics,
    ):
        for key in (
            "match_status",
            "match_kind",
            "resolution_reason",
            "resolution_kind",
            "source_table_index",
            "source_row_index",
            "resolved_table_index",
            "resolved_peak_index",
            "source_branch_index",
        ):
            assert result_entry.get(key) == baseline_entry.get(key)
        for key in (
            "measured_x",
            "measured_y",
            "simulated_x",
            "simulated_y",
            "dx_px",
            "dy_px",
            "distance_px",
        ):
            assert np.isclose(
                float(result_entry.get(key, np.nan)),
                float(baseline_entry.get(key, np.nan)),
                equal_nan=True,
            )
    assert int(result.point_match_summary["matched_pair_count"]) == 1
    assert int(result.point_match_summary["matched_pair_count"]) == int(
        baseline.point_match_summary["matched_pair_count"]
    )
    assert isinstance(result.full_beam_polish_summary, dict)
    assert bool(result.full_beam_polish_summary["accepted"]) is False
    assert str(result.full_beam_polish_summary["reason"]) == "no_seed_correspondences"


def test_fit_geometry_parameters_uses_manual_peak_sigma_by_default(monkeypatch):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[50.0, 14.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "label": "1,0,0",
            "x": 4.0,
            "y": 4.0,
            "sigma_px": 5.0,
            "placement_error_px": 4.5,
            "source_table_index": 0,
            "source_row_index": 0,
        }
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
                "f_scale_px": 1.0,
            }
        },
    )

    assert result.success
    assert np.allclose(np.asarray(result.fun, dtype=float), [2.0, 0.0])
    diag = result.point_match_diagnostics[0]
    assert float(diag["distance_px"]) == 10.0
    assert float(diag["measurement_sigma_px"]) == 5.0
    assert np.isclose(float(diag["sigma_weight"]), 0.2)
    assert np.isclose(float(diag["weight"]), 0.2)
    assert np.isclose(float(diag["weighted_dx_px"]), 2.0)
    assert np.isclose(float(diag["placement_error_px"]), 4.5)
    assert str(result.point_match_summary["peak_weighting_mode"]) == "measurement_sigma"
    assert int(result.point_match_summary["custom_sigma_count"]) == 1
    assert np.isclose(float(result.point_match_summary["measurement_sigma_median_px"]), 5.0)
    assert np.isclose(float(result.rms_px), 10.0)
    assert float(result.weighted_residual_rms_px) < float(result.rms_px)


def test_fit_geometry_parameters_can_ignore_manual_peak_sigma_when_disabled(monkeypatch):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[50.0, 14.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "label": "1,0,0",
            "x": 4.0,
            "y": 4.0,
            "sigma_px": 5.0,
            "placement_error_px": 4.5,
            "source_table_index": 0,
            "source_row_index": 0,
        }
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
                "use_measurement_uncertainty": False,
                "f_scale_px": 1.0,
            }
        },
    )

    assert result.success
    assert np.allclose(np.asarray(result.fun, dtype=float), [10.0, 0.0])
    diag = result.point_match_diagnostics[0]
    assert float(diag["distance_px"]) == 10.0
    assert float(diag["measurement_sigma_px"]) == 5.0
    assert np.isclose(float(diag["sigma_weight"]), 1.0)
    assert np.isclose(float(diag["weight"]), 1.0)
    assert np.isclose(float(diag["weighted_dx_px"]), 10.0)
    assert np.isclose(float(diag["placement_error_px"]), 4.5)
    assert str(result.point_match_summary["peak_weighting_mode"]) == "uniform"
    assert int(result.point_match_summary["custom_sigma_count"]) == 0
    assert np.isclose(float(result.rms_px), 10.0)
    assert np.isclose(
        float(result.weighted_residual_rms_px),
        np.sqrt((10.0 ** 2 + 0.0 ** 2) / 2.0),
    )


def test_fit_geometry_parameters_joint_backgrounds_share_theta_offset(monkeypatch):
    target_offset = 0.75

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        theta_initial = float(args[27])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, theta_initial, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        best_sample_indices_out = kwargs.get("best_sample_indices_out")
        if isinstance(best_sample_indices_out, np.ndarray):
            best_sample_indices_out[:] = 0
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["theta_offset"] = 0.0

    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)
    dataset_specs = [
        {
            "dataset_index": 0,
            "label": "bg0",
            "theta_initial": 3.0,
            "measured_peaks": [
                {"label": "1,0,0", "x": 3.0 + target_offset, "y": 4.0}
            ],
            "experimental_image": experimental_image,
        },
        {
            "dataset_index": 1,
            "label": "bg1",
            "theta_initial": 7.0,
            "measured_peaks": [
                {"label": "1,0,0", "x": 7.0 + target_offset, "y": 4.0}
            ],
            "experimental_image": experimental_image,
        },
    ]

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=dataset_specs[0]["measured_peaks"],
        var_names=["theta_offset"],
        experimental_image=experimental_image,
        dataset_specs=dataset_specs,
        refinement_config={
            "solver": {"restarts": 0, "weighted_matching": False, "max_nfev": 40},
            "single_ray": {"enabled": False},
        },
    )

    assert result.success
    assert result.x.shape == (1,)
    assert abs(float(result.x[0]) - target_offset) < 1e-6
    assert np.allclose(np.asarray(result.fun, dtype=float), 0.0, atol=1e-6)
    assert int(result.point_match_summary["dataset_count"]) == 2
    assert int(result.point_match_summary["matched_pair_count"]) == 2
    assert len(result.point_match_summary["per_dataset"]) == 2
    assert len(result.point_match_diagnostics) == 2
    assert {
        int(entry["dataset_index"]) for entry in result.point_match_diagnostics
    } == {0, 1}


def test_fit_geometry_parameters_accepts_numpy_dataset_specs(monkeypatch):
    target_offset = 0.5

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        theta_initial = float(args[27])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, theta_initial, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["theta_offset"] = 0.0

    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)
    dataset_specs = np.array(
        [
            {
                "dataset_index": 0,
                "label": "bg0",
                "theta_initial": 3.0,
                "measured_peaks": [
                    {"label": "1,0,0", "x": 3.0 + target_offset, "y": 4.0}
                ],
                "experimental_image": experimental_image,
            },
            {
                "dataset_index": 1,
                "label": "bg1",
                "theta_initial": 7.0,
                "measured_peaks": [
                    {"label": "1,0,0", "x": 7.0 + target_offset, "y": 4.0}
                ],
                "experimental_image": experimental_image,
            },
        ],
        dtype=object,
    )

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=dataset_specs[0]["measured_peaks"],
        var_names=["theta_offset"],
        experimental_image=experimental_image,
        dataset_specs=dataset_specs,
        refinement_config={
            "solver": {"restarts": 0, "weighted_matching": False, "max_nfev": 40},
            "single_ray": {"enabled": False},
        },
    )

    assert result.success
    assert result.x.shape == (1,)
    assert abs(float(result.x[0]) - target_offset) < 1e-6
    assert int(result.point_match_summary["dataset_count"]) == 2


def test_fit_geometry_parameters_parallelizes_multi_dataset_point_matching(monkeypatch):
    threaded_calls = []

    def fake_threaded_map(fn, items, *, max_workers, numba_threads=None):
        item_list = list(items)
        threaded_calls.append(
            {
                "max_workers": int(max_workers),
                "numba_threads": numba_threads,
                "count": len(item_list),
            }
        )
        return [fn(item) for item in item_list]

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        theta_initial = float(args[27])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, theta_initial, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    monkeypatch.setattr(opt, "_threaded_map", fake_threaded_map)
    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["theta_offset"] = 0.0

    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)
    dataset_specs = [
        {
            "dataset_index": 0,
            "label": "bg0",
            "theta_initial": 3.0,
            "measured_peaks": [{"label": "1,0,0", "x": 3.5, "y": 4.0}],
            "experimental_image": experimental_image,
        },
        {
            "dataset_index": 1,
            "label": "bg1",
            "theta_initial": 7.0,
            "measured_peaks": [{"label": "1,0,0", "x": 7.5, "y": 4.0}],
            "experimental_image": experimental_image,
        },
    ]

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=dataset_specs[0]["measured_peaks"],
        var_names=["theta_offset"],
        experimental_image=experimental_image,
        dataset_specs=dataset_specs,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
                "max_nfev": 40,
                "workers": 2,
                "parallel_mode": "datasets",
                "worker_numba_threads": 3,
            },
            "single_ray": {"enabled": False},
        },
    )

    assert result.success
    assert threaded_calls
    assert any(call["max_workers"] == 2 and call["count"] == 2 for call in threaded_calls)
    assert any(call["numba_threads"] == 3 for call in threaded_calls)
    assert result.parallelization_summary["dataset_workers"] == 2
    assert result.parallelization_summary["restart_workers"] == 1


def test_compute_sensitivity_weights_can_equalize_roi_totals():
    image_size = 31
    params = _base_params(image_size, optics_mode=1)
    params["gamma"] = 0.0
    params["pixel_size"] = 0.01
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    max_positions = np.array(
        [
            [1.0, 8.0, 15.0, np.nan, np.nan, np.nan],
            [25.0, 22.0, 15.0, np.nan, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    yy, xx = np.mgrid[0:image_size, 0:image_size]

    def simulator(local_params):
        shift = float(local_params.get("gamma", 0.0))
        image = (
            np.exp(
                -(
                    (xx - (8.0 + shift)) ** 2
                    + (yy - 15.0) ** 2
                )
                / (2.0 * 0.8 ** 2)
            )
            + 25.0
            * np.exp(
                -(
                    (xx - (22.0 + shift)) ** 2
                    + (yy - 15.0) ** 2
                )
                / (2.0 * 0.8 ** 2)
            )
        )
        return image.astype(np.float64), max_positions

    base_sim, _ = simulator(params)
    rois_raw = opt.build_tube_rois(
        miller,
        max_positions,
        params,
        image_size,
        base_width=3.0,
    )
    opt.compute_sensitivity_weights(
        base_sim,
        params,
        ["gamma"],
        rois_raw,
        simulator,
        downsample_factor=1,
        percentile=80.0,
        huber_percentile=100.0,
        per_reflection_quota=25,
        off_tube_fraction=0.0,
        normalize_per_roi=False,
    )
    raw_sums = [float(np.sum(np.asarray(roi.weights, dtype=float))) for roi in rois_raw]
    assert raw_sums[1] > raw_sums[0] * 2.0

    rois_equal = opt.build_tube_rois(
        miller,
        max_positions,
        params,
        image_size,
        base_width=3.0,
    )
    opt.compute_sensitivity_weights(
        base_sim,
        params,
        ["gamma"],
        rois_equal,
        simulator,
        downsample_factor=1,
        percentile=80.0,
        huber_percentile=100.0,
        per_reflection_quota=25,
        off_tube_fraction=0.0,
        normalize_per_roi=True,
    )
    equal_sums = [float(np.sum(np.asarray(roi.weights, dtype=float))) for roi in rois_equal]
    assert np.isclose(equal_sums[0], 1.0)
    assert np.isclose(equal_sums[1], 1.0)


def test_fit_geometry_parameters_can_accept_roi_image_refinement(monkeypatch):
    captured_cfg = {}

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    def fake_stage_two(
        experimental_image,
        miller,
        intensities,
        image_size,
        params,
        var_names,
        simulator,
        measured_dict,
        *,
        cfg,
    ):
        captured_cfg.update(cfg)
        updated_params = dict(params)
        updated_params["gamma"] = 0.25
        updated_params["center"] = list(params.get("center", [image_size / 2.0, image_size / 2.0]))
        updated_params["center_x"] = float(updated_params["center"][0])
        updated_params["center_y"] = float(updated_params["center"][1])
        stage_result = opt.OptimizeResult(
            x=np.array([0.25], dtype=float),
            fun=np.zeros(1, dtype=float),
            success=True,
            status=1,
            message="roi-refine-ok",
            nfev=2,
            active_mask=np.zeros(1, dtype=int),
            optimality=0.0,
        )
        stage_result.initial_cost = 12.0
        stage_result.final_cost = 2.0
        return updated_params, stage_result, [object(), object(), object()], np.zeros((image_size, image_size), dtype=float), lambda x: np.zeros(1, dtype=float)

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", _fake_process_three_reflections)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)
    monkeypatch.setattr(opt, "_stage_two_refinement", fake_stage_two)

    image_size = 20
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([5.0, 4.0, 3.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {"label": "1,0,0", "x": 4.0, "y": 4.0},
        {"label": "0,1,0", "x": 10.0, "y": 10.0},
        {"label": "0,0,1", "x": 14.0, "y": 14.0},
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {"restarts": 0, "weighted_matching": False},
            "image_refinement": {"enabled": True, "min_rois": 3},
        },
    )

    assert result.success
    assert np.isclose(float(result.x[0]), 0.25)
    assert isinstance(result.image_refinement_summary, dict)
    assert bool(result.image_refinement_summary["accepted"]) is True
    assert str(result.image_refinement_summary["status"]) == "accepted"
    assert bool(captured_cfg["equal_peak_weights"]) is True
    assert int(result.image_refinement_summary["selected_roi_count"]) == 3
    assert "ROI/image refinement accepted" in str(result.message)
    assert int(result.point_match_summary["matched_pair_count"]) == 3


def test_fit_geometry_parameters_defaults_to_point_only_fit_without_image_stages(
    monkeypatch,
):
    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    def fail_stage_one(*args, **kwargs):
        raise AssertionError("ridge refinement should be disabled by default")

    def fail_stage_two(*args, **kwargs):
        raise AssertionError("image refinement should be disabled by default")

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", _fake_process_peaks)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)
    monkeypatch.setattr(opt, "_stage_one_initialize", fail_stage_one)
    monkeypatch.setattr(opt, "_stage_two_refinement", fail_stage_two)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([10.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={"solver": {"restarts": 0, "weighted_matching": False}},
    )

    assert result.success
    assert isinstance(result.ridge_refinement_summary, dict)
    assert bool(result.ridge_refinement_summary["enabled"]) is False
    assert str(result.ridge_refinement_summary["reason"]) == "disabled_by_config"
    assert isinstance(result.image_refinement_summary, dict)
    assert bool(result.image_refinement_summary["enabled"]) is False
    assert str(result.image_refinement_summary["reason"]) == "disabled_by_config"


def test_fit_geometry_parameters_manual_point_fit_mode_uses_lean_defaults(
    monkeypatch,
):
    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", _fake_process_peaks)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([10.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={"solver": {"manual_point_fit_mode": True}},
    )

    assert result.success
    assert isinstance(result.geometry_fit_debug_summary, dict)
    assert isinstance(result.geometry_fit_debug_summary["solver"], dict)
    solver_debug = result.geometry_fit_debug_summary["solver"]
    assert bool(solver_debug["manual_point_fit_mode"]) is True
    assert np.isclose(float(solver_debug["f_scale_px"]), 1.0)
    assert bool(solver_debug["q_group_line_constraints"]) is True
    assert np.isclose(float(solver_debug["hk0_peak_priority_weight"]), 6.0)
    assert int(solver_debug["restarts"]) >= 1
    assert bool(solver_debug["use_measurement_uncertainty"]) is False
    assert bool(solver_debug["full_beam_polish_enabled"]) is False
    assert isinstance(result.full_beam_polish_summary, dict)
    assert bool(result.full_beam_polish_summary["enabled"]) is False
    assert str(result.full_beam_polish_summary["reason"]) == "disabled_by_config"


def test_fit_geometry_parameters_manual_point_fit_adds_q_group_line_residuals(
    monkeypatch,
):
    captured: dict[str, np.ndarray] = {}

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 4.0, 5.0, 0.0, 1.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            np.array(
                [[1.0, 8.0, 9.0, 0.0, -1.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        residual = np.asarray(residual_fn(x), dtype=float)
        captured["residual"] = residual.copy()
        return opt.OptimizeResult(
            x=x,
            fun=residual,
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 32
    miller = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0, 1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["debye_x"] = 1.0
    params["debye_y"] = 1.0
    measured = [
        {
            "hkl": (1, 1, 0),
            "x": 4.0,
            "y": 5.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "q_group_key": ("q_group", "primary", 1, 0),
        },
        {
            "hkl": (-1, 1, 0),
            "x": 8.0,
            "y": 9.0,
            "source_table_index": 1,
            "source_row_index": 0,
            "q_group_key": ("q_group", "primary", 1, 0),
        },
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={"solver": {"manual_point_fit_mode": True}},
    )

    assert result.success
    assert captured["residual"].shape == (6,)
    assert np.allclose(captured["residual"], 0.0, atol=1.0e-6)
    assert int(result.point_match_summary["line_group_count"]) == 1
    assert int(result.point_match_summary["resolved_line_group_count"]) == 1
    assert int(result.point_match_summary["missing_line_group_count"]) == 0


def test_fit_geometry_parameters_hk0_peaks_receive_priority_weight(monkeypatch):
    captured: dict[str, np.ndarray] = {}

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 5.0, 5.0, 0.0, 0.0, 0.0, 2.0]],
                dtype=np.float64,
            ),
            np.array(
                [[1.0, 9.0, 9.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        residual = np.asarray(residual_fn(x), dtype=float)
        captured["residual"] = residual.copy()
        return opt.OptimizeResult(
            x=x,
            fun=residual,
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 24
    miller = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([10.0, 10.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (0, 0, 2),
            "x": 4.0,
            "y": 5.0,
            "source_table_index": 0,
            "source_row_index": 0,
        },
        {
            "hkl": (1, 0, 0),
            "x": 8.0,
            "y": 9.0,
            "source_table_index": 1,
            "source_row_index": 0,
        },
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
                "hk0_peak_priority_weight": 5.0,
            }
        },
    )

    assert result.success
    assert np.allclose(captured["residual"], [5.0, 0.0, 1.0, 0.0])
    diag_by_hkl = {
        tuple(diag["hkl"]): diag
        for diag in result.point_match_diagnostics
        if isinstance(diag.get("hkl"), tuple)
    }
    assert np.isclose(float(diag_by_hkl[(0, 0, 2)]["priority_weight"]), 5.0)
    assert str(diag_by_hkl[(0, 0, 2)]["priority_class"]) == "hk0"
    assert np.isclose(float(diag_by_hkl[(1, 0, 0)]["priority_weight"]), 1.0)
    assert str(diag_by_hkl[(1, 0, 0)]["priority_class"]) == "default"


def test_fit_geometry_parameters_manual_point_fit_guardrail_aborts_bound_hugging_bad_seed(
    monkeypatch,
):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[10.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x_bad = np.asarray(kwargs["bounds"][1], dtype=float)
        for _ in range(40):
            residual_fn(x_bad)
        raise AssertionError("manual fail-fast guardrail did not abort the solve")

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 1024
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 400.0, "y": 400.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma", "Gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "bounds": {
                "gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
                "Gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
            },
            "solver": {
                "manual_point_fit_mode": True,
            },
        },
    )

    assert not result.success
    assert "Manual point-fit guardrail stopped the solve" in str(result.message)
    assert "gamma, Gamma" in str(result.message)
    assert str(result.early_stop_reason).startswith(
        "Manual point-fit guardrail stopped the solve"
    )
    assert bool(result.geometry_fit_progress["aborted_early"]) is True
    assert float(result.point_match_summary["unweighted_peak_rms_px"]) > 250.0
    assert sorted(result.bound_hits) == ["Gamma", "gamma"]


def test_fit_geometry_parameters_can_accept_ridge_refinement(monkeypatch):
    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    def fake_stage_one_initialize(
        experimental_image,
        params,
        var_names,
        simulator,
        *,
        downsample_factor,
        max_nfev,
        bounds=None,
        x_scale=None,
    ):
        updated_params = dict(params)
        updated_params["gamma"] = 0.125
        stage_result = opt.OptimizeResult(
            x=np.array([0.125], dtype=float),
            fun=np.zeros(1, dtype=float),
            success=True,
            status=1,
            message="ridge-refine-ok",
            nfev=2,
            active_mask=np.zeros(1, dtype=int),
            optimality=0.0,
        )
        stage_result.initial_cost = 5.0
        stage_result.final_cost = 1.0
        stage_result.cost_reduction = 0.8
        return updated_params, stage_result

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", _fake_process_peaks)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)
    monkeypatch.setattr(opt, "_stage_one_initialize", fake_stage_one_initialize)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([10.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {"restarts": 0, "weighted_matching": False},
            "ridge_refinement": {"enabled": True},
            "image_refinement": {"enabled": False},
        },
    )

    assert result.success
    assert np.isclose(float(result.x[0]), 0.125)
    assert isinstance(result.ridge_refinement_summary, dict)
    assert bool(result.ridge_refinement_summary["accepted"]) is True
    assert str(result.ridge_refinement_summary["status"]) == "accepted"
    assert "Ridge refinement accepted" in str(result.message)


def test_fit_geometry_parameters_supports_anisotropic_measurement_weighting(monkeypatch):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[20.0, 14.0, 18.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "label": "1,0,0",
            "x": 14.0,
            "y": 10.0,
            "sigma_px": 2.0,
        }
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
                "use_measurement_uncertainty": True,
                "anisotropic_measurement_uncertainty": True,
                "radial_sigma_scale": 1.0,
                "tangential_sigma_scale": 4.0,
            },
            "ridge_refinement": {"enabled": False},
            "image_refinement": {"enabled": False},
        },
    )

    assert result.success
    assert np.allclose(np.asarray(result.fun, dtype=float), [0.0, 1.0])
    diag = result.point_match_diagnostics[0]
    assert np.isclose(float(diag["sigma_radial_px"]), 2.0)
    assert np.isclose(float(diag["sigma_tangential_px"]), 8.0)
    assert np.isclose(float(diag["weighted_dx_px"]), 0.0)
    assert np.isclose(float(diag["weighted_dy_px"]), 1.0)
    assert np.isclose(float(diag["weighted_tangential_residual_px"]), 1.0)
    assert bool(diag["anisotropic_sigma_used"]) is True
    assert str(result.point_match_summary["peak_weighting_mode"]) == "measurement_covariance"
    assert int(result.point_match_summary["anisotropic_sigma_count"]) == 1
    assert np.isclose(float(result.rms_px), 8.0)
    assert float(result.weighted_residual_rms_px) < float(result.rms_px)


def test_fit_geometry_parameters_reports_solver_and_data_only_identifiability(
    monkeypatch,
):
    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        return np.array([gamma - 1.0], dtype=np.float64)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "compute_peak_position_error_geometry_local", fake_compute)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)
    params["gamma"] = 1.0

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[[1.0, 0.0, 0.0, 4.0, 4.0]],
        var_names=["gamma", "Gamma"],
        experimental_image=None,
        refinement_config={
            "solver": {"restarts": 0},
            "identifiability": {"enabled": True},
        },
    )

    assert result.success

    solver_summary = result.identifiability_summary
    assert isinstance(solver_summary, dict)
    assert str(solver_summary["status"]) == "ok"
    assert str(solver_summary["diagnostic_scope"]) == "solver_conditioned_active"
    assert bool(solver_summary["includes_priors"]) is True
    assert int(solver_summary["num_parameters"]) == 2
    assert int(solver_summary["rank"]) == 1
    assert bool(solver_summary["underconstrained"]) is True
    solver_entries = solver_summary["parameter_entries"]
    assert [entry["name"] for entry in solver_entries] == ["gamma", "Gamma"]
    assert float(solver_entries[0]["column_norm"]) > 0.0
    assert np.isclose(float(solver_entries[1]["column_norm"]), 0.0)

    data_summary = result.data_only_identifiability_summary
    assert isinstance(data_summary, dict)
    assert str(data_summary["status"]) == "ok"
    assert str(data_summary["diagnostic_scope"]) == "data_only_all_selectable"
    assert bool(data_summary["includes_priors"]) is False
    assert int(data_summary["rank"]) == 1
    assert bool(data_summary["underconstrained"]) is True
    weak_parameters = data_summary["weak_parameters"]
    assert len(weak_parameters) == 1
    assert str(weak_parameters[0]["name"]) == "Gamma"
    assert list(result.next_stage_recommendations) == []
    assert result.next_stage_recommendation is None


def test_fit_geometry_parameters_correlated_inactive_block_is_recommended_together(
    monkeypatch,
):
    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        Gamma = float(args[1])
        coupled = gamma + Gamma - 1.0
        return np.array([coupled, 2.0 * coupled], dtype=np.float64)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "compute_peak_position_error_geometry_local", fake_compute)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[[1.0, 0.0, 0.0, 4.0, 4.0]],
        var_names=[],
        candidate_param_names=["gamma", "Gamma"],
        experimental_image=None,
        refinement_config={
            "solver": {"restarts": 0},
            "identifiability": {"enabled": True},
        },
    )

    assert bool(result.success) is False
    assert np.isclose(float(result.cost), 2.5)
    assert str(result.message) == "fixed-parameter mode identity"
    assert str(result.identifiability_summary["status"]) == "failed"
    assert str(result.identifiability_summary["reason"]) == "empty_parameter_vector"

    high_pairs = result.data_only_identifiability_summary["high_correlation_pairs"]
    assert len(high_pairs) == 1
    assert {
        str(high_pairs[0]["name_i"]),
        str(high_pairs[0]["name_j"]),
    } == {"gamma", "Gamma"}
    assert float(high_pairs[0]["abs_correlation"]) > 0.99

    recommendations = result.next_stage_recommendations
    assert len(recommendations) == 1
    assert set(recommendations[0]["params"]) == {"gamma", "Gamma"}
    assert "Correlated block" in str(recommendations[0]["reason"])
    assert isinstance(result.next_stage_recommendation, dict)
    assert set(result.next_stage_recommendation["params"]) == {"gamma", "Gamma"}


def test_fit_geometry_parameters_weak_inactive_parameter_is_not_recommended(
    monkeypatch,
):
    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        return np.array([gamma - 1.0], dtype=np.float64)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "compute_peak_position_error_geometry_local", fake_compute)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)
    params["gamma"] = 1.0

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[[1.0, 0.0, 0.0, 4.0, 4.0]],
        var_names=["gamma"],
        candidate_param_names=["gamma", "Gamma"],
        experimental_image=None,
        refinement_config={
            "solver": {"restarts": 0},
            "identifiability": {"enabled": True},
        },
    )

    assert result.success
    weak_parameters = result.data_only_identifiability_summary["weak_parameters"]
    assert len(weak_parameters) == 1
    assert str(weak_parameters[0]["name"]) == "Gamma"
    assert list(result.next_stage_recommendations) == []
    assert result.next_stage_recommendation is None


def test_fit_geometry_parameters_reports_retired_stage_placeholders(monkeypatch):
    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        Gamma = float(args[1])
        dist = float(args[2])
        return np.array([gamma - 1.0, Gamma - 2.0, dist - 3.0], dtype=np.float64)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "compute_peak_position_error_geometry_local", fake_compute)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)
    params["corto_detector"] = 0.1

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[],
        var_names=["gamma", "Gamma", "corto_detector"],
        experimental_image=None,
        refinement_config={
            "solver": {
                "restarts": 0,
                "staged_release": {"enabled": True},
                "reparameterize_pairs": {"enabled": True},
            },
            "identifiability": {
                "enabled": True,
                "adaptive_regularization": {"enabled": True},
                "auto_freeze": True,
                "selective_thaw": {"enabled": True},
            },
        },
    )

    assert result.success
    for summary in (
        result.reparameterization_summary,
        result.staged_release_summary,
        result.adaptive_regularization_summary,
        result.auto_freeze_summary,
        result.selective_thaw_summary,
    ):
        assert isinstance(summary, dict)
        assert str(summary["status"]) == "skipped"
        assert bool(summary["accepted"]) is False


def test_fit_geometry_parameters_selects_best_discrete_mode(monkeypatch):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 2.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)
    transformed = opt._transform_points_orientation_local(
        [(2.0, 4.0)],
        (image_size, image_size),
        indexing_mode="xy",
        k=1,
        flip_x=False,
        flip_y=False,
        flip_order="yx",
    )
    measured = [{"label": "1,0,0", "x": transformed[0][0], "y": transformed[0][1]}]

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=[],
        candidate_param_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "discrete_modes": {
                "enabled": True,
                "rot90": [0, 1, 2, 3],
                "flip_x": [False],
                "flip_y": [False],
            },
            "identifiability": {"enabled": False},
            "full_beam_polish": {"enabled": False},
        },
    )

    assert bool(result.success) is False
    assert np.isclose(float(result.cost), 0.0)
    assert str(result.message) == "fixed-parameter mode rot270"
    assert isinstance(result.chosen_discrete_mode, dict)
    assert int(result.chosen_discrete_mode["k"]) == 3
    assert str(result.discrete_mode_summary["selected_label"]) == "rot270"


def test_fit_geometry_parameters_emits_normalized_multistart_status_updates(
    monkeypatch,
):
    status_messages = []

    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        Gamma = float(args[1])
        dist = float(args[2])
        return np.array([gamma - 1.0, Gamma - 2.0, dist - 3.0], dtype=np.float64)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "compute_peak_position_error_geometry_local", fake_compute)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)
    params["corto_detector"] = 0.1

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[],
        var_names=["gamma", "Gamma", "corto_detector"],
        experimental_image=None,
        refinement_config={
            "solver": {"restarts": 0},
            "identifiability": {"enabled": False},
        },
        status_callback=status_messages.append,
    )

    assert result.success
    assert isinstance(result.geometry_fit_debug_summary, dict)
    assert int(result.geometry_fit_debug_summary["dataset_count"]) == 1
    assert list(result.geometry_fit_debug_summary["var_names"]) == [
        "gamma",
        "Gamma",
        "corto_detector",
    ]
    assert isinstance(result.geometry_fit_debug_summary.get("solve_progress"), dict)
    assert int(result.geometry_fit_debug_summary["solve_progress"]["evaluation_count"]) >= 1
    assert any("Geometry fit: setup mode=angle" in msg for msg in status_messages)
    assert any("running normalized-u multistart solve" in msg for msg in status_messages)
    assert any("Geometry fit: mode identity prescore" in msg for msg in status_messages)
    assert any("Geometry fit: multistart summary selected_mode=identity" in msg for msg in status_messages)
    assert any("identity seed" in msg and "cost=" in msg for msg in status_messages)
    assert any("complete" in msg and "metric=angle" in msg for msg in status_messages)
