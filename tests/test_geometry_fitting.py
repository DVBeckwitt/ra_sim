import numpy as np

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


def test_resolve_parallel_worker_count_auto_uses_full_thread_budget(monkeypatch):
    monkeypatch.setattr(opt, "_available_parallel_thread_budget", lambda: 12)

    assert opt._resolve_parallel_worker_count("auto", max_tasks=32) == 12
    assert opt._resolve_parallel_worker_count(None, max_tasks=8) == 8


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
        refinement_config={"solver": {"restarts": 0}},
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
        refinement_config={"solver": {"restarts": 0, "stagnation_probe": False}},
    )

    assert result.success
    assert solve_calls["count"] == 1
    assert isinstance(result.single_ray_polish_summary, dict)
    assert bool(result.single_ray_polish_summary["enabled"]) is False
    assert bool(result.single_ray_polish_summary["accepted"]) is False
    assert str(result.single_ray_polish_summary["status"]) == "skipped"
    assert isinstance(result.point_match_summary, dict)
    assert bool(result.point_match_summary["central_ray_mode"]) is True
    assert bool(result.point_match_summary["single_ray_polish_enabled"]) is False
    assert bool(result.point_match_summary["single_ray_polish_accepted"]) is False
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
        refinement_config={"solver": {"restarts": 0}},
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
        },
    )

    assert result.success
    assert np.allclose(captured_residuals["match"], [0.0, 0.0])
    assert np.allclose(captured_residuals["missing"], [11.0, 0.0])


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


def test_fit_geometry_parameters_pixel_path_tolerates_stale_in_range_source_indices(
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
    assert int(result.point_match_summary["fallback_entry_count"]) == 1
    assert isinstance(result.point_match_diagnostics, list)
    assert len(result.point_match_diagnostics) == 1
    assert result.point_match_diagnostics[0]["resolution_reason"] in {
        "missing_source_indices",
        "source_hkl_mismatch",
    }
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
        },
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 8.0,
            "y": 8.0,
            "overlay_match_index": 3,
            "source_table_index": 0,
            "source_row_index": 0,
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
        },
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 8.0,
            "y": 8.0,
            "overlay_match_index": 3,
            "source_table_index": 0,
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
        "directional probe gamma+" in str(entry.get("message", ""))
        for entry in getattr(result, "restart_history", [])
    )


def test_fit_geometry_parameters_pixel_path_pairwise_probe_escapes_coupled_flat_region(
    monkeypatch,
):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        Gamma = float(args[9])
        sim_col = 6.0 if gamma >= 0.25 and Gamma >= 0.25 else 2.0
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
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
                "stagnation_probe": True,
                "stagnation_probe_fraction": 0.5,
                "stagnation_probe_pairwise": True,
                "stagnation_probe_random_directions": 0,
            }
        },
    )

    assert result.x.shape == (2,)
    assert float(result.x[0]) >= 0.25
    assert float(result.x[1]) >= 0.25
    assert np.allclose(np.asarray(result.fun, dtype=float), [0.0, 0.0])
    assert any(
        "pairwise probe" in str(entry.get("message", ""))
        and "gamma" in str(entry.get("message", ""))
        and "Gamma" in str(entry.get("message", ""))
        for entry in getattr(result, "restart_history", [])
    )


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
    assert "restart corner seed" in str(result.message)
    assert any(
        "restart corner seed" in str(entry.get("message", ""))
        and "opposite-corner" in str(entry.get("message", ""))
        for entry in getattr(result, "restart_history", [])
    )


def test_fit_geometry_parameters_parallelizes_restart_seed_search(monkeypatch):
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

    monkeypatch.setattr(opt, "_threaded_map", fake_threaded_map)
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
    assert threaded_calls
    assert any(call["max_workers"] == 2 and call["count"] > 1 for call in threaded_calls)
    assert any(call["numba_threads"] == 2 for call in threaded_calls)
    assert result.parallelization_summary["dataset_workers"] == 1
    assert result.parallelization_summary["restart_workers"] == 2


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


def test_fit_geometry_parameters_reports_identifiability(monkeypatch):
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
    assert isinstance(result.identifiability_summary, dict)
    assert str(result.identifiability_summary["status"]) == "ok"
    assert int(result.identifiability_summary["num_parameters"]) == 2
    assert int(result.identifiability_summary["rank"]) == 1
    assert bool(result.identifiability_summary["underconstrained"]) is True
    assert str(result.identifiability_summary["dominant_group"]) == "underconstrained"
    parameter_entries = result.identifiability_summary["parameter_entries"]
    assert [entry["name"] for entry in parameter_entries] == ["gamma", "Gamma"]
    assert float(parameter_entries[0]["column_norm"]) > 0.0
    assert np.isclose(float(parameter_entries[1]["column_norm"]), 0.0)
    assert list(result.identifiability_summary["recommended_fixed_parameters"]) == [
        "Gamma"
    ]
    weak_parameters = result.identifiability_summary["weak_parameters"]
    assert len(weak_parameters) == 1
    assert str(weak_parameters[0]["name"]) == "Gamma"


def test_fit_geometry_parameters_can_auto_freeze_weak_parameter(monkeypatch):
    solve_dims = []

    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        return np.array([gamma - 1.0], dtype=np.float64)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x0_arr = np.asarray(x0, dtype=float)
        solve_dims.append(int(x0_arr.size))
        if x0_arr.size == 2:
            x = np.array([0.5, 2.0], dtype=float)
        else:
            x = np.array([1.0], dtype=float)
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
        var_names=["gamma", "Gamma"],
        experimental_image=None,
        refinement_config={
            "solver": {"restarts": 0},
            "identifiability": {"enabled": True, "auto_freeze": True},
        },
    )

    assert result.success
    assert solve_dims == [2, 1]
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([1.0, 2.0]))
    assert isinstance(result.auto_freeze_summary, dict)
    assert bool(result.auto_freeze_summary["accepted"]) is True
    assert list(result.auto_freeze_summary["fixed_parameters"]) == ["Gamma"]


def test_fit_geometry_parameters_can_seed_with_adaptive_regularization(monkeypatch):
    solve_starts = []
    status_messages = []

    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        Gamma = float(args[1])
        return np.array(
            [gamma - 1.0, 0.1 * gamma * (Gamma - 2.0)],
            dtype=np.float64,
        )

    def fake_least_squares(residual_fn, x0, **kwargs):
        x0_arr = np.asarray(x0, dtype=float)
        solve_starts.append(x0_arr.copy())
        if len(solve_starts) == 1:
            assert np.allclose(x0_arr, np.array([0.0, 0.0], dtype=float))
            x = np.array([1.0, 0.25], dtype=float)
        elif len(solve_starts) == 2:
            assert np.allclose(x0_arr, np.array([1.0, 0.25], dtype=float))
            x = np.array([1.0, 2.0], dtype=float)
        elif len(solve_starts) == 3:
            assert np.allclose(x0_arr, np.array([1.0, 2.0], dtype=float))
            x = np.array([1.0, 2.0], dtype=float)
        else:
            raise AssertionError("unexpected least_squares call count")
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
        var_names=["gamma", "Gamma"],
        experimental_image=None,
        refinement_config={
            "solver": {"restarts": 0},
            "identifiability": {
                "enabled": True,
                "adaptive_regularization": {
                    "enabled": True,
                    "max_parameters": 1,
                },
            },
        },
        status_callback=status_messages.append,
    )

    assert result.success
    assert len(solve_starts) == 3
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([1.0, 2.0]))
    assert isinstance(result.adaptive_regularization_summary, dict)
    assert bool(result.adaptive_regularization_summary["accepted"]) is True
    assert bool(result.adaptive_regularization_summary["release_accepted"]) is True
    assert list(result.adaptive_regularization_summary["applied_parameters"]) == [
        "Gamma"
    ]
    assert any("adaptive regularization evaluating" in msg for msg in status_messages)
    assert any("adaptive regularization releasing seed" in msg for msg in status_messages)
    assert any("adaptive regularization accepted" in msg for msg in status_messages)


def test_fit_geometry_parameters_can_selectively_thaw_after_auto_freeze(monkeypatch):
    solve_starts = []
    status_messages = []

    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        Gamma = float(args[1])
        return np.array(
            [gamma - 1.0, 0.1 * gamma * (Gamma - 2.0)],
            dtype=np.float64,
        )

    def fake_least_squares(residual_fn, x0, **kwargs):
        x0_arr = np.asarray(x0, dtype=float)
        solve_starts.append(x0_arr.copy())
        if len(solve_starts) == 1:
            x = np.array([0.0, 0.0], dtype=float)
        elif len(solve_starts) == 2:
            assert x0_arr.size == 1
            x = np.array([1.0], dtype=float)
        elif len(solve_starts) == 3:
            assert np.allclose(x0_arr, np.array([1.0, 0.0], dtype=float))
            x = np.array([1.0, 2.0], dtype=float)
        else:
            raise AssertionError("unexpected least_squares call count")
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
        var_names=["gamma", "Gamma"],
        experimental_image=None,
        refinement_config={
            "solver": {"restarts": 0},
            "identifiability": {
                "enabled": True,
                "auto_freeze": True,
                "selective_thaw": {
                    "enabled": True,
                    "max_condition_number": 1.0e8,
                },
            },
        },
        status_callback=status_messages.append,
    )

    assert result.success
    assert [arr.size for arr in solve_starts] == [2, 1, 2]
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([1.0, 2.0]))
    assert isinstance(result.auto_freeze_summary, dict)
    assert bool(result.auto_freeze_summary["accepted"]) is True
    assert list(result.auto_freeze_summary["fixed_parameters"]) == ["Gamma"]
    assert isinstance(result.selective_thaw_summary, dict)
    assert bool(result.selective_thaw_summary["accepted"]) is True
    assert list(result.selective_thaw_summary["thawed_parameters"]) == ["Gamma"]
    assert list(result.selective_thaw_summary["remaining_fixed_parameters"]) == []
    assert any("selective thaw step" in msg for msg in status_messages)
    assert any("selective thaw accepted" in msg for msg in status_messages)


def test_fit_geometry_parameters_reports_high_correlation_pairs(monkeypatch):
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
        var_names=["gamma", "Gamma"],
        experimental_image=None,
        refinement_config={
            "solver": {"restarts": 0},
            "identifiability": {"enabled": True},
        },
    )

    high_pairs = result.identifiability_summary["high_correlation_pairs"]
    assert len(high_pairs) == 1
    assert {
        str(high_pairs[0]["name_i"]),
        str(high_pairs[0]["name_j"]),
    } == {"gamma", "Gamma"}
    assert float(high_pairs[0]["abs_correlation"]) > 0.99
    assert len(result.identifiability_summary["recommended_fixed_parameters"]) == 1


def test_fit_geometry_parameters_can_stage_parameter_release(monkeypatch):
    solve_starts = []

    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        Gamma = float(args[1])
        dist = float(args[2])
        return np.array([gamma - 1.0, Gamma - 2.0, dist - 3.0], dtype=np.float64)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x0_arr = np.asarray(x0, dtype=float)
        solve_starts.append(x0_arr.copy())
        if x0_arr.size == 2:
            x = np.array([1.0, 2.0], dtype=float)
        else:
            assert np.allclose(x0_arr, np.array([1.0, 2.0, 0.1], dtype=float))
            x = np.array([1.0, 2.0, 3.0], dtype=float)
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
            },
            "identifiability": {"enabled": False},
        },
    )

    assert result.success
    assert len(solve_starts) == 2
    assert solve_starts[0].size == 2
    assert solve_starts[1].size == 3
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([1.0, 2.0, 3.0]))
    assert isinstance(result.staged_release_summary, dict)
    assert bool(result.staged_release_summary["accepted"]) is True
    assert int(result.staged_release_summary["accepted_stage_count"]) == 1
    stage_entries = result.staged_release_summary["stages"]
    assert len(stage_entries) == 1
    assert list(stage_entries[0]["active_parameters"]) == ["gamma", "Gamma"]


def test_fit_geometry_parameters_can_seed_with_pair_reparameterization(monkeypatch):
    solve_starts = []

    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        Gamma = float(args[1])
        dist = float(args[2])
        return np.array([gamma - 1.0, Gamma - 2.0, dist - 3.0], dtype=np.float64)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x0_arr = np.asarray(x0, dtype=float)
        solve_starts.append(x0_arr.copy())
        if len(solve_starts) == 1:
            assert np.allclose(x0_arr, np.array([0.0, 0.0, 0.1], dtype=float))
            x = np.array([1.5, -0.5, 0.1], dtype=float)
        else:
            assert np.allclose(x0_arr, np.array([1.0, 2.0, 0.1], dtype=float))
            x = np.array([1.0, 2.0, 3.0], dtype=float)
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
                "reparameterize_pairs": {"enabled": True},
            },
            "identifiability": {"enabled": False},
        },
    )

    assert result.success
    assert len(solve_starts) == 2
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([1.0, 2.0, 3.0]))
    assert isinstance(result.reparameterization_summary, dict)
    assert bool(result.reparameterization_summary["accepted"]) is True
    assert list(result.reparameterization_summary["pairs"]) == [["gamma", "Gamma"]]


def test_fit_geometry_parameters_emits_status_updates(monkeypatch):
    status_messages = []

    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        Gamma = float(args[1])
        dist = float(args[2])
        return np.array([gamma - 1.0, Gamma - 2.0, dist - 3.0], dtype=np.float64)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.size == 2:
            x = np.array([1.0, 2.0], dtype=float)
        else:
            x = np.array([1.0, 2.0, 3.0], dtype=float)
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
            },
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
    assert any("staged release enabled" in msg for msg in status_messages)
    assert any("Geometry fit: setup mode=angle" in msg for msg in status_messages)
    assert any("Geometry fit: main solve seed cost=" in msg for msg in status_messages)
    assert any("staged stage 1/1" in msg for msg in status_messages)
    assert any("running main solve" in msg for msg in status_messages)
    assert any("complete" in msg for msg in status_messages)
