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


def test_fit_geometry_parameters_pixel_path_enables_single_ray_by_default(monkeypatch):
    process_calls = []

    def fake_process(*args, **kwargs):
        process_calls.append(dict(kwargs))
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
    assert any(
        call.get("best_sample_indices_out") is not None for call in process_calls
    )
    assert any(
        isinstance(call.get("single_sample_indices"), np.ndarray)
        for call in process_calls
    )


def test_fit_geometry_parameters_pixel_path_can_disable_single_ray(monkeypatch):
    process_calls = []

    def fake_process(*args, **kwargs):
        process_calls.append(dict(kwargs))
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
            "single_ray": {"enabled": False},
        },
    )

    assert result.success
    assert process_calls
    assert not any(
        call.get("best_sample_indices_out") is not None for call in process_calls
    )
    assert all(
        call.get("single_sample_indices") is None for call in process_calls
    )


def test_fit_geometry_parameters_pixel_path_restricts_simulation_to_selected_reflections(
    monkeypatch,
):
    process_calls = []

    def fake_process(*args, **kwargs):
        process_calls.append(
            {
                "miller": np.asarray(args[0], dtype=np.float64).copy(),
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
    assert any(
        call["kwargs"].get("best_sample_indices_out") is not None
        for call in process_calls
    )
    assert any(
        isinstance(call["kwargs"].get("single_sample_indices"), np.ndarray)
        for call in process_calls
    )
    assert isinstance(result.point_match_summary, dict)
    assert int(result.point_match_summary["simulated_reflection_count"]) == 1
    assert int(result.point_match_summary["total_reflection_count"]) == 3
    assert bool(result.point_match_summary["subset_reduced"]) is True
    assert bool(result.point_match_summary["single_ray_enabled"]) is True
    assert int(result.point_match_summary["single_ray_forced_count"]) == 1


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
