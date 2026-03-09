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
