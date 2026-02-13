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
