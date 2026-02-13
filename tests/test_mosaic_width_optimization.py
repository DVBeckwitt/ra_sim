import numpy as np
import pytest

from ra_sim.fitting.optimization import fit_mosaic_widths_separable
from ra_sim.utils.calculations import d_spacing, two_theta


def _make_gaussian(center, sigma_px, gamma_px, size):
    rows = np.arange(size)
    cols = np.arange(size)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    cy, cx = center
    return np.exp(-((xx - cx) ** 2) / (2 * sigma_px ** 2) - ((yy - cy) ** 2) / (2 * gamma_px ** 2))


def test_fit_mosaic_widths_separable_recovers_true_widths(monkeypatch):
    image_size = 64
    miller = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [-1, 2, 0],
        [2, -1, 1],
    ], dtype=np.float64)
    intensities = np.array([3.0, 2.5, 2.2, 1.8, 1.6, 1.4], dtype=np.float64)
    centers = [
        (16, 16),
        (16, 48),
        (48, 16),
        (48, 48),
        (24, 32),
        (40, 32),
    ]
    center_map = {
        tuple(int(v) for v in hkl): center
        for hkl, center in zip(np.rint(miller).astype(int), centers)
    }

    sigma_true = 0.65
    gamma_true = 0.4
    eta_true = 0.12

    recorded_subsets = []

    def fake_process_peaks_parallel(
        miller_subset,
        intens_subset,
        image_size_subset,
        av,
        cv,
        lambda_array,
        buffer,
        dist,
        geom_gamma,
        geom_Gamma,
        chi,
        psi,
        psi_z,
        zs,
        zb,
        n2,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        sigma_deg,
        gamma_deg,
        eta,
        wavelength_array,
        debye_x,
        debye_y,
        center,
        theta_initial,
        cor_angle,
        unit_x,
        n_detector,
        save_flag,
        record_status=False,
        thickness=0.0,
    ):
        recorded_subsets.append(np.array(miller_subset, dtype=np.float64, copy=True))
        buffer.fill(0.0)
        sigma_px = max(1.0, sigma_deg * 8.0)
        gamma_px = max(1.0, gamma_deg * 8.0)
        hit_tables = []
        miss_tables = []
        for idx, hkl in enumerate(miller_subset):
            key = tuple(int(round(v)) for v in hkl)
            centre = center_map.get(key, centers[idx % len(centers)])
            narrow = _make_gaussian(centre, sigma_px, gamma_px, image_size_subset)
            wide = _make_gaussian(centre, sigma_px * 1.5, gamma_px * 1.5, image_size_subset)
            template = eta * narrow + (1.0 - eta) * wide
            contrib = intens_subset[idx] * template
            buffer += contrib
            max_intensity = float(np.max(contrib))
            hit_tables.append(
                np.array(
                    [[
                        max_intensity,
                        float(centre[1]),
                        float(centre[0]),
                        0.0,
                        float(hkl[0]),
                        float(hkl[1]),
                        float(hkl[2]),
                    ]],
                    dtype=np.float64,
                )
            )
            miss_tables.append(np.empty((0, 3), dtype=np.float64))
        return buffer.copy(), hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), miss_tables

    monkeypatch.setattr(
        "ra_sim.fitting.optimization.process_peaks_parallel", fake_process_peaks_parallel
    )

    mosaic_arrays = dict(
        beam_x_array=np.zeros(1),
        beam_y_array=np.zeros(1),
        theta_array=np.zeros(1),
        phi_array=np.zeros(1),
        wavelength_array=np.ones(1),
    )

    params = {
        "a": 4.0,
        "c": 7.0,
        "lambda": 1.0,
        "cor_angle": 0.0,
        "corto_detector": 0.5,
        "gamma": 0.0,
        "Gamma": 0.0,
        "chi": 0.0,
        "psi": 0.0,
        "psi_z": 0.0,
        "zs": 0.0,
        "zb": 0.0,
        "n2": 1.0,
        "debye_x": 0.0,
        "debye_y": 0.0,
        "theta_initial": 0.0,
        "center": (image_size / 2.0, image_size / 2.0),
        "uv1": np.array([1.0, 0.0, 0.0]),
        "uv2": np.array([0.0, 1.0, 0.0]),
    }

    buffer = np.zeros((image_size, image_size), dtype=np.float64)
    experimental_image, *_ = fake_process_peaks_parallel(
        miller,
        intensities,
        image_size,
        params["a"],
        params["c"],
        np.ones(1),
        buffer,
        params["corto_detector"],
        params["gamma"],
        params["Gamma"],
        params["chi"],
        params["psi"],
        params["psi_z"],
        params["zs"],
        params["zb"],
        params["n2"],
        mosaic_arrays["beam_x_array"],
        mosaic_arrays["beam_y_array"],
        mosaic_arrays["theta_array"],
        mosaic_arrays["phi_array"],
        sigma_true,
        gamma_true,
        eta_true,
        mosaic_arrays["wavelength_array"],
        params["debye_x"],
        params["debye_y"],
        params["center"],
        params["theta_initial"],
        params["cor_angle"],
        params["uv1"],
        params["uv2"],
        0,
    )
    experimental_image = experimental_image + 5.0

    # Drop the pre-fit call so only optimizer evaluations remain.
    recorded_subsets.clear()

    params_initial = dict(params)
    params_initial["mosaic_params"] = dict(
        mosaic_arrays,
        sigma_mosaic_deg=0.4,
        gamma_mosaic_deg=0.25,
        eta=0.05,
    )

    result = fit_mosaic_widths_separable(
        experimental_image,
        miller,
        intensities,
        image_size,
        params_initial,
        num_peaks=4,
        roi_half_width=5,
        min_peak_separation=12.0,
        stratify="L",
        max_nfev=80,
        peak_source="auto",
        max_restarts=0,
        roi_normalization="none",
        bounds=(
            np.array([0.01, 0.01, 0.0], dtype=np.float64),
            np.array([5.0, 5.0, 1.0], dtype=np.float64),
        ),
    )

    assert result.success
    assert abs(result.x[0] - sigma_true) < 0.1
    assert abs(result.x[1] - gamma_true) < 0.1
    assert abs(result.x[2] - eta_true) < 0.15
    assert np.isfinite(result.initial_cost)
    assert np.isfinite(result.final_cost)
    assert isinstance(result.roi_diagnostics, list)
    assert isinstance(result.rejected_rois, list)

    selected_hkls = {roi.hkl for roi in result.selected_rois}
    assert selected_hkls == {(0, 0, 1), (1, 1, 0), (-1, 2, 0), (2, -1, 1)}
    assert all((2 * h + k) % 3 == 0 for h, k, _ in selected_hkls)

    assert recorded_subsets, "optimizer should simulate at least once"
    for subset in recorded_subsets:
        if subset.size == 0:
            continue
        hk = np.rint(subset[:, :2]).astype(int)
        assert np.all((2 * hk[:, 0] + hk[:, 1]) % 3 == 0)
        hkl_full = np.rint(subset[:, :3]).astype(int)
        for h, k, l in hkl_full:
            d_hkl = d_spacing(h, k, l, params["a"], params["c"])
            tth = two_theta(d_hkl, params["lambda"])
            assert tth is not None
            if h == 0 and k == 0:
                continue
            assert tth <= 65.0 + 1e-8


def test_fit_mosaic_widths_separable_geometry_requires_measured_peaks():
    image_size = 32
    experimental_image = np.ones((image_size, image_size), dtype=np.float64)
    miller = np.array([[1, 1, 0], [-1, 2, 0], [0, 0, 1]], dtype=np.float64)
    intensities = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    params = {
        "a": 4.0,
        "c": 7.0,
        "lambda": 1.0,
        "cor_angle": 0.0,
        "corto_detector": 0.5,
        "gamma": 0.0,
        "Gamma": 0.0,
        "chi": 0.0,
        "psi": 0.0,
        "psi_z": 0.0,
        "zs": 0.0,
        "zb": 0.0,
        "n2": 1.0,
        "debye_x": 0.0,
        "debye_y": 0.0,
        "theta_initial": 0.0,
        "center": (image_size / 2.0, image_size / 2.0),
        "uv1": np.array([1.0, 0.0, 0.0]),
        "uv2": np.array([0.0, 1.0, 0.0]),
        "mosaic_params": {
            "beam_x_array": np.zeros(1),
            "beam_y_array": np.zeros(1),
            "theta_array": np.zeros(1),
            "phi_array": np.zeros(1),
            "wavelength_array": np.ones(1),
            "sigma_mosaic_deg": 0.4,
            "gamma_mosaic_deg": 0.2,
            "eta": 0.1,
        },
    }

    with pytest.raises(RuntimeError, match="Geometry-locked mosaic fit requires measured geometry peaks"):
        fit_mosaic_widths_separable(
            experimental_image,
            miller,
            intensities,
            image_size,
            params,
            peak_source="geometry",
        )


def test_fit_mosaic_widths_separable_geometry_source_uses_measured_peaks(monkeypatch):
    image_size = 96
    miller = np.array(
        [
            [1, 1, 0],
            [-1, 2, 0],
            [2, -1, 1],
            [0, 0, 1],
            [3, 0, 0],
            [0, 3, 0],
            [2, 2, 0],
            [-2, 1, 1],
            [1, -2, 2],
            [4, -2, 0],
            [-4, 2, 0],
            [0, 0, 2],
        ],
        dtype=np.float64,
    )
    intensities = np.linspace(3.0, 1.2, miller.shape[0], dtype=np.float64)

    centers = [
        (14, 14),
        (14, 34),
        (14, 54),
        (14, 74),
        (34, 14),
        (34, 34),
        (34, 54),
        (34, 74),
        (54, 14),
        (54, 34),
        (54, 54),
        (54, 74),
    ]
    center_map = {
        tuple(int(v) for v in hkl): center
        for hkl, center in zip(np.rint(miller).astype(int), centers)
    }

    recorded_subsets = []
    sigma_true = 0.55
    gamma_true = 0.35
    eta_true = 0.2

    def fake_process_peaks_parallel(
        miller_subset,
        intens_subset,
        image_size_subset,
        av,
        cv,
        lambda_array,
        buffer,
        dist,
        geom_gamma,
        geom_Gamma,
        chi,
        psi,
        psi_z,
        zs,
        zb,
        n2,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        sigma_deg,
        gamma_deg,
        eta,
        wavelength_array,
        debye_x,
        debye_y,
        center,
        theta_initial,
        cor_angle,
        unit_x,
        n_detector,
        save_flag,
        record_status=False,
        thickness=0.0,
    ):
        recorded_subsets.append(np.array(miller_subset, dtype=np.float64, copy=True))
        buffer.fill(0.0)
        sigma_px = max(1.0, sigma_deg * 8.0)
        gamma_px = max(1.0, gamma_deg * 8.0)
        hit_tables = []
        miss_tables = []
        for idx, hkl in enumerate(miller_subset):
            key = tuple(int(round(v)) for v in hkl)
            centre = center_map.get(key, centers[idx % len(centers)])
            narrow = _make_gaussian(centre, sigma_px, gamma_px, image_size_subset)
            wide = _make_gaussian(centre, sigma_px * 1.4, gamma_px * 1.4, image_size_subset)
            template = eta * narrow + (1.0 - eta) * wide
            contrib = intens_subset[idx] * template
            buffer += contrib
            hit_tables.append(
                np.array(
                    [[
                        float(np.max(contrib)),
                        float(centre[1]),
                        float(centre[0]),
                        0.0,
                        float(hkl[0]),
                        float(hkl[1]),
                        float(hkl[2]),
                    ]],
                    dtype=np.float64,
                )
            )
            miss_tables.append(np.empty((0, 3), dtype=np.float64))
        return buffer.copy(), hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), miss_tables

    monkeypatch.setattr(
        "ra_sim.fitting.optimization.process_peaks_parallel",
        fake_process_peaks_parallel,
    )

    params = {
        "a": 4.0,
        "c": 7.0,
        "lambda": 1.0,
        "cor_angle": 0.0,
        "corto_detector": 0.5,
        "gamma": 0.0,
        "Gamma": 0.0,
        "chi": 0.0,
        "psi": 0.0,
        "psi_z": 0.0,
        "zs": 0.0,
        "zb": 0.0,
        "n2": 1.0,
        "debye_x": 0.0,
        "debye_y": 0.0,
        "theta_initial": 0.0,
        "center": (image_size / 2.0, image_size / 2.0),
        "uv1": np.array([1.0, 0.0, 0.0]),
        "uv2": np.array([0.0, 1.0, 0.0]),
        "mosaic_params": {
            "beam_x_array": np.zeros(1),
            "beam_y_array": np.zeros(1),
            "theta_array": np.zeros(1),
            "phi_array": np.zeros(1),
            "wavelength_array": np.ones(1),
            "sigma_mosaic_deg": 0.4,
            "gamma_mosaic_deg": 0.2,
            "eta": 0.1,
        },
    }

    buffer = np.zeros((image_size, image_size), dtype=np.float64)
    experimental_image, *_ = fake_process_peaks_parallel(
        miller,
        intensities,
        image_size,
        params["a"],
        params["c"],
        np.ones(1),
        buffer,
        params["corto_detector"],
        params["gamma"],
        params["Gamma"],
        params["chi"],
        params["psi"],
        params["psi_z"],
        params["zs"],
        params["zb"],
        params["n2"],
        params["mosaic_params"]["beam_x_array"],
        params["mosaic_params"]["beam_y_array"],
        params["mosaic_params"]["theta_array"],
        params["mosaic_params"]["phi_array"],
        sigma_true,
        gamma_true,
        eta_true,
        params["mosaic_params"]["wavelength_array"],
        params["debye_x"],
        params["debye_y"],
        params["center"],
        params["theta_initial"],
        params["cor_angle"],
        params["uv1"],
        params["uv2"],
        0,
    )
    experimental_image = experimental_image + 3.0

    measured_hkls = [tuple(int(v) for v in hkl) for hkl in np.rint(miller[:9]).astype(int)]
    measured_peaks = [
        {
            "label": f"{h},{k},{l}",
            "x": float(center_map[(h, k, l)][1]),
            "y": float(center_map[(h, k, l)][0]),
        }
        for h, k, l in measured_hkls
    ]

    recorded_subsets.clear()
    result = fit_mosaic_widths_separable(
        experimental_image,
        miller,
        intensities,
        image_size,
        params,
        num_peaks=8,
        roi_half_width=5,
        min_peak_separation=10.0,
        peak_source="geometry",
        measured_peaks=measured_peaks,
        stratify="twotheta",
        max_nfev=60,
    )

    assert result.success
    assert len(result.selected_rois) >= 8
    selected_hkls = {roi.hkl for roi in result.selected_rois}
    assert selected_hkls.issubset(set(measured_hkls))
    assert isinstance(result.top_worst_rois, list)
    assert isinstance(result.restart_history, list)

    measured_set = set(measured_hkls)
    assert recorded_subsets, "optimizer should simulate at least once"
    for subset in recorded_subsets:
        hkl_subset = {tuple(int(round(v)) for v in row) for row in subset}
        assert hkl_subset.issubset(measured_set)
