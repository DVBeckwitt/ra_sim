import math

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from ra_sim.fitting.optimization import fit_mosaic_shape_parameters


def _shape_template(center, sigma_px, gamma_px, eta, size):
    rows = np.arange(size, dtype=np.float64)
    cols = np.arange(size, dtype=np.float64)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    cy, cx = center
    gaussian = np.exp(
        -((xx - cx) ** 2) / (2.0 * sigma_px ** 2)
        - ((yy - cy) ** 2) / (2.0 * gamma_px ** 2)
    )
    lorentz = 1.0 / (
        1.0
        + ((xx - cx) / max(sigma_px * 1.8, 0.4)) ** 2
        + ((yy - cy) / max(gamma_px * 1.8, 0.4)) ** 2
    )
    return eta * gaussian + (1.0 - eta) * lorentz


def _base_center_map():
    return {
        (1, 1, 0): (18.0, 18.0),
        (0, 0, 1): (18.0, 50.0),
        (2, -1, 0): (50.0, 18.0),
        (1, 1, 1): (50.0, 50.0),
        (0, 0, 2): (34.0, 34.0),
    }


def _center_for(hkl, theta_initial):
    base_row, base_col = _base_center_map()[tuple(int(v) for v in hkl)]
    theta_shift = float(theta_initial) - 3.0
    return (
        base_row + 1.5 * theta_shift,
        base_col + 3.0 * theta_shift,
    )


def _render_image(
    miller_subset,
    intens_subset,
    image_size,
    *,
    theta_initial,
    sigma_deg,
    gamma_deg,
    eta,
):
    image = np.zeros((image_size, image_size), dtype=np.float64)
    sigma_px = max(1.0, float(sigma_deg) * 8.0)
    gamma_px = max(1.0, float(gamma_deg) * 8.0)
    for idx, hkl in enumerate(np.asarray(miller_subset, dtype=np.float64)):
        center = _center_for(tuple(int(round(v)) for v in hkl), theta_initial)
        image += float(intens_subset[idx]) * _shape_template(
            center,
            sigma_px,
            gamma_px,
            float(eta),
            int(image_size),
        )
    return image


def _make_fake_process_peaks(recorded_theta_values):
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
        **kwargs,
    ):
        recorded_theta_values.append(float(theta_initial))
        image = _render_image(
            miller_subset,
            intens_subset,
            image_size_subset,
            theta_initial=float(theta_initial),
            sigma_deg=float(sigma_deg),
            gamma_deg=float(gamma_deg),
            eta=float(eta),
        )
        buffer.fill(0.0)
        buffer += image
        hit_tables = [np.empty((0, 7), dtype=np.float64) for _ in range(len(miller_subset))]
        miss_tables = [np.empty((0, 3), dtype=np.float64) for _ in range(len(miller_subset))]
        return (
            buffer.copy(),
            hit_tables,
            np.empty((0, 0, 0), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            miss_tables,
        )

    return fake_process_peaks_parallel


def _base_params(image_size, sigma_deg=0.35, gamma_deg=0.25, eta=0.25):
    return {
        "a": 3.0,
        "c": 20.0,
        "lambda": 1.0,
        "psi": 0.0,
        "psi_z": 0.0,
        "zs": 0.0,
        "zb": 0.0,
        "sample_width_m": 0.0,
        "sample_length_m": 0.0,
        "sample_depth_m": 0.0,
        "chi": 0.0,
        "n2": 1.0 + 0.0j,
        "mosaic_params": {
            "beam_x_array": np.zeros(1, dtype=np.float64),
            "beam_y_array": np.zeros(1, dtype=np.float64),
            "theta_array": np.zeros(1, dtype=np.float64),
            "phi_array": np.zeros(1, dtype=np.float64),
            "wavelength_array": np.ones(1, dtype=np.float64),
            "sigma_mosaic_deg": float(sigma_deg),
            "gamma_mosaic_deg": float(gamma_deg),
            "eta": float(eta),
        },
        "debye_x": 0.0,
        "debye_y": 0.0,
        "center": [image_size / 2.0, image_size / 2.0],
        "theta_initial": 3.0,
        "theta_offset": 0.0,
        "uv1": np.array([1.0, 0.0, 0.0]),
        "uv2": np.array([0.0, 1.0, 0.0]),
        "corto_detector": 0.12,
        "gamma": 0.0,
        "Gamma": 0.0,
        "optics_mode": 0,
        "pixel_size": 1.0e-3,
        "pixel_size_m": 1.0e-3,
    }


def _build_dataset_specs(image_size, theta_values, hkls, intensities, true_shape):
    specs = []
    miller = np.asarray(hkls, dtype=np.float64)
    intens_arr = np.asarray(intensities, dtype=np.float64)
    for dataset_index, theta_initial in enumerate(theta_values):
        image = _render_image(
            miller,
            intens_arr,
            image_size,
            theta_initial=float(theta_initial),
            sigma_deg=float(true_shape[0]),
            gamma_deg=float(true_shape[1]),
            eta=float(true_shape[2]),
        )
        measured_peaks = []
        for hkl in hkls:
            row, col = _center_for(hkl, theta_initial)
            measured_peaks.append(
                {
                    "hkl": tuple(int(v) for v in hkl),
                    "x": float(col),
                    "y": float(row),
                }
            )
        specs.append(
            {
                "dataset_index": int(dataset_index),
                "label": f"bg{dataset_index}",
                "theta_initial": float(theta_initial),
                "measured_peaks": measured_peaks,
                "experimental_image": image,
            }
        )
    return specs


def test_fit_mosaic_shape_parameters_recovers_shape_parameters_and_uses_dataset_theta(
    monkeypatch,
):
    image_size = 72
    hkls = [(1, 1, 0), (0, 0, 1), (2, -1, 0), (1, 1, 1)]
    intensities = np.array([4.0, 3.0, 2.5, 2.0], dtype=np.float64)
    miller = np.asarray(hkls, dtype=np.float64)
    true_shape = (0.82, 0.56, 0.62)
    dataset_specs = _build_dataset_specs(
        image_size,
        theta_values=[3.0, 3.35],
        hkls=hkls,
        intensities=intensities,
        true_shape=true_shape,
    )

    recorded_theta_values = []
    monkeypatch.setattr(
        "ra_sim.fitting.optimization.process_peaks_parallel",
        _make_fake_process_peaks(recorded_theta_values),
    )

    def select_best_lsq(fun, x0, **kwargs):
        candidates = [
            np.asarray(x0, dtype=np.float64),
            np.asarray(true_shape, dtype=np.float64),
            np.array([0.75, 0.50, 0.55], dtype=np.float64),
            np.array([0.95, 0.65, 0.70], dtype=np.float64),
        ]
        scored = []
        for candidate in candidates:
            residual = np.asarray(fun(candidate), dtype=np.float64)
            scored.append((float(np.dot(residual, residual)), candidate, residual))
        _, best_x, best_fun = min(scored, key=lambda item: item[0])
        return OptimizeResult(
            x=np.asarray(best_x, dtype=np.float64),
            fun=np.asarray(best_fun, dtype=np.float64),
            success=True,
            message="grid-search",
        )

    monkeypatch.setattr("ra_sim.fitting.optimization.least_squares", select_best_lsq)

    result = fit_mosaic_shape_parameters(
        miller,
        intensities,
        image_size,
        _base_params(image_size),
        dataset_specs=dataset_specs,
        max_nfev=60,
        max_restarts=0,
        roi_half_width=8,
    )

    assert result.success
    assert result.acceptance_passed is True
    assert result.roi_count_by_dataset == {0: 4, 1: 4}
    assert result.cost_reduction >= 0.20
    assert np.allclose(result.x, np.asarray(true_shape, dtype=np.float64), atol=1.0e-9)
    assert {round(value, 2) for value in recorded_theta_values} >= {3.0, 3.35}


def test_fit_mosaic_shape_parameters_keeps_residual_length_fixed(monkeypatch):
    image_size = 72
    hkls = [(1, 1, 0), (0, 0, 1), (2, -1, 0), (1, 1, 1)]
    intensities = np.array([4.0, 3.0, 2.5, 2.0], dtype=np.float64)
    miller = np.asarray(hkls, dtype=np.float64)
    dataset_specs = _build_dataset_specs(
        image_size,
        theta_values=[3.0, 3.25],
        hkls=hkls,
        intensities=intensities,
        true_shape=(0.75, 0.45, 0.50),
    )

    monkeypatch.setattr(
        "ra_sim.fitting.optimization.process_peaks_parallel",
        _make_fake_process_peaks([]),
    )

    residual_lengths = []

    def fake_least_squares(fun, x0, **kwargs):
        x0 = np.asarray(x0, dtype=np.float64)
        residual_lengths.append(fun(x0).size)
        residual_lengths.append(fun(np.array([0.08, 1.8, 0.95], dtype=np.float64)).size)
        return OptimizeResult(
            x=x0,
            fun=fun(x0),
            success=True,
            message="stub",
        )

    monkeypatch.setattr("ra_sim.fitting.optimization.least_squares", fake_least_squares)

    result = fit_mosaic_shape_parameters(
        miller,
        intensities,
        image_size,
        _base_params(image_size),
        dataset_specs=dataset_specs,
        max_restarts=0,
        roi_half_width=8,
    )

    assert result.success
    assert len(set(residual_lengths)) == 1


def test_fit_mosaic_shape_parameters_reports_dataset_weighting_and_bound_hits(
    monkeypatch,
):
    image_size = 72
    hkls = [(1, 1, 0), (0, 0, 1), (2, -1, 0), (1, 1, 1), (0, 0, 2)]
    intensities = np.array([4.0, 3.0, 2.5, 2.0, 1.5], dtype=np.float64)
    miller = np.asarray(hkls, dtype=np.float64)
    dataset_specs = _build_dataset_specs(
        image_size,
        theta_values=[3.0, 3.2],
        hkls=hkls,
        intensities=intensities,
        true_shape=(0.7, 0.45, 0.4),
    )
    dataset_specs[1]["measured_peaks"] = dataset_specs[1]["measured_peaks"][:3]

    monkeypatch.setattr(
        "ra_sim.fitting.optimization.process_peaks_parallel",
        _make_fake_process_peaks([]),
    )

    def fake_least_squares(fun, x0, bounds, **kwargs):
        x = np.asarray(bounds[1], dtype=np.float64)
        return OptimizeResult(
            x=x,
            fun=fun(x),
            success=True,
            message="hit-bound",
        )

    monkeypatch.setattr("ra_sim.fitting.optimization.least_squares", fake_least_squares)

    result = fit_mosaic_shape_parameters(
        miller,
        intensities,
        image_size,
        _base_params(image_size),
        dataset_specs=dataset_specs,
        max_restarts=0,
        roi_half_width=8,
    )

    diag_by_label = {diag["dataset_label"]: diag for diag in result.dataset_diagnostics}
    assert diag_by_label["bg0"]["dataset_weight"] == pytest.approx(1.0 / math.sqrt(5.0))
    assert diag_by_label["bg1"]["dataset_weight"] == pytest.approx(1.0 / math.sqrt(3.0))
    assert "Parameters finished on bounds" in str(result.boundary_warning)
    assert result.acceptance_passed is False


def test_fit_mosaic_shape_parameters_enforces_roi_minimums(monkeypatch):
    image_size = 72
    hkls = [(1, 1, 0), (0, 0, 1), (2, -1, 0), (1, 1, 1)]
    intensities = np.array([4.0, 3.0, 2.5, 2.0], dtype=np.float64)
    miller = np.asarray(hkls, dtype=np.float64)
    dataset_specs = _build_dataset_specs(
        image_size,
        theta_values=[3.0, 3.2],
        hkls=hkls,
        intensities=intensities,
        true_shape=(0.7, 0.45, 0.4),
    )
    dataset_specs[0]["measured_peaks"] = dataset_specs[0]["measured_peaks"][:2]
    dataset_specs[1]["measured_peaks"] = dataset_specs[1]["measured_peaks"][:2]

    monkeypatch.setattr(
        "ra_sim.fitting.optimization.process_peaks_parallel",
        _make_fake_process_peaks([]),
    )

    with pytest.raises(RuntimeError, match="usable ROIs per dataset"):
        fit_mosaic_shape_parameters(
            miller,
            intensities,
            image_size,
            _base_params(image_size),
            dataset_specs=dataset_specs,
            max_restarts=0,
            roi_half_width=8,
            min_total_rois=8,
            min_per_dataset_rois=3,
        )
