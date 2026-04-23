import math

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from ra_sim.fitting.optimization import (
    fit_mosaic_shape_parameters,
    focus_mosaic_profile_dataset_specs,
)
from ra_sim.fitting.optimization_mosaic_profiles import (
    MosaicPhiProfile,
    MosaicPointPair,
    _mosaic_profile_entry_priority,
    compare_centered_phi_profiles,
    fit_lorentzian_plus_gaussian_profile,
    fit_mosaic_parameters_from_centered_phi_profiles,
    integrate_selected_qr_phi_profiles,
    lorentzian_plus_gaussian_profile,
    pair_selected_qr_and_background_points,
)

TT_SCALE = 0.06
PHI_SCALE = 0.08


def _pseudo_voigt(delta_deg, sigma_deg, gamma_deg, eta):
    sigma = max(float(sigma_deg), 1.0e-3)
    gamma = max(float(gamma_deg), 1.0e-3)
    gaussian = np.exp(-0.5 * (np.asarray(delta_deg, dtype=np.float64) / sigma) ** 2)
    lorentz = 1.0 / (1.0 + (np.asarray(delta_deg, dtype=np.float64) / gamma) ** 2)
    return float(eta) * gaussian + (1.0 - float(eta)) * lorentz


def _base_center_map():
    return {
        (1, 1, 0): (16.0, 18.0),
        (1, 1, 1): (30.0, 18.0),
        (0, 0, 1): (18.0, 56.0),
        (0, 0, 2): (34.0, 56.0),
        (0, 0, 3): (50.0, 56.0),
        (2, -1, 0): (62.0, 18.0),
    }


def _center_for(hkl, theta_initial):
    base_row, base_col = _base_center_map()[tuple(int(v) for v in hkl)]
    theta_shift = float(theta_initial) - 3.0
    return (
        base_row + 2.0 * theta_shift,
        base_col + 1.0 * theta_shift,
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
    in_plane_scale=1.0,
    specular_scale=1.0,
):
    rows = np.arange(image_size, dtype=np.float64)
    cols = np.arange(image_size, dtype=np.float64)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    image = np.full((image_size, image_size), 0.002, dtype=np.float64)

    for idx, hkl in enumerate(np.asarray(miller_subset, dtype=np.float64)):
        hkl_key = tuple(int(round(v)) for v in hkl)
        center_row, center_col = _center_for(hkl_key, theta_initial)
        delta_tt = (yy - center_row) * TT_SCALE
        delta_phi = (xx - center_col) * PHI_SCALE

        if hkl_key[0] == 0 and hkl_key[1] == 0:
            width_tt = 0.12 + 0.28 * float(sigma_deg) + 0.22 * float(gamma_deg)
            width_tt += 0.08 * float(eta)
            raw = _pseudo_voigt(delta_tt, width_tt, 1.25 * width_tt, eta)
            raw *= np.exp(-0.5 * (delta_phi / 0.10) ** 2)
            l_val = abs(int(hkl_key[2]))
            area_scale = float(intens_subset[idx]) * float(specular_scale)
            area_scale *= (
                1.0
                + 0.18 * float(sigma_deg) * (1.0 + 0.2 * l_val)
                + 0.12 * float(gamma_deg) * max(l_val - 0.5, 0.5)
                + 0.42 * float(eta) * (0.6 + 0.15 * l_val)
            )
        else:
            width_phi = 0.18 + 0.40 * float(sigma_deg) + 0.18 * float(gamma_deg)
            width_phi += 0.06 * float(eta)
            raw = _pseudo_voigt(delta_phi, width_phi, 1.3 * width_phi, eta)
            raw *= np.exp(-0.5 * (delta_tt / 0.06) ** 2)
            area_scale = float(intens_subset[idx]) * float(in_plane_scale)

        total = float(np.sum(raw))
        if total > 0.0 and np.isfinite(total):
            image += raw / total * area_scale

    return image


def _fake_detector_pixels_to_fit_space(
    cols,
    rows,
    *,
    center,
    detector_distance,
    pixel_size,
    gamma_deg=0.0,
    Gamma_deg=0.0,
):
    del center
    del detector_distance
    del pixel_size
    del gamma_deg
    del Gamma_deg
    cols_arr = np.asarray(cols, dtype=np.float64).reshape(-1)
    rows_arr = np.asarray(rows, dtype=np.float64).reshape(-1)
    return rows_arr * TT_SCALE, cols_arr * PHI_SCALE


def _make_fake_process_peaks(
    recorded_theta_values,
    *,
    sim_in_plane_scale=1.0,
    sim_specular_scale=1.0,
    recorded_subsets=None,
    recorded_kernel_kwargs=None,
):
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
        del av
        del cv
        del lambda_array
        del dist
        del geom_gamma
        del geom_Gamma
        del chi
        del psi
        del psi_z
        del zs
        del zb
        del n2
        del beam_x_array
        del beam_y_array
        del theta_array
        del phi_array
        del wavelength_array
        del debye_x
        del debye_y
        del center
        del cor_angle
        del unit_x
        del n_detector
        del save_flag
        if recorded_kernel_kwargs is not None:
            recorded_kernel_kwargs.append(dict(kwargs))
        del kwargs
        recorded_theta_values.append(float(theta_initial))
        if recorded_subsets is not None:
            recorded_subsets.append(np.array(miller_subset, dtype=np.float64, copy=True))
        image = _render_image(
            miller_subset,
            intens_subset,
            image_size_subset,
            theta_initial=float(theta_initial),
            sigma_deg=float(sigma_deg),
            gamma_deg=float(gamma_deg),
            eta=float(eta),
            in_plane_scale=float(sim_in_plane_scale),
            specular_scale=float(sim_specular_scale),
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


def _build_dataset_specs(
    image_size,
    theta_values,
    hkls,
    intensities,
    true_shape,
    *,
    measured_in_plane_scale=1.0,
    measured_specular_scale=1.0,
):
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
            in_plane_scale=float(measured_in_plane_scale),
            specular_scale=float(measured_specular_scale),
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


def test_fit_mosaic_shape_parameters_recovers_profile_parameters_and_uses_dataset_theta(
    monkeypatch,
):
    image_size = 84
    hkls = [(1, 1, 0), (1, 1, 1), (0, 0, 1), (0, 0, 2), (0, 0, 3)]
    intensities = np.array([4.0, 3.0, 2.8, 2.2, 1.7], dtype=np.float64)
    miller = np.asarray(hkls, dtype=np.float64)
    true_shape = (0.82, 0.54, 0.63)
    dataset_specs = _build_dataset_specs(
        image_size,
        theta_values=[3.0, 3.3],
        hkls=hkls,
        intensities=intensities,
        true_shape=true_shape,
    )

    recorded_theta_values = []
    monkeypatch.setattr(
        "ra_sim.fitting.optimization._detector_pixels_to_fit_space",
        _fake_detector_pixels_to_fit_space,
    )
    monkeypatch.setattr(
        "ra_sim.fitting.optimization.process_peaks_parallel",
        _make_fake_process_peaks(recorded_theta_values, sim_in_plane_scale=3.5),
    )

    def select_best_lsq(fun, x0, **kwargs):
        candidates = [
            np.asarray(x0, dtype=np.float64),
            np.asarray(true_shape, dtype=np.float64),
            np.array([0.68, 0.44, 0.48], dtype=np.float64),
            np.array([1.05, 0.78, 0.82], dtype=np.float64),
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
        specular_relative_intensity_weight=1.0,
        fit_theta_i=False,
    )

    assert result.success
    assert result.acceptance_passed is True
    assert result.roi_count_by_dataset == {0: 5, 1: 5}
    assert result.cost_reduction >= 0.20
    assert np.allclose(result.x, np.asarray(true_shape, dtype=np.float64), atol=1.0e-9)
    assert {round(value, 2) for value in recorded_theta_values} >= {3.0, 3.3}
    assert result.final_residual_rms >= 0.0
    assert result.parameter_bounds["sigma_mosaic_deg"]["final"] == pytest.approx(true_shape[0])
    diag_by_label = {diag["dataset_label"]: diag for diag in result.dataset_diagnostics}
    assert diag_by_label["bg0"]["in_plane_roi_count"] == 2
    assert diag_by_label["bg0"]["specular_roi_count"] == 3
    assert diag_by_label["bg0"]["relative_intensity_term_count"] == 2
    debug_summary = result.mosaic_fit_debug_summary
    assert debug_summary["inputs"]["dataset_count"] == 2
    assert debug_summary["acceptance"]["passed"] is True
    assert debug_summary["objective_terms"]["profile_shape_enabled"] is True
    assert debug_summary["objective_terms"]["specular_relative_intensity_enabled"] is True


def test_fit_mosaic_shape_parameters_refines_per_dataset_theta_i(monkeypatch):
    image_size = 84
    hkls = [(1, 1, 0), (1, 1, 1), (0, 0, 1), (0, 0, 2), (0, 0, 3)]
    intensities = np.array([4.0, 3.0, 2.8, 2.2, 1.7], dtype=np.float64)
    miller = np.asarray(hkls, dtype=np.float64)
    true_shape = (0.82, 0.54, 0.63)
    true_theta_values = [3.0, 3.3]
    dataset_specs = _build_dataset_specs(
        image_size,
        theta_values=true_theta_values,
        hkls=hkls,
        intensities=intensities,
        true_shape=true_shape,
    )
    dataset_specs[0]["theta_initial"] = 2.7
    dataset_specs[1]["theta_initial"] = 3.0

    recorded_theta_values = []
    monkeypatch.setattr(
        "ra_sim.fitting.optimization._detector_pixels_to_fit_space",
        _fake_detector_pixels_to_fit_space,
    )
    monkeypatch.setattr(
        "ra_sim.fitting.optimization.process_peaks_parallel",
        _make_fake_process_peaks(recorded_theta_values, sim_in_plane_scale=3.5),
    )

    def select_best_lsq(fun, x0, **kwargs):
        x0 = np.asarray(x0, dtype=np.float64)
        candidates = [
            x0,
            np.array([*true_shape, *true_theta_values], dtype=np.float64),
            np.array([0.68, 0.44, 0.48, 2.9, 3.2], dtype=np.float64),
            np.array([1.05, 0.78, 0.82, 2.8, 3.1], dtype=np.float64),
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
        specular_relative_intensity_weight=1.0,
        fit_theta_i=True,
        theta_i_mode="per_dataset",
        theta_i_bounds_deg=(-0.5, 0.5),
    )

    assert result.success
    assert result.theta_refinement_mode == "per_dataset"
    assert result.theta_param_names == ["theta_initial[0]", "theta_initial[1]"]
    assert result.refined_theta_values_by_dataset == pytest.approx({0: 3.0, 1: 3.3})
    assert result.parameter_bounds["theta_initial[0]"]["final"] == pytest.approx(3.0)
    assert result.parameter_bounds["theta_initial[1]"]["final"] == pytest.approx(3.3)
    assert {round(value, 2) for value in recorded_theta_values} >= {2.7, 3.0, 3.3}
    debug_summary = result.mosaic_fit_debug_summary
    assert debug_summary["theta"]["optimize_theta"] is True
    assert debug_summary["theta"]["resolved_mode"] == "per_dataset"
    assert debug_summary["theta"]["refined_theta_values_by_dataset"] == pytest.approx(
        {"0": 3.0, "1": 3.3}
    )


def test_fit_mosaic_shape_parameters_only_simulates_selected_profile_reflections(
    monkeypatch,
):
    image_size = 84
    hkls = [(1, 1, 0), (1, 1, 1), (0, 0, 1), (0, 0, 2), (0, 0, 3)]
    intensities = np.array([4.0, 3.0, 2.8, 2.2, 1.7], dtype=np.float64)
    miller = np.asarray(hkls, dtype=np.float64)
    dataset_specs = _build_dataset_specs(
        image_size,
        theta_values=[3.0],
        hkls=hkls,
        intensities=intensities,
        true_shape=(0.80, 0.50, 0.55),
    )
    selected_hkls = {(1, 1, 0), (0, 0, 1), (0, 0, 3)}
    dataset_specs[0]["measured_peaks"] = [
        dict(entry)
        for entry in dataset_specs[0]["measured_peaks"]
        if tuple(int(v) for v in entry["hkl"]) in selected_hkls
    ]

    recorded_subsets = []
    recorded_kernel_kwargs = []
    monkeypatch.setattr(
        "ra_sim.fitting.optimization._detector_pixels_to_fit_space",
        _fake_detector_pixels_to_fit_space,
    )
    monkeypatch.setattr(
        "ra_sim.fitting.optimization.process_peaks_parallel",
        _make_fake_process_peaks(
            [],
            recorded_subsets=recorded_subsets,
            recorded_kernel_kwargs=recorded_kernel_kwargs,
        ),
    )

    def fake_least_squares(fun, x0, **kwargs):
        return OptimizeResult(
            x=np.asarray(x0, dtype=np.float64),
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
        specular_relative_intensity_weight=0.0,
        fit_theta_i=False,
        min_total_rois=3,
        min_per_dataset_rois=3,
    )

    assert result.roi_count_by_dataset == {0: 3}
    assert recorded_subsets, "optimizer should simulate at least one selected subset"
    assert recorded_kernel_kwargs
    assert all(kw.get("collect_hit_tables") is False for kw in recorded_kernel_kwargs)
    for subset in recorded_subsets:
        simulated_hkls = {
            tuple(int(round(v)) for v in row) for row in np.asarray(subset, dtype=np.float64)
        }
        assert simulated_hkls == selected_hkls


def test_focus_mosaic_profile_dataset_specs_keeps_all_specular_and_top_three_q_groups():
    dataset_specs = [
        {
            "dataset_index": 1,
            "label": "bg1",
            "theta_initial": 3.0,
            "measured_peaks": [
                {"hkl": (0, 0, 1), "x": 1.0, "y": 1.0},
                {"hkl": (0, 0, 2), "x": 2.0, "y": 2.0},
                {
                    "hkl": (1, 0, 0),
                    "x": 10.0,
                    "y": 10.0,
                    "weight": 5.0,
                    "q_group_key": ("q_group", "primary", 1, 0),
                    "source_peak_index": 4,
                },
                {
                    "hkl": (-1, 0, 0),
                    "x": 11.0,
                    "y": 10.0,
                    "weight": 4.0,
                    "q_group_key": ("q_group", "primary", 1, 0),
                    "source_peak_index": 7,
                },
                {
                    "hkl": (1, 1, 0),
                    "x": 20.0,
                    "y": 10.0,
                    "weight": 9.0,
                    "q_group_key": ("q_group", "primary", 3, 0),
                    "source_peak_index": 2,
                },
                {
                    "hkl": (2, -1, 0),
                    "x": 30.0,
                    "y": 10.0,
                    "weight": 8.0,
                    "q_group_key": ("q_group", "primary", 7, 0),
                    "source_peak_index": 3,
                },
                {
                    "hkl": (2, 0, 0),
                    "x": 40.0,
                    "y": 10.0,
                    "weight": 1.0,
                    "q_group_key": ("q_group", "primary", 4, 0),
                    "source_peak_index": 9,
                },
            ],
        },
        {
            "dataset_index": 2,
            "label": "bg2",
            "theta_initial": 3.2,
            "measured_peaks": [
                {"hkl": (0, 0, 1), "x": 1.5, "y": 1.2},
                {"hkl": (0, 0, 2), "x": 2.5, "y": 2.2},
                {
                    "hkl": (1, 0, 0),
                    "x": 10.5,
                    "y": 11.0,
                    "weight": 5.5,
                    "q_group_key": ("q_group", "primary", 1, 0),
                    "source_peak_index": 5,
                },
                {
                    "hkl": (-1, 0, 0),
                    "x": 12.5,
                    "y": 11.0,
                    "weight": 5.1,
                    "q_group_key": ("q_group", "primary", 1, 0),
                    "source_peak_index": 8,
                },
                {
                    "hkl": (1, 1, 0),
                    "x": 21.0,
                    "y": 10.5,
                    "weight": 9.5,
                    "q_group_key": ("q_group", "primary", 3, 0),
                    "source_peak_index": 1,
                },
                {
                    "hkl": (2, -1, 0),
                    "x": 31.0,
                    "y": 10.5,
                    "weight": 7.5,
                    "q_group_key": ("q_group", "primary", 7, 0),
                    "source_peak_index": 4,
                },
                {
                    "hkl": (2, 0, 0),
                    "x": 41.0,
                    "y": 10.5,
                    "weight": 0.5,
                    "q_group_key": ("q_group", "primary", 4, 0),
                    "source_peak_index": 10,
                },
            ],
        },
    ]

    focused_specs, summary = focus_mosaic_profile_dataset_specs(
        dataset_specs,
        source_miller=np.asarray(
            [
                (1, 0, 0),
                (-1, 0, 0),
                (1, 1, 0),
                (2, -1, 0),
                (2, 0, 0),
                (0, 0, 1),
                (0, 0, 2),
            ],
            dtype=np.float64,
        ),
        source_intensities=np.asarray([5.0, 4.0, 9.0, 8.0, 1.0, 3.0, 2.0]),
        reference_dataset_index=1,
        max_in_plane_groups=3,
    )

    assert summary["selected_specular_hkls"] == [[0, 0, 1], [0, 0, 2]]
    assert summary["selected_in_plane_hkls"] == [[1, 1, 0], [1, 0, 0], [2, -1, 0]]
    assert summary["selected_peak_count_by_dataset"] == {"1": 5, "2": 5}

    for spec in focused_specs:
        kept_hkls = [tuple(entry["hkl"]) for entry in spec["measured_peaks"]]
        assert kept_hkls[:2] == [(0, 0, 1), (0, 0, 2)]
        assert (2, 0, 0) not in kept_hkls
        assert sum(1 for hkl in kept_hkls if hkl in {(1, 0, 0), (-1, 0, 0)}) == 1


def test_focus_mosaic_profile_dataset_specs_uses_source_intensity_when_weight_missing():
    dataset_specs = [
        {
            "dataset_index": 0,
            "label": "bg0",
            "theta_initial": 3.0,
            "measured_peaks": [
                {
                    "hkl": (1, 0, 0),
                    "x": 10.0,
                    "y": 10.0,
                    "q_group_key": ("q_group", "primary", 1, 0),
                    "source_table_index": 0,
                    "source_peak_index": 5,
                },
                {
                    "hkl": (-1, 0, 0),
                    "x": 12.0,
                    "y": 10.0,
                    "q_group_key": ("q_group", "primary", 1, 0),
                    "source_table_index": 1,
                    "source_peak_index": 6,
                },
                {
                    "hkl": (1, 1, 0),
                    "x": 20.0,
                    "y": 10.0,
                    "q_group_key": ("q_group", "primary", 3, 0),
                    "source_table_index": 2,
                    "source_peak_index": 2,
                },
                {
                    "hkl": (2, -1, 0),
                    "x": 30.0,
                    "y": 10.0,
                    "q_group_key": ("q_group", "primary", 7, 0),
                    "source_table_index": 3,
                    "source_peak_index": 3,
                },
                {
                    "hkl": (2, 0, 0),
                    "x": 40.0,
                    "y": 10.0,
                    "q_group_key": ("q_group", "primary", 4, 0),
                    "source_table_index": 4,
                    "source_peak_index": 4,
                },
                {"hkl": (0, 0, 1), "x": 1.0, "y": 1.0, "source_table_index": 5},
            ],
        }
    ]

    focused_specs, summary = focus_mosaic_profile_dataset_specs(
        dataset_specs,
        source_miller=np.asarray(
            [
                (1, 0, 0),
                (-1, 0, 0),
                (1, 1, 0),
                (2, -1, 0),
                (2, 0, 0),
                (0, 0, 1),
            ],
            dtype=np.float64,
        ),
        source_intensities=np.asarray([8.0, 7.5, 6.0, 5.0, 1.0, 3.0]),
        reference_dataset_index=0,
        max_in_plane_groups=3,
    )

    assert summary["selected_in_plane_hkls"] == [[1, 0, 0], [1, 1, 0], [2, -1, 0]]
    kept_hkls = [tuple(entry["hkl"]) for entry in focused_specs[0]["measured_peaks"]]
    assert kept_hkls == [(0, 0, 1), (1, 0, 0), (1, 1, 0), (2, -1, 0)]


def test_mosaic_profile_entry_priority_prefers_canonical_branch_over_dense_peak_alias():
    miller = np.asarray([(1, 0, 0)], dtype=np.float64)
    intensities = np.asarray([5.0], dtype=np.float64)
    intensity_lookup = {(1, 0, 0): 5.0}

    branch_zero = {
        "hkl": (1, 0, 0),
        "weight": 1.0,
        "source_branch_index": 0,
        "source_peak_index": 9,
        "source_row_index": 4,
    }
    branch_one = {
        "hkl": (1, 0, 0),
        "weight": 1.0,
        "source_branch_index": 1,
        "source_peak_index": 2,
        "source_row_index": 4,
    }

    assert _mosaic_profile_entry_priority(
        branch_zero,
        miller=miller,
        intensities=intensities,
        intensity_lookup=intensity_lookup,
    ) < _mosaic_profile_entry_priority(
        branch_one,
        miller=miller,
        intensities=intensities,
        intensity_lookup=intensity_lookup,
    )


def test_fit_mosaic_shape_parameters_accepts_generic_off_specular_peak_families(
    monkeypatch,
):
    image_size = 84
    hkls = [(1, 1, 0), (1, 1, 1), (0, 0, 1), (0, 0, 2), (0, 0, 3), (2, -1, 0)]
    intensities = np.array([4.0, 3.0, 2.8, 2.2, 1.7, 1.4], dtype=np.float64)
    miller = np.asarray(hkls, dtype=np.float64)
    dataset_specs = _build_dataset_specs(
        image_size,
        theta_values=[3.0],
        hkls=hkls,
        intensities=intensities,
        true_shape=(0.78, 0.48, 0.55),
    )

    monkeypatch.setattr(
        "ra_sim.fitting.optimization._detector_pixels_to_fit_space",
        _fake_detector_pixels_to_fit_space,
    )
    monkeypatch.setattr(
        "ra_sim.fitting.optimization.process_peaks_parallel",
        _make_fake_process_peaks([]),
    )

    def fake_least_squares(fun, x0, **kwargs):
        return OptimizeResult(
            x=np.asarray(x0, dtype=np.float64),
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
        specular_relative_intensity_weight=0.0,
        fit_theta_i=False,
        min_total_rois=5,
        min_per_dataset_rois=5,
    )

    assert result.roi_count_by_dataset == {0: 6}
    assert result.prepared_dataset_summaries[0]["roi_count"] == 6
    diag_by_label = {diag["dataset_label"]: diag for diag in result.dataset_diagnostics}
    assert diag_by_label["bg0"]["in_plane_roi_count"] == 3
    assert diag_by_label["bg0"]["specular_roi_count"] == 3
    assert "bg0:profile_prep:unsupported_peak_family" not in result.rejected_roi_reason_counts


def test_fit_mosaic_shape_parameters_keeps_residual_length_fixed(monkeypatch):
    image_size = 84
    hkls = [(1, 1, 0), (1, 1, 1), (0, 0, 1), (0, 0, 2), (0, 0, 3)]
    intensities = np.array([4.0, 3.0, 2.8, 2.2, 1.7], dtype=np.float64)
    miller = np.asarray(hkls, dtype=np.float64)
    dataset_specs = _build_dataset_specs(
        image_size,
        theta_values=[3.0, 3.25],
        hkls=hkls,
        intensities=intensities,
        true_shape=(0.75, 0.45, 0.50),
    )

    monkeypatch.setattr(
        "ra_sim.fitting.optimization._detector_pixels_to_fit_space",
        _fake_detector_pixels_to_fit_space,
    )
    monkeypatch.setattr(
        "ra_sim.fitting.optimization.process_peaks_parallel",
        _make_fake_process_peaks([]),
    )

    residual_lengths = []

    def fake_least_squares(fun, x0, **kwargs):
        x0 = np.asarray(x0, dtype=np.float64)
        residual_lengths.append(fun(x0).size)
        residual_lengths.append(fun(np.array([0.18, 1.5, 0.90], dtype=np.float64)).size)
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
        fit_theta_i=False,
    )

    assert result.success
    assert len(set(residual_lengths)) == 1


def test_fit_mosaic_shape_parameters_uses_dataset_parallel_workers(monkeypatch):
    image_size = 84
    hkls = [(1, 1, 0), (1, 1, 1), (0, 0, 1), (0, 0, 2), (0, 0, 3)]
    intensities = np.array([4.0, 3.0, 2.8, 2.2, 1.7], dtype=np.float64)
    miller = np.asarray(hkls, dtype=np.float64)
    dataset_specs = _build_dataset_specs(
        image_size,
        theta_values=[3.0, 3.25],
        hkls=hkls,
        intensities=intensities,
        true_shape=(0.75, 0.45, 0.50),
    )

    monkeypatch.setattr(
        "ra_sim.fitting.optimization._detector_pixels_to_fit_space",
        _fake_detector_pixels_to_fit_space,
    )
    monkeypatch.setattr(
        "ra_sim.fitting.optimization.process_peaks_parallel",
        _make_fake_process_peaks([]),
    )

    threaded_map_calls = []

    def fake_threaded_map(fn, items, *, max_workers, numba_threads=None):
        threaded_map_calls.append(
            {
                "item_count": len(items),
                "max_workers": int(max_workers),
                "numba_threads": numba_threads,
            }
        )
        return [fn(item) for item in items]

    def fake_least_squares(fun, x0, **kwargs):
        return OptimizeResult(
            x=np.asarray(x0, dtype=np.float64),
            fun=fun(x0),
            success=True,
            message="stub",
        )

    monkeypatch.setattr("ra_sim.fitting.optimization._threaded_map", fake_threaded_map)
    monkeypatch.setattr("ra_sim.fitting.optimization.least_squares", fake_least_squares)

    result = fit_mosaic_shape_parameters(
        miller,
        intensities,
        image_size,
        _base_params(image_size),
        dataset_specs=dataset_specs,
        max_restarts=0,
        roi_half_width=8,
        workers=8,
        parallel_mode="datasets",
        fit_theta_i=False,
    )

    assert result.success
    assert result.parallelization_summary["dataset_workers"] == 2
    assert result.parallelization_summary["restart_workers"] == 1
    assert any(call["max_workers"] == 2 for call in threaded_map_calls)


def test_fit_mosaic_shape_parameters_emits_live_progress_updates(monkeypatch):
    image_size = 84
    hkls = [(1, 1, 0), (1, 1, 1), (0, 0, 1), (0, 0, 2), (0, 0, 3)]
    intensities = np.array([4.0, 3.0, 2.8, 2.2, 1.7], dtype=np.float64)
    miller = np.asarray(hkls, dtype=np.float64)
    dataset_specs = _build_dataset_specs(
        image_size,
        theta_values=[3.0],
        hkls=hkls,
        intensities=intensities,
        true_shape=(0.75, 0.45, 0.50),
    )

    monkeypatch.setattr(
        "ra_sim.fitting.optimization._detector_pixels_to_fit_space",
        _fake_detector_pixels_to_fit_space,
    )
    monkeypatch.setattr(
        "ra_sim.fitting.optimization.process_peaks_parallel",
        _make_fake_process_peaks([]),
    )

    progress_lines = []

    def fake_least_squares(fun, x0, **kwargs):
        x0 = np.asarray(x0, dtype=np.float64)
        trial = np.array([0.78, 0.48, 0.55], dtype=np.float64)
        fun(x0)
        return OptimizeResult(
            x=trial,
            fun=fun(trial),
            success=True,
            message="stub-progress",
            nfev=2,
        )

    monkeypatch.setattr("ra_sim.fitting.optimization.least_squares", fake_least_squares)

    result = fit_mosaic_shape_parameters(
        miller,
        intensities,
        image_size,
        _base_params(image_size),
        dataset_specs=dataset_specs,
        max_restarts=1,
        roi_half_width=8,
        min_total_rois=5,
        min_per_dataset_rois=5,
        fit_theta_i=False,
        progress_callback=progress_lines.append,
    )

    assert result.success
    assert result.mosaic_fit_debug_summary["solver"]["progress_callback_enabled"] is True
    assert any(line.startswith("Preparing dataset ROIs:") for line in progress_lines)
    assert any(line.startswith("Prepared 1 dataset(s):") for line in progress_lines)
    assert any(line.startswith("Dataset bg0:") for line in progress_lines)
    assert any(line.startswith("Initial objective:") for line in progress_lines)
    assert any(line.startswith("Primary solve start:") for line in progress_lines)
    assert any(line.startswith("Primary eval 1:") for line in progress_lines)
    assert any(line.startswith("Primary solve done:") for line in progress_lines)
    assert any(line.startswith("Restarts scheduled:") for line in progress_lines)
    assert any(line.startswith("Restart 1/1 start:") for line in progress_lines)
    assert any(line.startswith("Restart 1/1 eval 1:") for line in progress_lines)
    assert any(line.startswith("Restart 1/1 done:") for line in progress_lines)
    assert any(line.startswith("Final result:") for line in progress_lines)
    assert any(line.startswith("Final dataset bg0:") for line in progress_lines)


def test_fit_mosaic_shape_parameters_reports_dataset_weighting_and_bound_hits(
    monkeypatch,
):
    image_size = 84
    hkls = [(1, 1, 0), (1, 1, 1), (0, 0, 1), (0, 0, 2), (0, 0, 3)]
    intensities = np.array([4.0, 3.0, 2.8, 2.2, 1.7], dtype=np.float64)
    miller = np.asarray(hkls, dtype=np.float64)
    dataset_specs = _build_dataset_specs(
        image_size,
        theta_values=[3.0, 3.2],
        hkls=hkls,
        intensities=intensities,
        true_shape=(0.70, 0.45, 0.40),
    )
    dataset_specs[1]["measured_peaks"] = dataset_specs[1]["measured_peaks"][:4]

    monkeypatch.setattr(
        "ra_sim.fitting.optimization._detector_pixels_to_fit_space",
        _fake_detector_pixels_to_fit_space,
    )
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
        fit_theta_i=False,
    )

    diag_by_label = {diag["dataset_label"]: diag for diag in result.dataset_diagnostics}
    assert diag_by_label["bg0"]["dataset_weight"] == pytest.approx(1.0 / math.sqrt(5.0))
    assert diag_by_label["bg1"]["dataset_weight"] == pytest.approx(1.0 / math.sqrt(4.0))
    assert "Parameters finished on bounds" in str(result.boundary_warning)
    assert result.acceptance_passed is False


def test_fit_mosaic_shape_parameters_enforces_roi_minimums(monkeypatch):
    image_size = 84
    hkls = [(1, 1, 0), (1, 1, 1), (0, 0, 1), (0, 0, 2), (0, 0, 3)]
    intensities = np.array([4.0, 3.0, 2.8, 2.2, 1.7], dtype=np.float64)
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
        "ra_sim.fitting.optimization._detector_pixels_to_fit_space",
        _fake_detector_pixels_to_fit_space,
    )
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
            min_total_rois=6,
            min_per_dataset_rois=3,
            fit_theta_i=False,
        )


def test_fit_mosaic_shape_parameters_requires_one_active_parameter(monkeypatch):
    image_size = 84
    hkls = [(1, 1, 0), (1, 1, 1), (0, 0, 1), (0, 0, 2), (0, 0, 3)]
    intensities = np.array([4.0, 3.0, 2.8, 2.2, 1.7], dtype=np.float64)
    miller = np.asarray(hkls, dtype=np.float64)
    dataset_specs = _build_dataset_specs(
        image_size,
        theta_values=[3.0],
        hkls=hkls,
        intensities=intensities,
        true_shape=(0.75, 0.45, 0.50),
    )

    monkeypatch.setattr(
        "ra_sim.fitting.optimization._detector_pixels_to_fit_space",
        _fake_detector_pixels_to_fit_space,
    )
    monkeypatch.setattr(
        "ra_sim.fitting.optimization.process_peaks_parallel",
        _make_fake_process_peaks([]),
    )

    with pytest.raises(ValueError, match="At least one mosaic fit parameter must be enabled"):
        fit_mosaic_shape_parameters(
            miller,
            intensities,
            image_size,
            _base_params(image_size),
            dataset_specs=dataset_specs,
            max_restarts=0,
            roi_half_width=8,
            min_total_rois=1,
            min_per_dataset_rois=1,
            fit_theta_i=False,
            fit_sigma_mosaic=False,
            fit_gamma_mosaic=False,
            fit_eta=False,
        )


def test_pair_selected_qr_and_background_points_matches_duplicate_q_groups_by_peak_index():
    selected = [
        {"q_group_key": ("q_group", "primary", 1, 5), "source_peak_index": 0},
        {"q_group_key": ("q_group", "primary", 1, 5), "source_peak_index": 1},
    ]
    background = [
        {"q_group_key": ("q_group", "primary", 1, 5), "source_peak_index": 1, "x": 2, "y": 3},
        {"q_group_key": ("q_group", "primary", 1, 5), "source_peak_index": 0, "x": 4, "y": 5},
    ]

    pairs = pair_selected_qr_and_background_points(selected, background)

    assert len(pairs) == 2
    assert pairs[0].background_point["x"] == 4
    assert pairs[1].background_point["x"] == 2


def test_fit_lorentzian_plus_gaussian_profile_recovers_center_and_widths():
    phi = np.linspace(-2.0, 2.0, 161, dtype=np.float64)
    expected = lorentzian_plus_gaussian_profile(
        phi,
        center_deg=0.23,
        gaussian_amplitude=12.0,
        gaussian_sigma_deg=0.18,
        lorentzian_amplitude=5.0,
        lorentzian_gamma_deg=0.42,
        baseline=0.7,
    )

    fit = fit_lorentzian_plus_gaussian_profile(phi, expected)

    assert fit.success
    assert fit.center_deg == pytest.approx(0.23, abs=0.02)
    assert fit.gaussian_sigma_deg > 0.0
    assert fit.lorentzian_gamma_deg > 0.0
    assert fit.area > 0.0


def test_integrate_selected_qr_phi_profiles_uses_background_anchor_roi_and_fit():
    image_size = 41
    rows = np.arange(image_size, dtype=np.float64)
    cols = np.arange(image_size, dtype=np.float64)
    phi_axis = np.linspace(-2.0, 2.0, image_size, dtype=np.float64)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    phi_line = lorentzian_plus_gaussian_profile(
        phi_axis,
        center_deg=0.20,
        gaussian_amplitude=9.0,
        gaussian_sigma_deg=0.16,
        lorentzian_amplitude=4.0,
        lorentzian_gamma_deg=0.32,
        baseline=0.0,
    )
    orthogonal = np.exp(-0.5 * ((xx - 20.0) / 1.6) ** 2)
    image = 2.0 + phi_line[:, None] * orthogonal
    manual_entry = {
        "q_group_key": ("q_group", "primary", 1, 5),
        "source_peak_index": 0,
        "hkl": (1, 0, 0),
        "x": 20.0,
        "y": 20.0,
        "background_phi_deg": 0.0,
    }

    profiles, rejected = integrate_selected_qr_phi_profiles(
        image,
        [manual_entry],
        phi_deg_map=phi_axis,
        phi_axis="row",
        roi_half_width_px=14,
        orthogonal_half_width_px=3,
        phi_bin_count=61,
    )

    assert rejected == []
    assert len(profiles) == 1
    assert profiles[0].pair.q_group_key == ("q_group", "primary", 1, 5)
    assert profiles[0].fit is not None
    assert profiles[0].fit.center_deg == pytest.approx(0.20, abs=0.08)
    assert float(np.sum(profiles[0].intensity)) > 0.0


def test_integrate_selected_qr_phi_profiles_rejects_ambiguous_square_1d_phi_axis():
    image = np.ones((21, 21), dtype=np.float64)
    manual_entry = {
        "q_group_key": ("q_group", "primary", 1, 5),
        "source_peak_index": 0,
        "hkl": (1, 0, 0),
        "x": 10.0,
        "y": 10.0,
        "background_phi_deg": 0.0,
    }

    profiles, rejected = integrate_selected_qr_phi_profiles(
        image,
        [manual_entry],
        phi_deg_map=np.linspace(-1.0, 1.0, 21, dtype=np.float64),
        roi_half_width_px=6,
    )

    assert profiles == []
    assert len(rejected) == 1
    assert "ambiguous 1D phi_deg_map" in rejected[0]["reason"]


def test_integrate_selected_qr_phi_profiles_accepts_explicit_column_axis_for_square_map():
    image_size = 41
    rows = np.arange(image_size, dtype=np.float64)
    cols = np.arange(image_size, dtype=np.float64)
    phi_axis = np.linspace(-2.0, 2.0, image_size, dtype=np.float64)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    phi_line = lorentzian_plus_gaussian_profile(
        phi_axis,
        center_deg=0.20,
        gaussian_amplitude=9.0,
        gaussian_sigma_deg=0.16,
        lorentzian_amplitude=4.0,
        lorentzian_gamma_deg=0.32,
        baseline=0.0,
    )
    orthogonal = np.exp(-0.5 * ((yy - 20.0) / 1.6) ** 2)
    image = 2.0 + phi_line[None, :] * orthogonal
    manual_entry = {
        "q_group_key": ("q_group", "primary", 1, 5),
        "source_peak_index": 0,
        "hkl": (1, 0, 0),
        "x": 20.0,
        "y": 20.0,
        "background_phi_deg": 0.0,
    }

    profiles, rejected = integrate_selected_qr_phi_profiles(
        image,
        [manual_entry],
        phi_deg_map=phi_axis,
        phi_axis="col",
        roi_half_width_px=14,
        orthogonal_half_width_px=3,
        phi_bin_count=61,
    )

    assert rejected == []
    assert len(profiles) == 1
    assert profiles[0].fit is not None
    assert profiles[0].fit.center_deg == pytest.approx(0.20, abs=0.08)


def _scaffold_profile(width: float, *, center: float = 0.0) -> MosaicPhiProfile:
    phi = np.linspace(-2.0, 2.0, 121, dtype=np.float64)
    intensity = lorentzian_plus_gaussian_profile(
        phi,
        center_deg=float(center),
        gaussian_amplitude=1.0,
        gaussian_sigma_deg=float(width),
        lorentzian_amplitude=0.45,
        lorentzian_gamma_deg=float(width) * 1.7,
        baseline=0.0,
    )
    pair = MosaicPointPair(
        pair_index=0,
        pair_key=("q_group", "primary", 1, 5),
        selected_qr_point={"q_group_key": ("q_group", "primary", 1, 5)},
        background_point={"q_group_key": ("q_group", "primary", 1, 5), "x": 0.0, "y": 0.0},
        q_group_key=("q_group", "primary", 1, 5),
        hkl=(1, 0, 0),
    )
    profile = MosaicPhiProfile(
        pair=pair,
        phi_deg=phi,
        intensity=intensity,
        signal_counts=np.ones_like(phi),
        background_level=np.zeros_like(phi),
        center_col=0.0,
        center_row=0.0,
        phi_anchor_deg=0.0,
        row_bounds=(0, 1),
        col_bounds=(0, 1),
        weight=1.0,
    )
    profile.fit = fit_lorentzian_plus_gaussian_profile(profile.phi_deg, profile.intensity)
    return profile


def test_compare_centered_phi_profiles_aligns_fitted_peak_centers():
    measured = _scaffold_profile(0.35, center=0.35)
    candidate = _scaffold_profile(0.35, center=-0.45)

    comparison = compare_centered_phi_profiles([measured], [candidate])

    assert comparison.profile_count == 1
    assert comparison.residual_rms < 2.0e-3
    assert comparison.per_profile[0]["matched"] is True


def test_compare_centered_phi_profiles_consumes_duplicate_pair_keys_in_order():
    measured = [
        _scaffold_profile(0.26, center=0.35),
        _scaffold_profile(0.62, center=-0.25),
    ]
    candidate = [
        _scaffold_profile(0.26, center=-0.45),
        _scaffold_profile(0.62, center=0.55),
    ]

    comparison = compare_centered_phi_profiles(measured, candidate)

    assert comparison.profile_count == 2
    assert comparison.residual_rms < 2.0e-3
    assert [entry["matched"] for entry in comparison.per_profile] == [True, True]


def test_compare_centered_phi_profiles_penalizes_extra_candidate_profiles():
    measured = [_scaffold_profile(0.35, center=0.1)]
    matching_candidate = _scaffold_profile(0.35, center=-0.2)
    extra_candidate = _scaffold_profile(0.62, center=0.45)

    baseline = compare_centered_phi_profiles(measured, [matching_candidate])
    comparison = compare_centered_phi_profiles(
        measured,
        [matching_candidate, extra_candidate],
    )

    assert baseline.profile_count == 1
    assert baseline.residual_rms < 2.0e-3
    assert comparison.profile_count == 2
    assert comparison.residual_rms > baseline.residual_rms
    assert comparison.per_profile[-1]["matched"] is False
    assert comparison.per_profile[-1]["extra_candidate"] is True


def test_fit_mosaic_parameters_from_centered_phi_profiles_refines_active_width():
    measured = [_scaffold_profile(0.46, center=0.15)]

    def simulate(params):
        return [_scaffold_profile(float(params["width"]), center=-0.2)]

    result = fit_mosaic_parameters_from_centered_phi_profiles(
        measured,
        {"width": 0.20, "fixed": 3.0},
        ["width"],
        simulate,
        bounds={"width": (0.05, 1.0)},
        max_nfev=40,
    )

    assert result.best_parameters["width"] == pytest.approx(0.46, abs=0.04)
    assert result.best_parameters["fixed"] == 3.0
    assert result.final_cost < result.initial_cost
