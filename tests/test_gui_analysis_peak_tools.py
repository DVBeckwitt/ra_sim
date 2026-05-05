import numpy as np

from ra_sim.gui import analysis_peak_tools


def _area_normalized_pseudo_voigt(
    x_values,
    *,
    baseline: float,
    area: float,
    center: float,
    fwhm: float,
    eta: float,
):
    x_arr = np.asarray(x_values, dtype=float)
    fwhm_value = max(abs(float(fwhm)), 1.0e-12)
    delta = x_arr - float(center)
    gaussian = (
        2.0
        * np.sqrt(np.log(2.0))
        / (np.sqrt(np.pi) * fwhm_value)
        * np.exp(-4.0 * np.log(2.0) * (delta / fwhm_value) ** 2)
    )
    lorentzian = 2.0 / (np.pi * fwhm_value) / (1.0 + 4.0 * (delta / fwhm_value) ** 2)
    eta_value = float(np.clip(float(eta), 0.0, 1.0))
    return float(baseline) + float(area) * ((1.0 - eta_value) * gaussian + eta_value * lorentzian)


def test_align_angle_to_axis_prefers_wrapped_axis_domain() -> None:
    axis = np.linspace(170.0, 190.0, 9)

    assert analysis_peak_tools.align_angle_to_axis(-170.0, axis) == 190.0
    assert analysis_peak_tools.align_angle_to_axis(175.0, axis) == 175.0


def test_integration_region_contains_supports_wrapped_phi_ranges() -> None:
    assert analysis_peak_tools.integration_region_contains(
        18.0,
        85.0,
        tth_min=10.0,
        tth_max=20.0,
        phi_min=70.0,
        phi_max=-70.0,
    )
    assert analysis_peak_tools.integration_region_contains(
        18.0,
        -82.0,
        tth_min=10.0,
        tth_max=20.0,
        phi_min=70.0,
        phi_max=-70.0,
    )
    assert not analysis_peak_tools.integration_region_contains(
        18.0,
        0.0,
        tth_min=10.0,
        tth_max=20.0,
        phi_min=70.0,
        phi_max=-70.0,
    )


def test_sample_curve_value_interpolates_sorted_curve() -> None:
    x_values = np.array([3.0, 1.0, 2.0], dtype=float)
    y_values = np.array([9.0, 1.0, 4.0], dtype=float)

    assert analysis_peak_tools.sample_curve_value(x_values, y_values, 2.5) == 6.5
    assert np.isnan(analysis_peak_tools.sample_curve_value(x_values, y_values, 0.5))


def test_subtract_linear_background_plane_recovers_peak_profile_from_sloped_roi() -> None:
    theta_axis = np.linspace(17.0, 19.0, 121)
    phi_axis = np.linspace(-4.0, 4.0, 101)
    theta_grid, phi_grid = np.meshgrid(theta_axis, phi_axis)
    theta_center = 18.2
    phi_center = -0.7
    theta_fwhm = 0.36
    phi_fwhm = 1.4
    plane = 12.0 + 1.7 * (theta_grid - 18.0) - 0.35 * phi_grid
    peak = 30.0 * np.exp(
        -4.0
        * np.log(2.0)
        * (
            ((theta_grid - theta_center) / theta_fwhm) ** 2
            + (((phi_grid - phi_center + 180.0) % 360.0 - 180.0) / phi_fwhm) ** 2
        )
    )
    roi = plane + peak

    result = analysis_peak_tools.subtract_linear_background_plane(
        theta_axis,
        phi_axis,
        roi,
        [{"two_theta_deg": theta_center, "phi_deg": phi_center}],
        theta_exclusion_half_width=0.45,
        phi_exclusion_half_width=1.6,
    )

    corrected = np.asarray(result["corrected"], dtype=float)
    off_peak = (
        (np.abs(theta_grid - theta_center) > 0.65)
        | (np.abs(((phi_grid - phi_center + 180.0) % 360.0) - 180.0) > 2.2)
    )
    radial_profile = np.nanmean(corrected, axis=0)
    fit = analysis_peak_tools.fit_composite_peak_profile(
        theta_axis,
        radial_profile,
        [theta_center],
        model=analysis_peak_tools.PROFILE_GAUSSIAN,
    )

    assert result["success"] is True
    assert abs(float(np.nanmedian(corrected[off_peak]))) < 0.05
    assert fit["success"] is True
    assert abs(float(fit["components"][0]["center"]) - theta_center) < 0.03
    assert abs(float(fit["components"][0]["fwhm"]) - theta_fwhm) < 0.07


def test_subtract_linear_background_plane_rejects_broad_peak_wings_by_default() -> None:
    theta_axis = np.linspace(17.0, 19.0, 121)
    phi_axis = np.linspace(-4.0, 4.0, 101)
    theta_grid, phi_grid = np.meshgrid(theta_axis, phi_axis)
    theta_center = 18.15
    phi_center = 0.6
    theta_fwhm = 0.42
    phi_fwhm = 1.7
    plane = 9.5 + 2.3 * (theta_grid - 18.0) - 0.55 * (phi_grid - 0.25)
    peak = 32.0 * np.exp(
        -4.0
        * np.log(2.0)
        * (
            ((theta_grid - theta_center) / theta_fwhm) ** 2
            + (((phi_grid - phi_center + 180.0) % 360.0 - 180.0) / phi_fwhm) ** 2
        )
    )
    roi = plane + peak

    result = analysis_peak_tools.subtract_linear_background_plane(
        theta_axis,
        phi_axis,
        roi,
        [{"two_theta_deg": theta_center, "phi_deg": phi_center}],
    )

    corrected = np.asarray(result["corrected"], dtype=float)
    off_peak = (
        (np.abs(theta_grid - theta_center) > 0.72)
        | (np.abs(((phi_grid - phi_center + 180.0) % 360.0) - 180.0) > 2.6)
    )
    radial_profile = np.nanmean(corrected, axis=0)
    fit = analysis_peak_tools.fit_composite_peak_profile(
        theta_axis,
        radial_profile,
        [theta_center],
        model=analysis_peak_tools.PROFILE_GAUSSIAN,
    )

    assert result["success"] is True
    assert abs(float(np.nanmedian(corrected[off_peak]))) < 0.08
    assert result["fit_sample_count"] <= result["initial_fit_sample_count"]
    assert fit["success"] is True
    assert abs(float(fit["components"][0]["center"]) - theta_center) < 0.03
    assert abs(float(fit["components"][0]["fwhm"]) - theta_fwhm) < 0.08


def test_subtract_linear_background_plane_fails_when_peak_mask_consumes_roi() -> None:
    theta_axis = np.linspace(10.0, 10.2, 9)
    phi_axis = np.linspace(-0.2, 0.2, 9)
    roi = np.ones((phi_axis.size, theta_axis.size), dtype=float)

    result = analysis_peak_tools.subtract_linear_background_plane(
        theta_axis,
        phi_axis,
        roi,
        [{"two_theta_deg": 10.1, "phi_deg": 0.0}],
        theta_exclusion_half_width=1.0,
        phi_exclusion_half_width=1.0,
    )

    assert result["success"] is False
    assert "Not enough background samples" in str(result["error"])
    assert result["fit_sample_count"] == 0
    assert result["used_fallback_fit_mask"] is False


def test_match_selected_peak_index_uses_radial_and_azimuth_tolerances() -> None:
    peaks = [
        {"two_theta_deg": 12.34, "phi_deg": -8.7},
        {"two_theta_deg": 18.9, "phi_deg": 15.0},
    ]

    assert (
        analysis_peak_tools.match_selected_peak_index(
            peaks,
            two_theta_deg=12.37,
            phi_deg=-8.9,
            radial_tolerance_deg=0.05,
            azimuth_tolerance_deg=0.5,
        )
        == 0
    )
    assert (
        analysis_peak_tools.match_selected_peak_index(
            peaks,
            two_theta_deg=12.6,
            phi_deg=-8.9,
            radial_tolerance_deg=0.05,
            azimuth_tolerance_deg=0.5,
        )
        is None
    )


def test_format_peak_fit_axis_summary_returns_empty_text_for_no_results() -> None:
    assert analysis_peak_tools.format_peak_fit_axis_summary("Radial", []) == ""


def test_format_peak_fit_axis_summary_reports_best_success_from_mixed_results() -> None:
    summary = analysis_peak_tools.format_peak_fit_axis_summary(
        "Radial",
        [
            {
                "success": False,
                "peak_index": 1,
                "label": "Gaussian",
                "selected_axis_value": 17.5,
                "error": "window too small",
            },
            {
                "success": True,
                "peak_index": 3,
                "label": "Lorentzian",
                "selected_axis_value": 18.75,
                "center": 18.7,
                "fwhm": 0.42,
                "rmse": 0.18,
            },
            {
                "success": True,
                "peak_index": 2,
                "label": "Pseudo-Voigt",
                "selected_axis_value": 18.25,
                "center": 18.2,
                "fwhm": 0.34,
                "rmse": 0.05,
            },
        ],
    )

    assert (
        summary == "Radial: 2/3 fits; best P2 Pseudo-Voigt @ 18.2500 deg; "
        "center 18.2000, FWHM 0.3400, RMSE 0.05"
    )


def test_format_peak_fit_axis_summary_picks_lowest_rmse_success() -> None:
    summary = analysis_peak_tools.format_peak_fit_axis_summary(
        "Azimuth",
        [
            {
                "success": True,
                "peak_index": 4,
                "model": analysis_peak_tools.PROFILE_GAUSSIAN,
                "selected_axis_value": -22.0,
                "center": -21.95,
                "fwhm": 1.7,
                "rmse": 0.24,
            },
            {
                "success": True,
                "peak_index": 1,
                "model": analysis_peak_tools.PROFILE_GAUSSIAN,
                "selected_axis_value": -35.0,
                "center": -35.1,
                "fwhm": 1.1,
                "rmse": 0.03,
            },
        ],
    )

    assert summary.startswith("Azimuth: 2/2 fits; best P1 Gaussian @ -35.0000 deg;")
    assert "RMSE 0.03" in summary


def test_format_peak_fit_axis_summary_reports_last_failure_when_none_succeed() -> None:
    summary = analysis_peak_tools.format_peak_fit_axis_summary(
        "Azimuth",
        [
            {
                "success": False,
                "peak_index": 1,
                "label": "Gaussian",
                "selected_axis_value": -20.0,
                "error": "fit failed",
            },
            {
                "success": False,
                "peak_index": 3,
                "label": "Lorentzian",
                "selected_axis_value": -14.5,
                "error": "optimizer diverged",
            },
        ],
    )

    assert (
        summary
        == "Azimuth: 0/2 fits; last failure P3 Lorentzian @ -14.5000 deg: optimizer diverged"
    )


def test_format_peak_fit_axis_table_reports_widths_and_mixture_percent() -> None:
    table = analysis_peak_tools.format_peak_fit_axis_table(
        "Radial",
        [
            {
                "success": True,
                "peak_index": 1,
                "model": analysis_peak_tools.PROFILE_GAUSSIAN,
                "selected_axis_value": 18.25,
                "center": 18.2,
                "fwhm": 0.34,
                "rmse": 0.02,
            },
            {
                "success": True,
                "peak_index": 2,
                "model": analysis_peak_tools.PROFILE_LORENTZIAN,
                "selected_axis_value": 18.75,
                "center": 18.7,
                "fwhm": 0.42,
                "rmse": 0.05,
            },
            {
                "success": True,
                "peak_index": 3,
                "model": analysis_peak_tools.PROFILE_PSEUDO_VOIGT,
                "selected_axis_value": 19.0,
                "center": 18.95,
                "fwhm": 0.38,
                "eta": 0.65,
                "rmse": 0.03,
            },
            {
                "success": False,
                "peak_index": 4,
                "label": "Gaussian",
                "selected_axis_value": 19.5,
                "error": "window too small",
            },
        ],
    )

    assert table.splitlines()[0] == "Radial: 3/4 fits; best P1 Gaussian, RMSE 0.02"
    assert "G-FWHM" in table
    assert "L-FWHM" in table
    assert "G/L%" in table
    assert "P1   Gaussian" in table
    assert "P2   Lorentz" in table
    assert "P3   P-Voigt" in table
    assert "100.0/ 0.0" in table
    assert " 0.0/100.0" in table
    assert "35.0/65.0" in table
    assert "Failures:" in table
    assert "P4 Gaussian @ 19.5000 deg: window too small" in table


def test_fit_peak_profile_recovers_gaussian_center_and_fwhm() -> None:
    x_values = np.linspace(10.0, 20.0, 400)
    y_values = analysis_peak_tools.gaussian_profile(
        x_values,
        baseline=0.6,
        amplitude=7.5,
        center=15.25,
        fwhm=0.42,
    )

    fit = analysis_peak_tools.fit_peak_profile(
        x_values,
        y_values,
        center_guess=15.2,
        model=analysis_peak_tools.PROFILE_GAUSSIAN,
        window_half_width=0.9,
    )

    assert fit["success"] is True
    assert fit["label"] == "Gaussian"
    assert abs(float(fit["center"]) - 15.25) < 0.01
    assert abs(float(fit["fwhm"]) - 0.42) < 0.03
    assert abs(float(fit["sigma"]) - (0.42 / (2.0 * np.sqrt(2.0 * np.log(2.0))))) < 0.03


def test_fit_peak_profile_recovers_pseudo_voigt_eta() -> None:
    x_values = np.linspace(-5.0, 5.0, 500)
    y_values = _area_normalized_pseudo_voigt(
        x_values,
        baseline=1.2,
        area=5.0,
        center=-0.4,
        fwhm=1.3,
        eta=0.65,
    )

    fit = analysis_peak_tools.fit_peak_profile(
        x_values,
        y_values,
        center_guess=-0.3,
        model=analysis_peak_tools.PROFILE_PSEUDO_VOIGT,
        window_half_width=2.4,
    )

    assert fit["success"] is True
    assert fit["label"] == "Pseudo-Voigt"
    assert abs(float(fit["center"]) + 0.4) < 0.03
    assert abs(float(fit["fwhm"]) - 1.3) < 0.08
    assert abs(float(fit["eta"]) - 0.65) < 0.08


def test_pseudo_voigt_profile_integrates_to_area_and_eta_is_lorentzian_fraction() -> None:
    x_values = np.linspace(-250.0, 250.0, 50001)
    fwhm = 1.4
    eta = 0.7
    gaussian = analysis_peak_tools._gaussian_unit_area_profile(
        x_values,
        center=0.0,
        fwhm=fwhm,
    )
    lorentzian = analysis_peak_tools._lorentzian_unit_area_profile(
        x_values,
        center=0.0,
        fwhm=fwhm,
    )
    mixed = analysis_peak_tools.pseudo_voigt_profile(
        x_values,
        baseline=0.0,
        amplitude=6.5,
        center=0.0,
        fwhm=fwhm,
        eta=eta,
    )

    assert abs(float(np.trapezoid(gaussian, x_values)) - 1.0) < 1.0e-6
    assert abs(float(np.trapezoid(lorentzian, x_values)) - 1.0) < 0.004
    assert abs(float(np.trapezoid(mixed, x_values)) - 6.5) < 0.02
    assert float(np.trapezoid(eta * lorentzian, x_values)) > float(
        np.trapezoid((1.0 - eta) * gaussian, x_values)
    )


def test_fit_composite_peak_profile_single_peak_uses_full_curve() -> None:
    x_values = np.linspace(-5.0, 6.0, 550)
    y_values = analysis_peak_tools.gaussian_profile(
        x_values,
        baseline=0.4,
        amplitude=5.2,
        center=1.35,
        fwhm=0.62,
    )

    fit = analysis_peak_tools.fit_composite_peak_profile(
        x_values,
        y_values,
        [1.3],
        model=analysis_peak_tools.PROFILE_GAUSSIAN,
    )

    assert fit["success"] is True
    assert len(fit["components"]) == 1
    assert len(fit["x_fit"]) == len(x_values)
    assert np.isclose(float(fit["x_fit"][0]), float(x_values[0]))
    assert np.isclose(float(fit["x_fit"][-1]), float(x_values[-1]))
    assert abs(float(fit["components"][0]["center"]) - 1.35) < 0.03
    assert abs(float(fit["components"][0]["fwhm"]) - 0.62) < 0.08


def test_fit_composite_peak_profile_two_gaussians_sums_components() -> None:
    x_values = np.linspace(-6.0, 6.0, 601)
    y_values = (
        0.8
        + analysis_peak_tools.gaussian_profile(
            x_values,
            baseline=0.0,
            amplitude=4.5,
            center=-1.4,
            fwhm=0.55,
        )
        + analysis_peak_tools.gaussian_profile(
            x_values,
            baseline=0.0,
            amplitude=3.8,
            center=1.8,
            fwhm=0.9,
        )
    )

    fit = analysis_peak_tools.fit_composite_peak_profile(
        x_values,
        y_values,
        [-1.35, 1.75],
        model=analysis_peak_tools.PROFILE_GAUSSIAN,
    )

    fitted_centers = [float(component["center"]) for component in fit["components"]]

    assert fit["success"] is True
    assert len(fit["components"]) == 2
    assert len(fit["x_fit"]) == len(x_values)
    assert np.isclose(float(fit["x_fit"][0]), float(x_values[0]))
    assert np.isclose(float(fit["x_fit"][-1]), float(x_values[-1]))
    assert np.allclose(fitted_centers, [-1.4, 1.8], atol=0.08)
    assert float(fit["rmse"]) < 0.1


def test_fit_composite_peak_profile_pseudo_voigt_returns_eta_and_full_curve() -> None:
    x_values = np.linspace(-4.0, 4.0, 480)
    y_values = _area_normalized_pseudo_voigt(
        x_values,
        baseline=0.9,
        area=4.2,
        center=-0.6,
        fwhm=1.1,
        eta=0.72,
    )

    fit = analysis_peak_tools.fit_composite_peak_profile(
        x_values,
        y_values,
        [-0.55],
        model=analysis_peak_tools.PROFILE_PSEUDO_VOIGT,
    )

    assert fit["success"] is True
    assert len(fit["components"]) == 1
    assert len(fit["x_fit"]) == len(x_values)
    assert abs(float(fit["components"][0]["center"]) + 0.6) < 0.05
    assert abs(float(fit["components"][0]["fwhm"]) - 1.1) < 0.12
    assert abs(float(fit["components"][0]["eta"]) - 0.72) < 0.1


def test_fit_composite_peak_profile_collapses_near_duplicate_centers() -> None:
    x_values = np.linspace(-2.5, 2.5, 500)
    y_values = analysis_peak_tools.gaussian_profile(
        x_values,
        baseline=0.3,
        amplitude=4.8,
        center=0.12,
        fwhm=0.45,
    )

    fit = analysis_peak_tools.fit_composite_peak_profile(
        x_values,
        y_values,
        [0.12, 0.125],
        model=analysis_peak_tools.PROFILE_GAUSSIAN,
    )

    assert fit["success"] is True
    assert len(fit["components"]) == 1
    assert fit["component_groups"] == [{"component_index": 0, "center_guess_indices": [0, 1]}]


def test_fit_composite_peak_profile_does_not_use_samples_outside_selected_window() -> None:
    x_values = np.linspace(-10.0, 10.0, 801)
    y_values = analysis_peak_tools.gaussian_profile(
        x_values,
        baseline=0.5,
        amplitude=3.2,
        center=1.1,
        fwhm=0.85,
    )
    selected_mask = (x_values >= -2.0) & (x_values <= 3.0)
    selected_x = x_values[selected_mask]
    selected_y = y_values[selected_mask]

    fit = analysis_peak_tools.fit_composite_peak_profile(
        selected_x,
        selected_y,
        [1.05],
        model=analysis_peak_tools.PROFILE_GAUSSIAN,
    )

    assert fit["success"] is True
    assert np.allclose(np.asarray(fit["x_fit"], dtype=float), selected_x)
    assert float(np.min(fit["x_fit"])) >= -2.0
    assert float(np.max(fit["x_fit"])) <= 3.0


def test_fit_composite_peak_profile_handles_wrapped_selected_azimuth_window() -> None:
    x_values = np.concatenate(
        (
            np.linspace(-179.5, -170.0, 96),
            np.linspace(170.0, 179.5, 96),
        )
    )
    baseline = 0.35
    amplitude = 4.1
    center = 179.1
    fwhm = 4.8

    wrapped_delta = ((x_values - center + 180.0) % 360.0) - 180.0
    y_values = baseline + amplitude * np.exp(-4.0 * np.log(2.0) * (wrapped_delta / fwhm) ** 2)

    fit = analysis_peak_tools.fit_composite_peak_profile(
        x_values,
        y_values,
        [178.8],
        model=analysis_peak_tools.PROFILE_GAUSSIAN,
    )

    fit_center = float(fit["components"][0]["center"])
    wrapped_center_error = ((fit_center - center + 180.0) % 360.0) - 180.0

    assert fit["success"] is True
    assert len(fit["components"]) == 1
    assert len(fit["x_fit"]) == len(x_values)
    assert np.all(np.asarray(fit["x_fit"], dtype=float) >= np.min(x_values))
    assert np.all(np.asarray(fit["x_fit"], dtype=float) <= np.max(x_values))
    assert abs(wrapped_center_error) < 0.3
    assert np.min(np.abs(np.asarray(fit["x_fit"], dtype=float) - fit_center)) < 0.25
    assert abs(float(fit["components"][0]["fwhm"]) - fwhm) < 0.5


def test_fit_composite_peak_profile_reports_wrapped_center_on_selected_axis_domain() -> None:
    x_values = np.concatenate(
        (
            np.linspace(-179.5, -170.0, 96),
            np.linspace(170.0, 179.5, 96),
        )
    )
    baseline = 0.35
    amplitude = 4.1
    center = -179.1
    fwhm = 4.8

    wrapped_delta = ((x_values - center + 180.0) % 360.0) - 180.0
    y_values = baseline + amplitude * np.exp(-4.0 * np.log(2.0) * (wrapped_delta / fwhm) ** 2)

    fit = analysis_peak_tools.fit_composite_peak_profile(
        x_values,
        y_values,
        [179.0],
        model=analysis_peak_tools.PROFILE_GAUSSIAN,
    )

    fit_center = float(fit["components"][0]["center"])
    wrapped_center_error = ((fit_center - center + 180.0) % 360.0) - 180.0

    assert fit["success"] is True
    assert abs(wrapped_center_error) < 0.3
    assert np.min(np.abs(np.asarray(fit["x_fit"], dtype=float) - fit_center)) < 0.25
    assert abs(float(fit["components"][0]["fwhm"]) - fwhm) < 0.5
