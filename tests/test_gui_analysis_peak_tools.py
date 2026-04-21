import numpy as np

from ra_sim.gui import analysis_peak_tools


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
        summary
        == "Radial: 2/3 fits; best P2 Pseudo-Voigt @ 18.2500 deg; "
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
    y_values = analysis_peak_tools.pseudo_voigt_profile(
        x_values,
        baseline=1.2,
        amplitude=5.0,
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
    y_values = analysis_peak_tools.pseudo_voigt_profile(
        x_values,
        baseline=0.9,
        amplitude=4.2,
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
    y_values = baseline + amplitude * np.exp(
        -4.0 * np.log(2.0) * (wrapped_delta / fwhm) ** 2
    )

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
    y_values = baseline + amplitude * np.exp(
        -4.0 * np.log(2.0) * (wrapped_delta / fwhm) ** 2
    )

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
