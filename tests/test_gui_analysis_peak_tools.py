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
