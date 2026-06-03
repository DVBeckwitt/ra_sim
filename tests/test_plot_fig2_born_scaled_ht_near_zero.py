from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "plot_fig2_born_scaled_ht_near_zero.py"
)


def load_script_module():
    spec = importlib.util.spec_from_file_location("plot_fig2_born_scaled_ht_near_zero", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_table_defaults_match_attached_parameters() -> None:
    module = load_script_module()

    assert module.DEFAULTS.qc_inv_angstrom == pytest.approx(0.0517)
    assert module.DEFAULTS.substrate_qc_inv_angstrom == pytest.approx(0.0305)
    assert module.DEFAULTS.film_density_e_per_a3 == pytest.approx(1.8870)
    assert module.DEFAULTS.substrate_density_e_per_a3 == pytest.approx(0.6567)
    assert module.DEFAULTS.thickness_nm == pytest.approx(50.0)
    assert module.DEFAULTS.top_roughness_angstrom == pytest.approx(0.0)
    assert module.DEFAULTS.bottom_roughness_angstrom == pytest.approx(0.0)
    assert module.DEFAULTS.bandwidth_fwhm == pytest.approx(0.005)
    assert module.DEFAULTS.n_wavelength_samples == 20
    assert module.DEFAULTS.divergence_fwhm_deg == pytest.approx(0.5)
    assert module.DEFAULTS.n_divergence_samples == 50
    assert module.DEFAULTS.ht_structure_scale == pytest.approx(1.0)
    assert module.DEFAULTS.ht_over_q2_scale == pytest.approx(1.0)


def test_curve_contract_includes_bare_ht_structure_term() -> None:
    module = load_script_module()

    assert module.HT_STRUCTURE_LABEL in module.FIG2_CURVE_ORDER
    assert module.FIG2_CURVE_ORDER[0] == module.HT_STRUCTURE_LABEL
    assert module.HT_OVER_Q2_LABEL in module.FIG2_CURVE_ORDER
    assert module.DEFAULT_GUI_VISIBLE_CURVE_LABELS == (
        module.HT_STRUCTURE_LABEL,
        module.HT_OVER_Q2_LABEL,
        module.PURE_PARRATT_LABEL,
    )
    assert module.FIG2_CURVE_DISPLAY_LABELS[module.HT_STRUCTURE_LABEL] == (
        r"$S_{\mathrm{HT},0}(L)$"
    )
    assert module.FIG2_CURVE_DISPLAY_LABELS[module.HT_OVER_Q2_LABEL] == (
        r"$S_{\mathrm{HT},0}(L) / Q_z^2$"
    )
    assert module.HT_STRUCTURE_LABEL in module.FIG2_CURVE_STYLE
    assert module.HT_OVER_Q2_LABEL in module.FIG2_CURVE_STYLE


def _synthetic_ht_parratt_curves(module, l_values, ht_over_q2, parratt):
    qz = module.qz_from_L(l_values, module.DEFAULTS.c_angstrom)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "L": l_values,
                    "Qz_Ainv": qz,
                    "intensity": ht_over_q2,
                    "label": module.HT_OVER_Q2_LABEL,
                }
            ),
            pd.DataFrame(
                {
                    "L": l_values,
                    "Qz_Ainv": qz,
                    "intensity": parratt,
                    "label": module.PURE_PARRATT_LABEL,
                }
            ),
        ],
        ignore_index=True,
    )


def _synthetic_stitch_curves(
    module,
    qz,
    *,
    pure,
    unscaled,
    scaled,
    stitched,
    weight,
):
    l_values = module.L_from_qz(qz, module.DEFAULTS.c_angstrom)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "L": l_values,
                    "Qz_Ainv": qz,
                    "intensity": intensity,
                    "label": label,
                }
            )
            for label, intensity in (
                (module.PURE_PARRATT_LABEL, pure),
                (module.STITCH_UNSCALED_HT_OVER_Q2_LABEL, unscaled),
                (module.STITCH_SCALED_HT_OVER_Q2_LABEL, scaled),
                (module.STITCHED_PARRATT_HT_OVER_Q2_LABEL, stitched),
                (module.STITCH_WEIGHT_LABEL, weight),
            )
        ],
        ignore_index=True,
    )


def test_fit_ht_over_q2_scale_to_parratt_matches_smooth_synthetic_curve() -> None:
    module = load_script_module()
    l_values = np.linspace(1.0, 10.0, 300)
    parratt = 1.0 / (1.0 + 0.25 * (l_values - 1.0))
    ht_base = parratt / 2.5
    curves = _synthetic_ht_parratt_curves(module, l_values, ht_base, parratt)

    fitted_scale = module.fit_ht_over_q2_scale_to_parratt(curves)

    assert fitted_scale == pytest.approx(2.5, rel=1e-6)


def test_fit_ht_over_q2_scale_removes_pseudo_voigt_peaks_before_scaling() -> None:
    module = load_script_module()
    l_values = np.linspace(1.0, 10.0, 1801)
    parratt = 1.0 / (1.0 + 0.18 * (l_values - 1.0))
    expected_scale = 1.7
    ht_base = parratt / expected_scale
    peak_sum = sum(
        module.pseudo_voigt_profile(l_values, center, 1.2, 0.08, 0.35) for center in (3.0, 6.0, 9.0)
    )
    curves = _synthetic_ht_parratt_curves(module, l_values, ht_base + peak_sum, parratt)

    fitted_scale = module.fit_ht_over_q2_scale_to_parratt(curves)

    uncorrected_scale = np.sum((ht_base + peak_sum) * parratt) / np.sum((ht_base + peak_sum) ** 2)
    assert abs(fitted_scale - expected_scale) < abs(uncorrected_scale - expected_scale)
    assert fitted_scale == pytest.approx(expected_scale, rel=0.08)


def test_fit_ht_over_q2_scale_falls_back_when_peak_subtraction_consumes_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    l_values = np.linspace(1.0, 10.0, 300)
    parratt = 1.0 / (1.0 + 0.2 * (l_values - 1.0))
    expected_scale = 2.5
    ht_base = parratt / expected_scale
    curves = _synthetic_ht_parratt_curves(module, l_values, ht_base, parratt)

    def _consume_background(*args, **kwargs):
        _l_values, observed, _background = args[:3]
        return np.asarray(observed, dtype=np.float64) * 1.1

    monkeypatch.setattr(module, "fit_pseudo_voigt_peak_sum", _consume_background)

    fitted_scale = module.fit_ht_over_q2_scale_to_parratt(curves)

    assert fitted_scale == pytest.approx(expected_scale, rel=1e-6)


def test_fit_ht_over_q2_scale_accounts_for_current_curve_scale() -> None:
    module = load_script_module()
    l_values = np.linspace(1.0, 10.0, 300)
    parratt = 1.0 / (1.0 + 0.2 * (l_values - 1.0))
    expected_scale = 2.0
    current_scale = 4.0
    stored_ht = (parratt / expected_scale) * current_scale
    curves = _synthetic_ht_parratt_curves(module, l_values, stored_ht, parratt)

    fitted_scale = module.fit_ht_over_q2_scale_to_parratt(curves, current_scale=current_scale)

    assert fitted_scale == pytest.approx(expected_scale, rel=1e-6)


def test_fit_ht_over_q2_scale_requires_full_l_range() -> None:
    module = load_script_module()
    l_values = np.linspace(0.0, 0.5, 50)
    curves = _synthetic_ht_parratt_curves(
        module,
        l_values,
        np.ones_like(l_values),
        np.ones_like(l_values),
    )

    with pytest.raises(ValueError, match="1.0 <= L <= 10.0"):
        module.fit_ht_over_q2_scale_to_parratt(curves)


def test_curves_span_l_range_detects_autofit_window() -> None:
    module = load_script_module()
    narrow_l = np.linspace(0.0, 0.5, 50)
    wide_l = np.linspace(1.0, 10.0, 300)

    narrow_curves = _synthetic_ht_parratt_curves(
        module,
        narrow_l,
        np.ones_like(narrow_l),
        np.ones_like(narrow_l),
    )
    wide_curves = _synthetic_ht_parratt_curves(
        module,
        wide_l,
        np.ones_like(wide_l),
        np.ones_like(wide_l),
    )

    assert not module.curves_span_l_range(narrow_curves, 1.0, 10.0)
    assert module.curves_span_l_range(wide_curves, 1.0, 10.0)


def test_curves_span_l_range_allows_float_grid_endpoint_roundoff() -> None:
    module = load_script_module()
    l_values = np.arange(1.0, 10.0 + 0.5 * 0.0005, 0.0005)
    curves = _synthetic_ht_parratt_curves(
        module,
        l_values,
        np.ones_like(l_values),
        np.ones_like(l_values),
    )

    assert l_values[-1] < 10.0
    assert module.curves_span_l_range(curves, 1.0, 10.0)


def test_autofit_args_for_l_range_clones_args_without_changing_visible_range() -> None:
    module = load_script_module()
    args = module.parse_args(["--from-csv", "curves.csv", "--L-min", "0", "--L-max", "0.5"])

    fit_args = module.autofit_args_for_l_range(args, 1.0, 10.0)

    assert fit_args.L_min == pytest.approx(1.0)
    assert fit_args.L_max == pytest.approx(10.0)
    assert fit_args.from_csv is None
    assert args.L_min == pytest.approx(0.0)
    assert args.L_max == pytest.approx(0.5)
    assert args.from_csv == "curves.csv"


def test_fit_pseudo_voigt_peak_sum_uses_parratt_background() -> None:
    module = load_script_module()
    l_values = np.linspace(2.5, 3.5, 501)
    background = 0.4 + 0.02 * (l_values - 3.0)
    peak = module.pseudo_voigt_profile(l_values, 3.0, 0.9, 0.07, 0.4)
    observed = background + peak

    fitted_peak = module.fit_pseudo_voigt_peak_sum(
        l_values,
        observed,
        background,
        peak_centers=(3.0,),
    )

    before = np.mean((observed - background) ** 2)
    after = np.mean((observed - fitted_peak - background) ** 2)
    assert np.all(fitted_peak >= 0.0)
    assert after < before * 0.05


def test_ht_over_q2_positive_qz_division_masks_zero_qz() -> None:
    module = load_script_module()

    intensity = module.ht_over_q2_positive_qz_division(
        np.array([1.0, 0.8]),
        np.array([0.0, 0.02]),
        qz_min=1.0e-8,
        scale=2.0,
    )

    assert np.isnan(intensity[0])
    assert intensity[1] == pytest.approx(0.8 / 0.02**2 * 2.0)


def test_divergence_safe_ht_over_q2_average_renormalizes_valid_samples() -> None:
    module = load_script_module()
    structure_samples = np.array(
        [
            [10.0, 20.0],
            [1.0, 2.0],
            [4.0, 8.0],
        ]
    )
    qz_samples = np.array(
        [
            [-0.1, 0.0],
            [0.0, 0.2],
            [0.1, 0.4],
        ]
    )
    divergence_weights = np.array([0.2, 0.3, 0.5])

    averaged = module.divergence_safe_ht_over_q2_average(
        structure_samples,
        qz_samples,
        divergence_weights,
        qz_min=1.0e-8,
    )

    assert averaged[0] == pytest.approx(4.0 / 0.1**2)
    expected_second = (0.3 * (2.0 / 0.2**2) + 0.5 * (8.0 / 0.4**2)) / (0.3 + 0.5)
    assert averaged[1] == pytest.approx(expected_second)


def test_stitch_helper_math_contract() -> None:
    module = load_script_module()

    assert module.parse_float_list("") == ()
    assert module.parse_float_list("3, 6,9") == pytest.approx((3.0, 6.0, 9.0))
    assert module.surface_cell_area_hex(4.0) == pytest.approx(0.5 * np.sqrt(3.0) * 16.0)
    assert module.miceli_cell_scale(4.0) > 0.0

    mask = module.bragg_exclusion_mask(
        np.array([2.9, 3.0, 3.4, 4.0, 6.0]),
        centers=(3.0, 6.0),
        half_width=0.2,
    )
    assert mask.tolist() == [False, False, True, True, False]

    model = np.array([1.0, 2.0, 4.0, 8.0])
    target = 3.5 * model
    assert module.fit_log_median_scale(model, target, np.ones_like(model, dtype=bool)) == (
        pytest.approx(3.5)
    )

    x = np.array([1.0, 2.0, 3.5, 5.0, 6.0])
    weight = module.smooth_stitch_weight(x, x1=2.0, x2=5.0)
    assert weight[0] == pytest.approx(1.0)
    assert weight[1] == pytest.approx(1.0)
    assert 0.0 < weight[2] < 1.0
    assert weight[3] == pytest.approx(0.0)
    assert weight[4] == pytest.approx(0.0)

    low = np.array([1.0, 10.0])
    high = np.array([100.0, 1000.0])
    assert np.allclose(module.log_blend(low, high, np.ones(2)), low)
    assert np.allclose(module.log_blend(low, high, np.zeros(2)), high)


def test_stitch_cli_defaults_and_choices() -> None:
    module = load_script_module()

    default_args = module.parse_args([])
    assert default_args.plot_mode == "fig2"
    assert not default_args.stitch
    assert default_args.stitch_x1_q_over_qc == pytest.approx(2.0)
    assert default_args.stitch_x2_q_over_qc == pytest.approx(5.0)
    assert default_args.stitch_branch == "normalized-ht-q2"
    assert default_args.scale_mode == "log-median"
    assert default_args.fit_exclude_centers_L == "3,6,9"

    stitch_args = module.parse_args(
        [
            "--stitch",
            "--plot-mode",
            "stitch",
            "--scale-mode",
            "manual",
            "--manual-stitch-scale",
            "2.5",
        ]
    )
    assert stitch_args.stitch
    assert stitch_args.plot_mode == "stitch"
    assert stitch_args.scale_mode == "manual"
    assert stitch_args.manual_stitch_scale == pytest.approx(2.5)


def test_default_stitch_computes_hidden_fit_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_script_module()

    def _fake_ht_module(**kwargs):
        l_max = float(kwargs["L_max"])
        return {(0, 0): {"L": np.linspace(0.0, l_max, 200), "I": np.ones(200)}}

    monkeypatch.setattr(module, "import_ht_module", lambda _repo_root: _fake_ht_module)
    args = module.parse_args(
        [
            "--plot-mode",
            "stitch",
            "--L-step",
            "0.1",
            "--bandwidth-fwhm",
            "0",
            "--n-wavelength-samples",
            "1",
            "--divergence-fwhm-deg",
            "0",
            "--n-divergence-samples",
            "1",
        ]
    )

    curves = module.compute_curves(args)
    stitch_l = curves[curves["label"] == module.STITCHED_PARRATT_HT_OVER_Q2_LABEL]["L"]

    assert module.STITCHED_PARRATT_HT_OVER_Q2_LABEL in set(curves["label"])
    assert stitch_l.max() > args.L_max


def test_compute_curves_ht_over_q2_avoids_clipped_zero_divergence_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()

    def _fake_ht_module(**_kwargs):
        return {(0, 0): {"L": np.linspace(0.0, 1.0, 11), "I": np.ones(11)}}

    monkeypatch.setattr(module, "import_ht_module", lambda _repo_root: _fake_ht_module)
    args = module.parse_args(
        [
            "--L-min",
            "0",
            "--L-max",
            "0.6",
            "--L-step",
            "0.3",
            "--bandwidth-fwhm",
            "0",
            "--n-wavelength-samples",
            "1",
            "--divergence-fwhm-deg",
            "0.5",
            "--n-divergence-samples",
            "3",
        ]
    )

    curves = module.compute_curves(args)
    ht_over_q2 = curves[curves["label"] == module.HT_OVER_Q2_LABEL].sort_values("L")

    assert ht_over_q2["intensity"].iloc[0] < 1.0e10


def test_compute_curves_skips_raw_miceli_branch_unless_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()

    def _fake_ht_module(**_kwargs):
        return {(0, 0): {"L": np.linspace(0.0, 1.0, 11), "I": np.ones(11)}}

    calls: list[object] = []
    original_average = module.divergence_safe_ht_over_q2_average

    def _counting_average(*args, **kwargs):
        calls.append(args[0])
        return original_average(*args, **kwargs)

    monkeypatch.setattr(module, "import_ht_module", lambda _repo_root: _fake_ht_module)
    monkeypatch.setattr(module, "divergence_safe_ht_over_q2_average", _counting_average)

    common_args = [
        "--L-min",
        "0.1",
        "--L-max",
        "0.3",
        "--L-step",
        "0.1",
        "--bandwidth-fwhm",
        "0",
        "--n-wavelength-samples",
        "1",
        "--divergence-fwhm-deg",
        "0",
        "--n-divergence-samples",
        "1",
    ]
    module.compute_curves(module.parse_args(common_args))
    assert len(calls) == 1

    calls.clear()
    module.compute_curves(
        module.parse_args(
            [
                *common_args,
                "--stitch",
                "--stitch-branch",
                "raw-miceli-ht-q2",
                "--scale-mode",
                "manual",
            ]
        )
    )
    assert len(calls) == 2


def test_stitch_curves_are_emitted_and_match_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_script_module()

    def _fake_ht_module(**_kwargs):
        return {(0, 0): {"L": np.linspace(0.0, 4.0, 41), "I": np.ones(41)}}

    monkeypatch.setattr(module, "import_ht_module", lambda _repo_root: _fake_ht_module)
    args = module.parse_args(
        [
            "--stitch",
            "--plot-mode",
            "stitch",
            "--scale-mode",
            "manual",
            "--manual-stitch-scale",
            "2.0",
            "--L-min",
            "0.2",
            "--L-max",
            "3.0",
            "--L-step",
            "0.25",
            "--bandwidth-fwhm",
            "0",
            "--n-wavelength-samples",
            "1",
            "--divergence-fwhm-deg",
            "0",
            "--n-divergence-samples",
            "1",
        ]
    )

    curves = module.compute_curves(args)
    labels = set(curves["label"])
    assert module.STITCH_UNSCALED_HT_OVER_Q2_LABEL in labels
    assert module.STITCH_SCALED_HT_OVER_Q2_LABEL in labels
    assert module.STITCHED_PARRATT_HT_OVER_Q2_LABEL in labels
    assert module.STITCH_WEIGHT_LABEL in labels

    parratt = curves[curves["label"] == module.PURE_PARRATT_LABEL].sort_values("L")
    scaled = curves[curves["label"] == module.STITCH_SCALED_HT_OVER_Q2_LABEL].sort_values("L")
    stitched = curves[curves["label"] == module.STITCHED_PARRATT_HT_OVER_Q2_LABEL].sort_values("L")
    weight = curves[curves["label"] == module.STITCH_WEIGHT_LABEL].sort_values("L")

    weights = weight["intensity"].to_numpy()
    low_index = int(np.flatnonzero(np.isclose(weights, 1.0))[0])
    high_index = int(np.flatnonzero(np.isclose(weights, 0.0))[0])
    assert stitched["intensity"].iloc[low_index] == pytest.approx(
        parratt["intensity"].iloc[low_index]
    )
    assert stitched["intensity"].iloc[high_index] == pytest.approx(
        scaled["intensity"].iloc[high_index]
    )


def test_curve_labels_for_plot_modes() -> None:
    module = load_script_module()

    assert module.curve_labels_for_plot_mode(module.parse_args([])) == module.FIG2_CURVE_ORDER
    assert module.curve_labels_for_plot_mode(module.parse_args(["--plot-mode", "stitch"])) == (
        module.PURE_PARRATT_LABEL,
        module.STITCH_SCALED_HT_OVER_Q2_LABEL,
        module.STITCHED_PARRATT_HT_OVER_Q2_LABEL,
    )
    diagnostic_labels = module.curve_labels_for_plot_mode(
        module.parse_args(["--plot-mode", "stitch-diagnostics"])
    )
    assert module.STITCH_UNSCALED_HT_OVER_Q2_LABEL in diagnostic_labels
    assert module.STITCH_WEIGHT_LABEL in diagnostic_labels


def test_write_stitch_diagnostics_reports_residual_warning(tmp_path, capsys) -> None:
    module = load_script_module()
    qz = module.DEFAULTS.qc_inv_angstrom * np.array([5.0, 6.0, 7.0, 8.0])
    pure = np.ones_like(qz)
    scaled = np.array([1.0, 10.0, 100.0, 1000.0])
    curves = _synthetic_stitch_curves(
        module,
        qz,
        pure=pure,
        unscaled=np.ones_like(qz),
        scaled=scaled,
        stitched=scaled,
        weight=np.zeros_like(qz),
    )
    args = module.parse_args(["--stitch", "--write-stitch-diagnostics"])

    csv_path = module.write_stitch_diagnostics(curves, args, tmp_path)

    diagnostics = pd.read_csv(csv_path)
    assert set(
        [
            "material",
            "stitch_branch",
            "scale_mode",
            "stitch_scale",
            "n_fit_points",
            "median_log10_residual",
            "mad_log10_residual",
        ]
    ).issubset(diagnostics.columns)
    assert diagnostics["n_fit_points"].iloc[0] == 4
    assert diagnostics["mad_log10_residual"].iloc[0] > 0.3
    assert "visual composite" in capsys.readouterr().out


def test_write_stitch_diagnostic_plot_creates_file(tmp_path) -> None:
    module = load_script_module()
    qz = module.DEFAULTS.qc_inv_angstrom * np.array([5.0, 6.0, 7.0, 8.0])
    curves = _synthetic_stitch_curves(
        module,
        qz,
        pure=np.ones_like(qz),
        unscaled=np.ones_like(qz),
        scaled=np.ones_like(qz),
        stitched=np.ones_like(qz),
        weight=np.zeros_like(qz),
    )

    figure_path = module.write_stitch_diagnostic_plot(
        curves, module.parse_args(["--stitch"]), tmp_path
    )

    assert figure_path.name == "fig_stitch_scale_diagnostic.png"
    assert figure_path.exists()


@pytest.mark.parametrize("qc", [0.0305, 0.0517])
def test_qc_density_conversion_round_trips(qc: float) -> None:
    module = load_script_module()

    density = module.density_e_per_a3_from_qc(qc)
    assert density > 0
    assert module.qc_from_density_e_per_a3(density) == pytest.approx(qc)


def test_parratt_stack_uses_top_and_bottom_roughness() -> None:
    module = load_script_module()
    args = module.parse_args(
        [
            "--top-roughness-angstrom",
            "7.5",
            "--bottom-roughness-angstrom",
            "9.5",
        ]
    )

    stack = module.make_air_film_substrate_stack(args)

    assert stack[0].name == "air"
    assert stack[1].name == "Bi2Se3"
    assert stack[1].roughness_angstrom == pytest.approx(7.5)
    assert stack[2].name == "SiO2"
    assert stack[2].roughness_angstrom == pytest.approx(9.5)


def test_xlim_validation_keeps_positive_width() -> None:
    module = load_script_module()

    assert module.validated_l_limits(0.02, 0.2) == pytest.approx((0.02, 0.2))
    assert module.validated_l_limits(0.4, 0.2) == pytest.approx((0.4, 0.401))


def test_expanded_slider_bounds_includes_typed_value() -> None:
    module = load_script_module()

    assert module.expanded_slider_bounds(0.02, 1.0, 5.0) == pytest.approx((0.02, 5.0))
    assert module.expanded_slider_bounds(50.0, 2000.0, 5000.0) == pytest.approx((50.0, 5000.0))
    assert module.expanded_slider_bounds(0.02, 1.0, -0.5) == pytest.approx((-0.5, 1.0))
    assert module.expanded_slider_bounds(0.02, 1.0, 0.5) == pytest.approx((0.02, 1.0))


def test_gui_flag_routes_to_gui_launcher(monkeypatch) -> None:
    module = load_script_module()
    calls: list[object] = []

    monkeypatch.setattr(module, "run_fig2_gui", lambda args: calls.append(args))
    monkeypatch.setattr(
        module, "compute_curves", lambda args: pytest.fail("GUI should own compute")
    )
    monkeypatch.setattr(
        module, "plot_fig2", lambda curves, args: pytest.fail("GUI should own plot")
    )

    module.main(["--gui"])

    assert len(calls) == 1


def test_gui_parameter_state_updates_args_with_thickness_in_angstrom() -> None:
    module = load_script_module()
    args = module.parse_args([])

    state = module.Fig2GuiParameterState.from_args(args)
    state.thickness_angstrom = 750.0
    state.film_qc_inv_angstrom = 0.0600
    state.bandwidth_percent = 7.5
    state.n_wavelength_samples = 123.0
    state.divergence_fwhm_deg = 0.75
    state.n_divergence_samples = 99.0
    state.ht_structure_scale = 4.0
    state.ht_over_q2_scale = 3.5
    state.sync_density_from_qc("film")
    updated = state.updated_args(args)

    assert updated.thickness_nm == pytest.approx(75.0)
    assert updated.qc_inv_angstrom == pytest.approx(0.0600)
    assert updated.film_density_e_per_a3 == pytest.approx(module.density_e_per_a3_from_qc(0.0600))
    assert updated.bandwidth_fwhm == pytest.approx(0.075)
    assert updated.n_wavelength_samples == 123
    assert updated.divergence_fwhm_deg == pytest.approx(0.75)
    assert updated.n_divergence_samples == 99
    assert updated.ht_structure_scale == pytest.approx(4.0)
    assert updated.ht_over_q2_scale == pytest.approx(3.5)


def test_gui_parameter_state_can_sync_qc_from_density() -> None:
    module = load_script_module()
    args = module.parse_args([])
    state = module.Fig2GuiParameterState.from_args(args)

    state.substrate_density_e_per_a3 = 1.0
    state.sync_qc_from_density("substrate")

    assert state.substrate_qc_inv_angstrom == pytest.approx(module.qc_from_density_e_per_a3(1.0))


def test_draw_fig2_curves_applies_xlim_and_secondary_axis() -> None:
    module = load_script_module()
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure

    args = module.parse_args(["--L-min", "0.02", "--L-max", "0.25"])
    l_values = np.array([0.02, 0.10, 0.25])
    curves = pd.concat(
        [
            pd.DataFrame(
                {
                    "L": l_values,
                    "Qz_Ainv": module.qz_from_L(l_values, args.c_angstrom),
                    "intensity": np.array([1.0, 0.5, 0.25]),
                    "label": label,
                }
            )
            for label in module.FIG2_CURVE_ORDER
        ],
        ignore_index=True,
    )
    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot()

    module.draw_fig2_curves(ax, curves, args)

    assert ax.get_xlim() == pytest.approx((0.02, 0.25))
    curve_lines = ax.lines[: len(module.FIG2_CURVE_ORDER)]
    assert [line.get_label() for line in curve_lines] == [
        module.FIG2_CURVE_DISPLAY_LABELS[label] for label in module.FIG2_CURVE_ORDER
    ]
    assert ax.get_yscale() == "log"
    assert ax.get_ylabel() == r"$I/I_0$ or $R$"
    assert len(ax.child_axes) == 1


def test_draw_fig2_curves_can_use_linear_y_axis() -> None:
    module = load_script_module()
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure

    args = module.parse_args(["--L-min", "0.02", "--L-max", "0.25"])
    l_values = np.array([0.02, 0.10, 0.25])
    curves = pd.concat(
        [
            pd.DataFrame(
                {
                    "L": l_values,
                    "Qz_Ainv": module.qz_from_L(l_values, args.c_angstrom),
                    "intensity": np.array([1.0, 0.5, 0.25]),
                    "label": label,
                }
            )
            for label in module.FIG2_CURVE_ORDER
        ],
        ignore_index=True,
    )
    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot()

    module.draw_fig2_curves(ax, curves, args, use_log_y_axis=False)

    assert ax.get_yscale() == "linear"


def test_draw_fig2_curves_can_hide_unselected_curves() -> None:
    module = load_script_module()
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure

    args = module.parse_args(["--L-min", "0.02", "--L-max", "0.25"])
    l_values = np.array([0.02, 0.10, 0.25])
    curves = pd.concat(
        [
            pd.DataFrame(
                {
                    "L": l_values,
                    "Qz_Ainv": module.qz_from_L(l_values, args.c_angstrom),
                    "intensity": np.array([1.0, 0.5, 0.25]),
                    "label": label,
                }
            )
            for label in module.FIG2_CURVE_ORDER
        ],
        ignore_index=True,
    )
    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot()
    selected_label = module.FIG2_CURVE_ORDER[1]

    module.draw_fig2_curves(ax, curves, args, visible_curve_labels={selected_label})

    plotted_labels = [line.get_label() for line in ax.lines]
    assert module.FIG2_CURVE_DISPLAY_LABELS[selected_label] in plotted_labels
    for hidden_label in set(module.FIG2_CURVE_ORDER) - {selected_label}:
        assert module.FIG2_CURVE_DISPLAY_LABELS[hidden_label] not in plotted_labels


def test_draw_fig2_curves_adds_bragg_markers_when_enabled() -> None:
    module = load_script_module()
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure

    args = module.parse_args(
        ["--L-min", "0", "--L-max", "9", "--show-bragg-markers", "--bragg-marker-L", "3", "6"]
    )
    l_values = np.array([0.0, 3.0, 6.0])
    curves = pd.concat(
        [
            pd.DataFrame(
                {
                    "L": l_values,
                    "Qz_Ainv": module.qz_from_L(l_values, args.c_angstrom),
                    "intensity": np.array([1.0, 0.8, 0.5]),
                    "label": label,
                }
            )
            for label in module.FIG2_CURVE_ORDER
        ],
        ignore_index=True,
    )
    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot()

    module.draw_fig2_curves(ax, curves, args)

    labels = [line.get_label() for line in ax.lines]
    assert r"$(00L)$ markers" in labels


def test_bragg_view_cli_sets_high_l_range_and_markers() -> None:
    module = load_script_module()

    args = module.parse_args(["--bragg-view"])

    assert args.bragg_view
    assert args.show_bragg_markers
    assert args.L_min == pytest.approx(0.0)
    assert args.L_max == pytest.approx(9.0)
    assert args.bragg_marker_L == pytest.approx([3.0, 6.0, 9.0])


def test_bragg_view_preserves_explicit_l_range() -> None:
    module = load_script_module()

    args = module.parse_args(["--bragg-view", "--L-max", "6"])

    assert args.show_bragg_markers
    assert args.L_min == pytest.approx(0.0)
    assert args.L_max == pytest.approx(6.0)


def test_compute_curves_includes_bare_structure_without_changing_pure_parratt_formula() -> None:
    module = load_script_module()
    source = inspect.getsource(module.compute_curves)

    assert "structure = np.zeros_like" in source
    assert "ht_over_q2 = np.zeros_like" in source
    assert "structure += weight * S" in source
    assert "divergence_safe_ht_over_q2_average" in source
    assert "ht_over_q2 += weight * S / qz_safe**2" not in source
    assert "ht_over_q2 = ht_over_q2_values * float(args.ht_over_q2_scale)" in source
    assert "normalized_positive_curve(ht_over_q2)" not in source
    assert '"label": HT_STRUCTURE_LABEL' in source
    assert '"label": HT_OVER_Q2_LABEL' in source
    assert "pure_parratt += weight * Rp\n" in source
    assert "pure_parratt += weight * Rp * S" not in source


def test_load_curves_from_csv_recognizes_bare_ht_structure_term(tmp_path) -> None:
    module = load_script_module()
    csv_path = tmp_path / "curves.csv"
    pd.DataFrame(
        {
            "L": [0.0, 0.1],
            "Qz_Ainv": [0.0, 0.02],
            "S_HT0": [1.0, 0.8],
            "Parratt_reflectivity": [1.0, 0.9],
        }
    ).to_csv(csv_path, index=False)

    curves = module.load_curves_from_csv(
        csv_path, module.parse_args(["--ht-structure-scale", "3", "--ht-over-q2-scale", "2"])
    )

    labels = set(curves["label"])
    assert module.HT_STRUCTURE_LABEL in labels
    assert module.HT_OVER_Q2_LABEL in labels
    assert module.PURE_PARRATT_LABEL in labels

    ht_structure = curves[curves["label"] == module.HT_STRUCTURE_LABEL]["intensity"]
    assert ht_structure.max() == pytest.approx(3.0)
    ht_over_q2 = curves[curves["label"] == module.HT_OVER_Q2_LABEL].sort_values("L")
    assert np.isnan(ht_over_q2["intensity"].iloc[0])
    assert ht_over_q2["intensity"].iloc[1] == pytest.approx(0.8 / 0.02**2 * 2.0)


def test_load_curves_from_csv_can_emit_stitch_curves(tmp_path) -> None:
    module = load_script_module()
    csv_path = tmp_path / "curves.csv"
    qz = module.DEFAULTS.qc_inv_angstrom * np.array([2.0, 5.0, 6.0, 7.0, 8.0])
    pd.DataFrame(
        {
            "L": module.L_from_qz(qz, module.DEFAULTS.c_angstrom),
            "Qz_Ainv": qz,
            "S_HT0": np.ones_like(qz),
            "Parratt_reflectivity": np.linspace(1.0, 0.2, qz.size),
        }
    ).to_csv(csv_path, index=False)

    curves = module.load_curves_from_csv(
        csv_path,
        module.parse_args(
            [
                "--from-csv",
                str(csv_path),
                "--stitch",
                "--scale-mode",
                "manual",
                "--manual-stitch-scale",
                "2.0",
            ]
        ),
    )

    assert module.STITCH_SCALED_HT_OVER_Q2_LABEL in set(curves["label"])
    assert module.STITCHED_PARRATT_HT_OVER_Q2_LABEL in set(curves["label"])


def test_apply_l_limits_updates_axis_without_recompute() -> None:
    module = load_script_module()
    calls: list[str] = []

    class DummyCanvas:
        def draw_idle(self) -> None:
            calls.append("draw_idle")

    axis = SimpleNamespace(set_xlim=lambda left, right: calls.append(f"{left:g}:{right:g}"))

    module.apply_l_limits_to_axis(axis, DummyCanvas(), 0.05, 0.15)

    assert calls == ["0.05:0.15", "draw_idle"]


def test_apply_y_log_limits_updates_axis_without_recompute() -> None:
    module = load_script_module()
    calls: list[str] = []

    class DummyCanvas:
        def draw_idle(self) -> None:
            calls.append("draw_idle")

    axis = SimpleNamespace(set_ylim=lambda bottom, top: calls.append(f"{bottom:g}:{top:g}"))

    module.apply_y_log_limits_to_axis(axis, DummyCanvas(), -6.0, 2.0)

    assert calls == ["1e-06:100", "draw_idle"]


def test_draw_fig2_curves_leaves_ht_over_q2_nan_as_plot_gap() -> None:
    module = load_script_module()
    plotted: dict[str, np.ndarray] = {}

    class DummySecondaryAxis:
        def set_xlabel(self, _label: str) -> None:
            return

    class DummyAxis:
        def plot(self, _x, y, *, label: str, **_kwargs) -> None:
            plotted[label] = np.asarray(y, dtype=float)

        def axvline(self, *_args, **_kwargs) -> None:
            return

        def set_xlim(self, *_args) -> None:
            return

        def set_yscale(self, *_args) -> None:
            return

        def set_ylim(self, *_args) -> None:
            return

        def set_xlabel(self, *_args) -> None:
            return

        def set_ylabel(self, *_args) -> None:
            return

        def set_title(self, *_args) -> None:
            return

        def secondary_xaxis(self, *_args, **_kwargs):
            return DummySecondaryAxis()

        def grid(self, *_args, **_kwargs) -> None:
            return

        def legend(self, *_args, **_kwargs) -> None:
            return

    curves = pd.concat(
        [
            pd.DataFrame(
                {
                    "L": [0.0, 0.1],
                    "Qz_Ainv": [0.0, 0.02],
                    "intensity": [np.nan, 2.0],
                    "label": module.HT_OVER_Q2_LABEL,
                }
            ),
            pd.DataFrame(
                {
                    "L": [0.0, 0.1],
                    "Qz_Ainv": [0.0, 0.02],
                    "intensity": [0.0, 1.0],
                    "label": module.PURE_PARRATT_LABEL,
                }
            ),
        ],
        ignore_index=True,
    )

    module.draw_fig2_curves(DummyAxis(), curves, module.parse_args([]))

    ht_label = module.FIG2_CURVE_DISPLAY_LABELS[module.HT_OVER_Q2_LABEL]
    parratt_label = module.FIG2_CURVE_DISPLAY_LABELS[module.PURE_PARRATT_LABEL]
    assert np.isnan(plotted[ht_label][0])
    assert plotted[ht_label][1] == pytest.approx(2.0)
    assert plotted[parratt_label][0] > 0.0


def test_gui_launcher_uses_tkagg_ttk_and_background_compute() -> None:
    module = load_script_module()
    source = inspect.getsource(module.run_fig2_gui)

    forbidden_snippets = (
        "NotImplementedError",
        "Apply parameters",
        "Click Apply",
        "debounce_ms",
        "pending_recompute_id",
        "for index, label in enumerate(FIG2_CURVE_ORDER)",
        "HT structure scale",
        "ttk.Notebook",
        'text="View"',
        'text="Curves"',
        'text="Beam"',
        'text="Sample"',
    )
    for snippet in forbidden_snippets:
        assert snippet not in source

    required_snippets = (
        "FigureCanvasTkAgg",
        "NavigationToolbar2Tk",
        "tk.Canvas",
        "ttk.Scrollbar",
        "ttk.LabelFrame",
        "Axes and limits",
        "Displayed curves",
        "Scale factors",
        "Spectral bandwidth",
        "Angular divergence",
        "Film and substrate",
        "Film geometry",
        "ttk.Scale",
        "ttk.Entry",
        '"<Return>"',
        '"<FocusOut>"',
        "threading.Thread",
        ".after(",
        "is_current",
        "request_live_compute",
        "finish_live_compute",
        "y_min_log",
        "y_max_log",
        "Y min (log10)",
        "Y max (log10)",
        "bandwidth_percent",
        "Bandwidth FWHM (%)",
        "n_wavelength_samples",
        "Wavelength samples",
        "divergence_fwhm_deg",
        "Divergence FWHM (deg)",
        "n_divergence_samples",
        "Divergence samples",
        "ht_over_q2_scale",
        "HT / Qz² scale",
        "Auto-fit HT/Qz²",
        "def _auto_fit_ht_over_q2_scale",
        "def _finish_auto_fit_ht_over_q2_scale",
        "def _start_auto_fit_ht_over_q2_scale",
        "curves_span_l_range",
        "autofit_args_for_l_range",
        "fit_ht_over_q2_scale_to_parratt",
        "Film ρ (e/Å³)",
        "Substrate ρ (e/Å³)",
        "σt (Å)",
        "σb (Å)",
        "expanded_slider_bounds(from_, to, variable.get())",
        "Curve visibility",
        "curve_visible_vars",
        "DEFAULT_GUI_VISIBLE_CURVE_LABELS",
        "ttk.Checkbutton",
        "for index, label in enumerate(DEFAULT_GUI_VISIBLE_CURVE_LABELS)",
        "visible_curve_labels=_selected_curve_labels()",
        "y_log_var",
        "Log Y axis",
        "use_log_y_axis=y_log_var.get()",
        "def _on_y_scale_changed",
    )
    for snippet in required_snippets:
        assert snippet in source


def test_l_range_gui_callback_recomputes_but_y_range_callback_is_axis_only() -> None:
    module = load_script_module()
    source = inspect.getsource(module.run_fig2_gui)

    l_handler = source.split("def _on_l_limits_changed", 1)[1].split("def _on_y_limits_changed", 1)[
        0
    ]
    y_handler = source.split("def _on_y_limits_changed", 1)[1].split("def _draw_curves", 1)[0]

    assert "_schedule_live_recompute()" in l_handler
    assert "_schedule_live_recompute()" not in y_handler


def test_live_compute_gate_marks_old_generation_stale() -> None:
    module = load_script_module()
    gate = module.LiveComputeGate()

    first = gate.next_generation()
    second = gate.next_generation()

    assert first == 1
    assert second == 2
    assert gate.is_current(second)
    assert not gate.is_current(first)


def test_live_compute_gate_queues_latest_args_during_active_compute() -> None:
    module = load_script_module()
    gate = module.LiveComputeGate()

    first = gate.request_live_compute("first")
    second = gate.request_live_compute("second")
    third = gate.request_live_compute("third")

    assert first == (1, "first")
    assert second is None
    assert third is None
    assert not gate.is_current(1)

    queued = gate.finish_live_compute(1)

    assert queued == (3, "third")
    assert gate.is_current(3)
    assert gate.finish_live_compute(3) is None
