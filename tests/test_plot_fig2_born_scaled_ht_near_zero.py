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


def _record_stack_thicknesses(module, monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    original_stack = module.make_air_film_substrate_stack

    def _recording_stack(args, *, thickness_angstrom):
        recorded.append(float(thickness_angstrom))
        return original_stack(args, thickness_angstrom=thickness_angstrom)

    monkeypatch.setattr(module, "make_air_film_substrate_stack", _recording_stack)
    return recorded


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
    assert module.DEFAULTS.L_max == pytest.approx(10.0)


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
    assert module.GUI_CURVE_ORDER == (
        module.HT_STRUCTURE_LABEL,
        module.HT_OVER_Q2_LABEL,
        module.PURE_PARRATT_LABEL,
        module.STITCHED_PARRATT_HT_OVER_Q2_LABEL,
    )
    assert module.DEFAULT_GUI_CURVE_VISIBILITY == {
        module.HT_STRUCTURE_LABEL: True,
        module.HT_OVER_Q2_LABEL: True,
        module.PURE_PARRATT_LABEL: True,
        module.STITCHED_PARRATT_HT_OVER_Q2_LABEL: False,
    }
    assert module.FIG2_CURVE_DISPLAY_LABELS[module.HT_STRUCTURE_LABEL] == (
        r"$S_{\mathrm{HT},0}(L)$"
    )
    assert module.FIG2_CURVE_DISPLAY_LABELS[module.HT_OVER_Q2_LABEL] == (
        r"Scaled $S_{\mathrm{HT},0}(L) / Q_z^2$, automatic $A$"
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
                (module.LEGACY_STITCH_WEIGHT_LABEL, weight),
            )
        ],
        ignore_index=True,
    )


def test_estimate_ht_q2_scale_uses_log_median_and_bragg_masks() -> None:
    module = load_script_module()
    x = np.linspace(5.1, 9.9, 60)
    qz = module.DEFAULTS.qc_inv_angstrom * x
    l_values = np.linspace(1.0, 2.0, 60)
    ht_over_q2 = np.linspace(0.2, 2.0, 60)
    pure = 2.5 * ht_over_q2
    l_values[10] = 3.0
    l_values[20] = 6.0
    pure[[10, 20]] = 1.0e9

    estimate = module.estimate_ht_q2_scale(
        L=l_values,
        qz=qz,
        qc=module.DEFAULTS.qc_inv_angstrom,
        ht_over_q2=ht_over_q2,
        pure_parratt=pure,
        bragg_centers_L=(3.0, 6.0),
        bragg_half_width_L=0.05,
    )

    assert isinstance(estimate, module.ScaleEstimate)
    assert estimate.scale == pytest.approx(2.5)
    assert estimate.n_fit_points == 58
    assert estimate.fit_x_min_q_over_qc == pytest.approx(5.0)
    assert estimate.fit_x_max_q_over_qc == pytest.approx(10.0)
    assert estimate.mad_log10_residual == pytest.approx(0.0)


def test_estimate_ht_q2_scale_requires_enough_valid_points() -> None:
    module = load_script_module()
    x = np.linspace(5.1, 9.9, 9)
    qz = module.DEFAULTS.qc_inv_angstrom * x
    l_values = np.linspace(1.0, 2.0, 9)

    with pytest.raises(ValueError, match="Need at least ten finite positive points"):
        module.estimate_ht_q2_scale(
            L=l_values,
            qz=qz,
            qc=module.DEFAULTS.qc_inv_angstrom,
            ht_over_q2=np.ones_like(l_values),
            pure_parratt=np.ones_like(l_values),
        )


def test_stitch_scale_path_does_not_use_peak_fitting() -> None:
    module = load_script_module()
    source = inspect.getsource(module.build_stitch_curve_frames)

    assert "pseudo_voigt" not in source
    assert "curve_fit" not in source


def test_common_thickness_50nm_bi2se3_matches_parratt_and_ht() -> None:
    module = load_script_module()

    thickness = module.resolve_common_thickness(
        thickness_nm=50.0,
        c_angstrom=28.6360,
        stack_layers=0,
    )

    assert isinstance(thickness, module.ThicknessResolution)
    assert thickness.target_thickness_angstrom == pytest.approx(500.0)
    assert thickness.stack_layers == 17
    assert thickness.effective_thickness_angstrom == pytest.approx(17 * 28.6360)
    assert thickness.effective_thickness_nm == pytest.approx(48.6812)
    assert thickness.fringe_period_L == pytest.approx(1.0 / 17.0)
    assert thickness.thickness_mismatch_angstrom == pytest.approx(17 * 28.6360 - 500.0)
    assert thickness.thickness_mismatch_percent == pytest.approx(
        100.0 * (17 * 28.6360 - 500.0) / 500.0
    )


def test_thickness_summary_reports_effective_stack_and_period() -> None:
    module = load_script_module()

    text = module.thickness_summary_text(
        module.resolve_common_thickness(
            thickness_nm=50.0,
            c_angstrom=28.6360,
            stack_layers=0,
        )
    )

    assert "target d = 50.00 nm" in text
    assert "N = 17" in text
    assert "effective d = 48.68 nm" in text
    assert "fringe period" in text


def test_thickness_warning_reports_nominal_effective_mismatch() -> None:
    module = load_script_module()

    warning = module.thickness_warning_text(
        module.resolve_common_thickness(
            thickness_nm=50.0,
            c_angstrom=28.6360,
            stack_layers=0,
        )
    )
    quiet = module.thickness_warning_text(
        module.resolve_common_thickness(
            thickness_nm=17 * 28.6360 / 10.0,
            c_angstrom=28.6360,
            stack_layers=0,
        )
    )

    assert warning is not None
    assert "target thickness 50.000 nm maps to N=17 repeats" in warning
    assert "Parratt thickness is synchronized to HT: 48.681 nm" in warning
    assert quiet is None


def test_explicit_stack_layers_override_target_thickness_rounding() -> None:
    module = load_script_module()

    thickness = module.resolve_common_thickness(
        thickness_nm=50.0,
        c_angstrom=28.6360,
        stack_layers=18,
    )

    assert thickness.stack_layers == 18
    assert thickness.effective_thickness_angstrom == pytest.approx(18 * 28.6360)


def test_parratt_stack_uses_effective_thickness_not_nominal() -> None:
    module = load_script_module()
    args = module.parse_args(["--thickness-nm", "50", "--c-angstrom", "28.636"])
    thickness = module.resolve_common_thickness(
        thickness_nm=args.thickness_nm,
        c_angstrom=args.c_angstrom,
        stack_layers=args.stack_layers,
    )

    layers = module.make_air_film_substrate_stack(
        args,
        thickness_angstrom=thickness.effective_thickness_angstrom,
    )

    assert layers[1].thickness_angstrom == pytest.approx(thickness.effective_thickness_angstrom)
    assert layers[1].thickness_angstrom != pytest.approx(500.0)


def test_fringe_periods_match_when_thickness_is_synchronized() -> None:
    module = load_script_module()
    thickness = module.resolve_common_thickness(
        thickness_nm=50.0,
        c_angstrom=28.6360,
        stack_layers=0,
    )

    assert thickness.fringe_period_L_parratt == pytest.approx(thickness.fringe_period_L)
    assert thickness.fringe_period_L_parratt == pytest.approx(28.6360 / (17 * 28.6360))


def test_compute_curves_uses_common_effective_thickness_for_ht_and_parratt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    ht_stack_layers: list[int] = []
    parratt_thicknesses = _record_stack_thicknesses(module, monkeypatch)

    def _fake_ht_module(**kwargs):
        ht_stack_layers.append(int(kwargs["stack_layers"]))
        l_max = float(kwargs["L_max"])
        return {(0, 0): {"L": np.linspace(0.0, l_max, 400), "I": np.ones(400)}}

    monkeypatch.setattr(module, "import_ht_module", lambda _repo_root: _fake_ht_module)
    args = module.parse_args(
        [
            "--thickness-nm",
            "50",
            "--c-angstrom",
            "28.636",
            "--L-min",
            "0",
            "--L-max",
            "1",
            "--L-step",
            "0.2",
            "--bandwidth-fwhm",
            "0",
            "--n-wavelength-samples",
            "1",
            "--divergence-fwhm-deg",
            "0",
            "--n-divergence-samples",
            "1",
            "--fit-x-min-q-over-qc",
            "0.1",
            "--fit-x-max-q-over-qc",
            "20",
        ]
    )

    curves = module.compute_curves(args)

    assert ht_stack_layers == [17]
    assert parratt_thicknesses == pytest.approx([17 * 28.636])
    assert curves["target_thickness_nm"].iloc[0] == pytest.approx(50.0)
    assert curves["effective_thickness_nm"].iloc[0] == pytest.approx(17 * 28.636 / 10.0)
    assert curves["stack_layers"].iloc[0] == 17
    assert curves["fringe_period_L_ht"].iloc[0] == pytest.approx(1.0 / 17.0)
    assert curves["fringe_period_L_parratt"].iloc[0] == pytest.approx(1.0 / 17.0)


def test_average_pure_parratt_for_L_uses_effective_thickness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    parratt_thicknesses = _record_stack_thicknesses(module, monkeypatch)

    args = module.parse_args(
        [
            "--thickness-nm",
            "50",
            "--c-angstrom",
            "28.636",
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

    module.average_pure_parratt_for_L(np.array([0.0, 0.2, 0.4]), args)

    assert parratt_thicknesses == pytest.approx([17 * 28.636])


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

    out, meta = module.hard_piecewise_stitch(
        np.ones(5),
        10.0 * np.ones(5),
        np.array([1.0, 2.0, 3.5, 5.0, 6.0]),
        x_cut=5.0,
    )
    assert out.tolist() == [1.0, 1.0, 1.0, 1.0, 10.0]
    assert meta["used_morph"] is False


def test_piecewise_stitch_is_exact_on_both_sides() -> None:
    module = load_script_module()
    x = np.array([1.0, 4.9, 5.0, 5.1, 9.0])
    low = np.ones_like(x)
    high = 10.0 * np.ones_like(x)

    out, meta = module.hard_piecewise_stitch(
        low,
        high,
        x,
        x_cut=5.0,
    )

    assert np.allclose(out[x <= 5.0], low[x <= 5.0])
    assert np.allclose(out[x > 5.0], high[x > 5.0])
    assert not meta["used_morph"]
    assert meta["x1"] == pytest.approx(5.0)
    assert meta["x2"] == pytest.approx(5.0)


def test_choose_stitch_cut_best_match_avoids_bragg_mask() -> None:
    module = load_script_module()
    L = np.array([2.8, 3.0, 4.0, 5.0])
    x = np.array([4.0, 5.0, 6.0, 7.0])
    low = np.ones_like(x)
    high = np.array([1.01, 1.0, 1.5, 2.0])

    x_cut = module.choose_stitch_cut(
        L,
        x,
        low,
        high,
        mode="best-match",
        search_min=3.0,
        search_max=7.0,
        exclude_centers=(3.0,),
        exclude_half_width=0.35,
    )

    assert x_cut != 5.0
    assert x_cut == pytest.approx(6.0)


def test_contiguous_true_segments_finds_runs() -> None:
    module = load_script_module()

    segments = module.contiguous_true_segments(
        np.array([False, True, True, False, True, False, True, True])
    )

    assert segments == [(1, 3), (4, 5), (6, 8)]


def test_best_continuous_cut_rejects_single_point_match() -> None:
    module = load_script_module()
    x = np.linspace(3.0, 6.0, 31)
    L = np.linspace(1.0, 2.0, x.size)
    low = np.ones_like(x)
    high = np.full_like(x, 10.0)
    high[15] = 1.0

    with pytest.raises(ValueError, match="No continuous stitch region"):
        module.choose_continuous_stitch_region(
            L,
            x,
            low,
            high,
            max_abs_log10_jump=0.08,
            min_width_q_over_qc=0.25,
            min_points=5,
        )


def test_best_continuous_cut_finds_interval() -> None:
    module = load_script_module()
    x = np.linspace(3.0, 6.0, 61)
    L = np.linspace(1.0, 2.0, x.size)
    low = np.ones_like(x)
    high = np.full_like(x, 5.0)
    interval = (x >= 4.0) & (x <= 4.7)
    high[interval] = 10.0 ** (0.02 * np.sin(np.linspace(0.0, np.pi, interval.sum())))

    cut = module.choose_continuous_stitch_region(
        L,
        x,
        low,
        high,
        max_abs_log10_jump=0.08,
        min_width_q_over_qc=0.25,
        min_points=5,
    )

    assert cut["continuous_region_found"] is True
    assert 4.0 <= cut["x_cut"] <= 4.7
    assert 4.0 <= cut["continuous_region_x1_q_over_qc"] <= cut["x_cut"]
    assert cut["x_cut"] <= cut["continuous_region_x2_q_over_qc"] <= 4.7
    assert cut["continuous_region_width_q_over_qc"] >= 0.25
    assert cut["continuous_region_points"] >= 5
    assert abs(cut["log10_jump_at_cut"]) <= 0.08


def test_stitch_cli_defaults_and_choices() -> None:
    module = load_script_module()

    default_args = module.parse_args([])
    assert default_args.plot_mode == "fig2"
    assert not default_args.stitch
    assert default_args.stitch_cut_mode == "best-continuous"
    assert default_args.stitch_cut_q_over_qc == pytest.approx(5.0)
    assert default_args.stitch_x1_q_over_qc == pytest.approx(2.0)
    assert default_args.stitch_x2_q_over_qc == pytest.approx(5.0)
    assert default_args.stitch_branch == "normalized-ht-q2"
    assert default_args.scale_mode == "log-median"
    assert default_args.fit_exclude_centers_L == "3,6,9"
    assert default_args.max_abs_log10_jump_allowed == pytest.approx(0.08)
    assert default_args.min_continuous_width_q_over_qc == pytest.approx(0.25)
    assert default_args.min_continuous_points == 10
    assert default_args.workers == "1"

    stitch_args = module.parse_args(
        [
            "--stitch",
            "--plot-mode",
            "stitch",
            "--scale-mode",
            "manual",
            "--manual-stitch-scale",
            "2.5",
            "--stitch-cut-mode",
            "best-match",
            "--max-abs-log10-jump-allowed",
            "0.12",
            "--min-continuous-width-q-over-qc",
            "0.4",
            "--min-continuous-points",
            "7",
            "--workers",
            "4",
        ]
    )
    assert stitch_args.stitch
    assert stitch_args.plot_mode == "stitch"
    assert stitch_args.scale_mode == "manual"
    assert stitch_args.manual_stitch_scale == pytest.approx(2.5)
    assert stitch_args.stitch_cut_mode == "best-match"
    assert stitch_args.max_abs_log10_jump_allowed == pytest.approx(0.12)
    assert stitch_args.min_continuous_width_q_over_qc == pytest.approx(0.4)
    assert stitch_args.min_continuous_points == 7
    assert stitch_args.workers == "4"

    auto_args = module.parse_args(["--workers", "auto"])
    assert auto_args.workers == "auto"

    legacy_cut_args = module.parse_args(["--stitch-x2-q-over-qc", "6.25"])
    assert legacy_cut_args.stitch_cut_q_over_qc == pytest.approx(6.25)


def test_worker_count_parser_and_resolver() -> None:
    module = load_script_module()

    assert module.resolve_worker_count("1", n_jobs=20) == 1
    assert module.resolve_worker_count("4", n_jobs=20) == 4
    assert module.resolve_worker_count("999", n_jobs=3) == 3
    assert module.resolve_worker_count("auto", n_jobs=2) <= 2
    assert module.resolve_worker_count("auto", n_jobs=0) == 1

    for invalid in ("0", "-2", "abc", ""):
        with pytest.raises(SystemExit):
            module.parse_args(["--workers", invalid])


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
            "--L-max",
            "0.5",
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

    l_min, l_max = module.compute_l_grid_limits(args)

    assert l_min == pytest.approx(args.L_min)
    assert l_max > args.L_max
    with pytest.raises(ValueError, match="No continuous stitch region"):
        module.compute_curves(args)


def test_best_match_stitch_cut_computes_hidden_search_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()

    def _fake_ht_module(**kwargs):
        l_max = float(kwargs["L_max"])
        return {(0, 0): {"L": np.linspace(0.0, l_max, 200), "I": np.ones(200)}}

    monkeypatch.setattr(module, "import_ht_module", lambda _repo_root: _fake_ht_module)
    args = module.parse_args(
        [
            "--stitch",
            "--stitch-cut-mode",
            "best-match",
            "--scale-mode",
            "manual",
            "--L-min",
            "0.0",
            "--L-max",
            "0.5",
            "--cut-search-min-q-over-qc",
            "3.0",
            "--cut-search-max-q-over-qc",
            "6.0",
        ]
    )

    curves = module.compute_curves(args)
    stitched = curves[curves["label"] == module.STITCHED_PARRATT_HT_OVER_Q2_LABEL]

    assert len(stitched) > 0
    assert stitched["L"].max() > args.L_max


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
                "--stitch-cut-mode",
                "fixed",
                "--scale-mode",
                "manual",
            ]
        )
    )
    assert len(calls) == 2


def test_compute_curves_parallel_workers_match_serial_and_use_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()

    def _fake_ht_module(**kwargs):
        l_max = float(kwargs["L_max"])
        l_values = np.linspace(0.0, l_max, 300)
        return {(0, 0): {"L": l_values, "I": 1.0 + 0.2 * l_values}}

    executor_calls: list[int] = []
    real_executor = module.ThreadPoolExecutor

    class RecordingExecutor(real_executor):
        def __init__(self, *args, **kwargs):
            executor_calls.append(int(kwargs["max_workers"]))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(module, "import_ht_module", lambda _repo_root: _fake_ht_module)
    monkeypatch.setattr(module, "ThreadPoolExecutor", RecordingExecutor)
    common_args = [
        "--L-min",
        "0.0",
        "--L-max",
        "1.0",
        "--L-step",
        "0.2",
        "--bandwidth-fwhm",
        "0.02",
        "--n-wavelength-samples",
        "3",
        "--divergence-fwhm-deg",
        "0.2",
        "--n-divergence-samples",
        "3",
        "--fit-x-min-q-over-qc",
        "0.1",
        "--fit-x-max-q-over-qc",
        "20",
    ]

    serial = module.compute_curves(module.parse_args([*common_args, "--workers", "1"]))
    assert executor_calls == []
    parallel = module.compute_curves(module.parse_args([*common_args, "--workers", "2"]))

    assert executor_calls == [2]
    serial = serial.sort_values(["label", "L"]).reset_index(drop=True)
    parallel = parallel.sort_values(["label", "L"]).reset_index(drop=True)
    assert serial[["label", "L", "Qz_Ainv"]].equals(parallel[["label", "L", "Qz_Ainv"]])
    assert np.allclose(
        serial["intensity"].to_numpy(),
        parallel["intensity"].to_numpy(),
        rtol=1.0e-10,
        atol=1.0e-12,
        equal_nan=True,
    )


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
            "--stitch-cut-mode",
            "fixed",
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
    assert module.LEGACY_STITCH_WEIGHT_LABEL not in labels

    parratt = curves[curves["label"] == module.PURE_PARRATT_LABEL].sort_values("L")
    unscaled = curves[curves["label"] == module.STITCH_UNSCALED_HT_OVER_Q2_LABEL].sort_values("L")
    scaled = curves[curves["label"] == module.STITCH_SCALED_HT_OVER_Q2_LABEL].sort_values("L")
    stitched = curves[curves["label"] == module.STITCHED_PARRATT_HT_OVER_Q2_LABEL].sort_values("L")
    x_cut = float(stitched["stitch_cut_q_over_qc"].iloc[0])
    qz_over_qc = stitched["Qz_Ainv"].to_numpy() / args.qc_inv_angstrom
    scale_a = float(stitched["scale_A"].iloc[0])
    expected = np.where(
        qz_over_qc <= x_cut,
        parratt["intensity"].to_numpy(),
        scaled["intensity"].to_numpy(),
    )

    assert scale_a > 0.0
    assert np.allclose(
        scaled["intensity"].to_numpy(),
        scale_a * unscaled["intensity"].to_numpy(),
        equal_nan=True,
    )
    assert np.allclose(stitched["intensity"].to_numpy(), expected, equal_nan=True)
    assert stitched["used_morph"].iloc[0] == np.False_


def test_build_stitch_curve_frames_records_continuous_region_metadata() -> None:
    module = load_script_module()
    x = np.linspace(3.0, 6.0, 61)
    qz = module.DEFAULTS.qc_inv_angstrom * x
    L = module.L_from_qz(qz, module.DEFAULTS.c_angstrom)
    pure = np.ones_like(x)
    ht = np.full_like(x, 5.0)
    interval = (x >= 4.0) & (x <= 4.7)
    ht[interval] = 1.0
    args = module.parse_args(
        [
            "--stitch",
            "--scale-mode",
            "manual",
            "--manual-stitch-scale",
            "1.0",
            "--stitch-cut-mode",
            "best-continuous",
            "--min-continuous-points",
            "5",
        ]
    )

    frames = module.build_stitch_curve_frames(L, qz, pure, ht, None, args)
    stitched = next(
        frame
        for frame in frames
        if frame["label"].iloc[0] == module.STITCHED_PARRATT_HT_OVER_Q2_LABEL
    )

    assert stitched["continuous_region_found"].iloc[0] == np.True_
    assert stitched["continuous_region_x1_q_over_qc"].iloc[0] == pytest.approx(4.0)
    assert stitched["continuous_region_x2_q_over_qc"].iloc[0] == pytest.approx(4.7)
    assert stitched["continuous_region_width_q_over_qc"].iloc[0] >= 0.25
    assert stitched["continuous_region_points"].iloc[0] >= 5
    assert stitched["max_abs_log10_jump_allowed"].iloc[0] == pytest.approx(0.08)
    assert abs(stitched["log10_jump_at_cut"].iloc[0]) <= 0.08


def test_stitched_curve_stays_hard_piecewise_when_handoff_jumps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()

    def _fake_ht_module(**_kwargs):
        return {(0, 0): {"L": np.linspace(0.0, 4.0, 41), "I": np.ones(41)}}

    monkeypatch.setattr(module, "import_ht_module", lambda _repo_root: _fake_ht_module)
    args = module.parse_args(
        [
            "--stitch",
            "--plot-mode",
            "stitch",
            "--stitch-cut-mode",
            "fixed",
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
    parratt = curves[curves["label"] == module.PURE_PARRATT_LABEL].sort_values("L")
    scaled = curves[curves["label"] == module.STITCH_SCALED_HT_OVER_Q2_LABEL].sort_values("L")
    stitched = curves[curves["label"] == module.STITCHED_PARRATT_HT_OVER_Q2_LABEL].sort_values("L")

    x = stitched["Qz_Ainv"].to_numpy() / args.qc_inv_angstrom
    x_cut = float(stitched["stitch_cut_q_over_qc"].iloc[0])
    low_side = x <= x_cut
    high_side = x > x_cut

    assert stitched["used_morph"].iloc[0] == np.False_
    assert np.allclose(
        stitched["intensity"].to_numpy()[low_side],
        parratt["intensity"].to_numpy()[low_side],
        equal_nan=True,
    )
    assert np.allclose(
        stitched["intensity"].to_numpy()[high_side],
        scaled["intensity"].to_numpy()[high_side],
        equal_nan=True,
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
    assert module.LEGACY_STITCH_WEIGHT_LABEL not in diagnostic_labels


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
            "stitch_cut_mode",
            "stitch_cut_q_over_qc",
            "stitch_cut_L",
            "morph_x1_q_over_qc",
            "morph_x2_q_over_qc",
            "used_morph",
            "scale_method",
            "scale_mode",
            "scale_A",
            "fit_x_min_q_over_qc",
            "fit_x_max_q_over_qc",
            "fit_exclude_centers_L",
            "fit_exclude_half_width_L",
            "n_fit_points",
            "median_log10_residual",
            "mad_log10_residual",
            "log10_jump_at_cut",
            "continuous_region_found",
            "continuous_region_x1_q_over_qc",
            "continuous_region_x2_q_over_qc",
            "continuous_region_width_q_over_qc",
            "continuous_region_points",
            "continuous_region_score",
            "max_abs_log10_jump_allowed",
            "visible_L_min",
            "visible_L_max",
            "scale_grid_L_min",
            "scale_grid_L_max",
            "target_thickness_nm",
            "effective_thickness_nm",
            "target_thickness_angstrom",
            "effective_thickness_angstrom",
            "thickness_mismatch_angstrom",
            "thickness_mismatch_percent",
            "stack_layers",
            "c_angstrom",
            "fringe_period_L_ht",
            "fringe_period_L_parratt",
        ]
    ).issubset(diagnostics.columns)
    assert diagnostics["n_fit_points"].iloc[0] == 4
    assert diagnostics["mad_log10_residual"].iloc[0] > 0.3
    assert diagnostics["stitch_cut_q_over_qc"].iloc[0] == pytest.approx(5.0)
    assert diagnostics["scale_method"].iloc[0] == "log-median"
    assert diagnostics["stack_layers"].iloc[0] == 17
    assert diagnostics["effective_thickness_angstrom"].iloc[0] == pytest.approx(17 * 28.636)
    assert diagnostics["fringe_period_L_ht"].iloc[0] == pytest.approx(
        diagnostics["fringe_period_L_parratt"].iloc[0]
    )
    assert "visual handoff" in capsys.readouterr().out


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
    state.cpu_workers = 6.0
    state.sync_density_from_qc("film")
    updated = state.updated_args(args)

    assert updated.thickness_nm == pytest.approx(75.0)
    assert updated.qc_inv_angstrom == pytest.approx(0.0600)
    assert updated.film_density_e_per_a3 == pytest.approx(module.density_e_per_a3_from_qc(0.0600))
    assert updated.bandwidth_fwhm == pytest.approx(0.075)
    assert updated.n_wavelength_samples == 123
    assert updated.divergence_fwhm_deg == pytest.approx(0.75)
    assert updated.n_divergence_samples == 99
    assert updated.workers == "6"

    auto_state = module.Fig2GuiParameterState.from_args(
        module.parse_args(["--workers", "auto", "--n-wavelength-samples", "2"])
    )
    assert 1.0 <= auto_state.cpu_workers <= 2.0


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
    assert "structure_lam += weight * S" in source
    assert "structure[:] += contribution.structure" in source
    assert "divergence_safe_ht_over_q2_average" in source
    assert "ht_over_q2 += weight * S / qz_safe**2" not in source
    assert "scale_estimate = estimate_ht_q2_scale_for_args(" in source
    assert "ht_over_q2 = scale_estimate.scale * ht_over_q2_values" in source
    assert "normalized_positive_curve(ht_over_q2)" not in source
    assert "_curve_frame(HT_STRUCTURE_LABEL" in source
    assert '"label": HT_OVER_Q2_LABEL' in source
    assert "pure_parratt_lam += weight * Rp\n" in source
    assert "pure_parratt[:] += contribution.pure_parratt" in source
    assert "pure_parratt += weight * Rp * S" not in source


def test_load_curves_from_csv_recognizes_bare_ht_structure_term(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script_module()
    csv_path = tmp_path / "curves.csv"
    qz = np.concatenate(([0.0], module.DEFAULTS.qc_inv_angstrom * np.linspace(5.1, 9.9, 12)))
    l_values = module.L_from_qz(qz, module.DEFAULTS.c_angstrom)
    structure = np.concatenate(([1.0], np.linspace(0.8, 0.3, 12)))
    csv_pure = np.full_like(qz, 123.0)
    recomputed_pure = np.linspace(1.0, 0.25, qz.size)

    def _fake_pure_parratt(l_nom, _args):
        return np.interp(l_nom, l_values, recomputed_pure)

    monkeypatch.setattr(module, "average_pure_parratt_for_L", _fake_pure_parratt)
    pd.DataFrame(
        {
            "L": l_values,
            "Qz_Ainv": qz,
            "S_HT0": structure,
            "Parratt_reflectivity": csv_pure,
        }
    ).to_csv(csv_path, index=False)

    curves = module.load_curves_from_csv(
        csv_path,
        module.parse_args(
            [
                "--ht-structure-scale",
                "3",
                "--L-max",
                "3",
                "--scale-mode",
                "manual",
                "--manual-stitch-scale",
                "1",
            ]
        ),
    )

    labels = set(curves["label"])
    assert module.HT_STRUCTURE_LABEL in labels
    assert module.HT_OVER_Q2_LABEL in labels
    assert module.PURE_PARRATT_LABEL in labels

    ht_structure = curves[curves["label"] == module.HT_STRUCTURE_LABEL]["intensity"]
    assert ht_structure.max() == pytest.approx(3.0)
    parratt = curves[curves["label"] == module.PURE_PARRATT_LABEL].sort_values("L")
    assert parratt["intensity"].to_numpy() == pytest.approx(recomputed_pure)
    assert not np.allclose(parratt["intensity"].to_numpy(), csv_pure)
    ht_over_q2 = curves[curves["label"] == module.HT_OVER_Q2_LABEL].sort_values("L")
    assert np.isnan(ht_over_q2["intensity"].iloc[0])
    assert ht_over_q2["intensity"].iloc[1] == pytest.approx(structure[1] / qz[1] ** 2)


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
                "--stitch-cut-mode",
                "fixed",
                "--scale-mode",
                "manual",
                "--manual-stitch-scale",
                "2.0",
            ]
        ),
    )

    assert module.STITCH_SCALED_HT_OVER_Q2_LABEL in set(curves["label"])
    assert module.STITCHED_PARRATT_HT_OVER_Q2_LABEL in set(curves["label"])
    parratt = curves[curves["label"] == module.PURE_PARRATT_LABEL].sort_values("L")
    scaled = curves[curves["label"] == module.STITCH_SCALED_HT_OVER_Q2_LABEL].sort_values("L")
    stitched = curves[curves["label"] == module.STITCHED_PARRATT_HT_OVER_Q2_LABEL].sort_values("L")
    x_cut = float(stitched["stitch_cut_q_over_qc"].iloc[0])
    qz_over_qc = stitched["Qz_Ainv"].to_numpy() / module.DEFAULTS.qc_inv_angstrom
    expected = np.where(
        qz_over_qc <= x_cut,
        parratt["intensity"].to_numpy(),
        scaled["intensity"].to_numpy(),
    )
    assert np.allclose(stitched["intensity"].to_numpy(), expected, equal_nan=True)


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


def test_draw_fig2_curves_can_show_optional_stitched_curve() -> None:
    module = load_script_module()
    plotted: list[str] = []

    class DummySecondaryAxis:
        def set_xlabel(self, _label: str) -> None:
            return

    class DummyAxis:
        def plot(self, _x, _y, *, label: str, **_kwargs) -> None:
            plotted.append(label)

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

    l_values = np.array([0.1, 0.2])
    curves = pd.concat(
        [
            pd.DataFrame(
                {
                    "L": l_values,
                    "Qz_Ainv": module.qz_from_L(l_values, module.DEFAULTS.c_angstrom),
                    "intensity": np.ones_like(l_values),
                    "label": label,
                }
            )
            for label in (
                *module.DEFAULT_GUI_VISIBLE_CURVE_LABELS,
                module.STITCHED_PARRATT_HT_OVER_Q2_LABEL,
            )
        ],
        ignore_index=True,
    )

    module.draw_fig2_curves(
        DummyAxis(),
        curves,
        module.parse_args([]),
        visible_curve_labels=module.GUI_CURVE_ORDER,
    )

    assert module.FIG2_CURVE_DISPLAY_LABELS[module.STITCHED_PARRATT_HT_OVER_Q2_LABEL] in plotted


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
        "Auto-fit HT/Qz²",
        "def _auto_fit_ht_over_q2_scale",
        "def _finish_auto_fit_ht_over_q2_scale",
        "def _start_auto_fit_ht_over_q2_scale",
        "autofit_args_for_l_range",
        "fit_ht_over_q2_scale_to_parratt",
        "HT / Qz² scale",
        "Morph width Qz/Qc",
        "Max hard jump log10",
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
        "Stitch handoff",
        "Spectral bandwidth",
        "Angular divergence",
        "Film and substrate",
        "Film geometry",
        "ttk.Scale",
        "ttk.Entry",
        "ttk.Combobox",
        '"<Return>"',
        '"<FocusOut>"',
        '"<<ComboboxSelected>>"',
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
        "cpu_workers",
        "CPU workers",
        "HT/Qz² automatic scale",
        "fit window: 5 < Qz/Qc < 10",
        "thickness_summary_var",
        "thickness_summary_text",
        "resolve_common_thickness_for_args",
        "stitch_cut_mode_var",
        "best-continuous",
        "Cut Qz/Qc",
        "Search min Qz/Qc",
        "Search max Qz/Qc",
        "def _on_option_parameter_changed",
        "Film ρ (e/Å³)",
        "Substrate ρ (e/Å³)",
        "σt (Å)",
        "σb (Å)",
        "expanded_slider_bounds(from_, to, variable.get())",
        "Curve visibility",
        "curve_visible_vars",
        "GUI_CURVE_ORDER",
        "DEFAULT_GUI_CURVE_VISIBILITY",
        "ttk.Checkbutton",
        "for index, label in enumerate(GUI_CURVE_ORDER)",
        "visible_curve_labels=_selected_curve_labels()",
        "y_log_var",
        "Log Y axis",
        "use_log_y_axis=y_log_var.get()",
        "def _on_y_scale_changed",
        "def _stitch_curve_visible",
        "STITCHED_PARRATT_HT_OVER_Q2_LABEL",
        "updated.stitch = _stitch_curve_visible()",
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


def test_stitched_gui_visibility_recomputes_when_curve_is_missing() -> None:
    module = load_script_module()
    source = inspect.getsource(module.run_fig2_gui)

    handler = source.split("def _on_curve_visibility_changed", 1)[1].split(
        "def _on_option_parameter_changed", 1
    )[0]

    assert "_stitch_curve_visible()" in handler
    assert "_stitch_curve_missing(current_curves)" in handler
    assert "_schedule_live_recompute()" in handler
    assert "_draw_curves(current_curves, current_args)" in handler


def test_stitched_gui_visibility_recomputes_after_initial_compute_if_missing() -> None:
    module = load_script_module()
    source = inspect.getsource(module.run_fig2_gui)

    finish_handler = source.split("def _finish_compute_success", 1)[1].split(
        "def _finish_compute_error", 1
    )[0]

    assert "_stitch_curve_visible()" in finish_handler
    assert "_stitch_curve_missing(curves)" in finish_handler
    assert "_schedule_live_recompute()" in finish_handler


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
