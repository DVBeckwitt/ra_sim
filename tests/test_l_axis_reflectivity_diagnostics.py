import csv
import importlib

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from ra_sim.analysis import l_axis_reflectivity_diagnostics as module_under_test
from ra_sim.analysis.l_axis_reflectivity_diagnostics import (
    LAxisDiagnosticMaterial,
    add_l_qz_top_axis,
    build_unaveraged_model_curves,
    get_hk_curve,
    run_l_axis_reflectivity_diagnostics,
    simulate_ht_p0_dense,
    stable_zero_reference,
    wavelength_average_ht_p0_on_L,
    wavelength_average_fresnel_born_on_qz,
    wavelength_average_parratt_on_L,
    wavelength_samples,
)
from ra_sim.analysis.parratt import (
    ParrattLayer,
    born_fresnel_asymptote,
    fresnel_reflectivity_single_interface,
    make_bi_chalcogenide_stack,
    miceli_correction_factor,
    parratt_reflectivity,
)


def _material() -> LAxisDiagnosticMaterial:
    return LAxisDiagnosticMaterial(
        name="Bi2Se3",
        cif_path="unused.cif",
        a_angstrom=4.14,
        c_angstrom=28.64,
        qc_inv_angstrom=0.0517,
        layers=[
            ParrattLayer("air", 0.0, None),
            ParrattLayer("Bi2Se3", 0.0517, 1000.0),
            ParrattLayer("SiO2", 0.0305, None),
        ],
        stack_layers=100,
    )


def test_make_bi_chalcogenide_stack_defaults_to_air_film_sio2() -> None:
    layers = make_bi_chalcogenide_stack("Bi2Se3", thickness_nm=100.0)

    assert [layer.name for layer in layers] == ["air", "Bi2Se3", "SiO2"]
    assert layers[0].qc_inv_angstrom == 0.0
    assert layers[1].qc_inv_angstrom == 0.0517
    assert layers[1].thickness_angstrom == 1000.0
    assert layers[2].qc_inv_angstrom == 0.0305
    assert layers[2].thickness_angstrom is None


def test_make_bi_chalcogenide_stack_supports_air_substrate() -> None:
    layers = make_bi_chalcogenide_stack("Bi2Te3", thickness_nm=25.0, substrate="air")

    assert [layer.name for layer in layers] == ["air", "Bi2Te3", "air"]
    assert layers[1].qc_inv_angstrom == 0.0519
    assert layers[1].thickness_angstrom == 250.0
    assert layers[2].qc_inv_angstrom == 0.0


def test_miceli_correction_factor_tends_to_one() -> None:
    qc = 0.05
    qz = np.array([10.0, 20.0]) * qc

    correction = miceli_correction_factor(qz, qc)

    assert np.allclose(correction, 1.0, rtol=0.02)


def test_fresnel_matches_born_high_q() -> None:
    qc = 0.05
    qz = np.array([10.0]) * qc

    fresnel = fresnel_reflectivity_single_interface(qz, qc)
    born = born_fresnel_asymptote(qz, qc)

    assert np.allclose(fresnel, born, rtol=0.02)


def test_parratt_bounded_near_zero() -> None:
    layers = [
        ParrattLayer("air", 0.0, None),
        ParrattLayer("film", 0.05, None),
    ]
    qz = np.array([1e-5, 1e-4, 1e-3])

    reflectivity = parratt_reflectivity(qz, layers)

    assert np.all(reflectivity <= 1.000001)
    assert np.all(reflectivity > 0.99)


def test_l_axis_plot_has_qz_top_axis() -> None:
    fig, ax = plt.subplots()

    secax = add_l_qz_top_axis(ax, c_angstrom=28.64)

    assert "L" in ax.get_xlabel()
    assert "Q_z" in secax.get_xlabel()
    assert ax.get_xlabel() != "Q"
    assert secax.get_xlabel() != "Q"
    plt.close(fig)


@pytest.mark.parametrize(
    "curves",
    [
        {(0, 0): {"L": np.array([0.0]), "I": np.array([1.0])}},
        {"0,0": {"L": np.array([0.0]), "I": np.array([1.0])}},
        {"(0, 0)": {"L": np.array([0.0]), "I": np.array([1.0])}},
    ],
)
def test_get_hk_curve_accepts_repo_key_variants(curves) -> None:
    curve = get_hk_curve(curves, h=0, k=0)

    assert np.array_equal(curve["L"], [0.0])
    assert np.array_equal(curve["I"], [1.0])


def test_stable_zero_reference_uses_near_zero_fallback() -> None:
    L = np.array([0.0, 0.01, 0.1])
    intensity = np.array([0.0, 2.0, 1.0])

    assert stable_zero_reference(L, intensity) == 2.0


def test_ht_called_with_p_user_zero(monkeypatch) -> None:
    seen = {}

    def fake_ht_Iinf_dict(**kwargs):
        seen.update(kwargs)
        return {(0, 0): {"L": np.array([0.0, 1.0]), "I": np.array([1.0, 0.5])}}

    monkeypatch.setattr(module_under_test, "ht_Iinf_dict", fake_ht_Iinf_dict)

    L, intensity = simulate_ht_p0_dense(
        _material(),
        lambda_angstrom=1.23,
        L_step_dense=0.2,
        L_max_padded=1.0,
    )

    assert seen["p"] == 0.0
    assert seen["hk_list"] == [(0, 0)]
    assert seen["finite_stack"] is True
    assert seen["stack_layers"] == 100
    assert seen["lambda_"] == 1.23
    assert np.array_equal(L, [0.0, 1.0])
    assert np.array_equal(intensity, [1.0, 0.5])


def test_born_scaled_curve_uses_ht_p0_not_sf_only() -> None:
    material = _material()
    L = np.array([0.5, 1.0, 2.0])
    ht_p0 = np.array([10.0, 5.0, 2.0])
    sf_only = np.array([10.0, 9.0, 9.0])

    curves = build_unaveraged_model_curves(
        material,
        L,
        ht_p0,
        sf_only_intensity=sf_only,
    )

    qz = 2.0 * np.pi * L / material.c_angstrom
    expected = (material.qc_inv_angstrom / (2.0 * qz)) ** 4 * (ht_p0 / ht_p0[0])
    sf_only_wrong = (material.qc_inv_angstrom / (2.0 * qz)) ** 4 * (sf_only / sf_only[0])

    assert np.allclose(curves["Born_scaled_HT_p0"], expected)
    assert not np.allclose(curves["Born_scaled_HT_p0"], sf_only_wrong)


def test_wavelength_samples_are_normalized() -> None:
    lambdas, weights = wavelength_samples(lambda0=1.5418, fwhm=0.05, n=9)

    assert lambdas.shape == (9,)
    assert weights.shape == (9,)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0.0)


def test_wavelength_average_uses_scaled_L() -> None:
    lambda0 = 1.0
    lambda_j = 1.05
    L_nom = np.array([1.0])

    L_true = L_nom * lambda0 / lambda_j

    assert np.allclose(L_true, [0.95238095238])


def test_wavelength_average_parratt_uses_scaled_l(monkeypatch) -> None:
    seen_qz = []

    monkeypatch.setattr(
        module_under_test,
        "wavelength_samples",
        lambda lambda0, bandwidth_fwhm, n_samples: (
            np.array([1.0, 2.0]),
            np.array([0.25, 0.75]),
        ),
    )

    def fake_parratt(qz, layers, wavelength_angstrom):
        del layers, wavelength_angstrom
        seen_qz.append(np.asarray(qz, dtype=float).copy())
        return np.ones_like(qz, dtype=float)

    monkeypatch.setattr(module_under_test, "parratt_reflectivity", fake_parratt)

    wavelength_average_parratt_on_L(
        np.array([1.0]),
        c_angstrom=10.0,
        layers=_material().layers,
        lambda0=1.0,
        n_samples=2,
    )

    assert np.allclose(seen_qz[0], [2.0 * np.pi / 10.0])
    assert np.allclose(seen_qz[1], [np.pi / 10.0])


def test_wavelength_average_fresnel_born_uses_full_bandwidth_scaling(monkeypatch) -> None:
    monkeypatch.setattr(
        module_under_test,
        "wavelength_samples",
        lambda lambda0, bandwidth_fwhm, n_samples: (
            np.array([1.0, 2.0]),
            np.array([0.25, 0.75]),
        ),
    )
    qz_nominal = np.array([0.1])
    qc = 0.05

    fresnel_avg, born_avg, correction = wavelength_average_fresnel_born_on_qz(
        qz_nominal,
        qc,
        lambda0=1.0,
        n_samples=2,
    )

    qz_true = np.array([0.1, 0.05])
    expected_fresnel = (
        0.25 * fresnel_reflectivity_single_interface(qz_true[0], qc)
        + 0.75 * fresnel_reflectivity_single_interface(qz_true[1], qc)
    )
    expected_born = (
        0.25 * born_fresnel_asymptote(qz_true[0], qc)
        + 0.75 * born_fresnel_asymptote(qz_true[1], qc)
    )

    assert np.allclose(fresnel_avg, expected_fresnel)
    assert np.allclose(born_avg, expected_born)
    assert np.allclose(correction, expected_fresnel / expected_born)


def test_wavelength_average_ht_resimulate_calls_ht_for_each_wavelength(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        module_under_test,
        "wavelength_samples",
        lambda lambda0, bandwidth_fwhm, n_samples: (
            np.array([1.0, 2.0]),
            np.array([0.5, 0.5]),
        ),
    )

    def fake_simulate(material, *, lambda_angstrom, L_step_dense=0.002, L_max_padded=9.5):
        del material, L_step_dense, L_max_padded
        calls.append(lambda_angstrom)
        return np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(module_under_test, "simulate_ht_p0_dense", fake_simulate)

    wavelength_average_ht_p0_on_L(
        np.array([1.0]),
        _material(),
        lambda0=1.0,
        n_samples=2,
        ht_bandwidth_mode="resimulate",
    )

    assert calls == [1.0, 2.0]


def test_wavelength_average_ht_interpolate_only_calls_ht_once(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        module_under_test,
        "wavelength_samples",
        lambda lambda0, bandwidth_fwhm, n_samples: (
            np.array([1.0, 2.0]),
            np.array([0.5, 0.5]),
        ),
    )

    def fake_simulate(material, *, lambda_angstrom, L_step_dense=0.002, L_max_padded=9.5):
        del material, L_step_dense, L_max_padded
        calls.append(lambda_angstrom)
        return np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(module_under_test, "simulate_ht_p0_dense", fake_simulate)

    wavelength_average_ht_p0_on_L(
        np.array([1.0]),
        _material(),
        lambda0=1.0,
        n_samples=2,
        ht_bandwidth_mode="interpolate-only",
    )

    assert calls == [1.0]


def test_run_writes_required_artifacts_and_metadata(tmp_path, monkeypatch) -> None:
    def fake_simulate(material, *, lambda_angstrom, L_step_dense=0.002, L_max_padded=9.5):
        del material, lambda_angstrom, L_step_dense, L_max_padded
        L = np.linspace(0.0, 9.5, 96)
        return L, 10.0 / (1.0 + L)

    monkeypatch.setattr(module_under_test, "simulate_ht_p0_dense", fake_simulate)

    run_l_axis_reflectivity_diagnostics(
        output_dir=tmp_path,
        material=_material(),
        L_values=np.linspace(0.0, 9.0, 10),
        lambda0_angstrom=1.5418,
        bandwidth_fwhm=0.05,
        n_wavelength_samples=1,
        ht_bandwidth_mode="interpolate-only",
        allow_missing_rois=True,
    )

    expected_files = {
        "fig_L0_L9_born_vs_fresnel_corrected_HT0.png",
        "fig_L0_L9_born_vs_fresnel_vs_parratt_envelope_HT0.png",
        "fig_miceli_RF_vs_Born_asymptote.png",
        "fig_miceli_correction_factor.png",
        "fig_L0_L9_parratt_vs_born_ht0.png",
        "fig_L0_L9_miceli_correction_factor.png",
        "fig_near_Qc_parratt_vs_born_ht0.png",
        "fig_Qz0_parratt_flat_born_diverges.png",
        "fig_HT_p0_vs_SF_only.png",
        "fig_detector_roi_overlay.png",
        "model_curves_L_Qz.csv",
        "marker_table_L_Qz.csv",
        "report.md",
    }

    assert expected_files <= {path.name for path in tmp_path.iterdir()}

    with (tmp_path / "model_curves_L_Qz.csv").open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    required_curve_columns = {
        "material",
        "curve_name",
        "L",
        "Qz_Ainv",
        "intensity",
        "source",
        "p_user",
        "stack_layers",
        "thickness_nm",
        "bandwidth_fwhm",
        "bandwidth_mode",
        "substrate",
        "n_wavelength_samples",
        "lambda0_angstrom",
        "qz_floor_for_born_scaling",
    }
    assert required_curve_columns <= set(rows[0])
    assert {
        "HT_p0_normalized",
        "Born_scaled_HT_p0",
        "Fresnel_corrected_HT_p0",
        "Parratt_envelope_HT_p0",
        "Parratt_reflectivity",
        "Miceli_correction_factor",
    } <= {row["curve_name"] for row in rows}
    assert {row["bandwidth_mode"] for row in rows} == {"interpolate-only"}
    assert {row["qz_floor_for_born_scaling"] for row in rows} == {"1e-08"}

    with (tmp_path / "marker_table_L_Qz.csv").open(newline="", encoding="utf-8") as fp:
        marker_rows = list(csv.DictReader(fp))

    assert {"material", "feature", "Qz_Ainv", "L", "source"} <= set(marker_rows[0])
    assert "Qc" in {row["feature"] for row in marker_rows}

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Born-scaled HT p=0" in report
    assert "Fresnel-corrected HT p=0" in report
    assert "Parratt-envelope HT p=0" in report
    assert "air / Bi2Se3(100 nm) / SiO2" in report
    assert "interpolate-only" in report
    assert "does not prove total external reflection" in report
    assert "exact diffraction" not in report


def test_detector_roi_overlay_requires_input_unless_allowed(tmp_path, monkeypatch) -> None:
    def fake_simulate(material, *, lambda_angstrom, L_step_dense=0.002, L_max_padded=9.5):
        del material, lambda_angstrom, L_step_dense, L_max_padded
        L = np.linspace(0.0, 9.5, 96)
        return L, 10.0 / (1.0 + L)

    monkeypatch.setattr(module_under_test, "simulate_ht_p0_dense", fake_simulate)

    with pytest.raises(ValueError, match="ROI"):
        run_l_axis_reflectivity_diagnostics(
            output_dir=tmp_path,
            material=_material(),
            L_values=np.linspace(0.0, 9.0, 10),
            n_wavelength_samples=1,
            ht_bandwidth_mode="interpolate-only",
            allow_missing_rois=False,
        )


def test_cli_delegates_to_diagnostic_runner(tmp_path, monkeypatch) -> None:
    script = importlib.import_module("scripts.plot_l_axis_reflectivity_diagnostics")
    seen = {}

    def fake_run_l_axis_reflectivity_diagnostics(**kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(
        script,
        "run_l_axis_reflectivity_diagnostics",
        fake_run_l_axis_reflectivity_diagnostics,
    )

    result = script.main(
        [
            "--output-dir",
            str(tmp_path),
            "--allow-missing-rois",
            "--ht-bandwidth-mode",
            "interpolate-only",
            "--n-wavelength-samples",
            "1",
        ]
    )

    assert result == 0
    assert seen["output_dir"] == tmp_path
    assert seen["allow_missing_rois"] is True
    assert seen["ht_bandwidth_mode"] == "interpolate-only"
    assert seen["n_wavelength_samples"] == 1
    assert seen["material"].name == "Bi2Se3"
