import json
import os
from pathlib import Path

import pytest

from ra_sim.structure_factors.options import StructureFactorOptions
from ra_sim.structure_factors.raw_f import compute_raw_f_debug_payload
from ra_sim.structure_factors.sweep import build_switch_sweep
from ra_sim.validation.environment import collect_structure_factor_environment
from tests.helpers.vesta_reference import (
    EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM,
    compare_hkl_tables,
    infer_lambda_from_vesta_rows,
    parse_vesta_structure_factor_txt,
    summarize_f_errors,
    write_comparison_csv,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
CIF_PATH = FIXTURE_DIR / "Bi2Se3.cif"
TXT_PATH = FIXTURE_DIR / "Bi2Se3_vesta_cu_ka1_dmin_0p7.txt"
ARTIFACT_DIR = Path("artifacts/bi2se3_vesta_reference")


def _comparison():
    pytest.importorskip("Dans_Diffraction")
    rows = parse_vesta_structure_factor_txt(TXT_PATH)
    wavelength = EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM
    options = StructureFactorOptions.vesta_cu_ka1()
    sim = [compute_raw_f_debug_payload(CIF_PATH, row.hkl, wavelength, options) for row in rows]
    return rows, wavelength, options, compare_hkl_tables(rows, sim)


def _write_debug_artifacts(rows, wavelength, options, comparison):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "environment.json").write_text(
        json.dumps(
            collect_structure_factor_environment(CIF_PATH, wavelength, options),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_comparison_csv(ARTIFACT_DIR / "comparison.csv", comparison)
    worst = sorted(comparison, key=lambda row: row.f_abs_rel_error, reverse=True)[:10]
    (ARTIFACT_DIR / "top_10_worst_hkls.json").write_text(
        json.dumps([row.__dict__ for row in worst], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    worst_hkl = (worst[0].h, worst[0].k, worst[0].l)
    (ARTIFACT_DIR / "worst_hkl_debug.json").write_text(
        json.dumps(
            compute_raw_f_debug_payload(CIF_PATH, worst_hkl, wavelength, options),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "switch_sweep_summary.json").write_text(
        json.dumps(
            build_switch_sweep(rows, str(CIF_PATH), wavelength, options),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_bi2se3_vesta_reference_twotheta_matches():
    _rows, _wavelength, _options, comparison = _comparison()
    errors = [
        row.two_theta_abs_error
        for row in comparison
        if row.vesta_two_theta is not None and row.two_theta_abs_error is not None
    ]

    assert max(errors) < 1e-4


def test_bi2se3_vesta_reference_f_abs_matches():
    rows, wavelength, options, comparison = _comparison()
    summary = summarize_f_errors(comparison)

    if os.environ.get("RA_SIM_WRITE_BI2SE3_DEBUG") == "1":
        _write_debug_artifacts(rows, wavelength, options, comparison)
    try:
        assert summary["median_rel_error"] < 0.001
        assert summary["max_rel_error"] < 0.002
    except AssertionError:
        _write_debug_artifacts(rows, wavelength, options, comparison)
        raise


def test_bi2se3_vesta_reference_failure_artifact_debug_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("RA_SIM_WRITE_BI2SE3_DEBUG", "1")
    monkeypatch.setattr(
        "tests.test_bi2se3_vesta_reference_regression.ARTIFACT_DIR",
        tmp_path / "debug",
    )
    rows, wavelength, options, comparison = _comparison()

    _write_debug_artifacts(rows, wavelength, options, comparison)

    assert (tmp_path / "debug" / "environment.json").exists()
    assert (tmp_path / "debug" / "comparison.csv").exists()
    assert (tmp_path / "debug" / "worst_hkl_debug.json").exists()


def test_no_silent_reference_update_combines_all_guards():
    rows = parse_vesta_structure_factor_txt(TXT_PATH)
    summary = summarize_f_errors(_comparison()[3])

    assert len(rows) == 206
    assert sum(row.two_theta is not None for row in rows) == 161
    assert infer_lambda_from_vesta_rows(rows) == pytest.approx(1.5405929254021151)
    assert summary["median_rel_error"] < 0.001
    assert summary["max_rel_error"] < 0.002


def test_wrong_se1_occupancy_fails_vesta_parity(tmp_path):
    pytest.importorskip("Dans_Diffraction")
    wrong_cif = tmp_path / "Bi2Se3_wrong_se1_occupancy.cif"
    cif_text = CIF_PATH.read_text(encoding="utf-8")
    assert "Se1        1.0000" in cif_text
    wrong_cif.write_text(
        cif_text.replace("Se1        1.0000", "Se1        0.9000"),
        encoding="utf-8",
    )
    rows = parse_vesta_structure_factor_txt(TXT_PATH)
    wavelength = EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM
    options = StructureFactorOptions.vesta_cu_ka1()
    sim = [compute_raw_f_debug_payload(wrong_cif, row.hkl, wavelength, options) for row in rows]
    summary = summarize_f_errors(compare_hkl_tables(rows, sim))

    assert not (summary["median_rel_error"] < 0.001 and summary["max_rel_error"] < 0.002)
