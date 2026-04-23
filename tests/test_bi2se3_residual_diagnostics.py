from dataclasses import replace
import os
from pathlib import Path

import pytest

from ra_sim.structure_factors.options import StructureFactorOptions
from ra_sim.structure_factors.raw_f import compute_raw_f_debug_payload
from tests.helpers.vesta_reference import (
    compare_hkl_tables,
    infer_lambda_from_vesta_rows,
    parse_vesta_structure_factor_txt,
    residuals_binned_by_s,
    summarize_f_errors,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
CIF_PATH = FIXTURE_DIR / "Bi2Se3.cif"
TXT_PATH = FIXTURE_DIR / "Bi2Se3_vesta_cu_ka1_dmin_0p7.txt"


def _comparison(options: StructureFactorOptions):
    pytest.importorskip("Dans_Diffraction")
    rows = parse_vesta_structure_factor_txt(TXT_PATH)
    wavelength = infer_lambda_from_vesta_rows(rows)
    sim = [compute_raw_f_debug_payload(CIF_PATH, row.hkl, wavelength, options) for row in rows]
    return compare_hkl_tables(rows, sim)


def test_error_summary_reproduces_package_default_diagnostic_baseline():
    if os.environ.get("RA_SIM_RUN_BI2SE3_DIAGNOSTIC_BASELINE") != "1":
        pytest.skip("Set RA_SIM_RUN_BI2SE3_DIAGNOSTIC_BASELINE=1 for diagnostic baseline.")
    options = replace(StructureFactorOptions.package_default(), occupancy_mode="unit")
    summary = summarize_f_errors(_comparison(options))

    assert summary["mean_rel_error"] == pytest.approx(0.019, abs=0.003)
    assert summary["max_rel_error"] == pytest.approx(0.053, abs=0.005)
    assert summary["correlation"] > 0.999


def test_residuals_report_worst_hkls():
    options = replace(StructureFactorOptions.package_default(), occupancy_mode="unit")
    comparison = _comparison(options)
    worst = sorted(comparison, key=lambda row: row.f_abs_rel_error, reverse=True)[:10]

    assert (4, 0, 19) in {(row.h, row.k, row.l) for row in worst}


def test_residuals_binned_by_s_are_reported():
    binned = residuals_binned_by_s(_comparison(StructureFactorOptions.vesta_cu_ka1()))

    assert set(binned) == {"low_s", "mid_s", "high_s"}
    assert all(value["count"] > 0 for value in binned.values())
    assert all("mean_rel_error" in value for value in binned.values())
