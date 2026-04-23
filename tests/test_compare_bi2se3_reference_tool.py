import json
from pathlib import Path
import subprocess
import sys

import pytest

from ra_sim.tools.compare_bi2se3_reference import compare_bi2se3_reference
from tests.helpers.vesta_reference import (
    EXPECTED_BI2SE3_CIF_SHA256,
    EXPECTED_BI2SE3_FINITE_TWOTHETA_COUNT,
    EXPECTED_BI2SE3_NAN_TWOTHETA_COUNT,
    EXPECTED_BI2SE3_ROW_COUNT,
    EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM,
    EXPECTED_BI2SE3_TXT_SHA256,
    EXPECTED_BI2SE3_WAVELENGTH_ANGSTROM,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]
CIF_PATH = FIXTURE_DIR / "Bi2Se3.cif"
TXT_PATH = FIXTURE_DIR / "Bi2Se3_vesta_cu_ka1_dmin_0p7.txt"


def test_compare_tool_default_uses_two_theta_balanced_wavelength():
    pytest.importorskip("Dans_Diffraction")

    summary, _comparison = compare_bi2se3_reference(TXT_PATH, CIF_PATH)

    assert summary["wavelength_mode"] == "two_theta_balanced"
    assert summary["wavelength_angstrom"] == EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM
    assert summary["max_two_theta_abs_error"] < 1e-4
    assert summary["row_count"] == EXPECTED_BI2SE3_ROW_COUNT
    assert summary["finite_two_theta_count"] == EXPECTED_BI2SE3_FINITE_TWOTHETA_COUNT
    assert summary["nan_two_theta_count"] == EXPECTED_BI2SE3_NAN_TWOTHETA_COUNT
    assert summary["cif_sha256"] == EXPECTED_BI2SE3_CIF_SHA256
    assert summary["txt_sha256"] == EXPECTED_BI2SE3_TXT_SHA256
    assert summary["se1_occupancy"] == 1.0
    assert summary["site_occupancy"]["Se1"] == 1.0
    assert summary["factor_convention"] == "VESTA Cu Kalpha1 legacy mode"


def test_compare_tool_cli_default_uses_two_theta_balanced_wavelength():
    pytest.importorskip("Dans_Diffraction")

    completed = subprocess.run(
        [sys.executable, "-m", "ra_sim.tools.compare_bi2se3_reference", "--json"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)

    assert summary["wavelength_mode"] == "two_theta_balanced"
    assert summary["wavelength_angstrom"] == EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM
    assert summary["max_two_theta_abs_error"] < 1e-4
    assert summary["row_count"] == EXPECTED_BI2SE3_ROW_COUNT
    assert summary["finite_two_theta_count"] == EXPECTED_BI2SE3_FINITE_TWOTHETA_COUNT
    assert summary["nan_two_theta_count"] == EXPECTED_BI2SE3_NAN_TWOTHETA_COUNT
    assert summary["cif_sha256"] == EXPECTED_BI2SE3_CIF_SHA256
    assert summary["txt_sha256"] == EXPECTED_BI2SE3_TXT_SHA256
    assert summary["se1_occupancy"] == 1.0
    assert summary["factor_convention"] == "VESTA Cu Kalpha1 legacy mode"


def test_compare_tool_inferred_mode_reports_inferred_wavelength():
    pytest.importorskip("Dans_Diffraction")

    summary, _comparison = compare_bi2se3_reference(
        TXT_PATH,
        CIF_PATH,
        wavelength_mode="inferred",
    )

    assert summary["wavelength_mode"] == "inferred"
    assert summary["wavelength_angstrom"] == EXPECTED_BI2SE3_WAVELENGTH_ANGSTROM
    assert summary["inferred_wavelength_angstrom"] == EXPECTED_BI2SE3_WAVELENGTH_ANGSTROM
