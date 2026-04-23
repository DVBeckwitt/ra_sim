import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.helpers.vesta_reference import (
    EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM,
    EXPECTED_BI2SE3_WAVELENGTH_ANGSTROM,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "debug" / "compare_single_hkl_vesta_vs_sim.py"


def _run_script(*args: str) -> dict:
    pytest.importorskip("Dans_Diffraction")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_single_hkl_debug_script_defaults_to_two_theta_balanced_wavelength():
    result = _run_script("--h", "0", "--k", "0", "--l", "3")

    assert result["wavelength_mode"] == "two_theta_balanced"
    assert result["wavelength_angstrom"] == EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM
    assert result["simulation"]["wavelength"] == EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM
    assert result["simulation"]["two_theta"] == pytest.approx(
        result["reference"]["two_theta"],
        abs=1e-4,
    )


def test_single_hkl_debug_script_inferred_mode_reports_inferred_wavelength():
    result = _run_script("--h", "0", "--k", "0", "--l", "3", "--wavelength-mode", "inferred")

    assert result["wavelength_mode"] == "inferred"
    assert result["wavelength_angstrom"] == EXPECTED_BI2SE3_WAVELENGTH_ANGSTROM
    assert result["inferred_wavelength_angstrom"] == EXPECTED_BI2SE3_WAVELENGTH_ANGSTROM
