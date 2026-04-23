from pathlib import Path

import pytest

from ra_sim.structure_factors.options import StructureFactorOptions
from ra_sim.structure_factors.sweep import build_switch_sweep
from tests.helpers.vesta_reference import (
    EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM,
    parse_vesta_structure_factor_txt,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
CIF_PATH = FIXTURE_DIR / "Bi2Se3.cif"
TXT_PATH = FIXTURE_DIR / "Bi2Se3_vesta_cu_ka1_dmin_0p7.txt"


def _sweep():
    pytest.importorskip("Dans_Diffraction")
    rows = parse_vesta_structure_factor_txt(TXT_PATH)
    return build_switch_sweep(
        rows,
        str(CIF_PATH),
        EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM,
        StructureFactorOptions.vesta_cu_ka1(),
    )


def test_switch_sweep_changes_one_variable_at_a_time():
    sweep = _sweep()

    assert sweep[0]["changed_options"] == []
    assert all(len(row["changed_options"]) == 1 for row in sweep[1:])


def test_switch_sweep_outputs_error_summary():
    row = _sweep()[0]

    assert {
        "mean_abs_error",
        "median_rel_error",
        "mean_rel_error",
        "max_rel_error",
        "worst_hkl",
        "mean_f_real_abs_error",
        "mean_f_imag_abs_error",
        "finite_two_theta_count",
        "mean_two_theta_abs_error",
        "max_two_theta_abs_error",
        "max_d_abs_error",
    } <= set(row["summary"])


def test_wavelength_sweep_reports_geometry_signal():
    sweep = {row["name"]: row for row in _sweep()}

    assert (
        sweep["wavelength_ui_1p5409"]["summary"]["max_two_theta_abs_error"]
        > sweep["baseline"]["summary"]["max_two_theta_abs_error"]
    )


def test_anomalous_toggle_has_nonzero_effect():
    sweep = {row["name"]: row for row in _sweep()}

    assert sweep["anomalous_off"]["summary"]["max_rel_error"] > 1e-4


def test_debye_waller_toggle_reports_effect():
    sweep = {row["name"]: row for row in _sweep()}

    assert sweep["debye_waller_off"]["summary"]["max_rel_error"] > 1e-4


def test_occupancy_toggle_reports_effect():
    sweep = {row["name"]: row for row in _sweep()}

    assert sweep["unit_occupancy"]["summary"]["max_rel_error"] > 1e-4
