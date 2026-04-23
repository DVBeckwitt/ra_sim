import math
from pathlib import Path

import pytest

from ra_sim.structure_factors.options import StructureFactorOptions
from ra_sim.structure_factors.raw_f import compute_raw_f_debug_payload
from tests.helpers.vesta_reference import (
    EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM,
    parse_vesta_structure_factor_txt,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
CIF_PATH = FIXTURE_DIR / "Bi2Se3.cif"
TXT_PATH = FIXTURE_DIR / "Bi2Se3_vesta_cu_ka1_dmin_0p7.txt"


def _payloads():
    pytest.importorskip("Dans_Diffraction")
    rows = parse_vesta_structure_factor_txt(TXT_PATH)
    wavelength = EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM
    options = StructureFactorOptions.vesta_cu_ka1()
    return rows, [
        compute_raw_f_debug_payload(CIF_PATH, row.hkl, wavelength, options) for row in rows
    ]


def test_bi2se3_d_spacing_matches_vesta():
    rows, payloads = _payloads()

    assert max(abs(payload["d"] - row.d) for row, payload in zip(rows, payloads)) < 1e-5


def test_bi2se3_twotheta_matches_vesta_for_finite_rows():
    rows, payloads = _payloads()

    errors = [
        abs(payload["two_theta"] - row.two_theta)
        for row, payload in zip(rows, payloads)
        if row.two_theta is not None
    ]
    assert max(errors) < 1e-4


def test_f_abs_comparison_does_not_require_finite_twotheta():
    rows, payloads = _payloads()

    assert len(payloads) == 206
    assert sum(row.two_theta is None for row in rows) == 45
    assert all(math.isfinite(payload["f_abs"]) and payload["f_abs"] > 0.0 for payload in payloads)
