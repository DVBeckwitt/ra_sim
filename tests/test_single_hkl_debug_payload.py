import json
from pathlib import Path

import pytest

from ra_sim.structure_factors.options import StructureFactorOptions
from ra_sim.structure_factors.raw_f import compute_raw_f_debug_payload
from tests.helpers.vesta_reference import (
    infer_lambda_from_vesta_rows,
    parse_vesta_structure_factor_txt,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
CIF_PATH = FIXTURE_DIR / "Bi2Se3.cif"
TXT_PATH = FIXTURE_DIR / "Bi2Se3_vesta_cu_ka1_dmin_0p7.txt"


def _payload(hkl=(1, 0, 7)):
    pytest.importorskip("Dans_Diffraction")
    rows = parse_vesta_structure_factor_txt(TXT_PATH)
    return compute_raw_f_debug_payload(
        CIF_PATH,
        hkl,
        infer_lambda_from_vesta_rows(rows),
        StructureFactorOptions.vesta_cu_ka1(),
    )


def test_single_hkl_debug_payload_has_all_terms():
    payload = _payload()
    site = payload["expanded_atom_list"][0]

    assert {
        "label",
        "element",
        "occupancy",
        "s",
        "f0",
        "f_prime",
        "f_double_prime",
        "f_total_real",
        "f_total_imag",
        "debye_waller",
        "phase_real",
        "phase_imag",
        "contribution_real",
        "contribution_imag",
    } <= set(site)


def test_single_hkl_contributions_sum_to_reported_f():
    payload = _payload((0, 1, 5))

    real = sum(site["contribution_real"] for site in payload["expanded_atom_list"])
    imag = sum(site["contribution_imag"] for site in payload["expanded_atom_list"])

    assert real == pytest.approx(payload["f_real"], abs=1e-10)
    assert imag == pytest.approx(payload["f_imag"], abs=1e-10)


def test_single_hkl_debug_payload_serializes_to_json():
    payload = _payload((4, 0, 19))

    assert json.loads(json.dumps(payload))["h"] == 4


def test_nan_twotheta_hkl_has_raw_f_debug_payload():
    rows = parse_vesta_structure_factor_txt(TXT_PATH)
    nan_row = next(row for row in rows if row.two_theta is None)
    payload = _payload(nan_row.hkl)

    assert payload["two_theta"] is None
    assert payload["f_abs"] > 0.0
