import math
from pathlib import Path

import pytest

from ra_sim.structure_factors.options import StructureFactorOptions
from ra_sim.structure_factors.raw_f import compute_raw_complex_f, compute_raw_f_debug_payload
from tests.helpers.vesta_reference import (
    compare_hkl_tables,
    infer_lambda_from_vesta_rows,
    parse_vesta_structure_factor_txt,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
CIF_PATH = FIXTURE_DIR / "Bi2Se3.cif"
TXT_PATH = FIXTURE_DIR / "Bi2Se3_vesta_cu_ka1_dmin_0p7.txt"


def _p1_cif(tmp_path: Path, atoms: str) -> Path:
    path = tmp_path / "p1.cif"
    path.write_text(
        "\n".join(
            [
                "data_test",
                "_cell_length_a 10",
                "_cell_length_b 10",
                "_cell_length_c 10",
                "_cell_angle_alpha 90",
                "_cell_angle_beta 90",
                "_cell_angle_gamma 90",
                "_space_group_name_H-M_alt 'P 1'",
                "_space_group_IT_number 1",
                "loop_",
                "_space_group_symop_operation_xyz",
                "'x,y,z'",
                "loop_",
                "_atom_site_label",
                "_atom_site_type_symbol",
                "_atom_site_occupancy",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "_atom_site_U_iso_or_equiv",
                atoms,
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_one_atom_p1_matches_occupancy_factor_and_debye_waller(tmp_path):
    pytest.importorskip("Dans_Diffraction")
    cif = _p1_cif(tmp_path, "C1 C 0.5 0 0 0 0.01")
    options = StructureFactorOptions(
        scattering_table="constant",
        anomalous_mode="off",
        constant_factors={"C": 10.0},
    )

    payload = compute_raw_f_debug_payload(cif, (0, 0, 1), 1.0, options)
    site = payload["expanded_atom_list"][0]
    expected = site["occupancy"] * site["f_total_real"] * site["debye_waller"]

    assert math.isclose(payload["f_real"], expected, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(payload["f_imag"], 0.0, abs_tol=1e-12)


def test_two_atom_p1_phase_cancellation(tmp_path):
    pytest.importorskip("Dans_Diffraction")
    cif = _p1_cif(tmp_path, "C1 C 1 0 0 0 0\nC2 C 1 0.5 0 0 0")
    options = StructureFactorOptions(
        scattering_table="constant",
        anomalous_mode="off",
        debye_waller_mode="off",
        constant_factors={"C": 10.0},
    )

    value = compute_raw_complex_f(cif, (1, 0, 0), 1.0, options)

    assert abs(value) < 1e-12


def test_raw_f_returns_complex_number():
    pytest.importorskip("Dans_Diffraction")
    value = compute_raw_complex_f(
        CIF_PATH,
        (0, 0, 3),
        1.5405929254021151,
        StructureFactorOptions.vesta_cu_ka1(),
    )

    assert isinstance(value, complex)
    assert not math.isclose(abs(value), abs(value) ** 2)


def test_multiplicity_does_not_scale_raw_f():
    pytest.importorskip("Dans_Diffraction")
    rows = parse_vesta_structure_factor_txt(TXT_PATH)
    row = next(item for item in rows if item.hkl == (1, 0, 7))
    value = compute_raw_complex_f(
        CIF_PATH,
        row.hkl,
        infer_lambda_from_vesta_rows(rows),
        StructureFactorOptions.vesta_cu_ka1(),
    )

    assert not math.isclose(abs(value), row.multiplicity * row.f_abs, rel_tol=1e-3)
    assert not math.isclose(abs(value), math.sqrt(row.multiplicity) * row.f_abs, rel_tol=1e-3)


def test_lorentz_polarization_not_in_raw_f():
    pytest.importorskip("Dans_Diffraction")
    payload = compute_raw_f_debug_payload(
        CIF_PATH,
        (0, 1, 5),
        1.5405929254021151,
        StructureFactorOptions.vesta_cu_ka1(),
    )

    text = str(payload["options"]).lower()
    forbidden = ("lorentz", "polarization", "powder", "profile", "preferred", "scale_factor")
    assert not any(term in text for term in forbidden)


def test_intensity_api_not_used_for_vesta_f_comparison(monkeypatch):
    pytest.importorskip("Dans_Diffraction")
    rows = parse_vesta_structure_factor_txt(TXT_PATH)[:1]

    def bad_intensity(*_args, **_kwargs):
        raise AssertionError("intensity API must not be used for VESTA F comparison")

    import Dans_Diffraction.classes_scattering as classes_scattering

    monkeypatch.setattr(classes_scattering.Scattering, "intensity", bad_intensity)
    payload = compute_raw_f_debug_payload(
        CIF_PATH,
        rows[0].hkl,
        infer_lambda_from_vesta_rows(rows),
        StructureFactorOptions.vesta_cu_ka1(),
    )
    comparison = compare_hkl_tables(rows, [payload])

    assert comparison[0].f_abs_rel_error < 0.001
