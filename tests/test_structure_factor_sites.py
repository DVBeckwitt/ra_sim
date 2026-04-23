import hashlib
import json
import math
from pathlib import Path

import pytest

from ra_sim.structure_factors.options import StructureFactorOptions
from ra_sim.structure_factors.raw_f import compute_raw_complex_f, compute_raw_f_debug_payload


FIXTURE_DIR = Path(__file__).parent / "fixtures"
CIF_PATH = FIXTURE_DIR / "Bi2Se3.cif"


def test_expanded_site_list_is_deterministic():
    pytest.importorskip("Dans_Diffraction")
    payload = compute_raw_f_debug_payload(
        CIF_PATH,
        (1, 0, 7),
        1.5405929254021151,
        StructureFactorOptions.vesta_cu_ka1(),
    )
    serialized = json.dumps(payload["expanded_atom_list"], sort_keys=True)

    assert hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def test_no_unrecognized_elements():
    pytest.importorskip("Dans_Diffraction")
    payload = compute_raw_f_debug_payload(
        CIF_PATH,
        (1, 0, 7),
        1.5405929254021151,
        StructureFactorOptions.vesta_cu_ka1(),
    )

    assert {site["element"] for site in payload["expanded_atom_list"]} == {"Bi", "Se"}


def test_bi2se3_se1_occupancy_is_unit():
    pytest.importorskip("Dans_Diffraction")
    payload = compute_raw_f_debug_payload(
        CIF_PATH,
        (1, 0, 7),
        1.5405929254021151,
        StructureFactorOptions.vesta_cu_ka1(),
    )
    se1_occupancies = {
        site["occupancy"] for site in payload["expanded_atom_list"] if site["label"] == "Se1"
    }

    assert se1_occupancies == {1.0}


def test_occupancy_applied_once(tmp_path):
    pytest.importorskip("Dans_Diffraction")
    cif = tmp_path / "partial.cif"
    cif.write_text(
        "\n".join(
            [
                "data_partial",
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
                "C1 C 0.25 0 0 0 0",
            ]
        ),
        encoding="utf-8",
    )
    base = StructureFactorOptions(
        scattering_table="constant",
        anomalous_mode="off",
        debye_waller_mode="off",
        constant_factors={"C": 8.0},
    )

    cif_occ = abs(compute_raw_complex_f(cif, (0, 0, 1), 1.0, base))
    unit_occ = abs(
        compute_raw_complex_f(
            cif,
            (0, 0, 1),
            1.0,
            StructureFactorOptions(
                scattering_table="constant",
                anomalous_mode="off",
                debye_waller_mode="off",
                occupancy_mode="unit",
                constant_factors={"C": 8.0},
            ),
        )
    )

    assert math.isclose(cif_occ, 2.0, rel_tol=1e-12)
    assert math.isclose(unit_occ, 8.0, rel_tol=1e-12)


def test_origin_shift_preserves_abs_f(tmp_path):
    pytest.importorskip("Dans_Diffraction")
    options = StructureFactorOptions(
        scattering_table="constant",
        anomalous_mode="off",
        debye_waller_mode="off",
        constant_factors={"C": 8.0},
    )
    paths = []
    for shift in (0.0, 0.125):
        path = tmp_path / f"origin_{shift}.cif"
        path.write_text(
            "\n".join(
                [
                    "data_origin",
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
                    f"C1 C 1 {shift} {shift} {shift} 0",
                    f"C2 C 1 {0.25 + shift} {shift} {shift} 0",
                ]
            ),
            encoding="utf-8",
        )
        paths.append(path)

    first = compute_raw_complex_f(paths[0], (1, 0, 0), 1.0, options)
    shifted = compute_raw_complex_f(paths[1], (1, 0, 0), 1.0, options)

    assert abs(first) == pytest.approx(abs(shifted), abs=1e-12)
