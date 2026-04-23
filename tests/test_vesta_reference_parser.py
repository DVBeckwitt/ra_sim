import hashlib
import math
from pathlib import Path

import numpy as np

from tests.helpers.vesta_reference import (
    EXPECTED_BI2SE3_CIF_SHA256,
    EXPECTED_BI2SE3_FINITE_TWOTHETA_COUNT,
    EXPECTED_BI2SE3_NAN_TWOTHETA_COUNT,
    EXPECTED_BI2SE3_ROW_COUNT,
    EXPECTED_BI2SE3_TXT_SHA256,
    EXPECTED_BI2SE3_WAVELENGTH_ANGSTROM,
    infer_lambda_from_vesta_rows,
    load_fixture_metadata,
    parse_vesta_structure_factor_txt,
    write_comparison_csv,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
TXT_PATH = FIXTURE_DIR / "Bi2Se3_vesta_cu_ka1_dmin_0p7.txt"
CIF_PATH = FIXTURE_DIR / "Bi2Se3.cif"
METADATA_PATH = FIXTURE_DIR / "Bi2Se3_vesta_cu_ka1_dmin_0p7.metadata.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parse_vesta_txt_schema():
    rows = parse_vesta_structure_factor_txt(TXT_PATH)

    assert len(rows) == 206
    assert sum(row.two_theta is not None for row in rows) == 161
    assert all(isinstance(row.h, int) for row in rows)
    assert all(isinstance(row.k, int) for row in rows)
    assert all(isinstance(row.l, int) for row in rows)
    assert all(math.isfinite(row.d) and row.d > 0.0 for row in rows)
    assert all(math.isfinite(row.f_abs) and row.f_abs > 0.0 for row in rows)


def test_vesta_abs_f_matches_real_imag():
    rows = parse_vesta_structure_factor_txt(TXT_PATH)

    for row in rows:
        reconstructed = math.hypot(row.f_real, row.f_imag)
        assert math.isclose(reconstructed, row.f_abs, rel_tol=5e-5, abs_tol=5e-4)


def test_vesta_txt_implied_wavelength():
    rows = parse_vesta_structure_factor_txt(TXT_PATH)

    assert math.isclose(
        infer_lambda_from_vesta_rows(rows),
        EXPECTED_BI2SE3_WAVELENGTH_ANGSTROM,
        abs_tol=1e-5,
    )


def test_nan_twotheta_rows_are_beyond_bragg_condition():
    rows = parse_vesta_structure_factor_txt(TXT_PATH)
    wavelength = infer_lambda_from_vesta_rows(rows)
    cutoff = wavelength / 2.0

    finite = [row for row in rows if row.two_theta is not None]
    nan_rows = [row for row in rows if row.two_theta is None]

    assert finite
    assert nan_rows
    assert all(row.d >= cutoff for row in finite)
    assert all(row.d < cutoff for row in nan_rows)


def test_vesta_fixture_metadata_has_export_settings():
    metadata = load_fixture_metadata(METADATA_PATH)

    assert metadata["fixture_role"] == "VESTA parity reference, not absolute physical truth"
    assert metadata["radiation"] == "X-ray"
    assert metadata["line"] == "Cu Kalpha1"
    assert metadata["factor_convention"] == "VESTA Cu Kalpha1 legacy mode"
    assert metadata["dmin_angstrom"] == 0.7
    assert metadata["row_count"] == EXPECTED_BI2SE3_ROW_COUNT
    assert metadata["finite_two_theta_count"] == EXPECTED_BI2SE3_FINITE_TWOTHETA_COUNT
    assert metadata["nan_two_theta_count"] == EXPECTED_BI2SE3_NAN_TWOTHETA_COUNT
    assert metadata["cif_sha256"] == EXPECTED_BI2SE3_CIF_SHA256
    assert metadata["txt_sha256"] == EXPECTED_BI2SE3_TXT_SHA256
    assert metadata["site_occupancy"]["Se1"] == 1.0
    assert "vesta_version" in metadata
    assert "anomalous_dispersion_setting" in metadata
    assert "thermal_factor_setting" in metadata
    assert metadata["inferred_lambda_angstrom"] == 1.540592925
    assert metadata["lambda_inferred_from_txt_angstrom"] == EXPECTED_BI2SE3_WAVELENGTH_ANGSTROM


def test_no_accidental_reference_update():
    rows = parse_vesta_structure_factor_txt(TXT_PATH)

    assert _sha256(CIF_PATH) == EXPECTED_BI2SE3_CIF_SHA256
    assert _sha256(TXT_PATH) == EXPECTED_BI2SE3_TXT_SHA256
    assert len(rows) == EXPECTED_BI2SE3_ROW_COUNT
    assert sum(row.two_theta is not None for row in rows) == EXPECTED_BI2SE3_FINITE_TWOTHETA_COUNT
    assert sum(row.two_theta is None for row in rows) == EXPECTED_BI2SE3_NAN_TWOTHETA_COUNT
    assert np.isclose(infer_lambda_from_vesta_rows(rows), EXPECTED_BI2SE3_WAVELENGTH_ANGSTROM)


def test_empty_comparison_csv_writes_header(tmp_path):
    output = tmp_path / "empty.csv"

    write_comparison_csv(output, [])

    assert output.read_text(encoding="utf-8").startswith("h,k,l,vesta_d,sim_d")
