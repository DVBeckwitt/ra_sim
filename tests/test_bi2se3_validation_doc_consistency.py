from pathlib import Path
import re

from tests.helpers.vesta_reference import (
    EXPECTED_BI2SE3_CIF_SHA256,
    EXPECTED_BI2SE3_FINITE_TWOTHETA_COUNT,
    EXPECTED_BI2SE3_NAN_TWOTHETA_COUNT,
    EXPECTED_BI2SE3_ROW_COUNT,
    EXPECTED_BI2SE3_TXT_SHA256,
    load_fixture_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "validation" / "bi2se3_vesta_structure_factor_validation.md"
METADATA_PATH = REPO_ROOT / "tests" / "fixtures" / "Bi2Se3_vesta_cu_ka1_dmin_0p7.metadata.json"


def test_bi2se3_validation_doc_matches_fixture_metadata():
    text = DOC_PATH.read_text(encoding="utf-8")
    metadata = load_fixture_metadata(METADATA_PATH)

    assert EXPECTED_BI2SE3_CIF_SHA256 in text
    assert EXPECTED_BI2SE3_TXT_SHA256 in text
    assert re.search(rf"\b{EXPECTED_BI2SE3_ROW_COUNT}\s+total HKL rows\b", text)
    assert re.search(rf"\b{EXPECTED_BI2SE3_FINITE_TWOTHETA_COUNT}\s+finite `2theta` rows\b", text)
    assert re.search(
        rf"\b{EXPECTED_BI2SE3_NAN_TWOTHETA_COUNT}\s+rows with non-finite `2theta`", text
    )
    assert f"Se1 occupancy = {metadata['site_occupancy']['Se1']:.4f}" in text
    assert "max relative `|F|` error | < 0.002" in text
