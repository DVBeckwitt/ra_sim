from pathlib import Path

import pytest

from ra_sim.cli import _parse_cif_cell_a_c


def test_parse_cif_cell_a_c_reads_raw_values():
    a_val, c_val = _parse_cif_cell_a_c(str(Path("tests/local_test.cif")))
    assert a_val == pytest.approx(4.0)
    assert c_val == pytest.approx(10.0)


def test_parse_cif_cell_a_c_handles_uncertainty_suffix(tmp_path):
    cif_path = tmp_path / "uncertainty.cif"
    cif_path.write_text(
        "\n".join(
            [
                "data_test",
                "_cell_length_a    4.123(4)",
                "_cell_length_c    30.456(7)",
            ]
        ),
        encoding="utf-8",
    )
    a_val, c_val = _parse_cif_cell_a_c(str(cif_path))
    assert a_val == pytest.approx(4.123)
    assert c_val == pytest.approx(30.456)
