from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ra_sim.utils.diffraction_tools import miller_generator
from ra_sim.utils.pbi2_ht_shift_cif import generate_pbii_ht_shifted_cif


def _write_charged_pbii_cif(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "data_charged_pbii",
                "_cell_length_a 4.557",
                "_cell_length_b 4.557",
                "_cell_length_c 6.979",
                "_cell_angle_alpha 90",
                "_cell_angle_beta 90",
                "_cell_angle_gamma 120",
                "loop_",
                "_space_group_symop_operation_xyz",
                "'x, y, z'",
                "loop_",
                "_atom_site_label",
                "_atom_site_occupancy",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "_atom_site_type_symbol",
                "Pb1 1.0 0 0 0 Pb2+",
                "I1 1.0 0.333333 0.666667 0.2675 I1-",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _run_real_miller(cif_path: Path):
    lambda_value = 1.5406
    return miller_generator(
        4,
        str(cif_path),
        [1.0],
        lambda_value,
        energy=12.398419843320026 / lambda_value,
        intensity_threshold=0.0,
        two_theta_range=(0.0, 70.0),
    )


def test_real_generated_disordered_cif_produces_miller_rows(tmp_path):
    pytest.importorskip("CifFile")
    pytest.importorskip("Dans_Diffraction")

    source_2h = Path("tests/Diffuse/PbI2_2H.cif")
    packaged_6h = Path("ra_sim/config/materials/PbI2_6H.cif")
    assert source_2h.is_file()
    assert packaged_6h.is_file()

    generated = generate_pbii_ht_shifted_cif(
        source_cif=source_2h,
        output_dir=tmp_path,
        mode="compact_6h",
    )
    generated_path = Path(generated.cif_path)

    generated_miller, generated_intensities, *_ = _run_real_miller(generated_path)
    packaged_miller, packaged_intensities, *_ = _run_real_miller(packaged_6h)

    assert generated_path.is_file()
    assert np.asarray(generated_miller).shape[0] > 0
    assert np.asarray(generated_intensities).size > 0
    assert np.asarray(packaged_miller).shape[0] > 0
    assert np.asarray(packaged_intensities).size > 0


def test_real_charged_active_pbii_cif_generates_disordered_cif(tmp_path):
    source = _write_charged_pbii_cif(tmp_path / "charged-pbi2.cif")

    generated = generate_pbii_ht_shifted_cif(
        source_cif=source,
        output_dir=tmp_path,
        mode="compact_6h",
    )

    assert Path(generated.cif_path).is_file()


def test_real_charged_active_pbii_cif_produces_miller_rows_when_available(tmp_path):
    pytest.importorskip("CifFile")
    pytest.importorskip("Dans_Diffraction")
    source = _write_charged_pbii_cif(tmp_path / "charged-pbi2.cif")
    generated = generate_pbii_ht_shifted_cif(
        source_cif=source,
        output_dir=tmp_path,
        mode="compact_6h",
    )

    miller_values, intensity_values, *_ = _run_real_miller(Path(generated.cif_path))

    assert np.asarray(miller_values).shape[0] > 0
    assert np.asarray(intensity_values).size > 0
