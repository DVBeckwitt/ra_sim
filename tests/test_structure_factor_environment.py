from pathlib import Path

import pytest

from ra_sim.structure_factors.options import StructureFactorOptions
from ra_sim.validation.environment import collect_structure_factor_environment
from tests.helpers.vesta_reference import (
    EXPECTED_BI2SE3_CIF_SHA256,
    EXPECTED_BI2SE3_WAVELENGTH_ANGSTROM,
)


CIF_PATH = Path(__file__).parent / "fixtures" / "Bi2Se3.cif"


def test_environment_snapshot_has_required_keys():
    env = collect_structure_factor_environment(
        CIF_PATH,
        EXPECTED_BI2SE3_WAVELENGTH_ANGSTROM,
        StructureFactorOptions.vesta_cu_ka1(),
    )

    assert {
        "python_version",
        "numpy_version",
        "Dans_Diffraction_version",
        "xraydb_version",
        "spglib_version",
        "cif_sha256",
        "wavelength_angstrom",
        "anomalous_mode",
        "debye_waller_mode",
        "occupancy_mode",
        "scattering_factor_source",
    } <= set(env)


def test_cif_hash_is_stable():
    env = collect_structure_factor_environment(CIF_PATH, EXPECTED_BI2SE3_WAVELENGTH_ANGSTROM)

    assert env["cif_sha256"] == EXPECTED_BI2SE3_CIF_SHA256


def test_wavelength_must_be_explicit():
    with pytest.raises(ValueError, match="explicit"):
        collect_structure_factor_environment(CIF_PATH, None)  # type: ignore[arg-type]


def test_no_default_wavelength_or_default_energy_is_used():
    env = collect_structure_factor_environment(
        CIF_PATH,
        EXPECTED_BI2SE3_WAVELENGTH_ANGSTROM,
        StructureFactorOptions.vesta_cu_ka1(),
    )

    assert env["wavelength_angstrom"] == EXPECTED_BI2SE3_WAVELENGTH_ANGSTROM
    assert env["anomalous_mode"] == "vesta_cu_ka1"
    assert env["scattering_factor_source"] == "waaskirf"
