"""Environment snapshots for structure-factor parity tests."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
from pathlib import Path
import platform

import numpy as np

from ra_sim.structure_factors.options import StructureFactorOptions


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def collect_structure_factor_environment(
    cif_path: str | Path,
    wavelength_angstrom: float,
    options: StructureFactorOptions | None = None,
) -> dict[str, object]:
    if wavelength_angstrom is None:
        raise ValueError("wavelength_angstrom must be explicit.")
    options = options or StructureFactorOptions.package_default()
    cif_bytes = Path(cif_path).read_bytes()
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "Dans_Diffraction_version": _version("Dans_Diffraction"),
        "xraydb_version": _version("xraydb"),
        "spglib_version": _version("spglib"),
        "cif_sha256": hashlib.sha256(cif_bytes).hexdigest(),
        "wavelength_angstrom": float(wavelength_angstrom),
        "anomalous_mode": options.anomalous_mode,
        "debye_waller_mode": options.debye_waller_mode,
        "occupancy_mode": options.occupancy_mode,
        "scattering_factor_source": options.scattering_table,
    }
