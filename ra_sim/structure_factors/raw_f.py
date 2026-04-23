"""Raw complex structure-factor calculation and debug payloads."""

from __future__ import annotations

import contextlib
from functools import lru_cache
import io
import math
from pathlib import Path
from typing import Any

import numpy as np

from ra_sim.structure_factors.options import StructureFactorOptions
from ra_sim.structure_factors.vesta_like_atomic_factors import (
    anomalous_terms,
    f0,
    f_total,
)
from ra_sim.utils.calculations import two_theta


@lru_cache(maxsize=16)
def _load_crystal_cached(cif_path: str):
    import Dans_Diffraction as dif

    with contextlib.redirect_stdout(io.StringIO()):
        xtl = dif.Crystal(cif_path)
        xtl.Symmetry.generate_matrices()
        xtl.generate_structure()
    return xtl


def _load_crystal(cif_path: str | Path):
    return _load_crystal_cached(str(Path(cif_path).resolve()))


def _debye_waller(uiso, qmag: float, options: StructureFactorOptions) -> np.ndarray:
    if options.debye_waller_mode == "off":
        return np.ones(len(uiso), dtype=float)
    if options.debye_waller_mode == "cif":
        import Dans_Diffraction.functions_crystallography as fc

        return np.asarray(fc.debyewaller(uiso, [qmag]), dtype=float).reshape(1, -1)[0]
    raise ValueError(f"Unsupported Debye-Waller mode: {options.debye_waller_mode}")


def _occupancy(occ, options: StructureFactorOptions) -> np.ndarray:
    if options.occupancy_mode == "unit":
        return np.ones(len(occ), dtype=float)
    if options.occupancy_mode == "cif":
        return np.asarray(occ, dtype=float)
    raise ValueError(f"Unsupported occupancy mode: {options.occupancy_mode}")


def _atomic_factors(
    atom_type,
    s_value: float,
    wavelength_angstrom: float,
    options: StructureFactorOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    atom_type = np.asarray(atom_type, dtype=str)
    if options.scattering_table == "constant":
        missing = sorted({el for el in atom_type if el not in options.constant_factors})
        if missing:
            raise ValueError(f"Missing constant factors for elements: {missing}")
        total = np.array([complex(options.constant_factors[el]) for el in atom_type])
        zeros = np.zeros(len(atom_type), dtype=float)
        return total.real, zeros, total.imag, total

    base = f0(atom_type, [s_value], table=options.scattering_table)[0]
    f_prime, f_double_prime = anomalous_terms(
        atom_type,
        wavelength_angstrom,
        mode=options.anomalous_mode,
    )
    total = f_total(
        atom_type,
        [s_value],
        wavelength_angstrom,
        table=options.scattering_table,
        anomalous_mode=options.anomalous_mode,
    )[0]
    return base, f_prime, f_double_prime, total


def compute_raw_complex_f(
    cif_path: str | Path,
    hkl: tuple[int, int, int],
    wavelength_angstrom: float,
    options: StructureFactorOptions,
) -> complex:
    payload = compute_raw_f_debug_payload(cif_path, hkl, wavelength_angstrom, options)
    return complex(payload["f_real"], payload["f_imag"])


def compute_raw_f_debug_payload(
    cif_path: str | Path,
    hkl: tuple[int, int, int],
    wavelength_angstrom: float,
    options: StructureFactorOptions,
) -> dict[str, Any]:
    if wavelength_angstrom is None:
        raise ValueError("wavelength_angstrom must be explicit.")
    wavelength = float(wavelength_angstrom)
    if not math.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("wavelength_angstrom must be positive and finite.")
    if options.phase_sign not in {-1, 1}:
        raise ValueError("phase_sign must be -1 or 1.")

    xtl = _load_crystal(cif_path)
    uvw, atom_type, label, occ, uiso, _mxmymz = xtl.Structure.get()
    hkl_array = np.asarray(hkl, dtype=float).reshape(1, 3)
    q_cart = np.asarray(xtl.Cell.calculateQ(hkl_array), dtype=float).reshape(1, 3)[0]
    qmag = float(np.linalg.norm(q_cart))
    d_spacing = (2.0 * np.pi / qmag) if qmag > 0.0 else math.inf
    s_value = qmag / (4.0 * np.pi)
    r_cart = np.asarray(xtl.Cell.calculateR(uvw), dtype=float)
    phase = np.exp(options.phase_sign * 1j * np.dot(q_cart, r_cart.T))
    occupancy = _occupancy(occ, options)
    debye_waller = _debye_waller(uiso, qmag, options)
    f0_values, f_prime, f_double_prime, atomic_f = _atomic_factors(
        atom_type,
        s_value,
        wavelength,
        options,
    )

    contributions = atomic_f * occupancy * debye_waller * phase
    scale = float(getattr(xtl, "scale", 1.0))
    summed = np.sum(contributions) / scale

    sites = []
    for index, contribution in enumerate(contributions):
        phase_value = phase[index]
        sites.append(
            {
                "site_index": index,
                "label": str(label[index]),
                "element": str(atom_type[index]),
                "fract_x": float(uvw[index][0]),
                "fract_y": float(uvw[index][1]),
                "fract_z": float(uvw[index][2]),
                "occupancy": float(occupancy[index]),
                "uiso": float(uiso[index]),
                "b_iso": float(8.0 * np.pi**2 * float(uiso[index])),
                "s": float(s_value),
                "f0": float(f0_values[index]),
                "f_prime": float(f_prime[index]),
                "f_double_prime": float(f_double_prime[index]),
                "f_total_real": float(atomic_f[index].real),
                "f_total_imag": float(atomic_f[index].imag),
                "debye_waller": float(debye_waller[index]),
                "phase_real": float(phase_value.real),
                "phase_imag": float(phase_value.imag),
                "contribution_real": float((contribution / scale).real),
                "contribution_imag": float((contribution / scale).imag),
            }
        )

    return {
        "h": int(hkl[0]),
        "k": int(hkl[1]),
        "l": int(hkl[2]),
        "d": float(d_spacing),
        "s": float(s_value),
        "qmag": float(qmag),
        "wavelength": wavelength,
        "two_theta": None
        if two_theta(d_spacing, wavelength) is None
        else float(two_theta(d_spacing, wavelength)),
        "scale": scale,
        "options": options.to_dict(),
        "expanded_atom_count": len(sites),
        "expanded_atom_list": sites,
        "f_real": float(summed.real),
        "f_imag": float(summed.imag),
        "f_abs": float(abs(summed)),
    }
