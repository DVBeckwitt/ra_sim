"""Atomic factor helpers for VESTA-parity diagnostics."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


VESTA_CU_KA1_ANOMALOUS_TERMS = {
    "Bi": (-4.23706, 8.83640),
    "Se": (-0.787865, 1.13462),
}


def s_from_d(d_angstrom: float) -> float:
    if not math.isfinite(d_angstrom) or d_angstrom <= 0.0:
        raise ValueError("d spacing must be positive and finite.")
    return 1.0 / (2.0 * d_angstrom)


def _qmag_from_s(s_values) -> np.ndarray:
    return 4.0 * np.pi * np.asarray(s_values, dtype=float).reshape(-1)


def _normalize_elements(elements: str | Iterable[str]) -> np.ndarray:
    return np.asarray([elements] if isinstance(elements, str) else list(elements), dtype=str)


def f0(
    elements: str | Iterable[str],
    s_values,
    *,
    table: str = "waaskirf",
) -> np.ndarray:
    import Dans_Diffraction.functions_crystallography as fc

    element_array = _normalize_elements(elements)
    qmag = _qmag_from_s(s_values)
    if table == "waaskirf":
        return fc.xray_scattering_factor_WaasKirf(element_array, qmag)
    if table == "itc":
        return fc.xray_scattering_factor(element_array, qmag)
    raise ValueError(f"Unsupported scattering table: {table}")


def anomalous_terms(
    elements: str | Iterable[str],
    wavelength_angstrom: float,
    *,
    mode: str = "vesta_cu_ka1",
) -> tuple[np.ndarray, np.ndarray]:
    element_array = _normalize_elements(elements)
    if mode == "off":
        return np.zeros(len(element_array)), np.zeros(len(element_array))
    if mode == "vesta_cu_ka1":
        missing = sorted({el for el in element_array if el not in VESTA_CU_KA1_ANOMALOUS_TERMS})
        if missing:
            raise ValueError(f"No VESTA Cu Kalpha anomalous terms for: {missing}")
        f_prime = np.array([VESTA_CU_KA1_ANOMALOUS_TERMS[el][0] for el in element_array])
        f_double_prime = np.array([VESTA_CU_KA1_ANOMALOUS_TERMS[el][1] for el in element_array])
        return f_prime, f_double_prime
    if mode == "xraydb":
        import Dans_Diffraction.functions_crystallography as fc

        energy_kev = 12.398419843320026 / float(wavelength_angstrom)
        f_prime, package_f2 = fc.xray_dispersion_corrections(
            element_array,
            np.array([energy_kev]),
        )
        return np.asarray(f_prime[0], dtype=float), -np.asarray(package_f2[0], dtype=float)
    raise ValueError(f"Unsupported anomalous mode: {mode}")


def f_total(
    elements: str | Iterable[str],
    s_values,
    wavelength_angstrom: float,
    *,
    table: str = "waaskirf",
    anomalous_mode: str = "vesta_cu_ka1",
) -> np.ndarray:
    element_array = _normalize_elements(elements)
    base = f0(element_array, s_values, table=table).astype(complex)
    f_prime, f_double_prime = anomalous_terms(
        element_array,
        wavelength_angstrom,
        mode=anomalous_mode,
    )
    return base + f_prime.reshape(1, -1) + 1j * f_double_prime.reshape(1, -1)


def atomic_factor_debug_table(
    elements: Iterable[str],
    s_value: float,
    wavelength_angstrom: float,
    *,
    table: str = "waaskirf",
    anomalous_mode: str = "vesta_cu_ka1",
) -> list[dict[str, float | str]]:
    element_array = _normalize_elements(elements)
    f0_values = f0(element_array, [s_value], table=table)[0]
    f_prime, f_double_prime = anomalous_terms(
        element_array,
        wavelength_angstrom,
        mode=anomalous_mode,
    )
    total = f_total(
        element_array,
        [s_value],
        wavelength_angstrom,
        table=table,
        anomalous_mode=anomalous_mode,
    )[0]
    return [
        {
            "element": str(element),
            "s": float(s_value),
            "f0": float(f0_values[index]),
            "f_prime": float(f_prime[index]),
            "f_double_prime": float(f_double_prime[index]),
            "f_total_real": float(total[index].real),
            "f_total_imag": float(total[index].imag),
        }
        for index, element in enumerate(element_array)
    ]
