"""Parratt and single-interface reflectivity helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParrattLayer:
    name: str
    qc_inv_angstrom: float
    thickness_angstrom: float | None = None
    beta: float = 0.0
    roughness_angstrom: float = 0.0


def make_bi_chalcogenide_stack(
    material: str,
    thickness_nm: float = 100.0,
    substrate: str = "SiO2",
) -> list[ParrattLayer]:
    film_qc = {
        "Bi2Se3": 0.0517,
        "Bi2Te3": 0.0519,
    }[material]

    substrate_qc = {
        "air": 0.0,
        "SiO2": 0.0305,
    }[substrate]

    return [
        ParrattLayer("air", 0.0, None),
        ParrattLayer(material, film_qc, float(thickness_nm) * 10.0),
        ParrattLayer(substrate, substrate_qc, None),
    ]


def parratt_reflectivity(
    qz_inv_angstrom: np.ndarray,
    layers: Sequence[ParrattLayer],
    wavelength_angstrom: float = 1.5418,
) -> np.ndarray:
    qz = np.asarray(qz_inv_angstrom, dtype=float)
    k0 = 2.0 * np.pi / float(wavelength_angstrom)

    alpha = np.arcsin(np.clip(qz / (2.0 * k0), -1.0, 1.0))
    kx = k0 * np.cos(alpha)

    kz = []
    for layer in layers:
        qc = float(layer.qc_inv_angstrom)
        delta = 0.5 * (qc / (2.0 * k0)) ** 2
        n = 1.0 - delta + 1j * float(layer.beta)
        kz.append(np.sqrt((n * k0) ** 2 - kx**2 + 0j))

    r_eff = np.zeros_like(qz, dtype=np.complex128)

    for j in range(len(layers) - 2, -1, -1):
        kz_j = kz[j]
        kz_next = kz[j + 1]

        r_j = (kz_j - kz_next) / (kz_j + kz_next)

        sigma = float(layers[j + 1].roughness_angstrom)
        if sigma > 0.0:
            r_j *= np.exp(-2.0 * kz_j * kz_next * sigma**2)

        d_next = layers[j + 1].thickness_angstrom
        phase = 1.0 if d_next is None else np.exp(2j * kz_next * float(d_next))

        r_eff = (r_j + r_eff * phase) / (1.0 + r_j * r_eff * phase)

    return np.abs(r_eff) ** 2


def fresnel_reflectivity_single_interface(qz, qc):
    qz_arr = np.asarray(qz, dtype=np.complex128)
    qp = np.sqrt(qz_arr**2 - float(qc) ** 2 + 0j)
    r = (qz_arr - qp) / (qz_arr + qp)
    return np.abs(r) ** 2


def born_fresnel_asymptote(qz, qc):
    qz_arr = np.asarray(qz, dtype=float)
    return (float(qc) / (2.0 * qz_arr)) ** 4


def miceli_correction_factor(qz, qc):
    return fresnel_reflectivity_single_interface(qz, qc) / born_fresnel_asymptote(qz, qc)

