"""Numerical helper functions used by the simulator."""

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
try:
    import pyFAI
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
except Exception:  # pragma: no cover - optional
    pyFAI = None
    class AzimuthalIntegrator:
        pass
import json
import matplotlib.pyplot as plt
from numba import njit
import math
import numba

from ra_sim.config import get_material_config

try:  # pragma: no cover - optional dependency used at runtime
    import Dans_Diffraction as dif
except Exception:  # pragma: no cover - optional dependency
    dif = None

try:  # pragma: no cover - optional dependency used at runtime
    import xraydb
except Exception:  # pragma: no cover - optional dependency
    xraydb = None

# ``materials.yaml`` is discovered through :mod:`ra_sim.config` so that the
# same configuration mechanism can be used by both the GUI and command line
# tooling.
_MATERIAL_CFG = get_material_config()
_MATERIAL_CONSTANTS = _MATERIAL_CFG.get("constants", {})
_MATERIAL_DATA = _MATERIAL_CFG["material"]

R_E = float(_MATERIAL_CONSTANTS["classical_electron_radius"])
LAMBDA_DEFAULT = float(_MATERIAL_CONSTANTS["default_wavelength"])
N_A = float(_MATERIAL_CONSTANTS["avogadro_number"])

_COMPOSITION = _MATERIAL_DATA.get("composition", {})
_ELEMENTS = _MATERIAL_DATA.get("elements", {})
_DENSITY = float(_MATERIAL_DATA["density"])

_ELEMENT_KEYS = tuple(_COMPOSITION.keys())
_STOICHIOMETRY = np.array([float(_COMPOSITION[key]) for key in _ELEMENT_KEYS], dtype=np.float64)
_ATOMIC_MASSES = np.array([float(_ELEMENTS[key]["atomic_mass"]) for key in _ELEMENT_KEYS], dtype=np.float64)
_MASS_ATTENUATION = np.array([float(_ELEMENTS[key]["mass_attenuation_coefficient"]) for key in _ELEMENT_KEYS], dtype=np.float64)
_ATOMIC_NUMBERS = np.array([float(_ELEMENTS[key]["atomic_number"]) for key in _ELEMENT_KEYS], dtype=np.float64)

_FORMULA_MASS = float(np.dot(_STOICHIOMETRY, _ATOMIC_MASSES))
_MASS_FRACTIONS = (_STOICHIOMETRY * _ATOMIC_MASSES) / _FORMULA_MASS
_TOTAL_ATOMIC_NUMBER = float(np.dot(_STOICHIOMETRY, _ATOMIC_NUMBERS))
_WEIGHTED_MASS_ATTENUATION = float(np.dot(_MASS_FRACTIONS, _MASS_ATTENUATION))
_HC_ANGSTROM_EV = 12398.419843320026
_ELEMENT_SYMBOL_RE = re.compile(r"[A-Z][a-z]?")


# Function to calculate d-spacing for hexagonal crystals
def d_spacing(h, k, l, av, cv):
    if (h, k, l) == (0, 0, 0):
        return None
    term1 = 4 / 3 * (h**2 + h * k + k**2) / av**2
    term2 = (l**2) / cv**2
    return 1 / np.sqrt(term1 + term2)

# Function to calculate 2theta using Bragg's law
def two_theta(d, wavelength):
    if d is None:
        return None
    sin_theta = wavelength / (2 * d)
    if sin_theta > 1:  # This means the reflection is not physically possible
        return None
    theta = np.arcsin(sin_theta)
    return 2 * np.degrees(theta)


SOURCE_BRANCH_PHI_ZERO_DEADBAND_DEG = 1.0e-3


def normalize_signed_phi_deg(phi_deg):
    """Wrap one azimuth into the stable interval ``(-180, 180]`` degrees."""

    try:
        value = float(phi_deg)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    wrapped = float(((value + 180.0) % 360.0) - 180.0)
    if wrapped <= -180.0 + 1.0e-12:
        wrapped = 180.0
    return float(wrapped)


def source_branch_index_from_phi_deg(
    phi_deg,
    *,
    zero_deadband_deg: float = SOURCE_BRANCH_PHI_ZERO_DEADBAND_DEG,
):
    """Return stable detector-side branch label ``0``/``1`` from signed azimuth."""

    wrapped = normalize_signed_phi_deg(phi_deg)
    if wrapped is None:
        return None
    try:
        deadband = abs(float(zero_deadband_deg))
    except (TypeError, ValueError):
        deadband = float(SOURCE_BRANCH_PHI_ZERO_DEADBAND_DEG)
    if not np.isfinite(deadband):
        deadband = float(SOURCE_BRANCH_PHI_ZERO_DEADBAND_DEG)
    if abs(float(wrapped)) <= float(deadband):
        return None
    return 0 if float(wrapped) < 0.0 else 1


def resolve_canonical_branch(
    entry: Mapping[str, object] | None,
    *,
    allow_legacy_peak_fallback: bool = False,
    zero_deadband_deg: float = SOURCE_BRANCH_PHI_ZERO_DEADBAND_DEG,
) -> tuple[int | None, str | None, str | None]:
    """Resolve canonical detector-side branch identity for one entry.

    Canonical runtime behavior accepts only ``source_branch_index`` or signed-phi
    geometry hints. Legacy migration may additionally consult cached caked
    azimuth values and, as a final fallback only, ``source_peak_index`` when it
    already lives in the compact ``{0, 1}`` branch namespace.
    """

    if not isinstance(entry, Mapping):
        return None, None, None

    try:
        explicit_branch = int(entry.get("source_branch_index"))
    except Exception:
        explicit_branch = -1
    if explicit_branch in {0, 1}:
        return int(explicit_branch), "source_branch_index", None

    try:
        deadband = abs(float(zero_deadband_deg))
    except (TypeError, ValueError):
        deadband = float(SOURCE_BRANCH_PHI_ZERO_DEADBAND_DEG)
    if not np.isfinite(deadband):
        deadband = float(SOURCE_BRANCH_PHI_ZERO_DEADBAND_DEG)

    def _resolve_signed_value(
        value: object,
        *,
        source_name: str,
    ) -> tuple[int | None, str | None, str | None]:
        wrapped = normalize_signed_phi_deg(value)
        if wrapped is None:
            return None, None, None
        if abs(float(wrapped)) <= float(deadband):
            return None, str(source_name), "ambiguous_branch_deadband"
        return (
            0 if float(wrapped) < 0.0 else 1,
            str(source_name),
            None,
        )

    for key in (
        "simulated_phi_deg",
        "background_phi_deg",
        "phi_deg",
        "phi",
    ):
        branch_idx, branch_source, branch_reason = _resolve_signed_value(
            entry.get(key),
            source_name=str(key),
        )
        if branch_reason is not None:
            return None, branch_source, branch_reason
        if branch_idx in {0, 1}:
            return int(branch_idx), branch_source, None

    if allow_legacy_peak_fallback:
        for key in (
            "refined_sim_caked_y",
            "caked_y",
            "raw_caked_y",
        ):
            branch_idx, branch_source, branch_reason = _resolve_signed_value(
                entry.get(key),
                source_name=str(key),
            )
            if branch_reason is not None:
                return None, branch_source, branch_reason
            if branch_idx in {0, 1}:
                return int(branch_idx), branch_source, None

        try:
            legacy_branch = int(entry.get("source_peak_index"))
        except Exception:
            legacy_branch = -1
        if legacy_branch in {0, 1}:
            return int(legacy_branch), "source_peak_index", None

    return None, None, None

@njit
def IoR(lambda_, rho_e, r, mu):
    # delta = (lambda^2 * rho_e * r_e)/(2 pi)
    delta = (lambda_**2 * rho_e * r) / (2.0 * np.pi)
    # beta  = mu * lambda / (4 pi)
    beta  = (mu * lambda_) / (4.0 * np.pi)
    return delta, beta

@njit
def IndexofRefraction(lambda_m=LAMBDA_DEFAULT):
    """Return the complex X-ray index of refraction for the active material.

    Parameters
    ----------
    lambda_m:
        Wavelength in meters. Defaults to the configured reference wavelength.
    """

    if not np.isfinite(lambda_m) or lambda_m <= 0.0:
        lambda_m = LAMBDA_DEFAULT

    # Linear attenuation coefficient (m^-1)
    mu_cm = _DENSITY * _WEIGHTED_MASS_ATTENUATION
    mu_m = mu_cm * 1.0e2

    # Electron density (m^-3)
    rho_g_m3 = _DENSITY * 1.0e6
    V_mol = _FORMULA_MASS / rho_g_m3
    n_formulas = 1.0 / V_mol
    rho_e = _TOTAL_ATOMIC_NUMBER * (n_formulas * N_A)

    # delta and beta
    delta = (R_E * (lambda_m ** 2) * rho_e) / (2.0 * math.pi)
    beta = (mu_m * lambda_m) / (4.0 * math.pi)

    return 1.0 - delta + 1.0j * beta


def _sanitize_lambda_m(lambda_m) -> float:
    try:
        value = float(lambda_m)
    except (TypeError, ValueError):
        return LAMBDA_DEFAULT
    if not np.isfinite(value) or value <= 0.0:
        return LAMBDA_DEFAULT
    return value


def _sanitize_lambda_array(lambda_m_array) -> np.ndarray:
    arr = np.asarray(lambda_m_array, dtype=np.float64)
    out = np.array(arr, copy=True, dtype=np.float64)
    invalid = (~np.isfinite(out)) | (out <= 0.0)
    if np.any(invalid):
        out[invalid] = LAMBDA_DEFAULT
    return out


def _normalize_element_symbol(raw_value) -> str | None:
    text = str(raw_value).strip()
    if not text:
        return None
    match = _ELEMENT_SYMBOL_RE.search(text)
    if match is None:
        return None
    return match.group(0)


def _complex_index_from_density(lambda_m, electron_density_m3, mu_m):
    lambda_arr = _sanitize_lambda_array(lambda_m)
    delta = (R_E * np.square(lambda_arr) * float(electron_density_m3)) / (2.0 * np.pi)
    beta = (np.asarray(mu_m, dtype=np.float64) * lambda_arr) / (4.0 * np.pi)
    return (1.0 - delta).astype(np.complex128) + 1.0j * beta


@lru_cache(maxsize=32)
def _material_optics_properties(material: str | None = None) -> dict[str, object]:
    cfg = get_material_config(material)
    material_block = cfg["material"]
    composition = material_block.get("composition", {})
    elements = material_block.get("elements", {})
    element_keys = tuple(str(key) for key in composition.keys())
    stoichiometry = np.array(
        [float(composition[key]) for key in element_keys],
        dtype=np.float64,
    )
    atomic_masses = np.array(
        [float(elements[key]["atomic_mass"]) for key in element_keys],
        dtype=np.float64,
    )
    mass_attenuation = np.array(
        [float(elements[key]["mass_attenuation_coefficient"]) for key in element_keys],
        dtype=np.float64,
    )
    atomic_numbers = np.array(
        [float(elements[key]["atomic_number"]) for key in element_keys],
        dtype=np.float64,
    )

    formula_mass = float(np.dot(stoichiometry, atomic_masses))
    if formula_mass <= 0.0 or not np.isfinite(formula_mass):
        raise ValueError(f"Configured material {cfg['name']!r} has invalid formula mass.")

    density_g_cm3 = float(material_block["density"])
    if density_g_cm3 <= 0.0 or not np.isfinite(density_g_cm3):
        raise ValueError(f"Configured material {cfg['name']!r} has invalid density.")

    mass_fractions = (stoichiometry * atomic_masses) / formula_mass
    total_atomic_number = float(np.dot(stoichiometry, atomic_numbers))
    rho_g_m3 = density_g_cm3 * 1.0e6
    molar_volume_m3 = formula_mass / rho_g_m3
    formulas_per_m3 = N_A / molar_volume_m3
    electron_density_m3 = total_atomic_number * formulas_per_m3

    return {
        "name": str(cfg["name"]),
        "density_g_cm3": density_g_cm3,
        "electron_density_m3": float(electron_density_m3),
        "mass_fractions": mass_fractions,
        "mass_attenuation_cm2_g": mass_attenuation,
    }


@lru_cache(maxsize=32)
def _cif_optics_properties(cif_path: str) -> dict[str, object]:
    if dif is None or xraydb is None:
        raise RuntimeError("CIF-based optics require Dans_Diffraction and xraydb.")

    resolved_path = str(Path(str(cif_path)).expanduser().resolve())
    xtl = dif.Crystal(resolved_path)
    xtl.Symmetry.generate_matrices()
    xtl.generate_structure()
    structure = xtl.Structure

    raw_types = np.asarray(getattr(structure, "type", []), dtype=object).reshape(-1)
    if raw_types.size <= 0:
        raise ValueError(f"Could not determine atom types from CIF {resolved_path!r}.")

    occupancies = np.asarray(
        getattr(structure, "occupancy", np.ones(raw_types.shape[0], dtype=np.float64)),
        dtype=np.float64,
    ).reshape(-1)
    if occupancies.size != raw_types.size:
        occupancies = np.ones(raw_types.shape[0], dtype=np.float64)

    composition_counts: dict[str, float] = {}
    for raw_type, occ in zip(raw_types, occupancies):
        symbol = _normalize_element_symbol(raw_type)
        if symbol is None:
            continue
        occ_value = float(occ)
        if not np.isfinite(occ_value) or occ_value <= 0.0:
            continue
        composition_counts[symbol] = composition_counts.get(symbol, 0.0) + occ_value

    if not composition_counts:
        raise ValueError(f"Could not derive unit-cell composition from CIF {resolved_path!r}.")

    element_symbols = tuple(sorted(composition_counts))
    counts = np.array([composition_counts[sym] for sym in element_symbols], dtype=np.float64)
    atomic_masses = np.array(
        [float(xraydb.atomic_mass(sym)) for sym in element_symbols],
        dtype=np.float64,
    )
    atomic_numbers = np.array(
        [float(xraydb.atomic_number(sym)) for sym in element_symbols],
        dtype=np.float64,
    )

    cell_volume_ang3 = float(xtl.Cell.volume())
    if not np.isfinite(cell_volume_ang3) or cell_volume_ang3 <= 0.0:
        raise ValueError(f"CIF {resolved_path!r} has invalid unit-cell volume.")

    cell_molar_mass_g = float(np.dot(counts, atomic_masses))
    if not np.isfinite(cell_molar_mass_g) or cell_molar_mass_g <= 0.0:
        raise ValueError(f"CIF {resolved_path!r} has invalid unit-cell mass.")

    mass_fractions = (counts * atomic_masses) / cell_molar_mass_g
    density_g_cm3 = (cell_molar_mass_g / N_A) / (cell_volume_ang3 * 1.0e-24)
    electron_density_m3 = float(np.dot(counts, atomic_numbers)) / (cell_volume_ang3 * 1.0e-30)

    return {
        "path": resolved_path,
        "density_g_cm3": float(density_g_cm3),
        "electron_density_m3": float(electron_density_m3),
        "element_symbols": element_symbols,
        "mass_fractions": mass_fractions,
    }


def _weighted_mass_attenuation_from_cif(props: dict[str, object], energy_kev) -> np.ndarray:
    if xraydb is None:
        raise RuntimeError("CIF-based optics require xraydb.")
    weighted = np.zeros_like(np.asarray(energy_kev, dtype=np.float64), dtype=np.float64)
    symbols = tuple(props["element_symbols"])
    fractions = np.asarray(props["mass_fractions"], dtype=np.float64)
    for symbol, fraction in zip(symbols, fractions):
        weighted += float(fraction) * np.asarray(
            xraydb.mu_elam(str(symbol), energy_kev),
            dtype=np.float64,
        )
    return weighted


def _index_of_refraction_array_from_material_props(
    lambda_m_array,
    props: dict[str, object],
) -> np.ndarray:
    lambda_arr = _sanitize_lambda_array(lambda_m_array)
    density_g_cm3 = float(props["density_g_cm3"])
    weighted_mass_attn = float(
        np.dot(
            np.asarray(props["mass_fractions"], dtype=np.float64),
            np.asarray(props["mass_attenuation_cm2_g"], dtype=np.float64),
        )
    )
    mu_m = density_g_cm3 * weighted_mass_attn * 1.0e2
    return _complex_index_from_density(
        lambda_arr,
        float(props["electron_density_m3"]),
        mu_m,
    )


def _index_of_refraction_array_from_cif_props(
    lambda_m_array,
    props: dict[str, object],
) -> np.ndarray:
    lambda_arr = _sanitize_lambda_array(lambda_m_array)
    energy_ev = _HC_ANGSTROM_EV / (lambda_arr * 1.0e10)
    weighted_mass_attn = _weighted_mass_attenuation_from_cif(props, energy_ev)
    mu_m = float(props["density_g_cm3"]) * weighted_mass_attn * 1.0e2
    return _complex_index_from_density(
        lambda_arr,
        float(props["electron_density_m3"]),
        mu_m,
    )


def resolve_index_of_refraction_array(
    lambda_m_array,
    *,
    cif_path: str | None = None,
    material: str | None = None,
) -> np.ndarray:
    """Return wavelength-specific complex indices for the active optics source.

    The preferred source is the provided CIF path. When CIF-based composition or
    attenuation lookup is unavailable, the configured materials table is used as
    a fallback.
    """

    if cif_path:
        try:
            return _index_of_refraction_array_from_cif_props(
                lambda_m_array,
                _cif_optics_properties(str(cif_path)),
            )
        except Exception:
            pass

    return _index_of_refraction_array_from_material_props(
        lambda_m_array,
        _material_optics_properties(material),
    )


def resolve_index_of_refraction(
    lambda_m=LAMBDA_DEFAULT,
    *,
    cif_path: str | None = None,
    material: str | None = None,
) -> complex:
    """Return the complex index of refraction for one wavelength."""

    lambda_value = _sanitize_lambda_m(lambda_m)
    return complex(
        resolve_index_of_refraction_array(
            np.array([lambda_value], dtype=np.float64),
            cif_path=cif_path,
            material=material,
        )[0]
    )

@njit
def complex_sqrt(z):
    """
    Compute the principal square root of a complex number z
    in a Numba-friendly way using polar form.
    """
    r = math.hypot(z.real, z.imag)  # sqrt(x^2 + y^2)
    phi = math.atan2(z.imag, z.real)
    sqrt_r = math.sqrt(r)
    half_phi = 0.5 * phi
    return complex(sqrt_r * math.cos(half_phi),
                   sqrt_r * math.sin(half_phi))
@njit
def fresnel_transmission(grazing_angle, refractive_index, s_polarization=True, incoming=True):
    """
    Fresnel E-field amplitude transmission for vacuum <-> medium.
    grazing_angle: radians from the surface (0 parallel).
    refractive_index: complex n = 1 - delta + 1j*beta.
    incoming=True means vacuum -> medium, False means medium -> vacuum.
    For energy flux, use T = (kz1.real / kz0.real) * abs(t)**2.
    """
    n = refractive_index
    kz0 = math.sin(grazing_angle) + 0j
    c = math.cos(grazing_angle)
    kz1 = complex_sqrt(n*n - c*c)  # principal branch

    if s_polarization:
        num = 2.0 * (kz0 if incoming else kz1)
        den = kz0 + kz1
    else:
        if incoming:  # vacuum -> medium
            num = 2.0 * n * kz0
            den = n*n * kz0 + kz1
        else:         # medium -> vacuum
            num = 2.0 * n * kz1
            den = kz1 + n*n * kz0

    if abs(den) < 1e-30:
        return 0j
    return num / den
