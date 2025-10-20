"""Numerical helper functions used by the simulator."""

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

from ra_sim.path_config import get_material_config

# ``materials.yaml`` is discovered through :mod:`ra_sim.path_config` so that the
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
@njit
def IoR(lambda_, rho_e, r, mu):
    # delta = (lambda^2 * rho_e * r_e)/(2 pi)
    delta = (lambda_**2 * rho_e * r) / (2.0 * np.pi)
    # beta  = mu * lambda / (4 pi)
    beta  = (mu * lambda_) / (4.0 * np.pi)
    return delta, beta

@njit
def IndexofRefraction():
    """Return the complex X-ray index of refraction for the active material."""

    # Linear attenuation coefficient (m^-1)
    mu_cm = _DENSITY * _WEIGHTED_MASS_ATTENUATION
    mu_m = mu_cm * 1.0e2

    # Electron density (m^-3)
    rho_g_m3 = _DENSITY * 1.0e6
    V_mol = _FORMULA_MASS / rho_g_m3
    n_formulas = 1.0 / V_mol
    rho_e = _TOTAL_ATOMIC_NUMBER * (n_formulas * N_A)

    # delta and beta
    delta = (R_E * (LAMBDA_DEFAULT ** 2) * rho_e) / (2.0 * math.pi)
    beta = (mu_m * LAMBDA_DEFAULT) / (4.0 * math.pi)

    return 1.0 - delta + 1.0j * beta

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
