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
from pathlib import Path
import yaml
from numba import njit
import math
import numba

# ``ior_params.yaml`` lives at the repository root. ``calculations.py`` sits
# inside ``ra_sim/utils`` so we need to go two directories up to reach the
# repository root before appending the file name.  Using ``parents[2]`` keeps
# the path correct even if ``ra_sim`` is installed as a package.
_IOR_YAML = Path(__file__).resolve().parents[2] / "ior_params.yaml"
with open(_IOR_YAML, "r", encoding="utf-8") as fh:
    _IOR_PARAMS = yaml.safe_load(fh)

R_E = float(_IOR_PARAMS["classical_electron_radius"])
LAMBDA_DEFAULT = float(_IOR_PARAMS["default_wavelength"])
N_A = float(_IOR_PARAMS["avogadro_number"])
M_BI = float(_IOR_PARAMS["atomic_masses"]["Bi"])
M_SE = float(_IOR_PARAMS["atomic_masses"]["Se"])
MU_MASS_BI = float(_IOR_PARAMS["mass_attenuation_coefficients"]["Bi"])
MU_MASS_SE = float(_IOR_PARAMS["mass_attenuation_coefficients"]["Se"])
RHO_BI2SE3 = float(_IOR_PARAMS["compound_density"])
Z_BI = int(_IOR_PARAMS["atomic_numbers"]["Bi"])
Z_SE = int(_IOR_PARAMS["atomic_numbers"]["Se"])


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
    """Return the complex X-ray index of refraction for Bi2Se3."""
    # Formula mass of Bi2Se3 (g/mol)
    M_Bi2Se3 = 2.0 * M_BI + 3.0 * M_SE

    # Mass fractions of Bi and Se in Bi2Se3
    w_Bi = (2.0 * M_BI) / M_Bi2Se3
    w_Se = (3.0 * M_SE) / M_Bi2Se3

    # Linear attenuation coefficient (m^-1)
    mu_Bi2Se3_cm = RHO_BI2SE3 * (w_Bi * MU_MASS_BI + w_Se * MU_MASS_SE)
    mu_Bi2Se3 = mu_Bi2Se3_cm * 1.0e2

    # Electron density (m^-3)
    Z_total = 2.0 * Z_BI + 3.0 * Z_SE
    rho_g_m3 = RHO_BI2SE3 * 1.0e6
    V_mol = M_Bi2Se3 / rho_g_m3
    n_formulas = 1.0 / V_mol
    rho_e = Z_total * (n_formulas * N_A)

    # delta and beta
    delta = (R_E * LAMBDA_DEFAULT ** 2 * rho_e) / (2.0 * math.pi)
    beta = (mu_Bi2Se3 * LAMBDA_DEFAULT) / (4.0 * math.pi)

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
