import numpy as np
import pyFAI
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
import json
import matplotlib.pyplot as plt
# import njit
from numba import njit
import math
import numba


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
    """
    Computes the X-ray index of refraction n = 1 - delta + i beta
    for Bi2Se3 at a given wavelength using the correct mixture rule.
    """
    # Classical electron radius
    r_e = 2.8179403267e-15  # m

    # Wavelength in meters (example: Cu K-alpha ~1.54 Å)
    lambda_ = 1.54e-10

    # Avogadro's number
    N_A = 6.022e23

    # Atomic masses (g/mol)
    M_Bi = 208.98
    M_Se = 78.96

    # Formula mass of Bi2Se3 (g/mol)
    M_Bi2Se3 = 2.0 * M_Bi + 3.0 * M_Se

    # Mass fractions of Bi and Se in Bi2Se3
    w_Bi = (2.0 * M_Bi) / M_Bi2Se3
    w_Se = (3.0 * M_Se) / M_Bi2Se3

    # >>> 1) Mass attenuation coefficients (MAC) in cm^2/g at this wavelength

    mu_mass_Bi = 237.8   # cm^2/g
    mu_mass_Se = 81.16   # cm^2/g

    # >>> 2) Compound density in g/cm^3 (approx. 6.82 for Bi2Se3)
    rho_Bi2Se3 = 6.82    # g/cm^3

    # >>> 3) Compute the linear attenuation coefficient for Bi2Se3 in cm^-1
    #        mixture rule: mu_compound = rho * [ w_Bi * mu_mass_Bi + w_Se * mu_mass_Se ]
    mu_Bi2Se3_cm = rho_Bi2Se3 * (w_Bi * mu_mass_Bi + w_Se * mu_mass_Se)

    # Convert to m^-1
    mu_Bi2Se3 = mu_Bi2Se3_cm * 1.0e2

    # >>> 4) Compute electron density rho_e for Bi2Se3
    #        Number of electrons: Z_Bi = 83, Z_Se = 34
    #        Z_total = 2*83 + 3*34 = 268 e/formula
    Z_Bi = 83
    Z_Se = 34
    Z_total = 2.0 * Z_Bi + 3.0 * Z_Se  # 268

    # Convert compound density from g/cm^3 to g/m^3
    rho_g_m3 = rho_Bi2Se3 * 1.0e6

    # Molar volume in m^3/mol for Bi2Se3
    V_mol = M_Bi2Se3 / rho_g_m3

    # Number of formula units per m^3
    n_formulas = 1.0 / V_mol  # mol/m^3

    # Number of electrons per m^3
    rho_e = Z_total * (n_formulas * N_A)

    # >>> 5) Compute delta and beta
    # delta = (r_e * lambda^2 * rho_e)/(2*pi)
    # beta  = (mu * lambda)/(4*pi)
    delta = (r_e * lambda_**2 * rho_e) / (2.0 * math.pi)
    beta  = (mu_Bi2Se3 * lambda_) / (4.0 * math.pi)

    # >>> 6) Complex refractive index
    n = 1.0 - delta + 1.0j * beta
    return n

import math
import numba
from numba import njit

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
def fresnel_transmission(grazing_angle, refractive_index, s_polarization=True, direction="in"):
    """
    Calculate the Fresnel amplitude transmission coefficient for an interface
    between vacuum (n0 = 1) and a medium (complex index 'refractive_index').
    
    The angle 'grazing_angle' is measured from the sample surface 
    (0 => beam parallel to the surface). The parameter 'direction' can be:
      - "in"  : vacuum -> medium  (t_i)
      - "out" : medium -> vacuum  (t_f)
      
    Args:
        grazing_angle (float): Grazing angle (radians) from the surface plane.
        refractive_index (complex): Medium’s complex refractive index (1 - delta + i*beta).
        s_polarization (bool): True => s-polarization (TE), False => p-polarization (TM).
        direction (str): "in" or "out" for incoming or outgoing transmission.
        
    Returns:
        complex: The complex amplitude transmission coefficient.
                 Use abs(...)**2 to get intensity transmission.
    """
    # Validate direction
    direction = direction.lower()
    if direction not in ("in", "out"):
        raise ValueError("direction must be 'in' or 'out'.")

    # k_z0 = normal component in vacuum = sin(grazing_angle)
    k_z0 = math.sin(grazing_angle) + 0j
    # k_z1 = normal component in medium = sqrt(n^2 - cos^2(grazing_angle))
    cos_angle = math.cos(grazing_angle)
    k_z1 = complex_sqrt(refractive_index * refractive_index - cos_angle * cos_angle)

    if direction == "in":
        # Vacuum -> Medium
        if s_polarization:
            # s-pol: t_s = 2 k_z0 / (k_z0 + k_z1)
            numerator = 2.0 * k_z0
            denominator = k_z0 + k_z1
        else:
            # p-pol: t_p = 2 n^2 k_z0 / (n^2 k_z0 + k_z1)
            numerator = 2.0 * refractive_index * refractive_index * k_z0
            denominator = refractive_index * refractive_index * k_z0 + k_z1
    else:
        # direction == "out" => Medium -> Vacuum
        if s_polarization:
            # s-pol: t_s = 2 k_z1 / (k_z1 + k_z0)
            numerator = 2.0 * k_z1
            denominator = k_z1 + k_z0
        else:
            # p-pol: t_p = 2 n^2 k_z1 / (n^2 k_z1 + k_z0)
            numerator = 2.0 * refractive_index * refractive_index * k_z1
            denominator = refractive_index * refractive_index * k_z1 + k_z0

    # Avoid division by near-zero
    if abs(denominator) < 1e-30:
        return 0j

    return numerator / denominator
