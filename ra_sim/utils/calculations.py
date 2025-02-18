import numpy as np
import pyFAI
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
import json
import matplotlib.pyplot as plt
# import njit
from numba import njit


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

    r_e = 2.8179403267e-15
    lambda_ = 1.54e-10  # 1.54 Å in meters
    
    rho_Bi = 9.807       # g/cm^3
    rho_Se = 4.81        # g/cm^3
    mu_Bi  = 237.8* rho_Bi  # (cm^2/g) * (g/cm^3) = cm^-1
    mu_Se  = 81.16 * rho_Se  # cm^-1

    # Molar masses
    M_Bi = 208.98
    M_Se = 78.96
    # Bi2Se3 formula mass
    M_Bi2Se3 = 2.0*M_Bi + 3.0*M_Se  # g/mol

    # Weighted linear attenuation
    mu_Bi2Se3 = (
        (2.0*M_Bi)*mu_Bi + (3.0*M_Se)*mu_Se
    ) / M_Bi2Se3  # still in cm^-1

    # Convert cm^-1 to m^-1
    mu_Bi2Se3_m = mu_Bi2Se3 * 1.0e2

    # Density in g/cm^3 => convert to g/m^3
    rho_g_m3 = 6.82e6   # was 6.82 g/cm^3 => 6.82e6 g/m^3

    # Avogadro's number
    N_A = 6.022e23

    # Volume (m^3) per mole
    V_mol = M_Bi2Se3 / rho_g_m3  # g/mol / (g/m^3) = m^3/mol

    # *** Multiply by total electrons (Z_total = 268) per formula ***
    Z_total = 2*83 + 3*34  # 166 + 102 = 268
    rho_e =  Z_total * (N_A / V_mol)  # e/m^3

    # Compute delta, beta
    delta, beta = IoR(lambda_, rho_e, r_e, mu_Bi2Se3_m)

    # Refractive index
    n = 1.0 - delta + 1.0j*beta
    return n
import math
import numba

@numba.njit
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

@numba.njit
def fresnel_transmission(alpha, n, pol_s=True):
    """
    Fresnel transmission amplitude for a grazing-incidence wave
    from vacuum (n0=1) into a medium with complex index n,
    where alpha is the angle from the SAMPLE SURFACE (not the normal).

    Args:
        alpha : float
            Grazing angle in radians from the surface plane
            (small alpha => beam nearly parallel to surface).
        n : complex
            Complex refractive index (1 - delta + i*beta).
        pol_s : bool
            True => s-polarization (TE). False => p-polarization (TM).

    Returns:
        t_amp : complex
            The complex Fresnel transmission amplitude.
            (For intensity transmission, use abs(t_amp)**2)
    """
    # Normal component in vacuum is sin(alpha), since alpha is from the surface.
    k_z0 = math.sin(alpha) + 0j  # make it complex

    # k_z1 = sqrt(n^2 - cos^2(alpha)) in the medium
    cos_a = math.cos(alpha)
    z = n*n - cos_a*cos_a
    k_z1 = complex_sqrt(z)

    if pol_s:
        # s-polarization:
        # t_s = 2 k_z0 / (k_z0 + k_z1)
        numerator = 2.0 * k_z0
        denominator = k_z0 + k_z1
    else:
        # p-polarization:
        # t_p = 2 n^2 k_z0 / (n^2 k_z0 + k_z1)
        numerator = 2.0 * n*n * k_z0
        denominator = n*n * k_z0 + k_z1

    if abs(denominator) < 1e-30:
        return 0j
    
    return numerator / denominator
