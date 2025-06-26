
"""Random mosaic block profile generation for diffraction simulations."""

import numpy as np
from numba import njit, int64, float64, prange
import matplotlib.pyplot as plt


def sample_pseudo_voigt_2d(n, eta, sigma, gamma):
    """Return ``n`` samples from a simple 2D pseudo-Voigt distribution.

    Parameters
    ----------
    n : int
        Number of samples to draw.
    eta : float
        Mixing fraction between Gaussian (0) and Lorentzian (1).
    sigma : float
        Standard deviation of the Gaussian component.
    gamma : float
        Half width at half maximum of the Lorentzian component.

    Returns
    -------
    ndarray
        Array of shape ``(n, 2)`` containing the sampled points.
    """

    gauss = np.random.normal(0.0, sigma, (n, 2))
    lorentz = gamma * np.random.standard_cauchy((n, 2))
    return (1.0 - eta) * gauss + eta * lorentz


def sample_2d_gaussian(n, sigma):
    return np.random.normal(0, sigma, (n, 2))


def generate_random_profiles(num_samples, divergence_sigma, bw_sigma, lambda0, bandwidth):
    # Generate divergence profile
    Divergence_samples = sample_2d_gaussian(num_samples, divergence_sigma)
    theta_array = Divergence_samples[:, 0]
    phi_array = Divergence_samples[:, 1]

    # Generate beam profile
    Beam_samples = sample_2d_gaussian(num_samples, bw_sigma)
    beam_x_array = Beam_samples[:, 0]
    beam_y_array = Beam_samples[:, 1]

    # Sample wavelengths: assume a normal distribution with mean lambda0
    # and standard deviation lambda0 * bandwidth.
    wavelength_array = np.random.normal(lambda0, lambda0 * bandwidth, num_samples)

    return (beam_x_array, beam_y_array,
            theta_array, phi_array,
            wavelength_array)

