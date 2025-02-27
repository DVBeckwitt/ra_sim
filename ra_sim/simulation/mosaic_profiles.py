
import numpy as np
from numba import njit, int64, float64, prange
import matplotlib.pyplot as plt


def sample_2d_gaussian(n, sigma):
    return np.random.normal(0, sigma, (n, 2))

def sample_2d_cauchy(n, gamma):
    # For a 1D Cauchy, x = gamma * tan(pi*(u-0.5)), u in (0,1).
    # For 2D, we can sample each dimension independently:
    u1 = np.random.rand(n)
    u2 = np.random.rand(n)
    x = gamma * np.tan(np.pi*(u1 - 0.5))
    y = gamma * np.tan(np.pi*(u2 - 0.5))
    return np.column_stack((x, y))

def sample_pseudo_voigt_2d(n, eta, sigma, gamma):
    u = np.random.rand(n)
    gaussian_indices = (u >= eta)
    cauchy_indices = (u < eta)

    samples = np.zeros((n, 2))
    # Gaussian samples
    g_count = np.sum(gaussian_indices)
    if g_count > 0:
        samples[gaussian_indices,:] = sample_2d_gaussian(g_count, sigma)

    # Cauchy samples
    c_count = np.sum(cauchy_indices)
    if c_count > 0:
        samples[cauchy_indices,:] = sample_2d_cauchy(c_count, gamma)

    return samples

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

