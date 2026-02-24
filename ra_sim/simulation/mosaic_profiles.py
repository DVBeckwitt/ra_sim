
"""Random mosaic block profile generation for diffraction simulations."""

from __future__ import annotations

from typing import Any

import numpy as np


def _normal(rng: Any, loc: float, scale: float, size):
    if rng is None:
        return np.random.normal(loc, scale, size)
    return rng.normal(loc, scale, size)


def sample_2d_gaussian(n, sigma, rng=None):
    return _normal(rng, 0.0, sigma, (n, 2))


def generate_random_profiles(
    num_samples,
    divergence_sigma,
    bw_sigma,
    lambda0,
    bandwidth,
    rng=None,
):
    # Generate divergence profile
    Divergence_samples = sample_2d_gaussian(num_samples, divergence_sigma, rng=rng)
    theta_array = Divergence_samples[:, 0]
    phi_array = Divergence_samples[:, 1]

    # Generate beam profile
    Beam_samples = sample_2d_gaussian(num_samples, bw_sigma, rng=rng)
    beam_x_array = Beam_samples[:, 0]
    beam_y_array = Beam_samples[:, 1]

    # Sample wavelengths: assume a normal distribution with mean lambda0
    # and standard deviation lambda0 * bandwidth.
    wavelength_array = _normal(rng, lambda0, lambda0 * bandwidth, num_samples)

    return (beam_x_array, beam_y_array,
            theta_array, phi_array,
            wavelength_array)

