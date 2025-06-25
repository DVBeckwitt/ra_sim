"""High-level orchestration of full diffraction simulations."""

import numpy as np

from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction import process_peaks_parallel
from ra_sim.utils.calculations import IndexofRefraction, fresnel_transmission


def simulate_diffraction(
    theta_initial,
    gamma,
    Gamma,
    chi,
    zs,
    zb,
    debye_x_value,
    debye_y_value,
    corto_detector_value,
    miller,
    intensities,
    image_size,
    av,
    cv,
    lambda_,
    psi,
    n2,
    center,
    num_samples,
    divergence_sigma,
    bw_sigma,
    sigma_mosaic_var,
    gamma_mosaic_var,
    eta_var,
    bandwidth=0.007,
):
    """Run a standalone diffraction simulation.

    Parameters are largely mirrored from :func:`process_peaks_parallel` but the
    random beam/mosaic profiles are regenerated on each call using
    :func:`generate_random_profiles`.
    """

    center = np.asarray(center, dtype=np.float64)
    current_sigma_mosaic = np.radians(sigma_mosaic_var.get())
    current_gamma_mosaic = np.radians(gamma_mosaic_var.get())
    current_eta = eta_var.get()

    beam_x_array, beam_y_array, theta_array, phi_array, wavelength_array = generate_random_profiles(
        num_samples,
        divergence_sigma,
        bw_sigma,
        lambda_,
        bandwidth,
    )

    unit_x = np.array([1.0, 0.0, 0.0])
    n_detector = np.array([0.0, 1.0, 0.0])

    simulated_image, *_ = process_peaks_parallel(
        np.ascontiguousarray(miller, dtype=np.float64),
        np.ascontiguousarray(intensities, dtype=np.float64),
        image_size,
        av,
        cv,
        lambda_,
        np.zeros((image_size, image_size)),
        corto_detector_value,
        gamma,
        Gamma,
        chi,
        psi,
        zs,
        zb,
        n2,
        np.ascontiguousarray(beam_x_array, dtype=np.float64),
        np.ascontiguousarray(beam_y_array, dtype=np.float64),
        np.ascontiguousarray(theta_array, dtype=np.float64),
        np.ascontiguousarray(phi_array, dtype=np.float64),
        np.degrees(current_sigma_mosaic),
        np.degrees(current_gamma_mosaic),
        current_eta,
        np.ascontiguousarray(wavelength_array, dtype=np.float64),
        debye_x_value,
        debye_y_value,
        center,
        theta_initial,
        unit_x,
        n_detector,
        save_flag=0,
    )

    return simulated_image