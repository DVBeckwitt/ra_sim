"""High-level orchestration of full diffraction simulations."""

import numpy as np

from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction import (
    DEFAULT_SOLVE_Q_MODE,
    DEFAULT_SOLVE_Q_REL_TOL,
    OPTICS_MODE_FAST,
    process_peaks_parallel_safe,
)
from ra_sim.utils.calculations import IndexofRefraction, fresnel_transmission


def _resolve_profile_samples(profile_samples):
    """Normalize optional precomputed beam samples into five aligned arrays."""

    if profile_samples is None:
        return None

    if isinstance(profile_samples, dict):
        wavelength = profile_samples.get("wavelength_array")
        if wavelength is None:
            wavelength = profile_samples.get("wavelength_i_array")
        arrays = (
            profile_samples.get("beam_x_array"),
            profile_samples.get("beam_y_array"),
            profile_samples.get("theta_array"),
            profile_samples.get("phi_array"),
            wavelength,
        )
    else:
        arrays = tuple(profile_samples)
        if len(arrays) != 5:
            raise ValueError("profile_samples must provide five arrays")

    resolved = tuple(np.asarray(arr, dtype=np.float64).reshape(-1) for arr in arrays)
    sample_count = int(resolved[0].size) if resolved else 0
    if sample_count <= 0:
        raise ValueError("profile_samples must not be empty")
    if any(arr.size != sample_count for arr in resolved[1:]):
        raise ValueError("profile_samples arrays must all have the same length")
    return resolved


def simulate_diffraction(
    theta_initial,
    cor_angle,
    gamma,
    Gamma,
    chi,
    psi_z,
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
    optics_mode=OPTICS_MODE_FAST,
    solve_q_steps=1000,
    solve_q_rel_tol=DEFAULT_SOLVE_Q_REL_TOL,
    solve_q_mode=DEFAULT_SOLVE_Q_MODE,
    profile_samples=None,
    profile_rng=None,
):
    """Run a standalone diffraction simulation.

    Parameters are largely mirrored from :func:`process_peaks_parallel` but the
    random beam/mosaic profiles are regenerated on each call using
    :func:`generate_random_profiles` unless ``profile_samples`` is supplied.
    """

    current_sigma_mosaic = np.radians(sigma_mosaic_var.get())
    current_gamma_mosaic = np.radians(gamma_mosaic_var.get())
    current_eta = eta_var.get()

    resolved_profile_samples = _resolve_profile_samples(profile_samples)
    if resolved_profile_samples is None:
        beam_x_array, beam_y_array, theta_array, phi_array, wavelength_array = (
            generate_random_profiles(
                num_samples,
                divergence_sigma,
                bw_sigma,
                lambda_,
                bandwidth,
                rng=profile_rng,
            )
        )
    else:
        beam_x_array, beam_y_array, theta_array, phi_array, wavelength_array = (
            resolved_profile_samples
        )

    unit_x = np.array([1.0, 0.0, 0.0])
    n_detector = np.array([0.0, 1.0, 0.0])

    simulated_image, *_ = process_peaks_parallel_safe(
        miller,
        intensities,
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
        psi_z,
        zs,
        zb,
        n2,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        np.degrees(current_sigma_mosaic),
        np.degrees(current_gamma_mosaic),
        current_eta,
        wavelength_array,
        debye_x_value,
        debye_y_value,
        center,
        theta_initial,
        cor_angle,
        unit_x,
        n_detector,
        save_flag=0,
        optics_mode=optics_mode,
        solve_q_steps=solve_q_steps,
        solve_q_rel_tol=solve_q_rel_tol,
        solve_q_mode=solve_q_mode,
    )

    return simulated_image
