"""Compatibility wrapper around the typed simulation request API."""

from __future__ import annotations

from typing import Any

import numpy as np

from ra_sim.simulation.diffraction import (
    DEFAULT_SOLVE_Q_MODE,
    DEFAULT_SOLVE_Q_REL_TOL,
    OPTICS_MODE_FAST,
    process_peaks_parallel_safe,
)
from ra_sim.simulation.engine import simulate
from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.types import (
    BeamSamples,
    DebyeWallerParams,
    DetectorGeometry,
    MosaicParams,
    SimulationRequest,
)


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


def _read_runtime_value(value: Any) -> Any:
    """Return ``value`` or ``value.get()`` for Tk-like variables."""

    getter = getattr(value, "get", None)
    if callable(getter):
        return getter()
    return value


def _build_legacy_request(
    *,
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
    bandwidth,
    optics_mode,
    solve_q_steps,
    solve_q_rel_tol,
    solve_q_mode,
    profile_samples,
    profile_rng,
    pixel_size_m=100e-6,
    sample_width_m=0.0,
    sample_length_m=0.0,
    thickness=0.0,
    n2_sample_array=None,
) -> SimulationRequest:
    """Build one typed simulation request from the legacy positional inputs."""

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

    image_size_int = int(image_size)
    geometry = DetectorGeometry(
        image_size=image_size_int,
        av=float(av),
        cv=float(cv),
        lambda_angstrom=float(lambda_),
        distance_m=float(corto_detector_value),
        gamma_deg=float(gamma),
        Gamma_deg=float(Gamma),
        chi_deg=float(chi),
        psi_deg=float(psi),
        psi_z_deg=float(psi_z),
        zs=float(zs),
        zb=float(zb),
        center=np.asarray(center, dtype=np.float64),
        theta_initial_deg=float(theta_initial),
        cor_angle_deg=float(cor_angle),
        unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        pixel_size_m=float(pixel_size_m),
        sample_width_m=float(sample_width_m),
        sample_length_m=float(sample_length_m),
    )
    beam = BeamSamples(
        beam_x_array=np.asarray(beam_x_array, dtype=np.float64),
        beam_y_array=np.asarray(beam_y_array, dtype=np.float64),
        theta_array=np.asarray(theta_array, dtype=np.float64),
        phi_array=np.asarray(phi_array, dtype=np.float64),
        wavelength_array=np.asarray(wavelength_array, dtype=np.float64),
        n2_sample_array=(
            None
            if n2_sample_array is None
            else np.asarray(n2_sample_array, dtype=np.complex128)
        ),
    )
    mosaic = MosaicParams(
        sigma_mosaic_deg=float(_read_runtime_value(sigma_mosaic_var)),
        gamma_mosaic_deg=float(_read_runtime_value(gamma_mosaic_var)),
        eta=float(_read_runtime_value(eta_var)),
        solve_q_steps=int(solve_q_steps),
        solve_q_rel_tol=float(solve_q_rel_tol),
        solve_q_mode=int(solve_q_mode),
    )

    return SimulationRequest(
        miller=np.asarray(miller, dtype=np.float64),
        intensities=np.asarray(intensities, dtype=np.float64).reshape(-1),
        geometry=geometry,
        beam=beam,
        mosaic=mosaic,
        debye_waller=DebyeWallerParams(
            x=float(debye_x_value),
            y=float(debye_y_value),
        ),
        n2=n2,
        image_buffer=np.zeros((image_size_int, image_size_int), dtype=np.float64),
        save_flag=0,
        thickness=float(thickness),
        optics_mode=OPTICS_MODE_FAST if optics_mode is None else int(optics_mode),
        collect_hit_tables=True,
    )


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
    pixel_size_m=100e-6,
    sample_width_m=0.0,
    sample_length_m=0.0,
    thickness=0.0,
    n2_sample_array=None,
):
    """Run one legacy positional simulation through the typed engine API."""

    request = _build_legacy_request(
        theta_initial=theta_initial,
        cor_angle=cor_angle,
        gamma=gamma,
        Gamma=Gamma,
        chi=chi,
        psi_z=psi_z,
        zs=zs,
        zb=zb,
        debye_x_value=debye_x_value,
        debye_y_value=debye_y_value,
        corto_detector_value=corto_detector_value,
        miller=miller,
        intensities=intensities,
        image_size=image_size,
        av=av,
        cv=cv,
        lambda_=lambda_,
        psi=psi,
        n2=n2,
        center=center,
        num_samples=num_samples,
        divergence_sigma=divergence_sigma,
        bw_sigma=bw_sigma,
        sigma_mosaic_var=sigma_mosaic_var,
        gamma_mosaic_var=gamma_mosaic_var,
        eta_var=eta_var,
        bandwidth=bandwidth,
        optics_mode=optics_mode,
        solve_q_steps=solve_q_steps,
        solve_q_rel_tol=solve_q_rel_tol,
        solve_q_mode=solve_q_mode,
        profile_samples=profile_samples,
        profile_rng=profile_rng,
        pixel_size_m=pixel_size_m,
        sample_width_m=sample_width_m,
        sample_length_m=sample_length_m,
        thickness=thickness,
        n2_sample_array=n2_sample_array,
    )
    return simulate(request, peak_runner=process_peaks_parallel_safe).image
