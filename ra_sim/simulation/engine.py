"""Typed simulation entrypoints built on top of legacy diffraction kernels."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .diffraction import process_peaks_parallel_safe, process_qr_rods_parallel_safe
from .types import SimulationRequest, SimulationResult


PeakRunner = Callable[..., tuple[Any, Any, Any, Any, Any, Any]]
RodRunner = Callable[..., tuple[Any, Any, Any, Any, Any, Any, Any]]


def _default_image_buffer(request: SimulationRequest) -> np.ndarray:
    if request.image_buffer is not None:
        return request.image_buffer
    size = int(request.geometry.image_size)
    return np.zeros((size, size), dtype=np.float64)


def simulate(
    request: SimulationRequest,
    *,
    peak_runner: PeakRunner = process_peaks_parallel_safe,
) -> SimulationResult:
    """Run a diffraction simulation from a typed request."""

    image_buffer = _default_image_buffer(request)

    image, hit_tables, q_data, q_count, all_status, miss_tables = peak_runner(
        request.miller,
        request.intensities,
        request.geometry.image_size,
        request.geometry.av,
        request.geometry.cv,
        request.geometry.lambda_angstrom,
        image_buffer,
        request.geometry.distance_m,
        request.geometry.gamma_deg,
        request.geometry.Gamma_deg,
        request.geometry.chi_deg,
        request.geometry.psi_deg,
        request.geometry.psi_z_deg,
        request.geometry.zs,
        request.geometry.zb,
        request.n2,
        request.beam.beam_x_array,
        request.beam.beam_y_array,
        request.beam.theta_array,
        request.beam.phi_array,
        request.mosaic.sigma_mosaic_deg,
        request.mosaic.gamma_mosaic_deg,
        request.mosaic.eta,
        request.beam.wavelength_array,
        request.debye_waller.x,
        request.debye_waller.y,
        request.geometry.center,
        request.geometry.theta_initial_deg,
        request.geometry.cor_angle_deg,
        request.geometry.unit_x,
        request.geometry.n_detector,
        save_flag=request.save_flag,
        record_status=request.record_status,
        thickness=request.thickness,
    )

    return SimulationResult(
        image=np.asarray(image, dtype=np.float64),
        hit_tables=hit_tables,
        q_data=q_data,
        q_count=q_count,
        all_status=all_status,
        miss_tables=miss_tables,
        degeneracy=None,
    )


def simulate_qr_rods(
    qr_dict: dict,
    request: SimulationRequest,
    *,
    peak_runner: RodRunner = process_qr_rods_parallel_safe,
) -> SimulationResult:
    """Run rod-based simulation through the typed request API."""

    image_buffer = _default_image_buffer(request)

    image, hit_tables, q_data, q_count, all_status, miss_tables, degeneracy = peak_runner(
        qr_dict,
        request.geometry.image_size,
        request.geometry.av,
        request.geometry.cv,
        request.geometry.lambda_angstrom,
        image_buffer,
        request.geometry.distance_m,
        request.geometry.gamma_deg,
        request.geometry.Gamma_deg,
        request.geometry.chi_deg,
        request.geometry.psi_deg,
        request.geometry.psi_z_deg,
        request.geometry.zs,
        request.geometry.zb,
        request.n2,
        request.beam.beam_x_array,
        request.beam.beam_y_array,
        request.beam.theta_array,
        request.beam.phi_array,
        request.mosaic.sigma_mosaic_deg,
        request.mosaic.gamma_mosaic_deg,
        request.mosaic.eta,
        request.beam.wavelength_array,
        request.debye_waller.x,
        request.debye_waller.y,
        request.geometry.center,
        request.geometry.theta_initial_deg,
        request.geometry.cor_angle_deg,
        request.geometry.unit_x,
        request.geometry.n_detector,
        save_flag=request.save_flag,
        record_status=request.record_status,
        thickness=request.thickness,
    )

    return SimulationResult(
        image=np.asarray(image, dtype=np.float64),
        hit_tables=hit_tables,
        q_data=q_data,
        q_count=q_count,
        all_status=all_status,
        miss_tables=miss_tables,
        degeneracy=degeneracy,
    )
