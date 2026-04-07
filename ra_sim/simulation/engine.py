"""Typed simulation entrypoints built on top of legacy diffraction kernels."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .diffraction import (
    build_intersection_cache,
    process_peaks_parallel_safe,
    process_qr_rods_parallel_safe,
)
from .types import SimulationRequest, SimulationResult


PeakRunner = Callable[..., tuple[Any, Any, Any, Any, Any, Any]]
RodRunner = Callable[..., tuple[Any, Any, Any, Any, Any, Any, Any]]


def _default_image_buffer(request: SimulationRequest) -> np.ndarray:
    if request.image_buffer is not None:
        return request.image_buffer
    size = int(request.geometry.image_size)
    return np.zeros((size, size), dtype=np.float64)


def _capture_hit_tables_for_intersection_cache(
    *,
    hit_tables: Any,
    collect_hit_tables_requested: bool,
    runner: Callable[..., tuple[Any, ...]],
    runner_args: tuple[Any, ...],
    runner_kwargs: dict[str, Any],
    image_arg_index: int,
) -> Any:
    """Return hit tables suitable for building the per-peak intersection cache."""

    try:
        if hit_tables is not None and len(hit_tables) > 0:
            return hit_tables
    except TypeError:
        if hit_tables is not None:
            return hit_tables

    if collect_hit_tables_requested:
        return hit_tables

    rerun_args = list(runner_args)
    rerun_args[image_arg_index] = np.zeros_like(
        np.asarray(rerun_args[image_arg_index], dtype=np.float64)
    )
    rerun_kwargs = dict(runner_kwargs)
    rerun_kwargs["collect_hit_tables"] = True
    rerun_kwargs["accumulate_image"] = False

    rerun_result = runner(*rerun_args, **rerun_kwargs)
    return rerun_result[1]


def simulate(
    request: SimulationRequest,
    *,
    peak_runner: PeakRunner = process_peaks_parallel_safe,
) -> SimulationResult:
    """Run a diffraction simulation from a typed request."""

    image_buffer = _default_image_buffer(request)

    peak_kwargs: dict[str, Any] = dict(
        save_flag=request.save_flag,
        record_status=request.record_status,
        thickness=request.thickness,
        pixel_size_m=request.geometry.pixel_size_m,
        sample_width_m=request.geometry.sample_width_m,
        sample_length_m=request.geometry.sample_length_m,
        solve_q_steps=request.mosaic.solve_q_steps,
        solve_q_rel_tol=request.mosaic.solve_q_rel_tol,
        solve_q_mode=request.mosaic.solve_q_mode,
        collect_hit_tables=request.collect_hit_tables,
        accumulate_image=request.accumulate_image,
    )
    if request.optics_mode is not None:
        peak_kwargs["optics_mode"] = request.optics_mode
    if request.beam.sample_weights is not None:
        peak_kwargs["sample_weights"] = request.beam.sample_weights
    if request.beam.n2_sample_array is not None:
        peak_kwargs["n2_sample_array_override"] = request.beam.n2_sample_array
    if request.single_sample_indices is not None:
        peak_kwargs["single_sample_indices"] = request.single_sample_indices
    if request.best_sample_indices_out is not None:
        peak_kwargs["best_sample_indices_out"] = request.best_sample_indices_out

    peak_args = (
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
    )
    image, hit_tables, q_data, q_count, all_status, miss_tables = peak_runner(
        *peak_args,
        **peak_kwargs,
    )
    cache_hit_tables = _capture_hit_tables_for_intersection_cache(
        hit_tables=hit_tables,
        collect_hit_tables_requested=bool(request.collect_hit_tables),
        runner=peak_runner,
        runner_args=peak_args,
        runner_kwargs=peak_kwargs,
        image_arg_index=6,
    )
    intersection_cache = build_intersection_cache(
        cache_hit_tables,
        request.geometry.av,
        request.geometry.cv,
        beam_x_array=request.beam.beam_x_array,
        beam_y_array=request.beam.beam_y_array,
        theta_array=request.beam.theta_array,
        phi_array=request.beam.phi_array,
        wavelength_array=request.beam.wavelength_array,
        best_sample_indices_out=request.best_sample_indices_out,
        single_sample_indices=request.single_sample_indices,
    )

    return SimulationResult(
        image=np.asarray(image, dtype=np.float64),
        hit_tables=hit_tables,
        q_data=q_data,
        q_count=q_count,
        all_status=all_status,
        miss_tables=miss_tables,
        degeneracy=None,
        intersection_cache=intersection_cache,
    )


def simulate_qr_rods(
    qr_dict: dict,
    request: SimulationRequest,
    *,
    peak_runner: RodRunner = process_qr_rods_parallel_safe,
) -> SimulationResult:
    """Run rod-based simulation through the typed request API."""

    image_buffer = _default_image_buffer(request)

    rod_kwargs: dict[str, Any] = dict(
        save_flag=request.save_flag,
        record_status=request.record_status,
        thickness=request.thickness,
        pixel_size_m=request.geometry.pixel_size_m,
        sample_width_m=request.geometry.sample_width_m,
        sample_length_m=request.geometry.sample_length_m,
        solve_q_steps=request.mosaic.solve_q_steps,
        solve_q_rel_tol=request.mosaic.solve_q_rel_tol,
        solve_q_mode=request.mosaic.solve_q_mode,
        collect_hit_tables=request.collect_hit_tables,
        accumulate_image=request.accumulate_image,
    )
    if request.optics_mode is not None:
        rod_kwargs["optics_mode"] = request.optics_mode
    if request.beam.sample_weights is not None:
        rod_kwargs["sample_weights"] = request.beam.sample_weights
    if request.beam.n2_sample_array is not None:
        rod_kwargs["n2_sample_array_override"] = request.beam.n2_sample_array

    rod_args = (
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
    )
    image, hit_tables, q_data, q_count, all_status, miss_tables, degeneracy = peak_runner(
        *rod_args,
        **rod_kwargs,
    )
    cache_hit_tables = _capture_hit_tables_for_intersection_cache(
        hit_tables=hit_tables,
        collect_hit_tables_requested=bool(request.collect_hit_tables),
        runner=peak_runner,
        runner_args=rod_args,
        runner_kwargs=rod_kwargs,
        image_arg_index=5,
    )
    intersection_cache = build_intersection_cache(
        cache_hit_tables,
        request.geometry.av,
        request.geometry.cv,
        beam_x_array=request.beam.beam_x_array,
        beam_y_array=request.beam.beam_y_array,
        theta_array=request.beam.theta_array,
        phi_array=request.beam.phi_array,
        wavelength_array=request.beam.wavelength_array,
        best_sample_indices_out=request.best_sample_indices_out,
        single_sample_indices=request.single_sample_indices,
    )

    return SimulationResult(
        image=np.asarray(image, dtype=np.float64),
        hit_tables=hit_tables,
        q_data=q_data,
        q_count=q_count,
        all_status=all_status,
        miss_tables=miss_tables,
        degeneracy=degeneracy,
        intersection_cache=intersection_cache,
    )
