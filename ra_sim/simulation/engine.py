"""Typed simulation entrypoints built on top of legacy diffraction kernels."""

from __future__ import annotations

from threading import Lock, Thread
from typing import Any, Callable

import numpy as np

from ra_sim.utils.parallel import (
    default_reserved_cpu_worker_count,
    temporary_numba_thread_limit,
)

from .diffraction import (
    build_intersection_cache,
    process_peaks_parallel_safe,
    process_qr_rods_parallel_safe,
)
from .types import (
    BeamSamples,
    DebyeWallerParams,
    DetectorGeometry,
    MosaicParams,
    SimulationRequest,
    SimulationResult,
)


PeakRunner = Callable[..., tuple[Any, Any, Any, Any, Any, Any]]
RodRunner = Callable[..., tuple[Any, Any, Any, Any, Any, Any, Any]]
_FORWARD_SIMULATION_NUMBA_WARMUP_LOCK = Lock()
_FORWARD_SIMULATION_NUMBA_WARMED = False
_FORWARD_SIMULATION_NUMBA_WARMUP_THREAD: Thread | None = None


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
    rerun_kwargs.pop("projection_debug_counters", None)
    rerun_kwargs.pop("projection_debug_reject_counts", None)
    rerun_kwargs.pop("projection_debug_reject_records", None)
    rerun_kwargs.pop("projection_debug_row_hit_counts", None)
    rerun_kwargs.pop("projection_debug_row_tthp_sums", None)
    rerun_kwargs.pop("projection_debug_row_tth_sums", None)

    rerun_result = runner(*rerun_args, **rerun_kwargs)
    return rerun_result[1]


def _run_simulation_request(
    request: SimulationRequest,
    *,
    peak_runner: PeakRunner = process_peaks_parallel_safe,
) -> SimulationResult:
    """Run a diffraction simulation from a typed request."""

    from ra_sim.simulation.projection_debug import allocate_projection_debug_buffers
    from ra_sim.simulation.projection_debug import append_projection_debug_background
    from ra_sim.simulation.projection_debug import build_projection_debug_background
    from ra_sim.simulation.projection_debug import finalize_projection_debug_session
    from ra_sim.simulation.projection_debug import get_current_projection_debug_session
    from ra_sim.simulation.projection_debug import projection_debug_logging_enabled
    from ra_sim.simulation.projection_debug import projection_debug_request_settings
    from ra_sim.simulation.projection_debug import resolve_exit_projection_mode_flag
    from ra_sim.simulation.projection_debug import start_projection_debug_session

    image_buffer = _default_image_buffer(request)
    worker_count = default_reserved_cpu_worker_count()
    projection_debug_active = projection_debug_logging_enabled()
    projection_debug_buffers = (
        allocate_projection_debug_buffers(
            request.miller.shape[0],
            request.geometry.image_size,
            worker_count,
        )
        if projection_debug_active
        else {}
    )
    projection_debug_settings = (
        projection_debug_request_settings(request, source="simulation")
        if projection_debug_active
        else None
    )

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
        exit_projection_mode=resolve_exit_projection_mode_flag(request.exit_projection_mode),
    )
    if projection_debug_active:
        peak_kwargs.update(projection_debug_buffers)
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
    if peak_runner is process_peaks_parallel_safe and projection_debug_active:
        peak_kwargs["enable_safe_cache"] = False

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
    with temporary_numba_thread_limit(worker_count):
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
    projection_debug_background = None
    projection_debug_log_path = None
    if projection_debug_active:
        projection_debug_background = build_projection_debug_background(
            projection_debug_buffers,
            request.miller,
            None,
            projection_debug_settings,
        )
        projection_debug_session = get_current_projection_debug_session()
        if projection_debug_session is None:
            projection_debug_session = start_projection_debug_session(
                projection_debug_settings,
                source="simulation",
            )
            append_projection_debug_background(
                projection_debug_session,
                projection_debug_background,
            )
            projection_debug_log_path = finalize_projection_debug_session(
                projection_debug_session
            )
        else:
            append_projection_debug_background(
                projection_debug_session,
                projection_debug_background,
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
        projection_debug={
            "log_path": projection_debug_log_path,
            "background": projection_debug_background,
        },
    )


def _build_forward_simulation_numba_warmup_request() -> SimulationRequest:
    return SimulationRequest(
        miller=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        geometry=DetectorGeometry(
            image_size=2,
            av=1.0,
            cv=1.0,
            lambda_angstrom=1.0,
            distance_m=0.1,
            gamma_deg=0.0,
            Gamma_deg=0.0,
            chi_deg=0.0,
            psi_deg=0.0,
            psi_z_deg=0.0,
            zs=0.0,
            zb=0.0,
            center=np.array([1.0, 1.0], dtype=np.float64),
            theta_initial_deg=0.0,
            cor_angle_deg=0.0,
            unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            pixel_size_m=1.0e-4,
            sample_width_m=0.0,
            sample_length_m=0.0,
        ),
        beam=BeamSamples(
            beam_x_array=np.array([0.0], dtype=np.float64),
            beam_y_array=np.array([0.0], dtype=np.float64),
            theta_array=np.array([0.0], dtype=np.float64),
            phi_array=np.array([0.0], dtype=np.float64),
            wavelength_array=np.array([1.0], dtype=np.float64),
            sample_weights=None,
            n2_sample_array=None,
        ),
        mosaic=MosaicParams(
            sigma_mosaic_deg=0.2,
            gamma_mosaic_deg=0.1,
            eta=0.05,
            solve_q_steps=1000,
            solve_q_rel_tol=5.0e-4,
            solve_q_mode=0,
        ),
        debye_waller=DebyeWallerParams(x=0.0, y=0.0),
        n2=1.0 + 0.0j,
        image_buffer=np.zeros((2, 2), dtype=np.float64),
        save_flag=0,
        record_status=False,
        thickness=0.0,
        optics_mode=0,
        collect_hit_tables=True,
        accumulate_image=False,
        exit_projection_mode="internal",
    )


def warmup_forward_simulation_numba() -> bool:
    """Compile the forward-simulation hot path once with tiny dummy inputs."""

    global _FORWARD_SIMULATION_NUMBA_WARMED
    with _FORWARD_SIMULATION_NUMBA_WARMUP_LOCK:
        if _FORWARD_SIMULATION_NUMBA_WARMED:
            return False
        try:
            _run_simulation_request(
                _build_forward_simulation_numba_warmup_request(),
                peak_runner=process_peaks_parallel_safe,
            )
            _FORWARD_SIMULATION_NUMBA_WARMED = True
            return True
        except Exception:
            return False


def start_forward_simulation_numba_warmup_in_background() -> bool:
    """Start one daemon warmup thread if available and not yet warmed."""

    global _FORWARD_SIMULATION_NUMBA_WARMUP_THREAD
    with _FORWARD_SIMULATION_NUMBA_WARMUP_LOCK:
        if _FORWARD_SIMULATION_NUMBA_WARMED:
            return False
        thread = _FORWARD_SIMULATION_NUMBA_WARMUP_THREAD
        if thread is not None and thread.is_alive():
            return False
        thread = Thread(
            target=warmup_forward_simulation_numba,
            name="forward-simulation-numba-warmup",
            daemon=True,
        )
        _FORWARD_SIMULATION_NUMBA_WARMUP_THREAD = thread
        thread.start()
        return True


def simulate(
    request: SimulationRequest,
    *,
    peak_runner: PeakRunner = process_peaks_parallel_safe,
) -> SimulationResult:
    """Run a diffraction simulation from a typed request."""

    if peak_runner is process_peaks_parallel_safe:
        warmup_forward_simulation_numba()

    return _run_simulation_request(request, peak_runner=peak_runner)


def simulate_qr_rods(
    qr_dict: dict,
    request: SimulationRequest,
    *,
    peak_runner: RodRunner = process_qr_rods_parallel_safe,
) -> SimulationResult:
    """Run rod-based simulation through the typed request API."""

    from ra_sim.simulation.projection_debug import allocate_projection_debug_buffers
    from ra_sim.simulation.projection_debug import append_projection_debug_background
    from ra_sim.simulation.projection_debug import build_projection_debug_background
    from ra_sim.simulation.projection_debug import finalize_projection_debug_session
    from ra_sim.simulation.projection_debug import get_current_projection_debug_session
    from ra_sim.simulation.projection_debug import projection_debug_logging_enabled
    from ra_sim.simulation.projection_debug import projection_debug_request_settings
    from ra_sim.simulation.projection_debug import resolve_exit_projection_mode_flag
    from ra_sim.simulation.projection_debug import start_projection_debug_session
    from ra_sim.utils.stacking_fault import qr_dict_to_arrays

    image_buffer = _default_image_buffer(request)
    worker_count = default_reserved_cpu_worker_count()
    debug_miller = request.miller
    if debug_miller.size == 0:
        debug_miller, _, _, _ = qr_dict_to_arrays(qr_dict)
    projection_debug_active = projection_debug_logging_enabled()
    projection_debug_buffers = (
        allocate_projection_debug_buffers(
            debug_miller.shape[0],
            request.geometry.image_size,
            worker_count,
        )
        if projection_debug_active
        else {}
    )
    projection_debug_settings = (
        projection_debug_request_settings(
            request,
            source="simulation_qr_rods",
        )
        if projection_debug_active
        else None
    )

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
        exit_projection_mode=resolve_exit_projection_mode_flag(request.exit_projection_mode),
    )
    if projection_debug_active:
        rod_kwargs.update(projection_debug_buffers)
    if request.optics_mode is not None:
        rod_kwargs["optics_mode"] = request.optics_mode
    if request.beam.sample_weights is not None:
        rod_kwargs["sample_weights"] = request.beam.sample_weights
    if request.beam.n2_sample_array is not None:
        rod_kwargs["n2_sample_array_override"] = request.beam.n2_sample_array
    if peak_runner is process_qr_rods_parallel_safe and projection_debug_active:
        rod_kwargs["enable_safe_cache"] = False

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
    with temporary_numba_thread_limit(worker_count):
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
    projection_debug_background = None
    projection_debug_log_path = None
    if projection_debug_active:
        projection_debug_background = build_projection_debug_background(
            projection_debug_buffers,
            np.asarray(debug_miller, dtype=np.float64),
            None,
            projection_debug_settings,
        )
        projection_debug_session = get_current_projection_debug_session()
        if projection_debug_session is None:
            projection_debug_session = start_projection_debug_session(
                projection_debug_settings,
                source="simulation_qr_rods",
            )
            append_projection_debug_background(
                projection_debug_session,
                projection_debug_background,
            )
            projection_debug_log_path = finalize_projection_debug_session(
                projection_debug_session
            )
        else:
            append_projection_debug_background(
                projection_debug_session,
                projection_debug_background,
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
        projection_debug={
            "log_path": projection_debug_log_path,
            "background": projection_debug_background,
        },
    )
