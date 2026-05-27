"""Typed simulation entrypoints built on top of legacy diffraction kernels."""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Iterator, Mapping
from threading import Lock, Thread, local
from typing import Any, Callable

import numpy as np

from ra_sim.timing import timing_span
from ra_sim.utils.calculations import _legacy_kernel_n2_sample_array_from_angstrom
from ra_sim.utils.parallel import (
    current_parallel_thread_budget,
    default_reserved_cpu_worker_count,
    temporary_numba_thread_limit,
)

from .diffraction import (
    OPTICS_MODE_EXACT,
    _set_last_process_peaks_representative_hit_tables,
    build_intersection_cache,  # noqa: F401 - public monkeypatch surface for engine tests
    build_branch_representative_intersection_cache,
    get_process_peaks_runtime_kwargs,
    get_last_process_peaks_representative_hit_tables,
    normalize_events_per_beam_phase_backend,
    process_peaks_parallel_safe,
    process_qr_rods_parallel_safe,
    require_exact_optics_mode,
)
from .types import (
    BeamSamples,
    DebyeWallerParams,
    DetectorGeometry,
    MosaicParams,
    SimulationRequest,
    SimulationResult,
)


PeakRunner = Callable[..., tuple[Any, ...]]
RodRunner = Callable[..., tuple[Any, ...]]
_FORWARD_SIMULATION_NUMBA_WARMUP_LOCK = Lock()
_FORWARD_SIMULATION_NUMBA_WARMED = False
_FORWARD_SIMULATION_NUMBA_WARMUP_FAILED = False
_FORWARD_SIMULATION_NUMBA_WEIGHTED_WARMED = False
_FORWARD_SIMULATION_NUMBA_WEIGHTED_WARMUP_FAILED = False
_FORWARD_SIMULATION_NUMBA_DISABLED = False
_FORWARD_SIMULATION_NUMBA_WARMUP_THREAD: Thread | None = None
_FORWARD_SIMULATION_SAFE_RUN_STATE = local()
_QR_ROD_SIMULATION_NUMBA_WARMUP_LOCK = Lock()
_QR_ROD_SIMULATION_NUMBA_WARMED = False
_QR_ROD_SIMULATION_NUMBA_WARMUP_FAILED = False
_QR_ROD_SIMULATION_NUMBA_WARMUP_THREAD: Thread | None = None
_QR_ROD_SIMULATION_SAFE_RUN_STATE = local()


@contextlib.contextmanager
def _simulation_timing_span(
    timing_fields: Mapping[str, object] | None,
    name: str,
    **fields: object,
) -> Iterator[None]:
    if not timing_fields:
        yield
        return
    payload = dict(timing_fields)
    event_root = str(payload.pop("event_root", "simulation") or "simulation")
    payload.update(fields)
    with timing_span(
        f"{event_root}.{name}",
        phase="calculation",
        **payload,
    ):
        yield


def _set_last_forward_simulation_safe_run_used_python_runner(
    used_python_runner: bool | None,
) -> None:
    _FORWARD_SIMULATION_SAFE_RUN_STATE.used_python_runner = used_python_runner


def _set_last_qr_rod_simulation_safe_run_used_python_runner(
    used_python_runner: bool | None,
) -> None:
    _QR_ROD_SIMULATION_SAFE_RUN_STATE.used_python_runner = used_python_runner


def _default_image_buffer(request: SimulationRequest) -> np.ndarray:
    if request.image_buffer is not None:
        return request.image_buffer
    size = int(request.geometry.image_size)
    return np.zeros((size, size), dtype=np.float64)


def _simulation_worker_count() -> int:
    reserved = max(int(default_reserved_cpu_worker_count()), 1)
    budget = max(int(current_parallel_thread_budget()), 1)
    return max(1, min(reserved, budget))


def _ensure_best_sample_buffer(
    request: SimulationRequest,
    *,
    peak_count: int,
    auto_allocate: bool,
) -> np.ndarray | None:
    if request.best_sample_indices_out is not None:
        normalized = np.asarray(
            request.best_sample_indices_out,
            dtype=np.int64,
        ).reshape(-1)
        if normalized.shape[0] != int(peak_count):
            raise ValueError(
                "best_sample_indices_out length "
                f"{normalized.shape[0]} does not match peak_count {int(peak_count)}"
            )
        if request.best_sample_indices_out is not normalized:
            request.best_sample_indices_out = normalized
        return normalized
    if not auto_allocate:
        return None
    request.best_sample_indices_out = np.full(int(peak_count), -1, dtype=np.int64)
    return request.best_sample_indices_out


def _normalize_representative_hit_tables_candidate(
    candidate: object,
) -> tuple[bool, list[object] | None]:
    if candidate is None:
        return True, None
    if isinstance(candidate, np.ndarray):
        return False, None
    try:
        tables = list(candidate)  # type: ignore[arg-type]
    except TypeError:
        return False, None
    normalized: list[object] = []
    for table in tables:
        try:
            arr = np.asarray(table, dtype=np.float64)
        except Exception:
            return False, None
        if arr.ndim != 2:
            return False, None
        normalized.append(table)
    return True, normalized


def _representative_hit_tables_from_runner_result(
    result: tuple[Any, ...],
    *,
    representative_index: int,
) -> list[object] | None:
    if len(result) > representative_index:
        valid, tables = _normalize_representative_hit_tables_candidate(
            result[representative_index]
        )
        if valid:
            return tables
    return get_last_process_peaks_representative_hit_tables()


def _ensure_request_beam_n2_sample_array(request: SimulationRequest) -> np.ndarray:
    sample_count = int(np.asarray(request.beam.beam_x_array, dtype=np.float64).size)
    existing = request.beam.n2_sample_array
    if existing is not None:
        normalized = np.asarray(existing, dtype=np.complex128).reshape(-1)
        if normalized.size == sample_count:
            request.beam.n2_sample_array = np.ascontiguousarray(
                normalized,
                dtype=np.complex128,
            )
            return request.beam.n2_sample_array

    request.beam.n2_sample_array = _legacy_kernel_n2_sample_array_from_angstrom(
        request.beam.wavelength_array,
        nominal_n2=request.n2,
        sample_count=sample_count,
    )
    return request.beam.n2_sample_array


def _last_forward_simulation_safe_run_used_python_runner() -> bool | None:
    used_python_runner = getattr(
        _FORWARD_SIMULATION_SAFE_RUN_STATE,
        "used_python_runner",
        None,
    )
    if used_python_runner is None:
        return None
    return bool(used_python_runner)


def _last_qr_rod_simulation_safe_run_used_python_runner() -> bool | None:
    used_python_runner = getattr(
        _QR_ROD_SIMULATION_SAFE_RUN_STATE,
        "used_python_runner",
        None,
    )
    if used_python_runner is None:
        return None
    return bool(used_python_runner)


def _apply_forward_simulation_numba_safe_run_result(
    used_python_runner: bool | None,
) -> None:
    global _FORWARD_SIMULATION_NUMBA_WARMED
    global _FORWARD_SIMULATION_NUMBA_WARMUP_FAILED
    global _FORWARD_SIMULATION_NUMBA_DISABLED

    if used_python_runner is True:
        _FORWARD_SIMULATION_NUMBA_WARMED = False
        _FORWARD_SIMULATION_NUMBA_WARMUP_FAILED = True
        _FORWARD_SIMULATION_NUMBA_DISABLED = True
        return
    if _FORWARD_SIMULATION_NUMBA_DISABLED:
        _FORWARD_SIMULATION_NUMBA_WARMED = False
        _FORWARD_SIMULATION_NUMBA_WARMUP_FAILED = True
        _FORWARD_SIMULATION_NUMBA_DISABLED = True
        return
    _FORWARD_SIMULATION_NUMBA_WARMED = True
    _FORWARD_SIMULATION_NUMBA_WARMUP_FAILED = False
    _FORWARD_SIMULATION_NUMBA_DISABLED = False


def _apply_forward_simulation_numba_warmup_result(
    used_python_runner: bool | None,
) -> bool:
    global _FORWARD_SIMULATION_NUMBA_WARMED
    global _FORWARD_SIMULATION_NUMBA_WARMUP_FAILED
    global _FORWARD_SIMULATION_NUMBA_DISABLED

    if used_python_runner is True:
        _FORWARD_SIMULATION_NUMBA_WARMED = False
        _FORWARD_SIMULATION_NUMBA_WARMUP_FAILED = True
        _FORWARD_SIMULATION_NUMBA_DISABLED = False
        return False
    _FORWARD_SIMULATION_NUMBA_WARMED = True
    _FORWARD_SIMULATION_NUMBA_WARMUP_FAILED = False
    _FORWARD_SIMULATION_NUMBA_DISABLED = False
    return True


def _record_forward_simulation_numba_safe_run_result() -> None:
    used_python_runner = _last_forward_simulation_safe_run_used_python_runner()
    with _FORWARD_SIMULATION_NUMBA_WARMUP_LOCK:
        _apply_forward_simulation_numba_safe_run_result(used_python_runner)


def _run_simulation_request(
    request: SimulationRequest,
    *,
    peak_runner: PeakRunner = process_peaks_parallel_safe,
    timing_fields: Mapping[str, object] | None = None,
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

    with _simulation_timing_span(timing_fields, "beam_sample_generation", runner="peaks"):
        image_buffer = _default_image_buffer(request)
        worker_count = _simulation_worker_count()
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
        best_sample_indices_out = _ensure_best_sample_buffer(
            request,
            peak_count=int(np.asarray(request.miller, dtype=np.float64).shape[0]),
            auto_allocate=bool(request.build_intersection_cache),
        )
        beam_n2_sample_array = _ensure_request_beam_n2_sample_array(request)
    collect_for_cache = bool(request.collect_hit_tables or request.build_intersection_cache)

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
        events_per_beam_phase=normalize_events_per_beam_phase_backend(
            request.mosaic.events_per_beam_phase
        ),
        collect_hit_tables=collect_for_cache,
        accumulate_image=request.accumulate_image,
        exit_projection_mode=resolve_exit_projection_mode_flag(request.exit_projection_mode),
    )
    if projection_debug_active:
        peak_kwargs.update(projection_debug_buffers)
    peak_kwargs["optics_mode"] = require_exact_optics_mode(request.optics_mode)
    if request.beam.sample_weights is not None:
        peak_kwargs["sample_weights"] = request.beam.sample_weights
    peak_kwargs["n2_sample_array_override"] = beam_n2_sample_array
    if best_sample_indices_out is not None:
        peak_kwargs["best_sample_indices_out"] = best_sample_indices_out
    if peak_runner is process_peaks_parallel_safe and projection_debug_active:
        peak_kwargs["enable_safe_cache"] = False
    if peak_runner is process_peaks_parallel_safe and _FORWARD_SIMULATION_NUMBA_DISABLED:
        peak_kwargs["prefer_python_runner"] = True
    if peak_runner is process_peaks_parallel_safe:
        peak_kwargs.update(get_process_peaks_runtime_kwargs(numba_thread_count=worker_count))
    safe_run_stats_out: dict[str, Any] | None = None
    used_python_runner: bool | None = None
    if peak_runner is process_peaks_parallel_safe:
        safe_run_stats_out = {}
        peak_kwargs["_safe_stats_out"] = safe_run_stats_out
        _set_last_forward_simulation_safe_run_used_python_runner(None)

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
    with _simulation_timing_span(
        timing_fields,
        "kernel_call",
        runner="peaks",
        row_count=int(np.asarray(request.miller, dtype=np.float64).shape[0]),
        collect_hit_tables=bool(collect_for_cache),
        accumulate_image=bool(request.accumulate_image),
    ):
        with temporary_numba_thread_limit(worker_count):
            _set_last_process_peaks_representative_hit_tables(None)
            peak_result = peak_runner(
                *peak_args,
                **peak_kwargs,
            )
            image, hit_tables, q_data, q_count, all_status, miss_tables = peak_result[:6]
            representative_hit_tables = _representative_hit_tables_from_runner_result(
                peak_result,
                representative_index=6,
            )
            if safe_run_stats_out is not None:
                _set_last_forward_simulation_safe_run_used_python_runner(
                    safe_run_stats_out.get("used_python_runner")
                )
                if "used_python_runner" in safe_run_stats_out:
                    used_python_runner = bool(safe_run_stats_out.get("used_python_runner"))
                peak_kwargs.pop("_safe_stats_out", None)
    with _simulation_timing_span(timing_fields, "result_ready", runner="peaks"):
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
        if request.build_intersection_cache:
            intersection_cache = build_branch_representative_intersection_cache(
                representative_hit_tables if representative_hit_tables is not None else hit_tables,
                request.geometry.av,
                request.geometry.cv,
                beam_x_array=request.beam.beam_x_array,
                beam_y_array=request.beam.beam_y_array,
                theta_array=request.beam.theta_array,
                phi_array=request.beam.phi_array,
                wavelength_array=request.beam.wavelength_array,
                best_sample_indices_out=best_sample_indices_out,
                group_by_qr_set=False,
            )
        else:
            intersection_cache = []
        public_hit_tables = hit_tables if request.collect_hit_tables else []

        return SimulationResult(
            image=np.asarray(image, dtype=np.float64),
            hit_tables=public_hit_tables,
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
            used_python_runner=used_python_runner,
        )


def _build_forward_simulation_numba_warmup_request() -> SimulationRequest:
    return _build_forward_simulation_numba_warmup_request_with_overrides()


def _build_forward_simulation_numba_warmup_request_with_overrides(
    *,
    sample_weights: np.ndarray | None = None,
) -> SimulationRequest:
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
            sample_weights=sample_weights,
            n2_sample_array=np.array([1.0 + 0.0j], dtype=np.complex128),
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
        optics_mode=OPTICS_MODE_EXACT,
        collect_hit_tables=True,
        accumulate_image=True,
        exit_projection_mode="external",
    )


def _build_weighted_forward_simulation_numba_warmup_request() -> SimulationRequest:
    return _build_forward_simulation_numba_warmup_request_with_overrides(
        sample_weights=np.array([1.0], dtype=np.float64)
    )


def _build_qr_rod_simulation_numba_warmup_qr_dict() -> dict[int, dict[str, Any]]:
    return {
        1: {
            "hk": (1, 0),
            "L": np.array([0.0], dtype=np.float64),
            "I": np.array([1.0], dtype=np.float64),
            "deg": 1,
        }
    }


def _apply_qr_rod_simulation_numba_warmup_result(
    used_python_runner: bool | None,
) -> bool:
    global _QR_ROD_SIMULATION_NUMBA_WARMED
    global _QR_ROD_SIMULATION_NUMBA_WARMUP_FAILED

    if used_python_runner is True:
        _QR_ROD_SIMULATION_NUMBA_WARMED = False
        _QR_ROD_SIMULATION_NUMBA_WARMUP_FAILED = True
        return False
    _QR_ROD_SIMULATION_NUMBA_WARMED = True
    _QR_ROD_SIMULATION_NUMBA_WARMUP_FAILED = False
    return True


def _warmup_forward_simulation_numba_weighted_locked() -> None:
    global _FORWARD_SIMULATION_NUMBA_WEIGHTED_WARMED
    global _FORWARD_SIMULATION_NUMBA_WEIGHTED_WARMUP_FAILED

    if (
        _FORWARD_SIMULATION_NUMBA_WEIGHTED_WARMED
        or _FORWARD_SIMULATION_NUMBA_WEIGHTED_WARMUP_FAILED
        or _FORWARD_SIMULATION_NUMBA_DISABLED
    ):
        return
    try:
        _run_simulation_request(
            _build_weighted_forward_simulation_numba_warmup_request(),
            peak_runner=process_peaks_parallel_safe,
        )
    except Exception:
        _FORWARD_SIMULATION_NUMBA_WEIGHTED_WARMED = False
        _FORWARD_SIMULATION_NUMBA_WEIGHTED_WARMUP_FAILED = True
        return
    if _last_forward_simulation_safe_run_used_python_runner() is True:
        _FORWARD_SIMULATION_NUMBA_WEIGHTED_WARMED = False
        _FORWARD_SIMULATION_NUMBA_WEIGHTED_WARMUP_FAILED = True
        return
    _FORWARD_SIMULATION_NUMBA_WEIGHTED_WARMED = True
    _FORWARD_SIMULATION_NUMBA_WEIGHTED_WARMUP_FAILED = False


def warmup_forward_simulation_numba() -> bool:
    """Compile the forward-simulation hot path once with tiny dummy inputs."""

    global _FORWARD_SIMULATION_NUMBA_WARMED
    global _FORWARD_SIMULATION_NUMBA_WARMUP_FAILED
    global _FORWARD_SIMULATION_NUMBA_DISABLED
    with _FORWARD_SIMULATION_NUMBA_WARMUP_LOCK:
        if _FORWARD_SIMULATION_NUMBA_WARMUP_FAILED or _FORWARD_SIMULATION_NUMBA_DISABLED:
            return False
        if not _FORWARD_SIMULATION_NUMBA_WARMED:
            try:
                _run_simulation_request(
                    _build_forward_simulation_numba_warmup_request(),
                    peak_runner=process_peaks_parallel_safe,
                )
                warmed = _apply_forward_simulation_numba_warmup_result(
                    _last_forward_simulation_safe_run_used_python_runner()
                )
            except Exception:
                _FORWARD_SIMULATION_NUMBA_WARMED = False
                _FORWARD_SIMULATION_NUMBA_WARMUP_FAILED = True
                _FORWARD_SIMULATION_NUMBA_DISABLED = False
                return False
            if not warmed:
                return False
            _warmup_forward_simulation_numba_weighted_locked()
            return True
        _warmup_forward_simulation_numba_weighted_locked()
        return False


def warmup_qr_rod_simulation_numba() -> bool:
    """Compile the rod-simulation hot path once with tiny dummy inputs."""

    global _QR_ROD_SIMULATION_NUMBA_WARMED
    global _QR_ROD_SIMULATION_NUMBA_WARMUP_FAILED

    with _QR_ROD_SIMULATION_NUMBA_WARMUP_LOCK:
        if _QR_ROD_SIMULATION_NUMBA_WARMED or _QR_ROD_SIMULATION_NUMBA_WARMUP_FAILED:
            return False
        try:
            simulate_qr_rods(
                _build_qr_rod_simulation_numba_warmup_qr_dict(),
                _build_forward_simulation_numba_warmup_request(),
                peak_runner=process_qr_rods_parallel_safe,
            )
            return _apply_qr_rod_simulation_numba_warmup_result(
                _last_qr_rod_simulation_safe_run_used_python_runner()
            )
        except Exception:
            _QR_ROD_SIMULATION_NUMBA_WARMED = False
            _QR_ROD_SIMULATION_NUMBA_WARMUP_FAILED = True
            return False


def start_forward_simulation_numba_warmup_in_background() -> bool:
    """Start one daemon warmup thread if available and not yet warmed."""

    global _FORWARD_SIMULATION_NUMBA_WARMUP_THREAD
    with _FORWARD_SIMULATION_NUMBA_WARMUP_LOCK:
        if (
            _FORWARD_SIMULATION_NUMBA_WARMED
            or _FORWARD_SIMULATION_NUMBA_WARMUP_FAILED
            or _FORWARD_SIMULATION_NUMBA_DISABLED
        ):
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


def start_qr_rod_simulation_numba_warmup_in_background() -> bool:
    """Start one daemon rod warmup thread if not yet warmed."""

    global _QR_ROD_SIMULATION_NUMBA_WARMUP_THREAD
    with _QR_ROD_SIMULATION_NUMBA_WARMUP_LOCK:
        if _QR_ROD_SIMULATION_NUMBA_WARMED or _QR_ROD_SIMULATION_NUMBA_WARMUP_FAILED:
            return False
        thread = _QR_ROD_SIMULATION_NUMBA_WARMUP_THREAD
        if thread is not None and thread.is_alive():
            return False
        thread = Thread(
            target=warmup_qr_rod_simulation_numba,
            name="qr-rod-simulation-numba-warmup",
            daemon=True,
        )
        _QR_ROD_SIMULATION_NUMBA_WARMUP_THREAD = thread
        thread.start()
        return True


def _call_run_simulation_request(
    request: SimulationRequest,
    *,
    peak_runner: PeakRunner,
    timing_fields: Mapping[str, object] | None,
) -> SimulationResult:
    """Call the request runner while preserving legacy test doubles."""

    runner = _run_simulation_request
    supports_timing_fields = True
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        parameters = signature.parameters.values()
        supports_timing_fields = "timing_fields" in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
    if supports_timing_fields:
        return runner(
            request,
            peak_runner=peak_runner,
            timing_fields=timing_fields,
        )
    return runner(request, peak_runner=peak_runner)


def simulate(
    request: SimulationRequest,
    *,
    peak_runner: PeakRunner = process_peaks_parallel_safe,
    timing_fields: Mapping[str, object] | None = None,
) -> SimulationResult:
    """Run a diffraction simulation from a typed request."""

    global _FORWARD_SIMULATION_NUMBA_WARMED
    global _FORWARD_SIMULATION_NUMBA_WARMUP_FAILED
    global _FORWARD_SIMULATION_NUMBA_DISABLED

    if peak_runner is process_peaks_parallel_safe:
        if not _FORWARD_SIMULATION_NUMBA_DISABLED:
            warmup_forward_simulation_numba()
        if not _FORWARD_SIMULATION_NUMBA_WARMED and not _FORWARD_SIMULATION_NUMBA_DISABLED:
            with _FORWARD_SIMULATION_NUMBA_WARMUP_LOCK:
                if not _FORWARD_SIMULATION_NUMBA_WARMED and not _FORWARD_SIMULATION_NUMBA_DISABLED:
                    try:
                        result = _call_run_simulation_request(
                            request,
                            peak_runner=peak_runner,
                            timing_fields=timing_fields,
                        )
                    except Exception:
                        _FORWARD_SIMULATION_NUMBA_WARMED = False
                        _FORWARD_SIMULATION_NUMBA_WARMUP_FAILED = True
                        _FORWARD_SIMULATION_NUMBA_DISABLED = True
                        raise
                    _apply_forward_simulation_numba_safe_run_result(
                        _last_forward_simulation_safe_run_used_python_runner()
                    )
                    return result

    result = _call_run_simulation_request(
        request,
        peak_runner=peak_runner,
        timing_fields=timing_fields,
    )
    if peak_runner is process_peaks_parallel_safe:
        _record_forward_simulation_numba_safe_run_result()
    return result


def simulate_qr_rods(
    qr_dict: dict,
    request: SimulationRequest,
    *,
    peak_runner: RodRunner = process_qr_rods_parallel_safe,
    timing_fields: Mapping[str, object] | None = None,
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

    with _simulation_timing_span(timing_fields, "beam_sample_generation", runner="qr_rods"):
        image_buffer = _default_image_buffer(request)
        worker_count = _simulation_worker_count()
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
        best_sample_indices_out = _ensure_best_sample_buffer(
            request,
            peak_count=int(np.asarray(debug_miller, dtype=np.float64).shape[0]),
            auto_allocate=bool(request.build_intersection_cache),
        )
        beam_n2_sample_array = _ensure_request_beam_n2_sample_array(request)
    collect_for_cache = bool(request.collect_hit_tables or request.build_intersection_cache)

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
        events_per_beam_phase=normalize_events_per_beam_phase_backend(
            request.mosaic.events_per_beam_phase
        ),
        collect_hit_tables=collect_for_cache,
        accumulate_image=request.accumulate_image,
        exit_projection_mode=resolve_exit_projection_mode_flag(request.exit_projection_mode),
    )
    if projection_debug_active:
        rod_kwargs.update(projection_debug_buffers)
    rod_kwargs["optics_mode"] = require_exact_optics_mode(request.optics_mode)
    if request.beam.sample_weights is not None:
        rod_kwargs["sample_weights"] = request.beam.sample_weights
    rod_kwargs["n2_sample_array_override"] = beam_n2_sample_array
    if best_sample_indices_out is not None:
        rod_kwargs["best_sample_indices_out"] = best_sample_indices_out
    if peak_runner is process_qr_rods_parallel_safe and projection_debug_active:
        rod_kwargs["enable_safe_cache"] = False
    safe_run_stats_out: dict[str, Any] | None = None
    used_python_runner: bool | None = None
    if peak_runner is process_qr_rods_parallel_safe:
        safe_run_stats_out = {}
        rod_kwargs["_safe_stats_out"] = safe_run_stats_out
        _set_last_qr_rod_simulation_safe_run_used_python_runner(None)

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
    with _simulation_timing_span(
        timing_fields,
        "kernel_call",
        runner="qr_rods",
        row_count=int(np.asarray(debug_miller, dtype=np.float64).shape[0]),
        collect_hit_tables=bool(collect_for_cache),
        accumulate_image=bool(request.accumulate_image),
    ):
        with temporary_numba_thread_limit(worker_count):
            _set_last_process_peaks_representative_hit_tables(None)
            rod_result = peak_runner(
                *rod_args,
                **rod_kwargs,
            )
            image, hit_tables, q_data, q_count, all_status, miss_tables, degeneracy = (
                rod_result[:7]
            )
            representative_hit_tables = _representative_hit_tables_from_runner_result(
                rod_result,
                representative_index=7,
            )
            if safe_run_stats_out is not None:
                _set_last_qr_rod_simulation_safe_run_used_python_runner(
                    safe_run_stats_out.get("used_python_runner")
                )
                if "used_python_runner" in safe_run_stats_out:
                    used_python_runner = bool(safe_run_stats_out.get("used_python_runner"))
                rod_kwargs.pop("_safe_stats_out", None)
    with _simulation_timing_span(timing_fields, "result_ready", runner="qr_rods"):
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
        if request.build_intersection_cache:
            intersection_cache = build_branch_representative_intersection_cache(
                representative_hit_tables if representative_hit_tables is not None else hit_tables,
                request.geometry.av,
                request.geometry.cv,
                beam_x_array=request.beam.beam_x_array,
                beam_y_array=request.beam.beam_y_array,
                theta_array=request.beam.theta_array,
                phi_array=request.beam.phi_array,
                wavelength_array=request.beam.wavelength_array,
                best_sample_indices_out=best_sample_indices_out,
                group_by_qr_set=True,
            )
        else:
            intersection_cache = []
        public_hit_tables = hit_tables if request.collect_hit_tables else []

        return SimulationResult(
            image=np.asarray(image, dtype=np.float64),
            hit_tables=public_hit_tables,
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
            used_python_runner=used_python_runner,
        )
