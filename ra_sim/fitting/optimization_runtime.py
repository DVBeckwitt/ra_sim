"""Shared runtime helpers extracted from the monolithic optimization module."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ra_sim.debug_controls import retain_optional_cache
from ra_sim.simulation.diffraction import (
    get_last_process_peaks_safe_stats,
    process_peaks_parallel_safe as _DIFFRACTION_PROCESS_PEAKS_SAFE_WRAPPER,
)
from ra_sim.utils.parallel import (
    call_with_numba_thread_limit as _shared_call_with_numba_thread_limit,
    current_parallel_thread_budget,
    numba_threads_per_worker as _shared_numba_threads_per_worker,
    reserved_worker_count,
)

process_peaks_parallel = _DIFFRACTION_PROCESS_PEAKS_SAFE_WRAPPER

_USE_NUMBA_PROCESS_PEAKS = True
_NUMBA_PROCESS_PEAKS_WARMED = False
_NUMBA_PROCESS_PEAKS_WARMUP_LOCK = Lock()


def retain_fit_simulation_cache() -> bool:
    return retain_optional_cache("fit_simulation", feature_needed=True)


def available_parallel_thread_budget() -> int:
    """Return CPU thread budget available to outer geometry-fit workers."""

    return int(current_parallel_thread_budget())


def coerce_sequence_items(values: Optional[Sequence[object]]) -> List[object]:
    """Return sequence contents without relying on ambiguous truthiness."""

    if values is None:
        return []
    try:
        return list(values)
    except TypeError:
        return []


def resolve_parallel_worker_count(
    raw_value: object,
    *,
    max_tasks: int,
) -> int:
    """Normalize one worker-count config value against a concrete task count."""

    if max_tasks <= 1:
        return 1

    requested = 0
    if isinstance(raw_value, str):
        text = raw_value.strip().lower()
        if text in {"", "auto", "default"}:
            requested = 0
        else:
            try:
                requested = int(float(text))
            except Exception:
                requested = 1
    elif raw_value is None:
        requested = 0
    elif isinstance(raw_value, (int, float)):
        requested = int(raw_value)
    else:
        try:
            requested = int(str(raw_value).strip())
        except Exception:
            requested = 1

    if requested <= 0:
        requested = reserved_worker_count(
            thread_budget=available_parallel_thread_budget(),
        )
    return max(1, min(int(requested), int(max_tasks)))


def resolve_numba_threads_per_worker(
    worker_count: int,
    raw_value: object,
) -> Optional[int]:
    """Return Numba thread mask to use inside each outer worker."""

    if worker_count <= 1:
        return None

    requested = 0
    if isinstance(raw_value, str):
        text = raw_value.strip().lower()
        if text not in {"", "auto", "default"}:
            try:
                requested = int(float(text))
            except Exception:
                requested = 0
    elif isinstance(raw_value, (int, float)):
        requested = int(raw_value)
    elif raw_value is not None:
        try:
            requested = int(str(raw_value).strip())
        except Exception:
            requested = 0

    if requested > 0:
        return max(int(requested), 1)

    return _shared_numba_threads_per_worker(
        worker_count,
        thread_budget=available_parallel_thread_budget(),
    )


def call_with_numba_thread_limit(
    fn: Callable[..., object],
    *args,
    numba_threads: Optional[int] = None,
    **kwargs,
):
    """Run *fn* while temporarily masking Numba's worker thread count."""

    return _shared_call_with_numba_thread_limit(
        fn,
        *args,
        numba_threads=numba_threads,
        **kwargs,
    )


def threaded_map(
    fn: Callable[[object], object],
    items: Sequence[object],
    *,
    max_workers: int,
    numba_threads: Optional[int] = None,
) -> List[object]:
    """Map *fn* over *items* using a thread pool while preserving order."""

    if max_workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]

    def _run(item: object) -> object:
        return call_with_numba_thread_limit(
            fn,
            item,
            numba_threads=numba_threads,
        )

    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        return list(executor.map(_run, items))


def set_numba_usage_from_config(
    refinement_config: Optional[Dict[str, Dict[str, float]]],
) -> None:
    """Apply per-fit config flags such as disabling numba for fitting."""

    global _USE_NUMBA_PROCESS_PEAKS
    global _NUMBA_PROCESS_PEAKS_WARMED

    if isinstance(refinement_config, dict):
        use_numba = refinement_config.get("use_numba", True)
        _USE_NUMBA_PROCESS_PEAKS = bool(use_numba)
        if not _USE_NUMBA_PROCESS_PEAKS:
            _NUMBA_PROCESS_PEAKS_WARMED = False


def last_process_peaks_used_python_runner() -> Optional[bool]:
    """Return whether last safe-wrapper call fell back to Python runner."""

    if process_peaks_parallel is not _DIFFRACTION_PROCESS_PEAKS_SAFE_WRAPPER:
        return None
    try:
        stats = get_last_process_peaks_safe_stats()
    except Exception:
        return None
    if "used_python_runner" not in stats:
        return None
    return bool(stats.get("used_python_runner", False))


def process_peaks_parallel_safe(*args, **kwargs):
    """Call numba-backed process-peaks wrapper with Python fallback."""

    global _USE_NUMBA_PROCESS_PEAKS
    global _NUMBA_PROCESS_PEAKS_WARMED

    def _invoke(fn, *, prefer_python_runner: bool = False):
        call_kwargs = dict(kwargs)
        if prefer_python_runner:
            call_kwargs["prefer_python_runner"] = True

        try:
            return fn(*args, **call_kwargs)
        except TypeError as exc:
            if (
                "optics_mode" in call_kwargs
                or "solve_q_steps" in call_kwargs
                or "solve_q_rel_tol" in call_kwargs
                or "solve_q_mode" in call_kwargs
                or "thickness" in call_kwargs
                or "pixel_size_m" in call_kwargs
                or "sample_width_m" in call_kwargs
                or "sample_length_m" in call_kwargs
                or "n2_sample_array_override" in call_kwargs
                or "prefer_python_runner" in call_kwargs
            ) and "unexpected keyword" in str(exc):
                reduced_kwargs = dict(call_kwargs)
                reduced_kwargs.pop("optics_mode", None)
                reduced_kwargs.pop("solve_q_steps", None)
                reduced_kwargs.pop("solve_q_rel_tol", None)
                reduced_kwargs.pop("solve_q_mode", None)
                reduced_kwargs.pop("thickness", None)
                reduced_kwargs.pop("pixel_size_m", None)
                reduced_kwargs.pop("sample_width_m", None)
                reduced_kwargs.pop("sample_length_m", None)
                reduced_kwargs.pop("n2_sample_array_override", None)
                reduced_kwargs.pop("prefer_python_runner", None)
                return fn(*args, **reduced_kwargs)
            raise

    def _invoke_numba_path():
        global _USE_NUMBA_PROCESS_PEAKS
        global _NUMBA_PROCESS_PEAKS_WARMED

        result = _invoke(process_peaks_parallel)
        if last_process_peaks_used_python_runner() is True:
            _USE_NUMBA_PROCESS_PEAKS = False
            _NUMBA_PROCESS_PEAKS_WARMED = False
        else:
            _NUMBA_PROCESS_PEAKS_WARMED = True
        return result

    if _USE_NUMBA_PROCESS_PEAKS:
        try:
            if not _NUMBA_PROCESS_PEAKS_WARMED:
                with _NUMBA_PROCESS_PEAKS_WARMUP_LOCK:
                    if _USE_NUMBA_PROCESS_PEAKS and not _NUMBA_PROCESS_PEAKS_WARMED:
                        return _invoke_numba_path()
            if _USE_NUMBA_PROCESS_PEAKS:
                return _invoke_numba_path()
        except Exception:
            _USE_NUMBA_PROCESS_PEAKS = False
            _NUMBA_PROCESS_PEAKS_WARMED = False
    return _invoke(process_peaks_parallel, prefer_python_runner=True)


@dataclass
class SimulationCache:
    """Simple cache for simulated detector images keyed by parameter vectors."""

    keys: Sequence[str]
    images: Dict[Tuple[float, ...], np.ndarray] = field(default_factory=dict)
    max_positions: Dict[Tuple[float, ...], np.ndarray] = field(default_factory=dict)

    def _flatten_value(self, value: object) -> Iterable[float]:
        if isinstance(value, np.ndarray):
            return value.ravel()
        if isinstance(value, (list, tuple)):
            return np.asarray(value, dtype=float).ravel()
        if isinstance(value, (int, float)):
            return (float(value),)
        return (float(str(value)),)

    def key_for(self, params: Dict[str, float]) -> Tuple[float, ...]:
        parts: List[float] = []
        for key in self.keys:
            value = params[key]
            parts.extend(float(f"{v:.8f}") for v in self._flatten_value(value))
        return tuple(parts)

    def get(
        self,
        params: Dict[str, float],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not retain_fit_simulation_cache():
            self.images.clear()
            self.max_positions.clear()
            return None
        key = self.key_for(params)
        if key in self.images:
            return self.images[key], self.max_positions[key]
        return None

    def store(
        self,
        params: Dict[str, float],
        image: np.ndarray,
        max_positions: np.ndarray,
    ) -> None:
        if not retain_fit_simulation_cache():
            self.images.clear()
            self.max_positions.clear()
            return
        key = self.key_for(params)
        self.images[key] = image
        self.max_positions[key] = max_positions


__all__ = [
    "SimulationCache",
    "available_parallel_thread_budget",
    "coerce_sequence_items",
    "process_peaks_parallel",
    "process_peaks_parallel_safe",
    "resolve_numba_threads_per_worker",
    "resolve_parallel_worker_count",
    "retain_fit_simulation_cache",
    "set_numba_usage_from_config",
    "threaded_map",
]
