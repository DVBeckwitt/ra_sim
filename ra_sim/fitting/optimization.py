"""Optimization routines for fitting simulated data to experiments."""

import copy
import hashlib
from dataclasses import dataclass
import math
from threading import Lock
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import (
    OptimizeResult,
    differential_evolution,
    least_squares,
    linear_sum_assignment,
)
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel, zoom
from scipy.spatial import cKDTree

from ra_sim.fitting import optimization_mosaic_profiles as _mosaic_profiles
from ra_sim.fitting import optimization_runtime as _runtime
from ra_sim.fitting.optimization_runtime import SimulationCache
from ra_sim.simulation.diffraction import (
    get_last_process_peaks_safe_stats,
    hit_tables_to_max_positions,
    process_peaks_parallel_safe as _DIFFRACTION_PROCESS_PEAKS_SAFE_WRAPPER,
)
from ra_sim.utils.calculations import (
    d_spacing,
    resolve_canonical_branch,
    source_branch_index_from_phi_deg,
    two_theta,
)
from ra_sim.utils.parallel import (
    numba_threads_per_worker as _shared_numba_threads_per_worker,
    reserved_worker_count,
)

RNG = np.random.default_rng(42)

process_peaks_parallel = getattr(
    _runtime,
    "process_peaks_parallel",
    _DIFFRACTION_PROCESS_PEAKS_SAFE_WRAPPER,
)
_USE_NUMBA_PROCESS_PEAKS = bool(getattr(_runtime, "_USE_NUMBA_PROCESS_PEAKS", True))
_NUMBA_PROCESS_PEAKS_WARMED = bool(getattr(_runtime, "_NUMBA_PROCESS_PEAKS_WARMED", False))
_NUMBA_PROCESS_PEAKS_WARMUP_LOCK = getattr(
    _runtime,
    "_NUMBA_PROCESS_PEAKS_WARMUP_LOCK",
    Lock(),
)


def _retain_fit_simulation_cache() -> bool:
    """Compatibility shim around extracted fit-simulation cache helper."""

    return _runtime.retain_fit_simulation_cache()


def _available_parallel_thread_budget() -> int:
    """Compatibility shim around extracted parallel-budget helper."""

    return _runtime.available_parallel_thread_budget()


def _coerce_sequence_items(values: Optional[Sequence[object]]) -> List[object]:
    """Compatibility shim around extracted sequence-normalization helper."""

    return _runtime.coerce_sequence_items(values)


def _resolve_parallel_worker_count(
    raw_value: object,
    *,
    max_tasks: int,
) -> int:
    """Resolve worker count while preserving local monkeypatch seams."""

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
            thread_budget=_available_parallel_thread_budget(),
        )
    return max(1, min(int(requested), int(max_tasks)))


def _resolve_numba_threads_per_worker(
    worker_count: int,
    raw_value: object,
) -> Optional[int]:
    """Resolve per-worker numba thread mask with local patchable budget."""

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
        thread_budget=_available_parallel_thread_budget(),
    )


def _threaded_map(
    fn: Callable[[object], object],
    items: Sequence[object],
    *,
    max_workers: int,
    numba_threads: Optional[int] = None,
) -> List[object]:
    """Compatibility shim around extracted threaded-map helper."""

    return _runtime.threaded_map(
        fn,
        items,
        max_workers=max_workers,
        numba_threads=numba_threads,
    )


def _set_numba_usage_from_config(
    refinement_config: Optional[Dict[str, Dict[str, float]]],
) -> None:
    """Apply runtime numba flag while keeping local compatibility globals."""

    global _USE_NUMBA_PROCESS_PEAKS
    global _NUMBA_PROCESS_PEAKS_WARMED

    if isinstance(refinement_config, dict):
        use_numba = refinement_config.get("use_numba", True)
        _USE_NUMBA_PROCESS_PEAKS = bool(use_numba)
        if not _USE_NUMBA_PROCESS_PEAKS:
            _NUMBA_PROCESS_PEAKS_WARMED = False

    _runtime._USE_NUMBA_PROCESS_PEAKS = bool(_USE_NUMBA_PROCESS_PEAKS)
    _runtime._NUMBA_PROCESS_PEAKS_WARMED = bool(_NUMBA_PROCESS_PEAKS_WARMED)


def _last_process_peaks_used_python_runner() -> Optional[bool]:
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


def _process_peaks_parallel_safe(*args, **kwargs):
    """Call numba-backed process-peaks wrapper with Python fallback."""

    global _USE_NUMBA_PROCESS_PEAKS
    global _NUMBA_PROCESS_PEAKS_WARMED

    def _coerce_python_runner_args(call_args: Tuple[object, ...]) -> Tuple[object, ...]:
        if len(call_args) < 24:
            return call_args

        coerced_args = list(call_args)
        zero_ray = np.zeros(1, dtype=np.float64)
        coerced_args[16] = zero_ray.copy()
        coerced_args[17] = zero_ray.copy()
        coerced_args[18] = zero_ray.copy()
        coerced_args[19] = zero_ray.copy()

        wavelength_scalar = 1.0
        try:
            wavelength_values = np.asarray(call_args[5], dtype=np.float64).reshape(-1)
        except Exception:
            wavelength_values = np.asarray([], dtype=np.float64)
        if (
            wavelength_values.size == 1
            and np.isfinite(wavelength_values[0])
            and wavelength_values[0] > 0.0
        ):
            wavelength_scalar = float(wavelength_values[0])
        wavelength_array = np.asarray([wavelength_scalar], dtype=np.float64)
        coerced_args[5] = wavelength_array.copy()
        coerced_args[23] = wavelength_array.copy()
        return tuple(coerced_args)

    def _invoke(fn, *, prefer_python_runner: bool = False):
        call_kwargs = dict(kwargs)
        call_args = args
        if prefer_python_runner:
            call_kwargs["prefer_python_runner"] = True
            call_args = _coerce_python_runner_args(call_args)

        try:
            return fn(*call_args, **call_kwargs)
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
                return fn(*call_args, **reduced_kwargs)
            raise

    def _invoke_numba_path():
        global _USE_NUMBA_PROCESS_PEAKS
        global _NUMBA_PROCESS_PEAKS_WARMED

        result = _invoke(process_peaks_parallel)
        if _last_process_peaks_used_python_runner() is True:
            _USE_NUMBA_PROCESS_PEAKS = False
            _NUMBA_PROCESS_PEAKS_WARMED = False
        else:
            _NUMBA_PROCESS_PEAKS_WARMED = True
        _runtime._USE_NUMBA_PROCESS_PEAKS = bool(_USE_NUMBA_PROCESS_PEAKS)
        _runtime._NUMBA_PROCESS_PEAKS_WARMED = bool(_NUMBA_PROCESS_PEAKS_WARMED)
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
            _runtime._USE_NUMBA_PROCESS_PEAKS = False
            _runtime._NUMBA_PROCESS_PEAKS_WARMED = False
    return _invoke(process_peaks_parallel, prefer_python_runner=True)


@dataclass
class TubeROI:
    """Representation of a physics-motivated tube ROI around a reflection."""

    reflection: Tuple[int, int, int]
    centerline: np.ndarray
    width: float
    bounds: Tuple[int, int, int, int]
    mask: np.ndarray
    off_tube_mask: np.ndarray
    sampling_probability: float = 1.0
    weights: Optional[np.ndarray] = None
    active_pixels: Optional[np.ndarray] = None
    full_weight_map: Optional[np.ndarray] = None
    active_mask: Optional[np.ndarray] = None
    centerline_mask: Optional[np.ndarray] = None
    tile_size: int = 8
    tile_probabilities: Optional[np.ndarray] = None
    identifier: int = 0


@dataclass
class PeakROI:
    """Lightweight container describing a square ROI around a simulated peak."""

    reflection_index: int
    hkl: Tuple[int, int, int]
    center: Tuple[float, float]
    row_indices: np.ndarray
    col_indices: np.ndarray
    flat_indices: np.ndarray
    observed: np.ndarray
    observed_sum: float
    observed_mean: float
    num_pixels: int
    simulated_intensity: float
    candidate_snr: float = float("nan")
    source: str = "auto"
    score: float = 0.0


@dataclass
class MosaicShapeROI:
    """Fixed-size detector ROI used by the geometry-cached mosaic shape fit."""

    dataset_index: int
    dataset_label: str
    reflection_index: int
    hkl: Tuple[int, int, int]
    center_row: float
    center_col: float
    row_bounds: Tuple[int, int]
    col_bounds: Tuple[int, int]
    measured_mask: np.ndarray
    measured_distance: np.ndarray
    measured_active_pixels: int
    measured_two_theta: float


@dataclass
class MosaicShapeDatasetContext:
    """Prepared per-dataset inputs for detector-shape mosaic fitting."""

    dataset_index: int
    label: str
    theta_initial: float
    experimental_image: np.ndarray
    geometry_context: "GeometryFitDatasetContext"
    miller: np.ndarray
    intensities: np.ndarray
    rois: List[MosaicShapeROI]
    measured_peak_count: int


MosaicProfileROI = _mosaic_profiles.MosaicProfileROI
MosaicProfileDatasetContext = _mosaic_profiles.MosaicProfileDatasetContext


@dataclass
class IterativeRefinementResult:
    """Container mimicking scipy's ``OptimizeResult`` with extra context."""

    x: np.ndarray
    fun: np.ndarray
    success: bool
    message: str
    best_params: Dict[str, float]
    history: List[Dict[str, float]]
    stage_summaries: List[Dict[str, float]]

    def __iter__(self):
        yield from (
            ("x", self.x),
            ("fun", self.fun),
            ("success", self.success),
            ("message", self.message),
        )


def _downsample_with_antialiasing(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample *image* by *factor* using Gaussian pre-filtering."""

    if factor <= 1:
        return image
    sigma = 0.5 * factor
    blurred = gaussian_filter(image, sigma=sigma, mode="reflect")
    return zoom(blurred, 1.0 / factor, order=1, prefilter=False)


def _compute_ridge_map(image: np.ndarray, percentile: float = 85.0) -> np.ndarray:
    """Return a binary ridge map derived from gradient magnitude."""

    grad_x = sobel(image, axis=1, mode="reflect")
    grad_y = sobel(image, axis=0, mode="reflect")
    magnitude = np.hypot(grad_x, grad_y)
    threshold = np.percentile(magnitude, percentile)
    if not np.isfinite(threshold) or threshold <= 0:
        threshold = float(np.mean(magnitude))
    return magnitude > threshold


def _update_params(
    params: Dict[str, float],
    var_names: Sequence[str],
    values: Sequence[float],
) -> Dict[str, float]:
    updated = dict(params)
    try:
        center_seed = updated.get(
            "center", (updated.get("center_x", 0.0), updated.get("center_y", 0.0))
        )
        center_row = float(center_seed[0])
        center_col = float(center_seed[1])
    except Exception:
        center_row = float(updated.get("center_x", 0.0))
        center_col = float(updated.get("center_y", 0.0))
    for name, val in zip(var_names, values):
        val_float = float(val)
        if name == "center_x":
            center_row = val_float
            updated["center_x"] = val_float
        elif name == "center_y":
            center_col = val_float
            updated["center_y"] = val_float
        else:
            updated[name] = val_float
    updated["center"] = [float(center_row), float(center_col)]
    updated["center_x"] = float(center_row)
    updated["center_y"] = float(center_col)
    return updated


def _allowed_reflection_mask(miller: np.ndarray) -> np.ndarray:
    """Return a mask selecting reflections with ``2h + k`` divisible by 3."""

    if miller.ndim != 2 or miller.shape[1] < 2:
        raise ValueError("miller array must have shape (N, >=3)")
    hk = np.rint(miller[:, :2]).astype(np.int64, copy=False)
    return (2 * hk[:, 0] + hk[:, 1]) % 3 == 0


def _parse_hkl_label(value: str) -> Optional[Tuple[int, int, int]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(",")
    if len(parts) != 3:
        return None
    try:
        return tuple(int(round(float(part.strip()))) for part in parts)  # type: ignore[return-value]
    except Exception:
        return None


def _normalize_measured_peaks(
    measured_peaks: Optional[Sequence[object]],
) -> List[Dict[str, object]]:
    """Normalize mixed measured-peak inputs into dicts with HKL + detector coords."""

    normalized: List[Dict[str, object]] = []
    measured_entries = _coerce_sequence_items(measured_peaks)
    if not measured_entries:
        return normalized

    for entry in measured_entries:
        hkl: Optional[Tuple[int, int, int]] = None
        x_val = None
        y_val = None
        label = None
        normalized_entry: Dict[str, object] = {}

        if isinstance(entry, dict):
            normalized_entry = dict(entry)
            if "hkl" in entry:
                try:
                    raw_hkl = entry["hkl"]
                    if isinstance(raw_hkl, (list, tuple, np.ndarray)) and len(raw_hkl) >= 3:
                        hkl = (
                            int(round(float(raw_hkl[0]))),
                            int(round(float(raw_hkl[1]))),
                            int(round(float(raw_hkl[2]))),
                        )
                except Exception:
                    hkl = None
            if hkl is None:
                hkl = _parse_hkl_label(entry.get("label"))  # type: ignore[arg-type]
            label = entry.get("label")
            x_val = entry.get("x")
            y_val = entry.get("y")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 5:
            try:
                hkl = (
                    int(round(float(entry[0]))),
                    int(round(float(entry[1]))),
                    int(round(float(entry[2]))),
                )
                x_val = entry[3]
                y_val = entry[4]
                label = f"{hkl[0]},{hkl[1]},{hkl[2]}"
            except Exception:
                hkl = None
        else:
            continue

        if hkl is None:
            continue
        try:
            x = float(x_val)  # type: ignore[arg-type]
            y = float(y_val)  # type: ignore[arg-type]
        except Exception:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        normalized_entry["hkl"] = hkl
        normalized_entry["label"] = (
            str(label) if label is not None else f"{hkl[0]},{hkl[1]},{hkl[2]}"
        )
        normalized_entry["x"] = x
        normalized_entry["y"] = y

        for key in ("raw_x", "raw_y", "placement_error_px"):
            raw_value = normalized_entry.get(key)
            if raw_value is None:
                continue
            try:
                numeric = float(raw_value)
            except Exception:
                normalized_entry.pop(key, None)
                continue
            if np.isfinite(numeric):
                normalized_entry[key] = float(numeric)
            else:
                normalized_entry.pop(key, None)

        sigma_value = (
            normalized_entry.get("sigma_px")
            if normalized_entry.get("sigma_px") is not None
            else normalized_entry.get(
                "position_sigma_px",
                normalized_entry.get("measurement_sigma_px"),
            )
        )
        if sigma_value is not None:
            try:
                sigma_px = float(sigma_value)
            except Exception:
                sigma_px = float("nan")
            if np.isfinite(sigma_px) and sigma_px > 0.0:
                normalized_entry["sigma_px"] = float(sigma_px)
            else:
                normalized_entry.pop("sigma_px", None)

        for source_key, target_key in (
            ("sigma_radial_px", "sigma_radial_px"),
            ("radial_sigma_px", "sigma_radial_px"),
            ("sigma_tangential_px", "sigma_tangential_px"),
            ("tangential_sigma_px", "sigma_tangential_px"),
        ):
            raw_value = normalized_entry.get(source_key)
            if raw_value is None:
                continue
            try:
                sigma_component = float(raw_value)
            except Exception:
                normalized_entry.pop(target_key, None)
                continue
            if np.isfinite(sigma_component) and sigma_component > 0.0:
                normalized_entry[target_key] = float(sigma_component)
            else:
                normalized_entry.pop(target_key, None)
        normalized.append(normalized_entry)
    return normalized


def _entry_has_geometry_source_identity(entry: Mapping[str, object]) -> bool:
    """Return whether one measured entry can be resolved back to a cached sim peak."""

    try:
        reflection_idx = int(entry.get("source_reflection_index", -1))
    except Exception:
        reflection_idx = -1
    try:
        table_idx = int(entry.get("source_table_index", -1))
    except Exception:
        table_idx = -1
    if reflection_idx < 0 and table_idx < 0:
        return False

    try:
        row_idx = int(entry.get("source_row_index", -1))
    except Exception:
        row_idx = -1
    if row_idx >= 0:
        return True

    branch_idx = _measured_source_peak_index(entry)
    return branch_idx in {0, 1}


def _measured_entries_have_geometry_source_identities(
    measured_entries: Optional[Sequence[object]],
) -> bool:
    """Return whether every measured entry carries cached source identity metadata."""

    entries = _coerce_sequence_items(measured_entries)
    if not entries:
        return False

    saw_entry = False
    for entry in entries:
        if not isinstance(entry, Mapping):
            return False
        saw_entry = True
        if not _entry_has_geometry_source_identity(entry):
            return False
    return saw_entry


def _dataset_specs_have_geometry_source_identities(
    dataset_specs: Optional[Sequence[object]],
) -> bool:
    """Return whether every dataset spec uses cache-backed measured peak identities."""

    specs = _coerce_sequence_items(dataset_specs)
    if not specs:
        return False

    saw_dataset = False
    for spec in specs:
        if not isinstance(spec, Mapping):
            return False
        saw_dataset = True
        if not _measured_entries_have_geometry_source_identities(spec.get("measured_peaks")):
            return False
    return saw_dataset


def _measured_entry_sigma_px(
    entry: Dict[str, object],
    *,
    default_sigma_px: float = 1.0,
) -> Tuple[float, bool]:
    """Return one measured-peak sigma in pixels and whether it was user-specified."""

    sigma_value = (
        entry.get("sigma_px")
        if entry.get("sigma_px") is not None
        else entry.get(
            "position_sigma_px",
            entry.get("measurement_sigma_px"),
        )
    )
    if sigma_value is None:
        return float(default_sigma_px), False
    try:
        sigma_px = float(sigma_value)
    except Exception:
        return float(default_sigma_px), False
    if not np.isfinite(sigma_px) or sigma_px <= 0.0:
        return float(default_sigma_px), False
    return float(sigma_px), True


def _measured_entry_sigma_components(
    entry: Dict[str, object],
    *,
    default_sigma_px: float = 1.0,
    anisotropic_enabled: bool = False,
    radial_scale: float = 1.0,
    tangential_scale: float = 1.0,
) -> Tuple[float, float, float, bool, bool]:
    """Return isotropic and radial/tangential measurement sigmas in pixels."""

    sigma_px, has_custom_sigma = _measured_entry_sigma_px(
        entry,
        default_sigma_px=default_sigma_px,
    )
    sigma_radial = float(
        entry.get(
            "sigma_radial_px",
            entry.get("radial_sigma_px", float("nan")),
        )
    )
    sigma_tangential = float(
        entry.get(
            "sigma_tangential_px",
            entry.get("tangential_sigma_px", float("nan")),
        )
    )

    has_custom_anisotropy = bool(np.isfinite(sigma_radial) or np.isfinite(sigma_tangential))
    radial_scale = float(radial_scale) if np.isfinite(radial_scale) else 1.0
    tangential_scale = float(tangential_scale) if np.isfinite(tangential_scale) else 1.0
    radial_scale = max(radial_scale, 1.0e-6)
    tangential_scale = max(tangential_scale, 1.0e-6)

    if not np.isfinite(sigma_radial) or sigma_radial <= 0.0:
        sigma_radial = float(sigma_px) * (radial_scale if anisotropic_enabled else 1.0)
    if not np.isfinite(sigma_tangential) or sigma_tangential <= 0.0:
        sigma_tangential = float(sigma_px) * (tangential_scale if anisotropic_enabled else 1.0)

    sigma_radial = max(float(sigma_radial), 1.0e-6)
    sigma_tangential = max(float(sigma_tangential), 1.0e-6)
    anisotropic_used = bool(
        has_custom_anisotropy
        or (
            anisotropic_enabled
            and not math.isclose(
                float(sigma_radial),
                float(sigma_tangential),
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            )
        )
    )
    return (
        float(sigma_px),
        float(sigma_radial),
        float(sigma_tangential),
        bool(has_custom_sigma),
        bool(anisotropic_used),
    )


def _radial_tangential_basis(
    point: Tuple[float, float],
    center: Sequence[float],
) -> Optional[np.ndarray]:
    """Return a detector-pixel radial/tangential basis at *point*."""

    if center is None or len(center) < 2:
        return None
    try:
        center_row = float(center[0])
        center_col = float(center[1])
        point_col = float(point[0])
        point_row = float(point[1])
    except Exception:
        return None
    if not (
        np.isfinite(center_row)
        and np.isfinite(center_col)
        and np.isfinite(point_col)
        and np.isfinite(point_row)
    ):
        return None

    radial = np.array(
        [point_col - center_col, point_row - center_row],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(radial))
    if norm <= 1.0e-9:
        return None
    radial_unit = radial / norm
    tangential_unit = np.array(
        [-radial_unit[1], radial_unit[0]],
        dtype=np.float64,
    )
    return np.column_stack((radial_unit, tangential_unit))


def _weight_measurement_residual(
    dx: float,
    dy: float,
    *,
    measured_point: Tuple[float, float],
    center: Sequence[float],
    entry: Dict[str, object],
    distance_weight: float,
    use_measurement_uncertainty: bool,
    anisotropic_enabled: bool,
    radial_scale: float,
    tangential_scale: float,
) -> Dict[str, float | bool]:
    """Apply scalar or covariance-style measurement weighting to one match."""

    sigma_px, sigma_radial, sigma_tangential, has_custom_sigma, anisotropic_used = (
        _measured_entry_sigma_components(
            entry,
            anisotropic_enabled=anisotropic_enabled,
            radial_scale=radial_scale,
            tangential_scale=tangential_scale,
        )
    )
    residual_vec = np.array([float(dx), float(dy)], dtype=np.float64)
    radial_component = float("nan")
    tangential_component = float("nan")
    weighted_radial = float("nan")
    weighted_tangential = float("nan")
    effective_distance_weight = float(distance_weight)

    if not bool(use_measurement_uncertainty):
        reported_sigma_px = float(
            entry.get(
                "reported_measurement_sigma_px",
                entry.get("measurement_sigma_px", sigma_px),
            )
        )
        if not np.isfinite(reported_sigma_px) or reported_sigma_px <= 0.0:
            reported_sigma_px = float(sigma_px)
        basis = _radial_tangential_basis(measured_point, center)
        weighted_vec = effective_distance_weight * residual_vec
        if basis is not None:
            radial_unit = basis[:, 0]
            tangential_unit = basis[:, 1]
            radial_component = float(np.dot(radial_unit, residual_vec))
            tangential_component = float(np.dot(tangential_unit, residual_vec))
            weighted_radial = float(effective_distance_weight * radial_component)
            weighted_tangential = float(effective_distance_weight * tangential_component)
        else:
            weighted_radial = float(weighted_vec[0])
            weighted_tangential = float(weighted_vec[1])
        return {
            "measurement_sigma_px": float(reported_sigma_px),
            "sigma_radial_px": float(sigma_radial),
            "sigma_tangential_px": float(sigma_tangential),
            "sigma_is_custom": False,
            "anisotropic_sigma_used": False,
            "sigma_weight": 1.0,
            "weighted_dx_px": float(weighted_vec[0]),
            "weighted_dy_px": float(weighted_vec[1]),
            "radial_residual_px": float(radial_component),
            "tangential_residual_px": float(tangential_component),
            "weighted_radial_residual_px": float(weighted_radial),
            "weighted_tangential_residual_px": float(weighted_tangential),
        }

    basis = _radial_tangential_basis(measured_point, center)
    if anisotropic_used and basis is not None:
        radial_unit = basis[:, 0]
        tangential_unit = basis[:, 1]
        radial_component = float(np.dot(radial_unit, residual_vec))
        tangential_component = float(np.dot(tangential_unit, residual_vec))
        weighted_radial = float(distance_weight * radial_component / sigma_radial)
        weighted_tangential = float(distance_weight * tangential_component / sigma_tangential)
        inv_sqrt_cov = basis @ np.diag([1.0 / sigma_radial, 1.0 / sigma_tangential]) @ basis.T
        weighted_vec = float(distance_weight) * (inv_sqrt_cov @ residual_vec)
    else:
        scalar_sigma_weight = 1.0 / float(sigma_px)
        weighted_vec = float(distance_weight * scalar_sigma_weight) * residual_vec
        if basis is not None:
            radial_unit = basis[:, 0]
            tangential_unit = basis[:, 1]
            radial_component = float(np.dot(radial_unit, residual_vec))
            tangential_component = float(np.dot(tangential_unit, residual_vec))
            weighted_radial = float(distance_weight * scalar_sigma_weight * radial_component)
            weighted_tangential = float(
                distance_weight * scalar_sigma_weight * tangential_component
            )
        else:
            weighted_radial = float(distance_weight * residual_vec[0] / sigma_px)
            weighted_tangential = float(distance_weight * residual_vec[1] / sigma_px)

    sigma_weight_equivalent = 1.0 / math.sqrt(
        max(float(sigma_radial) * float(sigma_tangential), 1.0e-12)
    )
    return {
        "measurement_sigma_px": float(sigma_px),
        "sigma_radial_px": float(sigma_radial),
        "sigma_tangential_px": float(sigma_tangential),
        "sigma_is_custom": bool(has_custom_sigma),
        "anisotropic_sigma_used": bool(anisotropic_used),
        "sigma_weight": float(sigma_weight_equivalent),
        "weighted_dx_px": float(weighted_vec[0]),
        "weighted_dy_px": float(weighted_vec[1]),
        "radial_residual_px": float(radial_component),
        "tangential_residual_px": float(tangential_component),
        "weighted_radial_residual_px": float(weighted_radial),
        "weighted_tangential_residual_px": float(weighted_tangential),
    }


@dataclass
class ReflectionSimulationSubset:
    """Reduced reflection list and remapped measured entries for fitting."""

    miller: np.ndarray
    intensities: np.ndarray
    measured_entries: List[Dict[str, object]]
    original_indices: np.ndarray
    total_reflection_count: int
    fixed_source_reflection_count: int
    fallback_hkl_count: int
    reduced: bool


@dataclass
class GeometryFitDatasetContext:
    """One measured-peak dataset used in a geometry fit."""

    dataset_index: int
    label: str
    theta_initial: float
    subset: ReflectionSimulationSubset
    experimental_image: Optional[np.ndarray] = None
    single_ray_indices: Optional[np.ndarray] = None
    dynamic_reanchor_enabled: bool = False
    dynamic_reanchor_callback: Optional[Callable[..., Mapping[str, object] | None]] = None


@dataclass(frozen=True)
class ParamSpec:
    """Normalized local parameter specification for one fit variable."""

    name: str
    ref_value: float
    lower: float
    upper: float
    scale: float
    prior_mean: float | None = None
    prior_sigma: float | None = None
    seed_group: str = "auto"


def _default_scale(
    value: float,
    lower: float,
    upper: float,
    *,
    prior_sigma: float | None = None,
    nominal_scale: float | None = None,
) -> float:
    """Choose one local normalized-variable scale."""

    if prior_sigma is not None and np.isfinite(float(prior_sigma)) and float(prior_sigma) > 0.0:
        return float(prior_sigma)
    span = float(upper) - float(lower)
    if np.isfinite(span) and span > 0.0:
        return max(0.25 * float(span), 1.0e-12)
    if nominal_scale is not None and np.isfinite(float(nominal_scale)):
        if float(nominal_scale) > 0.0:
            return float(nominal_scale)
    return max(abs(float(value)), 1.0, 1.0e-12)


def _build_u_bounds(specs: Sequence[ParamSpec]) -> tuple[np.ndarray, np.ndarray]:
    """Convert physical bounds into normalized ``u`` bounds."""

    lb = np.array(
        [(float(spec.lower) - float(spec.ref_value)) / float(spec.scale) for spec in specs],
        dtype=float,
    )
    ub = np.array(
        [(float(spec.upper) - float(spec.ref_value)) / float(spec.scale) for spec in specs],
        dtype=float,
    )
    return lb, ub


def _physical_from_u(specs: Sequence[ParamSpec], u: Sequence[float]) -> np.ndarray:
    """Map normalized local coordinates back into physical parameter values."""

    u_arr = np.asarray(u, dtype=float).reshape(-1)
    values = np.empty(len(specs), dtype=float)
    for idx, spec in enumerate(specs):
        uj = float(u_arr[idx]) if idx < u_arr.size else 0.0
        values[idx] = float(spec.ref_value + spec.scale * uj)
    return values


def _u_from_physical(
    specs: Sequence[ParamSpec],
    theta: Sequence[float],
) -> np.ndarray:
    """Map physical parameter values into normalized local coordinates."""

    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    values = np.empty(len(specs), dtype=float)
    for idx, spec in enumerate(specs):
        theta_j = float(theta_arr[idx]) if idx < theta_arr.size else float(spec.ref_value)
        values[idx] = float((theta_j - float(spec.ref_value)) / float(spec.scale))
    return values


def _matrix_rank_from_singular_values(
    singular_values: Sequence[float],
    *,
    rcond: float,
) -> int:
    """Return the numerical rank implied by one singular spectrum."""

    s = np.asarray(singular_values, dtype=float).reshape(-1)
    if s.size == 0 or not np.isfinite(s[0]) or s[0] <= 0.0:
        return 0
    threshold = max(float(rcond), 0.0) * float(s[0])
    return int(np.count_nonzero(s > threshold))


def _min_nonzero_singular_value(
    singular_values: Sequence[float],
    *,
    rcond: float,
) -> float:
    """Return the smallest non-negligible singular value or zero."""

    s = np.asarray(singular_values, dtype=float).reshape(-1)
    if s.size == 0 or not np.isfinite(s[0]) or s[0] <= 0.0:
        return 0.0
    threshold = max(float(rcond), 0.0) * float(s[0])
    usable = s[np.isfinite(s) & (s > threshold)]
    if usable.size == 0:
        return 0.0
    return float(np.min(usable))


def _finite_diff_jacobian_abs(
    fun: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    abs_step: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> np.ndarray:
    """Central finite-difference Jacobian with absolute steps and bound guards."""

    x_ref = np.asarray(x, dtype=float).reshape(-1)
    step_arr = np.asarray(abs_step, dtype=float).reshape(-1)
    f0 = np.asarray(fun(np.asarray(x_ref, dtype=float)), dtype=float).reshape(-1)
    J = np.empty((f0.size, x_ref.size), dtype=float)
    lower_arr = None if lower is None else np.asarray(lower, dtype=float).reshape(-1)
    upper_arr = None if upper is None else np.asarray(upper, dtype=float).reshape(-1)

    for j in range(x_ref.size):
        h = float(step_arr[j]) if j < step_arr.size else 0.0
        if not np.isfinite(h) or h <= 0.0:
            h = 1.0e-6
        xp = np.asarray(x_ref, dtype=float).copy()
        xm = np.asarray(x_ref, dtype=float).copy()
        if lower_arr is not None and j < lower_arr.size:
            xm[j] = max(float(lower_arr[j]), float(xm[j] - h))
        else:
            xm[j] = float(xm[j] - h)
        if upper_arr is not None and j < upper_arr.size:
            xp[j] = min(float(upper_arr[j]), float(xp[j] + h))
        else:
            xp[j] = float(xp[j] + h)

        delta_plus = float(xp[j] - x_ref[j])
        delta_minus = float(x_ref[j] - xm[j])
        if delta_plus > 0.0 and delta_minus > 0.0:
            fp = np.asarray(fun(xp), dtype=float).reshape(-1)
            fm = np.asarray(fun(xm), dtype=float).reshape(-1)
            J[:, j] = (fp - fm) / float(delta_plus + delta_minus)
        elif delta_plus > 0.0:
            fp = np.asarray(fun(xp), dtype=float).reshape(-1)
            J[:, j] = (fp - f0) / float(delta_plus)
        elif delta_minus > 0.0:
            fm = np.asarray(fun(xm), dtype=float).reshape(-1)
            J[:, j] = (f0 - fm) / float(delta_minus)
        else:
            J[:, j] = 0.0
    return J


def _soft_l1_row_scale(f: np.ndarray, f_scale: float) -> np.ndarray:
    """Return the row multiplier that matches SciPy's scalar ``soft_l1`` loss."""

    f_arr = np.asarray(f, dtype=float)
    scale = max(float(f_scale), 1.0e-12)
    z = (f_arr / scale) ** 2
    return np.power(1.0 + z, -0.25)


def _low_discrepancy_points(count: int, ndim: int) -> np.ndarray:
    """Generate a simple deterministic low-discrepancy point set on ``[0, 1]^d``."""

    n = max(int(count), 0)
    d = max(int(ndim), 0)
    if n <= 0 or d <= 0:
        return np.empty((0, d), dtype=float)
    irrational = np.modf(np.sqrt(np.arange(2, d + 2, dtype=float)))[0]
    irrational = np.where(np.abs(irrational) > 1.0e-12, irrational, 0.5)
    pts = np.empty((n, d), dtype=float)
    for idx in range(n):
        pts[idx, :] = np.mod(0.5 + (idx + 1) * irrational, 1.0)
    return pts


def _map_box_points(
    points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Affine-map unit-box samples into arbitrary axis-aligned bounds."""

    pts = np.asarray(points, dtype=float)
    lb = np.asarray(lower, dtype=float).reshape(-1)
    ub = np.asarray(upper, dtype=float).reshape(-1)
    if pts.size == 0 or lb.size == 0:
        return np.empty((0, lb.size), dtype=float)
    return lb[None, :] + pts * (ub - lb)[None, :]


def _deduplicate_seed_vectors(
    seeds: Sequence[Sequence[float]],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    min_dist: float,
) -> List[np.ndarray]:
    """Drop near-duplicate seed vectors in normalized space."""

    lb = np.asarray(lower, dtype=float).reshape(-1)
    ub = np.asarray(upper, dtype=float).reshape(-1)
    threshold = max(float(min_dist), 0.0)
    unique: List[np.ndarray] = []
    seen_exact: Set[Tuple[float, ...]] = set()
    for raw_seed in seeds:
        seed = np.asarray(raw_seed, dtype=float).reshape(-1)
        if seed.shape != lb.shape:
            continue
        seed = np.minimum(np.maximum(seed, lb), ub)
        if not np.all(np.isfinite(seed)):
            continue
        key = tuple(np.round(seed, 12).tolist())
        if key in seen_exact:
            continue
        if threshold > 0.0 and any(
            float(np.linalg.norm(seed - existing)) < threshold for existing in unique
        ):
            continue
        seen_exact.add(key)
        unique.append(seed)
    return unique


def _format_discrete_mode_label(mode: Mapping[str, object] | None) -> str:
    """Return one compact user-facing label for a solver-side discrete mode."""

    mode_dict = dict(mode or {})
    if not mode_dict:
        return "identity"
    label = str(mode_dict.get("label", "") or "").strip()
    if label:
        return label
    pieces: List[str] = []
    k = int(mode_dict.get("k", 0)) % 4
    if k:
        pieces.append(f"rot{90 * k}")
    if bool(mode_dict.get("flip_x", False)):
        pieces.append("flip_x")
    if bool(mode_dict.get("flip_y", False)):
        pieces.append("flip_y")
    return " + ".join(pieces) if pieces else "identity"


def _iter_discrete_mode_specs(
    config: Mapping[str, object] | None,
) -> List[Dict[str, object]]:
    """Enumerate solver-side discrete symmetry modes."""

    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return [
            {
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "indexing_mode": "xy",
                "label": "identity",
            }
        ]

    rot90_values = cfg.get("rot90", [0])
    flip_x_values = cfg.get("flip_x", [False])
    flip_y_values = cfg.get("flip_y", [False])
    try:
        rot90_list = [int(v) % 4 for v in list(rot90_values)]
    except Exception:
        rot90_list = [0]
    try:
        flip_x_list = [bool(v) for v in list(flip_x_values)]
    except Exception:
        flip_x_list = [False]
    try:
        flip_y_list = [bool(v) for v in list(flip_y_values)]
    except Exception:
        flip_y_list = [False]
    if not rot90_list:
        rot90_list = [0]
    if not flip_x_list:
        flip_x_list = [False]
    if not flip_y_list:
        flip_y_list = [False]

    seen: Set[Tuple[int, bool, bool]] = set()
    modes: List[Dict[str, object]] = []
    for k in rot90_list:
        for flip_x in flip_x_list:
            for flip_y in flip_y_list:
                key = (int(k) % 4, bool(flip_x), bool(flip_y))
                if key in seen:
                    continue
                seen.add(key)
                mode = {
                    "k": int(k) % 4,
                    "flip_x": bool(flip_x),
                    "flip_y": bool(flip_y),
                    "flip_order": "yx",
                    "indexing_mode": "xy",
                }
                mode["label"] = _format_discrete_mode_label(mode)
                modes.append(mode)
    if not modes:
        modes.append(
            {
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "indexing_mode": "xy",
                "label": "identity",
            }
        )
    return modes


def _rotate_point_orientation(
    col: float,
    row: float,
    shape: tuple[int, int],
    k: int,
) -> tuple[float, float]:
    """Rotate one point using the same convention as ``np.rot90``."""

    height, width = int(shape[0]), int(shape[1])
    col_new = float(col)
    row_new = float(row)
    for _ in range(int(k) % 4):
        row_new, col_new, height, width = (
            width - 1.0 - col_new,
            row_new,
            width,
            height,
        )
    return float(col_new), float(row_new)


def _transform_points_orientation_local(
    points: Sequence[tuple[float, float]],
    shape: tuple[int, int],
    *,
    indexing_mode: str = "xy",
    k: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_order: str = "yx",
) -> list[tuple[float, float]]:
    """Apply the discrete orientation transform to point coordinates."""

    base_height, base_width = int(shape[0]), int(shape[1])
    mode = str(indexing_mode or "xy").lower()
    if mode == "yx":
        height, width = base_width, base_height
    else:
        height, width = base_height, base_width

    order = str(flip_order or "yx").lower()
    transformed: list[tuple[float, float]] = []

    def _flip_xy(col_t: float, row_t: float) -> tuple[float, float]:
        if flip_x:
            col_t = width - 1.0 - col_t
        if flip_y:
            row_t = height - 1.0 - row_t
        return float(col_t), float(row_t)

    def _flip_yx(col_t: float, row_t: float) -> tuple[float, float]:
        if flip_y:
            row_t = height - 1.0 - row_t
        if flip_x:
            col_t = width - 1.0 - col_t
        return float(col_t), float(row_t)

    flipper = _flip_xy if order == "xy" else _flip_yx
    for col, row in points:
        col_t = float(col)
        row_t = float(row)
        if mode == "yx":
            col_t, row_t = row_t, col_t
        col_t, row_t = flipper(col_t, row_t)
        transformed.append(_rotate_point_orientation(col_t, row_t, (height, width), int(k)))
    return transformed


def _apply_discrete_mode_to_entries(
    entries: Sequence[object],
    shape: tuple[int, int],
    *,
    mode: Mapping[str, object] | None,
) -> List[Dict[str, object]]:
    """Apply one solver-side discrete mode to measured-peak entry coordinates."""

    mode_dict = dict(mode or {})
    k = int(mode_dict.get("k", 0)) % 4
    flip_x = bool(mode_dict.get("flip_x", False))
    flip_y = bool(mode_dict.get("flip_y", False))
    flip_order = str(mode_dict.get("flip_order", "yx"))
    indexing_mode = str(mode_dict.get("indexing_mode", "xy"))
    if k == 0 and not flip_x and not flip_y and indexing_mode == "xy":
        return [
            dict(entry) if isinstance(entry, Mapping) else dict(entry)  # type: ignore[arg-type]
            for entry in entries
            if isinstance(entry, Mapping)
        ]

    def _apply_pair(x_val: float, y_val: float) -> tuple[float, float]:
        return _transform_points_orientation_local(
            [(float(x_val), float(y_val))],
            shape,
            indexing_mode=indexing_mode,
            k=k,
            flip_x=flip_x,
            flip_y=flip_y,
            flip_order=flip_order,
        )[0]

    transformed: List[Dict[str, object]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        for x_key, y_key in (
            ("x", "y"),
            ("detector_x", "detector_y"),
            ("x_pix", "y_pix"),
            ("raw_x", "raw_y"),
        ):
            if x_key not in entry or y_key not in entry:
                continue
            try:
                x_val = float(entry.get(x_key))
                y_val = float(entry.get(y_key))
            except Exception:
                continue
            if not (np.isfinite(x_val) and np.isfinite(y_val)):
                continue
            entry[x_key], entry[y_key] = _apply_pair(x_val, y_val)
        transformed.append(entry)
    return transformed


def _apply_discrete_mode_to_image(
    image: np.ndarray | None,
    *,
    mode: Mapping[str, object] | None,
) -> np.ndarray | None:
    """Apply one solver-side discrete mode to an experimental image."""

    if image is None:
        return None
    mode_dict = dict(mode or {})
    oriented = np.asarray(image)
    if str(mode_dict.get("indexing_mode", "xy")).lower() == "yx":
        oriented = np.swapaxes(oriented, 0, 1)
    order = str(mode_dict.get("flip_order", "yx")).lower()
    if order == "xy":
        if bool(mode_dict.get("flip_x", False)):
            oriented = np.flip(oriented, axis=1)
        if bool(mode_dict.get("flip_y", False)):
            oriented = np.flip(oriented, axis=0)
    else:
        if bool(mode_dict.get("flip_y", False)):
            oriented = np.flip(oriented, axis=0)
        if bool(mode_dict.get("flip_x", False)):
            oriented = np.flip(oriented, axis=1)
    k = int(mode_dict.get("k", 0)) % 4
    if k:
        oriented = np.rot90(oriented, k)
    return np.asarray(oriented, dtype=np.float64)


def _apply_discrete_mode_to_dataset_contexts(
    dataset_contexts: Sequence[GeometryFitDatasetContext],
    *,
    image_size: int,
    mode: Mapping[str, object] | None,
) -> List[GeometryFitDatasetContext]:
    """Copy dataset contexts with one solver-side discrete orientation applied."""

    shape = (int(image_size), int(image_size))
    mode_dict = dict(mode or {})
    preserve_dynamic_reanchor = bool(
        int(mode_dict.get("k", 0)) % 4 == 0
        and not bool(mode_dict.get("flip_x", False))
        and not bool(mode_dict.get("flip_y", False))
        and str(mode_dict.get("indexing_mode", "xy")).lower() == "xy"
    )
    out: List[GeometryFitDatasetContext] = []
    for dataset_ctx in dataset_contexts:
        subset = dataset_ctx.subset
        transformed_entries = _apply_discrete_mode_to_entries(
            subset.measured_entries,
            shape,
            mode=mode,
        )
        transformed_subset = ReflectionSimulationSubset(
            miller=np.asarray(subset.miller, dtype=np.float64),
            intensities=np.asarray(subset.intensities, dtype=np.float64),
            measured_entries=transformed_entries,
            original_indices=np.asarray(subset.original_indices, dtype=np.int64),
            total_reflection_count=int(subset.total_reflection_count),
            fixed_source_reflection_count=int(subset.fixed_source_reflection_count),
            fallback_hkl_count=int(subset.fallback_hkl_count),
            reduced=bool(subset.reduced),
        )
        out.append(
            GeometryFitDatasetContext(
                dataset_index=int(dataset_ctx.dataset_index),
                label=str(dataset_ctx.label),
                theta_initial=float(dataset_ctx.theta_initial),
                subset=transformed_subset,
                experimental_image=_apply_discrete_mode_to_image(
                    dataset_ctx.experimental_image,
                    mode=mode,
                ),
                single_ray_indices=(
                    None
                    if dataset_ctx.single_ray_indices is None
                    else np.asarray(dataset_ctx.single_ray_indices, dtype=np.int64)
                ),
                dynamic_reanchor_enabled=bool(
                    preserve_dynamic_reanchor
                    and dataset_ctx.dynamic_reanchor_enabled
                    and callable(dataset_ctx.dynamic_reanchor_callback)
                ),
                dynamic_reanchor_callback=(
                    dataset_ctx.dynamic_reanchor_callback
                    if preserve_dynamic_reanchor and callable(dataset_ctx.dynamic_reanchor_callback)
                    else None
                ),
            )
        )
    return out


def _svd_summary(
    J: np.ndarray,
    param_names: Sequence[str],
    *,
    scales: Sequence[float] | None = None,
    rcond: float = 1.0e-8,
    weak_singular_rel: float = 1.0e-3,
    combo_thresh: float = 0.2,
) -> Dict[str, object]:
    """Summarize one Jacobian with singular values and weak parameter combinations."""

    jacobian = np.asarray(J, dtype=float)
    names = [str(name) for name in param_names]
    scale_arr = None if scales is None else np.asarray(scales, dtype=float).reshape(-1)
    if jacobian.ndim != 2:
        jacobian = np.empty((0, len(names)), dtype=float)
    if jacobian.shape[1] != len(names):
        jacobian = np.empty((jacobian.shape[0], len(names)), dtype=float)
    summary: Dict[str, object] = {
        "status": "ok",
        "num_parameters": int(len(names)),
        "num_residuals": int(jacobian.shape[0]),
        "rank": 0,
        "condition_number": float("inf"),
        "singular_values": [],
        "relative_singular_values": [],
        "parameter_entries": [],
        "covariance_u": [],
        "correlation_u": [],
        "covariance_theta": [],
        "weak_combinations": [],
        "underconstrained": False,
        "warning_flags": [],
    }
    if jacobian.size == 0 or jacobian.shape[1] == 0:
        summary["status"] = "failed"
        summary["reason"] = "empty_jacobian"
        return summary

    try:
        _U, s, Vt = np.linalg.svd(jacobian, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        summary["status"] = "failed"
        summary["reason"] = f"svd_failed: {exc}"
        return summary

    s = np.asarray(s, dtype=float).reshape(-1)
    rel_s = (
        (s / float(s[0])).tolist()
        if s.size and np.isfinite(s[0]) and s[0] > 0.0
        else [0.0 for _ in s]
    )
    rank = _matrix_rank_from_singular_values(s, rcond=rcond)
    min_nonzero_sv = _min_nonzero_singular_value(s, rcond=rcond)
    max_sv = float(s[0]) if s.size else 0.0
    if max_sv > 0.0 and min_nonzero_sv > 0.0:
        condition_number = float(max_sv / min_nonzero_sv)
    else:
        condition_number = float("inf")

    JTJ = jacobian.T @ jacobian
    try:
        covariance_u = np.linalg.pinv(JTJ, rcond=rcond)
    except TypeError:
        covariance_u = np.linalg.pinv(JTJ)
    diag_u = np.sqrt(np.clip(np.diag(covariance_u), 0.0, None))
    denom = np.outer(diag_u, diag_u)
    correlation_u = np.divide(
        covariance_u,
        denom,
        out=np.zeros_like(covariance_u),
        where=denom > 0.0,
    )
    correlation_u[~np.isfinite(correlation_u)] = 0.0

    covariance_theta = np.asarray(covariance_u, dtype=float)
    if scale_arr is not None and scale_arr.size == len(names):
        D = np.diag(scale_arr)
        covariance_theta = D @ covariance_u @ D

    column_norms = np.linalg.norm(jacobian, axis=0)
    total_norm = float(np.sum(column_norms))
    parameter_entries: List[Dict[str, object]] = []
    for idx, name in enumerate(names):
        column_norm = float(column_norms[idx]) if idx < column_norms.size else 0.0
        parameter_entries.append(
            {
                "name": str(name),
                "index": int(idx),
                "valid": bool(np.all(np.isfinite(jacobian[:, idx]))),
                "column_norm": float(column_norm),
                "relative_sensitivity": (
                    float(column_norm / total_norm) if total_norm > 1.0e-12 else 0.0
                ),
                "std_u": float(diag_u[idx]) if idx < diag_u.size else float("nan"),
                "std_theta": (
                    float(math.sqrt(max(float(covariance_theta[idx, idx]), 0.0)))
                    if idx < covariance_theta.shape[0]
                    else float("nan")
                ),
            }
        )

    weak_combinations: List[Dict[str, object]] = []
    if s.size and np.isfinite(max_sv) and max_sv > 0.0:
        threshold_rel = max(float(weak_singular_rel), 0.0)
        for idx, sv in enumerate(s.tolist()):
            rel = float(sv / max_sv) if max_sv > 0.0 else 0.0
            if rel >= threshold_rel:
                continue
            combo = {
                name: float(weight)
                for name, weight in zip(names, Vt[idx].tolist())
                if abs(float(weight)) >= float(combo_thresh)
            }
            weak_combinations.append(
                {
                    "sv": float(sv),
                    "sv_rel": float(rel),
                    "combo": combo,
                }
            )

    warning_flags: List[str] = []
    underconstrained = bool(
        rank < len(names) or not np.isfinite(condition_number) or jacobian.shape[0] < len(names)
    )
    if underconstrained:
        warning_flags.append("underconstrained")
    if weak_combinations:
        warning_flags.append("weak_combinations")

    summary.update(
        {
            "rank": int(rank),
            "condition_number": float(condition_number),
            "singular_values": s.tolist(),
            "relative_singular_values": rel_s,
            "parameter_entries": parameter_entries,
            "covariance_u": covariance_u.tolist(),
            "correlation_u": correlation_u.tolist(),
            "covariance_theta": covariance_theta.tolist(),
            "weak_combinations": weak_combinations,
            "underconstrained": bool(underconstrained),
            "warning_flags": warning_flags,
        }
    )
    return summary


def recommend_next_stage(
    J_all: np.ndarray,
    *,
    param_names: Sequence[str],
    active_names: Sequence[str],
    rcond: float = 1.0e-8,
    block_corr_abs: float = 0.90,
    weak_column_rel: float = 0.10,
    single_release_min_indep: float = 0.35,
    block_release_min_indep: float = 0.20,
) -> List[Dict[str, object]]:
    """Recommend the next inactive parameter block to thaw from one data-only Jacobian."""

    jacobian = np.asarray(J_all, dtype=float)
    names = [str(name) for name in param_names]
    active_name_set = {str(name) for name in active_names}
    if jacobian.ndim != 2 or jacobian.shape[1] != len(names) or not names:
        return []

    active_indices = [idx for idx, name in enumerate(names) if name in active_name_set]
    inactive_indices = [idx for idx, name in enumerate(names) if name not in active_name_set]
    if not inactive_indices:
        return []

    J_A = jacobian[:, active_indices] if active_indices else np.empty((jacobian.shape[0], 0))
    J_I = jacobian[:, inactive_indices]
    if J_A.size:
        P_A = J_A @ np.linalg.pinv(J_A, rcond=rcond)
    else:
        P_A = np.zeros((jacobian.shape[0], jacobian.shape[0]), dtype=float)
    identity = np.eye(jacobian.shape[0], dtype=float)
    residual_projector = identity - P_A

    inactive_norms = np.linalg.norm(J_I, axis=0)
    nonzero_norms = inactive_norms[inactive_norms > 0.0]
    median_norm = float(np.median(nonzero_norms)) if nonzero_norms.size else 0.0
    weak_threshold = float(max(float(weak_column_rel), 0.0) * median_norm)

    inactive_corr = np.eye(len(inactive_indices), dtype=float)
    for i_local in range(len(inactive_indices)):
        col_i = J_I[:, i_local]
        norm_i = float(np.linalg.norm(col_i))
        if norm_i <= 1.0e-12:
            continue
        for j_local in range(i_local + 1, len(inactive_indices)):
            col_j = J_I[:, j_local]
            norm_j = float(np.linalg.norm(col_j))
            if norm_j <= 1.0e-12:
                continue
            corr = float(np.dot(col_i, col_j) / (norm_i * norm_j))
            if np.isfinite(corr):
                inactive_corr[i_local, j_local] = corr
                inactive_corr[j_local, i_local] = corr

    adjacency: Dict[int, Set[int]] = {
        local_idx: {local_idx} for local_idx in range(len(inactive_indices))
    }
    for i_local in range(len(inactive_indices)):
        for j_local in range(i_local + 1, len(inactive_indices)):
            if abs(float(inactive_corr[i_local, j_local])) >= float(block_corr_abs):
                adjacency[i_local].add(j_local)
                adjacency[j_local].add(i_local)

    blocks: List[List[int]] = []
    visited: Set[int] = set()
    for start in range(len(inactive_indices)):
        if start in visited:
            continue
        stack = [start]
        block: List[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            block.append(current)
            stack.extend(sorted(adjacency.get(current, set()) - visited))
        blocks.append(sorted(block))

    base_singular_values = (
        np.linalg.svd(J_A, full_matrices=False, compute_uv=False)
        if J_A.size
        else np.array([], dtype=float)
    )
    base_rank = _matrix_rank_from_singular_values(base_singular_values, rcond=rcond)
    base_min_sv = _min_nonzero_singular_value(base_singular_values, rcond=rcond)
    eps = 1.0e-12

    recommendations: List[Dict[str, object]] = []
    for block_local in blocks:
        block_global = [inactive_indices[idx] for idx in block_local]
        block_names = [names[idx] for idx in block_global]
        J_block = jacobian[:, block_global]
        block_norm = float(np.linalg.norm(J_block))
        if block_norm <= 1.0e-12:
            continue
        block_independence = float(
            np.linalg.norm(residual_projector @ J_block) / (block_norm + eps)
        )
        block_singular_values = np.linalg.svd(
            np.hstack([J_A, J_block]) if J_A.size else J_block,
            full_matrices=False,
            compute_uv=False,
        )
        rank_gain = int(
            _matrix_rank_from_singular_values(block_singular_values, rcond=rcond) - base_rank
        )
        min_sv_gain = float(
            _min_nonzero_singular_value(block_singular_values, rcond=rcond) / (base_min_sv + eps)
        )
        weak_block = bool(
            all(
                float(inactive_norms[idx_local]) <= weak_threshold + eps
                for idx_local in block_local
            )
        )
        if len(block_global) == 1:
            idx_local = block_local[0]
            indep_j = block_independence
            if weak_block or indep_j < float(single_release_min_indep):
                continue
            recommendations.append(
                {
                    "params": block_names,
                    "reason": "Adds an independent data direction beyond the current active set.",
                    "rank_gain": int(rank_gain),
                    "min_sv_gain": float(min_sv_gain),
                    "block_independence": float(block_independence),
                    "use_soft_prior": True,
                    "weak": bool(weak_block),
                }
            )
            continue

        if weak_block or block_independence < float(block_release_min_indep):
            continue
        recommendations.append(
            {
                "params": block_names,
                "reason": "Correlated block. Good incremental information only when thawed together.",
                "rank_gain": int(rank_gain),
                "min_sv_gain": float(min_sv_gain),
                "block_independence": float(block_independence),
                "use_soft_prior": True,
                "weak": bool(weak_block),
            }
        )

    recommendations.sort(
        key=lambda item: (
            -int(item.get("rank_gain", 0)),
            -float(item.get("min_sv_gain", 0.0)),
            -float(item.get("block_independence", 0.0)),
            tuple(str(name) for name in item.get("params", [])),
        )
    )
    return recommendations


_miller_key_from_row = _mosaic_profiles._miller_key_from_row
_normalized_q_group_key = _mosaic_profiles._normalized_q_group_key
_normalized_hkl_key = _mosaic_profiles._normalized_hkl_key
_mosaic_profile_group_key = _mosaic_profiles._mosaic_profile_group_key
_mosaic_profile_intensity_lookup = _mosaic_profiles._mosaic_profile_intensity_lookup
_mosaic_profile_peak_score = _mosaic_profiles._mosaic_profile_peak_score
_mosaic_profile_entry_priority = _mosaic_profiles._mosaic_profile_entry_priority


def focus_mosaic_profile_dataset_specs(
    dataset_specs: Sequence[Dict[str, object]],
    *,
    source_miller: np.ndarray,
    source_intensities: np.ndarray,
    reference_dataset_index: object = None,
    max_in_plane_groups: int = 3,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Reduce mosaic-fit datasets to all 00L peaks plus the top in-plane groups."""

    return _mosaic_profiles.focus_mosaic_profile_dataset_specs(
        dataset_specs,
        source_miller=source_miller,
        source_intensities=source_intensities,
        reference_dataset_index=reference_dataset_index,
        max_in_plane_groups=max_in_plane_groups,
        coerce_sequence_items=_coerce_sequence_items,
    )


def _geometry_peak_priority_metadata(
    entry: Mapping[str, object],
    *,
    hk0_peak_priority_weight: float,
) -> Dict[str, object]:
    """Return the objective-priority metadata for one measured peak."""

    priority_weight = 1.0
    priority_class = "default"
    hkl_key = _normalized_hkl_key(entry.get("hkl"))
    if (
        hkl_key is not None
        and hkl_key != (0, 0, 0)
        and hkl_key[0] == 0
        and hkl_key[1] == 0
        and np.isfinite(hk0_peak_priority_weight)
        and hk0_peak_priority_weight > 1.0
    ):
        priority_weight = float(hk0_peak_priority_weight)
        priority_class = "hk0"

    return {
        "priority_weight": float(priority_weight),
        "priority_class": str(priority_class),
        "is_hk0_peak": bool(priority_class == "hk0"),
    }


def _geometry_line_group_ids(
    entry: Mapping[str, object],
) -> Tuple[Tuple[object, ...], ...]:
    """Return the Qr-family line groups that one measured entry belongs to."""

    group_ids: list[tuple[object, ...]] = []
    q_group_key = _normalized_q_group_key(entry.get("q_group_key"))
    if q_group_key is not None:
        group_ids.append(("q_group_line",) + q_group_key)

    hkl_key = _normalized_hkl_key(entry.get("hkl"))
    if hkl_key is not None and hkl_key[0] == 0 and hkl_key[1] == 0:
        source_label = (
            str(q_group_key[1])
            if isinstance(q_group_key, tuple) and len(q_group_key) >= 2
            else str(entry.get("source_label", "primary"))
        )
        group_ids.append(("qz_axis_line", str(source_label)))

    return tuple(group_ids)


def _eligible_geometry_line_group_ids(
    entries: Sequence[Mapping[str, object]],
) -> List[Tuple[object, ...]]:
    """Return stable line groups backed by at least two measured entries."""

    grouped: Dict[Tuple[object, ...], List[int]] = {}
    ordered: List[Tuple[object, ...]] = []
    for idx, entry in enumerate(entries):
        for group_id in _geometry_line_group_ids(entry):
            if group_id not in grouped:
                grouped[group_id] = []
                ordered.append(group_id)
            grouped[group_id].append(int(idx))
    return [group_id for group_id in ordered if len(grouped.get(group_id, ())) >= 2]


def _fit_geometry_line_model(
    points: Sequence[Sequence[float]] | np.ndarray,
) -> Optional[Dict[str, np.ndarray | float]]:
    """Fit one 2D PCA line model and return centroid/direction/span metadata."""

    arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] < 2:
        return None
    finite_mask = np.isfinite(arr).all(axis=1)
    arr = arr[finite_mask]
    if arr.shape[0] < 2:
        return None

    centroid = np.mean(arr, axis=0)
    centered = arr - centroid
    if arr.shape[0] == 2:
        direction = centered[1] - centered[0]
    else:
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        if vh.size < 2:
            return None
        direction = vh[0]

    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm <= 1.0e-12:
        return None
    direction = direction / direction_norm
    if float(direction[0]) < 0.0 or (
        abs(float(direction[0])) <= 1.0e-12 and float(direction[1]) < 0.0
    ):
        direction = -direction

    projections = centered @ direction
    proj_min = float(np.min(projections))
    proj_max = float(np.max(projections))
    if not (np.isfinite(proj_min) and np.isfinite(proj_max)):
        return None
    span = float(proj_max - proj_min)
    if span <= 1.0e-9:
        return None

    return {
        "centroid": np.asarray(centroid, dtype=np.float64),
        "direction": np.asarray(direction, dtype=np.float64),
        "span": float(span),
    }


def _unwrap_fit_space_phi_deg(
    values: Sequence[float] | np.ndarray,
    *,
    reference: float | None = None,
) -> np.ndarray:
    """Unwrap phi values around one local reference without reordering points."""

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return out

    finite_vals = arr[finite_mask]
    ref = (
        float(reference)
        if reference is not None and np.isfinite(float(reference))
        else float(finite_vals[0])
    )
    first_pass = np.array(
        [ref + _angular_difference_deg(float(value), ref) for value in finite_vals],
        dtype=np.float64,
    )
    if first_pass.size:
        ref = float(np.mean(first_pass))
    out[finite_mask] = np.array(
        [ref + _angular_difference_deg(float(value), ref) for value in finite_vals],
        dtype=np.float64,
    )
    return out


def _build_geometry_line_constraint_residuals_from_detector_matches(
    matched_pairs: Sequence[Mapping[str, object]],
    eligible_group_ids: Sequence[Tuple[object, ...]],
    *,
    center: Sequence[float] | None,
    detector_distance: float,
    pixel_size: float,
    gamma_deg: float = 0.0,
    Gamma_deg: float = 0.0,
    angle_weight: float = 1.0,
    offset_weight: float = 1.0,
    missing_penalty_px: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Turn Qr-family line mismatch into extra pixel-scale residuals."""

    fit_space_available = True
    try:
        centre_row = float(center[0]) if center is not None else float("nan")
        centre_col = float(center[1]) if center is not None else float("nan")
    except Exception:
        centre_row = float("nan")
        centre_col = float("nan")
    if not (
        np.isfinite(centre_row)
        and np.isfinite(centre_col)
        and np.isfinite(detector_distance)
        and detector_distance > 0.0
        and np.isfinite(pixel_size)
        and pixel_size > 0.0
    ):
        fit_space_available = False

    residual_components = np.zeros((len(eligible_group_ids), 2), dtype=np.float64)
    summary: Dict[str, object] = {
        "line_group_count": int(len(eligible_group_ids)),
        "resolved_line_group_count": 0,
        "missing_line_group_count": 0,
        "line_angle_rms_px": float("nan"),
        "line_offset_rms_px": float("nan"),
        "line_fit_space_span_deg_mean": float("nan"),
        "line_constraints_enabled": bool(fit_space_available),
    }
    if not eligible_group_ids or not fit_space_available:
        return np.array([], dtype=float), summary

    indexed_groups = {group_id: idx for idx, group_id in enumerate(eligible_group_ids)}
    grouped_pairs: Dict[Tuple[object, ...], List[Mapping[str, object]]] = {
        group_id: [] for group_id in eligible_group_ids
    }
    for pair in matched_pairs:
        raw_group_ids = pair.get("group_ids", ())
        if not isinstance(raw_group_ids, (list, tuple)):
            continue
        for raw_group_id in raw_group_ids:
            if raw_group_id in indexed_groups:
                grouped_pairs[raw_group_id].append(pair)

    resolved_angle_px: List[float] = []
    resolved_offset_px: List[float] = []
    resolved_spans_deg: List[float] = []

    def _assign_missing(slot: int) -> None:
        residual_components[int(slot), 0] = float(angle_weight) * float(missing_penalty_px)
        residual_components[int(slot), 1] = float(offset_weight) * float(missing_penalty_px)
        summary["missing_line_group_count"] = int(summary["missing_line_group_count"]) + 1

    for slot, group_id in enumerate(eligible_group_ids):
        records = grouped_pairs.get(group_id, [])
        if len(records) < 2:
            _assign_missing(slot)
            continue

        meas_points: list[tuple[float, float]] = []
        sim_points: list[tuple[float, float]] = []
        for record in records:
            try:
                measured_point = tuple(record.get("measured_point", (np.nan, np.nan)))
                simulated_point = tuple(record.get("simulated_point", (np.nan, np.nan)))
                measured_xy = (float(measured_point[0]), float(measured_point[1]))
                simulated_xy = (float(simulated_point[0]), float(simulated_point[1]))
            except Exception:
                continue
            if not (
                np.isfinite(measured_xy[0])
                and np.isfinite(measured_xy[1])
                and np.isfinite(simulated_xy[0])
                and np.isfinite(simulated_xy[1])
            ):
                continue
            meas_points.append(measured_xy)
            sim_points.append(simulated_xy)

        if len(meas_points) < 2 or len(sim_points) < 2:
            _assign_missing(slot)
            continue

        meas_cols = np.array([point[0] for point in meas_points], dtype=np.float64)
        meas_rows = np.array([point[1] for point in meas_points], dtype=np.float64)
        sim_cols = np.array([point[0] for point in sim_points], dtype=np.float64)
        sim_rows = np.array([point[1] for point in sim_points], dtype=np.float64)
        meas_two_theta, meas_phi = _detector_pixels_to_fit_space(
            meas_cols,
            meas_rows,
            center=center,
            detector_distance=float(detector_distance),
            pixel_size=float(pixel_size),
            gamma_deg=float(gamma_deg),
            Gamma_deg=float(Gamma_deg),
        )
        sim_two_theta, sim_phi = _detector_pixels_to_fit_space(
            sim_cols,
            sim_rows,
            center=center,
            detector_distance=float(detector_distance),
            pixel_size=float(pixel_size),
            gamma_deg=float(gamma_deg),
            Gamma_deg=float(Gamma_deg),
        )
        valid_mask = (
            np.isfinite(meas_two_theta)
            & np.isfinite(meas_phi)
            & np.isfinite(sim_two_theta)
            & np.isfinite(sim_phi)
        )
        if int(np.count_nonzero(valid_mask)) < 2:
            _assign_missing(slot)
            continue

        meas_cols = meas_cols[valid_mask]
        meas_rows = meas_rows[valid_mask]
        meas_two_theta = meas_two_theta[valid_mask]
        meas_phi = meas_phi[valid_mask]
        sim_two_theta = sim_two_theta[valid_mask]
        sim_phi = sim_phi[valid_mask]
        if meas_two_theta.size < 2:
            _assign_missing(slot)
            continue

        meas_phi_unwrapped = _unwrap_fit_space_phi_deg(meas_phi)
        measured_fit = np.column_stack((meas_two_theta, meas_phi_unwrapped))
        measured_model = _fit_geometry_line_model(measured_fit)
        if measured_model is None:
            _assign_missing(slot)
            continue

        sim_phi_unwrapped = _unwrap_fit_space_phi_deg(
            sim_phi,
            reference=float(np.asarray(measured_model["centroid"], dtype=np.float64)[1]),
        )
        simulated_fit = np.column_stack((sim_two_theta, sim_phi_unwrapped))
        simulated_model = _fit_geometry_line_model(simulated_fit)
        if simulated_model is None:
            _assign_missing(slot)
            continue

        detector_model = _fit_geometry_line_model(np.column_stack((meas_cols, meas_rows)))
        fit_span_deg = float(
            max(
                float(measured_model["span"]),
                float(simulated_model["span"]),
            )
        )
        if not np.isfinite(fit_span_deg) or fit_span_deg <= 1.0e-9:
            _assign_missing(slot)
            continue

        detector_span_px = (
            float(detector_model["span"]) if detector_model is not None else float("nan")
        )
        if not np.isfinite(detector_span_px) or detector_span_px <= 1.0e-9:
            centered_detector = np.column_stack((meas_cols, meas_rows))
            centered_detector = centered_detector - np.mean(centered_detector, axis=0)
            detector_span_px = float(2.0 * np.max(np.linalg.norm(centered_detector, axis=1)))
        if not np.isfinite(detector_span_px) or detector_span_px <= 1.0e-9:
            _assign_missing(slot)
            continue

        px_per_fit_deg = float(detector_span_px / max(fit_span_deg, 1.0e-9))
        measured_direction = np.asarray(measured_model["direction"], dtype=np.float64)
        simulated_direction = np.asarray(simulated_model["direction"], dtype=np.float64)
        dot_abs = float(abs(np.clip(np.dot(measured_direction, simulated_direction), -1.0, 1.0)))
        angle_diff_rad = float(math.acos(dot_abs))
        measured_normal = np.array(
            [-float(measured_direction[1]), float(measured_direction[0])],
            dtype=np.float64,
        )
        centroid_delta = np.asarray(simulated_model["centroid"], dtype=np.float64) - np.asarray(
            measured_model["centroid"], dtype=np.float64
        )
        offset_deg = float(np.dot(centroid_delta, measured_normal))

        angle_residual_px = float(
            abs(math.sin(angle_diff_rad)) * 0.5 * fit_span_deg * px_per_fit_deg
        )
        offset_residual_px = float(abs(offset_deg) * px_per_fit_deg)
        residual_components[slot, 0] = float(angle_weight) * float(angle_residual_px)
        residual_components[slot, 1] = float(offset_weight) * float(offset_residual_px)
        summary["resolved_line_group_count"] = int(summary["resolved_line_group_count"]) + 1
        resolved_angle_px.append(float(angle_residual_px))
        resolved_offset_px.append(float(offset_residual_px))
        resolved_spans_deg.append(float(fit_span_deg))

    if resolved_angle_px:
        angle_arr = np.asarray(resolved_angle_px, dtype=float)
        summary["line_angle_rms_px"] = float(np.sqrt(np.mean(angle_arr * angle_arr)))
    if resolved_offset_px:
        offset_arr = np.asarray(resolved_offset_px, dtype=float)
        summary["line_offset_rms_px"] = float(np.sqrt(np.mean(offset_arr * offset_arr)))
    if resolved_spans_deg:
        span_arr = np.asarray(resolved_spans_deg, dtype=float)
        summary["line_fit_space_span_deg_mean"] = float(np.mean(span_arr))

    return residual_components.reshape(-1), summary


def _build_geometry_line_constraint_residuals_from_fit_space_matches(
    matched_pairs: Sequence[Mapping[str, object]],
    eligible_group_ids: Sequence[Tuple[object, ...]],
    *,
    angle_weight: float = 1.0,
    offset_weight: float = 1.0,
    missing_penalty_deg: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Turn Qr-family line mismatch into extra fit-space residuals."""

    residual_components = np.zeros((len(eligible_group_ids), 2), dtype=np.float64)
    summary: Dict[str, object] = {
        "line_group_count": int(len(eligible_group_ids)),
        "resolved_line_group_count": 0,
        "missing_line_group_count": 0,
        "line_angle_rms_px": float("nan"),
        "line_offset_rms_px": float("nan"),
        "line_angle_rms_deg": float("nan"),
        "line_offset_rms_deg": float("nan"),
        "line_fit_space_span_deg_mean": float("nan"),
        "line_constraints_enabled": True,
    }
    if not eligible_group_ids:
        return np.array([], dtype=float), summary

    indexed_groups = {group_id: idx for idx, group_id in enumerate(eligible_group_ids)}
    grouped_pairs: Dict[Tuple[object, ...], List[Mapping[str, object]]] = {
        group_id: [] for group_id in eligible_group_ids
    }
    for pair in matched_pairs:
        raw_group_ids = pair.get("group_ids", ())
        if not isinstance(raw_group_ids, (list, tuple)):
            continue
        for raw_group_id in raw_group_ids:
            if raw_group_id in indexed_groups:
                grouped_pairs[raw_group_id].append(pair)

    resolved_angle_deg: List[float] = []
    resolved_offset_deg: List[float] = []
    resolved_spans_deg: List[float] = []

    def _assign_missing(slot: int) -> None:
        residual_components[int(slot), 0] = float(angle_weight) * float(missing_penalty_deg)
        residual_components[int(slot), 1] = float(offset_weight) * float(missing_penalty_deg)
        summary["missing_line_group_count"] = int(summary["missing_line_group_count"]) + 1

    for slot, group_id in enumerate(eligible_group_ids):
        records = grouped_pairs.get(group_id, [])
        if len(records) < 2:
            _assign_missing(slot)
            continue

        meas_points: list[tuple[float, float]] = []
        sim_points: list[tuple[float, float]] = []
        for record in records:
            try:
                measured_point = tuple(record.get("measured_fit_point", (np.nan, np.nan)))
                simulated_point = tuple(record.get("simulated_fit_point", (np.nan, np.nan)))
                measured_xy = (float(measured_point[0]), float(measured_point[1]))
                simulated_xy = (float(simulated_point[0]), float(simulated_point[1]))
            except Exception:
                continue
            if not (
                np.isfinite(measured_xy[0])
                and np.isfinite(measured_xy[1])
                and np.isfinite(simulated_xy[0])
                and np.isfinite(simulated_xy[1])
            ):
                continue
            meas_points.append(measured_xy)
            sim_points.append(simulated_xy)

        if len(meas_points) < 2 or len(sim_points) < 2:
            _assign_missing(slot)
            continue

        meas_points_arr = np.asarray(meas_points, dtype=np.float64)
        sim_points_arr = np.asarray(sim_points, dtype=np.float64)
        meas_phi_unwrapped = _unwrap_fit_space_phi_deg(meas_points_arr[:, 1])
        measured_fit = np.column_stack((meas_points_arr[:, 0], meas_phi_unwrapped))
        measured_model = _fit_geometry_line_model(measured_fit)
        if measured_model is None:
            _assign_missing(slot)
            continue

        sim_phi_unwrapped = _unwrap_fit_space_phi_deg(
            sim_points_arr[:, 1],
            reference=float(np.asarray(measured_model["centroid"], dtype=np.float64)[1]),
        )
        simulated_fit = np.column_stack((sim_points_arr[:, 0], sim_phi_unwrapped))
        simulated_model = _fit_geometry_line_model(simulated_fit)
        if simulated_model is None:
            _assign_missing(slot)
            continue

        fit_span_deg = float(
            max(
                float(measured_model["span"]),
                float(simulated_model["span"]),
            )
        )
        if not np.isfinite(fit_span_deg) or fit_span_deg <= 1.0e-9:
            _assign_missing(slot)
            continue

        measured_direction = np.asarray(measured_model["direction"], dtype=np.float64)
        simulated_direction = np.asarray(simulated_model["direction"], dtype=np.float64)
        dot_abs = float(abs(np.clip(np.dot(measured_direction, simulated_direction), -1.0, 1.0)))
        angle_diff_rad = float(math.acos(dot_abs))
        measured_normal = np.array(
            [-float(measured_direction[1]), float(measured_direction[0])],
            dtype=np.float64,
        )
        centroid_delta = np.asarray(simulated_model["centroid"], dtype=np.float64) - np.asarray(
            measured_model["centroid"], dtype=np.float64
        )
        offset_deg = float(np.dot(centroid_delta, measured_normal))

        angle_residual_deg = float(abs(math.sin(angle_diff_rad)) * 0.5 * fit_span_deg)
        offset_residual_deg = float(abs(offset_deg))
        residual_components[slot, 0] = float(angle_weight) * float(angle_residual_deg)
        residual_components[slot, 1] = float(offset_weight) * float(offset_residual_deg)
        summary["resolved_line_group_count"] = int(summary["resolved_line_group_count"]) + 1
        resolved_angle_deg.append(float(angle_residual_deg))
        resolved_offset_deg.append(float(offset_residual_deg))
        resolved_spans_deg.append(float(fit_span_deg))

    if resolved_angle_deg:
        angle_arr = np.asarray(resolved_angle_deg, dtype=float)
        angle_rms = float(np.sqrt(np.mean(angle_arr * angle_arr)))
        summary["line_angle_rms_deg"] = float(angle_rms)
        summary["line_angle_rms_px"] = float(angle_rms)
    if resolved_offset_deg:
        offset_arr = np.asarray(resolved_offset_deg, dtype=float)
        offset_rms = float(np.sqrt(np.mean(offset_arr * offset_arr)))
        summary["line_offset_rms_deg"] = float(offset_rms)
        summary["line_offset_rms_px"] = float(offset_rms)
    if resolved_spans_deg:
        span_arr = np.asarray(resolved_spans_deg, dtype=float)
        summary["line_fit_space_span_deg_mean"] = float(np.mean(span_arr))

    return residual_components.reshape(-1), summary


def _prepare_reflection_subset(
    miller: np.ndarray,
    intensities: np.ndarray,
    measured_peaks: Optional[Sequence[object]],
) -> ReflectionSimulationSubset:
    """Restrict simulation to reflections referenced by the measured peaks."""

    miller_arr = np.asarray(miller, dtype=np.float64)
    intensities_arr = np.asarray(intensities, dtype=np.float64)
    total_reflections = int(miller_arr.shape[0]) if miller_arr.ndim >= 1 else 0
    normalized_measured = _normalize_measured_peaks(measured_peaks)
    if normalized_measured:
        normalized_measured = [
            {
                **dict(entry),
                **(
                    {"q_group_key": group_key}
                    if (group_key := _normalized_q_group_key(dict(entry).get("q_group_key")))
                    is not None
                    else {}
                ),
            }
            for entry in normalized_measured
        ]

    if (
        miller_arr.ndim != 2
        or miller_arr.shape[1] != 3
        or intensities_arr.ndim != 1
        or intensities_arr.shape[0] != total_reflections
        or total_reflections <= 0
        or not normalized_measured
    ):
        return ReflectionSimulationSubset(
            miller=miller_arr,
            intensities=intensities_arr,
            measured_entries=normalized_measured,
            original_indices=np.arange(max(total_reflections, 0), dtype=np.int64),
            total_reflection_count=total_reflections,
            fixed_source_reflection_count=0,
            fallback_hkl_count=0,
            reduced=False,
        )

    hkl_to_reflection_indices: Dict[Tuple[int, int, int], List[int]] = {}
    for reflection_idx, row in enumerate(miller_arr):
        hkl_key = _miller_key_from_row(row)
        if hkl_key is None:
            continue
        hkl_to_reflection_indices.setdefault(hkl_key, []).append(int(reflection_idx))

    def _entry_hkl_key(entry: Mapping[str, object]) -> Optional[Tuple[int, int, int]]:
        raw_hkl = entry.get("hkl")
        if not isinstance(raw_hkl, tuple) or len(raw_hkl) != 3:
            return None
        try:
            return (int(raw_hkl[0]), int(raw_hkl[1]), int(raw_hkl[2]))
        except Exception:
            return None

    def _coerce_index(value: object) -> Optional[int]:
        try:
            idx = int(value)
        except Exception:
            return None
        return int(idx) if idx >= 0 else None

    def _trusted_full_reflection_index(
        entry: Mapping[str, object],
    ) -> Optional[int]:
        reflection_idx = _coerce_index(entry.get("source_reflection_index"))
        if reflection_idx is None:
            return None
        namespace = str(entry.get("source_reflection_namespace", "") or "").strip().lower()
        if namespace in {"full", "full_reflection", "miller"}:
            return int(reflection_idx)
        try:
            source_is_full = bool(entry.get("source_reflection_is_full", False))
        except Exception:
            source_is_full = False
        return int(reflection_idx) if source_is_full else None

    def _resolve_reflection_index(entry: Mapping[str, object]) -> Optional[int]:
        measured_hkl = _entry_hkl_key(entry)
        reflection_idx = _trusted_full_reflection_index(entry)
        if reflection_idx is None:
            return None

        if measured_hkl is not None:
            matching_indices = hkl_to_reflection_indices.get(measured_hkl, [])
            if reflection_idx is not None and reflection_idx in matching_indices:
                return int(reflection_idx)
            if len(matching_indices) == 1:
                return int(matching_indices[0])

        if reflection_idx < total_reflections:
            return int(reflection_idx)
        return None

    selected_original_indices: List[int] = []
    selected_lookup: set[int] = set()
    fixed_source_entry_indices: set[int] = set()
    resolved_reflection_indices: Dict[int, int] = {}
    for entry_index, entry in enumerate(normalized_measured):
        reflection_idx = _resolve_reflection_index(entry)
        if reflection_idx is None:
            continue
        resolved_reflection_indices[int(entry_index)] = int(reflection_idx)
        raw_hkl = entry.get("hkl")
        if isinstance(raw_hkl, tuple) and len(raw_hkl) == 3:
            source_hkl = _miller_key_from_row(miller_arr[reflection_idx])
            try:
                measured_hkl = (
                    int(raw_hkl[0]),
                    int(raw_hkl[1]),
                    int(raw_hkl[2]),
                )
            except Exception:
                measured_hkl = None
            if measured_hkl is not None and source_hkl != measured_hkl:
                continue
        if reflection_idx not in selected_lookup:
            selected_lookup.add(int(reflection_idx))
            selected_original_indices.append(int(reflection_idx))
        fixed_source_entry_indices.add(int(entry_index))

    fixed_source_reflection_count = int(len(selected_original_indices))

    fallback_hkl_keys: set[Tuple[int, int, int]] = set()
    for entry_index, entry in enumerate(normalized_measured):
        if entry_index in fixed_source_entry_indices:
            continue
        raw_hkl = entry.get("hkl")
        if not isinstance(raw_hkl, tuple) or len(raw_hkl) != 3:
            continue
        try:
            hkl_key = (
                int(raw_hkl[0]),
                int(raw_hkl[1]),
                int(raw_hkl[2]),
            )
        except Exception:
            continue
        fallback_hkl_keys.add(hkl_key)

    if fallback_hkl_keys:
        for idx, row in enumerate(miller_arr):
            hkl_key = _miller_key_from_row(row)
            if hkl_key is None or hkl_key not in fallback_hkl_keys:
                continue
            if idx in selected_lookup:
                continue
            selected_lookup.add(int(idx))
            selected_original_indices.append(int(idx))

    if not selected_original_indices:
        selected_original_indices = list(range(total_reflections))

    original_indices = np.asarray(selected_original_indices, dtype=np.int64)
    local_index_map = {
        int(original_idx): int(local_idx)
        for local_idx, original_idx in enumerate(original_indices.tolist())
    }
    full_identity_preserved = bool(
        int(original_indices.shape[0]) == int(total_reflections)
        and np.array_equal(original_indices, np.arange(total_reflections, dtype=np.int64))
    )

    remapped_measured: List[Dict[str, object]] = []
    for entry_index, entry in enumerate(normalized_measured):
        remapped_entry = dict(entry)
        reflection_idx = resolved_reflection_indices.get(int(entry_index))
        if reflection_idx is not None:
            remapped_entry["source_reflection_index"] = int(reflection_idx)
            remapped_entry["source_reflection_namespace"] = "full_reflection"
            remapped_entry["source_reflection_is_full"] = True
            local_idx = local_index_map.get(int(reflection_idx))
            if local_idx is not None:
                remapped_entry["resolved_table_index"] = int(local_idx)
                try:
                    row_idx = int(entry.get("source_row_index"))
                except Exception:
                    row_idx = -1
                if row_idx >= 0:
                    remapped_entry["source_row_index"] = int(row_idx)
                else:
                    remapped_entry.pop("source_row_index", None)
                branch_idx, _branch_source = _measured_source_peak_index_with_source(
                    entry
                )
                if branch_idx in {0, 1}:
                    remapped_entry["source_branch_index"] = int(branch_idx)
                    remapped_entry["resolved_peak_index"] = int(branch_idx)
                    remapped_entry["source_peak_index"] = int(branch_idx)
                else:
                    remapped_entry.pop("source_branch_index", None)
                    remapped_entry.pop("resolved_peak_index", None)
            else:
                remapped_entry.pop("resolved_table_index", None)
                remapped_entry.pop("resolved_peak_index", None)
                remapped_entry.pop("source_row_index", None)
                remapped_entry.pop("source_branch_index", None)
                remapped_entry.pop("source_peak_index", None)
                remapped_entry.pop("source_table_index", None)
        else:
            local_table_idx = _coerce_index(entry.get("source_table_index"))
            if (
                full_identity_preserved
                and local_table_idx is not None
                and local_table_idx < total_reflections
            ):
                remapped_entry["resolved_table_index"] = int(local_table_idx)
                try:
                    row_idx = int(entry.get("source_row_index"))
                except Exception:
                    row_idx = -1
                if row_idx >= 0:
                    remapped_entry["source_row_index"] = int(row_idx)
                else:
                    remapped_entry.pop("source_row_index", None)
                branch_idx, _branch_source = _measured_source_peak_index_with_source(
                    entry
                )
                if branch_idx in {0, 1}:
                    remapped_entry["resolved_peak_index"] = int(branch_idx)
                    if "source_branch_index" in entry:
                        remapped_entry["source_branch_index"] = int(branch_idx)
                    remapped_entry["source_peak_index"] = int(branch_idx)
                else:
                    remapped_entry.pop("resolved_peak_index", None)
                    remapped_entry.pop("source_branch_index", None)
                    remapped_entry.pop("source_peak_index", None)
                remapped_entry.pop("source_reflection_index", None)
                remapped_entry.pop("source_reflection_namespace", None)
                remapped_entry.pop("source_reflection_is_full", None)
            else:
                remapped_entry.pop("resolved_table_index", None)
                remapped_entry.pop("resolved_peak_index", None)
                remapped_entry.pop("source_reflection_index", None)
                remapped_entry.pop("source_reflection_namespace", None)
                remapped_entry.pop("source_reflection_is_full", None)
                remapped_entry.pop("source_table_index", None)
                remapped_entry.pop("source_row_index", None)
                remapped_entry.pop("source_branch_index", None)
                remapped_entry.pop("source_peak_index", None)
        remapped_measured.append(remapped_entry)

    reduced = len(original_indices) < total_reflections
    return ReflectionSimulationSubset(
        miller=miller_arr[original_indices],
        intensities=intensities_arr[original_indices],
        measured_entries=remapped_measured,
        original_indices=original_indices,
        total_reflection_count=total_reflections,
        fixed_source_reflection_count=fixed_source_reflection_count,
        fallback_hkl_count=int(len(fallback_hkl_keys)),
        reduced=reduced,
    )


def _build_geometry_fit_dataset_contexts(
    miller: np.ndarray,
    intensities: np.ndarray,
    params: Dict[str, object],
    measured_peaks: Optional[Sequence[object]],
    experimental_image: Optional[np.ndarray],
    dataset_specs: Optional[Sequence[Dict[str, object]]] = None,
) -> List[GeometryFitDatasetContext]:
    """Normalize one or more geometry-fit datasets into internal contexts."""

    default_theta = float(params.get("theta_initial", 0.0))
    raw_specs: List[Dict[str, object]] = []
    dataset_spec_entries = _coerce_sequence_items(dataset_specs)

    if dataset_spec_entries:
        for dataset_index, raw_entry in enumerate(dataset_spec_entries):
            if not isinstance(raw_entry, dict):
                raise TypeError("geometry fit dataset_specs entries must be dictionaries")
            entry = dict(raw_entry)
            entry.setdefault("dataset_index", int(dataset_index))
            entry.setdefault("label", f"dataset_{dataset_index}")
            entry.setdefault("theta_initial", default_theta)
            raw_specs.append(entry)
    else:
        raw_specs.append(
            {
                "dataset_index": 0,
                "label": "dataset_0",
                "theta_initial": default_theta,
                "measured_peaks": measured_peaks,
                "experimental_image": experimental_image,
            }
        )

    contexts: List[GeometryFitDatasetContext] = []
    for fallback_index, entry in enumerate(raw_specs):
        try:
            dataset_index = int(entry.get("dataset_index", fallback_index))
        except Exception:
            dataset_index = int(fallback_index)
        if dataset_index < 0:
            dataset_index = int(fallback_index)

        try:
            theta_initial = float(entry.get("theta_initial", default_theta))
        except Exception:
            theta_initial = float(default_theta)
        if not np.isfinite(theta_initial):
            theta_initial = float(default_theta)

        label = str(entry.get("label", f"dataset_{dataset_index}"))
        measured_local = entry.get("measured_peaks", measured_peaks)
        experimental_local = entry.get("experimental_image", experimental_image)
        dynamic_reanchor_enabled = bool(entry.get("dynamic_reanchor_enabled", False))
        dynamic_reanchor_callback = entry.get("dynamic_reanchor_callback", None)
        if not callable(dynamic_reanchor_callback):
            dynamic_reanchor_callback = None
        subset = _prepare_reflection_subset(miller, intensities, measured_local)
        contexts.append(
            GeometryFitDatasetContext(
                dataset_index=int(dataset_index),
                label=label,
                theta_initial=float(theta_initial),
                subset=subset,
                experimental_image=(
                    None
                    if experimental_local is None
                    else np.asarray(experimental_local, dtype=np.float64)
                ),
                dynamic_reanchor_enabled=bool(
                    dynamic_reanchor_enabled and callable(dynamic_reanchor_callback)
                ),
                dynamic_reanchor_callback=dynamic_reanchor_callback,
            )
        )
    return contexts


def _build_global_point_matches(
    simulated_points: Sequence[Tuple[float, float]],
    measured_points: Sequence[Tuple[float, float]],
    *,
    max_distance: float = np.inf,
) -> List[Tuple[np.ndarray, np.ndarray, float, int, int]]:
    """Globally optimal one-to-one matching with optional unmatched points."""

    simulated_entries = _coerce_sequence_items(simulated_points)
    measured_entries = _coerce_sequence_items(measured_points)
    if not simulated_entries or not measured_entries:
        return []

    sim = np.asarray(simulated_entries, dtype=float)
    meas = np.asarray(measured_entries, dtype=float)
    if sim.ndim != 2 or meas.ndim != 2 or sim.shape[1] != 2 or meas.shape[1] != 2:
        return []

    dist = np.linalg.norm(meas[:, None, :] - sim[None, :, :], axis=2)
    if not np.isfinite(max_distance):
        mask = np.isfinite(dist)
    else:
        mask = np.isfinite(dist) & (dist <= float(max_distance))
    if not np.any(mask):
        return []

    finite_dist = dist[mask]
    if finite_dist.size == 0:
        return []

    if np.isfinite(max_distance):
        dummy_cost = max(float(max_distance), 0.0) + 1.0e-6
    else:
        dummy_cost = max(float(np.max(finite_dist)), 0.0) + 1.0
    invalid_cost = dummy_cost + max(dummy_cost, 1.0) * 1.0e6

    num_meas, num_sim = dist.shape
    total_size = num_meas + num_sim
    cost = np.full((total_size, total_size), invalid_cost, dtype=float)
    cost[:num_meas, :num_sim] = np.where(mask, dist, invalid_cost)

    for meas_idx in range(num_meas):
        cost[meas_idx, num_sim + meas_idx] = dummy_cost
    for sim_idx in range(num_sim):
        cost[num_meas + sim_idx, sim_idx] = dummy_cost
    cost[num_meas:, num_sim:] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)
    matches: List[Tuple[np.ndarray, np.ndarray, float, int, int]] = []
    for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
        if row_idx >= num_meas or col_idx >= num_sim:
            continue
        if not bool(mask[row_idx, col_idx]):
            continue
        pair_dist = float(dist[row_idx, col_idx])
        matches.append((sim[col_idx], meas[row_idx], pair_dist, int(col_idx), int(row_idx)))

    matches.sort(key=lambda item: (float(item[2]), int(item[4]), int(item[3])))
    return matches


def _build_greedy_point_matches(
    simulated_points: Sequence[Tuple[float, float]],
    measured_points: Sequence[Tuple[float, float]],
    *,
    max_distance: float = np.inf,
) -> List[Tuple[np.ndarray, np.ndarray, float, int, int]]:
    """Backward-compatible wrapper for global point assignment."""

    return _build_global_point_matches(
        simulated_points,
        measured_points,
        max_distance=max_distance,
    )


def _uses_central_geometry_ray(
    mosaic_params: Mapping[str, object] | None,
    *,
    single_ray_indices: np.ndarray | None = None,
) -> bool:
    """Return whether one fit is effectively using only central geometry ray."""

    if isinstance(single_ray_indices, np.ndarray) and np.any(single_ray_indices >= 0):
        return False
    if not isinstance(mosaic_params, Mapping):
        return True

    def _as_array(key: str, *, default: float = 0.0) -> np.ndarray:
        raw = mosaic_params.get(key, None)
        if raw is None:
            return np.asarray([default], dtype=np.float64)
        try:
            return np.asarray(raw, dtype=np.float64).reshape(-1)
        except Exception:
            return np.asarray([], dtype=np.float64)

    beam_x = _as_array("beam_x_array")
    beam_y = _as_array("beam_y_array")
    theta = _as_array("theta_array")
    phi = _as_array("phi_array")
    if any(arr.size != 1 for arr in (beam_x, beam_y, theta, phi)):
        return False
    if not (
        np.all(np.isfinite(beam_x))
        and np.all(np.isfinite(beam_y))
        and np.all(np.isfinite(theta))
        and np.all(np.isfinite(phi))
    ):
        return False
    return bool(
        np.allclose(beam_x, 0.0)
        and np.allclose(beam_y, 0.0)
        and np.allclose(theta, 0.0)
        and np.allclose(phi, 0.0)
    )


def _evaluate_geometry_fit_dataset_point_matches(
    local: Dict[str, object],
    dataset_ctx: GeometryFitDatasetContext,
    *,
    image_size: int,
    pixel_tol: float,
    weighted_matching: bool,
    solver_f_scale: float,
    missing_pair_penalty: float,
    use_single_ray: bool,
    theta_value: float,
    use_measurement_uncertainty: bool = False,
    anisotropic_uncertainty: bool = False,
    radial_sigma_scale: float = 1.0,
    tangential_sigma_scale: float = 1.0,
    collect_diagnostics: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
    """Evaluate one measured dataset against one simulated geometry state."""

    simulation_subset = dataset_ctx.subset
    fit_miller = simulation_subset.miller
    fit_intensities = simulation_subset.intensities
    normalized_measured = [
        dict(entry)
        for entry in (simulation_subset.measured_entries or [])
        if isinstance(entry, Mapping)
    ]
    single_ray_indices = dataset_ctx.single_ray_indices
    try:
        hk0_peak_priority_weight = float(local.get("_hk0_peak_priority_weight", 1.0))
    except Exception:
        hk0_peak_priority_weight = 1.0
    if not np.isfinite(hk0_peak_priority_weight) or hk0_peak_priority_weight < 1.0:
        hk0_peak_priority_weight = 1.0
    line_constraints_enabled = bool(local.get("_q_group_line_constraints_enabled", False))
    try:
        q_group_line_angle_weight = float(local.get("_q_group_line_angle_weight", 0.6))
    except Exception:
        q_group_line_angle_weight = 0.6
    if not np.isfinite(q_group_line_angle_weight) or q_group_line_angle_weight < 0.0:
        q_group_line_angle_weight = 0.6
    try:
        q_group_line_offset_weight = float(local.get("_q_group_line_offset_weight", 1.0))
    except Exception:
        q_group_line_offset_weight = 1.0
    if not np.isfinite(q_group_line_offset_weight) or q_group_line_offset_weight < 0.0:
        q_group_line_offset_weight = 1.0
    try:
        q_group_line_missing_penalty_scale = float(
            local.get("_q_group_line_missing_penalty_scale", 0.35)
        )
    except Exception:
        q_group_line_missing_penalty_scale = 0.35
    if (
        not np.isfinite(q_group_line_missing_penalty_scale)
        or q_group_line_missing_penalty_scale < 0.0
    ):
        q_group_line_missing_penalty_scale = 0.35
    central_ray_mode = _uses_central_geometry_ray(
        local.get("mosaic_params"),
        single_ray_indices=single_ray_indices,
    )

    if not normalized_measured:
        return (
            np.array([], dtype=float),
            [],
            {
                "dataset_index": int(dataset_ctx.dataset_index),
                "dataset_label": str(dataset_ctx.label),
                "theta_initial_deg": float(theta_value),
                "measured_count": 0,
                "fixed_source_resolved_count": 0,
                "fallback_entry_count": 0,
                "missing_pair_count": 0,
                "simulated_reflection_count": int(fit_miller.shape[0]),
                "total_reflection_count": int(simulation_subset.total_reflection_count),
                "subset_reduced": bool(simulation_subset.reduced),
                "central_ray_mode": bool(central_ray_mode),
                "single_ray_enabled": bool(use_single_ray),
                "single_ray_forced_count": 0,
                "line_group_count": 0,
                "resolved_line_group_count": 0,
                "missing_line_group_count": 0,
                "line_angle_rms_px": float("nan"),
                "line_offset_rms_px": float("nan"),
                "line_fit_space_span_deg_mean": float("nan"),
                "line_constraints_enabled": bool(line_constraints_enabled),
            },
        )

    mosaic = local["mosaic_params"]
    wavelength_array = mosaic.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic.get("wavelength_i_array")

    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
    _, hit_tables, *_ = _process_peaks_parallel_safe(
        fit_miller,
        fit_intensities,
        image_size,
        local["a"],
        local["c"],
        wavelength_array,
        sim_buffer,
        local["corto_detector"],
        local["gamma"],
        local["Gamma"],
        local["chi"],
        local.get("psi", 0.0),
        local.get("psi_z", 0.0),
        local["zs"],
        local["zb"],
        local["n2"],
        mosaic["beam_x_array"],
        mosaic["beam_y_array"],
        mosaic["theta_array"],
        mosaic["phi_array"],
        mosaic["sigma_mosaic_deg"],
        mosaic["gamma_mosaic_deg"],
        mosaic["eta"],
        wavelength_array,
        local["debye_x"],
        local["debye_y"],
        local["center"],
        theta_value,
        local.get("cor_angle", 0.0),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        save_flag=0,
        **_simulation_kernel_kwargs(local, mosaic),
        single_sample_indices=single_ray_indices,
    )

    def _point_radius_px(point: Tuple[float, float]) -> float:
        try:
            center_row = float(local["center"][0])
            center_col = float(local["center"][1])
        except Exception:
            return float("nan")
        if not (
            np.isfinite(point[0])
            and np.isfinite(point[1])
            and np.isfinite(center_row)
            and np.isfinite(center_col)
        ):
            return float("nan")
        return float(math.hypot(float(point[0]) - center_col, float(point[1]) - center_row))

    maxpos = hit_tables_to_max_positions(hit_tables)
    measured_index_lookup = {id(entry): int(idx) for idx, entry in enumerate(normalized_measured)}
    # Keep one fixed two-component residual block per measured point. SciPy's
    # finite-difference Jacobian estimation requires a stable residual shape
    # even when one point flips between matched and missing.
    residual_components = np.zeros((len(normalized_measured), 2), dtype=np.float64)
    unresolved_indices = set(measured_index_lookup.values())
    diagnostics: List[Dict[str, object]] = []
    custom_sigma_values: List[float] = []
    matched_distances: List[float] = []
    matched_pair_count = 0
    anisotropic_sigma_count = 0
    eligible_line_group_ids = (
        _eligible_geometry_line_group_ids(normalized_measured)
        if bool(line_constraints_enabled)
        else []
    )
    line_match_records: List[Dict[str, object]] = []
    fixed_matches, fallback_measured, resolution_lookup = _resolve_fixed_source_matches(
        normalized_measured,
        hit_tables,
    )

    fallback_entries_by_hkl: Dict[Tuple[int, int, int], List[Dict[str, object]]] = {}
    for entry in fallback_measured:
        raw_hkl = entry.get("hkl")
        if not isinstance(raw_hkl, tuple) or len(raw_hkl) != 3:
            continue
        hkl_key = (int(raw_hkl[0]), int(raw_hkl[1]), int(raw_hkl[2]))
        fallback_entries_by_hkl.setdefault(hkl_key, []).append(entry)

    def _add_diag(base_entry: Dict[str, object], payload: Dict[str, object]) -> None:
        if not collect_diagnostics:
            return
        diag = dict(base_entry)
        diag.update(payload)
        diag["dataset_index"] = int(dataset_ctx.dataset_index)
        diag["dataset_label"] = str(dataset_ctx.label)
        diag["theta_initial_deg"] = float(theta_value)
        diagnostics.append(diag)

    def _weight_fields(
        entry: Dict[str, object],
        *,
        measured_point: Tuple[float, float],
        dx: float = 0.0,
        dy: float = 0.0,
        distance_weight: float = 1.0,
    ) -> Dict[str, float | bool]:
        nonlocal anisotropic_sigma_count
        fields = _weight_measurement_residual(
            dx,
            dy,
            measured_point=measured_point,
            center=local.get("center", []),
            entry=entry,
            distance_weight=distance_weight,
            use_measurement_uncertainty=bool(use_measurement_uncertainty),
            anisotropic_enabled=bool(anisotropic_uncertainty),
            radial_scale=float(radial_sigma_scale),
            tangential_scale=float(tangential_sigma_scale),
        )
        if bool(fields.get("sigma_is_custom", False)):
            custom_sigma_values.append(float(fields.get("measurement_sigma_px", np.nan)))
        if bool(fields.get("anisotropic_sigma_used", False)):
            anisotropic_sigma_count += 1
        priority_fields = _geometry_peak_priority_metadata(
            entry,
            hk0_peak_priority_weight=float(hk0_peak_priority_weight),
        )
        priority_weight = float(priority_fields["priority_weight"])
        for key in (
            "weighted_dx_px",
            "weighted_dy_px",
            "weighted_radial_residual_px",
            "weighted_tangential_residual_px",
        ):
            try:
                value = float(fields.get(key, np.nan))
            except Exception:
                continue
            if np.isfinite(value):
                fields[key] = float(priority_weight * value)
        fields.update(priority_fields)
        return fields

    def _placement_error_px(entry: Dict[str, object]) -> float:
        try:
            value = float(entry.get("placement_error_px", np.nan))
        except Exception:
            return float("nan")
        return float(value) if np.isfinite(value) else float("nan")

    def _assign_residual_pair(
        entry: Dict[str, object],
        primary: float,
        secondary: float = 0.0,
    ) -> None:
        slot = measured_index_lookup.get(id(entry))
        if slot is None:
            return
        residual_components[int(slot), 0] = float(primary)
        residual_components[int(slot), 1] = float(secondary)
        unresolved_indices.discard(int(slot))

    for measured_entry, sim_pt, _sim_hkl in fixed_matches:
        try:
            meas_pt = (float(measured_entry["x"]), float(measured_entry["y"]))
        except Exception:
            continue
        if not (
            np.isfinite(sim_pt[0])
            and np.isfinite(sim_pt[1])
            and np.isfinite(meas_pt[0])
            and np.isfinite(meas_pt[1])
        ):
            continue
        dx = float(sim_pt[0] - meas_pt[0])
        dy = float(sim_pt[1] - meas_pt[1])
        pair_dist = math.hypot(dx, dy)
        matched_pair_count += 1
        matched_distances.append(float(pair_dist))
        if weighted_matching:
            distance_weight = 1.0 / math.sqrt(1.0 + (pair_dist / solver_f_scale) ** 2)
        else:
            distance_weight = 1.0
        weight_fields = _weight_fields(
            measured_entry,
            measured_point=meas_pt,
            dx=dx,
            dy=dy,
            distance_weight=float(distance_weight),
        )
        weighted_dx = float(weight_fields["weighted_dx_px"])
        weighted_dy = float(weight_fields["weighted_dy_px"])
        _assign_residual_pair(measured_entry, weighted_dx, weighted_dy)
        group_ids = _geometry_line_group_ids(measured_entry)
        if group_ids:
            line_match_records.append(
                {
                    "group_ids": group_ids,
                    "measured_point": meas_pt,
                    "simulated_point": sim_pt,
                }
            )
        _add_diag(
            dict(resolution_lookup.get(id(measured_entry), {})),
            {
                "match_kind": "fixed_source",
                "match_status": "matched",
                "measured_x": float(meas_pt[0]),
                "measured_y": float(meas_pt[1]),
                "simulated_x": float(sim_pt[0]),
                "simulated_y": float(sim_pt[1]),
                "dx_px": float(sim_pt[0] - meas_pt[0]),
                "dy_px": float(sim_pt[1] - meas_pt[1]),
                "distance_px": float(pair_dist),
                "placement_error_px": _placement_error_px(measured_entry),
                "distance_weight": float(distance_weight),
                "weight": float(
                    float(distance_weight)
                    * float(weight_fields.get("sigma_weight", 1.0))
                    * float(weight_fields.get("priority_weight", 1.0))
                ),
                "measured_radius_px": _point_radius_px(meas_pt),
                "simulated_radius_px": _point_radius_px(sim_pt),
                **weight_fields,
            },
        )

    measured_dict = build_measured_dict(fallback_measured)
    simulated_candidates_by_hkl = _collect_geometry_fit_simulated_candidates(
        fit_miller,
        hit_tables,
        original_indices=simulation_subset.original_indices,
        measured_hkls=set(measured_dict),
    )

    missing_pairs = 0
    for hkl_key in measured_dict:
        sim_candidates = simulated_candidates_by_hkl.get(hkl_key, [])
        sim_list = [
            (
                float(candidate.get("simulated_x", np.nan)),
                float(candidate.get("simulated_y", np.nan)),
            )
            for candidate in sim_candidates
        ]
        measured_entries_hkl = fallback_entries_by_hkl.get(hkl_key, [])
        valid_entries_hkl: List[Dict[str, object]] = []
        measured_points: List[Tuple[float, float]] = []
        for entry in measured_entries_hkl:
            try:
                mx = float(entry["x"])
                my = float(entry["y"])
            except Exception:
                continue
            if not (np.isfinite(mx) and np.isfinite(my)):
                continue
            valid_entries_hkl.append(entry)
            measured_points.append((mx, my))
        if not measured_points:
            continue
        if not sim_list:
            missing_pairs += len(valid_entries_hkl)
            for entry in valid_entries_hkl:
                measured_point = (float(entry["x"]), float(entry["y"]))
                weight_fields = _weight_fields(
                    entry,
                    measured_point=measured_point,
                    distance_weight=1.0,
                )
                penalty_weight = float(weight_fields.get("sigma_weight", 1.0))
                penalty_weight *= float(weight_fields.get("priority_weight", 1.0))
                missing_penalty = float(missing_pair_penalty) * penalty_weight
                _assign_residual_pair(entry, missing_penalty, 0.0)
                _add_diag(
                    dict(resolution_lookup.get(id(entry), {})),
                    {
                        "match_kind": "hkl_fallback",
                        "match_status": "missing_pair",
                        "hkl": tuple(int(v) for v in hkl_key),
                        "source_peak_index": None,
                        "resolution_kind": str(
                            dict(resolution_lookup.get(id(entry), {})).get(
                                "resolution_kind", "hkl_fallback"
                            )
                        ),
                        "resolution_reason": str(
                            dict(resolution_lookup.get(id(entry), {})).get(
                                "resolution_reason", "no_simulated_candidates"
                            )
                        ),
                        "measured_x": float(entry["x"]),
                        "measured_y": float(entry["y"]),
                        "simulated_x": float("nan"),
                        "simulated_y": float("nan"),
                        "dx_px": float("nan"),
                        "dy_px": float("nan"),
                        "distance_px": float("nan"),
                        "placement_error_px": _placement_error_px(entry),
                        "distance_weight": 1.0,
                        "weight": float(penalty_weight),
                        "weighted_missing_penalty_px": float(missing_pair_penalty)
                        * float(penalty_weight),
                        "measured_radius_px": _point_radius_px(measured_point),
                        "simulated_radius_px": float("nan"),
                        **weight_fields,
                    },
                )
            continue

        matches = _build_greedy_point_matches(sim_list, measured_points, max_distance=pixel_tol)
        matched_meas_indices: set[int] = set()
        for sim_pt, meas_pt, pair_dist, sim_idx, meas_idx in matches:
            matched_meas_indices.add(int(meas_idx))
            entry = valid_entries_hkl[meas_idx] if 0 <= meas_idx < len(valid_entries_hkl) else {}
            matched_pair_count += 1
            matched_distances.append(float(pair_dist))
            dx = float(sim_pt[0] - meas_pt[0])
            dy = float(sim_pt[1] - meas_pt[1])
            if weighted_matching:
                distance_weight = 1.0 / math.sqrt(1.0 + (pair_dist / solver_f_scale) ** 2)
            else:
                distance_weight = 1.0
            weight_fields = _weight_fields(
                entry,
                measured_point=(float(meas_pt[0]), float(meas_pt[1])),
                dx=dx,
                dy=dy,
                distance_weight=float(distance_weight),
            )
            weighted_dx = float(weight_fields["weighted_dx_px"])
            weighted_dy = float(weight_fields["weighted_dy_px"])
            _assign_residual_pair(entry, weighted_dx, weighted_dy)
            group_ids = _geometry_line_group_ids(entry)
            if group_ids:
                line_match_records.append(
                    {
                        "group_ids": group_ids,
                        "measured_point": (float(meas_pt[0]), float(meas_pt[1])),
                        "simulated_point": (float(sim_pt[0]), float(sim_pt[1])),
                    }
                )
            if 0 <= meas_idx < len(valid_entries_hkl):
                sim_candidate = (
                    sim_candidates[sim_idx] if 0 <= sim_idx < len(sim_candidates) else {}
                )
                _add_diag(
                    dict(resolution_lookup.get(id(entry), {})),
                    {
                        "match_kind": "hkl_fallback",
                        "match_status": "matched",
                        "hkl": tuple(int(v) for v in hkl_key),
                        "sim_list_index": int(sim_idx),
                        "meas_list_index": int(meas_idx),
                        "source_table_index": None,
                        "resolved_table_index": sim_candidate.get("resolved_table_index"),
                        "source_reflection_index": sim_candidate.get("source_reflection_index"),
                        "source_reflection_namespace": None,
                        "source_reflection_is_full": False,
                        "source_branch_index": sim_candidate.get("source_branch_index"),
                        "source_peak_index": sim_candidate.get("source_peak_index"),
                        "resolved_peak_index": sim_candidate.get("resolved_peak_index"),
                        "measured_x": float(meas_pt[0]),
                        "measured_y": float(meas_pt[1]),
                        "simulated_x": float(sim_pt[0]),
                        "simulated_y": float(sim_pt[1]),
                        "dx_px": float(sim_pt[0] - meas_pt[0]),
                        "dy_px": float(sim_pt[1] - meas_pt[1]),
                        "distance_px": float(pair_dist),
                        "placement_error_px": _placement_error_px(entry),
                        "distance_weight": float(distance_weight),
                        "weight": float(
                            float(distance_weight)
                            * float(weight_fields.get("sigma_weight", 1.0))
                            * float(weight_fields.get("priority_weight", 1.0))
                        ),
                        "measured_radius_px": _point_radius_px(
                            (float(meas_pt[0]), float(meas_pt[1]))
                        ),
                        "simulated_radius_px": _point_radius_px(
                            (float(sim_pt[0]), float(sim_pt[1]))
                        ),
                        **weight_fields,
                    },
                )

        unmatched_meas_indices = [
            idx for idx in range(len(valid_entries_hkl)) if idx not in matched_meas_indices
        ]
        missing_pairs += len(unmatched_meas_indices)
        for meas_idx in unmatched_meas_indices:
            entry = valid_entries_hkl[meas_idx]
            measured_point = (float(entry["x"]), float(entry["y"]))
            weight_fields = _weight_fields(
                entry,
                measured_point=measured_point,
                distance_weight=1.0,
            )
            penalty_weight = float(weight_fields.get("sigma_weight", 1.0))
            penalty_weight *= float(weight_fields.get("priority_weight", 1.0))
            missing_penalty = float(missing_pair_penalty) * penalty_weight
            _assign_residual_pair(entry, missing_penalty, 0.0)
            _add_diag(
                dict(resolution_lookup.get(id(entry), {})),
                {
                    "match_kind": "hkl_fallback",
                    "match_status": "missing_pair",
                    "hkl": tuple(int(v) for v in hkl_key),
                    "source_peak_index": None,
                    "resolution_kind": str(
                        dict(resolution_lookup.get(id(entry), {})).get(
                            "resolution_kind", "hkl_fallback"
                        )
                    ),
                    "resolution_reason": str(
                        dict(resolution_lookup.get(id(entry), {})).get(
                            "resolution_reason", "unmatched_after_assignment"
                        )
                    ),
                    "measured_x": float(entry["x"]),
                    "measured_y": float(entry["y"]),
                    "simulated_x": float("nan"),
                    "simulated_y": float("nan"),
                    "dx_px": float("nan"),
                    "dy_px": float("nan"),
                    "distance_px": float("nan"),
                    "placement_error_px": _placement_error_px(entry),
                    "distance_weight": 1.0,
                    "weight": float(penalty_weight),
                    "weighted_missing_penalty_px": float(missing_pair_penalty)
                    * float(penalty_weight),
                    "measured_radius_px": _point_radius_px(measured_point),
                    "simulated_radius_px": float("nan"),
                    **weight_fields,
                },
            )

    for unresolved_idx in list(unresolved_indices):
        if unresolved_idx < 0 or unresolved_idx >= len(normalized_measured):
            continue
        entry = normalized_measured[unresolved_idx]
        measured_point = (
            float(entry.get("x", np.nan)),
            float(entry.get("y", np.nan)),
        )
        weight_fields = _weight_fields(
            entry,
            measured_point=measured_point,
            distance_weight=1.0,
        )
        penalty_weight = float(weight_fields.get("sigma_weight", 1.0))
        penalty_weight *= float(weight_fields.get("priority_weight", 1.0))
        residual_components[int(unresolved_idx), 0] = float(missing_pair_penalty) * float(
            penalty_weight
        )
        residual_components[int(unresolved_idx), 1] = 0.0
        missing_pairs += 1
        if collect_diagnostics:
            _add_diag(
                dict(resolution_lookup.get(id(entry), {})),
                {
                    "match_kind": "unresolved",
                    "match_status": "missing_pair",
                    "hkl": tuple(int(v) for v in entry.get("hkl", ()))
                    if isinstance(entry.get("hkl"), tuple)
                    else entry.get("hkl"),
                    "resolution_kind": str(
                        dict(resolution_lookup.get(id(entry), {})).get(
                            "resolution_kind", "unresolved"
                        )
                    ),
                    "resolution_reason": str(
                        dict(resolution_lookup.get(id(entry), {})).get(
                            "resolution_reason", "entry_not_evaluated"
                        )
                    ),
                    "measured_x": float(entry.get("x", np.nan)),
                    "measured_y": float(entry.get("y", np.nan)),
                    "simulated_x": float("nan"),
                    "simulated_y": float("nan"),
                    "dx_px": float("nan"),
                    "dy_px": float("nan"),
                    "distance_px": float("nan"),
                    "placement_error_px": _placement_error_px(entry),
                    "distance_weight": 1.0,
                    "weight": float(penalty_weight),
                    "weighted_missing_penalty_px": float(missing_pair_penalty)
                    * float(penalty_weight),
                    "measured_radius_px": _point_radius_px(
                        (
                            float(entry.get("x", np.nan)),
                            float(entry.get("y", np.nan)),
                        )
                    ),
                    "simulated_radius_px": float("nan"),
                    **weight_fields,
                },
            )

    line_summary = {
        "line_group_count": 0,
        "resolved_line_group_count": 0,
        "missing_line_group_count": 0,
        "line_angle_rms_px": float("nan"),
        "line_offset_rms_px": float("nan"),
        "line_fit_space_span_deg_mean": float("nan"),
        "line_constraints_enabled": bool(line_constraints_enabled),
    }
    residual_arr = residual_components.reshape(-1)
    if bool(line_constraints_enabled):
        line_residual_arr, line_summary = (
            _build_geometry_line_constraint_residuals_from_detector_matches(
                line_match_records,
                eligible_line_group_ids,
                center=local.get("center", []),
                detector_distance=float(local.get("corto_detector", np.nan)),
                pixel_size=_estimate_pixel_size(local),
                gamma_deg=float(local.get("gamma", 0.0)),
                Gamma_deg=float(local.get("Gamma", 0.0)),
                angle_weight=float(q_group_line_angle_weight),
                offset_weight=float(q_group_line_offset_weight),
                missing_penalty_px=float(missing_pair_penalty)
                * float(q_group_line_missing_penalty_scale),
            )
        )
        if line_residual_arr.size:
            residual_arr = np.concatenate((residual_arr, line_residual_arr))
    summary: Dict[str, object] = {
        "dataset_index": int(dataset_ctx.dataset_index),
        "dataset_label": str(dataset_ctx.label),
        "theta_initial_deg": float(theta_value),
        "measured_count": int(len(normalized_measured)),
        "fixed_source_resolved_count": int(len(fixed_matches)),
        "fallback_entry_count": int(len(fallback_measured)),
        "fallback_hkl_count": int(len(measured_dict)),
        "matched_pair_count": int(matched_pair_count),
        "missing_pair_count": int(missing_pairs),
        "simulated_reflection_count": int(fit_miller.shape[0]),
        "total_reflection_count": int(simulation_subset.total_reflection_count),
        "fixed_source_reflection_count": int(simulation_subset.fixed_source_reflection_count),
        "subset_fallback_hkl_count": int(simulation_subset.fallback_hkl_count),
        "subset_reduced": bool(simulation_subset.reduced),
        "central_ray_mode": bool(central_ray_mode),
        "single_ray_enabled": bool(use_single_ray),
        "single_ray_forced_count": int(np.count_nonzero(single_ray_indices >= 0))
        if isinstance(single_ray_indices, np.ndarray)
        else 0,
        "center_row": float(local["center"][0])
        if len(local.get("center", [])) >= 2
        else float("nan"),
        "center_col": float(local["center"][1])
        if len(local.get("center", [])) >= 2
        else float("nan"),
    }
    summary["anisotropic_sigma_count"] = int(anisotropic_sigma_count)
    if custom_sigma_values:
        sigma_arr = np.asarray(custom_sigma_values, dtype=float)
        sigma_arr = sigma_arr[np.isfinite(sigma_arr) & (sigma_arr > 0.0)]
        if sigma_arr.size:
            summary["custom_sigma_count"] = int(sigma_arr.size)
            summary["measurement_sigma_median_px"] = float(np.median(sigma_arr))
            summary["measurement_sigma_mean_px"] = float(np.mean(sigma_arr))
            summary["measurement_sigma_max_px"] = float(np.max(sigma_arr))
        else:
            summary["custom_sigma_count"] = 0
    else:
        summary["custom_sigma_count"] = 0
    if matched_distances:
        matched_dist_arr = np.asarray(matched_distances, dtype=float)
        matched_dist_arr = matched_dist_arr[np.isfinite(matched_dist_arr)]
        if matched_dist_arr.size:
            summary["unweighted_peak_rms_px"] = float(
                np.sqrt(np.mean(matched_dist_arr * matched_dist_arr))
            )
            summary["unweighted_peak_mean_px"] = float(np.mean(matched_dist_arr))
            summary["unweighted_peak_max_px"] = float(np.max(matched_dist_arr))
        else:
            summary["unweighted_peak_rms_px"] = float("nan")
            summary["unweighted_peak_mean_px"] = float("nan")
            summary["unweighted_peak_max_px"] = float("nan")
    else:
        summary["unweighted_peak_rms_px"] = float("nan")
        summary["unweighted_peak_mean_px"] = float("nan")
        summary["unweighted_peak_max_px"] = float("nan")
    if anisotropic_sigma_count > 0:
        summary["peak_weighting_mode"] = (
            "measurement_covariance+distance" if weighted_matching else "measurement_covariance"
        )
    elif summary.get("custom_sigma_count", 0):
        summary["peak_weighting_mode"] = (
            "measurement_sigma+distance" if weighted_matching else "measurement_sigma"
        )
    else:
        summary["peak_weighting_mode"] = "uniform"
    summary.update(line_summary)
    return residual_arr, diagnostics, summary


def _evaluate_geometry_fit_dataset_dynamic_point_matches(
    local: Dict[str, object],
    dataset_ctx: GeometryFitDatasetContext,
    *,
    image_size: int,
    missing_pair_penalty_deg: float,
    theta_value: float,
    collect_diagnostics: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
    """Evaluate one dataset with fixed source identities and angular residuals."""

    simulation_subset = dataset_ctx.subset
    fit_miller = simulation_subset.miller
    fit_intensities = simulation_subset.intensities
    normalized_measured = [
        dict(entry) if isinstance(entry, Mapping) else entry
        for entry in (simulation_subset.measured_entries or [])
    ]
    single_ray_indices = dataset_ctx.single_ray_indices
    try:
        hk0_peak_priority_weight = float(local.get("_hk0_peak_priority_weight", 1.0))
    except Exception:
        hk0_peak_priority_weight = 1.0
    if not np.isfinite(hk0_peak_priority_weight) or hk0_peak_priority_weight < 1.0:
        hk0_peak_priority_weight = 1.0
    line_constraints_enabled = bool(local.get("_q_group_line_constraints_enabled", False))
    try:
        q_group_line_angle_weight = float(local.get("_q_group_line_angle_weight", 0.6))
    except Exception:
        q_group_line_angle_weight = 0.6
    if not np.isfinite(q_group_line_angle_weight) or q_group_line_angle_weight < 0.0:
        q_group_line_angle_weight = 0.6
    try:
        q_group_line_offset_weight = float(local.get("_q_group_line_offset_weight", 1.0))
    except Exception:
        q_group_line_offset_weight = 1.0
    if not np.isfinite(q_group_line_offset_weight) or q_group_line_offset_weight < 0.0:
        q_group_line_offset_weight = 1.0
    try:
        q_group_line_missing_penalty_scale = float(
            local.get("_q_group_line_missing_penalty_scale", 0.35)
        )
    except Exception:
        q_group_line_missing_penalty_scale = 0.35
    if (
        not np.isfinite(q_group_line_missing_penalty_scale)
        or q_group_line_missing_penalty_scale < 0.0
    ):
        q_group_line_missing_penalty_scale = 0.35

    fit_space_summary_defaults = _fit_space_provenance_summary(local)
    central_ray_mode = _uses_central_geometry_ray(
        local.get("mosaic_params"),
        single_ray_indices=single_ray_indices,
    )

    if not normalized_measured:
        return (
            np.array([], dtype=float),
            [],
            {
                "dataset_index": int(dataset_ctx.dataset_index),
                "dataset_label": str(dataset_ctx.label),
                "theta_initial_deg": float(theta_value),
                "measured_count": 0,
                "fixed_source_resolved_count": 0,
                "fallback_entry_count": 0,
                "missing_pair_count": 0,
                "simulated_reflection_count": int(fit_miller.shape[0]),
                "total_reflection_count": int(simulation_subset.total_reflection_count),
                "subset_reduced": bool(simulation_subset.reduced),
                "central_ray_mode": bool(central_ray_mode),
                "single_ray_enabled": False,
                "single_ray_forced_count": 0,
                "peak_weighting_mode": "uniform",
                "line_group_count": 0,
                "resolved_line_group_count": 0,
                "missing_line_group_count": 0,
                "line_angle_rms_px": float("nan"),
                "line_offset_rms_px": float("nan"),
                "line_angle_rms_deg": float("nan"),
                "line_offset_rms_deg": float("nan"),
                "line_fit_space_span_deg_mean": float("nan"),
                "line_constraints_enabled": bool(line_constraints_enabled),
                "_live_cache_records": [],
                **fit_space_summary_defaults,
            },
        )

    mosaic = local["mosaic_params"]
    wavelength_array = mosaic.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic.get("wavelength_i_array")

    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
    _, hit_tables, *_ = _process_peaks_parallel_safe(
        fit_miller,
        fit_intensities,
        image_size,
        local["a"],
        local["c"],
        wavelength_array,
        sim_buffer,
        local["corto_detector"],
        local["gamma"],
        local["Gamma"],
        local["chi"],
        local.get("psi", 0.0),
        local.get("psi_z", 0.0),
        local["zs"],
        local["zb"],
        local["n2"],
        mosaic["beam_x_array"],
        mosaic["beam_y_array"],
        mosaic["theta_array"],
        mosaic["phi_array"],
        mosaic["sigma_mosaic_deg"],
        mosaic["gamma_mosaic_deg"],
        mosaic["eta"],
        wavelength_array,
        local["debye_x"],
        local["debye_y"],
        local["center"],
        theta_value,
        local.get("cor_angle", 0.0),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        save_flag=0,
        **_simulation_kernel_kwargs(local, mosaic),
        single_sample_indices=single_ray_indices,
    )

    def _overlay_index(entry: Dict[str, object], fallback: int) -> int:
        try:
            out = int(entry.get("overlay_match_index", fallback))
        except Exception:
            out = int(fallback)
        if out < 0:
            return int(fallback)
        return int(out)

    maxpos = hit_tables_to_max_positions(hit_tables)
    residual_components = np.zeros((len(normalized_measured), 2), dtype=np.float64)
    diagnostics: List[Dict[str, object]] = []
    pixel_distances: List[float] = []
    angular_distances: List[float] = []
    matched_pair_count = 0
    missing_pairs = 0
    fixed_source_resolved_count = 0
    fit_space_anchor_count_cached = 0
    fit_space_anchor_count_detector = 0
    fit_space_two_theta_adjustments: List[float] = []
    pixel_size_provenance = _fit_space_pixel_size_provenance(local)
    pixel_size = float(pixel_size_provenance.get("value", np.nan))
    detector_distance = float(local.get("corto_detector", np.nan))
    gamma_deg = float(local.get("gamma", 0.0))
    Gamma_deg = float(local.get("Gamma", 0.0))
    fit_center = local.get("center", [])
    a_lattice = float(local.get("a", np.nan))
    c_lattice = float(local.get("c", np.nan))
    wavelength_scalar = _wavelength_scalar_from_local(local)
    eligible_line_group_ids = (
        _eligible_geometry_line_group_ids(normalized_measured)
        if bool(line_constraints_enabled)
        else []
    )
    line_match_records: List[Dict[str, object]] = []
    live_cache_records: List[Dict[str, object]] = []
    dynamic_reanchor_enabled = bool(
        dataset_ctx.dynamic_reanchor_enabled and callable(dataset_ctx.dynamic_reanchor_callback)
    )
    measured_anchor_reanchor_attempt_count = 0
    measured_anchor_reanchor_count = 0
    measured_anchor_reanchor_fail_count = 0
    measured_anchor_motion_px: List[float] = []

    def _entry_diag(entry: Dict[str, object], idx: int) -> Dict[str, object]:
        source_branch_index, source_branch_source = _measured_source_peak_index_with_source(
            entry
        )
        return {
            "match_input_index": int(idx),
            "overlay_match_index": int(_overlay_index(entry, idx)),
            "pair_id": entry.get("pair_id"),
            "fit_run_id": entry.get("fit_run_id"),
            "label": str(entry.get("label", "")),
            "hkl": tuple(entry.get("hkl", ()))
            if isinstance(entry.get("hkl"), tuple)
            else entry.get("hkl"),
            "q_group_key": entry.get("q_group_key"),
            "source_table_index": entry.get("source_table_index"),
            "source_reflection_index": entry.get("source_reflection_index"),
            "source_reflection_namespace": entry.get("source_reflection_namespace"),
            "source_reflection_is_full": entry.get("source_reflection_is_full"),
            "source_row_index": entry.get("source_row_index"),
            "source_branch_index": source_branch_index,
            "source_branch_resolution_source": source_branch_source,
            "source_peak_index": entry.get("source_peak_index"),
            "fit_source_resolution_kind": entry.get("fit_source_resolution_kind"),
        }

    for idx, measured_entry in enumerate(normalized_measured):
        diag = _entry_diag(measured_entry, idx)
        priority_fields = _geometry_peak_priority_metadata(
            measured_entry,
            hk0_peak_priority_weight=float(hk0_peak_priority_weight),
        )
        priority_weight = float(priority_fields["priority_weight"])
        sim_point, sim_reason = _geometry_fit_correspondence_simulated_point(
            measured_entry,
            hit_tables=hit_tables,
            max_positions=maxpos,
        )
        if sim_point is not None:
            sim_col = float(sim_point[0])
            sim_row = float(sim_point[1])
            if (
                not np.isfinite(sim_col)
                or not np.isfinite(sim_row)
                or sim_col < 0.0
                or sim_row < 0.0
                or sim_col > float(image_size - 1)
                or sim_row > float(image_size - 1)
            ):
                sim_point = None
                sim_reason = "off_detector"

        reanchor_attempted = False
        reanchor_status = "disabled"
        reanchor_motion_px = float("nan")
        if dynamic_reanchor_enabled and sim_point is not None:
            reanchor_attempted = True
            measured_anchor_reanchor_attempt_count += 1
            reanchor_status = "callback_failed"
            try:
                rebuilt_entry = dataset_ctx.dynamic_reanchor_callback(
                    dict(measured_entry),
                    (float(sim_point[0]), float(sim_point[1])),
                    local_params=dict(local),
                    dataset_ctx=dataset_ctx,
                )
            except Exception:
                rebuilt_entry = None
            if isinstance(rebuilt_entry, Mapping):
                updated_entry = dict(measured_entry)
                updated_entry.update(dict(rebuilt_entry))
                try:
                    reanchor_motion_px = float(
                        rebuilt_entry.get("measured_reanchor_motion_px", np.nan)
                    )
                except Exception:
                    reanchor_motion_px = float("nan")
                if not np.isfinite(reanchor_motion_px):
                    old_detector_anchor, _old_reason = _measured_detector_anchor(measured_entry)
                    new_detector_anchor, _new_reason = _measured_detector_anchor(updated_entry)
                    if old_detector_anchor is not None and new_detector_anchor is not None:
                        reanchor_motion_px = float(
                            math.hypot(
                                float(new_detector_anchor[0]) - float(old_detector_anchor[0]),
                                float(new_detector_anchor[1]) - float(old_detector_anchor[1]),
                            )
                        )
                if np.isfinite(reanchor_motion_px):
                    measured_anchor_motion_px.append(float(reanchor_motion_px))
                measured_entry = updated_entry
                diag = _entry_diag(measured_entry, idx)
                measured_anchor_reanchor_count += 1
                reanchor_status = "updated"
            else:
                measured_anchor_reanchor_fail_count += 1

        measured_fit_anchor, measured_reason, measured_anchor_metadata = _measured_fit_space_anchor(
            measured_entry,
            center=fit_center,
            detector_distance=detector_distance,
            pixel_size=pixel_size,
            gamma_deg=gamma_deg,
            Gamma_deg=Gamma_deg,
            a_lattice=a_lattice,
            c_lattice=c_lattice,
            wavelength=wavelength_scalar if np.isfinite(wavelength_scalar) else None,
        )
        if measured_fit_anchor is not None:
            if measured_reason == "cached_fit_space_anchor":
                fit_space_anchor_count_cached += 1
            elif measured_reason in {
                "detector_fit_space_anchor",
                "display_fit_space_anchor",
                "background_detector_fit_space_anchor",
            }:
                fit_space_anchor_count_detector += 1
        try:
            two_theta_adjustment_deg = float(
                measured_anchor_metadata.get("two_theta_adjustment_deg", np.nan)
            )
        except Exception:
            two_theta_adjustment_deg = float("nan")
        if np.isfinite(two_theta_adjustment_deg) and abs(two_theta_adjustment_deg) > 1.0e-12:
            fit_space_two_theta_adjustments.append(float(two_theta_adjustment_deg))
        measured_detector_anchor, detector_reason = _measured_detector_anchor(measured_entry)
        if measured_fit_anchor is None:
            residual_components[idx, 0] = float(missing_pair_penalty_deg) * float(priority_weight)
            residual_components[idx, 1] = 0.0
            missing_pairs += 1
            if collect_diagnostics:
                diagnostics.append(
                    {
                        **diag,
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "theta_initial_deg": float(theta_value),
                        "match_kind": "fixed_source",
                        "match_status": "missing_pair",
                        "resolution_kind": "measured_fit_anchor",
                        "resolution_reason": str(measured_reason),
                        "measured_anchor_source": str(
                            measured_anchor_metadata.get("anchor_source", measured_reason)
                        ),
                        "measured_reanchor_attempted": bool(reanchor_attempted),
                        "measured_reanchor_status": str(reanchor_status),
                        "measured_reanchor_motion_px": float(reanchor_motion_px),
                        "two_theta_adjustment_deg": float(two_theta_adjustment_deg),
                        "measured_x": (
                            float(measured_detector_anchor[0])
                            if measured_detector_anchor is not None
                            else float("nan")
                        ),
                        "measured_y": (
                            float(measured_detector_anchor[1])
                            if measured_detector_anchor is not None
                            else float("nan")
                        ),
                        "simulated_x": float("nan"),
                        "simulated_y": float("nan"),
                        "dx_px": float("nan"),
                        "dy_px": float("nan"),
                        "distance_px": float("nan"),
                        "measured_two_theta_deg": float("nan"),
                        "measured_phi_deg": float("nan"),
                        "delta_two_theta_deg": float("nan"),
                        "delta_phi_deg": float("nan"),
                        "angular_distance_deg": float("nan"),
                        "weighted_missing_penalty_deg": float(missing_pair_penalty_deg)
                        * float(priority_weight),
                        **priority_fields,
                    }
                )
            continue

        measured_two_theta = float(measured_fit_anchor[0])
        measured_phi = float(measured_fit_anchor[1])
        if sim_point is None:
            residual_components[idx, 0] = float(missing_pair_penalty_deg) * float(priority_weight)
            residual_components[idx, 1] = 0.0
            missing_pairs += 1
            if collect_diagnostics:
                diagnostics.append(
                    {
                        **diag,
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "theta_initial_deg": float(theta_value),
                        "match_kind": "fixed_source",
                        "match_status": "missing_pair",
                        "resolution_kind": "fixed_source",
                        "resolution_reason": str(sim_reason),
                        "measured_anchor_source": str(
                            measured_anchor_metadata.get("anchor_source", measured_reason)
                        ),
                        "measured_reanchor_attempted": bool(reanchor_attempted),
                        "measured_reanchor_status": str(reanchor_status),
                        "measured_reanchor_motion_px": float(reanchor_motion_px),
                        "two_theta_adjustment_deg": float(two_theta_adjustment_deg),
                        "measured_x": (
                            float(measured_detector_anchor[0])
                            if measured_detector_anchor is not None
                            else float("nan")
                        ),
                        "measured_y": (
                            float(measured_detector_anchor[1])
                            if measured_detector_anchor is not None
                            else float("nan")
                        ),
                        "simulated_x": float("nan"),
                        "simulated_y": float("nan"),
                        "dx_px": float("nan"),
                        "dy_px": float("nan"),
                        "distance_px": float("nan"),
                        "measured_two_theta_deg": float(measured_two_theta),
                        "measured_phi_deg": float(measured_phi),
                        "delta_two_theta_deg": float("nan"),
                        "delta_phi_deg": float("nan"),
                        "angular_distance_deg": float("nan"),
                        "weighted_missing_penalty_deg": float(missing_pair_penalty_deg)
                        * float(priority_weight),
                        **priority_fields,
                    }
                )
            continue

        fixed_source_resolved_count += 1
        sim_col = float(sim_point[0])
        sim_row = float(sim_point[1])
        sim_two_theta_arr, sim_phi_arr = _detector_pixels_to_fit_space(
            np.array([sim_col], dtype=np.float64),
            np.array([sim_row], dtype=np.float64),
            center=fit_center,
            detector_distance=detector_distance,
            pixel_size=pixel_size,
            gamma_deg=gamma_deg,
            Gamma_deg=Gamma_deg,
        )
        if (
            sim_two_theta_arr.size < 1
            or sim_phi_arr.size < 1
            or not np.isfinite(sim_two_theta_arr[0])
            or not np.isfinite(sim_phi_arr[0])
        ):
            residual_components[idx, 0] = float(missing_pair_penalty_deg) * float(priority_weight)
            residual_components[idx, 1] = 0.0
            missing_pairs += 1
            if collect_diagnostics:
                diagnostics.append(
                    {
                        **diag,
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "theta_initial_deg": float(theta_value),
                        "match_kind": "fixed_source",
                        "match_status": "missing_pair",
                        "resolution_kind": "fit_space",
                        "resolution_reason": "invalid_fit_space",
                        "measured_anchor_source": str(
                            measured_anchor_metadata.get("anchor_source", measured_reason)
                        ),
                        "measured_reanchor_attempted": bool(reanchor_attempted),
                        "measured_reanchor_status": str(reanchor_status),
                        "measured_reanchor_motion_px": float(reanchor_motion_px),
                        "two_theta_adjustment_deg": float(two_theta_adjustment_deg),
                        "measured_x": (
                            float(measured_detector_anchor[0])
                            if measured_detector_anchor is not None
                            else float("nan")
                        ),
                        "measured_y": (
                            float(measured_detector_anchor[1])
                            if measured_detector_anchor is not None
                            else float("nan")
                        ),
                        "simulated_x": float(sim_col),
                        "simulated_y": float(sim_row),
                        "dx_px": float("nan"),
                        "dy_px": float("nan"),
                        "distance_px": float("nan"),
                        "measured_two_theta_deg": float(measured_two_theta),
                        "measured_phi_deg": float(measured_phi),
                        "delta_two_theta_deg": float("nan"),
                        "delta_phi_deg": float("nan"),
                        "angular_distance_deg": float("nan"),
                        "weighted_missing_penalty_deg": float(missing_pair_penalty_deg)
                        * float(priority_weight),
                        **priority_fields,
                    }
                )
            continue

        sim_two_theta = float(sim_two_theta_arr[0])
        sim_phi = float(sim_phi_arr[0])
        delta_two_theta = float(sim_two_theta - measured_two_theta)
        delta_phi = float(_angular_difference_deg(sim_phi, measured_phi))
        residual_components[idx, 0] = float(priority_weight) * float(delta_two_theta)
        residual_components[idx, 1] = float(priority_weight) * float(delta_phi)
        matched_pair_count += 1
        if measured_detector_anchor is not None:
            pixel_dx = float(sim_col - float(measured_detector_anchor[0]))
            pixel_dy = float(sim_row - float(measured_detector_anchor[1]))
            pixel_distance = float(math.hypot(pixel_dx, pixel_dy))
        else:
            pixel_dx = float("nan")
            pixel_dy = float("nan")
            pixel_distance = float("nan")
        angular_distance = float(math.hypot(delta_two_theta, delta_phi))
        if np.isfinite(pixel_distance):
            pixel_distances.append(pixel_distance)
        angular_distances.append(angular_distance)
        if eligible_line_group_ids:
            line_match_records.append(
                {
                    **diag,
                    "group_ids": _geometry_line_group_ids(measured_entry),
                    "measured_fit_point": (
                        float(measured_two_theta),
                        float(measured_phi),
                    ),
                    "simulated_fit_point": (
                        float(sim_two_theta),
                        float(sim_phi),
                    ),
                }
            )
        live_cache_records.append(
            {
                **diag,
                "dataset_index": int(dataset_ctx.dataset_index),
                "dataset_label": str(dataset_ctx.label),
                "simulated_detector_x": float(sim_col),
                "simulated_detector_y": float(sim_row),
                "simulated_two_theta_deg": float(sim_two_theta),
                "simulated_phi_deg": float(sim_phi),
            }
        )

        if collect_diagnostics:
            diagnostics.append(
                {
                    **diag,
                    "dataset_index": int(dataset_ctx.dataset_index),
                    "dataset_label": str(dataset_ctx.label),
                    "theta_initial_deg": float(theta_value),
                    "match_kind": "fixed_source",
                    "match_status": "matched",
                    "resolution_kind": "fixed_source",
                    "resolution_reason": str(sim_reason),
                    "measured_resolution_reason": str(measured_reason),
                    "detector_resolution_reason": str(detector_reason),
                    "measured_anchor_source": str(
                        measured_anchor_metadata.get("anchor_source", measured_reason)
                    ),
                    "measured_reanchor_attempted": bool(reanchor_attempted),
                    "measured_reanchor_status": str(reanchor_status),
                    "measured_reanchor_motion_px": float(reanchor_motion_px),
                    "two_theta_adjustment_deg": float(two_theta_adjustment_deg),
                    "measured_x": (
                        float(measured_detector_anchor[0])
                        if measured_detector_anchor is not None
                        else float("nan")
                    ),
                    "measured_y": (
                        float(measured_detector_anchor[1])
                        if measured_detector_anchor is not None
                        else float("nan")
                    ),
                    "simulated_x": float(sim_col),
                    "simulated_y": float(sim_row),
                    "dx_px": float(pixel_dx),
                    "dy_px": float(pixel_dy),
                    "distance_px": float(pixel_distance),
                    "measured_two_theta_deg": float(measured_two_theta),
                    "measured_phi_deg": float(measured_phi),
                    "simulated_two_theta_deg": float(sim_two_theta),
                    "simulated_phi_deg": float(sim_phi),
                    "delta_two_theta_deg": float(delta_two_theta),
                    "delta_phi_deg": float(delta_phi),
                    "angular_distance_deg": float(angular_distance),
                    **priority_fields,
                }
            )

    residual_arr = residual_components.reshape(-1)
    line_summary: Dict[str, object] = {
        "line_group_count": 0,
        "resolved_line_group_count": 0,
        "missing_line_group_count": 0,
        "line_angle_rms_px": float("nan"),
        "line_offset_rms_px": float("nan"),
        "line_angle_rms_deg": float("nan"),
        "line_offset_rms_deg": float("nan"),
        "line_fit_space_span_deg_mean": float("nan"),
        "line_constraints_enabled": bool(line_constraints_enabled),
    }
    if bool(line_constraints_enabled):
        line_residual_arr, line_summary = (
            _build_geometry_line_constraint_residuals_from_fit_space_matches(
                line_match_records,
                eligible_line_group_ids,
                angle_weight=float(q_group_line_angle_weight),
                offset_weight=float(q_group_line_offset_weight),
                missing_penalty_deg=float(missing_pair_penalty_deg)
                * float(q_group_line_missing_penalty_scale),
            )
        )
        if line_residual_arr.size:
            residual_arr = np.concatenate((residual_arr, line_residual_arr))
    summary: Dict[str, object] = {
        "dataset_index": int(dataset_ctx.dataset_index),
        "dataset_label": str(dataset_ctx.label),
        "theta_initial_deg": float(theta_value),
        "measured_count": int(len(normalized_measured)),
        "fixed_source_resolved_count": int(fixed_source_resolved_count),
        "fallback_entry_count": 0,
        "fallback_hkl_count": 0,
        "matched_pair_count": int(matched_pair_count),
        "missing_pair_count": int(missing_pairs),
        "simulated_reflection_count": int(fit_miller.shape[0]),
        "total_reflection_count": int(simulation_subset.total_reflection_count),
        "fixed_source_reflection_count": int(simulation_subset.fixed_source_reflection_count),
        "subset_fallback_hkl_count": int(simulation_subset.fallback_hkl_count),
        "subset_reduced": bool(simulation_subset.reduced),
        "central_ray_mode": bool(central_ray_mode),
        "single_ray_enabled": False,
        "single_ray_forced_count": int(np.count_nonzero(single_ray_indices >= 0))
        if isinstance(single_ray_indices, np.ndarray)
        else 0,
        "center_row": float(local["center"][0])
        if len(local.get("center", [])) >= 2
        else float("nan"),
        "center_col": float(local["center"][1])
        if len(local.get("center", [])) >= 2
        else float("nan"),
        "peak_weighting_mode": "uniform",
        "custom_sigma_count": 0,
        "anisotropic_sigma_count": 0,
        "line_group_count": int(line_summary.get("line_group_count", 0)),
        "resolved_line_group_count": int(line_summary.get("resolved_line_group_count", 0)),
        "missing_line_group_count": int(line_summary.get("missing_line_group_count", 0)),
        "line_angle_rms_px": float(line_summary.get("line_angle_rms_px", np.nan)),
        "line_offset_rms_px": float(line_summary.get("line_offset_rms_px", np.nan)),
        "line_angle_rms_deg": float(line_summary.get("line_angle_rms_deg", np.nan)),
        "line_offset_rms_deg": float(line_summary.get("line_offset_rms_deg", np.nan)),
        "line_fit_space_span_deg_mean": float(
            line_summary.get("line_fit_space_span_deg_mean", np.nan)
        ),
        "line_constraints_enabled": bool(
            line_summary.get("line_constraints_enabled", line_constraints_enabled)
        ),
        "measured_anchor_reanchor_enabled": bool(dynamic_reanchor_enabled),
        "measured_anchor_reanchor_attempt_count": int(measured_anchor_reanchor_attempt_count),
        "measured_anchor_reanchor_count": int(measured_anchor_reanchor_count),
        "measured_anchor_reanchor_fail_count": int(measured_anchor_reanchor_fail_count),
        "_live_cache_records": live_cache_records,
        **_fit_space_provenance_summary(
            local,
            cached_anchor_count=fit_space_anchor_count_cached,
            detector_anchor_count=fit_space_anchor_count_detector,
            two_theta_adjustments_deg=fit_space_two_theta_adjustments,
        ),
    }
    if pixel_distances:
        pixel_dist_arr = np.asarray(pixel_distances, dtype=float)
        pixel_dist_arr = pixel_dist_arr[np.isfinite(pixel_dist_arr)]
        if pixel_dist_arr.size:
            summary["unweighted_peak_rms_px"] = float(
                np.sqrt(np.mean(pixel_dist_arr * pixel_dist_arr))
            )
            summary["unweighted_peak_mean_px"] = float(np.mean(pixel_dist_arr))
            summary["unweighted_peak_max_px"] = float(np.max(pixel_dist_arr))
        else:
            summary["unweighted_peak_rms_px"] = float("nan")
            summary["unweighted_peak_mean_px"] = float("nan")
            summary["unweighted_peak_max_px"] = float("nan")
    else:
        summary["unweighted_peak_rms_px"] = float("nan")
        summary["unweighted_peak_mean_px"] = float("nan")
        summary["unweighted_peak_max_px"] = float("nan")
    if angular_distances:
        angular_dist_arr = np.asarray(angular_distances, dtype=float)
        angular_dist_arr = angular_dist_arr[np.isfinite(angular_dist_arr)]
        if angular_dist_arr.size:
            summary["unweighted_peak_rms_deg"] = float(
                np.sqrt(np.mean(angular_dist_arr * angular_dist_arr))
            )
            summary["unweighted_peak_mean_deg"] = float(np.mean(angular_dist_arr))
            summary["unweighted_peak_max_deg"] = float(np.max(angular_dist_arr))
        else:
            summary["unweighted_peak_rms_deg"] = float("nan")
            summary["unweighted_peak_mean_deg"] = float("nan")
            summary["unweighted_peak_max_deg"] = float("nan")
    else:
        summary["unweighted_peak_rms_deg"] = float("nan")
        summary["unweighted_peak_mean_deg"] = float("nan")
        summary["unweighted_peak_max_deg"] = float("nan")
    if measured_anchor_motion_px:
        motion_arr = np.asarray(measured_anchor_motion_px, dtype=float)
        motion_arr = motion_arr[np.isfinite(motion_arr)]
        if motion_arr.size:
            summary["measured_anchor_motion_mean_px"] = float(np.mean(motion_arr))
            summary["measured_anchor_motion_rms_px"] = float(
                np.sqrt(np.mean(motion_arr * motion_arr))
            )
            summary["measured_anchor_motion_max_px"] = float(np.max(motion_arr))
        else:
            summary["measured_anchor_motion_mean_px"] = float("nan")
            summary["measured_anchor_motion_rms_px"] = float("nan")
            summary["measured_anchor_motion_max_px"] = float("nan")
    else:
        summary["measured_anchor_motion_mean_px"] = float("nan")
        summary["measured_anchor_motion_rms_px"] = float("nan")
        summary["measured_anchor_motion_max_px"] = float("nan")
    return residual_arr, diagnostics, summary


def _evaluate_geometry_fit_dataset_fixed_correspondences(
    local: Dict[str, object],
    dataset_ctx: GeometryFitDatasetContext,
    correspondences: Sequence[Mapping[str, object]],
    *,
    image_size: int,
    weighted_matching: bool,
    solver_f_scale: float,
    missing_pair_penalty: float,
    theta_value: float,
    use_measurement_uncertainty: bool = False,
    anisotropic_uncertainty: bool = False,
    radial_sigma_scale: float = 1.0,
    tangential_sigma_scale: float = 1.0,
    match_radius_px: float = np.inf,
    collect_diagnostics: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
    """Evaluate one dataset using frozen per-point correspondences."""

    normalized_entries = [dict(entry) for entry in correspondences or ()]
    central_ray_mode = _uses_central_geometry_ray(local.get("mosaic_params"))
    if not normalized_entries:
        return (
            np.array([], dtype=float),
            [],
            {
                "dataset_index": int(dataset_ctx.dataset_index),
                "dataset_label": str(dataset_ctx.label),
                "theta_initial_deg": float(theta_value),
                "measured_count": 0,
                "fixed_source_resolved_count": 0,
                "fallback_entry_count": 0,
                "matched_pair_count": 0,
                "missing_pair_count": 0,
                "simulated_reflection_count": int(dataset_ctx.subset.miller.shape[0]),
                "total_reflection_count": int(dataset_ctx.subset.total_reflection_count),
                "subset_reduced": bool(dataset_ctx.subset.reduced),
                "central_ray_mode": bool(central_ray_mode),
                "single_ray_enabled": False,
                "single_ray_forced_count": 0,
                "match_radius_px": float(match_radius_px),
            },
        )
    try:
        hk0_peak_priority_weight = float(local.get("_hk0_peak_priority_weight", 1.0))
    except Exception:
        hk0_peak_priority_weight = 1.0
    if not np.isfinite(hk0_peak_priority_weight) or hk0_peak_priority_weight < 1.0:
        hk0_peak_priority_weight = 1.0

    fit_miller = dataset_ctx.subset.miller
    fit_intensities = dataset_ctx.subset.intensities
    mosaic = local["mosaic_params"]
    wavelength_array = mosaic.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic.get("wavelength_i_array")

    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
    _, hit_tables, *_ = _process_peaks_parallel_safe(
        fit_miller,
        fit_intensities,
        image_size,
        local["a"],
        local["c"],
        wavelength_array,
        sim_buffer,
        local["corto_detector"],
        local["gamma"],
        local["Gamma"],
        local["chi"],
        local.get("psi", 0.0),
        local.get("psi_z", 0.0),
        local["zs"],
        local["zb"],
        local["n2"],
        mosaic["beam_x_array"],
        mosaic["beam_y_array"],
        mosaic["theta_array"],
        mosaic["phi_array"],
        mosaic["sigma_mosaic_deg"],
        mosaic["gamma_mosaic_deg"],
        mosaic["eta"],
        wavelength_array,
        local["debye_x"],
        local["debye_y"],
        local["center"],
        theta_value,
        local.get("cor_angle", 0.0),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        save_flag=0,
        **_simulation_kernel_kwargs(local, mosaic),
    )
    maxpos = hit_tables_to_max_positions(hit_tables)

    def _point_radius_px(point: Tuple[float, float]) -> float:
        try:
            center_row = float(local["center"][0])
            center_col = float(local["center"][1])
        except Exception:
            return float("nan")
        if not (
            np.isfinite(point[0])
            and np.isfinite(point[1])
            and np.isfinite(center_row)
            and np.isfinite(center_col)
        ):
            return float("nan")
        return float(math.hypot(float(point[0]) - center_col, float(point[1]) - center_row))

    residual_components = np.zeros((len(normalized_entries), 2), dtype=np.float64)
    diagnostics: List[Dict[str, object]] = []
    custom_sigma_values: List[float] = []
    matched_distances: List[float] = []
    matched_pair_count = 0
    missing_pairs = 0
    anisotropic_sigma_count = 0
    fixed_source_resolved_count = 0
    fallback_entry_count = 0

    def _add_diag(entry: Mapping[str, object], payload: Dict[str, object]) -> None:
        if not collect_diagnostics:
            return
        diag = dict(entry)
        diag.update(payload)
        diag["dataset_index"] = int(dataset_ctx.dataset_index)
        diag["dataset_label"] = str(dataset_ctx.label)
        diag["theta_initial_deg"] = float(theta_value)
        diagnostics.append(diag)

    def _weight_fields(
        entry: Mapping[str, object],
        *,
        measured_point: Tuple[float, float],
        dx: float = 0.0,
        dy: float = 0.0,
        distance_weight: float = 1.0,
    ) -> Dict[str, float | bool]:
        nonlocal anisotropic_sigma_count
        fields = _weight_measurement_residual(
            dx,
            dy,
            measured_point=measured_point,
            center=local.get("center", []),
            entry=entry,
            distance_weight=distance_weight,
            use_measurement_uncertainty=bool(use_measurement_uncertainty),
            anisotropic_enabled=bool(anisotropic_uncertainty),
            radial_scale=float(radial_sigma_scale),
            tangential_scale=float(tangential_sigma_scale),
        )
        if bool(fields.get("sigma_is_custom", False)):
            custom_sigma_values.append(float(fields.get("measurement_sigma_px", np.nan)))
        if bool(fields.get("anisotropic_sigma_used", False)):
            anisotropic_sigma_count += 1
        priority_fields = _geometry_peak_priority_metadata(
            entry,
            hk0_peak_priority_weight=float(hk0_peak_priority_weight),
        )
        priority_weight = float(priority_fields["priority_weight"])
        for key in (
            "weighted_dx_px",
            "weighted_dy_px",
            "weighted_radial_residual_px",
            "weighted_tangential_residual_px",
        ):
            try:
                value = float(fields.get(key, np.nan))
            except Exception:
                continue
            if np.isfinite(value):
                fields[key] = float(priority_weight * value)
        fields.update(priority_fields)
        return fields

    def _placement_error_px(entry: Mapping[str, object]) -> float:
        try:
            value = float(entry.get("placement_error_px", np.nan))
        except Exception:
            return float("nan")
        return float(value) if np.isfinite(value) else float("nan")

    effective_match_radius = float(match_radius_px)
    if not np.isfinite(effective_match_radius) or effective_match_radius <= 0.0:
        effective_match_radius = float("inf")

    for entry_index, entry in enumerate(normalized_entries):
        if str(entry.get("resolution_kind", "")).strip().lower() == "fixed_source":
            fixed_source_resolved_count += 1
        else:
            fallback_entry_count += 1

        try:
            measured_point = (
                float(entry.get("measured_x", np.nan)),
                float(entry.get("measured_y", np.nan)),
            )
        except Exception:
            measured_point = (float("nan"), float("nan"))

        if not (np.isfinite(measured_point[0]) and np.isfinite(measured_point[1])):
            priority_fields = _geometry_peak_priority_metadata(
                entry,
                hk0_peak_priority_weight=float(hk0_peak_priority_weight),
            )
            priority_weight = float(priority_fields["priority_weight"])
            missing_pairs += 1
            residual_components[entry_index, 0] = float(missing_pair_penalty) * float(
                priority_weight
            )
            _add_diag(
                entry,
                {
                    "match_kind": "fixed_correspondence",
                    "match_status": "missing_pair",
                    "resolution_reason": "invalid_measured_point",
                    "simulated_x": float("nan"),
                    "simulated_y": float("nan"),
                    "dx_px": float("nan"),
                    "dy_px": float("nan"),
                    "distance_px": float("nan"),
                    "weighted_missing_penalty_px": float(missing_pair_penalty)
                    * float(priority_weight),
                    "measured_radius_px": float("nan"),
                    "simulated_radius_px": float("nan"),
                    "weight": float(priority_weight),
                    **priority_fields,
                },
            )
            continue

        weight_fields = _weight_fields(
            entry,
            measured_point=measured_point,
            distance_weight=1.0,
        )
        penalty_weight = float(weight_fields.get("sigma_weight", 1.0))
        penalty_weight *= float(weight_fields.get("priority_weight", 1.0))
        missing_penalty = float(missing_pair_penalty) * penalty_weight
        simulated_point, simulated_reason = _geometry_fit_correspondence_simulated_point(
            entry,
            hit_tables=hit_tables,
            max_positions=maxpos,
        )

        pair_dist = float("nan")
        if simulated_point is not None:
            dx = float(simulated_point[0] - measured_point[0])
            dy = float(simulated_point[1] - measured_point[1])
            pair_dist = float(math.hypot(dx, dy))
            if np.isfinite(pair_dist) and pair_dist <= effective_match_radius:
                if weighted_matching:
                    distance_weight = 1.0 / math.sqrt(1.0 + (pair_dist / solver_f_scale) ** 2)
                else:
                    distance_weight = 1.0
                weight_fields = _weight_fields(
                    entry,
                    measured_point=measured_point,
                    dx=dx,
                    dy=dy,
                    distance_weight=float(distance_weight),
                )
                residual_components[entry_index, 0] = float(weight_fields["weighted_dx_px"])
                residual_components[entry_index, 1] = float(weight_fields["weighted_dy_px"])
                matched_pair_count += 1
                matched_distances.append(float(pair_dist))
                _add_diag(
                    entry,
                    {
                        "match_kind": "fixed_correspondence",
                        "match_status": "matched",
                        "simulated_x": float(simulated_point[0]),
                        "simulated_y": float(simulated_point[1]),
                        "dx_px": float(dx),
                        "dy_px": float(dy),
                        "distance_px": float(pair_dist),
                        "placement_error_px": _placement_error_px(entry),
                        "distance_weight": float(distance_weight),
                        "weight": float(
                            float(distance_weight)
                            * float(weight_fields.get("sigma_weight", 1.0))
                            * float(weight_fields.get("priority_weight", 1.0))
                        ),
                        "measured_radius_px": _point_radius_px(measured_point),
                        "simulated_radius_px": _point_radius_px(simulated_point),
                        "correspondence_resolution_reason": str(simulated_reason),
                        **weight_fields,
                    },
                )
                continue

        missing_pairs += 1
        residual_components[entry_index, 0] = float(missing_penalty)
        residual_components[entry_index, 1] = 0.0
        reason = (
            "match_radius_exceeded"
            if np.isfinite(pair_dist) and pair_dist > effective_match_radius
            else str(simulated_reason)
        )
        _add_diag(
            entry,
            {
                "match_kind": "fixed_correspondence",
                "match_status": "missing_pair",
                "simulated_x": (
                    float(simulated_point[0]) if simulated_point is not None else float("nan")
                ),
                "simulated_y": (
                    float(simulated_point[1]) if simulated_point is not None else float("nan")
                ),
                "dx_px": (
                    float(simulated_point[0] - measured_point[0])
                    if simulated_point is not None
                    else float("nan")
                ),
                "dy_px": (
                    float(simulated_point[1] - measured_point[1])
                    if simulated_point is not None
                    else float("nan")
                ),
                "distance_px": float(pair_dist) if np.isfinite(pair_dist) else float("nan"),
                "placement_error_px": _placement_error_px(entry),
                "distance_weight": 1.0,
                "weight": float(penalty_weight),
                "weighted_missing_penalty_px": float(missing_penalty),
                "measured_radius_px": _point_radius_px(measured_point),
                "simulated_radius_px": (
                    _point_radius_px(simulated_point)
                    if simulated_point is not None
                    else float("nan")
                ),
                "resolution_reason": str(reason),
                "correspondence_resolution_reason": str(simulated_reason),
                **weight_fields,
            },
        )

    residual_arr = residual_components.reshape(-1)
    summary: Dict[str, object] = {
        "dataset_index": int(dataset_ctx.dataset_index),
        "dataset_label": str(dataset_ctx.label),
        "theta_initial_deg": float(theta_value),
        "measured_count": int(len(normalized_entries)),
        "fixed_source_resolved_count": int(fixed_source_resolved_count),
        "fallback_entry_count": int(fallback_entry_count),
        "matched_pair_count": int(matched_pair_count),
        "missing_pair_count": int(missing_pairs),
        "simulated_reflection_count": int(fit_miller.shape[0]),
        "total_reflection_count": int(dataset_ctx.subset.total_reflection_count),
        "fixed_source_reflection_count": int(dataset_ctx.subset.fixed_source_reflection_count),
        "subset_fallback_hkl_count": int(dataset_ctx.subset.fallback_hkl_count),
        "subset_reduced": bool(dataset_ctx.subset.reduced),
        "central_ray_mode": bool(central_ray_mode),
        "single_ray_enabled": False,
        "single_ray_forced_count": 0,
        "center_row": float(local["center"][0])
        if len(local.get("center", [])) >= 2
        else float("nan"),
        "center_col": float(local["center"][1])
        if len(local.get("center", [])) >= 2
        else float("nan"),
        "match_radius_px": float(effective_match_radius),
        "line_group_count": 0,
        "resolved_line_group_count": 0,
        "missing_line_group_count": 0,
        "line_angle_rms_px": float("nan"),
        "line_offset_rms_px": float("nan"),
        "line_fit_space_span_deg_mean": float("nan"),
        "line_constraints_enabled": False,
    }
    summary["anisotropic_sigma_count"] = int(anisotropic_sigma_count)
    if custom_sigma_values:
        sigma_arr = np.asarray(custom_sigma_values, dtype=float)
        sigma_arr = sigma_arr[np.isfinite(sigma_arr) & (sigma_arr > 0.0)]
        if sigma_arr.size:
            summary["custom_sigma_count"] = int(sigma_arr.size)
            summary["measurement_sigma_median_px"] = float(np.median(sigma_arr))
            summary["measurement_sigma_mean_px"] = float(np.mean(sigma_arr))
            summary["measurement_sigma_max_px"] = float(np.max(sigma_arr))
        else:
            summary["custom_sigma_count"] = 0
    else:
        summary["custom_sigma_count"] = 0
    if matched_distances:
        matched_dist_arr = np.asarray(matched_distances, dtype=float)
        matched_dist_arr = matched_dist_arr[np.isfinite(matched_dist_arr)]
        if matched_dist_arr.size:
            summary["unweighted_peak_rms_px"] = float(
                np.sqrt(np.mean(matched_dist_arr * matched_dist_arr))
            )
            summary["unweighted_peak_mean_px"] = float(np.mean(matched_dist_arr))
            summary["unweighted_peak_max_px"] = float(np.max(matched_dist_arr))
        else:
            summary["unweighted_peak_rms_px"] = float("nan")
            summary["unweighted_peak_mean_px"] = float("nan")
            summary["unweighted_peak_max_px"] = float("nan")
    else:
        summary["unweighted_peak_rms_px"] = float("nan")
        summary["unweighted_peak_mean_px"] = float("nan")
        summary["unweighted_peak_max_px"] = float("nan")
    if anisotropic_sigma_count > 0:
        summary["peak_weighting_mode"] = (
            "measurement_covariance+distance" if weighted_matching else "measurement_covariance"
        )
    elif summary.get("custom_sigma_count", 0):
        summary["peak_weighting_mode"] = (
            "measurement_sigma+distance" if weighted_matching else "measurement_sigma"
        )
    else:
        summary["peak_weighting_mode"] = "uniform"
    return residual_arr, diagnostics, summary


def _local_peak_snr(
    image: np.ndarray,
    row: int,
    col: int,
    roi_half_width: int,
) -> float:
    """Estimate local peak SNR using robust statistics in the candidate patch."""

    h, w = image.shape
    r0 = max(0, row - roi_half_width)
    r1 = min(h, row + roi_half_width + 1)
    c0 = max(0, col - roi_half_width)
    c1 = min(w, col + roi_half_width + 1)
    if r0 >= r1 or c0 >= c1:
        return 0.0

    patch = np.asarray(image[r0:r1, c0:c1], dtype=np.float64)
    finite = patch[np.isfinite(patch)]
    if finite.size < 5:
        return 0.0

    baseline = float(np.median(finite))
    mad = float(np.median(np.abs(finite - baseline)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = float(np.std(finite))
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = 1.0

    signal = float(np.max(finite) - baseline)
    if not np.isfinite(signal):
        return 0.0
    return max(0.0, signal / sigma)


def _robust_cost(residual: np.ndarray, loss: str, f_scale: float) -> float:
    """Compute the least_squares robust objective for a residual vector."""

    residual = np.asarray(residual, dtype=np.float64)
    if residual.size == 0:
        return 0.0
    f_scale = max(float(f_scale), 1e-12)
    z = (residual / f_scale) ** 2
    loss_key = str(loss).strip().lower()

    if loss_key == "linear":
        rho = z
    elif loss_key == "soft_l1":
        rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
    elif loss_key == "huber":
        rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
    elif loss_key == "cauchy":
        rho = np.log1p(z)
    elif loss_key == "arctan":
        rho = np.arctan(z)
    else:
        raise ValueError(f"Unsupported loss '{loss}'.")
    return 0.5 * (f_scale * f_scale) * float(np.sum(rho))


def _simulate_with_cache(
    params: Dict[str, float],
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    cache: SimulationCache,
) -> Tuple[np.ndarray, np.ndarray]:
    cached = cache.get(params)
    if cached is not None:
        return cached

    buffer = np.zeros((image_size, image_size), dtype=np.float64)

    mosaic = params["mosaic_params"]
    wavelength_array = mosaic.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic.get("wavelength_i_array")
    if wavelength_array is None:
        wavelength_array = params.get("lambda")

    image, hit_tables, *_ = _process_peaks_parallel_safe(
        miller,
        intensities,
        image_size,
        params["a"],
        params["c"],
        wavelength_array,
        buffer,
        params["corto_detector"],
        params["gamma"],
        params["Gamma"],
        params["chi"],
        params.get("psi", 0.0),
        params.get("psi_z", 0.0),
        params["zs"],
        params["zb"],
        params["n2"],
        mosaic["beam_x_array"],
        mosaic["beam_y_array"],
        mosaic["theta_array"],
        mosaic["phi_array"],
        mosaic["sigma_mosaic_deg"],
        mosaic["gamma_mosaic_deg"],
        mosaic["eta"],
        wavelength_array,
        params["debye_x"],
        params["debye_y"],
        params["center"],
        params["theta_initial"],
        params.get("cor_angle", 0.0),
        params.get("uv1", np.array([1.0, 0.0, 0.0])),
        params.get("uv2", np.array([0.0, 1.0, 0.0])),
        save_flag=0,
        **_simulation_kernel_kwargs(params, mosaic),
    )

    image = np.asarray(image, dtype=np.float64)
    maxpos = hit_tables_to_max_positions(hit_tables)

    cache.store(params, image, maxpos)
    return image, maxpos


def fit_mosaic_widths_separable(
    experimental_image: np.ndarray,
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, float],
    *,
    num_peaks: int = 36,
    roi_half_width: int = 8,
    min_peak_separation: Optional[float] = None,
    stratify: Optional[str] = None,
    stratify_bins: int = 6,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
    max_nfev: int = 80,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    measured_peaks: Optional[Sequence[object]] = None,
    peak_source: str = "geometry",
    max_restarts: int = 2,
    roi_normalization: str = "sqrt_npix",
) -> OptimizeResult:
    r"""Estimate mosaic pseudo-Voigt widths using separable non-linear least squares.

    Geometry, detector center, and per-peak locations are assumed to be fixed. Only
    the pseudo-Voigt mosaic widths (:math:`\sigma`, :math:`\gamma`) and mixing
    parameter :math:`\eta` are refined.  Peak amplitudes and a constant
    background are eliminated analytically within each ROI so that intensity
    outliers do not bias the width estimates.

    Parameters
    ----------
    experimental_image:
        Measured detector image at full resolution.
    miller, intensities:
        Arrays describing the simulated reflections.  These should match the
        arrays used to generate ``experimental_image``.
    image_size:
        Detector dimension (assumed square).
    params:
        Dictionary of simulation parameters.  ``params['mosaic_params']`` must
        contain the beam sample arrays plus initial values for
        ``sigma_mosaic_deg``, ``gamma_mosaic_deg`` and ``eta``.
    num_peaks:
        Number of peaks to include in the separable fit.
    roi_half_width:
        Half-width (in pixels) of the square ROI around each selected peak.
    min_peak_separation:
        Minimum Euclidean distance between ROI centres.  Defaults to ``2.5``
        times ``roi_half_width`` when ``None``.
    stratify:
        ``None`` (default) selects the globally brightest peaks.  ``"L"``
        enforces round-robin selection across distinct :math:`L` values while
        ``"twotheta"`` stratifies by equal-width 2θ bins.
    stratify_bins:
        Maximum number of bins when ``stratify='twotheta'``.
    loss, f_scale, max_nfev, bounds:
        Directly forwarded to :func:`scipy.optimize.least_squares`.
    measured_peaks:
        Optional measured peak list from geometry fitting. Entries may be dicts
        with ``label``/``x``/``y`` (or ``hkl``/``x``/``y``), or 5-tuples
        ``(h, k, l, x, y)``.
    peak_source:
        Candidate source strategy: ``"geometry"``, ``"auto"``, or ``"hybrid"``.
    max_restarts:
        Number of jittered restart solves after the primary solve.
    roi_normalization:
        ``"sqrt_npix"`` (default) scales each ROI residual block by the square
        root of its pixel count; ``"none"`` leaves raw residuals unchanged.

    Returns
    -------
    OptimizeResult
        ``x`` holds the refined ``[sigma_deg, gamma_deg, eta]`` vector.  The
        ``best_params`` attribute mirrors ``params`` with the optimized mosaic
        values inserted, and ``selected_rois`` enumerates the ROIs used during the
        fit. Additional attributes include diagnostics (`roi_diagnostics`,
        `rejected_rois`, `initial_cost`, `final_cost`, ...).
    """

    experimental_image = np.asarray(experimental_image, dtype=np.float64)
    if experimental_image.shape != (image_size, image_size):
        raise ValueError("experimental_image shape must match the provided image_size")

    peak_source_key = str(peak_source).strip().lower()
    if peak_source_key not in {"geometry", "auto", "hybrid"}:
        raise ValueError("peak_source must be one of {'geometry', 'auto', 'hybrid'}")

    roi_norm_key = str(roi_normalization).strip().lower()
    if roi_norm_key not in {"sqrt_npix", "none"}:
        raise ValueError("roi_normalization must be 'sqrt_npix' or 'none'")

    max_restarts = max(0, int(max_restarts))

    miller = np.asarray(miller, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    if miller.ndim != 2 or miller.shape[1] != 3:
        raise ValueError("miller must be an array of shape (N, 3)")
    if intensities.shape[0] != miller.shape[0]:
        raise ValueError("intensities and miller must have matching lengths")

    allowed_mask = _allowed_reflection_mask(miller)
    allowed_indices = np.flatnonzero(allowed_mask)
    if allowed_indices.size == 0:
        raise RuntimeError("No reflections satisfy 2h + k ≡ 0 (mod 3) for mosaic-width fitting")

    mosaic_params = dict(params.get("mosaic_params", {}))
    if not mosaic_params:
        raise ValueError("params['mosaic_params'] is required")

    beam_x = np.asarray(mosaic_params.get("beam_x_array"), dtype=np.float64)
    beam_y = np.asarray(mosaic_params.get("beam_y_array"), dtype=np.float64)
    theta_array = np.asarray(mosaic_params.get("theta_array"), dtype=np.float64)
    phi_array = np.asarray(mosaic_params.get("phi_array"), dtype=np.float64)
    if not (beam_x.size and beam_y.size and theta_array.size and phi_array.size):
        raise ValueError("mosaic_params must include beam and divergence samples")

    wavelength_array = mosaic_params.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic_params.get("wavelength_i_array")
    if wavelength_array is None:
        base_lambda = float(params.get("lambda", 1.0))
        wavelength_array = np.full(beam_x.shape, base_lambda, dtype=np.float64)
    else:
        wavelength_array = np.asarray(wavelength_array, dtype=np.float64)

    sigma0 = float(mosaic_params.get("sigma_mosaic_deg", 0.5))
    gamma0 = float(mosaic_params.get("gamma_mosaic_deg", 0.5))
    eta0 = float(mosaic_params.get("eta", 0.05))
    solve_q_steps = int(np.clip(int(mosaic_params.get("solve_q_steps", 1000)), 32, 8192))
    solve_q_rel_tol = float(
        np.clip(float(mosaic_params.get("solve_q_rel_tol", 5.0e-4)), 1.0e-6, 5.0e-2)
    )
    solve_q_mode = int(mosaic_params.get("solve_q_mode", 1))
    if solve_q_mode != 0:
        solve_q_mode = 1

    roi_half_width = int(roi_half_width)
    if roi_half_width <= 0:
        raise ValueError("roi_half_width must be a positive integer")
    if min_peak_separation is None:
        min_peak_separation = 2.5 * float(roi_half_width)
    min_peak_separation = float(min_peak_separation)
    min_peak_separation_sq = min_peak_separation * min_peak_separation

    num_peaks = int(num_peaks)
    if num_peaks <= 0:
        raise ValueError("num_peaks must be positive")

    a_lattice = float(params.get("a", 1.0))
    c_lattice = float(params.get("c", 1.0))
    lambda_scalar = float(params.get("lambda", float(np.mean(wavelength_array))))

    gamma_deg = float(params.get("gamma", 0.0))
    Gamma_deg = float(params.get("Gamma", 0.0))
    chi_deg = float(params.get("chi", 0.0))
    psi_deg = float(params.get("psi", 0.0))
    psi_z_deg = float(params.get("psi_z", 0.0))
    zs = float(params.get("zs", 0.0))
    zb = float(params.get("zb", 0.0))
    n2 = params.get("n2")
    if n2 is None:
        raise ValueError("params['n2'] (complex index of refraction) is required")
    debye_x = float(params.get("debye_x", 0.0))
    debye_y = float(params.get("debye_y", 0.0))
    theta_initial = float(params.get("theta_initial", 0.0))
    cor_angle = float(params.get("cor_angle", 0.0))
    corto_detector = float(params.get("corto_detector"))
    center = tuple(params.get("center", (image_size / 2.0, image_size / 2.0)))
    unit_x = np.asarray(params.get("uv1", np.array([1.0, 0.0, 0.0])), dtype=np.float64)
    n_detector = np.asarray(params.get("uv2", np.array([0.0, 1.0, 0.0])), dtype=np.float64)

    full_buffer = np.zeros((image_size, image_size), dtype=np.float64)

    def _simulate(
        miller_subset: np.ndarray,
        intens_subset: np.ndarray,
        sigma_deg: float,
        gamma_deg_: float,
        eta_: float,
        buffer: np.ndarray,
        *,
        record_hits: bool = False,
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        buffer.fill(0.0)
        image, hit_tables, *_ = _process_peaks_parallel_safe(
            miller_subset,
            intens_subset,
            image_size,
            a_lattice,
            c_lattice,
            wavelength_array,
            buffer,
            corto_detector,
            gamma_deg,
            Gamma_deg,
            chi_deg,
            psi_deg,
            psi_z_deg,
            zs,
            zb,
            n2,
            beam_x,
            beam_y,
            theta_array,
            phi_array,
            sigma_deg,
            gamma_deg_,
            eta_,
            wavelength_array,
            debye_x,
            debye_y,
            center,
            theta_initial,
            cor_angle,
            unit_x,
            n_detector,
            0,
            **_simulation_kernel_kwargs(
                {
                    "optics_mode": params.get("optics_mode", 0),
                    "sample_depth_m": params.get("sample_depth_m", params.get("thickness", 0.0)),
                    "pixel_size_m": params.get("pixel_size_m", params.get("pixel_size", 100e-6)),
                    "sample_width_m": params.get("sample_width_m", 0.0),
                    "sample_length_m": params.get("sample_length_m", 0.0),
                },
                {
                    "solve_q_steps": solve_q_steps,
                    "solve_q_rel_tol": solve_q_rel_tol,
                    "solve_q_mode": solve_q_mode,
                    "n2_sample_array": params.get("mosaic_params", {}).get("n2_sample_array"),
                },
            ),
        )
        image = np.asarray(image, dtype=np.float64)
        if not record_hits:
            return image, None
        hits_py = [np.asarray(tbl) for tbl in hit_tables]
        return image, hits_py

    allowed_miller = np.ascontiguousarray(miller[allowed_indices], dtype=np.float64)
    allowed_intensities = np.ascontiguousarray(intensities[allowed_indices], dtype=np.float64)

    two_theta_limit = 65.0
    kept_rows: List[int] = []
    allowed_two_theta: List[float] = []
    for local_idx, hkl_row in enumerate(allowed_miller):
        hkl_int = tuple(int(round(val)) for val in hkl_row)
        if all(v == 0 for v in hkl_int):
            continue
        d_hkl = d_spacing(hkl_int[0], hkl_int[1], hkl_int[2], a_lattice, c_lattice)
        tth_val = two_theta(d_hkl, lambda_scalar)
        if tth_val is None or not np.isfinite(tth_val):
            continue
        h_zero = hkl_int[0] == 0 and hkl_int[1] == 0
        if (not h_zero) and tth_val > two_theta_limit:
            continue
        kept_rows.append(local_idx)
        allowed_two_theta.append(float(tth_val) if tth_val is not None else float("nan"))

    if not kept_rows:
        raise RuntimeError("No reflections satisfy the 2h + k ≡ 0 constraint within 65° 2θ")

    allowed_indices = allowed_indices[kept_rows]
    allowed_miller = allowed_miller[kept_rows]
    allowed_intensities = allowed_intensities[kept_rows]
    allowed_two_theta = np.asarray(allowed_two_theta, dtype=np.float64)

    hkl_lookup: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    for local_idx, hkl_row in enumerate(allowed_miller):
        hkl = tuple(int(round(v)) for v in hkl_row)
        weight = abs(float(allowed_intensities[local_idx]))
        prev = hkl_lookup.get(hkl)
        if prev is None or weight > float(prev["weight"]):
            hkl_lookup[hkl] = {
                "reflection_index": float(allowed_indices[local_idx]),
                "weight": weight,
                "two_theta": float(allowed_two_theta[local_idx]),
                "L": float(hkl[2]),
            }

    rejected_rois: List[Dict[str, object]] = []
    candidates_auto: List[Dict[str, object]] = []
    if peak_source_key in {"auto", "hybrid"}:
        _, hit_tables = _simulate(
            allowed_miller,
            allowed_intensities,
            sigma0,
            gamma0,
            eta0,
            full_buffer,
            record_hits=True,
        )

        if not hit_tables:
            raise RuntimeError("Initial simulation produced no peak information")

        for local_idx, tbl in enumerate(hit_tables):
            arr = np.asarray(tbl, dtype=np.float64)
            if arr.size == 0:
                continue
            for row in arr:
                intensity = float(row[0])
                col = float(row[1])
                row_pix = float(row[2])
                if not (np.isfinite(intensity) and np.isfinite(col) and np.isfinite(row_pix)):
                    continue
                hkl = (int(round(row[4])), int(round(row[5])), int(round(row[6])))
                if all(v == 0 for v in hkl):
                    continue
                h, k, _ = hkl
                if (2 * h + k) % 3 != 0:
                    continue
                tth = float(allowed_two_theta[local_idx])
                candidates_auto.append(
                    {
                        "reflection_index": int(allowed_indices[local_idx]),
                        "intensity": intensity,
                        "row": row_pix,
                        "col": col,
                        "hkl": hkl,
                        "L": hkl[2],
                        "two_theta": tth,
                        "source": "auto",
                    }
                )

    normalized_measured = _normalize_measured_peaks(measured_peaks)
    candidates_geom: List[Dict[str, object]] = []
    if peak_source_key in {"geometry", "hybrid"}:
        if not normalized_measured and peak_source_key == "geometry":
            raise RuntimeError(
                "Geometry-locked mosaic fit requires measured geometry peaks; run geometry fitting first."
            )

        seen_geom: set[Tuple[int, int, int]] = set()
        for entry in normalized_measured:
            hkl = entry["hkl"]  # type: ignore[assignment]
            lookup = hkl_lookup.get(hkl)  # type: ignore[arg-type]
            if lookup is None:
                rejected_rois.append(
                    {
                        "stage": "candidate",
                        "reason": "hkl_not_in_current_reflections",
                        "hkl": hkl,
                        "x": float(entry["x"]),
                        "y": float(entry["y"]),
                        "source": "geometry",
                    }
                )
                continue
            row_pix = float(entry["y"])
            col = float(entry["x"])
            key = (
                int(lookup["reflection_index"]),
                int(round(row_pix)),
                int(round(col)),
            )
            if key in seen_geom:
                continue
            seen_geom.add(key)
            candidates_geom.append(
                {
                    "reflection_index": int(lookup["reflection_index"]),
                    "intensity": float(lookup["weight"]),
                    "row": row_pix,
                    "col": col,
                    "hkl": hkl,
                    "L": int(round(lookup["L"])),
                    "two_theta": float(lookup["two_theta"]),
                    "source": "geometry",
                }
            )

    if peak_source_key == "geometry":
        candidates: List[Dict[str, object]] = list(candidates_geom)
    elif peak_source_key == "auto":
        candidates = list(candidates_auto)
    else:
        candidates = list(candidates_geom)
        seen_hybrid = {
            (int(c["reflection_index"]), int(round(float(c["row"]))), int(round(float(c["col"]))))
            for c in candidates
        }
        for cand in candidates_auto:
            key = (
                int(cand["reflection_index"]),
                int(round(float(cand["row"]))),
                int(round(float(cand["col"]))),
            )
            if key in seen_hybrid:
                continue
            seen_hybrid.add(key)
            candidates.append(cand)

    if not candidates:
        raise RuntimeError("No peak candidates available for ROI selection")

    for cand in candidates:
        row_idx = int(round(float(cand["row"])))
        col_idx = int(round(float(cand["col"])))
        snr = _local_peak_snr(experimental_image, row_idx, col_idx, roi_half_width)
        cand["snr"] = float(snr)

    score_key = "snr" if peak_source_key in {"geometry", "hybrid"} else "intensity"
    for cand in candidates:
        cand["score"] = float(cand.get(score_key, cand.get("intensity", 0.0)))

    def _candidate_iter() -> Iterable[Dict[str, float]]:
        default_stratify = "twotheta" if peak_source_key in {"geometry", "hybrid"} else "none"
        key = (stratify or default_stratify).lower()
        if key not in {"none", "l", "twotheta"}:
            raise ValueError("stratify must be None, 'L', or 'twotheta'")
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (
                float(c.get("score", 0.0)),
                float(c.get("intensity", 0.0)),
            ),
            reverse=True,
        )
        if key == "none" or len(sorted_candidates) <= 1:
            return sorted_candidates

        if key == "l":
            groups: Dict[int, List[Dict[str, float]]] = {}
            for cand in sorted_candidates:
                groups.setdefault(int(cand["L"]), []).append(cand)
            for group in groups.values():
                group.sort(key=lambda c: float(c["score"]), reverse=True)
            order = sorted(groups.keys(), key=lambda val: (abs(val), val))
            round_robin: List[Dict[str, float]] = []
            while True:
                progressed = False
                for val in list(order):
                    group = groups.get(val)
                    if not group:
                        continue
                    round_robin.append(group.pop(0))
                    progressed = True
                    if not group:
                        groups.pop(val)
                if not progressed:
                    break
            return round_robin

        # stratify == 'twotheta'
        tth_values = [
            float(c["two_theta"])
            for c in sorted_candidates
            if c.get("two_theta") is not None and np.isfinite(float(c["two_theta"]))
        ]
        if not tth_values or np.allclose(tth_values, tth_values[0]):
            return sorted_candidates
        num_bins = max(1, min(int(stratify_bins), len(tth_values)))
        if num_bins == 1:
            return sorted_candidates
        edges = np.linspace(min(tth_values), max(tth_values), num_bins + 1)
        bins: Dict[int, List[Dict[str, float]]] = {i: [] for i in range(num_bins)}
        for cand in sorted_candidates:
            tth = cand["two_theta"]
            if tth is None or not np.isfinite(tth):
                bin_idx = 0
            else:
                bin_idx = int(np.searchsorted(edges, tth, side="right") - 1)
                bin_idx = max(0, min(num_bins - 1, bin_idx))
            bins[bin_idx].append(cand)
        for group in bins.values():
            group.sort(key=lambda c: float(c["score"]), reverse=True)
        round_robin: List[Dict[str, float]] = []
        active_bins = [idx for idx in range(num_bins) if bins[idx]]
        while active_bins:
            progressed = False
            for idx in list(active_bins):
                group = bins[idx]
                if not group:
                    active_bins.remove(idx)
                    continue
                round_robin.append(group.pop(0))
                progressed = True
                if not group:
                    active_bins.remove(idx)
            if not progressed:
                break
        return round_robin if round_robin else sorted_candidates

    ordered_candidates = list(_candidate_iter())

    rois: List[PeakROI] = []
    selected_centres: List[Tuple[float, float]] = []
    for cand in ordered_candidates:
        if len(rois) >= num_peaks:
            break
        row_c = float(cand["row"])
        col_c = float(cand["col"])
        row_idx = int(round(row_c))
        col_idx = int(round(col_c))
        if (
            row_idx - roi_half_width < 0
            or row_idx + roi_half_width >= image_size
            or col_idx - roi_half_width < 0
            or col_idx + roi_half_width >= image_size
        ):
            rejected_rois.append(
                {
                    "stage": "roi",
                    "reason": "out_of_bounds",
                    "hkl": tuple(int(v) for v in cand["hkl"]),
                    "center": (row_c, col_c),
                    "source": str(cand.get("source", "auto")),
                }
            )
            continue
        if selected_centres:
            if any(
                (row_c - r) ** 2 + (col_c - c) ** 2 < min_peak_separation_sq
                for r, c in selected_centres
            ):
                rejected_rois.append(
                    {
                        "stage": "roi",
                        "reason": "overlap",
                        "hkl": tuple(int(v) for v in cand["hkl"]),
                        "center": (row_c, col_c),
                        "source": str(cand.get("source", "auto")),
                    }
                )
                continue

        rows = np.arange(row_idx - roi_half_width, row_idx + roi_half_width + 1, dtype=int)
        cols = np.arange(col_idx - roi_half_width, col_idx + roi_half_width + 1, dtype=int)
        patch = experimental_image[np.ix_(rows, cols)]
        flat_patch = patch.ravel()
        valid_idx = np.flatnonzero(np.isfinite(flat_patch))
        if valid_idx.size == 0:
            rejected_rois.append(
                {
                    "stage": "roi",
                    "reason": "no_finite_pixels",
                    "hkl": tuple(int(v) for v in cand["hkl"]),
                    "center": (row_c, col_c),
                    "source": str(cand.get("source", "auto")),
                }
            )
            continue
        observed = flat_patch.take(valid_idx)
        observed = observed.astype(np.float64, copy=False)
        observed_sum = float(observed.sum())
        num_valid = int(observed.size)
        observed_mean = observed_sum / num_valid
        rois.append(
            PeakROI(
                reflection_index=int(cand["reflection_index"]),
                hkl=cand["hkl"],
                center=(row_c, col_c),
                row_indices=rows,
                col_indices=cols,
                flat_indices=valid_idx,
                observed=observed,
                observed_sum=observed_sum,
                observed_mean=observed_mean,
                num_pixels=num_valid,
                simulated_intensity=float(cand.get("intensity", 0.0)),
                candidate_snr=float(cand.get("snr", np.nan)),
                source=str(cand.get("source", "auto")),
                score=float(cand.get("score", 0.0)),
            )
        )
        selected_centres.append((row_c, col_c))

    if not rois:
        raise RuntimeError("No valid ROIs were selected for mosaic fitting")
    if peak_source_key == "geometry" and len(rois) < 8:
        raise RuntimeError(
            f"Geometry-locked mosaic fit needs at least 8 valid ROIs; got {len(rois)}."
        )

    unique_indices = sorted({roi.reflection_index for roi in rois})
    subset_miller = np.ascontiguousarray(miller[unique_indices], dtype=np.float64)
    subset_intensities = np.ones(subset_miller.shape[0], dtype=np.float64)

    subset_buffer = np.zeros((image_size, image_size), dtype=np.float64)

    def _evaluate_residual(
        theta: np.ndarray,
        *,
        collect_diagnostics: bool = False,
    ) -> Tuple[np.ndarray, Optional[List[Dict[str, object]]]]:
        sigma_deg, gamma_deg_local, eta_local = map(float, theta)
        sim_image, _ = _simulate(
            subset_miller,
            subset_intensities,
            sigma_deg,
            gamma_deg_local,
            eta_local,
            subset_buffer,
            record_hits=False,
        )

        residual_blocks: List[np.ndarray] = []
        diagnostics: List[Dict[str, object]] = []
        for roi in rois:
            block = sim_image[np.ix_(roi.row_indices, roi.col_indices)]
            flat = block.ravel()
            template = flat.take(roi.flat_indices)
            fallback_reason = ""
            amp = 0.0
            bkg = roi.observed_mean

            if template.size == 0:
                residual_raw = roi.observed - roi.observed_mean
                fallback_reason = "empty_template"
            else:
                tt = float(np.dot(template, template))
                if tt <= 1e-16:
                    residual_raw = roi.observed - roi.observed_mean
                    fallback_reason = "low_template_energy"
                else:
                    to = float(np.sum(template))
                    ty = float(np.dot(template, roi.observed))
                    oy = roi.observed_sum
                    oo = float(roi.num_pixels)
                    det = tt * oo - to * to
                    if abs(det) <= 1e-14 * tt * oo:
                        residual_raw = roi.observed - roi.observed_mean
                        fallback_reason = "ill_conditioned_affine"
                    else:
                        amp = (ty * oo - oy * to) / det
                        bkg = (tt * oy - to * ty) / det
                        model = amp * template + bkg
                        residual_raw = roi.observed - model

            if roi_norm_key == "sqrt_npix":
                norm = math.sqrt(max(float(roi.num_pixels), 1.0))
            else:
                norm = 1.0
            residual_normed = residual_raw / norm
            residual_blocks.append(residual_normed)

            if collect_diagnostics:
                rms = float(np.sqrt(np.mean(residual_raw * residual_raw)))
                diagnostics.append(
                    {
                        "hkl": tuple(int(v) for v in roi.hkl),
                        "reflection_index": int(roi.reflection_index),
                        "center": (float(roi.center[0]), float(roi.center[1])),
                        "num_pixels": int(roi.num_pixels),
                        "source": roi.source,
                        "candidate_snr": float(roi.candidate_snr),
                        "score": float(roi.score),
                        "amp": float(amp),
                        "bkg": float(bkg),
                        "rms": rms,
                        "fallback_reason": fallback_reason,
                        "normalization": roi_norm_key,
                        "outlier": False,
                    }
                )

        if not residual_blocks:
            empty = np.zeros(1, dtype=np.float64)
            if collect_diagnostics:
                return empty, diagnostics
            return empty, None
        merged = np.concatenate(residual_blocks)
        if collect_diagnostics:
            return merged, diagnostics
        return merged, None

    def residual(theta: np.ndarray) -> np.ndarray:
        out, _ = _evaluate_residual(theta, collect_diagnostics=False)
        return out

    if bounds is None:
        bounds = (
            np.array([0.03, 0.03, 0.0], dtype=np.float64),
            np.array([3.0, 3.0, 1.0], dtype=np.float64),
        )
    lower = np.asarray(bounds[0], dtype=np.float64).reshape(-1)
    upper = np.asarray(bounds[1], dtype=np.float64).reshape(-1)
    if lower.size != 3 or upper.size != 3:
        raise ValueError("bounds must contain exactly 3 lower/upper values")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("bounds must be finite")
    if np.any(lower >= upper):
        raise ValueError("each lower bound must be strictly less than upper bound")

    x0 = np.array([sigma0, gamma0, eta0], dtype=np.float64)
    x0 = np.clip(x0, lower, upper)

    def _run_solver(x_start: np.ndarray) -> OptimizeResult:
        return least_squares(
            residual,
            np.asarray(x_start, dtype=np.float64),
            bounds=(lower, upper),
            loss=loss,
            f_scale=f_scale,
            max_nfev=int(max_nfev),
        )

    initial_residual = residual(x0)
    initial_cost = _robust_cost(initial_residual, loss=loss, f_scale=f_scale)

    result = _run_solver(x0)
    best_result = result
    best_cost = _robust_cost(np.asarray(result.fun, dtype=np.float64), loss=loss, f_scale=f_scale)
    restart_history: List[Dict[str, object]] = [
        {
            "restart": 0,
            "start_x": x0.tolist(),
            "end_x": np.asarray(result.x, dtype=np.float64).tolist(),
            "cost": float(best_cost),
            "success": bool(result.success),
            "message": str(result.message),
        }
    ]

    restart_rng = np.random.default_rng(20260213)
    for restart_idx in range(max_restarts):
        anchor = np.asarray(best_result.x, dtype=np.float64)
        span = np.array(
            [
                0.1 * max(abs(float(anchor[0])), 0.03),
                0.1 * max(abs(float(anchor[1])), 0.03),
                0.05,
            ],
            dtype=np.float64,
        )
        trial_start = anchor + restart_rng.uniform(-1.0, 1.0, size=3) * span
        trial_start = np.clip(trial_start, lower, upper)
        trial = _run_solver(trial_start)
        trial_cost = _robust_cost(
            np.asarray(trial.fun, dtype=np.float64), loss=loss, f_scale=f_scale
        )
        restart_history.append(
            {
                "restart": restart_idx + 1,
                "start_x": np.asarray(trial_start, dtype=np.float64).tolist(),
                "end_x": np.asarray(trial.x, dtype=np.float64).tolist(),
                "cost": float(trial_cost),
                "success": bool(trial.success),
                "message": str(trial.message),
            }
        )
        if trial_cost < best_cost:
            best_result = trial
            best_cost = trial_cost

    result = best_result

    final_residual, roi_diagnostics = _evaluate_residual(
        np.asarray(result.x, dtype=np.float64),
        collect_diagnostics=True,
    )
    result.fun = np.asarray(final_residual, dtype=np.float64)
    final_cost = _robust_cost(result.fun, loss=loss, f_scale=f_scale)

    outlier_fraction = 0.0
    if roi_diagnostics:
        rms_vals = np.asarray([float(d["rms"]) for d in roi_diagnostics], dtype=np.float64)
        med = float(np.median(rms_vals))
        mad = float(np.median(np.abs(rms_vals - med)))
        sigma_rms = 1.4826 * mad
        if not np.isfinite(sigma_rms) or sigma_rms <= 1e-12:
            sigma_rms = float(np.std(rms_vals))
        if not np.isfinite(sigma_rms) or sigma_rms <= 1e-12:
            sigma_rms = 0.0
        threshold = med + 3.0 * sigma_rms if sigma_rms > 0 else med + 1e-12
        outliers = 0
        for diag in roi_diagnostics:
            is_outlier = bool(float(diag["rms"]) > threshold)
            diag["outlier"] = is_outlier
            if is_outlier:
                outliers += 1
        outlier_fraction = outliers / max(len(roi_diagnostics), 1)

    top_worst_rois = sorted(
        roi_diagnostics or [],
        key=lambda d: float(d["rms"]),
        reverse=True,
    )[:5]

    bound_names = ("sigma_mosaic_deg", "gamma_mosaic_deg", "eta")
    bound_hits: List[str] = []
    for idx, name in enumerate(bound_names):
        if math.isclose(float(result.x[idx]), float(lower[idx]), rel_tol=0.0, abs_tol=1e-6):
            bound_hits.append(f"{name}=lower({lower[idx]:.3f})")
        elif math.isclose(float(result.x[idx]), float(upper[idx]), rel_tol=0.0, abs_tol=1e-6):
            bound_hits.append(f"{name}=upper({upper[idx]:.3f})")
    boundary_warning = (
        "Possible identifiability issue: optimizer finished at parameter bounds ("
        + ", ".join(bound_hits)
        + ")."
        if bound_hits
        else ""
    )

    cost_reduction = 0.0
    if initial_cost > 1e-12 and np.isfinite(initial_cost):
        cost_reduction = (initial_cost - final_cost) / initial_cost

    best_params = dict(params)
    best_mosaic = dict(best_params.get("mosaic_params", {}))
    best_mosaic["sigma_mosaic_deg"] = float(result.x[0])
    best_mosaic["gamma_mosaic_deg"] = float(result.x[1])
    best_mosaic["eta"] = float(result.x[2])
    best_params["mosaic_params"] = best_mosaic

    result.cost = float(final_cost)
    result.best_params = best_params
    result.selected_rois = rois
    result.reflection_indices = unique_indices
    result.roi_diagnostics = roi_diagnostics or []
    result.rejected_rois = rejected_rois
    result.initial_cost = float(initial_cost)
    result.final_cost = float(final_cost)
    result.cost_reduction = float(cost_reduction)
    result.outlier_fraction = float(outlier_fraction)
    result.top_worst_rois = top_worst_rois
    result.restart_history = restart_history
    result.boundary_warning = boundary_warning
    result.acceptance_passed = bool(
        cost_reduction >= 0.20 and outlier_fraction <= 0.25 and not bound_hits
    )
    result.peak_source = peak_source_key
    result.roi_normalization = roi_norm_key
    return result


def _prepare_mosaic_shape_patch(
    patch: np.ndarray,
    *,
    smooth_sigma_px: float,
    ridge_percentile: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the centered/smoothed patch plus ridge mask and distance transform."""

    patch_arr = np.asarray(patch, dtype=np.float64)
    finite = patch_arr[np.isfinite(patch_arr)]
    if finite.size:
        centered = patch_arr - float(np.median(finite))
    else:
        centered = np.zeros_like(patch_arr, dtype=np.float64)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)

    smooth_sigma = max(float(smooth_sigma_px), 0.0)
    if smooth_sigma > 0.0:
        smoothed = gaussian_filter(centered, sigma=smooth_sigma, mode="reflect")
    else:
        smoothed = centered

    ridge_mask = np.asarray(
        _compute_ridge_map(smoothed, percentile=float(ridge_percentile)),
        dtype=bool,
    )
    return smoothed, ridge_mask, distance_transform_edt(~ridge_mask)


def _estimate_mosaic_shape_roi_half_width(
    params: Dict[str, object],
    upper_bounds: Sequence[float],
    image_size: int,
) -> int:
    """Estimate one fixed ROI radius from detector geometry and width bounds."""

    detector_distance = max(float(params.get("corto_detector", 0.0)), 1.0e-6)
    pixel_size = _estimate_pixel_size(params)
    width_upper = max(float(upper_bounds[0]), float(upper_bounds[1]), 0.03)
    projected_half_width_px = (
        detector_distance * math.radians(width_upper) / max(pixel_size, 1.0e-9)
    )
    half_width = int(
        np.clip(
            math.ceil(max(8.0, 2.0 * projected_half_width_px)),
            8,
            max(8, min(64, int(image_size) // 4)),
        )
    )
    return max(int(half_width), 1)


def _build_mosaic_shape_dataset_contexts(
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, object],
    dataset_specs: Sequence[Dict[str, object]],
    *,
    roi_half_width: int,
    smooth_sigma_px: float,
    ridge_percentile: float,
) -> Tuple[List[MosaicShapeDatasetContext], List[Dict[str, object]]]:
    """Prepare multi-dataset ROI contexts for the geometry-cached shape fit."""

    contexts = _build_geometry_fit_dataset_contexts(
        miller,
        intensities,
        params,
        measured_peaks=None,
        experimental_image=None,
        dataset_specs=dataset_specs,
    )
    a_lattice = float(params.get("a", 1.0))
    c_lattice = float(params.get("c", 1.0))
    mosaic_params = dict(params.get("mosaic_params", {}))
    wavelength_array = mosaic_params.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic_params.get("wavelength_i_array")
    if wavelength_array is None:
        lambda_scalar = float(params.get("lambda", 1.0))
    else:
        wave_arr = np.asarray(wavelength_array, dtype=np.float64).ravel()
        wave_arr = wave_arr[np.isfinite(wave_arr) & (wave_arr > 0.0)]
        lambda_scalar = (
            float(params.get("lambda", float(np.mean(wave_arr))))
            if wave_arr.size
            else float(params.get("lambda", 1.0))
        )
    lambda_scalar = max(float(lambda_scalar), 1.0e-6)

    prepared: List[MosaicShapeDatasetContext] = []
    rejected_rois: List[Dict[str, object]] = []

    for dataset_ctx in contexts:
        experimental_image = dataset_ctx.experimental_image
        if experimental_image is None:
            raise RuntimeError(
                f"Mosaic shape fit requires experimental_image for dataset '{dataset_ctx.label}'."
            )
        experimental_image = np.asarray(experimental_image, dtype=np.float64)
        if experimental_image.shape != (image_size, image_size):
            raise RuntimeError(
                f"Mosaic shape fit dataset '{dataset_ctx.label}' has image shape "
                f"{experimental_image.shape}, expected {(image_size, image_size)}."
            )

        subset = dataset_ctx.subset
        subset_miller = np.asarray(subset.miller, dtype=np.float64)
        subset_intensities = np.asarray(subset.intensities, dtype=np.float64)
        allowed_mask = _allowed_reflection_mask(subset_miller)
        keep_rows: List[int] = []
        reflection_lookup: Dict[Tuple[int, int, int], int] = {}
        two_theta_lookup: Dict[Tuple[int, int, int], float] = {}
        for local_idx, hkl_row in enumerate(subset_miller):
            if not bool(allowed_mask[local_idx]):
                continue
            hkl = tuple(int(round(val)) for val in hkl_row)
            if hkl == (0, 0, 0):
                continue
            try:
                spacing = d_spacing(hkl[0], hkl[1], hkl[2], a_lattice, c_lattice)
                two_theta_value = two_theta(spacing, lambda_scalar)
            except Exception:
                continue
            if two_theta_value is None or not np.isfinite(two_theta_value):
                continue
            if (hkl[0] != 0 or hkl[1] != 0) and float(two_theta_value) > 65.0:
                continue
            keep_rows.append(int(local_idx))
            reflection_lookup.setdefault(hkl, len(keep_rows) - 1)
            two_theta_lookup[hkl] = float(two_theta_value)

        if not keep_rows:
            raise RuntimeError(
                f"Mosaic shape fit found no usable reflections for dataset '{dataset_ctx.label}'."
            )

        usable_hkls = set(two_theta_lookup)
        normalized_measured = _normalize_measured_peaks(subset.measured_entries)
        usable_measured_entries: List[Dict[str, object]] = []
        rois: List[MosaicShapeROI] = []
        seen_centers: Set[Tuple[Tuple[int, int, int], int, int]] = set()

        for entry in normalized_measured:
            hkl = entry["hkl"]  # type: ignore[assignment]
            if hkl not in usable_hkls:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "stage": "candidate",
                        "reason": "hkl_not_usable_for_mosaic_shape",
                        "hkl": tuple(int(v) for v in hkl),
                        "x": float(entry["x"]),
                        "y": float(entry["y"]),
                    }
                )
                continue
            usable_measured_entries.append(dict(entry))

            center_row = float(entry["y"])
            center_col = float(entry["x"])
            row_idx = int(round(center_row))
            col_idx = int(round(center_col))
            roi_key = (tuple(int(v) for v in hkl), row_idx, col_idx)
            if roi_key in seen_centers:
                continue
            seen_centers.add(roi_key)

            row0 = row_idx - int(roi_half_width)
            row1 = row_idx + int(roi_half_width) + 1
            col0 = col_idx - int(roi_half_width)
            col1 = col_idx + int(roi_half_width) + 1
            if row0 < 0 or col0 < 0 or row1 > image_size or col1 > image_size:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "stage": "roi",
                        "reason": "out_of_bounds",
                        "hkl": tuple(int(v) for v in hkl),
                        "center": (float(center_col), float(center_row)),
                    }
                )
                continue

            patch = experimental_image[row0:row1, col0:col1]
            _, measured_mask, measured_distance = _prepare_mosaic_shape_patch(
                patch,
                smooth_sigma_px=float(smooth_sigma_px),
                ridge_percentile=float(ridge_percentile),
            )
            measured_active_pixels = int(np.count_nonzero(measured_mask))
            if measured_active_pixels <= 0:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "stage": "roi",
                        "reason": "empty_measured_ridge",
                        "hkl": tuple(int(v) for v in hkl),
                        "center": (float(center_col), float(center_row)),
                    }
                )
                continue

            rois.append(
                MosaicShapeROI(
                    dataset_index=int(dataset_ctx.dataset_index),
                    dataset_label=str(dataset_ctx.label),
                    reflection_index=int(reflection_lookup[hkl]),
                    hkl=tuple(int(v) for v in hkl),
                    center_row=float(center_row),
                    center_col=float(center_col),
                    row_bounds=(int(row0), int(row1)),
                    col_bounds=(int(col0), int(col1)),
                    measured_mask=measured_mask,
                    measured_distance=np.asarray(measured_distance, dtype=np.float64),
                    measured_active_pixels=int(measured_active_pixels),
                    measured_two_theta=float(two_theta_lookup[hkl]),
                )
            )

        point_match_subset = _prepare_reflection_subset(
            np.ascontiguousarray(subset_miller[keep_rows], dtype=np.float64),
            np.ascontiguousarray(subset_intensities[keep_rows], dtype=np.float64),
            usable_measured_entries,
        )
        geometry_context = GeometryFitDatasetContext(
            dataset_index=int(dataset_ctx.dataset_index),
            label=str(dataset_ctx.label),
            theta_initial=float(dataset_ctx.theta_initial),
            subset=point_match_subset,
            experimental_image=experimental_image,
        )

        prepared.append(
            MosaicShapeDatasetContext(
                dataset_index=int(dataset_ctx.dataset_index),
                label=str(dataset_ctx.label),
                theta_initial=float(dataset_ctx.theta_initial),
                experimental_image=experimental_image,
                geometry_context=geometry_context,
                miller=np.ascontiguousarray(subset_miller[keep_rows], dtype=np.float64),
                intensities=np.ascontiguousarray(
                    subset_intensities[keep_rows],
                    dtype=np.float64,
                ),
                rois=rois,
                measured_peak_count=int(len(normalized_measured)),
            )
        )

    return prepared, rejected_rois


def _fit_mosaic_shape_parameters_legacy(
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, object],
    *,
    dataset_specs: Sequence[Dict[str, object]],
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
    max_nfev: int = 80,
    max_restarts: int = 2,
    smooth_sigma_px: float = 1.0,
    ridge_percentile: float = 85.0,
    roi_half_width: Optional[int] = None,
    min_total_rois: int = 8,
    min_per_dataset_rois: int = 3,
    equal_dataset_weights: bool = True,
    workers: object = "auto",
    parallel_mode: str = "auto",
    worker_numba_threads: object = 0,
    restart_jitter: float = 0.15,
    ridge_weight: float = 1.0,
    point_match_weight: float = 0.0,
    point_match_f_scale: float = 6.0,
    point_match_weighted_matching: bool = False,
    point_match_missing_pair_penalty: float = 20.0,
    refine_theta: bool = False,
    theta_mode: str = "auto",
    theta_bounds: Optional[Tuple[float, float]] = None,
    fit_sigma_mosaic: bool = True,
    fit_gamma_mosaic: bool = True,
    fit_eta: bool = True,
) -> OptimizeResult:
    """Fit mosaic shape parameters from prepared experimental datasets."""

    dataset_spec_entries = _coerce_sequence_items(dataset_specs)
    miller = np.asarray(miller, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    if miller.ndim != 2 or miller.shape[1] != 3:
        raise ValueError("miller must be an array of shape (N, 3)")
    if intensities.ndim != 1 or intensities.shape[0] != miller.shape[0]:
        raise ValueError("intensities and miller must have matching lengths")
    if int(image_size) <= 0:
        raise ValueError("image_size must be positive")
    if not dataset_spec_entries:
        raise RuntimeError("Mosaic shape fit requires at least one prepared dataset_spec.")

    mosaic_params = dict(params.get("mosaic_params", {}))
    if not mosaic_params:
        raise ValueError("params['mosaic_params'] is required")
    required_keys = ("beam_x_array", "beam_y_array", "theta_array", "phi_array")
    missing_keys = [key for key in required_keys if not np.asarray(mosaic_params.get(key)).size]
    if missing_keys:
        raise ValueError("mosaic_params must include beam and divergence samples for shape fitting")

    wavelength_array = mosaic_params.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic_params.get("wavelength_i_array")
    if wavelength_array is None:
        base_lambda = float(params.get("lambda", 1.0))
        wavelength_array = np.full(
            np.asarray(mosaic_params["beam_x_array"]).shape,
            base_lambda,
            dtype=np.float64,
        )
    else:
        wavelength_array = np.asarray(wavelength_array, dtype=np.float64)

    sigma0 = float(mosaic_params.get("sigma_mosaic_deg", 0.5))
    gamma0 = float(mosaic_params.get("gamma_mosaic_deg", 0.5))
    eta0 = float(mosaic_params.get("eta", 0.05))
    if bounds is None:
        bounds = (
            np.array([0.03, 0.03, 0.0], dtype=np.float64),
            np.array([3.0, 3.0, 1.0], dtype=np.float64),
        )
    lower = np.asarray(bounds[0], dtype=np.float64).reshape(-1)
    upper = np.asarray(bounds[1], dtype=np.float64).reshape(-1)
    if lower.size != 3 or upper.size != 3:
        raise ValueError("bounds must contain exactly 3 lower/upper values")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("bounds must be finite")
    if np.any(lower >= upper):
        raise ValueError("each lower bound must be strictly less than upper bound")

    x0 = np.clip(np.array([sigma0, gamma0, eta0], dtype=np.float64), lower, upper)
    min_total_rois = max(int(min_total_rois), 1)
    min_per_dataset_rois = max(int(min_per_dataset_rois), 1)
    max_restarts = max(int(max_restarts), 0)
    smooth_sigma_px = max(float(smooth_sigma_px), 0.0)
    ridge_percentile = float(np.clip(float(ridge_percentile), 1.0, 99.9))
    restart_jitter = max(float(restart_jitter), 0.0)
    ridge_weight = max(float(ridge_weight), 0.0)
    point_match_weight = max(float(point_match_weight), 0.0)
    point_match_f_scale = max(float(point_match_f_scale), 1.0e-6)
    fit_sigma_mosaic = bool(fit_sigma_mosaic)
    fit_gamma_mosaic = bool(fit_gamma_mosaic)
    fit_eta = bool(fit_eta)
    point_match_missing_pair_penalty = max(
        float(point_match_missing_pair_penalty),
        0.0,
    )
    if ridge_weight <= 0.0 and point_match_weight <= 0.0:
        raise ValueError(
            "At least one mosaic objective term must be enabled "
            "(ridge_weight > 0 or point_match_weight > 0)."
        )

    if roi_half_width is None:
        roi_half_width = _estimate_mosaic_shape_roi_half_width(
            params,
            upper,
            int(image_size),
        )
    roi_half_width = int(roi_half_width)
    if roi_half_width <= 0:
        raise ValueError("roi_half_width must be a positive integer")

    prepared_datasets, rejected_rois = _build_mosaic_shape_dataset_contexts(
        miller,
        intensities,
        int(image_size),
        dict(params),
        list(dataset_spec_entries),
        roi_half_width=int(roi_half_width),
        smooth_sigma_px=float(smooth_sigma_px),
        ridge_percentile=float(ridge_percentile),
    )
    if not prepared_datasets:
        raise RuntimeError("Mosaic shape fit found no prepared datasets.")

    rejected_reason_counts_by_dataset: Dict[str, Dict[str, int]] = {}
    for rejected in rejected_rois:
        dataset_label = str(rejected.get("dataset_label") or "")
        reason = str(rejected.get("reason") or "unknown")
        if not dataset_label:
            continue
        bucket = rejected_reason_counts_by_dataset.setdefault(dataset_label, {})
        bucket[reason] = int(bucket.get(reason, 0)) + 1

    dataset_failures = [
        dataset_ctx
        for dataset_ctx in prepared_datasets
        if len(dataset_ctx.rois) < int(min_per_dataset_rois)
    ]
    if dataset_failures:
        dataset_text = ", ".join(f"{ctx.label}={len(ctx.rois)}" for ctx in dataset_failures)
        rejection_text = ", ".join(
            f"{ctx.label}["
            + ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(
                    rejected_reason_counts_by_dataset.get(str(ctx.label), {}).items(),
                    key=lambda item: (-int(item[1]), str(item[0])),
                )[:3]
            )
            + "]"
            for ctx in dataset_failures
            if rejected_reason_counts_by_dataset.get(str(ctx.label))
        )
        detail_suffix = f" Rejections: {rejection_text}." if rejection_text else ""
        raise RuntimeError(
            "Mosaic shape fit needs at least "
            f"{int(min_per_dataset_rois)} usable ROIs per dataset; got {dataset_text}."
            f"{detail_suffix}"
        )

    total_rois = int(sum(len(dataset_ctx.rois) for dataset_ctx in prepared_datasets))
    if total_rois < int(min_total_rois):
        raise RuntimeError(
            f"Mosaic shape fit needs at least {int(min_total_rois)} usable ROIs; got {total_rois}."
        )

    def _json_safe(value: object) -> object:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, complex):
            return {
                "real": _json_safe(value.real),
                "imag": _json_safe(value.imag),
            }
        if isinstance(value, np.generic):
            return _json_safe(value.item())
        if isinstance(value, np.ndarray):
            return [_json_safe(item) for item in value.tolist()]
        if isinstance(value, Mapping):
            return {str(key): _json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)

    def _array_summary(values: object) -> Dict[str, object]:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = arr[np.isfinite(arr)]
        summary: Dict[str, object] = {
            "count": int(arr.size),
            "finite_count": int(finite.size),
        }
        if finite.size:
            summary.update(
                {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                }
            )
        else:
            summary.update(
                {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                }
            )
        return summary

    def _image_summary(image: object) -> Dict[str, object]:
        arr = np.asarray(image, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        summary: Dict[str, object] = {
            "shape": [int(dim) for dim in arr.shape],
            "finite_count": int(finite.size),
        }
        if finite.size:
            summary.update(
                {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "sum": float(np.sum(finite)),
                }
            )
        else:
            summary.update(
                {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "sum": None,
                }
            )
        return summary

    def _measured_peak_summaries(entries: object) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for entry in entries if isinstance(entries, (list, tuple)) else []:
            if not isinstance(entry, Mapping):
                continue
            peak_summary: Dict[str, object] = {
                "hkl": _json_safe(entry.get("hkl")),
                "x": _json_safe(entry.get("x")),
                "y": _json_safe(entry.get("y")),
            }
            if "source_table_index" in entry:
                peak_summary["source_table_index"] = _json_safe(entry.get("source_table_index"))
            if "source_row_index" in entry:
                peak_summary["source_row_index"] = _json_safe(entry.get("source_row_index"))
            out.append(peak_summary)
        return out

    input_dataset_summaries: List[Dict[str, object]] = []
    for spec in dataset_spec_entries:
        if not isinstance(spec, Mapping):
            continue
        measured_peaks = spec.get("measured_peaks", [])
        experimental_image = spec.get("experimental_image")
        input_dataset_summaries.append(
            {
                "dataset_index": _json_safe(spec.get("dataset_index")),
                "dataset_label": _json_safe(spec.get("label", spec.get("dataset_index", "?"))),
                "theta_initial_deg": _json_safe(spec.get("theta_initial")),
                "measured_peak_count": int(
                    len(measured_peaks) if isinstance(measured_peaks, (list, tuple)) else 0
                ),
                "experimental_image": (
                    _image_summary(experimental_image) if experimental_image is not None else None
                ),
                "measured_peaks": _measured_peak_summaries(measured_peaks),
            }
        )

    prepared_dataset_summaries: List[Dict[str, object]] = []
    for dataset_ctx in prepared_datasets:
        roi_entries = [
            {
                "reflection_index": int(roi.reflection_index),
                "hkl": [int(v) for v in roi.hkl],
                "center_xy_px": [float(roi.center_col), float(roi.center_row)],
                "row_bounds": [int(v) for v in roi.row_bounds],
                "col_bounds": [int(v) for v in roi.col_bounds],
                "measured_two_theta_deg": float(roi.measured_two_theta),
                "measured_active_pixels": int(roi.measured_active_pixels),
            }
            for roi in dataset_ctx.rois
        ]
        prepared_dataset_summaries.append(
            {
                "dataset_index": int(dataset_ctx.dataset_index),
                "dataset_label": str(dataset_ctx.label),
                "theta_initial_deg": float(dataset_ctx.theta_initial),
                "experimental_image": _image_summary(dataset_ctx.experimental_image),
                "measured_peak_count": int(dataset_ctx.measured_peak_count),
                "reflection_count": int(dataset_ctx.miller.shape[0]),
                "intensity_count": int(dataset_ctx.intensities.shape[0]),
                "roi_count": int(len(dataset_ctx.rois)),
                "roi_hkls": [[int(v) for v in roi.hkl] for roi in dataset_ctx.rois],
                "rois": roi_entries,
            }
        )

    rejected_roi_reason_counts: Dict[str, int] = {}
    rejected_roi_counts_by_dataset: Dict[str, int] = {}
    for rejected in rejected_rois:
        if not isinstance(rejected, Mapping):
            continue
        dataset_key = str(rejected.get("dataset_label", rejected.get("dataset_index", "?")))
        reason_key = str(rejected.get("reason", "unknown"))
        stage_key = str(rejected.get("stage", "unknown"))
        rejected_roi_reason_counts[f"{dataset_key}:{stage_key}:{reason_key}"] = (
            rejected_roi_reason_counts.get(f"{dataset_key}:{stage_key}:{reason_key}", 0) + 1
        )
        rejected_roi_counts_by_dataset[dataset_key] = (
            rejected_roi_counts_by_dataset.get(dataset_key, 0) + 1
        )

    dataset_count = int(len(prepared_datasets))
    optimize_theta = bool(refine_theta)

    theta_mode_key = str(theta_mode).strip().lower()
    if theta_mode_key not in {"auto", "single", "shared_offset", "per_dataset"}:
        raise ValueError(
            "theta_mode must be one of {'auto', 'single', 'shared_offset', 'per_dataset'}"
        )
    if theta_mode_key == "auto":
        resolved_theta_mode = "single" if dataset_count == 1 else "per_dataset"
    else:
        resolved_theta_mode = str(theta_mode_key)
    if resolved_theta_mode == "single" and dataset_count != 1:
        raise ValueError("theta_mode='single' requires exactly one prepared dataset")

    parallel_mode_key = str(parallel_mode).strip().lower()
    if parallel_mode_key not in {"auto", "datasets", "restarts", "off"}:
        raise ValueError("parallel_mode must be one of {'auto', 'datasets', 'restarts', 'off'}")
    configured_parallel_workers = _resolve_parallel_worker_count(
        workers,
        max_tasks=max(len(prepared_datasets), max_restarts, 1),
    )
    dataset_parallel_workers = 1
    restart_parallel_workers = 1
    if configured_parallel_workers > 1 and parallel_mode_key != "off":
        if parallel_mode_key in {"auto", "datasets"} and len(prepared_datasets) > 1:
            dataset_parallel_workers = min(
                int(configured_parallel_workers),
                len(prepared_datasets),
            )
        elif parallel_mode_key in {"auto", "restarts"} and max_restarts > 1:
            restart_parallel_workers = min(
                int(configured_parallel_workers),
                int(max_restarts),
            )
    active_outer_workers = max(dataset_parallel_workers, restart_parallel_workers)
    numba_threads = _resolve_numba_threads_per_worker(
        active_outer_workers,
        worker_numba_threads,
    )
    parallelization_summary = {
        "mode": str(parallel_mode_key),
        "configured_workers": int(configured_parallel_workers),
        "dataset_workers": int(dataset_parallel_workers),
        "restart_workers": int(restart_parallel_workers),
        "worker_numba_threads": (None if numba_threads is None else int(numba_threads)),
        "numba_thread_budget": int(_available_parallel_thread_budget()),
    }

    base_params = dict(params)
    beam_x = np.asarray(mosaic_params.get("beam_x_array"), dtype=np.float64)
    beam_y = np.asarray(mosaic_params.get("beam_y_array"), dtype=np.float64)
    theta_array = np.asarray(mosaic_params.get("theta_array"), dtype=np.float64)
    phi_array = np.asarray(mosaic_params.get("phi_array"), dtype=np.float64)
    uv1 = np.asarray(
        base_params.get("uv1", np.array([1.0, 0.0, 0.0])),
        dtype=np.float64,
    )
    uv2 = np.asarray(
        base_params.get("uv2", np.array([0.0, 1.0, 0.0])),
        dtype=np.float64,
    )

    def _safe_float(value: object, fallback: float) -> float:
        try:
            out = float(value)
        except Exception:
            return float(fallback)
        if not np.isfinite(out):
            return float(fallback)
        return out

    theta_param_names: List[str] = []
    theta_param_dataset_indices: List[Optional[int]] = []
    if optimize_theta:
        theta_bound_lower: float
        theta_bound_upper: float
        if theta_bounds is None:
            theta_bound_lower = -0.5
            theta_bound_upper = 0.5
        else:
            theta_bound_lower = float(theta_bounds[0])
            theta_bound_upper = float(theta_bounds[1])
        if not (np.isfinite(theta_bound_lower) and np.isfinite(theta_bound_upper)):
            raise ValueError("theta_bounds must be finite when refine_theta is enabled")
        if theta_bound_lower >= theta_bound_upper:
            raise ValueError("theta_bounds lower bound must be strictly less than the upper bound")

        theta_seeds: List[float] = []
        theta_lowers: List[float] = []
        theta_uppers: List[float] = []
        if resolved_theta_mode == "shared_offset":
            theta_seed = _safe_float(params.get("theta_offset", 0.0), 0.0)
            theta_param_names.append("theta_offset")
            theta_param_dataset_indices.append(None)
            theta_seeds.append(float(np.clip(theta_seed, theta_bound_lower, theta_bound_upper)))
            theta_lowers.append(float(theta_bound_lower))
            theta_uppers.append(float(theta_bound_upper))
        elif resolved_theta_mode == "single":
            theta_seed = _safe_float(
                params.get("theta_initial", prepared_datasets[0].theta_initial),
                float(prepared_datasets[0].theta_initial),
            )
            theta_param_names.append("theta_initial")
            theta_param_dataset_indices.append(int(prepared_datasets[0].dataset_index))
            theta_seeds.append(float(np.clip(theta_seed, theta_bound_lower, theta_bound_upper)))
            theta_lowers.append(float(theta_bound_lower))
            theta_uppers.append(float(theta_bound_upper))
        else:
            theta_seed_map = dict(base_params.get("_mosaic_theta_initials_by_dataset", {}))
            for dataset_ctx in prepared_datasets:
                dataset_idx = int(dataset_ctx.dataset_index)
                dataset_theta = float(dataset_ctx.theta_initial)
                theta_seed = _safe_float(
                    theta_seed_map.get(dataset_idx, dataset_theta),
                    dataset_theta,
                )
                lower_i = float(dataset_theta + theta_bound_lower)
                upper_i = float(dataset_theta + theta_bound_upper)
                theta_param_names.append(f"theta_initial[{dataset_idx}]")
                theta_param_dataset_indices.append(int(dataset_idx))
                theta_seeds.append(float(np.clip(theta_seed, lower_i, upper_i)))
                theta_lowers.append(float(lower_i))
                theta_uppers.append(float(upper_i))

        x0 = np.concatenate([x0, np.asarray(theta_seeds, dtype=np.float64)])
        lower = np.concatenate([lower, np.asarray(theta_lowers, dtype=np.float64)])
        upper = np.concatenate([upper, np.asarray(theta_uppers, dtype=np.float64)])

    param_names = ["sigma_mosaic_deg", "gamma_mosaic_deg", "eta"]
    if optimize_theta and theta_param_names:
        param_names.extend(str(name) for name in theta_param_names)
    active_mask = np.array(
        [
            fit_sigma_mosaic,
            fit_gamma_mosaic,
            fit_eta,
            *([True] * len(theta_param_names) if optimize_theta else []),
        ],
        dtype=bool,
    )
    if active_mask.size != len(param_names):
        raise RuntimeError("mosaic fit parameter bookkeeping is inconsistent")
    active_indices = np.flatnonzero(active_mask)
    fixed_indices = np.flatnonzero(~active_mask)
    if active_indices.size == 0:
        raise ValueError(
            "At least one mosaic fit parameter must be enabled "
            "(fit_sigma_mosaic, fit_gamma_mosaic, fit_eta, or refine_theta)."
        )

    full_x0 = np.asarray(x0, dtype=np.float64)
    full_lower = np.asarray(lower, dtype=np.float64)
    full_upper = np.asarray(upper, dtype=np.float64)
    active_x0 = np.asarray(full_x0[active_indices], dtype=np.float64)
    active_lower = np.asarray(full_lower[active_indices], dtype=np.float64)
    active_upper = np.asarray(full_upper[active_indices], dtype=np.float64)
    active_parameter_names = [str(param_names[idx]) for idx in active_indices.tolist()]
    fixed_parameter_names = [str(param_names[idx]) for idx in fixed_indices.tolist()]

    def _expand_active_vector(x_trial: Sequence[float]) -> np.ndarray:
        x_arr = np.asarray(x_trial, dtype=np.float64).reshape(-1)
        if active_indices.size == full_x0.size:
            if x_arr.size != full_x0.size:
                raise ValueError("mosaic fit parameter vector length mismatch")
            return x_arr
        if x_arr.size != active_indices.size:
            raise ValueError("mosaic fit active parameter vector length mismatch")
        full = np.array(full_x0, copy=True)
        full[active_indices] = x_arr
        return full

    def _apply_trial_params(x: Sequence[float]) -> Dict[str, object]:
        local = dict(base_params)
        local_mosaic = dict(mosaic_params)
        local_mosaic["sigma_mosaic_deg"] = float(x[0])
        local_mosaic["gamma_mosaic_deg"] = float(x[1])
        local_mosaic["eta"] = float(x[2])
        local_mosaic["beam_x_array"] = beam_x
        local_mosaic["beam_y_array"] = beam_y
        local_mosaic["theta_array"] = theta_array
        local_mosaic["phi_array"] = phi_array
        local_mosaic["wavelength_array"] = wavelength_array
        local["mosaic_params"] = local_mosaic
        local.setdefault("theta_offset", 0.0)
        if optimize_theta:
            theta_slice = np.asarray(x[3:], dtype=np.float64)
            if resolved_theta_mode == "shared_offset" and theta_slice.size >= 1:
                local["theta_offset"] = float(theta_slice[0])
            elif resolved_theta_mode == "single" and theta_slice.size >= 1:
                local["theta_initial"] = float(theta_slice[0])
            elif resolved_theta_mode == "per_dataset" and theta_slice.size:
                local["_mosaic_theta_initials_by_dataset"] = {
                    int(dataset_idx): float(theta_slice[idx])
                    for idx, dataset_idx in enumerate(theta_param_dataset_indices)
                    if dataset_idx is not None and idx < theta_slice.size
                }
        return local

    def _theta_initial_for_dataset(
        local: Dict[str, object],
        dataset_ctx: MosaicShapeDatasetContext,
    ) -> float:
        theta_base = _safe_float(dataset_ctx.theta_initial, 0.0)
        if resolved_theta_mode == "shared_offset":
            return float(theta_base + _safe_float(local.get("theta_offset", 0.0), 0.0))
        if resolved_theta_mode == "single":
            return _safe_float(local.get("theta_initial", theta_base), theta_base)
        theta_map = local.get("_mosaic_theta_initials_by_dataset", {})
        if isinstance(theta_map, Mapping):
            return _safe_float(
                theta_map.get(int(dataset_ctx.dataset_index), theta_base), theta_base
            )
        return float(theta_base)

    def _simulate_dataset_image(
        local: Dict[str, object],
        dataset_ctx: MosaicShapeDatasetContext,
        *,
        theta_value: float,
    ) -> np.ndarray:
        local_mosaic = dict(local["mosaic_params"])
        wave_local = local_mosaic.get("wavelength_array")
        if wave_local is None:
            wave_local = local_mosaic.get("wavelength_i_array")
        if wave_local is None:
            wave_local = wavelength_array
        buffer = np.zeros((image_size, image_size), dtype=np.float64)
        image, *_ = _process_peaks_parallel_safe(
            dataset_ctx.miller,
            dataset_ctx.intensities,
            image_size,
            local["a"],
            local["c"],
            wave_local,
            buffer,
            local["corto_detector"],
            local["gamma"],
            local["Gamma"],
            local["chi"],
            local.get("psi", 0.0),
            local.get("psi_z", 0.0),
            local["zs"],
            local["zb"],
            local["n2"],
            local_mosaic["beam_x_array"],
            local_mosaic["beam_y_array"],
            local_mosaic["theta_array"],
            local_mosaic["phi_array"],
            local_mosaic["sigma_mosaic_deg"],
            local_mosaic["gamma_mosaic_deg"],
            local_mosaic["eta"],
            wave_local,
            local["debye_x"],
            local["debye_y"],
            local["center"],
            theta_value,
            local.get("cor_angle", 0.0),
            uv1,
            uv2,
            save_flag=0,
            **_simulation_kernel_kwargs(local, local_mosaic),
        )
        return np.asarray(image, dtype=np.float64)

    ridge_term_enabled = bool(ridge_weight > 0.0)
    point_match_term_enabled = bool(point_match_weight > 0.0)

    def _evaluate_one_dataset(
        item: Tuple[Dict[str, object], MosaicShapeDatasetContext, bool],
    ) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
        local, dataset_ctx, collect_diagnostics = item
        theta_value = _theta_initial_for_dataset(local, dataset_ctx)
        roi_blocks: List[np.ndarray] = []
        roi_diags: List[Dict[str, object]] = []
        if ridge_term_enabled:
            sim_image = _simulate_dataset_image(
                local,
                dataset_ctx,
                theta_value=float(theta_value),
            )
            for roi in dataset_ctx.rois:
                row0, row1 = roi.row_bounds
                col0, col1 = roi.col_bounds
                patch = sim_image[row0:row1, col0:col1]
                _, sim_mask, sim_distance = _prepare_mosaic_shape_patch(
                    patch,
                    smooth_sigma_px=float(smooth_sigma_px),
                    ridge_percentile=float(ridge_percentile),
                )
                measured_term = roi.measured_mask.astype(np.float64) * sim_distance
                sim_term = sim_mask.astype(np.float64) * roi.measured_distance
                residual_raw = np.concatenate([measured_term.ravel(), sim_term.ravel()]).astype(
                    np.float64, copy=False
                )
                sim_active_pixels = int(np.count_nonzero(sim_mask))
                active_pixels = int(roi.measured_active_pixels + sim_active_pixels)
                residual_normed = (residual_raw / math.sqrt(max(active_pixels, 1))) * float(
                    ridge_weight
                )
                roi_blocks.append(np.asarray(residual_normed, dtype=np.float64))

                if collect_diagnostics:
                    roi_diags.append(
                        {
                            "dataset_index": int(dataset_ctx.dataset_index),
                            "dataset_label": str(dataset_ctx.label),
                            "hkl": tuple(int(v) for v in roi.hkl),
                            "center": (float(roi.center_col), float(roi.center_row)),
                            "measured_active_pixels": int(roi.measured_active_pixels),
                            "sim_active_pixels": int(sim_active_pixels),
                            "active_mask_pixels": int(active_pixels),
                            "two_theta_deg": float(roi.measured_two_theta),
                            "rms": (
                                float(np.sqrt(np.mean(residual_raw * residual_raw)))
                                if residual_raw.size
                                else 0.0
                            ),
                        }
                    )

        point_residual = np.zeros(0, dtype=np.float64)
        point_summary: Dict[str, object] = {}
        if point_match_term_enabled:
            point_residual_raw, _point_diags, point_summary = (
                _evaluate_geometry_fit_dataset_point_matches(
                    local,
                    dataset_ctx.geometry_context,
                    image_size=image_size,
                    pixel_tol=float("inf"),
                    weighted_matching=bool(point_match_weighted_matching),
                    solver_f_scale=float(point_match_f_scale),
                    missing_pair_penalty=float(point_match_missing_pair_penalty),
                    use_single_ray=False,
                    theta_value=float(theta_value),
                    collect_diagnostics=bool(collect_diagnostics),
                )
            )
            point_residual = np.asarray(point_residual_raw, dtype=np.float64)
            if point_residual.size:
                point_residual = (point_residual / math.sqrt(max(point_residual.size, 1))) * float(
                    point_match_weight
                )

        residual_blocks: List[np.ndarray] = []
        if roi_blocks:
            residual_blocks.append(np.concatenate(roi_blocks))
        if point_residual.size:
            residual_blocks.append(point_residual)

        dataset_residual = (
            np.concatenate(residual_blocks) if residual_blocks else np.zeros(0, dtype=np.float64)
        )
        dataset_weight = (
            1.0 / math.sqrt(max(len(dataset_ctx.rois), 1)) if bool(equal_dataset_weights) else 1.0
        )
        dataset_residual = np.asarray(dataset_residual, dtype=np.float64) * dataset_weight

        dataset_summary = {
            "dataset_index": int(dataset_ctx.dataset_index),
            "dataset_label": str(dataset_ctx.label),
            "theta_initial_deg": float(theta_value),
            "roi_count": int(len(dataset_ctx.rois)),
            "measured_peak_count": int(dataset_ctx.measured_peak_count),
            "simulated_reflection_count": int(dataset_ctx.miller.shape[0]),
            "dataset_weight": float(dataset_weight),
            "ridge_weight": float(ridge_weight),
            "point_match_weight": float(point_match_weight),
        }
        if point_summary:
            dataset_summary.update(
                {
                    "matched_pair_count": int(point_summary.get("matched_pair_count", 0)),
                    "missing_pair_count": int(point_summary.get("missing_pair_count", 0)),
                    "fixed_source_resolved_count": int(
                        point_summary.get("fixed_source_resolved_count", 0)
                    ),
                    "unweighted_peak_rms_px": float(
                        point_summary.get("unweighted_peak_rms_px", float("nan"))
                    ),
                    "unweighted_peak_max_px": float(
                        point_summary.get("unweighted_peak_max_px", float("nan"))
                    ),
                }
            )
        if collect_diagnostics:
            ordered_roi_diags = sorted(
                roi_diags,
                key=lambda item: float(item["rms"]),
                reverse=True,
            )
            dataset_summary.update(
                {
                    "residual_norm": float(np.linalg.norm(dataset_residual)),
                    "cost": float(_robust_cost(dataset_residual, loss=loss, f_scale=f_scale)),
                    "worst_hkls": [
                        tuple(int(v) for v in diag["hkl"]) for diag in ordered_roi_diags[:3]
                    ],
                    "max_roi_rms_px": (
                        float(ordered_roi_diags[0]["rms"]) if ordered_roi_diags else 0.0
                    ),
                }
            )
        return dataset_residual, roi_diags, dataset_summary

    def _evaluate_residual(
        theta: np.ndarray,
        *,
        collect_diagnostics: bool = False,
    ) -> Tuple[
        np.ndarray,
        Optional[List[Dict[str, object]]],
        Optional[List[Dict[str, object]]],
        Optional[Dict[int, int]],
    ]:
        local = _apply_trial_params(theta)
        dataset_items = [
            (local, dataset_ctx, bool(collect_diagnostics)) for dataset_ctx in prepared_datasets
        ]
        if dataset_parallel_workers > 1 and len(dataset_items) > 1:
            results = _threaded_map(
                _evaluate_one_dataset,
                dataset_items,
                max_workers=dataset_parallel_workers,
                numba_threads=numba_threads,
            )
        else:
            results = [_evaluate_one_dataset(item) for item in dataset_items]

        residual_blocks: List[np.ndarray] = []
        roi_diags: List[Dict[str, object]] = []
        dataset_diags: List[Dict[str, object]] = []
        roi_counts: Dict[int, int] = {}
        for residual_i, roi_diag_i, summary_i in results:
            residual_i = np.asarray(residual_i, dtype=np.float64)
            if residual_i.size:
                residual_blocks.append(residual_i)
            roi_diags.extend(list(roi_diag_i))
            dataset_diags.append(dict(summary_i))
            roi_counts[int(summary_i["dataset_index"])] = int(summary_i["roi_count"])
        if not residual_blocks:
            empty = np.zeros(1, dtype=np.float64)
            if collect_diagnostics:
                return empty, roi_diags, dataset_diags, roi_counts
            return empty, None, None, None
        merged = np.concatenate(residual_blocks)
        if collect_diagnostics:
            return merged, roi_diags, dataset_diags, roi_counts
        return merged, None, None, None

    def residual(theta: np.ndarray) -> np.ndarray:
        residual_out, _, _, _ = _evaluate_residual(
            _expand_active_vector(np.asarray(theta, dtype=np.float64)),
            collect_diagnostics=False,
        )
        return residual_out

    def _run_solver(x_start: np.ndarray) -> OptimizeResult:
        return least_squares(
            residual,
            np.asarray(x_start, dtype=np.float64),
            bounds=(active_lower, active_upper),
            loss=loss,
            f_scale=f_scale,
            max_nfev=int(max_nfev),
        )

    initial_residual = residual(active_x0)
    initial_cost = _robust_cost(initial_residual, loss=loss, f_scale=f_scale)
    primary_result = _run_solver(active_x0)
    primary_cost = _robust_cost(
        np.asarray(primary_result.fun, dtype=np.float64),
        loss=loss,
        f_scale=f_scale,
    )
    restart_history: List[Dict[str, object]] = [
        {
            "restart": 0,
            "start_x": np.asarray(_expand_active_vector(active_x0), dtype=np.float64).tolist(),
            "end_x": np.asarray(
                _expand_active_vector(primary_result.x),
                dtype=np.float64,
            ).tolist(),
            "cost": float(primary_cost),
            "success": bool(primary_result.success),
            "message": str(primary_result.message),
        }
    ]
    best_result = primary_result
    best_cost = float(primary_cost)

    if max_restarts > 0:
        restart_rng = np.random.default_rng(20260329)
        anchor = np.asarray(primary_result.x, dtype=np.float64)
        range_span = np.abs(active_upper - active_lower)
        base_span = np.maximum(np.abs(anchor), range_span)
        for idx, name in enumerate(active_parameter_names):
            if name in {"sigma_mosaic_deg", "gamma_mosaic_deg"}:
                base_span[idx] = max(float(base_span[idx]), 0.03)
            elif name == "eta":
                base_span[idx] = max(float(base_span[idx]), 0.5)
            elif name.startswith("theta_"):
                base_span[idx] = max(float(range_span[idx]), 0.05)
        span = np.maximum(
            restart_jitter * np.asarray(base_span, dtype=np.float64),
            np.full(anchor.shape, 1.0e-3, dtype=np.float64),
        )
        restart_starts = [
            np.clip(
                anchor + restart_rng.uniform(-1.0, 1.0, size=anchor.size).astype(np.float64) * span,
                active_lower,
                active_upper,
            )
            for _ in range(int(max_restarts))
        ]

        def _solve_restart(seed: np.ndarray) -> Tuple[np.ndarray, OptimizeResult, float]:
            solved = _run_solver(seed)
            solved_cost = _robust_cost(
                np.asarray(solved.fun, dtype=np.float64),
                loss=loss,
                f_scale=f_scale,
            )
            return np.asarray(seed, dtype=np.float64), solved, float(solved_cost)

        if restart_parallel_workers > 1 and len(restart_starts) > 1:
            restart_results = _threaded_map(
                _solve_restart,
                restart_starts,
                max_workers=restart_parallel_workers,
                numba_threads=numba_threads,
            )
        else:
            restart_results = [_solve_restart(seed) for seed in restart_starts]

        for restart_idx, (seed, trial, trial_cost) in enumerate(restart_results, start=1):
            restart_history.append(
                {
                    "restart": int(restart_idx),
                    "start_x": np.asarray(
                        _expand_active_vector(seed),
                        dtype=np.float64,
                    ).tolist(),
                    "end_x": np.asarray(
                        _expand_active_vector(trial.x),
                        dtype=np.float64,
                    ).tolist(),
                    "cost": float(trial_cost),
                    "success": bool(trial.success),
                    "message": str(trial.message),
                }
            )
            if float(trial_cost) < float(best_cost):
                best_result = trial
                best_cost = float(trial_cost)

    best_full_x = _expand_active_vector(np.asarray(best_result.x, dtype=np.float64))
    final_residual, roi_diagnostics, dataset_diagnostics, roi_count_by_dataset = _evaluate_residual(
        best_full_x, collect_diagnostics=True
    )
    best_result.x = np.asarray(best_full_x, dtype=np.float64)
    best_result.fun = np.asarray(final_residual, dtype=np.float64)
    final_cost = _robust_cost(best_result.fun, loss=loss, f_scale=f_scale)
    initial_residual_count = int(np.asarray(initial_residual, dtype=np.float64).size)
    final_residual_count = int(np.asarray(best_result.fun, dtype=np.float64).size)
    initial_residual_norm = float(np.linalg.norm(initial_residual))
    final_residual_norm = float(np.linalg.norm(best_result.fun))
    initial_residual_rms = (
        float(np.sqrt(np.mean(np.asarray(initial_residual, dtype=np.float64) ** 2)))
        if initial_residual_count
        else 0.0
    )
    final_residual_rms = (
        float(np.sqrt(np.mean(np.asarray(best_result.fun, dtype=np.float64) ** 2)))
        if final_residual_count
        else 0.0
    )

    best_params = dict(base_params)
    best_params["mosaic_params"] = dict(mosaic_params)
    best_params["mosaic_params"].update(
        {
            "beam_x_array": beam_x,
            "beam_y_array": beam_y,
            "theta_array": theta_array,
            "phi_array": phi_array,
            "wavelength_array": wavelength_array,
            "sigma_mosaic_deg": float(best_result.x[0]),
            "gamma_mosaic_deg": float(best_result.x[1]),
            "eta": float(best_result.x[2]),
        }
    )
    refined_theta_values_by_dataset: Dict[int, float] = {}
    refined_theta_offset: Optional[float] = None
    if optimize_theta:
        theta_slice = np.asarray(best_result.x[3:], dtype=np.float64)
        if resolved_theta_mode == "shared_offset" and theta_slice.size >= 1:
            refined_theta_offset = float(theta_slice[0])
            best_params["theta_offset"] = float(refined_theta_offset)
            for dataset_ctx in prepared_datasets:
                refined_theta_values_by_dataset[int(dataset_ctx.dataset_index)] = float(
                    float(dataset_ctx.theta_initial) + refined_theta_offset
                )
        elif resolved_theta_mode == "single" and theta_slice.size >= 1:
            refined_theta = float(theta_slice[0])
            best_params["theta_initial"] = float(refined_theta)
            refined_theta_values_by_dataset[int(prepared_datasets[0].dataset_index)] = float(
                refined_theta
            )
        elif resolved_theta_mode == "per_dataset" and theta_slice.size:
            theta_map = {
                int(dataset_idx): float(theta_slice[idx])
                for idx, dataset_idx in enumerate(theta_param_dataset_indices)
                if dataset_idx is not None and idx < theta_slice.size
            }
            best_params["_mosaic_theta_initials_by_dataset"] = dict(theta_map)
            refined_theta_values_by_dataset.update(theta_map)

    cost_reduction = float(
        (float(initial_cost) - float(final_cost)) / max(float(initial_cost), 1.0e-12)
    )
    bound_hits = [
        str(param_names[idx])
        for idx in active_indices.tolist()
        if np.isclose(best_result.x[idx], full_lower[idx], rtol=0.0, atol=1.0e-6)
        or np.isclose(best_result.x[idx], full_upper[idx], rtol=0.0, atol=1.0e-6)
    ]
    boundary_warning = None
    if bound_hits:
        boundary_warning = "Parameters finished on bounds: " + ", ".join(
            str(name) for name in bound_hits
        )

    ordered_roi_diagnostics = sorted(
        list(roi_diagnostics or []),
        key=lambda item: float(item["rms"]),
        reverse=True,
    )
    point_match_pairs_ok = True
    if point_match_term_enabled:
        point_match_pairs_ok = all(
            int(diag.get("matched_pair_count", 0)) > 0 for diag in (dataset_diagnostics or [])
        )
    best_result.best_params = best_params
    best_result.initial_cost = float(initial_cost)
    best_result.final_cost = float(final_cost)
    best_result.cost_reduction = float(cost_reduction)
    best_result.restart_history = restart_history
    best_result.boundary_warning = boundary_warning
    best_result.bound_hits = list(bound_hits)
    best_result.roi_diagnostics = ordered_roi_diagnostics
    best_result.rejected_rois = list(rejected_rois)
    best_result.dataset_diagnostics = list(dataset_diagnostics or [])
    best_result.roi_count_by_dataset = dict(roi_count_by_dataset or {})
    best_result.top_worst_rois = ordered_roi_diagnostics[:10]
    best_result.parallelization_summary = parallelization_summary
    best_result.roi_half_width = int(roi_half_width)
    best_result.total_roi_count = int(total_rois)
    best_result.solver_loss = str(loss)
    best_result.solver_f_scale = float(f_scale)
    best_result.active_parameters = list(active_parameter_names)
    best_result.fixed_parameters = list(fixed_parameter_names)
    best_result.active_parameter_indices = active_indices.tolist()
    best_result.fixed_parameter_indices = fixed_indices.tolist()
    best_result.fit_sigma_mosaic = bool(fit_sigma_mosaic)
    best_result.fit_gamma_mosaic = bool(fit_gamma_mosaic)
    best_result.fit_eta = bool(fit_eta)
    best_result.theta_refinement_mode = str(resolved_theta_mode) if optimize_theta else None
    best_result.theta_param_name = (
        str(theta_param_names[0]) if optimize_theta and len(theta_param_names) == 1 else None
    )
    best_result.theta_param_names = list(theta_param_names)
    best_result.refined_theta_value = (
        float(best_result.x[3])
        if optimize_theta and len(theta_param_names) == 1 and best_result.x.size >= 4
        else None
    )
    best_result.refined_theta_offset = (
        float(refined_theta_offset) if refined_theta_offset is not None else None
    )
    best_result.refined_theta_values_by_dataset = dict(refined_theta_values_by_dataset)
    best_result.ridge_weight = float(ridge_weight)
    best_result.point_match_weight = float(point_match_weight)
    best_result.acceptance_passed = bool(
        float(cost_reduction) >= 0.20
        and not bound_hits
        and point_match_pairs_ok
        and all(
            int(count) >= int(min_per_dataset_rois)
            for count in (roi_count_by_dataset or {}).values()
        )
    )
    parameter_debug: Dict[str, Dict[str, object]] = {}
    for idx, name in enumerate(param_names):
        parameter_debug[str(name)] = {
            "index": int(idx),
            "active": bool(active_mask[idx]),
            "initial": float(full_x0[idx]),
            "final": float(best_result.x[idx]),
            "delta": float(best_result.x[idx] - full_x0[idx]),
            "lower": float(full_lower[idx]),
            "upper": float(full_upper[idx]),
        }
    geometry_parameter_summary = {
        "a": _json_safe(base_params.get("a")),
        "c": _json_safe(base_params.get("c")),
        "lambda": _json_safe(base_params.get("lambda")),
        "psi": _json_safe(base_params.get("psi")),
        "psi_z": _json_safe(base_params.get("psi_z")),
        "zs": _json_safe(base_params.get("zs")),
        "zb": _json_safe(base_params.get("zb")),
        "sample_width_m": _json_safe(base_params.get("sample_width_m")),
        "sample_length_m": _json_safe(base_params.get("sample_length_m")),
        "sample_depth_m": _json_safe(base_params.get("sample_depth_m")),
        "chi": _json_safe(base_params.get("chi")),
        "n2": _json_safe(base_params.get("n2")),
        "center_xy_px": _json_safe(base_params.get("center")),
        "theta_initial_deg": _json_safe(base_params.get("theta_initial")),
        "theta_offset_deg": _json_safe(base_params.get("theta_offset")),
        "corto_detector_m": _json_safe(base_params.get("corto_detector")),
        "gamma_deg": _json_safe(base_params.get("gamma")),
        "Gamma_deg": _json_safe(base_params.get("Gamma")),
        "optics_mode": _json_safe(base_params.get("optics_mode")),
        "pixel_size_m": _json_safe(base_params.get("pixel_size_m")),
        "pixel_size": _json_safe(base_params.get("pixel_size")),
        "uv1": _json_safe(uv1),
        "uv2": _json_safe(uv2),
    }
    mosaic_sample_summary = {
        "beam_x": _array_summary(beam_x),
        "beam_y": _array_summary(beam_y),
        "theta": _array_summary(theta_array),
        "phi": _array_summary(phi_array),
        "wavelength": _array_summary(wavelength_array),
        "n2_sample_array": _array_summary(mosaic_params.get("n2_sample_array", [])),
        "solve_q_steps": int(mosaic_params.get("solve_q_steps", 1000)),
        "solve_q_rel_tol": float(mosaic_params.get("solve_q_rel_tol", 5.0e-4)),
        "solve_q_mode": int(mosaic_params.get("solve_q_mode", 1)),
    }
    acceptance_summary = {
        "passed": bool(best_result.acceptance_passed),
        "cost_reduction_threshold": 0.20,
        "cost_reduction": float(cost_reduction),
        "bound_hits": list(bound_hits),
        "boundary_warning": boundary_warning,
        "point_match_pairs_ok": bool(point_match_pairs_ok),
        "min_total_rois": int(min_total_rois),
        "min_per_dataset_rois": int(min_per_dataset_rois),
        "total_roi_count": int(total_rois),
        "roi_count_by_dataset": {
            str(key): int(val) for key, val in dict(roi_count_by_dataset or {}).items()
        },
    }
    mosaic_fit_debug_summary = {
        "inputs": {
            "image_size": int(image_size),
            "dataset_count": int(dataset_count),
            "miller_count": int(miller.shape[0]),
            "intensity_count": int(intensities.shape[0]),
            "roi_half_width": int(roi_half_width),
            "input_datasets": input_dataset_summaries,
            "prepared_datasets": prepared_dataset_summaries,
            "rejected_rois": _json_safe(rejected_rois),
            "rejected_roi_reason_counts": _json_safe(rejected_roi_reason_counts),
            "rejected_roi_counts_by_dataset": _json_safe(rejected_roi_counts_by_dataset),
            "geometry_parameters": geometry_parameter_summary,
            "mosaic_samples": mosaic_sample_summary,
        },
        "solver": {
            "loss": str(loss),
            "f_scale": float(f_scale),
            "max_nfev": int(max_nfev),
            "max_restarts": int(max_restarts),
            "restart_jitter": float(restart_jitter),
            "success": bool(best_result.success),
            "status": _json_safe(getattr(best_result, "status", None)),
            "message": str(best_result.message),
            "nfev": _json_safe(getattr(best_result, "nfev", None)),
            "njev": _json_safe(getattr(best_result, "njev", None)),
            "optimality": _json_safe(getattr(best_result, "optimality", None)),
            "active_parameters": list(active_parameter_names),
            "fixed_parameters": list(fixed_parameter_names),
            "parameter_bounds": parameter_debug,
            "initial_cost": float(initial_cost),
            "final_cost": float(final_cost),
            "cost_reduction": float(cost_reduction),
            "initial_residual_count": int(initial_residual_count),
            "final_residual_count": int(final_residual_count),
            "initial_residual_norm": float(initial_residual_norm),
            "final_residual_norm": float(final_residual_norm),
            "initial_residual_rms": float(initial_residual_rms),
            "final_residual_rms": float(final_residual_rms),
            "restart_history": _json_safe(restart_history),
        },
        "objective_terms": {
            "ridge_enabled": bool(ridge_term_enabled),
            "point_match_enabled": bool(point_match_term_enabled),
            "ridge_weight": float(ridge_weight),
            "point_match_weight": float(point_match_weight),
            "point_match_f_scale": float(point_match_f_scale),
            "point_match_weighted_matching": bool(point_match_weighted_matching),
            "point_match_missing_pair_penalty_px": float(point_match_missing_pair_penalty),
            "equal_dataset_weights": bool(equal_dataset_weights),
        },
        "theta": {
            "optimize_theta": bool(optimize_theta),
            "requested_mode": str(theta_mode),
            "resolved_mode": str(resolved_theta_mode),
            "theta_bounds": _json_safe(theta_bounds),
            "theta_parameter_names": list(theta_param_names),
            "refined_theta_value": _json_safe(best_result.refined_theta_value),
            "refined_theta_offset": _json_safe(best_result.refined_theta_offset),
            "refined_theta_values_by_dataset": _json_safe(refined_theta_values_by_dataset),
        },
        "diagnostics": {
            "parallelization": _json_safe(parallelization_summary),
            "dataset_diagnostics": _json_safe(dataset_diagnostics or []),
            "roi_diagnostics": _json_safe(ordered_roi_diagnostics),
            "top_worst_rois": _json_safe(ordered_roi_diagnostics[:10]),
        },
        "acceptance": acceptance_summary,
    }
    best_result.parameter_bounds = parameter_debug
    best_result.initial_parameter_values = {
        str(name): float(full_x0[idx]) for idx, name in enumerate(param_names)
    }
    best_result.final_parameter_values = {
        str(name): float(best_result.x[idx]) for idx, name in enumerate(param_names)
    }
    best_result.initial_residual_count = int(initial_residual_count)
    best_result.final_residual_count = int(final_residual_count)
    best_result.initial_residual_norm = float(initial_residual_norm)
    best_result.final_residual_norm = float(final_residual_norm)
    best_result.initial_residual_rms = float(initial_residual_rms)
    best_result.final_residual_rms = float(final_residual_rms)
    best_result.input_dataset_summaries = list(input_dataset_summaries)
    best_result.prepared_dataset_summaries = list(prepared_dataset_summaries)
    best_result.rejected_roi_reason_counts = dict(rejected_roi_reason_counts)
    best_result.rejected_roi_counts_by_dataset = dict(rejected_roi_counts_by_dataset)
    best_result.geometry_parameter_summary = dict(geometry_parameter_summary)
    best_result.mosaic_sample_summary = dict(mosaic_sample_summary)
    best_result.acceptance_summary = dict(acceptance_summary)
    best_result.mosaic_fit_debug_summary = dict(mosaic_fit_debug_summary)
    return best_result


_angular_difference_array_deg = _mosaic_profiles._angular_difference_array_deg
_classify_mosaic_profile_family = _mosaic_profiles._classify_mosaic_profile_family
_profile_half_span_deg = _mosaic_profiles._profile_half_span_deg
_profile_bin_indices = _mosaic_profiles._profile_bin_indices
_extract_profile_from_flat_image = _mosaic_profiles._extract_profile_from_flat_image
_estimate_pixel_size = _mosaic_profiles._estimate_pixel_size
_fit_space_pixel_size_provenance = _mosaic_profiles._fit_space_pixel_size_provenance


def _build_mosaic_profile_dataset_contexts(
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, object],
    dataset_specs: Sequence[Dict[str, object]],
    *,
    roi_half_width: int,
) -> Tuple[List[MosaicProfileDatasetContext], List[Dict[str, object]]]:
    """Prepare fixed angular-profile windows for the sigma/gamma/eta fitter."""

    return _mosaic_profiles._build_mosaic_profile_dataset_contexts(
        miller,
        intensities,
        image_size,
        params,
        dataset_specs,
        roi_half_width=roi_half_width,
        build_geometry_fit_dataset_contexts=_build_geometry_fit_dataset_contexts,
        miller_key_from_row=_miller_key_from_row,
        normalized_hkl_key=_normalized_hkl_key,
        measured_detector_anchor=_measured_detector_anchor,
        detector_pixels_to_fit_space=_detector_pixels_to_fit_space,
    )


def _fit_mosaic_shape_parameters_profiles(
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, object],
    *,
    dataset_specs: Sequence[Dict[str, object]],
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
    max_nfev: int = 80,
    max_restarts: int = 2,
    smooth_sigma_px: float = 1.0,
    ridge_percentile: float = 85.0,
    roi_half_width: Optional[int] = None,
    min_total_rois: int = 8,
    min_per_dataset_rois: int = 3,
    equal_dataset_weights: bool = True,
    workers: object = "auto",
    parallel_mode: str = "auto",
    worker_numba_threads: object = 0,
    restart_jitter: float = 0.15,
    ridge_weight: float = 1.0,
    specular_relative_intensity_weight: float = 0.0,
    fit_theta_i: bool = True,
    theta_i_mode: str = "auto",
    theta_i_bounds_deg: Optional[Tuple[float, float]] = None,
    fit_sigma_mosaic: bool = True,
    fit_gamma_mosaic: bool = True,
    fit_eta: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> OptimizeResult:
    """Fit sigma/gamma/eta using in-plane phi profiles plus specular 00L profiles."""

    del smooth_sigma_px
    del ridge_percentile

    dataset_spec_entries = _coerce_sequence_items(dataset_specs)
    miller = np.asarray(miller, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    if miller.ndim != 2 or miller.shape[1] != 3:
        raise ValueError("miller must be an array of shape (N, 3)")
    if intensities.ndim != 1 or intensities.shape[0] != miller.shape[0]:
        raise ValueError("intensities and miller must have matching lengths")
    if int(image_size) <= 0:
        raise ValueError("image_size must be positive")
    if not dataset_spec_entries:
        raise RuntimeError("Mosaic shape fit requires at least one prepared dataset_spec.")

    mosaic_params = dict(params.get("mosaic_params", {}))
    if not mosaic_params:
        raise ValueError("params['mosaic_params'] is required")
    required_keys = ("beam_x_array", "beam_y_array", "theta_array", "phi_array")
    missing_keys = [key for key in required_keys if not np.asarray(mosaic_params.get(key)).size]
    if missing_keys:
        raise ValueError("mosaic_params must include beam and divergence samples for shape fitting")

    wavelength_array = mosaic_params.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic_params.get("wavelength_i_array")
    if wavelength_array is None:
        base_lambda = float(params.get("lambda", 1.0))
        wavelength_array = np.full(
            np.asarray(mosaic_params["beam_x_array"]).shape,
            base_lambda,
            dtype=np.float64,
        )
    else:
        wavelength_array = np.asarray(wavelength_array, dtype=np.float64)

    sigma0 = float(mosaic_params.get("sigma_mosaic_deg", 0.5))
    gamma0 = float(mosaic_params.get("gamma_mosaic_deg", 0.5))
    eta0 = float(mosaic_params.get("eta", 0.05))
    if bounds is None:
        bounds = (
            np.array([0.03, 0.03, 0.0], dtype=np.float64),
            np.array([3.0, 3.0, 1.0], dtype=np.float64),
        )
    lower = np.asarray(bounds[0], dtype=np.float64).reshape(-1)
    upper = np.asarray(bounds[1], dtype=np.float64).reshape(-1)
    if lower.size != 3 or upper.size != 3:
        raise ValueError("bounds must contain exactly 3 lower/upper values")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("bounds must be finite")
    if np.any(lower >= upper):
        raise ValueError("each lower bound must be strictly less than upper bound")

    profile_weight = max(float(ridge_weight), 0.0)
    specular_ratio_weight = max(float(specular_relative_intensity_weight), 0.0)
    if profile_weight <= 0.0 and specular_ratio_weight <= 0.0:
        raise ValueError(
            "At least one mosaic objective term must be enabled "
            "(ridge_weight > 0 or specular_relative_intensity_weight > 0)."
        )

    x0 = np.clip(np.array([sigma0, gamma0, eta0], dtype=np.float64), lower, upper)
    fit_sigma_mosaic = bool(fit_sigma_mosaic)
    fit_gamma_mosaic = bool(fit_gamma_mosaic)
    fit_eta = bool(fit_eta)
    min_total_rois = max(int(min_total_rois), 1)
    min_per_dataset_rois = max(int(min_per_dataset_rois), 1)
    max_restarts = max(int(max_restarts), 0)
    restart_jitter = max(float(restart_jitter), 0.0)
    progress_lock = Lock()

    def _emit_progress(message: str) -> None:
        if not callable(progress_callback):
            return
        text = str(message).strip()
        if not text:
            return
        with progress_lock:
            try:
                progress_callback(text)
            except Exception:
                pass

    if roi_half_width is None:
        roi_half_width = _estimate_mosaic_shape_roi_half_width(
            params,
            upper,
            int(image_size),
        )
    roi_half_width = int(roi_half_width)
    if roi_half_width <= 0:
        raise ValueError("roi_half_width must be a positive integer")

    _emit_progress(
        "Preparing dataset ROIs: "
        f"datasets={len(dataset_spec_entries)}, roi_half_width={int(roi_half_width)}"
    )
    prepared_datasets, rejected_rois = _build_mosaic_profile_dataset_contexts(
        miller,
        intensities,
        int(image_size),
        dict(params),
        list(dataset_spec_entries),
        roi_half_width=int(roi_half_width),
    )
    if not prepared_datasets:
        raise RuntimeError("Mosaic shape fit found no prepared datasets.")

    rejected_reason_counts_by_dataset: Dict[str, Dict[str, int]] = {}
    for rejected in rejected_rois:
        dataset_label = str(rejected.get("dataset_label") or "")
        reason = str(rejected.get("reason") or "unknown")
        if not dataset_label:
            continue
        bucket = rejected_reason_counts_by_dataset.setdefault(dataset_label, {})
        bucket[reason] = int(bucket.get(reason, 0)) + 1

    dataset_failures = [
        dataset_ctx
        for dataset_ctx in prepared_datasets
        if len(dataset_ctx.rois) < int(min_per_dataset_rois)
    ]
    if dataset_failures:
        dataset_text = ", ".join(f"{ctx.label}={len(ctx.rois)}" for ctx in dataset_failures)
        rejection_text = ", ".join(
            f"{ctx.label}["
            + ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(
                    rejected_reason_counts_by_dataset.get(str(ctx.label), {}).items(),
                    key=lambda item: (-int(item[1]), str(item[0])),
                )[:3]
            )
            + "]"
            for ctx in dataset_failures
            if rejected_reason_counts_by_dataset.get(str(ctx.label))
        )
        detail_suffix = f" Rejections: {rejection_text}." if rejection_text else ""
        raise RuntimeError(
            "Mosaic shape fit needs at least "
            f"{int(min_per_dataset_rois)} usable ROIs per dataset; got {dataset_text}."
            f"{detail_suffix}"
        )

    total_rois = int(sum(len(dataset_ctx.rois) for dataset_ctx in prepared_datasets))
    if total_rois < int(min_total_rois):
        raise RuntimeError(
            f"Mosaic shape fit needs at least {int(min_total_rois)} usable ROIs; got {total_rois}."
        )

    total_in_plane = int(sum(ctx.in_plane_roi_count for ctx in prepared_datasets))
    total_specular = int(sum(ctx.specular_roi_count for ctx in prepared_datasets))
    if total_in_plane <= 0:
        raise RuntimeError("Mosaic shape fit needs at least one usable off-specular profile.")
    if total_specular <= 0:
        raise RuntimeError("Mosaic shape fit needs at least one usable specular (00l) profile.")
    if specular_ratio_weight > 0.0 and total_specular < 2:
        raise RuntimeError(
            "Mosaic shape fit needs at least two specular (00l) profiles when relative-intensity fitting is enabled."
        )

    def _json_safe(value: object) -> object:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, complex):
            return {
                "real": _json_safe(value.real),
                "imag": _json_safe(value.imag),
            }
        if isinstance(value, np.generic):
            return _json_safe(value.item())
        if isinstance(value, np.ndarray):
            return [_json_safe(item) for item in value.tolist()]
        if isinstance(value, Mapping):
            return {str(key): _json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)

    def _array_summary(values: object) -> Dict[str, object]:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = arr[np.isfinite(arr)]
        summary: Dict[str, object] = {
            "count": int(arr.size),
            "finite_count": int(finite.size),
        }
        if finite.size:
            summary.update(
                {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                }
            )
        else:
            summary.update(
                {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                }
            )
        return summary

    def _image_summary(image: object) -> Dict[str, object]:
        arr = np.asarray(image, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        summary: Dict[str, object] = {
            "shape": [int(dim) for dim in arr.shape],
            "finite_count": int(finite.size),
        }
        if finite.size:
            summary.update(
                {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "sum": float(np.sum(finite)),
                }
            )
        else:
            summary.update(
                {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "sum": None,
                }
            )
        return summary

    def _measured_peak_summaries(entries: object) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for entry in entries if isinstance(entries, (list, tuple)) else []:
            if not isinstance(entry, Mapping):
                continue
            peak_summary: Dict[str, object] = {
                "hkl": _json_safe(entry.get("hkl")),
                "x": _json_safe(entry.get("x")),
                "y": _json_safe(entry.get("y")),
            }
            if "source_table_index" in entry:
                peak_summary["source_table_index"] = _json_safe(entry.get("source_table_index"))
            if "source_row_index" in entry:
                peak_summary["source_row_index"] = _json_safe(entry.get("source_row_index"))
            out.append(peak_summary)
        return out

    input_dataset_summaries: List[Dict[str, object]] = []
    for spec in dataset_spec_entries:
        if not isinstance(spec, Mapping):
            continue
        measured_peaks = spec.get("measured_peaks", [])
        experimental_image = spec.get("experimental_image")
        input_dataset_summaries.append(
            {
                "dataset_index": _json_safe(spec.get("dataset_index")),
                "dataset_label": _json_safe(spec.get("label", spec.get("dataset_index", "?"))),
                "theta_initial_deg": _json_safe(spec.get("theta_initial")),
                "measured_peak_count": int(
                    len(measured_peaks) if isinstance(measured_peaks, (list, tuple)) else 0
                ),
                "experimental_image": (
                    _image_summary(experimental_image) if experimental_image is not None else None
                ),
                "measured_peaks": _measured_peak_summaries(measured_peaks),
            }
        )

    prepared_dataset_summaries: List[Dict[str, object]] = []
    for dataset_ctx in prepared_datasets:
        roi_entries = [
            {
                "reflection_index": int(roi.reflection_index),
                "hkl": [int(v) for v in roi.hkl],
                "family": str(roi.family),
                "axis_name": str(roi.axis_name),
                "center_xy_px": [float(roi.center_col), float(roi.center_row)],
                "row_bounds": [int(v) for v in roi.row_bounds],
                "col_bounds": [int(v) for v in roi.col_bounds],
                "measured_two_theta_deg": float(roi.measured_two_theta_deg),
                "measured_phi_deg": float(roi.measured_phi_deg),
                "measured_area": float(roi.measured_area),
            }
            for roi in dataset_ctx.rois
        ]
        prepared_dataset_summaries.append(
            {
                "dataset_index": int(dataset_ctx.dataset_index),
                "dataset_label": str(dataset_ctx.label),
                "theta_initial_deg": float(dataset_ctx.theta_initial),
                "experimental_image": _image_summary(dataset_ctx.experimental_image),
                "measured_peak_count": int(dataset_ctx.measured_peak_count),
                "reflection_count": int(dataset_ctx.miller.shape[0]),
                "intensity_count": int(dataset_ctx.intensities.shape[0]),
                "roi_count": int(len(dataset_ctx.rois)),
                "in_plane_roi_count": int(dataset_ctx.in_plane_roi_count),
                "specular_roi_count": int(dataset_ctx.specular_roi_count),
                "roi_hkls": [[int(v) for v in roi.hkl] for roi in dataset_ctx.rois],
                "rois": roi_entries,
            }
        )

    rejected_roi_reason_counts: Dict[str, int] = {}
    rejected_roi_counts_by_dataset: Dict[str, int] = {}
    for rejected in rejected_rois:
        if not isinstance(rejected, Mapping):
            continue
        dataset_key = str(rejected.get("dataset_label", rejected.get("dataset_index", "?")))
        reason_key = str(rejected.get("reason", "unknown"))
        stage_key = str(rejected.get("stage", "unknown"))
        rejected_roi_reason_counts[f"{dataset_key}:{stage_key}:{reason_key}"] = (
            rejected_roi_reason_counts.get(f"{dataset_key}:{stage_key}:{reason_key}", 0) + 1
        )
        rejected_roi_counts_by_dataset[dataset_key] = (
            rejected_roi_counts_by_dataset.get(dataset_key, 0) + 1
        )

    base_params = dict(params)
    beam_x = np.asarray(mosaic_params.get("beam_x_array"), dtype=np.float64)
    beam_y = np.asarray(mosaic_params.get("beam_y_array"), dtype=np.float64)
    theta_array = np.asarray(mosaic_params.get("theta_array"), dtype=np.float64)
    phi_array = np.asarray(mosaic_params.get("phi_array"), dtype=np.float64)
    uv1 = np.asarray(
        base_params.get("uv1", np.array([1.0, 0.0, 0.0])),
        dtype=np.float64,
    )
    uv2 = np.asarray(
        base_params.get("uv2", np.array([0.0, 1.0, 0.0])),
        dtype=np.float64,
    )

    dataset_count = int(len(prepared_datasets))
    optimize_theta = bool(fit_theta_i)

    theta_mode_key = str(theta_i_mode).strip().lower()
    if theta_mode_key not in {"auto", "single", "shared_offset", "per_dataset"}:
        raise ValueError(
            "theta_i_mode must be one of {'auto', 'single', 'shared_offset', 'per_dataset'}"
        )
    if theta_mode_key == "auto":
        resolved_theta_mode = "single" if dataset_count == 1 else "per_dataset"
    else:
        resolved_theta_mode = str(theta_mode_key)
    if resolved_theta_mode == "single" and dataset_count != 1:
        raise ValueError("theta_i_mode='single' requires exactly one prepared dataset")

    def _safe_float(value: object, fallback: float) -> float:
        try:
            out = float(value)
        except Exception:
            return float(fallback)
        if not np.isfinite(out):
            return float(fallback)
        return out

    param_names = ["sigma_mosaic_deg", "gamma_mosaic_deg", "eta"]
    theta_param_names: List[str] = []
    theta_param_dataset_indices: List[Optional[int]] = []
    if optimize_theta:
        theta_bound_lower: float
        theta_bound_upper: float
        if theta_i_bounds_deg is None:
            theta_bound_lower = -0.5
            theta_bound_upper = 0.5
        else:
            theta_bound_lower = float(theta_i_bounds_deg[0])
            theta_bound_upper = float(theta_i_bounds_deg[1])
        if not (np.isfinite(theta_bound_lower) and np.isfinite(theta_bound_upper)):
            raise ValueError("theta_i_bounds_deg must be finite when fit_theta_i is enabled")
        if theta_bound_lower >= theta_bound_upper:
            raise ValueError(
                "theta_i_bounds_deg lower bound must be strictly less than the upper bound"
            )

        theta_seeds: List[float] = []
        theta_lowers: List[float] = []
        theta_uppers: List[float] = []
        if resolved_theta_mode == "shared_offset":
            theta_seed = _safe_float(params.get("theta_offset", 0.0), 0.0)
            theta_param_names.append("theta_offset")
            theta_param_dataset_indices.append(None)
            theta_seeds.append(float(np.clip(theta_seed, theta_bound_lower, theta_bound_upper)))
            theta_lowers.append(float(theta_bound_lower))
            theta_uppers.append(float(theta_bound_upper))
        elif resolved_theta_mode == "single":
            theta_seed = _safe_float(
                params.get("theta_initial", prepared_datasets[0].theta_initial),
                float(prepared_datasets[0].theta_initial),
            )
            theta_param_names.append("theta_initial")
            theta_param_dataset_indices.append(int(prepared_datasets[0].dataset_index))
            theta_seeds.append(float(np.clip(theta_seed, theta_bound_lower, theta_bound_upper)))
            theta_lowers.append(float(theta_bound_lower))
            theta_uppers.append(float(theta_bound_upper))
        else:
            theta_seed_map = dict(base_params.get("_mosaic_theta_initials_by_dataset", {}))
            for dataset_ctx in prepared_datasets:
                dataset_idx = int(dataset_ctx.dataset_index)
                dataset_theta = float(dataset_ctx.theta_initial)
                theta_seed = _safe_float(
                    theta_seed_map.get(dataset_idx, dataset_theta),
                    dataset_theta,
                )
                lower_i = float(dataset_theta + theta_bound_lower)
                upper_i = float(dataset_theta + theta_bound_upper)
                theta_param_names.append(f"theta_initial[{dataset_idx}]")
                theta_param_dataset_indices.append(int(dataset_idx))
                theta_seeds.append(float(np.clip(theta_seed, lower_i, upper_i)))
                theta_lowers.append(float(lower_i))
                theta_uppers.append(float(upper_i))

        x0 = np.concatenate([x0, np.asarray(theta_seeds, dtype=np.float64)])
        lower = np.concatenate([lower, np.asarray(theta_lowers, dtype=np.float64)])
        upper = np.concatenate([upper, np.asarray(theta_uppers, dtype=np.float64)])

    if optimize_theta and theta_param_names:
        param_names.extend(str(name) for name in theta_param_names)
    active_mask = np.array(
        [
            fit_sigma_mosaic,
            fit_gamma_mosaic,
            fit_eta,
            *([True] * len(theta_param_names) if optimize_theta else []),
        ],
        dtype=bool,
    )
    active_indices = np.flatnonzero(active_mask)
    fixed_indices = np.flatnonzero(~active_mask)
    if active_indices.size == 0:
        raise ValueError(
            "At least one mosaic fit parameter must be enabled "
            "(fit_sigma_mosaic, fit_gamma_mosaic, fit_eta, or fit_theta_i)."
        )

    full_x0 = np.asarray(x0, dtype=np.float64)
    full_lower = np.asarray(lower, dtype=np.float64)
    full_upper = np.asarray(upper, dtype=np.float64)
    active_x0 = np.asarray(full_x0[active_indices], dtype=np.float64)
    active_lower = np.asarray(full_lower[active_indices], dtype=np.float64)
    active_upper = np.asarray(full_upper[active_indices], dtype=np.float64)
    active_parameter_names = [str(param_names[idx]) for idx in active_indices.tolist()]
    fixed_parameter_names = [str(param_names[idx]) for idx in fixed_indices.tolist()]

    configured_parallel_workers = _resolve_parallel_worker_count(
        workers,
        max_tasks=max(len(prepared_datasets), max_restarts + 1, 1),
    )
    parallel_mode_key = str(parallel_mode).strip().lower()
    if parallel_mode_key not in {"auto", "datasets", "restarts", "off"}:
        raise ValueError("parallel_mode must be one of {'auto', 'datasets', 'restarts', 'off'}")
    dataset_parallel_workers = 1
    restart_parallel_workers = 1
    if configured_parallel_workers > 1 and parallel_mode_key != "off":
        if parallel_mode_key in {"auto", "datasets"} and len(prepared_datasets) > 1:
            dataset_parallel_workers = min(
                int(configured_parallel_workers),
                len(prepared_datasets),
            )
        elif parallel_mode_key in {"auto", "restarts"} and max_restarts > 1:
            restart_parallel_workers = min(
                int(configured_parallel_workers),
                int(max_restarts),
            )
    active_outer_workers = max(dataset_parallel_workers, restart_parallel_workers)
    numba_threads = _resolve_numba_threads_per_worker(
        active_outer_workers,
        worker_numba_threads,
    )
    parallelization_summary = {
        "mode": str(parallel_mode_key),
        "configured_workers": int(configured_parallel_workers),
        "dataset_workers": int(dataset_parallel_workers),
        "restart_workers": int(restart_parallel_workers),
        "worker_numba_threads": (None if numba_threads is None else int(numba_threads)),
        "numba_thread_budget": int(_available_parallel_thread_budget()),
    }

    def _format_cost(value: object) -> str:
        try:
            out = float(value)
        except Exception:
            return "n/a"
        if not np.isfinite(out):
            return "n/a"
        return f"{out:.6g}"

    def _format_rms(values: Sequence[float]) -> str:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return "0"
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return "n/a"
        return f"{float(np.sqrt(np.mean(finite * finite))):.6g}"

    def _format_params(x_values: Sequence[float]) -> str:
        arr = np.asarray(x_values, dtype=np.float64).reshape(-1)
        if arr.size < 3:
            return "params=n/a"
        parts = [
            f"sigma={float(arr[0]):.4f}",
            f"gamma={float(arr[1]):.4f}",
            f"eta={float(arr[2]):.4f}",
        ]
        if optimize_theta and arr.size > 3:
            theta_slice = np.asarray(arr[3:], dtype=np.float64).reshape(-1)
            if resolved_theta_mode == "shared_offset" and theta_slice.size >= 1:
                parts.append(f"theta_offset={float(theta_slice[0]):+.4f}")
            elif resolved_theta_mode == "single" and theta_slice.size >= 1:
                parts.append(f"theta={float(theta_slice[0]):.4f}")
            elif theta_slice.size:
                theta_finite = theta_slice[np.isfinite(theta_slice)]
                if theta_finite.size:
                    parts.append(
                        "theta_range="
                        f"[{float(np.min(theta_finite)):.4f},{float(np.max(theta_finite)):.4f}]"
                    )
        return ", ".join(parts)

    rejection_preview = sorted(
        rejected_roi_reason_counts.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )
    _emit_progress(
        "Prepared "
        f"{len(prepared_datasets)} dataset(s): total_rois={int(total_rois)}, "
        f"in_plane={int(total_in_plane)}, specular={int(total_specular)}, "
        f"active_params={','.join(active_parameter_names)}, "
        f"fixed_params={','.join(fixed_parameter_names) if fixed_parameter_names else '-'}"
    )
    for dataset_ctx in prepared_datasets:
        _emit_progress(
            "Dataset "
            f"{dataset_ctx.label}: theta={float(dataset_ctx.theta_initial):.4f} deg, "
            f"rois={len(dataset_ctx.rois)}, in_plane={int(dataset_ctx.in_plane_roi_count)}, "
            f"specular={int(dataset_ctx.specular_roi_count)}, "
            f"reflections={int(dataset_ctx.miller.shape[0])}"
        )
    if rejected_rois:
        preview_text = ", ".join(f"{key}={count}" for key, count in rejection_preview[:6])
        if len(rejection_preview) > 6:
            preview_text += ", ..."
        _emit_progress(f"ROI rejections during prep: count={len(rejected_rois)} [{preview_text}]")
    _emit_progress(
        "Parallel plan: "
        f"mode={parallel_mode_key}, dataset_workers={int(dataset_parallel_workers)}, "
        f"restart_workers={int(restart_parallel_workers)}, "
        "worker_numba_threads="
        f"{'auto' if numba_threads is None else int(numba_threads)}"
    )

    def _expand_active_vector(x_trial: Sequence[float]) -> np.ndarray:
        x_arr = np.asarray(x_trial, dtype=np.float64).reshape(-1)
        if active_indices.size == full_x0.size:
            if x_arr.size != full_x0.size:
                raise ValueError("mosaic fit parameter vector length mismatch")
            return x_arr
        if x_arr.size != active_indices.size:
            raise ValueError("mosaic fit active parameter vector length mismatch")
        full = np.array(full_x0, copy=True)
        full[active_indices] = x_arr
        return full

    def _apply_trial_params(x_values: Sequence[float]) -> Dict[str, object]:
        local = dict(base_params)
        local_mosaic = dict(mosaic_params)
        local_mosaic["sigma_mosaic_deg"] = float(x_values[0])
        local_mosaic["gamma_mosaic_deg"] = float(x_values[1])
        local_mosaic["eta"] = float(x_values[2])
        local_mosaic["beam_x_array"] = beam_x
        local_mosaic["beam_y_array"] = beam_y
        local_mosaic["theta_array"] = theta_array
        local_mosaic["phi_array"] = phi_array
        local_mosaic["wavelength_array"] = wavelength_array
        local["mosaic_params"] = local_mosaic
        local.setdefault("theta_offset", 0.0)
        if optimize_theta:
            theta_slice = np.asarray(x_values[3:], dtype=np.float64)
            if resolved_theta_mode == "shared_offset" and theta_slice.size >= 1:
                local["theta_offset"] = float(theta_slice[0])
            elif resolved_theta_mode == "single" and theta_slice.size >= 1:
                local["theta_initial"] = float(theta_slice[0])
            elif resolved_theta_mode == "per_dataset" and theta_slice.size:
                local["_mosaic_theta_initials_by_dataset"] = {
                    int(dataset_idx): float(theta_slice[idx])
                    for idx, dataset_idx in enumerate(theta_param_dataset_indices)
                    if dataset_idx is not None and idx < theta_slice.size
                }
        return local

    def _theta_initial_for_dataset(
        local: Dict[str, object],
        dataset_ctx: MosaicProfileDatasetContext,
    ) -> float:
        theta_base = _safe_float(dataset_ctx.theta_initial, 0.0)
        if resolved_theta_mode == "shared_offset":
            return float(theta_base + _safe_float(local.get("theta_offset", 0.0), 0.0))
        if resolved_theta_mode == "single":
            return _safe_float(local.get("theta_initial", theta_base), theta_base)
        theta_map = local.get("_mosaic_theta_initials_by_dataset", {})
        if isinstance(theta_map, Mapping):
            return _safe_float(
                theta_map.get(int(dataset_ctx.dataset_index), theta_base),
                theta_base,
            )
        return float(theta_base)

    image_cache: Dict[Tuple[int, float, float, float, float], np.ndarray] = {}
    image_cache_lock = Lock()

    def _simulate_dataset_image(
        local: Dict[str, object],
        dataset_ctx: MosaicProfileDatasetContext,
        *,
        theta_value: float,
    ) -> np.ndarray:
        key = (
            int(dataset_ctx.dataset_index),
            float(local["mosaic_params"]["sigma_mosaic_deg"]),
            float(local["mosaic_params"]["gamma_mosaic_deg"]),
            float(local["mosaic_params"]["eta"]),
            float(theta_value),
        )
        retain_image_cache = _retain_fit_simulation_cache()
        if not retain_image_cache:
            with image_cache_lock:
                image_cache.clear()
            cached = None
        else:
            with image_cache_lock:
                cached = image_cache.get(key)
            if cached is not None:
                return cached

        local_mosaic = dict(local["mosaic_params"])
        wave_local = local_mosaic.get("wavelength_array")
        if wave_local is None:
            wave_local = local_mosaic.get("wavelength_i_array")
        if wave_local is None:
            wave_local = wavelength_array

        buffer = np.zeros((image_size, image_size), dtype=np.float64)
        image, *_ = _process_peaks_parallel_safe(
            dataset_ctx.miller,
            dataset_ctx.intensities,
            image_size,
            local["a"],
            local["c"],
            wave_local,
            buffer,
            local["corto_detector"],
            local["gamma"],
            local["Gamma"],
            local["chi"],
            local.get("psi", 0.0),
            local.get("psi_z", 0.0),
            local["zs"],
            local["zb"],
            local["n2"],
            local_mosaic["beam_x_array"],
            local_mosaic["beam_y_array"],
            local_mosaic["theta_array"],
            local_mosaic["phi_array"],
            local_mosaic["sigma_mosaic_deg"],
            local_mosaic["gamma_mosaic_deg"],
            local_mosaic["eta"],
            wave_local,
            local["debye_x"],
            local["debye_y"],
            local["center"],
            float(theta_value),
            local.get("cor_angle", 0.0),
            uv1,
            uv2,
            save_flag=0,
            **_simulation_kernel_kwargs(local, local_mosaic),
        )
        cached = np.asarray(image, dtype=np.float64)
        if retain_image_cache:
            with image_cache_lock:
                existing = image_cache.get(key)
                if existing is not None:
                    return existing
                image_cache[key] = cached
        return cached

    def _evaluate_one_dataset(
        item: Tuple[Dict[str, object], MosaicProfileDatasetContext, bool],
    ) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
        local, dataset_ctx, collect_diagnostics = item
        theta_value = _theta_initial_for_dataset(local, dataset_ctx)
        sim_image = _simulate_dataset_image(
            local,
            dataset_ctx,
            theta_value=float(theta_value),
        )
        flat_sim = np.asarray(sim_image, dtype=np.float64).ravel()
        residual_blocks: List[np.ndarray] = []
        roi_diags: List[Dict[str, object]] = []
        specular_bundle: List[Tuple[MosaicProfileROI, np.ndarray, float]] = []

        for roi in dataset_ctx.rois:
            sim_profile = _extract_profile_from_flat_image(flat_sim, roi)
            sim_area = float(np.sum(sim_profile))
            if roi.family == "in_plane":
                sim_shape = (
                    np.asarray(sim_profile, dtype=np.float64) / sim_area
                    if np.isfinite(sim_area) and sim_area > 0.0
                    else np.zeros_like(roi.measured_shape_profile, dtype=np.float64)
                )
                residual_raw = sim_shape - roi.measured_shape_profile
                residual = np.asarray(residual_raw, dtype=np.float64)
                if residual.size:
                    residual = residual / math.sqrt(max(residual.size, 1))
                if profile_weight > 0.0:
                    residual_blocks.append(residual * float(profile_weight))
                if collect_diagnostics:
                    roi_diags.append(
                        {
                            "dataset_index": int(dataset_ctx.dataset_index),
                            "dataset_label": str(dataset_ctx.label),
                            "hkl": tuple(int(v) for v in roi.hkl),
                            "family": str(roi.family),
                            "axis_name": str(roi.axis_name),
                            "center": (float(roi.center_col), float(roi.center_row)),
                            "measured_area": float(roi.measured_area),
                            "simulated_area": float(sim_area),
                            "rms": (
                                float(np.sqrt(np.mean(residual_raw * residual_raw)))
                                if residual_raw.size
                                else 0.0
                            ),
                        }
                    )
            else:
                specular_bundle.append(
                    (roi, np.asarray(sim_profile, dtype=np.float64), float(sim_area))
                )

        shared_specular_scale = float("nan")
        ratio_term_count = 0
        if specular_bundle:
            sim_concat = np.concatenate([entry[1] for entry in specular_bundle]).astype(
                np.float64,
                copy=False,
            )
            meas_concat = np.concatenate(
                [entry[0].measured_profile for entry in specular_bundle]
            ).astype(np.float64, copy=False)
            denom = float(np.dot(sim_concat, sim_concat))
            if np.isfinite(denom) and denom > 1.0e-12:
                shared_specular_scale = max(
                    0.0,
                    float(np.dot(meas_concat, sim_concat) / denom),
                )
            else:
                shared_specular_scale = 0.0

            for roi, sim_profile, sim_area in specular_bundle:
                residual_raw = (
                    float(shared_specular_scale) * np.asarray(sim_profile, dtype=np.float64)
                    - roi.measured_profile
                )
                residual = np.asarray(residual_raw, dtype=np.float64)
                if residual.size:
                    residual = residual / math.sqrt(max(roi.measured_area, 1.0))
                if profile_weight > 0.0:
                    residual_blocks.append(residual * float(profile_weight))
                if collect_diagnostics:
                    roi_diags.append(
                        {
                            "dataset_index": int(dataset_ctx.dataset_index),
                            "dataset_label": str(dataset_ctx.label),
                            "hkl": tuple(int(v) for v in roi.hkl),
                            "family": str(roi.family),
                            "axis_name": str(roi.axis_name),
                            "center": (float(roi.center_col), float(roi.center_row)),
                            "measured_area": float(roi.measured_area),
                            "simulated_area": float(sim_area),
                            "shared_specular_scale": float(shared_specular_scale),
                            "rms": (
                                float(np.sqrt(np.mean(residual_raw * residual_raw)))
                                if residual_raw.size
                                else 0.0
                            ),
                        }
                    )

            if specular_ratio_weight > 0.0 and len(specular_bundle) > 1:
                ref_index = int(
                    np.argmax([float(entry[0].measured_area) for entry in specular_bundle])
                )
                ref_measured_area = max(
                    float(specular_bundle[ref_index][0].measured_area),
                    1.0e-12,
                )
                ref_simulated_area = max(
                    float(specular_bundle[ref_index][2]),
                    1.0e-12,
                )
                ratio_terms: List[float] = []
                for idx, (roi, _sim_profile, sim_area) in enumerate(specular_bundle):
                    if idx == ref_index:
                        continue
                    log_sim = math.log(max(float(sim_area), 1.0e-12) / ref_simulated_area)
                    log_meas = math.log(max(float(roi.measured_area), 1.0e-12) / ref_measured_area)
                    ratio_terms.append(float(log_sim - log_meas))
                ratio_term_count = int(len(ratio_terms))
                if ratio_terms:
                    residual_blocks.append(
                        np.asarray(ratio_terms, dtype=np.float64) * float(specular_ratio_weight)
                    )

        dataset_residual = (
            np.concatenate(residual_blocks) if residual_blocks else np.zeros(0, dtype=np.float64)
        )
        dataset_weight = (
            1.0 / math.sqrt(max(len(dataset_ctx.rois), 1)) if bool(equal_dataset_weights) else 1.0
        )
        dataset_residual = np.asarray(dataset_residual, dtype=np.float64) * dataset_weight

        ordered_roi_diags = sorted(
            roi_diags,
            key=lambda item: float(item["rms"]),
            reverse=True,
        )
        dataset_summary = {
            "dataset_index": int(dataset_ctx.dataset_index),
            "dataset_label": str(dataset_ctx.label),
            "theta_initial_deg": float(theta_value),
            "roi_count": int(len(dataset_ctx.rois)),
            "measured_peak_count": int(dataset_ctx.measured_peak_count),
            "simulated_reflection_count": int(dataset_ctx.miller.shape[0]),
            "dataset_weight": float(dataset_weight),
            "in_plane_roi_count": int(dataset_ctx.in_plane_roi_count),
            "specular_roi_count": int(dataset_ctx.specular_roi_count),
            "profile_weight": float(profile_weight),
            "specular_ratio_weight": float(specular_ratio_weight),
            "shared_specular_scale": (
                float(shared_specular_scale) if np.isfinite(shared_specular_scale) else None
            ),
            "relative_intensity_term_count": int(ratio_term_count),
            "worst_hkls": [tuple(int(v) for v in diag["hkl"]) for diag in ordered_roi_diags[:3]],
        }
        if collect_diagnostics:
            dataset_summary.update(
                {
                    "residual_norm": float(np.linalg.norm(dataset_residual)),
                    "cost": float(_robust_cost(dataset_residual, loss=loss, f_scale=f_scale)),
                    "max_roi_rms": (
                        float(ordered_roi_diags[0]["rms"]) if ordered_roi_diags else 0.0
                    ),
                }
            )
        return dataset_residual, roi_diags, dataset_summary

    def _evaluate_residual(
        theta: np.ndarray,
        *,
        collect_diagnostics: bool = False,
    ) -> Tuple[
        np.ndarray,
        Optional[List[Dict[str, object]]],
        Optional[List[Dict[str, object]]],
        Optional[Dict[int, int]],
    ]:
        local = _apply_trial_params(theta)
        dataset_items = [
            (local, dataset_ctx, bool(collect_diagnostics)) for dataset_ctx in prepared_datasets
        ]
        if dataset_parallel_workers > 1 and len(dataset_items) > 1:
            dataset_results = _threaded_map(
                _evaluate_one_dataset,
                dataset_items,
                max_workers=dataset_parallel_workers,
                numba_threads=numba_threads,
            )
        else:
            dataset_results = [_evaluate_one_dataset(item) for item in dataset_items]
        residual_blocks = [item[0] for item in dataset_results]
        all_roi_diags: List[Dict[str, object]] = []
        dataset_diagnostics: List[Dict[str, object]] = []
        roi_count_by_dataset: Dict[int, int] = {}
        for (_, roi_diags, dataset_summary), dataset_ctx in zip(
            dataset_results,
            prepared_datasets,
        ):
            if collect_diagnostics:
                all_roi_diags.extend(list(roi_diags))
                dataset_diagnostics.append(dict(dataset_summary))
            roi_count_by_dataset[int(dataset_ctx.dataset_index)] = int(len(dataset_ctx.rois))
        residual = (
            np.concatenate(residual_blocks) if residual_blocks else np.zeros(0, dtype=np.float64)
        )
        return (
            np.asarray(residual, dtype=np.float64),
            all_roi_diags if collect_diagnostics else None,
            dataset_diagnostics if collect_diagnostics else None,
            roi_count_by_dataset if collect_diagnostics else None,
        )

    initial_residual, _, _, _ = _evaluate_residual(full_x0)
    initial_cost = _robust_cost(initial_residual, loss=loss, f_scale=f_scale)
    _emit_progress(
        "Initial objective: "
        f"cost={_format_cost(initial_cost)}, residuals={int(np.asarray(initial_residual).size)}, "
        f"rms={_format_rms(initial_residual)}, {_format_params(full_x0)}"
    )

    def _run_solver(
        seed: np.ndarray,
        *,
        attempt_label: str,
        emit_evaluations: bool,
    ) -> Tuple[OptimizeResult, Dict[str, object]]:
        objective_state: Dict[str, object] = {
            "eval_count": 0,
            "best_cost": float("inf"),
        }

        def _objective_active(active_x: Sequence[float]) -> np.ndarray:
            full = _expand_active_vector(active_x)
            residual, _, _, _ = _evaluate_residual(full)
            if emit_evaluations:
                residual_arr = np.asarray(residual, dtype=np.float64)
                objective_state["eval_count"] = int(objective_state["eval_count"]) + 1
                eval_count = int(objective_state["eval_count"])
                cost = float(_robust_cost(residual_arr, loss=loss, f_scale=f_scale))
                best_cost = float(objective_state["best_cost"])
                improvement_margin = max(1.0e-6, 5.0e-3 * max(abs(best_cost), 1.0))
                improved = not np.isfinite(best_cost) or cost < (best_cost - improvement_margin)
                if improved:
                    objective_state["best_cost"] = float(cost)
                if eval_count == 1 or eval_count % 10 == 0 or improved:
                    _emit_progress(
                        f"{attempt_label} eval {eval_count}: "
                        f"cost={_format_cost(cost)}, rms={_format_rms(residual_arr)}, "
                        f"{_format_params(full)}"
                    )
            return np.asarray(residual, dtype=np.float64)

        result = least_squares(
            _objective_active,
            np.asarray(seed, dtype=np.float64),
            bounds=(active_lower, active_upper),
            loss=str(loss),
            f_scale=float(f_scale),
            max_nfev=int(max_nfev),
        )
        meta = {
            "eval_count": int(objective_state["eval_count"]),
            "best_cost": (
                None
                if not np.isfinite(float(objective_state["best_cost"]))
                else float(objective_state["best_cost"])
            ),
        }
        return result, meta

    _emit_progress(f"Primary solve start: max_nfev={int(max_nfev)}, {_format_params(full_x0)}")
    best_result, primary_meta = _run_solver(
        active_x0,
        attempt_label="Primary",
        emit_evaluations=True,
    )
    best_cost = _robust_cost(
        np.asarray(best_result.fun, dtype=np.float64),
        loss=loss,
        f_scale=f_scale,
    )
    _emit_progress(
        "Primary solve done: "
        f"success={bool(best_result.success)}, cost={_format_cost(best_cost)}, "
        f"nfev={_json_safe(getattr(best_result, 'nfev', None))}, "
        f"evals={int(primary_meta.get('eval_count', 0))}, "
        f"message={str(getattr(best_result, 'message', '') or '').strip() or 'n/a'}, "
        f"{_format_params(_expand_active_vector(np.asarray(best_result.x, dtype=np.float64)))}"
    )
    restart_history: List[Dict[str, object]] = []

    if max_restarts > 0 and active_x0.size:
        restart_rng = np.random.default_rng(42)
        span = (active_upper - active_lower) * float(restart_jitter)
        restart_starts = [
            np.clip(
                active_x0
                + restart_rng.uniform(-1.0, 1.0, size=active_x0.size).astype(np.float64) * span,
                active_lower,
                active_upper,
            )
            for _ in range(int(max_restarts))
        ]
        if restart_starts:
            _emit_progress(
                "Restarts scheduled: "
                f"count={len(restart_starts)}, mode="
                f"{'parallel' if restart_parallel_workers > 1 and len(restart_starts) > 1 else 'sequential'}"
            )

        def _solve_restart(
            seed: np.ndarray,
        ) -> Tuple[np.ndarray, OptimizeResult, float, Dict[str, object]]:
            trial, trial_meta = _run_solver(
                seed,
                attempt_label="Restart",
                emit_evaluations=False,
            )
            trial_cost = _robust_cost(
                np.asarray(trial.fun, dtype=np.float64),
                loss=loss,
                f_scale=f_scale,
            )
            return (
                np.asarray(seed, dtype=np.float64),
                trial,
                float(trial_cost),
                dict(trial_meta),
            )

        if restart_parallel_workers > 1 and len(restart_starts) > 1:
            restart_results = _threaded_map(
                _solve_restart,
                restart_starts,
                max_workers=restart_parallel_workers,
                numba_threads=numba_threads,
            )
        else:
            restart_results = []
            for restart_idx, seed in enumerate(restart_starts, start=1):
                _emit_progress(
                    f"Restart {restart_idx}/{len(restart_starts)} start: "
                    f"{_format_params(_expand_active_vector(seed))}"
                )
                trial, trial_meta = _run_solver(
                    seed,
                    attempt_label=f"Restart {restart_idx}/{len(restart_starts)}",
                    emit_evaluations=True,
                )
                trial_cost = _robust_cost(
                    np.asarray(trial.fun, dtype=np.float64),
                    loss=loss,
                    f_scale=f_scale,
                )
                restart_results.append(
                    (
                        np.asarray(seed, dtype=np.float64),
                        trial,
                        float(trial_cost),
                        dict(trial_meta),
                    )
                )

        for restart_idx, (seed, trial, trial_cost, trial_meta) in enumerate(
            restart_results,
            start=1,
        ):
            restart_history.append(
                {
                    "restart": int(restart_idx),
                    "start_x": np.asarray(_expand_active_vector(seed), dtype=np.float64).tolist(),
                    "end_x": np.asarray(_expand_active_vector(trial.x), dtype=np.float64).tolist(),
                    "cost": float(trial_cost),
                    "success": bool(trial.success),
                    "message": str(trial.message),
                }
            )
            _emit_progress(
                f"Restart {restart_idx}/{len(restart_starts)} done: "
                f"success={bool(trial.success)}, cost={_format_cost(trial_cost)}, "
                f"nfev={_json_safe(getattr(trial, 'nfev', None))}, "
                f"evals={int(trial_meta.get('eval_count', 0))}, "
                f"{_format_params(_expand_active_vector(np.asarray(trial.x, dtype=np.float64)))}"
            )
            if float(trial_cost) < float(best_cost):
                _emit_progress(
                    f"Restart {restart_idx}/{len(restart_starts)} is the new best solution."
                )
                best_result = trial
                best_cost = float(trial_cost)

    best_full_x = _expand_active_vector(np.asarray(best_result.x, dtype=np.float64))
    final_residual, roi_diagnostics, dataset_diagnostics, roi_count_by_dataset = _evaluate_residual(
        best_full_x, collect_diagnostics=True
    )
    best_result.x = np.asarray(best_full_x, dtype=np.float64)
    best_result.fun = np.asarray(final_residual, dtype=np.float64)
    final_cost = _robust_cost(best_result.fun, loss=loss, f_scale=f_scale)

    initial_residual_count = int(np.asarray(initial_residual, dtype=np.float64).size)
    final_residual_count = int(np.asarray(best_result.fun, dtype=np.float64).size)
    initial_residual_norm = float(np.linalg.norm(initial_residual))
    final_residual_norm = float(np.linalg.norm(best_result.fun))
    initial_residual_rms = (
        float(np.sqrt(np.mean(np.asarray(initial_residual, dtype=np.float64) ** 2)))
        if initial_residual_count
        else 0.0
    )
    final_residual_rms = (
        float(np.sqrt(np.mean(np.asarray(best_result.fun, dtype=np.float64) ** 2)))
        if final_residual_count
        else 0.0
    )

    best_params = dict(base_params)
    best_params["mosaic_params"] = dict(mosaic_params)
    best_params["mosaic_params"].update(
        {
            "beam_x_array": beam_x,
            "beam_y_array": beam_y,
            "theta_array": theta_array,
            "phi_array": phi_array,
            "wavelength_array": wavelength_array,
            "sigma_mosaic_deg": float(best_result.x[0]),
            "gamma_mosaic_deg": float(best_result.x[1]),
            "eta": float(best_result.x[2]),
        }
    )
    refined_theta_values_by_dataset: Dict[int, float] = {}
    refined_theta_offset: Optional[float] = None
    if optimize_theta:
        theta_slice = np.asarray(best_result.x[3:], dtype=np.float64)
        if resolved_theta_mode == "shared_offset" and theta_slice.size >= 1:
            refined_theta_offset = float(theta_slice[0])
            best_params["theta_offset"] = float(refined_theta_offset)
            for dataset_ctx in prepared_datasets:
                refined_theta_values_by_dataset[int(dataset_ctx.dataset_index)] = float(
                    float(dataset_ctx.theta_initial) + refined_theta_offset
                )
        elif resolved_theta_mode == "single" and theta_slice.size >= 1:
            refined_theta = float(theta_slice[0])
            best_params["theta_initial"] = float(refined_theta)
            refined_theta_values_by_dataset[int(prepared_datasets[0].dataset_index)] = float(
                refined_theta
            )
        elif resolved_theta_mode == "per_dataset" and theta_slice.size:
            theta_map = {
                int(dataset_idx): float(theta_slice[idx])
                for idx, dataset_idx in enumerate(theta_param_dataset_indices)
                if dataset_idx is not None and idx < theta_slice.size
            }
            best_params["_mosaic_theta_initials_by_dataset"] = dict(theta_map)
            refined_theta_values_by_dataset.update(theta_map)

    cost_reduction = 0.0
    if initial_cost > 1.0e-12 and np.isfinite(initial_cost):
        cost_reduction = float((float(initial_cost) - float(final_cost)) / float(initial_cost))
    bound_hits = [
        str(param_names[idx])
        for idx in active_indices.tolist()
        if np.isclose(best_result.x[idx], full_lower[idx], rtol=0.0, atol=1.0e-6)
        or np.isclose(best_result.x[idx], full_upper[idx], rtol=0.0, atol=1.0e-6)
    ]
    boundary_warning = None
    if bound_hits:
        boundary_warning = "Parameters finished on bounds: " + ", ".join(
            str(name) for name in bound_hits
        )

    ordered_roi_diagnostics = sorted(
        list(roi_diagnostics or []),
        key=lambda item: float(item["rms"]),
        reverse=True,
    )
    best_result.best_params = best_params
    best_result.initial_cost = float(initial_cost)
    best_result.final_cost = float(final_cost)
    best_result.cost_reduction = float(cost_reduction)
    best_result.restart_history = restart_history
    best_result.boundary_warning = boundary_warning
    best_result.bound_hits = list(bound_hits)
    best_result.roi_diagnostics = ordered_roi_diagnostics
    best_result.rejected_rois = list(rejected_rois)
    best_result.dataset_diagnostics = list(dataset_diagnostics or [])
    best_result.roi_count_by_dataset = dict(roi_count_by_dataset or {})
    best_result.top_worst_rois = ordered_roi_diagnostics[:10]
    best_result.parallelization_summary = dict(parallelization_summary)
    best_result.roi_half_width = int(roi_half_width)
    best_result.total_roi_count = int(total_rois)
    best_result.solver_loss = str(loss)
    best_result.solver_f_scale = float(f_scale)
    best_result.active_parameters = list(active_parameter_names)
    best_result.fixed_parameters = list(fixed_parameter_names)
    best_result.active_parameter_indices = active_indices.tolist()
    best_result.fixed_parameter_indices = fixed_indices.tolist()
    best_result.fit_sigma_mosaic = bool(fit_sigma_mosaic)
    best_result.fit_gamma_mosaic = bool(fit_gamma_mosaic)
    best_result.fit_eta = bool(fit_eta)
    best_result.fit_theta_i = bool(optimize_theta)
    best_result.theta_refinement_mode = str(resolved_theta_mode) if optimize_theta else None
    best_result.theta_param_name = (
        str(theta_param_names[0]) if optimize_theta and len(theta_param_names) == 1 else None
    )
    best_result.theta_param_names = list(theta_param_names)
    best_result.refined_theta_value = (
        float(best_result.x[3])
        if optimize_theta and len(theta_param_names) == 1 and best_result.x.size >= 4
        else None
    )
    best_result.refined_theta_offset = (
        float(refined_theta_offset) if refined_theta_offset is not None else None
    )
    best_result.refined_theta_values_by_dataset = dict(refined_theta_values_by_dataset)
    best_result.ridge_weight = float(profile_weight)
    best_result.specular_relative_intensity_weight = float(specular_ratio_weight)
    best_result.point_match_weight = float(specular_ratio_weight)
    best_result.acceptance_passed = bool(
        float(cost_reduction) >= 0.20
        and not bound_hits
        and total_in_plane > 0
        and total_specular > 0
        and all(
            int(count) >= int(min_per_dataset_rois)
            for count in (roi_count_by_dataset or {}).values()
        )
    )

    parameter_debug: Dict[str, Dict[str, object]] = {}
    for idx, name in enumerate(param_names):
        parameter_debug[str(name)] = {
            "index": int(idx),
            "active": bool(active_mask[idx]),
            "initial": float(full_x0[idx]),
            "final": float(best_result.x[idx]),
            "delta": float(best_result.x[idx] - full_x0[idx]),
            "lower": float(full_lower[idx]),
            "upper": float(full_upper[idx]),
        }

    geometry_parameter_summary = {
        "a": _json_safe(base_params.get("a")),
        "c": _json_safe(base_params.get("c")),
        "lambda": _json_safe(base_params.get("lambda")),
        "psi": _json_safe(base_params.get("psi")),
        "psi_z": _json_safe(base_params.get("psi_z")),
        "zs": _json_safe(base_params.get("zs")),
        "zb": _json_safe(base_params.get("zb")),
        "sample_width_m": _json_safe(base_params.get("sample_width_m")),
        "sample_length_m": _json_safe(base_params.get("sample_length_m")),
        "sample_depth_m": _json_safe(base_params.get("sample_depth_m")),
        "chi": _json_safe(base_params.get("chi")),
        "n2": _json_safe(base_params.get("n2")),
        "center_xy_px": _json_safe(base_params.get("center")),
        "theta_initial_deg": _json_safe(base_params.get("theta_initial")),
        "theta_offset_deg": _json_safe(base_params.get("theta_offset")),
        "corto_detector_m": _json_safe(base_params.get("corto_detector")),
        "gamma_deg": _json_safe(base_params.get("gamma")),
        "Gamma_deg": _json_safe(base_params.get("Gamma")),
        "optics_mode": _json_safe(base_params.get("optics_mode")),
        "pixel_size_m": _json_safe(base_params.get("pixel_size_m")),
        "pixel_size": _json_safe(base_params.get("pixel_size")),
        "uv1": _json_safe(uv1),
        "uv2": _json_safe(uv2),
    }
    mosaic_sample_summary = {
        "beam_x": _array_summary(beam_x),
        "beam_y": _array_summary(beam_y),
        "theta": _array_summary(theta_array),
        "phi": _array_summary(phi_array),
        "wavelength": _array_summary(wavelength_array),
        "n2_sample_array": _array_summary(mosaic_params.get("n2_sample_array", [])),
        "solve_q_steps": int(mosaic_params.get("solve_q_steps", 1000)),
        "solve_q_rel_tol": float(mosaic_params.get("solve_q_rel_tol", 5.0e-4)),
        "solve_q_mode": int(mosaic_params.get("solve_q_mode", 1)),
    }
    acceptance_summary = {
        "passed": bool(best_result.acceptance_passed),
        "cost_reduction_threshold": 0.20,
        "cost_reduction": float(cost_reduction),
        "bound_hits": list(bound_hits),
        "boundary_warning": boundary_warning,
        "min_total_rois": int(min_total_rois),
        "min_per_dataset_rois": int(min_per_dataset_rois),
        "total_roi_count": int(total_rois),
        "roi_count_by_dataset": {
            str(key): int(val) for key, val in dict(roi_count_by_dataset or {}).items()
        },
        "total_in_plane_roi_count": int(total_in_plane),
        "total_specular_roi_count": int(total_specular),
    }
    mosaic_fit_debug_summary = {
        "inputs": {
            "image_size": int(image_size),
            "dataset_count": int(len(prepared_datasets)),
            "miller_count": int(miller.shape[0]),
            "intensity_count": int(intensities.shape[0]),
            "roi_half_width": int(roi_half_width),
            "input_datasets": input_dataset_summaries,
            "prepared_datasets": prepared_dataset_summaries,
            "rejected_rois": _json_safe(rejected_rois),
            "rejected_roi_reason_counts": _json_safe(rejected_roi_reason_counts),
            "rejected_roi_counts_by_dataset": _json_safe(rejected_roi_counts_by_dataset),
            "geometry_parameters": geometry_parameter_summary,
            "mosaic_samples": mosaic_sample_summary,
        },
        "solver": {
            "loss": str(loss),
            "f_scale": float(f_scale),
            "max_nfev": int(max_nfev),
            "max_restarts": int(max_restarts),
            "restart_jitter": float(restart_jitter),
            "progress_callback_enabled": bool(callable(progress_callback)),
            "success": bool(best_result.success),
            "status": _json_safe(getattr(best_result, "status", None)),
            "message": str(best_result.message),
            "nfev": _json_safe(getattr(best_result, "nfev", None)),
            "njev": _json_safe(getattr(best_result, "njev", None)),
            "optimality": _json_safe(getattr(best_result, "optimality", None)),
            "active_parameters": list(active_parameter_names),
            "fixed_parameters": list(fixed_parameter_names),
            "parameter_bounds": parameter_debug,
            "initial_cost": float(initial_cost),
            "final_cost": float(final_cost),
            "cost_reduction": float(cost_reduction),
            "initial_residual_count": int(initial_residual_count),
            "final_residual_count": int(final_residual_count),
            "initial_residual_norm": float(initial_residual_norm),
            "final_residual_norm": float(final_residual_norm),
            "initial_residual_rms": float(initial_residual_rms),
            "final_residual_rms": float(final_residual_rms),
            "restart_history": _json_safe(restart_history),
        },
        "objective_terms": {
            "profile_shape_enabled": bool(profile_weight > 0.0),
            "specular_relative_intensity_enabled": bool(specular_ratio_weight > 0.0),
            "profile_weight": float(profile_weight),
            "specular_ratio_weight": float(specular_ratio_weight),
            "equal_dataset_weights": bool(equal_dataset_weights),
        },
        "theta": {
            "optimize_theta": bool(optimize_theta),
            "requested_mode": str(theta_i_mode),
            "resolved_mode": (str(resolved_theta_mode) if optimize_theta else None),
            "theta_i_bounds_deg": _json_safe(theta_i_bounds_deg),
            "theta_parameter_names": list(theta_param_names),
            "refined_theta_value": _json_safe(best_result.refined_theta_value),
            "refined_theta_offset": _json_safe(best_result.refined_theta_offset),
            "refined_theta_values_by_dataset": _json_safe(
                best_result.refined_theta_values_by_dataset
            ),
        },
        "diagnostics": {
            "parallelization": _json_safe(parallelization_summary),
            "dataset_diagnostics": _json_safe(dataset_diagnostics or []),
            "roi_diagnostics": _json_safe(ordered_roi_diagnostics),
            "top_worst_rois": _json_safe(ordered_roi_diagnostics[:10]),
        },
        "acceptance": acceptance_summary,
    }

    _emit_progress(
        "Final result: "
        f"success={bool(best_result.success)}, accepted={bool(best_result.acceptance_passed)}, "
        f"cost={_format_cost(final_cost)}, reduction={100.0 * float(cost_reduction):.1f}%, "
        f"rms={_format_rms(best_result.fun)}, "
        f"{_format_params(best_result.x)}"
    )
    if bound_hits:
        _emit_progress("Final bound hits: " + ", ".join(str(name) for name in bound_hits))
    for dataset_summary in list(dataset_diagnostics or []):
        label = str(dataset_summary.get("dataset_label", dataset_summary.get("dataset_index", "?")))
        worst_hkls = list(dataset_summary.get("worst_hkls", []) or [])
        worst_text = ", ".join(str(tuple(hkl)) for hkl in worst_hkls[:3]) if worst_hkls else "-"
        _emit_progress(
            f"Final dataset {label}: "
            f"theta={_format_cost(dataset_summary.get('theta_initial_deg'))} deg, "
            f"rois={int(dataset_summary.get('roi_count', 0))}, "
            f"cost={_format_cost(dataset_summary.get('cost'))}, "
            f"max_roi_rms={_format_cost(dataset_summary.get('max_roi_rms'))}, "
            f"worst_hkls={worst_text}"
        )

    best_result.parameter_bounds = parameter_debug
    best_result.initial_parameter_values = {
        str(name): float(full_x0[idx]) for idx, name in enumerate(param_names)
    }
    best_result.final_parameter_values = {
        str(name): float(best_result.x[idx]) for idx, name in enumerate(param_names)
    }
    best_result.initial_residual_count = int(initial_residual_count)
    best_result.final_residual_count = int(final_residual_count)
    best_result.initial_residual_norm = float(initial_residual_norm)
    best_result.final_residual_norm = float(final_residual_norm)
    best_result.initial_residual_rms = float(initial_residual_rms)
    best_result.final_residual_rms = float(final_residual_rms)
    best_result.input_dataset_summaries = list(input_dataset_summaries)
    best_result.prepared_dataset_summaries = list(prepared_dataset_summaries)
    best_result.rejected_roi_reason_counts = dict(rejected_roi_reason_counts)
    best_result.rejected_roi_counts_by_dataset = dict(rejected_roi_counts_by_dataset)
    best_result.geometry_parameter_summary = dict(geometry_parameter_summary)
    best_result.mosaic_sample_summary = dict(mosaic_sample_summary)
    best_result.acceptance_summary = dict(acceptance_summary)
    best_result.mosaic_fit_debug_summary = dict(mosaic_fit_debug_summary)
    return best_result


def fit_mosaic_shape_parameters(
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, object],
    *,
    dataset_specs: Sequence[Dict[str, object]],
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
    max_nfev: int = 80,
    max_restarts: int = 2,
    smooth_sigma_px: float = 1.0,
    ridge_percentile: float = 85.0,
    roi_half_width: Optional[int] = None,
    min_total_rois: int = 8,
    min_per_dataset_rois: int = 3,
    equal_dataset_weights: bool = True,
    workers: object = "auto",
    parallel_mode: str = "auto",
    worker_numba_threads: object = 0,
    restart_jitter: float = 0.15,
    ridge_weight: float = 1.0,
    specular_relative_intensity_weight: float = 0.0,
    fit_theta_i: bool = True,
    theta_i_mode: str = "auto",
    theta_i_bounds_deg: Optional[Tuple[float, float]] = None,
    fit_sigma_mosaic: bool = True,
    fit_gamma_mosaic: bool = True,
    fit_eta: bool = True,
    point_match_weight: Optional[float] = None,
    point_match_f_scale: Optional[float] = None,
    point_match_weighted_matching: Optional[bool] = None,
    point_match_missing_pair_penalty: Optional[float] = None,
    refine_theta: Optional[bool] = None,
    theta_mode: Optional[str] = None,
    theta_bounds: Optional[Tuple[float, float]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> OptimizeResult:
    """Public entry point for the profile-based mosaic fit."""

    del point_match_f_scale
    del point_match_weighted_matching
    del point_match_missing_pair_penalty

    if point_match_weight is not None:
        specular_relative_intensity_weight = float(point_match_weight)
    if refine_theta is not None:
        fit_theta_i = bool(refine_theta)
    if theta_mode is not None:
        theta_i_mode = str(theta_mode)
    if theta_bounds is not None:
        theta_i_bounds_deg = theta_bounds

    return _fit_mosaic_shape_parameters_profiles(
        miller,
        intensities,
        image_size,
        params,
        dataset_specs=dataset_specs,
        bounds=bounds,
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        max_restarts=max_restarts,
        smooth_sigma_px=smooth_sigma_px,
        ridge_percentile=ridge_percentile,
        roi_half_width=roi_half_width,
        min_total_rois=min_total_rois,
        min_per_dataset_rois=min_per_dataset_rois,
        equal_dataset_weights=equal_dataset_weights,
        workers=workers,
        parallel_mode=parallel_mode,
        worker_numba_threads=worker_numba_threads,
        restart_jitter=restart_jitter,
        ridge_weight=ridge_weight,
        specular_relative_intensity_weight=specular_relative_intensity_weight,
        fit_theta_i=fit_theta_i,
        theta_i_mode=theta_i_mode,
        theta_i_bounds_deg=theta_i_bounds_deg,
        fit_sigma_mosaic=fit_sigma_mosaic,
        fit_gamma_mosaic=fit_gamma_mosaic,
        fit_eta=fit_eta,
        progress_callback=progress_callback,
    )


def _simulation_kernel_kwargs(
    params: Dict[str, object],
    mosaic: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    mosaic_params = params.get("mosaic_params", {}) if mosaic is None else mosaic
    if not isinstance(mosaic_params, dict):
        mosaic_params = {}

    kwargs: Dict[str, object] = {
        "optics_mode": int(params.get("optics_mode", 0)),
        "solve_q_steps": int(mosaic_params.get("solve_q_steps", 1000)),
        "solve_q_rel_tol": float(mosaic_params.get("solve_q_rel_tol", 5.0e-4)),
        "solve_q_mode": int(mosaic_params.get("solve_q_mode", 1)),
        "thickness": float(params.get("sample_depth_m", params.get("thickness", 0.0))),
        "pixel_size_m": float(params.get("pixel_size_m", params.get("pixel_size", 100e-6))),
        "sample_width_m": float(params.get("sample_width_m", 0.0)),
        "sample_length_m": float(params.get("sample_length_m", 0.0)),
    }
    n2_sample_array = mosaic_params.get("n2_sample_array")
    if n2_sample_array is not None:
        kwargs["n2_sample_array_override"] = np.asarray(
            n2_sample_array,
            dtype=np.complex128,
        )
    return kwargs


def _interpolate_line(points: List[Tuple[float, float]]) -> np.ndarray:
    if len(points) == 1:
        return np.asarray(points, dtype=float)

    dense_points: List[np.ndarray] = []
    for start, end in zip(points[:-1], points[1:]):
        length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        steps = max(int(length * 2), 1)
        xs = np.linspace(start[0], end[0], steps)
        ys = np.linspace(start[1], end[1], steps)
        dense_points.append(np.column_stack((xs, ys)))
    return np.vstack(dense_points)


def build_tube_rois(
    miller: np.ndarray,
    max_positions: np.ndarray,
    params: Dict[str, float],
    image_size: int,
    *,
    measured_dict: Optional[Dict[Tuple[int, int, int], List[Tuple[float, float]]]] = None,
    base_width: Optional[float] = None,
) -> List[TubeROI]:
    """Construct tube-shaped ROIs following the manuscript guidance."""

    pixel_size = _estimate_pixel_size(params)
    mosaic = params["mosaic_params"]
    sigma_deg = float(mosaic.get("sigma_mosaic_deg", 0.3))
    mosaic_fwhm_rad = math.radians(sigma_deg) * 2.0 * math.sqrt(2.0 * math.log(2.0))
    divergence_rad = math.radians(float(mosaic.get("gamma_mosaic_deg", 0.3)))
    bandwidth_rad = float(params.get("bandwidth_rad", 0.0))
    detector_length = float(params.get("corto_detector", 1.0))

    nominal_width = mosaic_fwhm_rad * detector_length / pixel_size
    nominal_width += divergence_rad * detector_length / pixel_size
    nominal_width += bandwidth_rad * detector_length / pixel_size
    if base_width is not None:
        nominal_width = max(nominal_width, base_width)
    width_px = max(3.0, nominal_width)

    rois: List[TubeROI] = []
    measured_dict = measured_dict or {}

    for idx, reflection in enumerate(miller):
        I0, x0, y0, I1, x1, y1 = max_positions[idx]
        points: List[Tuple[float, float]] = []
        for x, y in ((x0, y0), (x1, y1)):
            if np.isfinite(x) and np.isfinite(y):
                points.append((float(x), float(y)))

        key = tuple(int(v) for v in reflection)
        for mx, my in measured_dict.get(key, []):
            points.append((float(mx), float(my)))

        if not points:
            continue

        # Sort points by polar angle around detector center to stabilise tubes
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] > 2:
            centroid = pts.mean(axis=0)
            angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
            order = np.argsort(angles)
            pts = pts[order]
        dense_centerline = _interpolate_line(list(map(tuple, pts)))

        min_x = max(int(np.floor(np.min(pts[:, 0] - width_px - 2))), 0)
        max_x = min(int(np.ceil(np.max(pts[:, 0] + width_px + 2))), image_size - 1)
        min_y = max(int(np.floor(np.min(pts[:, 1] - width_px - 2))), 0)
        max_y = min(int(np.ceil(np.max(pts[:, 1] + width_px + 2))), image_size - 1)

        if min_x >= max_x or min_y >= max_y:
            continue

        bounds = (min_x, max_x, min_y, max_y)
        grid_y, grid_x = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        query_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        tree = cKDTree(dense_centerline)
        distances, _ = tree.query(query_points, k=1)
        distance_map = distances.reshape(grid_x.shape)

        tube_mask = distance_map <= width_px
        centerline_mask = distance_map <= max(1.0, width_px * 0.25)
        off_mask = np.ones_like(tube_mask, dtype=bool)
        off_mask[tube_mask] = False

        roi = TubeROI(
            reflection=key,
            centerline=dense_centerline,
            width=float(width_px),
            bounds=bounds,
            mask=tube_mask,
            off_tube_mask=off_mask,
            identifier=idx,
        )
        roi.centerline_mask = centerline_mask
        rois.append(roi)

    return rois


def compute_sensitivity_weights(
    base_sim: np.ndarray,
    params: Dict[str, float],
    var_names: Sequence[str],
    rois: List[TubeROI],
    simulator,
    *,
    downsample_factor: int = 4,
    percentile: float = 90.0,
    huber_percentile: float = 97.0,
    per_reflection_quota: int = 200,
    off_tube_fraction: float = 0.05,
    normalize_per_roi: bool = False,
) -> None:
    """Populate each ROI with active pixels using a sensitivity-driven map."""

    base_down = _downsample_with_antialiasing(base_sim, downsample_factor)
    sensitivity = np.zeros_like(base_down, dtype=np.float64)

    for name in var_names:
        delta = max(abs(params.get(name, 1.0)) * 1e-3, 1e-3)
        shifted = dict(params)
        shifted[name] = params.get(name, 0.0) + delta
        sim_shift, _ = simulator(shifted)
        shift_down = _downsample_with_antialiasing(sim_shift, downsample_factor)
        grad = (shift_down - base_down) / delta
        sensitivity += grad * grad

    delta_s = np.sqrt(np.maximum(sensitivity, 0.0))
    safe_base = np.clip(base_down, 1e-6, None)
    importance_map = (delta_s * delta_s) / safe_base

    clip_level = np.percentile(importance_map, huber_percentile)
    if np.isfinite(clip_level) and clip_level > 0:
        importance_map = np.where(
            importance_map <= clip_level,
            importance_map,
            clip_level + 0.1 * (importance_map - clip_level),
        )

    upsample_factor = downsample_factor
    upsampled = zoom(importance_map, upsample_factor, order=1, prefilter=False)
    if upsampled.shape != base_sim.shape:
        pad_y = base_sim.shape[0] - upsampled.shape[0]
        pad_x = base_sim.shape[1] - upsampled.shape[1]
        upsampled = np.pad(
            upsampled,
            ((0, max(pad_y, 0)), (0, max(pad_x, 0))),
            mode="edge",
        )[: base_sim.shape[0], : base_sim.shape[1]]

    for roi in rois:
        min_x, max_x, min_y, max_y = roi.bounds
        weights_map = upsampled[min_y : max_y + 1, min_x : max_x + 1]
        weights_map = np.where(roi.mask, weights_map, 0.0)
        if not np.any(weights_map):
            weights_map = np.where(roi.mask, 1.0, 0.0)

        values = weights_map[roi.mask]
        if values.size == 0:
            continue

        threshold = np.percentile(values, percentile)
        active_mask = np.zeros_like(roi.mask, dtype=bool)
        active_mask[roi.mask] = weights_map[roi.mask] >= threshold

        # Ensure at least a quota of pixels per reflection
        if active_mask.sum() < per_reflection_quota:
            flat_weights = weights_map[roi.mask]
            order = np.argsort(flat_weights)[::-1]
            top = order[: min(per_reflection_quota, order.size)]
            template = np.zeros_like(flat_weights, dtype=bool)
            template[top] = True
            active_mask = np.zeros_like(roi.mask, dtype=bool)
            active_mask[roi.mask] = template

        # Add low-weight off-tube pixels
        off_candidates = np.argwhere(roi.off_tube_mask)
        if off_candidates.size:
            count = max(1, int(off_tube_fraction * off_candidates.shape[0]))
            chosen = RNG.choice(off_candidates.shape[0], size=count, replace=False)
            off_points = off_candidates[chosen]
            active_mask[off_points[:, 0], off_points[:, 1]] = True

        ys, xs = np.nonzero(active_mask)
        global_y = ys + min_y
        global_x = xs + min_x
        roi_weights = np.asarray(weights_map[ys, xs], dtype=np.float64)
        if normalize_per_roi and roi_weights.size:
            positive_weight_sum = float(np.sum(roi_weights[roi_weights > 0.0]))
            if np.isfinite(positive_weight_sum) and positive_weight_sum > 0.0:
                roi_weights = roi_weights / positive_weight_sum
            else:
                roi_weights = np.full(
                    roi_weights.shape,
                    1.0 / max(int(roi_weights.size), 1),
                    dtype=np.float64,
                )

        roi.active_pixels = np.stack((global_y, global_x), axis=1)
        roi.weights = roi_weights
        roi.full_weight_map = weights_map
        roi.active_mask = active_mask

        sampling_prob = roi.weights.size / max(1, roi.mask.sum())
        roi.sampling_probability = max(float(sampling_prob), 1e-3)


def sample_tiles(
    rois: Sequence[TubeROI],
    *,
    temperature: float,
    explore_fraction: float,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, float]]:
    """Sample weighted tiles from each ROI and return selected pixels."""

    selected: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}

    for roi in rois:
        if roi.active_pixels is None or roi.weights is None or roi.weights.size == 0:
            continue

        min_x, max_x, min_y, max_y = roi.bounds
        width = max_x - min_x + 1
        tiles_x = (width + roi.tile_size - 1) // roi.tile_size

        rel_y = roi.active_pixels[:, 0] - min_y
        rel_x = roi.active_pixels[:, 1] - min_x
        tile_y = rel_y // roi.tile_size
        tile_x = rel_x // roi.tile_size
        tile_ids = tile_y * tiles_x + tile_x
        tile_count = int(tile_ids.max()) + 1

        energies = np.bincount(tile_ids, weights=roi.weights, minlength=tile_count)
        if not np.any(energies):
            energies = np.ones_like(energies)

        scaled = energies.astype(np.float64)
        temperature = max(float(temperature), 1e-3)
        if temperature != 1.0:
            scaled = scaled ** (1.0 / temperature)
        probs = scaled / scaled.sum()

        base_tiles = max(1, int(round(roi.sampling_probability * tile_count)))
        base_tiles = min(base_tiles, tile_count)
        explore_tiles = max(1, int(round(explore_fraction * tile_count)))
        explore_tiles = min(explore_tiles, tile_count)

        base_selection = set(RNG.choice(tile_count, size=base_tiles, replace=False, p=probs))
        explore_selection = set(RNG.choice(tile_count, size=explore_tiles, replace=False))
        chosen_tiles = base_selection.union(explore_selection)

        mask = np.isin(tile_ids, list(chosen_tiles))
        pixels = roi.active_pixels[mask]
        weights = roi.weights[mask]
        if pixels.size == 0:
            continue

        sampling_prob = max(float(len(chosen_tiles)) / max(tile_count, 1), roi.sampling_probability)
        selected[roi.identifier] = (pixels, weights, sampling_prob)
        roi.tile_probabilities = probs

    return selected


def fit_local_background(
    experimental_image: np.ndarray,
    rois: Sequence[TubeROI],
    *,
    anchor_fraction: float = 0.01,
) -> np.ndarray:
    """Fit a smooth background inside ROI bounding boxes."""

    background = np.zeros_like(experimental_image, dtype=np.float64)
    coverage = np.zeros_like(experimental_image, dtype=np.float64)
    global_mask = np.zeros_like(experimental_image, dtype=bool)

    for roi in rois:
        min_x, max_x, min_y, max_y = roi.bounds
        patch = experimental_image[min_y : max_y + 1, min_x : max_x + 1]
        mask = roi.mask
        center_mask = (
            roi.centerline_mask
            if roi.centerline_mask is not None
            else np.zeros_like(mask, dtype=bool)
        )
        sample_mask = mask & ~center_mask
        if not np.any(sample_mask):
            sample_mask = mask

        ys, xs = np.nonzero(sample_mask)
        if ys.size < 6:
            ys, xs = np.nonzero(mask)

        if ys.size == 0:
            continue

        xx = xs.astype(np.float64)
        yy = ys.astype(np.float64)
        A = np.column_stack(
            [
                np.ones_like(xx),
                xx,
                yy,
                xx * xx,
                xx * yy,
                yy * yy,
            ]
        )
        b = patch[ys, xs]
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)

        grid_x, grid_y = np.meshgrid(
            np.arange(patch.shape[1], dtype=np.float64),
            np.arange(patch.shape[0], dtype=np.float64),
            indexing="xy",
        )
        fitted = (
            coeffs[0]
            + coeffs[1] * grid_x
            + coeffs[2] * grid_y
            + coeffs[3] * grid_x * grid_x
            + coeffs[4] * grid_x * grid_y
            + coeffs[5] * grid_y * grid_y
        )

        if roi.centerline_mask is not None:
            fitted = np.where(roi.centerline_mask, np.minimum(fitted, patch), fitted)

        background[min_y : max_y + 1, min_x : max_x + 1] += fitted
        coverage[min_y : max_y + 1, min_x : max_x + 1] += 1.0
        global_mask[min_y : max_y + 1, min_x : max_x + 1] |= mask

    mask = coverage > 0
    if np.any(mask):
        background[mask] /= np.maximum(coverage[mask], 1.0)

    smooth = gaussian_filter(background, sigma=3.0, mode="reflect")
    background[mask] = 0.7 * background[mask] + 0.3 * smooth[mask]

    outside_mask = ~global_mask
    anchors = np.argwhere(outside_mask)
    if anchors.size:
        count = max(1, int(anchor_fraction * anchors.shape[0]))
        chosen = RNG.choice(anchors.shape[0], size=count, replace=False)
        anchor_points = anchors[chosen]
        global_background = gaussian_filter(experimental_image, sigma=30.0, mode="reflect")
        background[anchor_points[:, 0], anchor_points[:, 1]] = global_background[
            anchor_points[:, 0], anchor_points[:, 1]
        ]

    return background


def _poisson_deviance(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    obs = np.clip(obs, 1e-9, None)
    pred = np.clip(pred, 1e-9, None)
    return 2.0 * (pred - obs + obs * np.log(obs / pred))


def _anscombe(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(np.clip(x, 0.0, None) + 3.0 / 8.0)


def _huber(residual: np.ndarray, delta: float) -> np.ndarray:
    delta = max(delta, 1e-6)
    abs_res = np.abs(residual)
    return np.where(abs_res <= delta, residual, delta * np.sign(residual))


def robust_residuals(
    obs: np.ndarray,
    pred: np.ndarray,
    weights: np.ndarray,
    sampling_probability: float,
    *,
    huber_delta: float,
    mixture: float = 0.1,
) -> np.ndarray:
    dev = _poisson_deviance(obs, pred)
    ans_res = _anscombe(obs) - _anscombe(pred)
    huber_res = _huber(ans_res, huber_delta)
    mixture = np.clip(mixture, 0.0, 1.0)
    mixture_weight = (1.0 - mixture) + mixture * np.exp(
        -0.5 * (ans_res / (huber_delta + 1e-6)) ** 2
    )
    combined = 0.5 * np.sqrt(np.abs(dev)) + 0.5 * huber_res
    scaling = np.sqrt(np.maximum(weights, 1e-8)) / max(sampling_probability, 1e-3)
    return combined * scaling * np.sqrt(mixture_weight)


def _select_active_reflections(
    rois: Sequence[TubeROI],
    *,
    max_reflections: int,
    random_fraction: float,
) -> List[TubeROI]:
    scored = [(float(np.sum(roi.weights)) if roi.weights is not None else 0.0, roi) for roi in rois]
    scored.sort(key=lambda t: t[0], reverse=True)

    selected = [roi for _, roi in scored[:max_reflections]]
    remainder = [roi for _, roi in scored[max_reflections:]]
    if remainder and random_fraction > 0:
        count = max(1, int(round(random_fraction * len(remainder))))
        count = min(count, len(remainder))
        selected.extend(RNG.choice(remainder, size=count, replace=False))
    return selected


def _centerline_shift(
    old: Sequence[Tuple[float, float]],
    new: Sequence[Tuple[float, float]],
) -> float:
    if not len(old) or not len(new):
        return np.inf
    old_arr = np.asarray(old)
    new_arr = np.asarray(new)
    tree = cKDTree(old_arr)
    distances, _ = tree.query(new_arr, k=1)
    return float(np.max(distances))


def _refresh_rois_if_needed(
    rois: List[TubeROI],
    miller: np.ndarray,
    max_positions: np.ndarray,
    params: Dict[str, float],
    image_size: int,
    *,
    measured_dict: Optional[Dict[Tuple[int, int, int], List[Tuple[float, float]]]] = None,
    threshold: float = 1.0,
) -> List[TubeROI]:
    updated: List[TubeROI] = []
    for roi in rois:
        idx = roi.identifier
        if idx >= len(miller):
            continue
        I0, x0, y0, I1, x1, y1 = max_positions[idx]
        points = []
        for x, y in ((x0, y0), (x1, y1)):
            if np.isfinite(x) and np.isfinite(y):
                points.append((float(x), float(y)))
        shift = _centerline_shift(roi.centerline, points)
        if shift <= threshold:
            updated.append(roi)
        else:
            # rebuild ROI when shift significant
            rebuilt = build_tube_rois(
                miller[idx : idx + 1],
                max_positions[idx : idx + 1],
                params,
                image_size,
                measured_dict=measured_dict,
                base_width=roi.width,
            )
            if rebuilt:
                new_roi = rebuilt[0]
                new_roi.identifier = roi.identifier
                updated.append(new_roi)
            else:
                updated.append(roi)
    return updated


def _stage_one_initialize(
    experimental_image: np.ndarray,
    params: Dict[str, float],
    var_names: Sequence[str],
    simulator,
    *,
    downsample_factor: int,
    max_nfev: int,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    x_scale: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, float], OptimizeResult]:
    """Stage 1 coarse alignment using Chamfer distance on ridge maps."""

    downsample_factor = max(int(downsample_factor), 1)
    exp_down = _downsample_with_antialiasing(experimental_image, downsample_factor)
    ridge_exp = _compute_ridge_map(exp_down)
    distance_exp = distance_transform_edt(~ridge_exp)

    x0 = np.array([params[name] for name in var_names], dtype=float)

    def residual(x):
        trial = _update_params(params, var_names, x)
        sim_img, _ = simulator(trial)
        sim_down = _downsample_with_antialiasing(sim_img, downsample_factor)
        ridge_sim = _compute_ridge_map(sim_down)
        distance_sim = distance_transform_edt(~ridge_sim)
        fwd = distance_sim[ridge_exp]
        back = distance_exp[ridge_sim]
        return np.concatenate((fwd.ravel(), back.ravel()))

    initial_residual = np.asarray(residual(x0), dtype=np.float64)
    lsq_kwargs: Dict[str, object] = {
        "max_nfev": int(max_nfev),
    }
    if bounds is not None:
        lsq_kwargs["bounds"] = bounds
    if x_scale is not None:
        lsq_kwargs["x_scale"] = np.asarray(x_scale, dtype=np.float64)
    result = least_squares(residual, x0, **lsq_kwargs)
    result.initial_cost = 0.5 * float(np.sum(initial_residual * initial_residual))
    result.final_cost = 0.5 * float(np.sum(np.asarray(result.fun, dtype=np.float64) ** 2))
    result.cost_reduction = (
        float((result.initial_cost - result.final_cost) / result.initial_cost)
        if np.isfinite(float(result.initial_cost)) and float(result.initial_cost) > 1.0e-12
        else 0.0
    )
    updated_params = _update_params(params, var_names, result.x)
    return updated_params, result


def _stage_two_refinement(
    experimental_image: np.ndarray,
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, float],
    var_names: Sequence[str],
    simulator,
    measured_dict: Dict[Tuple[int, int, int], List[Tuple[float, float]]],
    *,
    cfg: Dict[str, float],
) -> Tuple[
    Dict[str, float], OptimizeResult, List[TubeROI], np.ndarray, Callable[[np.ndarray], np.ndarray]
]:
    """Stage 2 refinement using Poisson deviance on active pixels."""

    base_sim, maxpos = simulator(params)
    rois = build_tube_rois(
        miller,
        maxpos,
        params,
        image_size,
        measured_dict=measured_dict,
    )

    compute_sensitivity_weights(
        base_sim,
        params,
        var_names,
        rois,
        simulator,
        downsample_factor=int(cfg.get("downsample_factor", 4)),
        percentile=float(cfg.get("percentile", 90.0)),
        huber_percentile=float(cfg.get("huber_percentile", 97.0)),
        per_reflection_quota=int(cfg.get("per_reflection_quota", 200)),
        off_tube_fraction=float(cfg.get("off_tube_fraction", 0.05)),
        normalize_per_roi=bool(cfg.get("equal_peak_weights", False)),
    )

    rois_state = rois

    def residual(x):
        nonlocal rois_state
        trial = _update_params(params, var_names, x)
        sim_img, trial_maxpos = simulator(trial)
        rois_state = _refresh_rois_if_needed(
            rois_state,
            miller,
            trial_maxpos,
            trial,
            image_size,
            measured_dict=measured_dict,
            threshold=float(cfg.get("roi_refresh_threshold", 1.0)),
        )

        missing = [roi for roi in rois_state if roi.weights is None]
        if missing:
            compute_sensitivity_weights(
                sim_img,
                trial,
                var_names,
                rois_state,
                simulator,
                downsample_factor=int(cfg.get("downsample_factor", 4)),
                percentile=float(cfg.get("percentile", 90.0)),
                huber_percentile=float(cfg.get("huber_percentile", 97.0)),
                per_reflection_quota=int(cfg.get("per_reflection_quota", 200)),
                off_tube_fraction=float(cfg.get("off_tube_fraction", 0.05)),
                normalize_per_roi=bool(cfg.get("equal_peak_weights", False)),
            )

        max_reflections = (
            len(rois_state)
            if bool(cfg.get("equal_peak_weights", False))
            else int(cfg.get("max_reflections", 12))
        )
        active = _select_active_reflections(
            rois_state,
            max_reflections=max(1, int(max_reflections)),
            random_fraction=float(cfg.get("random_reflection_fraction", 0.15)),
        )

        background = fit_local_background(experimental_image, rois_state)
        sampled = sample_tiles(
            active,
            temperature=float(cfg.get("sampling_temperature", 1.0)),
            explore_fraction=float(cfg.get("explore_fraction", 0.15)),
        )

        residuals_list: List[np.ndarray] = []
        for roi in active:
            selection = sampled.get(roi.identifier)
            if selection is None:
                continue
            pixels, weights, sampling_prob = selection
            obs = experimental_image[pixels[:, 0], pixels[:, 1]]
            model = sim_img[pixels[:, 0], pixels[:, 1]] + background[pixels[:, 0], pixels[:, 1]]
            res = robust_residuals(
                obs,
                model,
                weights,
                sampling_prob,
                huber_delta=float(cfg.get("huber_delta", 2.5)),
                mixture=float(cfg.get("outlier_mixture", 0.1)),
            )
            residuals_list.append(res)

        if not residuals_list:
            return np.zeros(1, dtype=float)
        return np.concatenate(residuals_list)

    x0 = np.array([params[name] for name in var_names], dtype=float)
    initial_residual = np.asarray(residual(x0), dtype=np.float64)
    initial_cost = 0.5 * float(np.sum(initial_residual * initial_residual))
    result = least_squares(residual, x0, max_nfev=int(cfg.get("max_nfev", 25)))
    updated_params = _update_params(params, var_names, result.x)
    final_sim, _ = simulator(updated_params)
    final_residual = np.asarray(result.fun, dtype=np.float64)
    final_cost = 0.5 * float(np.sum(final_residual * final_residual))
    rois_state = _refresh_rois_if_needed(
        rois_state,
        miller,
        maxpos,
        updated_params,
        image_size,
        measured_dict=measured_dict,
        threshold=float(cfg.get("roi_refresh_threshold", 1.0)),
    )
    result.initial_cost = float(initial_cost)
    result.final_cost = float(final_cost)
    result.cost_reduction = (
        float((initial_cost - final_cost) / initial_cost)
        if np.isfinite(initial_cost) and initial_cost > 1.0e-12
        else 0.0
    )
    result.selected_rois = list(rois_state)
    return updated_params, result, rois_state, final_sim, residual


def _stage_three_refinement(
    experimental_image: np.ndarray,
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, float],
    var_names: Sequence[str],
    simulator,
    rois: List[TubeROI],
    measured_dict: Dict[Tuple[int, int, int], List[Tuple[float, float]]],
    *,
    cfg: Dict[str, float],
) -> Tuple[Dict[str, float], OptimizeResult, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Final stage at native resolution with expanded active sets."""

    compute_sensitivity_weights(
        simulator(params)[0],
        params,
        var_names,
        rois,
        simulator,
        downsample_factor=int(cfg.get("downsample_factor", 1)),
        percentile=float(cfg.get("percentile", 85.0)),
        huber_percentile=float(cfg.get("huber_percentile", 98.0)),
        per_reflection_quota=int(cfg.get("per_reflection_quota", 400)),
        off_tube_fraction=float(cfg.get("off_tube_fraction", 0.1)),
        normalize_per_roi=bool(cfg.get("equal_peak_weights", False)),
    )

    rois_state = rois

    def residual(x):
        nonlocal rois_state
        trial = _update_params(params, var_names, x)
        sim_img, trial_maxpos = simulator(trial)
        rois_state = _refresh_rois_if_needed(
            rois_state,
            miller,
            trial_maxpos,
            trial,
            image_size,
            measured_dict=measured_dict,
            threshold=float(cfg.get("roi_refresh_threshold", 0.5)),
        )

        compute_sensitivity_weights(
            sim_img,
            trial,
            var_names,
            rois_state,
            simulator,
            downsample_factor=int(cfg.get("downsample_factor", 1)),
            percentile=float(cfg.get("percentile", 85.0)),
            huber_percentile=float(cfg.get("huber_percentile", 98.0)),
            per_reflection_quota=int(cfg.get("per_reflection_quota", 400)),
            off_tube_fraction=float(cfg.get("off_tube_fraction", 0.1)),
            normalize_per_roi=bool(cfg.get("equal_peak_weights", False)),
        )

        max_reflections = (
            len(rois_state)
            if bool(cfg.get("equal_peak_weights", False))
            else int(cfg.get("max_reflections", len(rois_state)))
        )
        active = _select_active_reflections(
            rois_state,
            max_reflections=max(1, int(max_reflections)),
            random_fraction=float(cfg.get("random_reflection_fraction", 0.1)),
        )

        background = fit_local_background(experimental_image, rois_state)
        sampled = sample_tiles(
            active,
            temperature=float(cfg.get("sampling_temperature", 0.7)),
            explore_fraction=float(cfg.get("explore_fraction", 0.1)),
        )

        residuals_list: List[np.ndarray] = []
        for roi in active:
            selection = sampled.get(roi.identifier)
            if selection is None:
                continue
            pixels, weights, sampling_prob = selection
            obs = experimental_image[pixels[:, 0], pixels[:, 1]]
            model = sim_img[pixels[:, 0], pixels[:, 1]] + background[pixels[:, 0], pixels[:, 1]]
            res = robust_residuals(
                obs,
                model,
                weights,
                sampling_prob,
                huber_delta=float(cfg.get("huber_delta", 2.0)),
                mixture=float(cfg.get("outlier_mixture", 0.05)),
            )
            residuals_list.append(res)

        if not residuals_list:
            return np.zeros(1, dtype=float)
        return np.concatenate(residuals_list)

    x0 = np.array([params[name] for name in var_names], dtype=float)
    result = least_squares(residual, x0, max_nfev=int(cfg.get("max_nfev", 35)))
    updated_params = _update_params(params, var_names, result.x)
    final_sim, _ = simulator(updated_params)
    return updated_params, result, final_sim, residual


def iterative_refinement(
    experimental_image: np.ndarray,
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, float],
    *,
    var_names: Optional[Sequence[str]] = None,
    measured_peaks: Optional[Sequence[Dict[str, float]]] = None,
    config: Optional[Dict[str, Dict[str, float]]] = None,
) -> IterativeRefinementResult:
    """Run the multi-stage refinement described in the manuscript."""

    if var_names is None:
        var_names = ("zb", "zs", "theta_initial", "chi")
    var_names = list(var_names)

    measured_dict = build_measured_dict(measured_peaks or [])

    cache_keys = [
        "gamma",
        "Gamma",
        "corto_detector",
        "theta_initial",
        "zs",
        "zb",
        "chi",
        "a",
        "c",
        "center",
    ]
    cache = SimulationCache(cache_keys)

    def simulator(local_params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        merged = dict(params)
        merged.update(local_params)
        return _simulate_with_cache(merged, miller, intensities, image_size, cache)

    config = config or {}
    stage1_cfg = {"downsample_factor": 8, "max_nfev": 20}
    stage1_cfg.update(config.get("stage1", {}))
    stage2_cfg = {
        "downsample_factor": 4,
        "percentile": 90.0,
        "huber_percentile": 97.0,
        "per_reflection_quota": 200,
        "off_tube_fraction": 0.05,
        "max_reflections": 12,
        "random_reflection_fraction": 0.15,
        "sampling_temperature": 1.0,
        "explore_fraction": 0.15,
        "huber_delta": 2.5,
        "outlier_mixture": 0.1,
        "max_nfev": 25,
        "roi_refresh_threshold": 1.0,
    }
    stage2_cfg.update(config.get("stage2", {}))
    stage3_cfg = {
        "downsample_factor": 1,
        "percentile": 85.0,
        "huber_percentile": 98.0,
        "per_reflection_quota": 400,
        "off_tube_fraction": 0.1,
        "max_reflections": len(miller),
        "random_reflection_fraction": 0.1,
        "sampling_temperature": 0.7,
        "explore_fraction": 0.1,
        "huber_delta": 2.0,
        "outlier_mixture": 0.05,
        "max_nfev": 35,
        "roi_refresh_threshold": 0.5,
    }
    stage3_cfg.update(config.get("stage3", {}))

    history: List[Dict[str, float]] = []
    stage_summaries: List[Dict[str, float]] = []
    current_params = dict(params)

    current_params, stage1_result = _stage_one_initialize(
        experimental_image,
        current_params,
        var_names,
        simulator,
        downsample_factor=stage1_cfg["downsample_factor"],
        max_nfev=stage1_cfg["max_nfev"],
    )
    history.append({name: current_params[name] for name in var_names})
    stage_summaries.append(
        {
            "stage": "level1",
            "cost": float(np.sum(stage1_result.fun**2)),
            "nfev": stage1_result.nfev,
        }
    )

    current_params, stage2_result, rois, stage2_sim, stage2_residual = _stage_two_refinement(
        experimental_image,
        miller,
        intensities,
        image_size,
        current_params,
        var_names,
        simulator,
        measured_dict,
        cfg=stage2_cfg,
    )
    history.append({name: current_params[name] for name in var_names})
    stage_summaries.append(
        {
            "stage": "level2",
            "cost": float(np.sum(stage2_result.fun**2)),
            "nfev": stage2_result.nfev,
        }
    )

    current_params, stage3_result, stage3_sim, stage3_residual = _stage_three_refinement(
        experimental_image,
        miller,
        intensities,
        image_size,
        current_params,
        var_names,
        simulator,
        rois,
        measured_dict,
        cfg=stage3_cfg,
    )
    history.append({name: current_params[name] for name in var_names})
    stage_summaries.append(
        {
            "stage": "level3",
            "cost": float(np.sum(stage3_result.fun**2)),
            "nfev": stage3_result.nfev,
        }
    )

    final_residual = stage3_residual(np.array([current_params[name] for name in var_names]))

    return IterativeRefinementResult(
        x=np.array([current_params[name] for name in var_names], dtype=float),
        fun=final_residual,
        success=bool(stage1_result.success and stage2_result.success and stage3_result.success),
        message="; ".join(
            filter(None, [stage1_result.message, stage2_result.message, stage3_result.message])
        ),
        best_params=dict(current_params),
        history=history,
        stage_summaries=stage_summaries,
    )


def build_measured_dict(
    measured_peaks: Optional[Sequence[object]],
) -> Dict[Tuple[int, int, int], List[Tuple[float, float]]]:
    """Return measured peaks grouped by HKL, skipping malformed entries."""

    measured_dict: Dict[Tuple[int, int, int], List[Tuple[float, float]]] = {}
    for entry in _normalize_measured_peaks(measured_peaks):
        raw_hkl = entry.get("hkl")
        if not isinstance(raw_hkl, tuple) or len(raw_hkl) != 3:
            continue
        try:
            hkl_key = (
                int(raw_hkl[0]),
                int(raw_hkl[1]),
                int(raw_hkl[2]),
            )
            x = float(entry.get("x"))
            y = float(entry.get("y"))
        except Exception:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        measured_dict.setdefault(hkl_key, []).append((x, y))
    return measured_dict


def _valid_hit_rows(hit_table: object) -> List[np.ndarray]:
    """Return filtered hit rows using the same rules as geometry auto-match."""

    hits = np.asarray(hit_table)
    if hits.ndim != 2 or hits.shape[1] < 7:
        return []

    rows: List[np.ndarray] = []
    for row in hits:
        if (
            not np.isfinite(row[0])
            or not np.isfinite(row[1])
            or not np.isfinite(row[2])
            or not np.isfinite(row[4])
            or not np.isfinite(row[5])
            or not np.isfinite(row[6])
        ):
            continue
        rows.append(np.asarray(row[:7], dtype=float))
    return rows


def _nonnegative_index(value: object) -> Optional[int]:
    try:
        idx = int(value)
    except Exception:
        return None
    return int(idx) if idx >= 0 else None


def _measured_source_table_index(entry: Mapping[str, object]) -> Optional[int]:
    table_idx = _nonnegative_index(entry.get("resolved_table_index"))
    if table_idx is not None:
        return int(table_idx)
    if _entry_trusted_full_reflection_identity(entry):
        return None
    return _nonnegative_index(entry.get("source_table_index"))


def _entry_trusted_full_reflection_identity(entry: Mapping[str, object]) -> bool:
    if not isinstance(entry, Mapping):
        return False
    reflection_idx = _nonnegative_index(entry.get("source_reflection_index"))
    if reflection_idx is None:
        return False
    namespace = str(entry.get("source_reflection_namespace", "") or "").strip().lower()
    if namespace in {"full", "full_reflection", "miller"}:
        return True
    return bool(entry.get("source_reflection_is_full", False))


def _diagnostic_source_table_index(entry: Mapping[str, object]) -> Optional[int]:
    if not isinstance(entry, Mapping):
        return None
    if _entry_trusted_full_reflection_identity(entry):
        return None
    if str(entry.get("resolution_kind", "")).strip().lower() != "fixed_source":
        return None
    return _nonnegative_index(entry.get("source_table_index"))


def _local_table_signature(original_indices: object) -> Optional[str]:
    try:
        indices_arr = np.asarray(original_indices, dtype=np.int64).reshape(-1)
    except Exception:
        return None
    try:
        digest = hashlib.sha1(
            np.ascontiguousarray(indices_arr, dtype=np.int64).tobytes()
        ).hexdigest()
    except Exception:
        return None
    return f"n{int(indices_arr.size)}:{digest}"


def _measured_source_peak_index_with_source(
    entry: Mapping[str, object],
    *,
    allow_legacy_peak_fallback: bool = False,
) -> Tuple[Optional[int], Optional[str]]:
    peak_idx, peak_source, _peak_reason = resolve_canonical_branch(
        entry,
        allow_legacy_peak_fallback=allow_legacy_peak_fallback,
    )
    if peak_idx in {0, 1}:
        return int(peak_idx), str(peak_source or "source_branch_index")
    peak_idx = _nonnegative_index(entry.get("resolved_peak_index"))
    if peak_idx in {0, 1}:
        return int(peak_idx), "resolved_peak_index"
    if allow_legacy_peak_fallback:
        peak_idx = _nonnegative_index(entry.get("source_peak_index"))
        if peak_idx in {0, 1}:
            return int(peak_idx), "source_peak_index"
    return None, None


def _measured_source_peak_index(
    entry: Mapping[str, object],
    *,
    allow_legacy_peak_fallback: bool = False,
) -> Optional[int]:
    peak_idx, _peak_source = _measured_source_peak_index_with_source(
        entry,
        allow_legacy_peak_fallback=allow_legacy_peak_fallback,
    )
    return peak_idx


def _branch_representative_row(
    rows: Sequence[np.ndarray],
    *,
    branch_index: int,
) -> Tuple[Optional[np.ndarray], str]:
    if branch_index not in {0, 1}:
        return None, "missing_source_peak_index"
    best_row: Optional[np.ndarray] = None
    best_intensity = -math.inf
    for raw_row in rows:
        try:
            row = np.asarray(raw_row, dtype=float).reshape(-1)
        except Exception:
            continue
        if row.shape[0] < 7:
            continue
        resolved_branch = source_branch_index_from_phi_deg(row[3])
        if resolved_branch is None:
            continue
        if int(resolved_branch) != int(branch_index):
            continue
        try:
            intensity = abs(float(row[0]))
        except Exception:
            intensity = 0.0
        if not np.isfinite(intensity):
            intensity = 0.0
        if best_row is None or float(intensity) > float(best_intensity):
            best_row = np.asarray(row[:7], dtype=float)
            best_intensity = float(intensity)
    if best_row is None:
        return None, "missing_source_peak"
    return best_row, "resolved_source_peak"


def _resolve_max_position_peak(
    max_positions: np.ndarray,
    *,
    table_idx: int,
    peak_index: int,
) -> Tuple[Optional[Tuple[float, float]], str]:
    if table_idx < 0 or table_idx >= int(max_positions.shape[0]):
        return None, "source_table_out_of_range"
    if peak_index not in {0, 1}:
        return None, "missing_source_peak_index"
    if int(peak_index) == 0:
        if not np.isfinite(max_positions[table_idx, 0]) or float(max_positions[table_idx, 0]) <= 0.0:
            return None, "missing_source_peak"
        sim_col = float(max_positions[table_idx, 1])
        sim_row = float(max_positions[table_idx, 2])
    else:
        if not np.isfinite(max_positions[table_idx, 3]) or float(max_positions[table_idx, 3]) <= 0.0:
            return None, "missing_source_peak"
        sim_col = float(max_positions[table_idx, 4])
        sim_row = float(max_positions[table_idx, 5])
    if not (np.isfinite(sim_col) and np.isfinite(sim_row)):
        return None, "invalid_source_peak_point"
    return (float(sim_col), float(sim_row)), "resolved_source_peak"


def _measured_source_indices(
    entry: Dict[str, object],
) -> Optional[Tuple[int, int]]:
    """Return the current hit-table row identity for a measured entry when present."""

    table_idx = _measured_source_table_index(entry)
    row_idx = _nonnegative_index(entry.get("source_row_index"))
    if table_idx is None or row_idx is None:
        return None
    return int(table_idx), int(row_idx)


def _measured_source_peak_indices(
    entry: Dict[str, object],
) -> Optional[Tuple[int, int]]:
    """Return the current hit-table branch identity for a measured entry when present."""

    table_idx = _measured_source_table_index(entry)
    peak_idx = _measured_source_peak_index(entry)
    if table_idx is None or peak_idx is None:
        return None
    return int(table_idx), int(peak_idx)


def _resolve_fixed_source_matches(
    measured_entries: Sequence[Dict[str, object]],
    hit_tables: Sequence[object],
) -> Tuple[
    List[Tuple[Dict[str, object], Tuple[float, float], Tuple[int, int, int]]],
    List[Dict[str, object]],
    Dict[int, Dict[str, object]],
]:
    """Resolve measured peaks back to their original simulated hit-table rows."""

    filtered_rows_cache: Dict[int, List[np.ndarray]] = {}
    resolved: List[Tuple[Dict[str, object], Tuple[float, float], Tuple[int, int, int]]] = []
    fallback_entries: List[Dict[str, object]] = []
    resolution_lookup: Dict[int, Dict[str, object]] = {}
    used_source_keys: set[Tuple[object, ...]] = set()
    try:
        max_positions = np.asarray(hit_tables_to_max_positions(hit_tables), dtype=float)
    except Exception:
        max_positions = np.empty((0, 6), dtype=np.float64)

    def _overlay_index(entry: Dict[str, object], fallback: int) -> int:
        try:
            out = int(entry.get("overlay_match_index", fallback))
        except Exception:
            out = int(fallback)
        if out < 0:
            return int(fallback)
        return int(out)

    for match_input_index, entry in enumerate(measured_entries):
        overlay_match_index = _overlay_index(entry, match_input_index)
        base_diag = {
            "match_input_index": int(match_input_index),
            "overlay_match_index": int(overlay_match_index),
            "label": str(entry.get("label", "")),
            "hkl": tuple(entry.get("hkl", ()))
            if isinstance(entry.get("hkl"), tuple)
            else entry.get("hkl"),
            "q_group_key": entry.get("q_group_key"),
            "source_table_index": entry.get("source_table_index"),
            "source_reflection_index": entry.get("source_reflection_index"),
            "source_row_index": entry.get("source_row_index"),
            "source_branch_index": entry.get("source_branch_index"),
            "source_peak_index": entry.get("source_peak_index"),
            "resolved_table_index": entry.get("resolved_table_index"),
            "resolved_peak_index": entry.get("resolved_peak_index"),
            "fit_source_resolution_kind": entry.get("fit_source_resolution_kind"),
        }
        trusted_full_reflection = _entry_trusted_full_reflection_identity(entry)
        source_peak_key = _measured_source_peak_indices(entry)
        source_row_key = _measured_source_indices(entry)
        source_key = (
            ("peak", int(source_peak_key[0]), int(source_peak_key[1]))
            if source_peak_key is not None
            else (
                ("row", int(source_row_key[0]), int(source_row_key[1]))
                if source_row_key is not None and not trusted_full_reflection
                else None
            )
        )
        if source_key is None:
            if trusted_full_reflection:
                resolution_lookup[id(entry)] = {
                    **base_diag,
                    "resolution_kind": "fixed_source",
                    "resolution_reason": "missing_source_peak_index",
                }
                continue
            fallback_entries.append(entry)
            resolution_lookup[id(entry)] = {
                **base_diag,
                "resolution_kind": "hkl_fallback",
                "resolution_reason": "missing_source_indices",
            }
            continue
        if source_key in used_source_keys:
            fallback_entries.append(entry)
            resolution_lookup[id(entry)] = {
                **base_diag,
                "resolution_kind": (
                    "fixed_source" if trusted_full_reflection else "hkl_fallback"
                ),
                "resolution_reason": "duplicate_source_key",
            }
            if trusted_full_reflection:
                continue
            continue

        table_idx = int(source_key[1])
        if table_idx not in filtered_rows_cache:
            if table_idx < 0 or table_idx >= len(hit_tables):
                filtered_rows_cache[table_idx] = []
            else:
                filtered_rows_cache[table_idx] = _valid_hit_rows(hit_tables[table_idx])

        rows = filtered_rows_cache.get(table_idx, [])
        resolved_from_peak = False
        sim_col = float("nan")
        sim_row = float("nan")
        sim_hkl: Tuple[int, int, int] | None = None
        if source_peak_key is not None:
            peak_idx = int(source_peak_key[1])
            peak_row, peak_reason = _branch_representative_row(
                rows,
                branch_index=int(peak_idx),
            )
            if peak_row is not None:
                sim_col = float(peak_row[1])
                sim_row = float(peak_row[2])
                resolved_from_peak = True
                try:
                    sim_hkl = (
                        int(round(float(peak_row[4]))),
                        int(round(float(peak_row[5]))),
                        int(round(float(peak_row[6]))),
                    )
                except Exception:
                    sim_hkl = None
            else:
                legacy_peak_point = None
                legacy_peak_reason = str(peak_reason)
                if not trusted_full_reflection:
                    legacy_peak_point, legacy_peak_reason = _resolve_max_position_peak(
                        max_positions,
                        table_idx=int(table_idx),
                        peak_index=int(peak_idx),
                    )
                if legacy_peak_point is not None:
                    sim_col = float(legacy_peak_point[0])
                    sim_row = float(legacy_peak_point[1])
                    resolved_from_peak = True
                elif source_row_key is None or trusted_full_reflection:
                    resolution_lookup[id(entry)] = {
                        **base_diag,
                        "resolution_kind": (
                            "fixed_source" if trusted_full_reflection else "hkl_fallback"
                        ),
                        "resolution_reason": str(legacy_peak_reason),
                    }
                    if not trusted_full_reflection:
                        fallback_entries.append(entry)
                    continue
            if not resolved_from_peak and (source_row_key is None or trusted_full_reflection):
                resolution_lookup[id(entry)] = {
                    **base_diag,
                    "resolution_kind": (
                        "fixed_source" if trusted_full_reflection else "hkl_fallback"
                    ),
                    "resolution_reason": "missing_source_peak_index",
                }
                if not trusted_full_reflection:
                    fallback_entries.append(entry)
                continue

        if not resolved_from_peak:
            if source_row_key is None or trusted_full_reflection:
                resolution_lookup[id(entry)] = {
                    **base_diag,
                    "resolution_kind": (
                        "fixed_source" if trusted_full_reflection else "hkl_fallback"
                    ),
                    "resolution_reason": (
                        "missing_source_peak_index"
                        if trusted_full_reflection
                        else "missing_source_indices"
                    ),
                }
                if not trusted_full_reflection:
                    fallback_entries.append(entry)
                continue
            row_idx = int(source_row_key[1])
            if row_idx < 0 or row_idx >= len(rows):
                resolution_lookup[id(entry)] = {
                    **base_diag,
                    "resolution_kind": "hkl_fallback",
                    "resolution_reason": "source_row_out_of_range",
                    "source_row_count": int(len(rows)),
                }
                fallback_entries.append(entry)
                continue

            row = rows[row_idx]
            try:
                sim_col = float(row[1])
                sim_row = float(row[2])
                sim_hkl = (
                    int(round(float(row[4]))),
                    int(round(float(row[5]))),
                    int(round(float(row[6]))),
                )
            except Exception:
                resolution_lookup[id(entry)] = {
                    **base_diag,
                    "resolution_kind": "hkl_fallback",
                    "resolution_reason": "source_row_parse_failed",
                }
                fallback_entries.append(entry)
                continue
            if not (np.isfinite(sim_col) and np.isfinite(sim_row)):
                resolution_lookup[id(entry)] = {
                    **base_diag,
                    "resolution_kind": "hkl_fallback",
                    "resolution_reason": "invalid_simulated_point",
                }
                fallback_entries.append(entry)
                continue
        measured_hkl = entry.get("hkl")
        if isinstance(measured_hkl, tuple) and len(measured_hkl) == 3:
            try:
                measured_hkl_key = (
                    int(measured_hkl[0]),
                    int(measured_hkl[1]),
                    int(measured_hkl[2]),
                )
            except Exception:
                measured_hkl_key = None
            if measured_hkl_key is not None and sim_hkl is None:
                sim_hkl = tuple(int(v) for v in measured_hkl_key)
            if measured_hkl_key is not None and sim_hkl is not None and measured_hkl_key != sim_hkl:
                resolution_lookup[id(entry)] = {
                    **base_diag,
                    "resolution_kind": (
                        "fixed_source" if trusted_full_reflection else "hkl_fallback"
                    ),
                    "resolution_reason": "source_hkl_mismatch",
                    "resolved_sim_hkl": tuple(int(v) for v in sim_hkl),
                }
                if not trusted_full_reflection:
                    fallback_entries.append(entry)
                continue

        resolved.append((entry, (sim_col, sim_row), tuple(int(v) for v in sim_hkl or (0, 0, 0))))
        resolution_lookup[id(entry)] = {
            **base_diag,
            "resolution_kind": "fixed_source",
            "resolution_reason": "resolved",
            "sim_x": float(sim_col),
            "sim_y": float(sim_row),
            "sim_hkl": tuple(int(v) for v in sim_hkl or (0, 0, 0)),
        }
        used_source_keys.add(source_key)

    return resolved, fallback_entries, resolution_lookup


def _collect_geometry_fit_simulated_candidates(
    fit_miller: np.ndarray,
    hit_tables: Sequence[object],
    *,
    original_indices: np.ndarray | None = None,
    measured_hkls: Optional[Set[Tuple[int, int, int]]] = None,
) -> Dict[Tuple[int, int, int], List[Dict[str, object]]]:
    """Collect simulated peak candidates with stable reflection/branch ids."""

    simulated_by_hkl: Dict[Tuple[int, int, int], List[Dict[str, object]]] = {}
    try:
        max_positions = np.asarray(hit_tables_to_max_positions(hit_tables), dtype=float)
    except Exception:
        max_positions = np.empty((0, 6), dtype=np.float64)
    for table_idx, (H, K, L) in enumerate(fit_miller):
        hkl_key = (int(round(H)), int(round(K)), int(round(L)))
        if measured_hkls is not None and hkl_key not in measured_hkls:
            continue
        if table_idx < 0 or table_idx >= len(hit_tables):
            continue
        rows = _valid_hit_rows(hit_tables[table_idx])
        emitted_candidate = False
        for branch_index in (0, 1):
            branch_row, _ = _branch_representative_row(rows, branch_index=int(branch_index))
            if branch_row is None:
                continue
            emitted_candidate = True
            simulated_by_hkl.setdefault(hkl_key, []).append(
                {
                    "hkl": tuple(int(v) for v in hkl_key),
                    "simulated_x": float(branch_row[1]),
                    "simulated_y": float(branch_row[2]),
                    "source_table_index": int(table_idx),
                    "resolved_table_index": int(table_idx),
                    "source_reflection_index": (
                        int(original_indices[table_idx])
                        if isinstance(original_indices, np.ndarray)
                        and table_idx < int(original_indices.shape[0])
                        else None
                    ),
                    "source_reflection_namespace": (
                        "full_reflection"
                        if isinstance(original_indices, np.ndarray)
                        and table_idx < int(original_indices.shape[0])
                        else None
                    ),
                    "source_reflection_is_full": bool(
                        isinstance(original_indices, np.ndarray)
                        and table_idx < int(original_indices.shape[0])
                    ),
                    "source_branch_index": int(branch_index),
                    "source_peak_index": int(branch_index),
                    "resolved_peak_index": int(branch_index),
                }
            )
        if emitted_candidate:
            continue
        for peak_index in (0, 1):
            peak_point, _ = _resolve_max_position_peak(
                max_positions,
                table_idx=int(table_idx),
                peak_index=int(peak_index),
            )
            if peak_point is None:
                continue
            simulated_by_hkl.setdefault(hkl_key, []).append(
                {
                    "hkl": tuple(int(v) for v in hkl_key),
                    "simulated_x": float(peak_point[0]),
                    "simulated_y": float(peak_point[1]),
                    "source_table_index": int(table_idx),
                    "resolved_table_index": int(table_idx),
                    "source_reflection_index": (
                        int(original_indices[table_idx])
                        if isinstance(original_indices, np.ndarray)
                        and table_idx < int(original_indices.shape[0])
                        else None
                    ),
                    "source_reflection_namespace": (
                        "full_reflection"
                        if isinstance(original_indices, np.ndarray)
                        and table_idx < int(original_indices.shape[0])
                        else None
                    ),
                    "source_reflection_is_full": bool(
                        isinstance(original_indices, np.ndarray)
                        and table_idx < int(original_indices.shape[0])
                    ),
                }
            )
    return simulated_by_hkl


def _build_geometry_fit_fixed_correspondence_groups(
    point_match_diagnostics: Sequence[object] | None,
) -> Dict[int, List[Dict[str, object]]]:
    """Freeze one per-manual-point correspondence from seed diagnostics."""

    grouped: Dict[int, List[Dict[str, object]]] = {}
    for fallback_index, raw_entry in enumerate(point_match_diagnostics or []):
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        try:
            dataset_index = int(entry.get("dataset_index", 0))
        except Exception:
            dataset_index = 0
        try:
            overlay_match_index = int(
                entry.get("overlay_match_index", entry.get("match_input_index", fallback_index))
            )
        except Exception:
            overlay_match_index = int(fallback_index)
        entry["overlay_match_index"] = int(max(overlay_match_index, 0))
        grouped.setdefault(int(dataset_index), []).append(entry)

    for dataset_entries in grouped.values():
        dataset_entries.sort(
            key=lambda entry: (
                int(entry.get("overlay_match_index", 0)),
                int(entry.get("match_input_index", 0)),
            )
        )
    return grouped


def _resolve_geometry_fit_correspondence(
    correspondence: Mapping[str, object],
    *,
    hit_tables: Sequence[object],
    max_positions: np.ndarray,
    filtered_rows_cache: Optional[Dict[int, List[np.ndarray]]] = None,
    current_local_table_signature: Optional[str] = None,
) -> Tuple[Optional[Tuple[float, float]], Dict[str, object]]:
    """Resolve one frozen correspondence back to a simulated detector point."""

    def _payload(
        reason: str,
        *,
        table_idx: Optional[int] = None,
        peak_idx: Optional[int] = None,
        sim_hkl: object = None,
        extra: Optional[Mapping[str, object]] = None,
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {"resolution_reason": str(reason)}
        if table_idx is not None:
            payload["resolved_table_index"] = int(table_idx)
        if peak_idx in {0, 1}:
            payload["resolved_peak_index"] = int(peak_idx)
        if sim_hkl is not None:
            payload["resolved_sim_hkl"] = sim_hkl
        if isinstance(extra, Mapping):
            payload.update(dict(extra))
        return payload

    def _rows_for_table(table_idx: int) -> List[np.ndarray]:
        if filtered_rows_cache is None:
            if 0 <= int(table_idx) < len(hit_tables):
                return _valid_hit_rows(hit_tables[int(table_idx)])
            return []
        if int(table_idx) not in filtered_rows_cache:
            if 0 <= int(table_idx) < len(hit_tables):
                filtered_rows_cache[int(table_idx)] = _valid_hit_rows(hit_tables[int(table_idx)])
            else:
                filtered_rows_cache[int(table_idx)] = []
        return filtered_rows_cache.get(int(table_idx), [])

    def _branch_sim_hkl(row: np.ndarray, fallback: object = None) -> object:
        try:
            return tuple(int(round(float(v))) for v in row[4:7])
        except Exception:
            return fallback

    trusted_full_reflection = _entry_trusted_full_reflection_identity(correspondence)
    frozen_kind = str(correspondence.get("frozen_locator_kind", "") or "").strip().lower()
    frozen_namespace = str(correspondence.get("frozen_table_namespace", "") or "").strip().lower()
    frozen_signature = correspondence.get("frozen_table_signature")
    table_idx: Optional[int] = None
    peak_idx: Optional[int] = None
    row_idx: Optional[int] = None

    if frozen_kind:
        table_idx = _nonnegative_index(correspondence.get("frozen_table_index"))
        if table_idx is None:
            return None, _payload("missing_source_table_index")
        if frozen_kind == "trusted_branch":
            if frozen_namespace != "full_reflection":
                return None, _payload("invalid_frozen_locator")
            peak_idx = _nonnegative_index(correspondence.get("frozen_branch_index"))
            if peak_idx not in {0, 1}:
                return None, _payload("missing_source_peak_index", table_idx=table_idx)
        elif frozen_kind == "local_branch":
            if frozen_namespace not in {"subset_local", "current_full_local"}:
                return None, _payload("invalid_frozen_locator")
            if (
                current_local_table_signature is not None
                and frozen_signature not in {None, "", current_local_table_signature}
            ):
                return None, _payload(
                    "frozen_table_signature_mismatch",
                    table_idx=table_idx,
                    extra={
                        "frozen_table_namespace": frozen_namespace,
                        "frozen_table_signature": frozen_signature,
                    },
                )
            peak_idx = _nonnegative_index(correspondence.get("frozen_branch_index"))
            if peak_idx not in {0, 1}:
                return None, _payload("missing_source_peak_index", table_idx=table_idx)
        elif frozen_kind == "local_row":
            if frozen_namespace not in {"subset_local", "current_full_local"}:
                return None, _payload("invalid_frozen_locator")
            if (
                current_local_table_signature is not None
                and frozen_signature not in {None, "", current_local_table_signature}
            ):
                return None, _payload(
                    "frozen_table_signature_mismatch",
                    table_idx=table_idx,
                    extra={
                        "frozen_table_namespace": frozen_namespace,
                        "frozen_table_signature": frozen_signature,
                    },
                )
            row_idx = _nonnegative_index(correspondence.get("frozen_row_index"))
            if row_idx is None:
                return None, _payload("missing_source_row_index", table_idx=table_idx)
        else:
            return None, _payload("invalid_frozen_locator")
    elif trusted_full_reflection:
        table_idx = _nonnegative_index(correspondence.get("source_reflection_index"))
        if table_idx is None:
            return None, _payload("missing_source_table_index")
        peak_idx = _measured_source_peak_index(correspondence)
        if peak_idx not in {0, 1}:
            return None, _payload("missing_source_peak_index", table_idx=table_idx)
        frozen_kind = "trusted_branch"
    else:
        resolved_table_idx = _nonnegative_index(correspondence.get("resolved_table_index"))
        if resolved_table_idx is not None:
            table_idx = int(resolved_table_idx)
        else:
            source_table_idx = _nonnegative_index(correspondence.get("source_table_index"))
            if source_table_idx is not None and source_table_idx < len(hit_tables):
                table_idx = int(source_table_idx)
        if table_idx is None:
            return None, _payload("missing_source_table_index")
        peak_idx = _measured_source_peak_index(correspondence)
        if peak_idx in {0, 1}:
            frozen_kind = "local_branch"
        else:
            row_idx = _nonnegative_index(correspondence.get("source_row_index"))
            if row_idx is None:
                return None, _payload("missing_source_peak_index", table_idx=table_idx)
            frozen_kind = "local_row"

    if table_idx < 0 or table_idx >= len(hit_tables):
        return None, _payload("source_table_out_of_range", table_idx=table_idx)

    rows = _rows_for_table(int(table_idx))
    if frozen_kind in {"trusted_branch", "local_branch"}:
        branch_row, branch_reason = _branch_representative_row(
            rows,
            branch_index=int(peak_idx),
        )
        if branch_row is not None:
            return (
                float(branch_row[1]),
                float(branch_row[2]),
            ), _payload(
                str(branch_reason),
                table_idx=table_idx,
                peak_idx=peak_idx,
                sim_hkl=_branch_sim_hkl(branch_row),
            )
        if frozen_kind != "trusted_branch":
            legacy_peak_point, legacy_peak_reason = _resolve_max_position_peak(
                max_positions,
                table_idx=int(table_idx),
                peak_index=int(peak_idx),
            )
            if legacy_peak_point is not None:
                fallback_hkl = correspondence.get("hkl")
                return legacy_peak_point, _payload(
                    str(legacy_peak_reason),
                    table_idx=table_idx,
                    peak_idx=peak_idx,
                    sim_hkl=fallback_hkl,
                )
        return None, _payload(str(branch_reason), table_idx=table_idx, peak_idx=peak_idx)

    if row_idx is None:
        return None, _payload("missing_source_row_index", table_idx=table_idx)
    if row_idx >= len(rows):
        return None, _payload(
            "source_row_out_of_range",
            table_idx=table_idx,
            extra={"source_row_count": int(len(rows))},
        )
    row = rows[int(row_idx)]
    try:
        sim_col = float(row[1])
        sim_row = float(row[2])
    except Exception:
        return None, _payload("source_row_parse_failed", table_idx=table_idx)
    if not (np.isfinite(sim_col) and np.isfinite(sim_row)):
        return None, _payload("invalid_source_row_point", table_idx=table_idx)
    return (
        float(sim_col),
        float(sim_row),
    ), _payload(
        "resolved_source_row",
        table_idx=table_idx,
        sim_hkl=_branch_sim_hkl(row, fallback=correspondence.get("hkl")),
    )


def _geometry_fit_correspondence_simulated_point(
    correspondence: Mapping[str, object],
    *,
    hit_tables: Sequence[object],
    max_positions: np.ndarray,
) -> Tuple[Optional[Tuple[float, float]], str]:
    """Resolve one frozen correspondence back to the current simulated point."""

    sim_point, resolution_payload = _resolve_geometry_fit_correspondence(
        correspondence,
        hit_tables=hit_tables,
        max_positions=max_positions,
    )
    return sim_point, str(resolution_payload.get("resolution_reason", "missing_source_peak_index"))


def _detector_pixels_to_fit_space(
    cols: Sequence[float] | np.ndarray,
    rows: Sequence[float] | np.ndarray,
    *,
    center: Sequence[float] | None,
    detector_distance: float,
    pixel_size: float,
    gamma_deg: float = 0.0,
    Gamma_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert detector pixels into `(2theta_deg, phi_deg)` using full detector tilt."""

    cols_arr = np.asarray(cols, dtype=float).reshape(-1)
    rows_arr = np.asarray(rows, dtype=float).reshape(-1)
    two_theta = np.full(cols_arr.shape, np.nan, dtype=np.float64)
    phi = np.full(cols_arr.shape, np.nan, dtype=np.float64)

    if cols_arr.shape != rows_arr.shape:
        return two_theta, phi
    if center is None or len(center) < 2:
        return two_theta, phi
    if not np.isfinite(detector_distance) or detector_distance <= 0.0:
        return two_theta, phi
    if not np.isfinite(pixel_size) or pixel_size <= 0.0:
        return two_theta, phi

    try:
        centre_row = float(center[0])
        centre_col = float(center[1])
    except (TypeError, ValueError, IndexError):
        return two_theta, phi

    cg = math.cos(math.radians(float(gamma_deg)))
    sg = math.sin(math.radians(float(gamma_deg)))
    cG = math.cos(math.radians(float(Gamma_deg)))
    sG = math.sin(math.radians(float(Gamma_deg)))

    R_x_det = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cg, sg],
            [0.0, -sg, cg],
        ],
        dtype=np.float64,
    )
    R_z_det = np.array(
        [
            [cG, sG, 0.0],
            [-sG, cG, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    n_detector = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    unit_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    n_det_rot = R_z_det @ (R_x_det @ n_detector)
    n_det_len = float(np.linalg.norm(n_det_rot))
    if not np.isfinite(n_det_len) or n_det_len <= 1.0e-14:
        return two_theta, phi
    n_det_rot /= n_det_len

    detector_pos = np.array(
        [0.0, float(detector_distance), 0.0],
        dtype=np.float64,
    )

    e1_det = unit_x - float(np.dot(unit_x, n_det_rot)) * n_det_rot
    e1_len = float(np.linalg.norm(e1_det))
    if not np.isfinite(e1_len) or e1_len <= 1.0e-14:
        e1_det = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        e1_det /= e1_len

    e2_det = -np.cross(n_det_rot, e1_det)
    e2_len = float(np.linalg.norm(e2_det))
    if not np.isfinite(e2_len) or e2_len <= 1.0e-14:
        e2_det = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        e2_det /= e2_len

    valid_mask = np.isfinite(cols_arr) & np.isfinite(rows_arr)
    if not np.any(valid_mask):
        return two_theta, phi

    x_det = (cols_arr[valid_mask] - centre_col) * float(pixel_size)
    y_det = (centre_row - rows_arr[valid_mask]) * float(pixel_size)
    world_points = (
        detector_pos[None, :] + x_det[:, None] * e1_det[None, :] + y_det[:, None] * e2_det[None, :]
    )
    radial = np.hypot(world_points[:, 0], world_points[:, 2])
    two_theta_valid = np.degrees(np.arctan2(radial, world_points[:, 1]))
    phi_valid = np.degrees(np.arctan2(world_points[:, 0], world_points[:, 2]))
    phi_valid = (phi_valid + 180.0) % 360.0 - 180.0

    two_theta[valid_mask] = two_theta_valid
    phi[valid_mask] = phi_valid
    return two_theta, phi


def _detector_anchor_from_entry(
    entry: Mapping[str, object],
    *anchor_keys: tuple[str, str, str],
) -> Tuple[Optional[Tuple[float, float]], str]:
    """Return one finite detector anchor from the requested entry key pairs."""

    for x_key, y_key, reason in anchor_keys:
        try:
            col = float(entry.get(x_key, np.nan))
            row = float(entry.get(y_key, np.nan))
        except Exception:
            continue
        if np.isfinite(col) and np.isfinite(row):
            return (float(col), float(row)), reason
    return None, "missing_detector_anchor"


def _measured_detector_anchor(
    entry: Mapping[str, object],
) -> Tuple[Optional[Tuple[float, float]], str]:
    """Return one measured detector anchor in native/oriented detector pixels."""

    return _detector_anchor_from_entry(
        entry,
        ("detector_x", "detector_y", "resolved_detector_anchor"),
        ("x", "y", "resolved_display_anchor"),
    )


def _wavelength_scalar_from_local(
    local: Mapping[str, object],
) -> float:
    """Return one representative wavelength in Angstrom from the trial params."""

    try:
        lambda_value = float(local.get("lambda", np.nan))
    except Exception:
        lambda_value = float("nan")
    if np.isfinite(lambda_value) and lambda_value > 0.0:
        return float(lambda_value)

    mosaic = local.get("mosaic_params", {})
    wavelength_array = None
    if isinstance(mosaic, Mapping):
        wavelength_array = mosaic.get("wavelength_array")
        if wavelength_array is None:
            wavelength_array = mosaic.get("wavelength_i_array")
    if wavelength_array is None:
        return float("nan")

    try:
        wave_arr = np.asarray(wavelength_array, dtype=np.float64).reshape(-1)
    except Exception:
        return float("nan")
    wave_arr = wave_arr[np.isfinite(wave_arr) & (wave_arr > 0.0)]
    if wave_arr.size <= 0:
        return float("nan")
    return float(np.mean(wave_arr))


def _entry_theoretical_two_theta_deg(
    entry: Mapping[str, object],
    *,
    a_lattice: float,
    c_lattice: float,
    wavelength: float,
) -> float | None:
    """Return the theoretical 2theta value for one measured peak entry."""

    hkl = entry.get("hkl")
    if (
        isinstance(hkl, (list, tuple, np.ndarray))
        and len(hkl) >= 3
        and np.isfinite(a_lattice)
        and a_lattice > 0.0
        and np.isfinite(c_lattice)
        and c_lattice > 0.0
        and np.isfinite(wavelength)
        and wavelength > 0.0
    ):
        try:
            spacing = d_spacing(
                int(hkl[0]),
                int(hkl[1]),
                int(hkl[2]),
                float(a_lattice),
                float(c_lattice),
            )
        except Exception:
            spacing = None
        if spacing is not None:
            try:
                two_theta_value = two_theta(float(spacing), float(wavelength))
            except Exception:
                two_theta_value = None
            if two_theta_value is not None and np.isfinite(float(two_theta_value)):
                return float(two_theta_value)

    try:
        qr_value = float(entry.get("background_reference_qr", entry.get("qr", np.nan)))
        qz_value = float(entry.get("background_reference_qz", entry.get("qz", np.nan)))
    except Exception:
        return None
    if (
        not np.isfinite(qr_value)
        or not np.isfinite(qz_value)
        or not np.isfinite(wavelength)
        or wavelength <= 0.0
    ):
        return None

    q_mag = float(math.hypot(qr_value, qz_value))
    if not np.isfinite(q_mag) or q_mag <= 0.0:
        return None
    bragg_arg = float(q_mag * float(wavelength) / (4.0 * np.pi))
    if not np.isfinite(bragg_arg) or abs(bragg_arg) > 1.0:
        return None
    return float(2.0 * np.degrees(np.arcsin(bragg_arg)))


def _fit_space_provenance_summary(
    params: Mapping[str, object],
    *,
    cached_anchor_count: int = 0,
    detector_anchor_count: int = 0,
    two_theta_adjustments_deg: Sequence[float] | None = None,
) -> Dict[str, object]:
    """Return one public fit-space provenance summary payload."""

    provenance = _fit_space_pixel_size_provenance(params)
    adjustment_values = np.asarray(
        list(two_theta_adjustments_deg or ()),
        dtype=np.float64,
    ).reshape(-1)
    adjustment_values = adjustment_values[np.isfinite(adjustment_values)]
    abs_adjustments = np.abs(adjustment_values)
    adjustment_count = int(abs_adjustments.size)
    total_abs_deg = float(np.sum(abs_adjustments)) if adjustment_count > 0 else 0.0
    max_abs_deg = float(np.max(abs_adjustments)) if adjustment_count > 0 else float("nan")
    mean_abs_deg = (
        float(total_abs_deg / float(adjustment_count)) if adjustment_count > 0 else float("nan")
    )
    return {
        "fit_space_pixel_size_source": str(provenance.get("source", "fallback")),
        "fit_space_pixel_size_value": float(provenance.get("value", np.nan)),
        "fit_space_pixel_size_raw": float(provenance.get("raw_pixel_size", np.nan)),
        "fit_space_pixel_size_m_raw": float(provenance.get("raw_pixel_size_m", np.nan)),
        "fit_space_debye_x_raw": float(provenance.get("raw_debye_x", np.nan)),
        "fit_space_debye_y_raw": float(provenance.get("raw_debye_y", np.nan)),
        "fit_space_anchor_count_cached": int(cached_anchor_count),
        "fit_space_anchor_count_detector": int(detector_anchor_count),
        "fit_space_anchor_source_counts": {
            "cached_fit_space_anchor": int(cached_anchor_count),
            "detector_fit_space_anchor": int(detector_anchor_count),
        },
        "fit_space_two_theta_adjustment_count": int(adjustment_count),
        "fit_space_two_theta_adjustment_total_abs_deg": float(total_abs_deg),
        "fit_space_two_theta_adjustment_mean_abs_deg": float(mean_abs_deg),
        "fit_space_two_theta_adjustment_max_abs_deg": float(max_abs_deg),
    }


def _measured_fit_space_anchor(
    entry: Mapping[str, object],
    *,
    center: Sequence[float] | None,
    detector_distance: float,
    pixel_size: float,
    gamma_deg: float = 0.0,
    Gamma_deg: float = 0.0,
    a_lattice: float | None = None,
    c_lattice: float | None = None,
    wavelength: float | None = None,
) -> Tuple[Optional[Tuple[float, float]], str, Dict[str, object]]:
    """Return one measured `(2theta, phi)` anchor plus fit-space provenance."""

    metadata: Dict[str, object] = {
        "anchor_source": "missing",
        "two_theta_adjustment_deg": float("nan"),
        "cached_two_theta_deg": float("nan"),
        "cached_phi_deg": float("nan"),
        "reference_two_theta_deg": float("nan"),
        "current_theoretical_two_theta_deg": float("nan"),
    }

    try:
        base_two_theta = float(
            entry.get(
                "background_two_theta_deg",
                entry.get("caked_x", entry.get("raw_caked_x", np.nan)),
            )
        )
        base_phi = float(
            entry.get(
                "background_phi_deg",
                entry.get("caked_y", entry.get("raw_caked_y", np.nan)),
            )
        )
    except Exception:
        base_two_theta = float("nan")
        base_phi = float("nan")

    try:
        reference_two_theta = float(entry.get("background_reference_two_theta_deg", np.nan))
    except Exception:
        reference_two_theta = float("nan")
    if not np.isfinite(reference_two_theta):
        try:
            ref_a = float(entry.get("background_reference_a", np.nan))
        except Exception:
            ref_a = float("nan")
        try:
            ref_c = float(entry.get("background_reference_c", np.nan))
        except Exception:
            ref_c = float("nan")
        try:
            ref_lambda = float(entry.get("background_reference_lambda", np.nan))
        except Exception:
            ref_lambda = float("nan")
        theoretical_ref = _entry_theoretical_two_theta_deg(
            entry,
            a_lattice=float(ref_a),
            c_lattice=float(ref_c),
            wavelength=float(ref_lambda),
        )
        if theoretical_ref is not None:
            reference_two_theta = float(theoretical_ref)
    if np.isfinite(reference_two_theta):
        metadata["reference_two_theta_deg"] = float(reference_two_theta)
    if a_lattice is not None and c_lattice is not None and wavelength is not None:
        current_theoretical = _entry_theoretical_two_theta_deg(
            entry,
            a_lattice=float(a_lattice),
            c_lattice=float(c_lattice),
            wavelength=float(wavelength),
        )
        if current_theoretical is not None and np.isfinite(current_theoretical):
            metadata["current_theoretical_two_theta_deg"] = float(current_theoretical)

    if (
        bool(entry.get("fit_space_anchor_override", False))
        and np.isfinite(base_two_theta)
        and np.isfinite(base_phi)
    ):
        metadata["anchor_source"] = "cached_fit_space_anchor"
        metadata["cached_two_theta_deg"] = float(base_two_theta)
        metadata["cached_phi_deg"] = float(base_phi)
        metadata["two_theta_adjustment_deg"] = 0.0
        return (
            (float(base_two_theta), float(base_phi)),
            "cached_fit_space_anchor",
            metadata,
        )

    measured_anchor, measured_reason = _detector_anchor_from_entry(
        entry,
        ("detector_x", "detector_y", "detector_fit_space_anchor"),
        ("x", "y", "display_fit_space_anchor"),
        (
            "background_detector_x",
            "background_detector_y",
            "background_detector_fit_space_anchor",
        ),
    )
    if measured_anchor is not None:
        two_theta_arr, phi_arr = _detector_pixels_to_fit_space(
            np.array([measured_anchor[0]], dtype=np.float64),
            np.array([measured_anchor[1]], dtype=np.float64),
            center=center,
            detector_distance=float(detector_distance),
            pixel_size=float(pixel_size),
            gamma_deg=float(gamma_deg),
            Gamma_deg=float(Gamma_deg),
        )
        if (
            two_theta_arr.size < 1
            or phi_arr.size < 1
            or not np.isfinite(two_theta_arr[0])
            or not np.isfinite(phi_arr[0])
        ):
            metadata["anchor_source"] = "invalid_fit_space_anchor"
            return None, "invalid_fit_space_anchor", metadata
        metadata["anchor_source"] = str(measured_reason)
        metadata["two_theta_adjustment_deg"] = 0.0
        return (
            (float(two_theta_arr[0]), float(phi_arr[0])),
            str(measured_reason),
            metadata,
        )

    if np.isfinite(base_two_theta) and np.isfinite(base_phi):
        metadata["anchor_source"] = "cached_fit_space_anchor"
        metadata["cached_two_theta_deg"] = float(base_two_theta)
        metadata["cached_phi_deg"] = float(base_phi)
        metadata["two_theta_adjustment_deg"] = 0.0
        return (
            (float(base_two_theta), float(base_phi)),
            "cached_fit_space_anchor",
            metadata,
        )

    metadata["anchor_source"] = str(measured_reason)
    return None, measured_reason, metadata


def _pixel_to_angles(
    col: float,
    row: float,
    center: Sequence[float],
    detector_distance: float,
    pixel_size: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Convert a detector pixel position to (2θ, φ) angles in degrees.

    Parameters
    ----------
    col, row
        Pixel coordinates using the same convention as the simulator, where
        ``col`` corresponds to the horizontal axis and ``row`` to the vertical
        axis.
    center
        Beam centre expressed as ``(row, col)`` as used throughout the
        simulation code.
    detector_distance
        Sample-to-detector distance in metres.
    pixel_size
        Pixel size in metres.
    """

    two_theta_arr, phi_arr = _detector_pixels_to_fit_space(
        np.array([col], dtype=np.float64),
        np.array([row], dtype=np.float64),
        center=center,
        detector_distance=float(detector_distance),
        pixel_size=float(pixel_size),
    )
    if (
        two_theta_arr.size < 1
        or phi_arr.size < 1
        or not np.isfinite(two_theta_arr[0])
        or not np.isfinite(phi_arr[0])
    ):
        return None, None
    return float(two_theta_arr[0]), float(phi_arr[0])


def _angular_difference_deg(a: float, b: float) -> float:
    """Return the signed minimal difference between two angles in degrees."""

    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    diff = float(a) - float(b)
    return (diff + 180.0) % 360.0 - 180.0


def simulate_and_compare_hkl(
    miller,
    intensities,
    image_size,
    params,
    measured_peaks,
    pixel_tol=np.inf,
    *,
    prefer_python_runner: bool = False,
):
    """Simulate reflections and pair them with measured peak positions.

    The routine performs a full-pattern simulation using
    :func:`process_peaks_parallel`, then, for each Miller index present in
    ``measured_peaks``, compares the simulated and measured peak *angles*.
    For every reflection we look at the maximum position extracted from the
    radial and azimuthal 1D slices.  The residuals therefore measure the
    difference in 2θ (radial) and φ (azimuthal) between simulation and
    experiment; peak shapes are ignored.  ``pixel_tol`` is interpreted as a
    tolerance on the combined angular difference (in degrees).

    Returns
    -------
    distances : :class:`numpy.ndarray`
        Residuals for the matched peaks.  Entries alternate between radial
        (2θ) and azimuthal (φ) absolute differences, all expressed in degrees.
    sim_coords : list[tuple[float, float]]
        Detector pixel coordinates of the simulated peaks that were matched.
    meas_coords : list[tuple[float, float]]
        Detector pixel coordinates of the measured peaks corresponding to
        ``sim_coords``.
    sim_millers : list[tuple[int, int, int]]
        Miller indices associated with each entry in ``sim_coords``.
    meas_millers : list[tuple[int, int, int]]
        Miller indices associated with each entry in ``meas_coords``.
    """
    sim_subset = _prepare_reflection_subset(miller, intensities, measured_peaks)
    normalized_measured = sim_subset.measured_entries
    sim_miller = sim_subset.miller
    sim_intensities = sim_subset.intensities
    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)

    # Unpack geometry & mosaic parameters
    a = params["a"]
    c = params["c"]
    dist = params["corto_detector"]
    gamma = params["gamma"]
    Gamma = params["Gamma"]
    chi = params["chi"]
    psi = params.get("psi", 0.0)
    psi_z = params.get("psi_z", 0.0)
    zs = params["zs"]
    zb = params["zb"]
    debye_x = params["debye_x"]
    debye_y = params["debye_y"]
    n2 = params["n2"]
    center = params["center"]
    theta_initial = params["theta_initial"]
    cor_angle = params.get("cor_angle", 0.0)

    mosaic = params["mosaic_params"]
    wavelength_array = mosaic.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic.get("wavelength_i_array")

    # Full-pattern simulation
    updated_image, hit_tables, *_ = _process_peaks_parallel_safe(
        sim_miller,
        sim_intensities,
        image_size,
        a,
        c,
        wavelength_array,
        sim_buffer,
        dist,
        gamma,
        Gamma,
        chi,
        psi,
        psi_z,
        zs,
        zb,
        n2,
        mosaic["beam_x_array"],
        mosaic["beam_y_array"],
        mosaic["theta_array"],
        mosaic["phi_array"],
        mosaic["sigma_mosaic_deg"],
        mosaic["gamma_mosaic_deg"],
        mosaic["eta"],
        wavelength_array,
        debye_x,
        debye_y,
        center,
        theta_initial,
        cor_angle,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        save_flag=0,
        prefer_python_runner=bool(prefer_python_runner),
        **_simulation_kernel_kwargs(params, mosaic),
    )
    maxpos = hit_tables_to_max_positions(hit_tables)

    distances: list[float] = []
    sim_coords: list[tuple[float, float]] = []
    meas_coords: list[tuple[float, float]] = []
    sim_millers: list[tuple[int, int, int]] = []
    meas_millers: list[tuple[int, int, int]] = []

    detector_distance = float(params.get("corto_detector", 0.0))
    pixel_size = _estimate_pixel_size(params)
    centre = params.get("center")

    def _match_points(
        simulated_points: list[tuple[float, float]],
        measured_points: list[tuple[float, float]],
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        assigned = _build_global_point_matches(
            simulated_points,
            measured_points,
            max_distance=np.inf,
        )
        return [
            (
                (float(sim_pt[0]), float(sim_pt[1])),
                (float(meas_pt[0]), float(meas_pt[1])),
            )
            for sim_pt, meas_pt, *_ in assigned
        ]

    fixed_matches, fallback_measured, _ = _resolve_fixed_source_matches(
        normalized_measured,
        hit_tables,
    )

    for measured_entry, (sim_center_col, sim_center_row), sim_hkl in fixed_matches:
        try:
            meas_center_col = float(measured_entry["x"])
            meas_center_row = float(measured_entry["y"])
        except Exception:
            continue
        if not (
            np.isfinite(sim_center_col)
            and np.isfinite(sim_center_row)
            and np.isfinite(meas_center_col)
            and np.isfinite(meas_center_row)
        ):
            continue

        sim_two_theta, sim_phi = _pixel_to_angles(
            sim_center_col, sim_center_row, centre, detector_distance, pixel_size
        )
        meas_two_theta, meas_phi = _pixel_to_angles(
            meas_center_col, meas_center_row, centre, detector_distance, pixel_size
        )
        if sim_two_theta is None or sim_phi is None or meas_two_theta is None or meas_phi is None:
            continue

        radial_diff = abs(sim_two_theta - meas_two_theta)
        azimuthal_diff = abs(_angular_difference_deg(sim_phi, meas_phi))
        combined = math.hypot(radial_diff, azimuthal_diff)
        if combined > pixel_tol:
            continue

        hkl_value = measured_entry.get("hkl")
        if isinstance(hkl_value, tuple) and len(hkl_value) == 3:
            hkl_key = (
                int(hkl_value[0]),
                int(hkl_value[1]),
                int(hkl_value[2]),
            )
        else:
            hkl_key = sim_hkl

        distances.extend([radial_diff, azimuthal_diff])
        sim_coords.append((sim_center_col, sim_center_row))
        meas_coords.append((meas_center_col, meas_center_row))
        sim_millers.append(hkl_key)
        meas_millers.append(hkl_key)

    measured_dict = build_measured_dict(fallback_measured)

    for i, (H, K, L) in enumerate(sim_miller):
        key = (int(round(H)), int(round(K)), int(round(L)))
        candidates = measured_dict.get(key)
        if not candidates:
            continue

        I0, x0, y0, I1, x1, y1 = maxpos[i]
        simulated_points: list[tuple[float, float]] = []
        for col, row in ((x0, y0), (x1, y1)):
            if not np.isfinite(col) or not np.isfinite(row):
                continue
            simulated_points.append((float(col), float(row)))

        if not simulated_points:
            continue

        sim_cols, sim_rows = zip(*simulated_points)
        sim_center_col = float(np.mean(sim_cols))
        sim_center_row = float(np.mean(sim_rows))
        sim_two_theta, sim_phi = _pixel_to_angles(
            sim_center_col, sim_center_row, centre, detector_distance, pixel_size
        )
        if sim_two_theta is None or sim_phi is None:
            continue

        measured_points: list[tuple[float, float]] = []
        for mx, my in candidates:
            if not np.isfinite(mx) or not np.isfinite(my):
                continue
            measured_points.append((float(mx), float(my)))

        if not measured_points:
            continue

        point_matches = _match_points(simulated_points, measured_points)
        if not point_matches:
            sim_center_col = float(simulated_points[0][0])
            sim_center_row = float(simulated_points[0][1])
            meas_center_col = float(measured_points[0][0])
            meas_center_row = float(measured_points[0][1])
            point_matches = [
                (
                    (sim_center_col, sim_center_row),
                    (meas_center_col, meas_center_row),
                )
            ]

        for (sim_center_col, sim_center_row), (meas_center_col, meas_center_row) in point_matches:
            sim_two_theta, sim_phi = _pixel_to_angles(
                sim_center_col, sim_center_row, centre, detector_distance, pixel_size
            )
            meas_two_theta, meas_phi = _pixel_to_angles(
                meas_center_col, meas_center_row, centre, detector_distance, pixel_size
            )
            if (
                sim_two_theta is None
                or sim_phi is None
                or meas_two_theta is None
                or meas_phi is None
            ):
                continue

            radial_diff = abs(sim_two_theta - meas_two_theta)
            azimuthal_diff = abs(_angular_difference_deg(sim_phi, meas_phi))
            combined = math.hypot(radial_diff, azimuthal_diff)

            if combined <= pixel_tol:
                distances.extend([radial_diff, azimuthal_diff])
                sim_coords.append((sim_center_col, sim_center_row))
                meas_coords.append((meas_center_col, meas_center_row))
                sim_millers.append(key)
                meas_millers.append(key)

    return (
        np.array(distances, dtype=float),
        sim_coords,
        meas_coords,
        sim_millers,
        meas_millers,
    )


def compute_peak_position_error_geometry_local(
    gamma,
    Gamma,
    dist,
    theta_initial,
    cor_angle,
    zs,
    zb,
    chi,
    a,
    c,
    center_x,
    center_y,
    measured_peaks,
    miller,
    intensities,
    image_size,
    mosaic_params,
    n2,
    psi,
    psi_z,
    debye_x,
    debye_y,
    wavelength,
    pixel_tol=np.inf,
    optics_mode=0,
):
    """
    Objective for DE: returns the 1D array of distances for all matched peaks.
    """
    params = {
        "gamma": gamma,
        "Gamma": Gamma,
        "corto_detector": dist,
        "theta_initial": theta_initial,
        "cor_angle": cor_angle,
        "zs": zs,
        "zb": zb,
        "chi": chi,
        "a": a,
        "c": c,
        "center": (center_x, center_y),
        "lambda": wavelength,
        "n2": n2,
        "psi": psi,
        "psi_z": psi_z,
        "debye_x": debye_x,
        "debye_y": debye_y,
        "mosaic_params": mosaic_params,
        "optics_mode": int(optics_mode),
    }
    D, *_ = simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks,
        pixel_tol,
    )
    return D


def fit_geometry_parameters(
    miller,
    intensities,
    image_size,
    params,
    measured_peaks,
    var_names,
    pixel_tol=np.inf,
    *,
    experimental_image: Optional[np.ndarray] = None,
    dataset_specs: Optional[Sequence[Dict[str, object]]] = None,
    candidate_param_names: Optional[Sequence[str]] = None,
    refinement_config: Optional[Dict[str, Dict[str, float]]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    live_update_callback: Optional[Callable[[Mapping[str, object]], None]] = None,
):
    """
    Least-squares fit for a subset of geometry parameters.
    var_names is a list of keys in `params` to optimize.
    """

    _set_numba_usage_from_config(refinement_config)

    def _emit_status(message: str) -> None:
        if not callable(status_callback):
            return
        try:
            text = str(message).strip()
        except Exception:
            text = ""
        if not text:
            return
        try:
            status_callback(text)
        except Exception:
            pass

    dataset_spec_entries = _coerce_sequence_items(dataset_specs)
    point_match_mode = experimental_image is not None or bool(dataset_spec_entries)

    use_single_ray = False

    optimizer_cfg: Dict[str, float] = {}
    if isinstance(refinement_config, dict):
        optimizer_cfg = (
            refinement_config.get("optimizer") or refinement_config.get("solver", {}) or {}
        )
    if not isinstance(optimizer_cfg, dict):
        optimizer_cfg = {}
    solver_cfg = optimizer_cfg

    prior_cfg: Dict[str, float] = {}
    if isinstance(refinement_config, dict):
        prior_cfg = refinement_config.get("priors", {}) or {}
    if not isinstance(prior_cfg, dict):
        prior_cfg = {}

    image_refinement_cfg: Dict[str, float] = {}
    if isinstance(refinement_config, dict):
        image_refinement_cfg = refinement_config.get("image_refinement", {}) or {}
    if not isinstance(image_refinement_cfg, dict):
        image_refinement_cfg = {}

    ridge_refinement_cfg: Dict[str, float] = {}
    if isinstance(refinement_config, dict):
        ridge_refinement_cfg = refinement_config.get("ridge_refinement", {}) or {}
    if not isinstance(ridge_refinement_cfg, dict):
        ridge_refinement_cfg = {}

    full_beam_polish_cfg: Dict[str, object] = {}
    if isinstance(refinement_config, dict):
        full_beam_polish_cfg = refinement_config.get("full_beam_polish", {}) or {}
    if not isinstance(full_beam_polish_cfg, dict):
        full_beam_polish_cfg = {}

    identifiability_cfg: Dict[str, float] = {}
    if isinstance(refinement_config, dict):
        identifiability_cfg = refinement_config.get("identifiability", {}) or {}
    if not isinstance(identifiability_cfg, dict):
        identifiability_cfg = {}
    seed_search_cfg: Dict[str, object] = {}
    if isinstance(refinement_config, dict):
        seed_search_cfg = refinement_config.get("seed_search", {}) or {}
    if not isinstance(seed_search_cfg, dict):
        seed_search_cfg = {}
    discrete_modes_cfg: Dict[str, object] = {}
    if isinstance(refinement_config, dict):
        discrete_modes_cfg = refinement_config.get("discrete_modes", {}) or {}
    if not isinstance(discrete_modes_cfg, dict):
        discrete_modes_cfg = {}
    manual_point_fit_mode = bool(solver_cfg.get("manual_point_fit_mode", False))
    manual_point_fit_has_cached_sources = bool(
        _dataset_specs_have_geometry_source_identities(dataset_spec_entries)
        or (
            not dataset_spec_entries
            and _measured_entries_have_geometry_source_identities(measured_peaks)
        )
    )

    solver_loss = str(solver_cfg.get("loss", "linear")).strip().lower()
    if solver_loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
        solver_loss = "linear"

    solver_f_scale = float(
        solver_cfg.get(
            "f_scale_px",
            1.0 if manual_point_fit_mode else (6.0 if point_match_mode else 1.0),
        )
    )
    solver_f_scale = max(solver_f_scale, 1e-6)

    solver_method = str(solver_cfg.get("method", "trf")).strip().lower()
    if solver_method != "trf":
        solver_method = "trf"

    solver_max_nfev = int(solver_cfg.get("max_nfev", 120 if point_match_mode else 60))
    solver_max_nfev = max(20, solver_max_nfev)
    full_beam_polish_enabled = bool(
        full_beam_polish_cfg.get(
            "enabled",
            point_match_mode and not manual_point_fit_mode,
        )
    )
    full_beam_polish_max_nfev = int(full_beam_polish_cfg.get("max_nfev", 40))
    full_beam_polish_max_nfev = max(5, full_beam_polish_max_nfev)
    full_beam_polish_match_radius_px = float(full_beam_polish_cfg.get("match_radius_px", 24.0))
    if not np.isfinite(full_beam_polish_match_radius_px) or full_beam_polish_match_radius_px <= 0.0:
        full_beam_polish_match_radius_px = 24.0

    solver_parallel_mode = str(solver_cfg.get("parallel_mode", "auto")).strip().lower()
    if solver_parallel_mode in {"false", "none", "disabled"}:
        solver_parallel_mode = "off"
    if solver_parallel_mode not in {"auto", "off", "datasets", "restarts"}:
        solver_parallel_mode = "auto"
    solver_workers_cfg = solver_cfg.get("workers", "auto")
    solver_worker_numba_threads_cfg = solver_cfg.get(
        "worker_numba_threads",
        0,
    )

    seed_prescore_top_k = int(seed_search_cfg.get("prescore_top_k", 8))
    seed_prescore_top_k = max(seed_prescore_top_k, 1)
    seed_n_global = int(seed_search_cfg.get("n_global", 24))
    seed_n_global = max(seed_n_global, 0)
    seed_n_jitter = int(seed_search_cfg.get("n_jitter", 12))
    seed_n_jitter = max(seed_n_jitter, 0)
    seed_jitter_sigma_u = float(seed_search_cfg.get("jitter_sigma_u", 0.5))
    if not np.isfinite(seed_jitter_sigma_u) or seed_jitter_sigma_u < 0.0:
        seed_jitter_sigma_u = 0.5
    seed_min_separation_u = float(seed_search_cfg.get("min_seed_separation_u", 0.5))
    if not np.isfinite(seed_min_separation_u) or seed_min_separation_u < 0.0:
        seed_min_separation_u = 0.5
    trusted_prior_fraction_of_span = float(
        seed_search_cfg.get("trusted_prior_fraction_of_span", 0.15)
    )
    if not np.isfinite(trusted_prior_fraction_of_span) or trusted_prior_fraction_of_span < 0.0:
        trusted_prior_fraction_of_span = 0.15
    raw_solver_restarts = solver_cfg.get("restarts", None)
    try:
        requested_solver_restarts = (
            None if raw_solver_restarts is None else int(raw_solver_restarts)
        )
    except Exception:
        requested_solver_restarts = None
    zero_seed_only_solver = bool(
        requested_solver_restarts == 0 and bool(full_beam_polish_cfg.get("enabled", False))
    )
    if requested_solver_restarts is None:
        solver_restarts = max(seed_prescore_top_k, 1) if point_match_mode else seed_prescore_top_k
    else:
        solver_restarts = max(int(requested_solver_restarts), 0)
        if zero_seed_only_solver:
            solver_restarts = 1
            seed_prescore_top_k = 1
            seed_n_global = 0
            seed_n_jitter = 0
    solver_restart_jitter = float(seed_search_cfg.get("restart_jitter", 0.15))
    if not np.isfinite(solver_restart_jitter) or solver_restart_jitter < 0.0:
        solver_restart_jitter = 0.15
    restart_selection_tol = float(seed_search_cfg.get("restart_selection_tol", 1.0e-12))
    if not np.isfinite(restart_selection_tol) or restart_selection_tol < 0.0:
        restart_selection_tol = 1.0e-12

    missing_pair_penalty = float(solver_cfg.get("missing_pair_penalty_px", 20.0))
    missing_pair_penalty = max(0.0, missing_pair_penalty)
    dynamic_point_geometry_fit = bool(solver_cfg.get("dynamic_point_geometry_fit", False))
    if manual_point_fit_mode and point_match_mode and manual_point_fit_has_cached_sources:
        dynamic_point_geometry_fit = True
    missing_pair_penalty_deg = float(solver_cfg.get("missing_pair_penalty_deg", 5.0))
    if not np.isfinite(missing_pair_penalty_deg) or missing_pair_penalty_deg < 0.0:
        missing_pair_penalty_deg = 5.0
    q_group_line_constraints_enabled = bool(
        solver_cfg.get(
            "q_group_line_constraints",
            solver_cfg.get("q_group_line_constraints_enabled", manual_point_fit_mode),
        )
    )
    q_group_line_angle_weight = float(solver_cfg.get("q_group_line_angle_weight", 0.6))
    if not np.isfinite(q_group_line_angle_weight) or q_group_line_angle_weight < 0.0:
        q_group_line_angle_weight = 0.6
    q_group_line_offset_weight = float(solver_cfg.get("q_group_line_offset_weight", 1.0))
    if not np.isfinite(q_group_line_offset_weight) or q_group_line_offset_weight < 0.0:
        q_group_line_offset_weight = 1.0
    q_group_line_missing_penalty_scale = float(
        solver_cfg.get("q_group_line_missing_penalty_scale", 0.35)
    )
    if (
        not np.isfinite(q_group_line_missing_penalty_scale)
        or q_group_line_missing_penalty_scale < 0.0
    ):
        q_group_line_missing_penalty_scale = 0.35
    hk0_peak_priority_weight = float(
        solver_cfg.get(
            "hk0_peak_priority_weight",
            6.0 if manual_point_fit_mode else 1.0,
        )
    )
    if not np.isfinite(hk0_peak_priority_weight) or hk0_peak_priority_weight < 1.0:
        hk0_peak_priority_weight = 1.0

    weighted_matching = bool(solver_cfg.get("weighted_matching", False))
    use_measurement_uncertainty = bool(
        solver_cfg.get(
            "use_measurement_uncertainty",
            not manual_point_fit_mode,
        )
    )
    stagnation_probe_enabled = bool(
        solver_cfg.get(
            "stagnation_probe",
            point_match_mode and not manual_point_fit_mode,
        )
    )
    stagnation_probe_fraction = float(solver_cfg.get("stagnation_probe_fraction", 0.35))
    if not np.isfinite(stagnation_probe_fraction) or stagnation_probe_fraction <= 0.0:
        stagnation_probe_fraction = 0.0
    stagnation_probe_min_improvement = float(
        solver_cfg.get("stagnation_probe_min_improvement", 1e-6)
    )
    if not np.isfinite(stagnation_probe_min_improvement):
        stagnation_probe_min_improvement = 1e-6
    stagnation_probe_min_improvement = max(0.0, stagnation_probe_min_improvement)
    stagnation_probe_pairwise = bool(solver_cfg.get("stagnation_probe_pairwise", True))
    stagnation_probe_pair_limit = int(solver_cfg.get("stagnation_probe_pair_limit", 6))
    stagnation_probe_pair_limit = max(0, stagnation_probe_pair_limit)
    stagnation_probe_random_directions = int(
        solver_cfg.get("stagnation_probe_random_directions", 0)
    )
    stagnation_probe_random_directions = max(0, stagnation_probe_random_directions)
    manual_fail_fast_enabled = bool(
        solver_cfg.get("manual_fail_fast", manual_point_fit_mode and point_match_mode)
    )
    if not point_match_mode:
        manual_fail_fast_enabled = False
    manual_fail_fast_eval_window = int(solver_cfg.get("manual_fail_fast_eval_window", 30))
    manual_fail_fast_eval_window = max(5, manual_fail_fast_eval_window)
    manual_fail_fast_probe_period = int(solver_cfg.get("manual_fail_fast_probe_period", 10))
    manual_fail_fast_probe_period = max(1, manual_fail_fast_probe_period)
    manual_fail_fast_peak_rms_px = float(solver_cfg.get("manual_fail_fast_peak_rms_px", 250.0))
    if not np.isfinite(manual_fail_fast_peak_rms_px) or manual_fail_fast_peak_rms_px <= 0.0:
        manual_fail_fast_peak_rms_px = 250.0
    manual_fail_fast_min_bound_parameters = int(
        solver_cfg.get("manual_fail_fast_min_bound_parameters", 2)
    )
    manual_fail_fast_min_bound_parameters = max(
        1,
        manual_fail_fast_min_bound_parameters,
    )
    staged_release_cfg_raw = solver_cfg.get("staged_release", {})
    if isinstance(staged_release_cfg_raw, bool):
        staged_release_cfg: Dict[str, object] = {"enabled": bool(staged_release_cfg_raw)}
    elif isinstance(staged_release_cfg_raw, dict):
        staged_release_cfg = dict(staged_release_cfg_raw)
    else:
        staged_release_cfg = {}
    staged_release_enabled = bool(staged_release_cfg.get("enabled", False))
    staged_release_max_nfev = int(
        staged_release_cfg.get("max_nfev", max(10, min(30, solver_max_nfev)))
    )
    staged_release_max_nfev = max(5, staged_release_max_nfev)
    staged_release_max_cost_increase_fraction = float(
        staged_release_cfg.get("max_cost_increase_fraction", 0.0)
    )
    if not np.isfinite(staged_release_max_cost_increase_fraction):
        staged_release_max_cost_increase_fraction = 0.0
    staged_release_max_cost_increase_fraction = max(
        0.0,
        staged_release_max_cost_increase_fraction,
    )
    staged_release_blocks_cfg = staged_release_cfg.get("blocks", None)
    reparameterize_cfg_raw = solver_cfg.get("reparameterize_pairs", {})
    if isinstance(reparameterize_cfg_raw, bool):
        reparameterize_cfg: Dict[str, object] = {"enabled": bool(reparameterize_cfg_raw)}
    elif isinstance(reparameterize_cfg_raw, dict):
        reparameterize_cfg = dict(reparameterize_cfg_raw)
    else:
        reparameterize_cfg = {}
    reparameterize_enabled = bool(reparameterize_cfg.get("enabled", False))
    reparameterize_max_nfev = int(
        reparameterize_cfg.get("max_nfev", max(10, min(30, solver_max_nfev)))
    )
    reparameterize_max_nfev = max(5, reparameterize_max_nfev)
    reparameterize_max_cost_increase_fraction = float(
        reparameterize_cfg.get("max_cost_increase_fraction", 0.0)
    )
    if not np.isfinite(reparameterize_max_cost_increase_fraction):
        reparameterize_max_cost_increase_fraction = 0.0
    reparameterize_max_cost_increase_fraction = max(
        0.0,
        reparameterize_max_cost_increase_fraction,
    )
    reparameterize_pairs_cfg = reparameterize_cfg.get("pairs", None)
    reparameterize_default_pairs = [
        ["gamma", "Gamma"],
        ["zs", "zb"],
        ["theta_initial", "cor_angle"],
    ]

    parameter_group_map = {
        "center_x": "center",
        "center_y": "center",
        "gamma": "tilt",
        "Gamma": "tilt",
        "chi": "tilt",
        "cor_angle": "tilt",
        "theta_initial": "tilt",
        "theta_offset": "tilt",
        "corto_detector": "distance",
        "zs": "distance",
        "zb": "distance",
        "a": "lattice",
        "c": "lattice",
        "psi_z": "lattice",
    }

    def _parameter_group(name: str) -> str:
        return str(parameter_group_map.get(str(name), "other"))

    staged_release_default_blocks = [
        ["center", "tilt"],
        ["distance"],
        ["lattice", "other"],
    ]

    image_refinement_enabled = bool(image_refinement_cfg.get("enabled", False))
    ridge_refinement_enabled = bool(ridge_refinement_cfg.get("enabled", False))
    identifiability_enabled = bool(identifiability_cfg.get("enabled", point_match_mode))
    identifiability_fd_step_fraction = float(identifiability_cfg.get("fd_step_fraction", 0.02))
    if not np.isfinite(identifiability_fd_step_fraction) or identifiability_fd_step_fraction <= 0.0:
        identifiability_fd_step_fraction = 0.02
    identifiability_fd_min_step = float(identifiability_cfg.get("fd_min_step", 1.0e-4))
    if not np.isfinite(identifiability_fd_min_step) or identifiability_fd_min_step <= 0.0:
        identifiability_fd_min_step = 1.0e-4
    identifiability_condition_warn = float(identifiability_cfg.get("condition_number_warn", 1.0e8))
    if not np.isfinite(identifiability_condition_warn) or identifiability_condition_warn <= 1.0:
        identifiability_condition_warn = 1.0e8
    identifiability_top_peaks_per_parameter = int(
        identifiability_cfg.get("top_peaks_per_parameter", 3)
    )
    identifiability_top_peaks_per_parameter = max(
        1,
        identifiability_top_peaks_per_parameter,
    )
    identifiability_correlation_warn = float(identifiability_cfg.get("correlation_warn", 0.95))
    if not np.isfinite(identifiability_correlation_warn) or identifiability_correlation_warn < 0.0:
        identifiability_correlation_warn = 0.95
    identifiability_correlation_warn = min(
        max(identifiability_correlation_warn, 0.0),
        0.999999,
    )
    identifiability_weak_norm_ratio = float(
        identifiability_cfg.get("weak_column_norm_ratio", 1.0e-6)
    )
    if not np.isfinite(identifiability_weak_norm_ratio) or identifiability_weak_norm_ratio < 0.0:
        identifiability_weak_norm_ratio = 1.0e-6
    identifiability_fd_abs_step_u = float(identifiability_cfg.get("fd_abs_step_u", 1.0e-3))
    if not np.isfinite(identifiability_fd_abs_step_u) or identifiability_fd_abs_step_u <= 0.0:
        identifiability_fd_abs_step_u = 1.0e-3
    identifiability_sv_rcond = float(identifiability_cfg.get("sv_rcond", 1.0e-8))
    if not np.isfinite(identifiability_sv_rcond) or identifiability_sv_rcond <= 0.0:
        identifiability_sv_rcond = 1.0e-8
    identifiability_weak_singular_rel = float(identifiability_cfg.get("weak_singular_rel", 1.0e-3))
    if (
        not np.isfinite(identifiability_weak_singular_rel)
        or identifiability_weak_singular_rel <= 0.0
    ):
        identifiability_weak_singular_rel = 1.0e-3
    identifiability_combo_thresh = float(identifiability_cfg.get("combo_thresh", 0.2))
    if not np.isfinite(identifiability_combo_thresh) or identifiability_combo_thresh <= 0.0:
        identifiability_combo_thresh = 0.2
    identifiability_block_corr_abs = float(identifiability_cfg.get("block_corr_abs", 0.90))
    if not np.isfinite(identifiability_block_corr_abs) or identifiability_block_corr_abs < 0.0:
        identifiability_block_corr_abs = 0.90
    identifiability_block_corr_abs = min(max(identifiability_block_corr_abs, 0.0), 0.999999)
    identifiability_weak_column_rel = float(identifiability_cfg.get("weak_column_rel", 0.10))
    if not np.isfinite(identifiability_weak_column_rel) or identifiability_weak_column_rel < 0.0:
        identifiability_weak_column_rel = 0.10
    identifiability_single_release_min_indep = float(
        identifiability_cfg.get("single_release_min_indep", 0.35)
    )
    if (
        not np.isfinite(identifiability_single_release_min_indep)
        or identifiability_single_release_min_indep < 0.0
    ):
        identifiability_single_release_min_indep = 0.35
    identifiability_block_release_min_indep = float(
        identifiability_cfg.get("block_release_min_indep", 0.20)
    )
    if (
        not np.isfinite(identifiability_block_release_min_indep)
        or identifiability_block_release_min_indep < 0.0
    ):
        identifiability_block_release_min_indep = 0.20
    auto_freeze_enabled = bool(identifiability_cfg.get("auto_freeze", False))
    auto_freeze_condition_number = float(
        identifiability_cfg.get(
            "auto_freeze_condition_number",
            identifiability_condition_warn,
        )
    )
    if not np.isfinite(auto_freeze_condition_number) or auto_freeze_condition_number <= 1.0:
        auto_freeze_condition_number = identifiability_condition_warn
    auto_freeze_correlation = float(
        identifiability_cfg.get(
            "auto_freeze_correlation",
            max(identifiability_correlation_warn, 0.98),
        )
    )
    if not np.isfinite(auto_freeze_correlation) or auto_freeze_correlation < 0.0:
        auto_freeze_correlation = max(identifiability_correlation_warn, 0.98)
    auto_freeze_correlation = min(max(auto_freeze_correlation, 0.0), 0.999999)
    auto_freeze_max_parameters = int(identifiability_cfg.get("auto_freeze_max_parameters", 2))
    auto_freeze_max_parameters = max(0, auto_freeze_max_parameters)
    auto_freeze_max_nfev = int(
        identifiability_cfg.get(
            "auto_freeze_max_nfev",
            max(20, min(60, solver_max_nfev)),
        )
    )
    auto_freeze_max_nfev = max(10, auto_freeze_max_nfev)
    auto_freeze_max_cost_increase_fraction = float(
        identifiability_cfg.get("auto_freeze_max_cost_increase_fraction", 0.0)
    )
    if not np.isfinite(auto_freeze_max_cost_increase_fraction):
        auto_freeze_max_cost_increase_fraction = 0.0
    auto_freeze_max_cost_increase_fraction = max(
        0.0,
        auto_freeze_max_cost_increase_fraction,
    )
    selective_thaw_cfg_raw = identifiability_cfg.get("selective_thaw", {})
    if isinstance(selective_thaw_cfg_raw, bool):
        selective_thaw_cfg: Dict[str, object] = {"enabled": bool(selective_thaw_cfg_raw)}
    elif isinstance(selective_thaw_cfg_raw, dict):
        selective_thaw_cfg = dict(selective_thaw_cfg_raw)
    else:
        selective_thaw_cfg = {}
    selective_thaw_enabled = bool(selective_thaw_cfg.get("enabled", False))
    selective_thaw_max_parameters = int(
        selective_thaw_cfg.get(
            "max_parameters",
            max(1, auto_freeze_max_parameters),
        )
    )
    selective_thaw_max_parameters = max(0, selective_thaw_max_parameters)
    selective_thaw_max_nfev = int(
        selective_thaw_cfg.get(
            "max_nfev",
            max(10, min(40, solver_max_nfev)),
        )
    )
    selective_thaw_max_nfev = max(5, selective_thaw_max_nfev)
    selective_thaw_condition_number = float(
        selective_thaw_cfg.get(
            "max_condition_number",
            identifiability_condition_warn,
        )
    )
    if not np.isfinite(selective_thaw_condition_number) or selective_thaw_condition_number <= 1.0:
        selective_thaw_condition_number = identifiability_condition_warn
    adaptive_regularization_cfg_raw = identifiability_cfg.get(
        "adaptive_regularization",
        {},
    )
    if isinstance(adaptive_regularization_cfg_raw, bool):
        adaptive_regularization_cfg: Dict[str, object] = {
            "enabled": bool(adaptive_regularization_cfg_raw)
        }
    elif isinstance(adaptive_regularization_cfg_raw, dict):
        adaptive_regularization_cfg = dict(adaptive_regularization_cfg_raw)
    else:
        adaptive_regularization_cfg = {}
    adaptive_regularization_enabled = bool(adaptive_regularization_cfg.get("enabled", False))
    adaptive_regularization_max_parameters = int(
        adaptive_regularization_cfg.get(
            "max_parameters",
            max(1, auto_freeze_max_parameters),
        )
    )
    adaptive_regularization_max_parameters = max(
        0,
        adaptive_regularization_max_parameters,
    )
    adaptive_regularization_max_nfev = int(
        adaptive_regularization_cfg.get(
            "max_nfev",
            max(10, min(30, solver_max_nfev)),
        )
    )
    adaptive_regularization_max_nfev = max(5, adaptive_regularization_max_nfev)
    adaptive_regularization_release_max_nfev = int(
        adaptive_regularization_cfg.get(
            "release_max_nfev",
            max(5, min(20, solver_max_nfev)),
        )
    )
    adaptive_regularization_release_max_nfev = max(
        3,
        adaptive_regularization_release_max_nfev,
    )
    adaptive_regularization_condition_number = float(
        adaptive_regularization_cfg.get(
            "condition_number_trigger",
            identifiability_condition_warn,
        )
    )
    if (
        not np.isfinite(adaptive_regularization_condition_number)
        or adaptive_regularization_condition_number <= 1.0
    ):
        adaptive_regularization_condition_number = identifiability_condition_warn
    adaptive_regularization_correlation = float(
        adaptive_regularization_cfg.get(
            "correlation_trigger",
            max(identifiability_correlation_warn, 0.98),
        )
    )
    if (
        not np.isfinite(adaptive_regularization_correlation)
        or adaptive_regularization_correlation < 0.0
    ):
        adaptive_regularization_correlation = max(
            identifiability_correlation_warn,
            0.98,
        )
    adaptive_regularization_correlation = min(
        max(adaptive_regularization_correlation, 0.0),
        0.999999,
    )
    adaptive_regularization_sigma_scale = float(adaptive_regularization_cfg.get("sigma_scale", 0.5))
    if (
        not np.isfinite(adaptive_regularization_sigma_scale)
        or adaptive_regularization_sigma_scale <= 0.0
    ):
        adaptive_regularization_sigma_scale = 0.5
    adaptive_regularization_min_sigma = float(adaptive_regularization_cfg.get("min_sigma", 0.05))
    if (
        not np.isfinite(adaptive_regularization_min_sigma)
        or adaptive_regularization_min_sigma <= 0.0
    ):
        adaptive_regularization_min_sigma = 0.05
    adaptive_regularization_max_cost_increase_fraction = float(
        adaptive_regularization_cfg.get("max_cost_increase_fraction", 0.02)
    )
    if not np.isfinite(adaptive_regularization_max_cost_increase_fraction):
        adaptive_regularization_max_cost_increase_fraction = 0.02
    adaptive_regularization_max_cost_increase_fraction = max(
        0.0,
        adaptive_regularization_max_cost_increase_fraction,
    )
    anisotropic_uncertainty_enabled = bool(
        solver_cfg.get("anisotropic_measurement_uncertainty", False)
    )
    radial_sigma_scale = float(solver_cfg.get("radial_sigma_scale", 1.0))
    tangential_sigma_scale = float(solver_cfg.get("tangential_sigma_scale", 1.0))
    if not np.isfinite(radial_sigma_scale) or radial_sigma_scale <= 0.0:
        radial_sigma_scale = 1.0
    if not np.isfinite(tangential_sigma_scale) or tangential_sigma_scale <= 0.0:
        tangential_sigma_scale = 1.0

    try:
        center_seed = params.get("center", (0.0, 0.0))
        center_row_default = float(center_seed[0])
        center_col_default = float(center_seed[1])
    except Exception:
        center_row_default = 0.0
        center_col_default = 0.0
    if not np.isfinite(center_row_default):
        center_row_default = 0.0
    if not np.isfinite(center_col_default):
        center_col_default = 0.0

    params = dict(params)
    full_beam_mosaic_params = copy.deepcopy(dict(params.get("mosaic_params", {}) or {}))
    params["center"] = [center_row_default, center_col_default]
    params.setdefault("center_x", center_row_default)
    params.setdefault("center_y", center_col_default)
    params.setdefault("theta_offset", 0.0)

    dataset_contexts = _build_geometry_fit_dataset_contexts(
        miller,
        intensities,
        params,
        measured_peaks,
        experimental_image,
        dataset_specs=dataset_spec_entries,
    )
    multi_dataset_mode = ("theta_offset" in var_names) and len(dataset_contexts) > 0
    dataset_parallel_workers = 1
    restart_parallel_workers = 1
    worker_numba_threads: Optional[int] = None
    configured_parallel_workers = _resolve_parallel_worker_count(
        solver_workers_cfg,
        max_tasks=max(len(dataset_contexts), solver_restarts, 1),
    )
    if configured_parallel_workers > 1 and solver_parallel_mode != "off":
        if solver_parallel_mode in {"auto", "datasets"} and len(dataset_contexts) > 1:
            dataset_parallel_workers = min(
                configured_parallel_workers,
                len(dataset_contexts),
            )
        elif solver_parallel_mode in {"auto", "restarts"} and solver_restarts > 1:
            restart_parallel_workers = min(
                configured_parallel_workers,
                solver_restarts,
            )
    active_outer_workers = max(dataset_parallel_workers, restart_parallel_workers)
    worker_numba_threads = _resolve_numba_threads_per_worker(
        active_outer_workers,
        solver_worker_numba_threads_cfg,
    )
    parallelization_summary = {
        "mode": str(solver_parallel_mode),
        "configured_workers": int(configured_parallel_workers),
        "dataset_workers": int(dataset_parallel_workers),
        "restart_workers": int(restart_parallel_workers),
        "worker_numba_threads": (
            None if worker_numba_threads is None else int(worker_numba_threads)
        ),
        "numba_thread_budget": int(_available_parallel_thread_budget()),
    }

    def _safe_float(value: object, fallback: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        if not np.isfinite(out):
            return float(fallback)
        return out

    base_mosaic_params = (
        copy.deepcopy(dict(params.get("mosaic_params", {})))
        if isinstance(params.get("mosaic_params", {}), Mapping)
        else {}
    )
    base_wave_key: str | None = None
    base_wavelength_template: np.ndarray | None = None
    for wave_key in ("wavelength_array", "wavelength_i_array"):
        wave_raw = base_mosaic_params.get(wave_key)
        if wave_raw is None:
            continue
        try:
            wave_arr = np.asarray(wave_raw, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        wave_arr = wave_arr[np.isfinite(wave_arr) & (wave_arr > 0.0)]
        if wave_arr.size <= 0:
            continue
        base_wave_key = str(wave_key)
        base_wavelength_template = np.asarray(wave_arr, dtype=np.float64)
        break
    base_lambda_scalar = _wavelength_scalar_from_local(params)

    def _apply_trial_params(
        x: Sequence[float],
        *,
        names: Optional[Sequence[str]] = None,
        base_params: Optional[Mapping[str, object]] = None,
    ) -> Dict[str, object]:
        local = params.copy()
        if isinstance(base_params, Mapping):
            local.update(dict(base_params))
        center_pair = local.get("center", [center_row_default, center_col_default])
        try:
            center_row = _safe_float(center_pair[0], center_row_default)
            center_col = _safe_float(center_pair[1], center_col_default)
        except Exception:
            center_row = center_row_default
            center_col = center_col_default

        center_row = _safe_float(local.get("center_x", center_row), center_row)
        center_col = _safe_float(local.get("center_y", center_col), center_col)

        active_names = [str(name) for name in (names if names is not None else var_names)]
        for name, v in zip(active_names, x):
            val = float(v)
            if name == "center_x":
                center_row = val
                local["center_x"] = val
            elif name == "center_y":
                center_col = val
                local["center_y"] = val
            else:
                local[name] = val

        local_mosaic = (
            copy.deepcopy(dict(local.get("mosaic_params", {})))
            if isinstance(local.get("mosaic_params", {}), Mapping)
            else {}
        )
        lambda_trial = _safe_float(local.get("lambda", base_lambda_scalar), base_lambda_scalar)
        if np.isfinite(lambda_trial) and lambda_trial > 0.0:
            if (
                isinstance(base_wavelength_template, np.ndarray)
                and base_wavelength_template.size > 0
                and np.isfinite(base_lambda_scalar)
                and base_lambda_scalar > 0.0
            ):
                scaled_wavelength = np.asarray(base_wavelength_template, dtype=np.float64) * (
                    float(lambda_trial) / float(base_lambda_scalar)
                )
                local_mosaic["wavelength_array"] = np.asarray(
                    scaled_wavelength,
                    dtype=np.float64,
                )
                if base_wave_key == "wavelength_i_array":
                    local_mosaic["wavelength_i_array"] = np.asarray(
                        scaled_wavelength,
                        dtype=np.float64,
                    )
            elif "beam_x_array" in local_mosaic:
                try:
                    beam_count = int(
                        np.asarray(local_mosaic.get("beam_x_array"), dtype=np.float64).size
                    )
                except Exception:
                    beam_count = 0
                if beam_count > 0:
                    local_mosaic["wavelength_array"] = np.full(
                        beam_count,
                        float(lambda_trial),
                        dtype=np.float64,
                    )
        local["mosaic_params"] = local_mosaic
        local["center"] = [float(center_row), float(center_col)]
        local["center_x"] = float(center_row)
        local["center_y"] = float(center_col)
        local["_q_group_line_constraints_enabled"] = bool(q_group_line_constraints_enabled)
        local["_q_group_line_angle_weight"] = float(q_group_line_angle_weight)
        local["_q_group_line_offset_weight"] = float(q_group_line_offset_weight)
        local["_q_group_line_missing_penalty_scale"] = float(q_group_line_missing_penalty_scale)
        local["_hk0_peak_priority_weight"] = float(hk0_peak_priority_weight)
        return local

    def _build_point_matches(
        simulated_points: Sequence[Tuple[float, float]],
        measured_points: Sequence[Tuple[float, float]],
        *,
        max_distance: float = np.inf,
    ) -> List[Tuple[np.ndarray, np.ndarray, float, int, int]]:
        """Backward-compatible wrapper around the global point matcher."""

        return _build_global_point_matches(
            simulated_points,
            measured_points,
            max_distance=max_distance,
        )

    def _legacy_cost_fn_unused(x):
        """Retained only as an inert compatibility shim for old notebooks."""

        return np.asarray(cost_fn(x), dtype=float)

    def _legacy_evaluate_pixel_matches_unused(
        local: Dict[str, object],
        *,
        collect_diagnostics: bool = False,
    ) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
        """Legacy shim. Real geometry point matching lives in `_evaluate_pixel_matches`."""

        del local, collect_diagnostics
        return (
            np.array([], dtype=float),
            [],
            {
                "status": "legacy_pixel_match_path_retired",
            },
        )

    def _theta_initial_for_dataset(
        local: Dict[str, object],
        dataset_ctx: GeometryFitDatasetContext,
    ) -> float:
        theta_base = _safe_float(dataset_ctx.theta_initial, 0.0)
        if multi_dataset_mode:
            theta_offset = _safe_float(local.get("theta_offset", 0.0), 0.0)
            return float(theta_base + theta_offset)
        return _safe_float(local.get("theta_initial", theta_base), theta_base)

    def _cost_fn_for_dataset(
        item: Tuple[Dict[str, object], GeometryFitDatasetContext],
    ) -> np.ndarray:
        local, dataset_ctx = item
        theta_value = _theta_initial_for_dataset(local, dataset_ctx)
        args = [
            local["gamma"],
            local["Gamma"],
            local["corto_detector"],
            theta_value,
            local.get("cor_angle", 0.0),
            local["zs"],
            local["zb"],
            local["chi"],
            local["a"],
            local["c"],
            local["center"][0],
            local["center"][1],
        ]
        residual = compute_peak_position_error_geometry_local(
            *args,
            measured_peaks=dataset_ctx.subset.measured_entries,
            miller=dataset_ctx.subset.miller,
            intensities=dataset_ctx.subset.intensities,
            image_size=image_size,
            mosaic_params=local["mosaic_params"],
            n2=local["n2"],
            psi=local.get("psi", 0.0),
            psi_z=local.get("psi_z", 0.0),
            debye_x=local["debye_x"],
            debye_y=local["debye_y"],
            wavelength=local["lambda"],
            pixel_tol=pixel_tol,
            optics_mode=local.get("optics_mode", 0),
        )
        return np.asarray(residual, dtype=float)

    def cost_fn(x):
        local = _apply_trial_params(x)
        residual_blocks: List[np.ndarray] = []
        dataset_items = [(local, dataset_ctx) for dataset_ctx in dataset_contexts]
        if dataset_parallel_workers > 1 and len(dataset_items) > 1:
            residual_results = _threaded_map(
                _cost_fn_for_dataset,
                dataset_items,
                max_workers=dataset_parallel_workers,
                numba_threads=worker_numba_threads,
            )
        else:
            residual_results = [_cost_fn_for_dataset(item) for item in dataset_items]
        for residual_arr in residual_results:
            if residual_arr.size:
                residual_blocks.append(residual_arr)
        prior_residual = _parameter_prior_residuals(x)
        if prior_residual.size:
            residual_blocks.append(np.asarray(prior_residual, dtype=float))
        return np.concatenate(residual_blocks) if residual_blocks else np.array([], dtype=float)

    def _evaluate_pixel_matches(
        local: Dict[str, object],
        *,
        collect_diagnostics: bool = False,
    ) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
        if not dataset_contexts:
            return (
                np.array([], dtype=float),
                [],
                {
                    "dataset_count": 0,
                    "measured_count": 0,
                    "fixed_source_resolved_count": 0,
                    "fallback_entry_count": 0,
                    "missing_pair_count": 0,
                    "simulated_reflection_count": 0,
                    "total_reflection_count": 0,
                    "fixed_source_reflection_count": 0,
                    "subset_fallback_hkl_count": 0,
                    "subset_reduced": False,
                    "central_ray_mode": False,
                    "single_ray_enabled": bool(use_single_ray),
                    "single_ray_forced_count": 0,
                },
            )

        residual_blocks: List[np.ndarray] = []
        diagnostics: List[Dict[str, object]] = []
        per_dataset_summaries: List[Dict[str, object]] = []
        summary: Dict[str, object] = {
            "dataset_count": int(len(dataset_contexts)),
            "measured_count": 0,
            "fixed_source_resolved_count": 0,
            "fallback_entry_count": 0,
            "matched_pair_count": 0,
            "missing_pair_count": 0,
            "simulated_reflection_count": 0,
            "total_reflection_count": 0,
            "fixed_source_reflection_count": 0,
            "subset_fallback_hkl_count": 0,
            "subset_reduced": False,
            "central_ray_mode": False,
            "single_ray_enabled": bool(use_single_ray),
            "single_ray_forced_count": 0,
            "center_row": float(local["center"][0])
            if len(local.get("center", [])) >= 2
            else float("nan"),
            "center_col": float(local["center"][1])
            if len(local.get("center", [])) >= 2
            else float("nan"),
            "line_group_count": 0,
            "resolved_line_group_count": 0,
            "missing_line_group_count": 0,
            "line_angle_rms_px": float("nan"),
            "line_offset_rms_px": float("nan"),
            "line_fit_space_span_deg_mean": float("nan"),
            "line_constraints_enabled": bool(q_group_line_constraints_enabled),
        }

        def _evaluate_pixel_matches_for_dataset(
            item: Tuple[Dict[str, object], GeometryFitDatasetContext],
        ) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
            local_item, dataset_ctx = item
            theta_value = _theta_initial_for_dataset(local_item, dataset_ctx)
            return _evaluate_geometry_fit_dataset_point_matches(
                local_item,
                dataset_ctx,
                image_size=image_size,
                pixel_tol=float(pixel_tol),
                weighted_matching=bool(weighted_matching),
                solver_f_scale=float(solver_f_scale),
                missing_pair_penalty=float(missing_pair_penalty),
                use_single_ray=bool(use_single_ray),
                theta_value=float(theta_value),
                use_measurement_uncertainty=bool(use_measurement_uncertainty),
                anisotropic_uncertainty=bool(anisotropic_uncertainty_enabled),
                radial_sigma_scale=float(radial_sigma_scale),
                tangential_sigma_scale=float(tangential_sigma_scale),
                collect_diagnostics=collect_diagnostics,
            )

        dataset_items = [(local, dataset_ctx) for dataset_ctx in dataset_contexts]
        if dataset_parallel_workers > 1 and len(dataset_items) > 1:
            dataset_results = _threaded_map(
                _evaluate_pixel_matches_for_dataset,
                dataset_items,
                max_workers=dataset_parallel_workers,
                numba_threads=worker_numba_threads,
            )
        else:
            dataset_results = [_evaluate_pixel_matches_for_dataset(item) for item in dataset_items]

        for residual_i, diagnostics_i, summary_i in dataset_results:
            residual_i = np.asarray(residual_i, dtype=float)
            if residual_i.size:
                residual_blocks.append(residual_i)
            per_dataset_summaries.append(dict(summary_i))
            summary["matched_pair_count"] = int(summary.get("matched_pair_count", 0)) + int(
                summary_i.get("matched_pair_count", 0)
            )
            for key in (
                "measured_count",
                "fixed_source_resolved_count",
                "fallback_entry_count",
                "missing_pair_count",
                "simulated_reflection_count",
                "total_reflection_count",
                "fixed_source_reflection_count",
                "subset_fallback_hkl_count",
                "single_ray_forced_count",
                "line_group_count",
                "resolved_line_group_count",
                "missing_line_group_count",
            ):
                summary[key] = int(summary.get(key, 0)) + int(summary_i.get(key, 0))
            summary["subset_reduced"] = bool(
                summary.get("subset_reduced", False) or bool(summary_i.get("subset_reduced", False))
            )
            summary["central_ray_mode"] = bool(
                summary.get("central_ray_mode", False)
                or bool(summary_i.get("central_ray_mode", False))
            )
            summary["line_constraints_enabled"] = bool(
                summary.get("line_constraints_enabled", False)
                or bool(summary_i.get("line_constraints_enabled", False))
            )
            if collect_diagnostics and diagnostics_i:
                diagnostics.extend(diagnostics_i)

        residual_arr = (
            np.concatenate(residual_blocks) if residual_blocks else np.array([], dtype=float)
        )
        summary["per_dataset"] = per_dataset_summaries

        custom_sigma_values = np.asarray(
            [
                float(entry.get("measurement_sigma_px", np.nan))
                for entry in diagnostics
                if bool(entry.get("sigma_is_custom", False))
            ],
            dtype=float,
        )
        custom_sigma_values = custom_sigma_values[
            np.isfinite(custom_sigma_values) & (custom_sigma_values > 0.0)
        ]

        matched_distances = np.asarray(
            [
                float(entry.get("distance_px", np.nan))
                for entry in diagnostics
                if str(entry.get("match_status", "")).lower() == "matched"
            ],
            dtype=float,
        )
        matched_distances = matched_distances[np.isfinite(matched_distances)]
        summary["custom_sigma_count"] = int(custom_sigma_values.size)
        anisotropic_count = int(
            sum(
                int(summary_i.get("anisotropic_sigma_count", 0))
                for summary_i in per_dataset_summaries
            )
        )
        summary["anisotropic_sigma_count"] = int(anisotropic_count)
        if anisotropic_count > 0:
            summary["peak_weighting_mode"] = (
                "measurement_covariance+distance"
                if bool(weighted_matching)
                else "measurement_covariance"
            )
        elif custom_sigma_values.size:
            summary["peak_weighting_mode"] = (
                "measurement_sigma+distance" if bool(weighted_matching) else "measurement_sigma"
            )
        else:
            summary["peak_weighting_mode"] = "uniform"
        if custom_sigma_values.size:
            summary["measurement_sigma_median_px"] = float(np.median(custom_sigma_values))
            summary["measurement_sigma_mean_px"] = float(np.mean(custom_sigma_values))
            summary["measurement_sigma_max_px"] = float(np.max(custom_sigma_values))
        if matched_distances.size:
            summary["unweighted_peak_rms_px"] = float(
                np.sqrt(np.mean(matched_distances * matched_distances))
            )
            summary["unweighted_peak_mean_px"] = float(np.mean(matched_distances))
            summary["unweighted_peak_max_px"] = float(np.max(matched_distances))
        else:
            matched_sq_sum = 0.0
            matched_sum = 0.0
            matched_max = float("nan")
            matched_count = int(summary.get("matched_pair_count", 0))
            has_sq_stat = False
            has_mean_stat = False
            for summary_i in per_dataset_summaries:
                try:
                    dataset_rms = float(summary_i.get("unweighted_peak_rms_px", np.nan))
                except Exception:
                    dataset_rms = float("nan")
                try:
                    dataset_mean = float(summary_i.get("unweighted_peak_mean_px", np.nan))
                except Exception:
                    dataset_mean = float("nan")
                try:
                    dataset_max = float(summary_i.get("unweighted_peak_max_px", np.nan))
                except Exception:
                    dataset_max = float("nan")
                dataset_count = int(summary_i.get("matched_pair_count", 0))
                if dataset_count <= 0:
                    continue
                if np.isfinite(dataset_rms):
                    matched_sq_sum += float(dataset_count) * float(dataset_rms) * float(dataset_rms)
                    has_sq_stat = True
                if np.isfinite(dataset_mean):
                    matched_sum += float(dataset_count) * float(dataset_mean)
                    has_mean_stat = True
                if np.isfinite(dataset_max):
                    if not np.isfinite(matched_max) or dataset_max > matched_max:
                        matched_max = float(dataset_max)
            if matched_count > 0 and has_sq_stat:
                summary["unweighted_peak_rms_px"] = float(
                    np.sqrt(matched_sq_sum / float(matched_count))
                )
            else:
                summary["unweighted_peak_rms_px"] = float("nan")
            if matched_count > 0 and has_mean_stat:
                summary["unweighted_peak_mean_px"] = float(matched_sum / float(matched_count))
            else:
                summary["unweighted_peak_mean_px"] = float("nan")
            summary["unweighted_peak_max_px"] = float(matched_max)

        line_resolved_total = int(summary.get("resolved_line_group_count", 0))
        for rms_key, count_key in (
            ("line_angle_rms_px", "resolved_line_group_count"),
            ("line_offset_rms_px", "resolved_line_group_count"),
        ):
            weighted_sq_sum = 0.0
            has_sq_stat = False
            for summary_i in per_dataset_summaries:
                try:
                    dataset_rms = float(summary_i.get(rms_key, np.nan))
                except Exception:
                    dataset_rms = float("nan")
                dataset_count = int(summary_i.get(count_key, 0))
                if dataset_count <= 0 or not np.isfinite(dataset_rms):
                    continue
                weighted_sq_sum += float(dataset_count) * float(dataset_rms) * float(dataset_rms)
                has_sq_stat = True
            if line_resolved_total > 0 and has_sq_stat:
                summary[rms_key] = float(np.sqrt(weighted_sq_sum / float(line_resolved_total)))
            else:
                summary[rms_key] = float("nan")

        line_span_sum = 0.0
        line_span_count = 0
        for summary_i in per_dataset_summaries:
            try:
                dataset_span = float(summary_i.get("line_fit_space_span_deg_mean", np.nan))
            except Exception:
                dataset_span = float("nan")
            dataset_count = int(summary_i.get("resolved_line_group_count", 0))
            if dataset_count <= 0 or not np.isfinite(dataset_span):
                continue
            line_span_sum += float(dataset_count) * float(dataset_span)
            line_span_count += int(dataset_count)
        if line_span_count > 0:
            summary["line_fit_space_span_deg_mean"] = float(line_span_sum / float(line_span_count))
        else:
            summary["line_fit_space_span_deg_mean"] = float("nan")
        return residual_arr, diagnostics, summary

    def _evaluate_dynamic_point_matches(
        local: Dict[str, object],
        *,
        collect_diagnostics: bool = False,
    ) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
        fit_space_summary_defaults = _fit_space_provenance_summary(local)
        if not dataset_contexts:
            return (
                np.array([], dtype=float),
                [],
                {
                    "dataset_count": 0,
                    "measured_count": 0,
                    "fixed_source_resolved_count": 0,
                    "fallback_entry_count": 0,
                    "matched_pair_count": 0,
                    "missing_pair_count": 0,
                    "simulated_reflection_count": 0,
                    "total_reflection_count": 0,
                    "fixed_source_reflection_count": 0,
                    "subset_fallback_hkl_count": 0,
                    "subset_reduced": False,
                    "central_ray_mode": False,
                    "single_ray_enabled": False,
                    "single_ray_forced_count": 0,
                    "peak_weighting_mode": "uniform",
                    "line_group_count": 0,
                    "resolved_line_group_count": 0,
                    "missing_line_group_count": 0,
                    "line_angle_rms_px": float("nan"),
                    "line_offset_rms_px": float("nan"),
                    "line_angle_rms_deg": float("nan"),
                    "line_offset_rms_deg": float("nan"),
                    "line_fit_space_span_deg_mean": float("nan"),
                    "line_constraints_enabled": bool(q_group_line_constraints_enabled),
                    "_live_cache_records": [],
                    **fit_space_summary_defaults,
                },
            )

        residual_blocks: List[np.ndarray] = []
        diagnostics: List[Dict[str, object]] = []
        per_dataset_summaries: List[Dict[str, object]] = []
        summary: Dict[str, object] = {
            "dataset_count": int(len(dataset_contexts)),
            "measured_count": 0,
            "fixed_source_resolved_count": 0,
            "fallback_entry_count": 0,
            "matched_pair_count": 0,
            "missing_pair_count": 0,
            "simulated_reflection_count": 0,
            "total_reflection_count": 0,
            "fixed_source_reflection_count": 0,
            "subset_fallback_hkl_count": 0,
            "subset_reduced": False,
            "central_ray_mode": False,
            "single_ray_enabled": False,
            "single_ray_forced_count": 0,
            "center_row": float(local["center"][0])
            if len(local.get("center", [])) >= 2
            else float("nan"),
            "center_col": float(local["center"][1])
            if len(local.get("center", [])) >= 2
            else float("nan"),
            "peak_weighting_mode": "uniform",
            "custom_sigma_count": 0,
            "anisotropic_sigma_count": 0,
            "line_group_count": 0,
            "resolved_line_group_count": 0,
            "missing_line_group_count": 0,
            "line_angle_rms_px": float("nan"),
            "line_offset_rms_px": float("nan"),
            "line_angle_rms_deg": float("nan"),
            "line_offset_rms_deg": float("nan"),
            "line_fit_space_span_deg_mean": float("nan"),
            "line_constraints_enabled": bool(q_group_line_constraints_enabled),
            **fit_space_summary_defaults,
        }
        live_cache_records: List[Dict[str, object]] = []

        def _evaluate_dynamic_matches_for_dataset(
            item: Tuple[Dict[str, object], GeometryFitDatasetContext],
        ) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
            local_item, dataset_ctx = item
            theta_value = _theta_initial_for_dataset(local_item, dataset_ctx)
            return _evaluate_geometry_fit_dataset_dynamic_point_matches(
                local_item,
                dataset_ctx,
                image_size=image_size,
                missing_pair_penalty_deg=float(missing_pair_penalty_deg),
                theta_value=float(theta_value),
                collect_diagnostics=collect_diagnostics,
            )

        dataset_items = [(local, dataset_ctx) for dataset_ctx in dataset_contexts]
        if dataset_parallel_workers > 1 and len(dataset_items) > 1:
            dataset_results = _threaded_map(
                _evaluate_dynamic_matches_for_dataset,
                dataset_items,
                max_workers=dataset_parallel_workers,
                numba_threads=worker_numba_threads,
            )
        else:
            dataset_results = [
                _evaluate_dynamic_matches_for_dataset(item) for item in dataset_items
            ]

        for residual_i, diagnostics_i, summary_i in dataset_results:
            residual_i = np.asarray(residual_i, dtype=float)
            if residual_i.size:
                residual_blocks.append(residual_i)
            per_dataset_summaries.append(dict(summary_i))
            for key in (
                "measured_count",
                "fixed_source_resolved_count",
                "fallback_entry_count",
                "matched_pair_count",
                "missing_pair_count",
                "simulated_reflection_count",
                "total_reflection_count",
                "fixed_source_reflection_count",
                "subset_fallback_hkl_count",
                "single_ray_forced_count",
                "line_group_count",
                "resolved_line_group_count",
                "missing_line_group_count",
            ):
                summary[key] = int(summary.get(key, 0)) + int(summary_i.get(key, 0))
            summary["subset_reduced"] = bool(
                summary.get("subset_reduced", False) or bool(summary_i.get("subset_reduced", False))
            )
            summary["central_ray_mode"] = bool(
                summary.get("central_ray_mode", False)
                or bool(summary_i.get("central_ray_mode", False))
            )
            summary["line_constraints_enabled"] = bool(
                summary.get("line_constraints_enabled", False)
                or bool(summary_i.get("line_constraints_enabled", False))
            )
            for record in summary_i.get("_live_cache_records", ()) or ():
                if isinstance(record, Mapping):
                    live_cache_records.append(dict(record))
            if collect_diagnostics and diagnostics_i:
                diagnostics.extend(diagnostics_i)

        residual_arr = (
            np.concatenate(residual_blocks) if residual_blocks else np.array([], dtype=float)
        )
        summary["per_dataset"] = per_dataset_summaries

        for prefix, diag_key in (
            ("px", "distance_px"),
            ("deg", "angular_distance_deg"),
        ):
            rms_key = f"unweighted_peak_rms_{prefix}"
            mean_key = f"unweighted_peak_mean_{prefix}"
            max_key = f"unweighted_peak_max_{prefix}"
            values = np.asarray(
                [
                    float(entry.get(diag_key, np.nan))
                    for entry in diagnostics
                    if str(entry.get("match_status", "")).lower() == "matched"
                ],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            if values.size:
                summary[rms_key] = float(np.sqrt(np.mean(values * values)))
                summary[mean_key] = float(np.mean(values))
                summary[max_key] = float(np.max(values))
                continue

            matched_sq_sum = 0.0
            matched_sum = 0.0
            matched_max = float("nan")
            matched_count = int(summary.get("matched_pair_count", 0))
            has_sq_stat = False
            has_mean_stat = False
            for summary_i in per_dataset_summaries:
                try:
                    dataset_rms = float(summary_i.get(rms_key, np.nan))
                except Exception:
                    dataset_rms = float("nan")
                try:
                    dataset_mean = float(summary_i.get(mean_key, np.nan))
                except Exception:
                    dataset_mean = float("nan")
                try:
                    dataset_max = float(summary_i.get(max_key, np.nan))
                except Exception:
                    dataset_max = float("nan")
                dataset_count = int(summary_i.get("matched_pair_count", 0))
                if dataset_count <= 0:
                    continue
                if np.isfinite(dataset_rms):
                    matched_sq_sum += float(dataset_count) * float(dataset_rms) * float(dataset_rms)
                    has_sq_stat = True
                if np.isfinite(dataset_mean):
                    matched_sum += float(dataset_count) * float(dataset_mean)
                    has_mean_stat = True
                if np.isfinite(dataset_max):
                    if not np.isfinite(matched_max) or dataset_max > matched_max:
                        matched_max = float(dataset_max)
            if matched_count > 0 and has_sq_stat:
                summary[rms_key] = float(np.sqrt(matched_sq_sum / float(matched_count)))
            else:
                summary[rms_key] = float("nan")
            if matched_count > 0 and has_mean_stat:
                summary[mean_key] = float(matched_sum / float(matched_count))
            else:
                summary[mean_key] = float("nan")
            summary[max_key] = float(matched_max)
        line_resolved_total = int(summary.get("resolved_line_group_count", 0))
        for rms_key in (
            "line_angle_rms_px",
            "line_offset_rms_px",
            "line_angle_rms_deg",
            "line_offset_rms_deg",
        ):
            weighted_sq_sum = 0.0
            has_sq_stat = False
            for summary_i in per_dataset_summaries:
                try:
                    dataset_rms = float(summary_i.get(rms_key, np.nan))
                except Exception:
                    dataset_rms = float("nan")
                dataset_count = int(summary_i.get("resolved_line_group_count", 0))
                if dataset_count <= 0 or not np.isfinite(dataset_rms):
                    continue
                weighted_sq_sum += float(dataset_count) * float(dataset_rms) * float(dataset_rms)
                has_sq_stat = True
            if line_resolved_total > 0 and has_sq_stat:
                summary[rms_key] = float(np.sqrt(weighted_sq_sum / float(line_resolved_total)))
            else:
                summary[rms_key] = float("nan")
        line_span_sum = 0.0
        line_span_count = 0
        for summary_i in per_dataset_summaries:
            try:
                dataset_span = float(summary_i.get("line_fit_space_span_deg_mean", np.nan))
            except Exception:
                dataset_span = float("nan")
            dataset_count = int(summary_i.get("resolved_line_group_count", 0))
            if dataset_count <= 0 or not np.isfinite(dataset_span):
                continue
            line_span_sum += float(dataset_count) * float(dataset_span)
            line_span_count += int(dataset_count)
        if line_span_count > 0:
            summary["line_fit_space_span_deg_mean"] = float(line_span_sum / float(line_span_count))
        else:
            summary["line_fit_space_span_deg_mean"] = float("nan")
        fit_space_anchor_count_cached = int(
            sum(
                int(summary_i.get("fit_space_anchor_count_cached", 0))
                for summary_i in per_dataset_summaries
            )
        )
        fit_space_anchor_count_detector = int(
            sum(
                int(summary_i.get("fit_space_anchor_count_detector", 0))
                for summary_i in per_dataset_summaries
            )
        )
        fit_space_two_theta_adjustment_count = int(
            sum(
                int(summary_i.get("fit_space_two_theta_adjustment_count", 0))
                for summary_i in per_dataset_summaries
            )
        )
        fit_space_two_theta_adjustment_total_abs_deg = float(
            sum(
                float(summary_i.get("fit_space_two_theta_adjustment_total_abs_deg", 0.0))
                for summary_i in per_dataset_summaries
            )
        )
        fit_space_two_theta_adjustment_max_abs_deg = float("nan")
        for summary_i in per_dataset_summaries:
            try:
                dataset_max = float(
                    summary_i.get("fit_space_two_theta_adjustment_max_abs_deg", np.nan)
                )
            except Exception:
                dataset_max = float("nan")
            if not np.isfinite(dataset_max):
                continue
            if (
                not np.isfinite(fit_space_two_theta_adjustment_max_abs_deg)
                or dataset_max > fit_space_two_theta_adjustment_max_abs_deg
            ):
                fit_space_two_theta_adjustment_max_abs_deg = float(dataset_max)
        summary["fit_space_anchor_count_cached"] = int(fit_space_anchor_count_cached)
        summary["fit_space_anchor_count_detector"] = int(fit_space_anchor_count_detector)
        summary["fit_space_anchor_source_counts"] = {
            "cached_fit_space_anchor": int(fit_space_anchor_count_cached),
            "detector_fit_space_anchor": int(fit_space_anchor_count_detector),
        }
        summary["fit_space_two_theta_adjustment_count"] = int(fit_space_two_theta_adjustment_count)
        summary["fit_space_two_theta_adjustment_total_abs_deg"] = float(
            fit_space_two_theta_adjustment_total_abs_deg
        )
        summary["fit_space_two_theta_adjustment_mean_abs_deg"] = (
            float(
                fit_space_two_theta_adjustment_total_abs_deg
                / float(fit_space_two_theta_adjustment_count)
            )
            if fit_space_two_theta_adjustment_count > 0
            else float("nan")
        )
        summary["fit_space_two_theta_adjustment_max_abs_deg"] = float(
            fit_space_two_theta_adjustment_max_abs_deg
        )
        summary["_live_cache_records"] = live_cache_records
        return residual_arr, diagnostics, summary

    fixed_correspondence_groups: Dict[int, List[Dict[str, object]]] = {}

    def _evaluate_fixed_correspondence_matches(
        local: Dict[str, object],
        *,
        collect_diagnostics: bool = False,
    ) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
        if not dataset_contexts:
            return (
                np.array([], dtype=float),
                [],
                {
                    "dataset_count": 0,
                    "measured_count": 0,
                    "fixed_source_resolved_count": 0,
                    "fallback_entry_count": 0,
                    "matched_pair_count": 0,
                    "missing_pair_count": 0,
                    "simulated_reflection_count": 0,
                    "total_reflection_count": 0,
                    "fixed_source_reflection_count": 0,
                    "subset_fallback_hkl_count": 0,
                    "subset_reduced": False,
                    "central_ray_mode": False,
                    "single_ray_enabled": False,
                    "single_ray_forced_count": 0,
                    "match_radius_px": float(full_beam_polish_match_radius_px),
                },
            )

        residual_blocks: List[np.ndarray] = []
        diagnostics: List[Dict[str, object]] = []
        per_dataset_summaries: List[Dict[str, object]] = []
        summary: Dict[str, object] = {
            "dataset_count": int(len(dataset_contexts)),
            "measured_count": 0,
            "fixed_source_resolved_count": 0,
            "fallback_entry_count": 0,
            "matched_pair_count": 0,
            "missing_pair_count": 0,
            "simulated_reflection_count": 0,
            "total_reflection_count": 0,
            "fixed_source_reflection_count": 0,
            "subset_fallback_hkl_count": 0,
            "subset_reduced": False,
            "central_ray_mode": False,
            "single_ray_enabled": False,
            "single_ray_forced_count": 0,
            "center_row": float(local["center"][0])
            if len(local.get("center", [])) >= 2
            else float("nan"),
            "center_col": float(local["center"][1])
            if len(local.get("center", [])) >= 2
            else float("nan"),
            "match_radius_px": float(full_beam_polish_match_radius_px),
        }

        for dataset_ctx in dataset_contexts:
            theta_value = _theta_initial_for_dataset(local, dataset_ctx)
            residual_i, diagnostics_i, summary_i = (
                _evaluate_geometry_fit_dataset_fixed_correspondences(
                    local,
                    dataset_ctx,
                    fixed_correspondence_groups.get(int(dataset_ctx.dataset_index), []),
                    image_size=image_size,
                    weighted_matching=bool(weighted_matching),
                    solver_f_scale=float(solver_f_scale),
                    missing_pair_penalty=float(missing_pair_penalty),
                    theta_value=float(theta_value),
                    use_measurement_uncertainty=bool(use_measurement_uncertainty),
                    anisotropic_uncertainty=bool(anisotropic_uncertainty_enabled),
                    radial_sigma_scale=float(radial_sigma_scale),
                    tangential_sigma_scale=float(tangential_sigma_scale),
                    match_radius_px=float(full_beam_polish_match_radius_px),
                    collect_diagnostics=collect_diagnostics,
                )
            )
            residual_i = np.asarray(residual_i, dtype=float)
            if residual_i.size:
                residual_blocks.append(residual_i)
            per_dataset_summaries.append(dict(summary_i))
            for key in (
                "measured_count",
                "fixed_source_resolved_count",
                "fallback_entry_count",
                "matched_pair_count",
                "missing_pair_count",
                "simulated_reflection_count",
                "total_reflection_count",
                "fixed_source_reflection_count",
                "subset_fallback_hkl_count",
            ):
                summary[key] = int(summary.get(key, 0)) + int(summary_i.get(key, 0))
            summary["subset_reduced"] = bool(
                summary.get("subset_reduced", False) or bool(summary_i.get("subset_reduced", False))
            )
            if collect_diagnostics and diagnostics_i:
                diagnostics.extend(diagnostics_i)

        residual_arr = (
            np.concatenate(residual_blocks) if residual_blocks else np.array([], dtype=float)
        )
        summary["per_dataset"] = per_dataset_summaries

        custom_sigma_values = np.asarray(
            [
                float(entry.get("measurement_sigma_px", np.nan))
                for entry in diagnostics
                if bool(entry.get("sigma_is_custom", False))
            ],
            dtype=float,
        )
        custom_sigma_values = custom_sigma_values[
            np.isfinite(custom_sigma_values) & (custom_sigma_values > 0.0)
        ]
        matched_distances = np.asarray(
            [
                float(entry.get("distance_px", np.nan))
                for entry in diagnostics
                if str(entry.get("match_status", "")).lower() == "matched"
            ],
            dtype=float,
        )
        matched_distances = matched_distances[np.isfinite(matched_distances)]
        summary["custom_sigma_count"] = int(custom_sigma_values.size)
        anisotropic_count = int(
            sum(
                int(summary_i.get("anisotropic_sigma_count", 0))
                for summary_i in per_dataset_summaries
            )
        )
        summary["anisotropic_sigma_count"] = int(anisotropic_count)
        if anisotropic_count > 0:
            summary["peak_weighting_mode"] = (
                "measurement_covariance+distance"
                if bool(weighted_matching)
                else "measurement_covariance"
            )
        elif custom_sigma_values.size:
            summary["peak_weighting_mode"] = (
                "measurement_sigma+distance" if bool(weighted_matching) else "measurement_sigma"
            )
        else:
            summary["peak_weighting_mode"] = "uniform"
        if custom_sigma_values.size:
            summary["measurement_sigma_median_px"] = float(np.median(custom_sigma_values))
            summary["measurement_sigma_mean_px"] = float(np.mean(custom_sigma_values))
            summary["measurement_sigma_max_px"] = float(np.max(custom_sigma_values))
        if matched_distances.size:
            summary["unweighted_peak_rms_px"] = float(
                np.sqrt(np.mean(matched_distances * matched_distances))
            )
            summary["unweighted_peak_mean_px"] = float(np.mean(matched_distances))
            summary["unweighted_peak_max_px"] = float(np.max(matched_distances))
        else:
            summary["unweighted_peak_rms_px"] = float("nan")
            summary["unweighted_peak_mean_px"] = float("nan")
            summary["unweighted_peak_max_px"] = float("nan")
        return residual_arr, diagnostics, summary

    point_match_evaluator = (
        _evaluate_dynamic_point_matches if dynamic_point_geometry_fit else _evaluate_pixel_matches
    )

    def _public_point_match_summary(
        summary_in: Mapping[str, object] | None,
    ) -> Dict[str, object]:
        if not isinstance(summary_in, Mapping):
            return {}
        return {
            str(key): value
            for key, value in dict(summary_in).items()
            if not str(key).startswith("_")
        }

    last_live_update_payload: Dict[str, object] | None = None

    def _collect_seed_debug_summary(x_trial: Sequence[float]) -> Dict[str, object]:
        x_arr = np.asarray(x_trial, dtype=float).reshape(-1)
        local = _apply_trial_params(x_arr)
        point_match_summary_local: Optional[Dict[str, object]] = None
        if point_match_mode:
            residual_arr, _, point_match_summary = point_match_evaluator(
                local,
                collect_diagnostics=True,
            )
            point_match_summary_local = _public_point_match_summary(point_match_summary)
        else:
            residual_arr = np.asarray(cost_fn(x_arr), dtype=float)
        prior_residual = _parameter_prior_residuals(x_arr)
        if prior_residual.size:
            if np.asarray(residual_arr, dtype=float).size:
                residual_arr = np.concatenate(
                    [
                        np.asarray(residual_arr, dtype=float),
                        np.asarray(prior_residual, dtype=float),
                    ]
                )
            else:
                residual_arr = np.asarray(prior_residual, dtype=float)

        summary: Dict[str, object] = {
            "cost": float(
                _robust_cost(
                    np.asarray(residual_arr, dtype=float),
                    loss=solver_loss,
                    f_scale=solver_f_scale,
                )
            ),
            "weighted_rms_px": float(_weighted_rms_px(residual_arr)),
        }
        if point_match_summary_local is not None:
            summary["point_match_summary"] = point_match_summary_local
        return summary

    def pixel_cost_fn(x):
        nonlocal last_live_update_payload
        local = _apply_trial_params(x)
        residual_arr, _, point_match_summary = point_match_evaluator(
            local,
            collect_diagnostics=False,
        )
        live_cache_records = []
        if isinstance(point_match_summary, Mapping):
            for record in point_match_summary.get("_live_cache_records", ()) or ():
                if isinstance(record, Mapping):
                    live_cache_records.append(dict(record))
        last_live_update_payload = {
            "params": {
                str(key): value
                for key, value in local.items()
                if str(key)
                not in {
                    "mosaic_params",
                    "uv1",
                    "uv2",
                }
            },
            "point_match_summary": _public_point_match_summary(point_match_summary),
            "live_cache_records": live_cache_records,
            "dynamic_point_geometry_fit": bool(dynamic_point_geometry_fit),
            "manual_point_fit_mode": bool(manual_point_fit_mode),
        }
        prior_residual = _parameter_prior_residuals(x)
        if prior_residual.size:
            if residual_arr.size:
                return np.concatenate(
                    [
                        np.asarray(residual_arr, dtype=float),
                        np.asarray(prior_residual, dtype=float),
                    ]
                )
            return np.asarray(prior_residual, dtype=float)
        return residual_arr

    def _apply_full_beam_trial_params(x_trial: Sequence[float]) -> Dict[str, object]:
        local = _apply_trial_params(x_trial)
        local["mosaic_params"] = copy.deepcopy(full_beam_mosaic_params)
        return local

    def full_beam_cost_fn(x):
        local = _apply_full_beam_trial_params(x)
        residual_arr, _, _ = _evaluate_fixed_correspondence_matches(
            local,
            collect_diagnostics=False,
        )
        prior_residual = _parameter_prior_residuals(x)
        if prior_residual.size:
            if residual_arr.size:
                return np.concatenate(
                    [
                        np.asarray(residual_arr, dtype=float),
                        np.asarray(prior_residual, dtype=float),
                    ]
                )
            return np.asarray(prior_residual, dtype=float)
        return residual_arr

    def _initial_value(name: str) -> float:
        if name == "center_x":
            return _safe_float(
                params.get(
                    "center_x", params.get("center", [center_row_default, center_col_default])[0]
                ),
                center_row_default,
            )
        if name == "center_y":
            return _safe_float(
                params.get(
                    "center_y", params.get("center", [center_row_default, center_col_default])[1]
                ),
                center_col_default,
            )
        return _safe_float(params.get(name, 0.0), 0.0)

    def _vector_from_params(
        local_params: Mapping[str, object],
        *,
        names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        values: List[float] = []
        active_names = [str(name) for name in (names if names is not None else var_names)]
        for name in active_names:
            if name == "center_x":
                values.append(
                    _safe_float(
                        local_params.get(
                            "center_x",
                            local_params.get("center", [center_row_default, center_col_default])[0],
                        ),
                        center_row_default,
                    )
                )
                continue
            if name == "center_y":
                values.append(
                    _safe_float(
                        local_params.get(
                            "center_y",
                            local_params.get("center", [center_row_default, center_col_default])[1],
                        ),
                        center_col_default,
                    )
                )
                continue
            values.append(_safe_float(local_params.get(name, params.get(name, 0.0)), 0.0))
        return np.asarray(values, dtype=float)

    x0 = [_initial_value(name) for name in var_names]
    requested_x0_arr = np.asarray(x0, dtype=float).copy()

    lower_bounds = []
    upper_bounds = []
    bounds_cfg = {}
    x_scale_cfg = {}
    if isinstance(refinement_config, dict):
        bounds_cfg = refinement_config.get("bounds", {}) or {}
        x_scale_cfg = refinement_config.get("x_scale", {}) or {}

    def _cfg_bounds(name: str, current_val: float) -> tuple[float, float]:
        entry = bounds_cfg.get(name)
        if entry is None:
            if name == "center_x" or name == "center_y":
                hi = max(float(image_size) - 1.0, 0.0)
                return 0.0, hi
            return -np.inf, np.inf
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            return float(entry[0]), float(entry[1])
        if not isinstance(entry, dict):
            return -np.inf, np.inf

        mode = str(entry.get("mode", "absolute")).lower()
        min_raw = entry.get("min", None)
        max_raw = entry.get("max", None)

        if mode in {"relative", "rel"}:
            if not np.isfinite(current_val):
                return -np.inf, np.inf
            lo = current_val + float(min_raw) if min_raw is not None else -np.inf
            hi = current_val + float(max_raw) if max_raw is not None else np.inf
            return lo, hi
        if mode in {"relative_min0", "rel_min0"}:
            if not np.isfinite(current_val):
                return -np.inf, np.inf
            lo = current_val + float(min_raw) if min_raw is not None else -np.inf
            if np.isfinite(lo):
                lo = max(0.0, lo)
            hi = current_val + float(max_raw) if max_raw is not None else np.inf
            return lo, hi

        lo = float(min_raw) if min_raw is not None else -np.inf
        hi = float(max_raw) if max_raw is not None else np.inf
        return lo, hi

    for name, val in zip(var_names, x0):
        lo, hi = _cfg_bounds(name, float(val))
        lower_bounds.append(lo)
        upper_bounds.append(hi)

    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    x0_arr = np.asarray(x0, dtype=float)
    prior_centers = np.full(x0_arr.shape, np.nan, dtype=float)
    prior_sigmas = np.full(x0_arr.shape, np.nan, dtype=float)
    parameter_prior_summary: List[Dict[str, float]] = []
    for idx, name in enumerate(var_names):
        entry = prior_cfg.get(name)
        if entry is None:
            continue

        center = float(x0_arr[idx])
        sigma = float("nan")
        enabled = True
        if isinstance(entry, (int, float)):
            sigma = float(entry)
        elif isinstance(entry, dict):
            enabled = bool(entry.get("enabled", True))
            center = _safe_float(entry.get("center", center), center)
            sigma_value = entry.get("sigma", entry.get("stdev", entry.get("std")))
            try:
                sigma = float(sigma_value)
            except (TypeError, ValueError):
                sigma = float("nan")
        if not enabled or not np.isfinite(sigma) or sigma <= 0.0:
            continue

        prior_centers[idx] = float(center)
        prior_sigmas[idx] = float(sigma)
        parameter_prior_summary.append(
            {
                "name": str(name),
                "center": float(center),
                "sigma": float(sigma),
            }
        )

    prior_active_mask = (
        np.isfinite(prior_centers) & np.isfinite(prior_sigmas) & (prior_sigmas > 0.0)
    )

    def _parameter_prior_residuals(x_trial: Sequence[float]) -> np.ndarray:
        if not np.any(prior_active_mask):
            return np.array([], dtype=float)
        x_arr = np.asarray(x_trial, dtype=float)
        return (x_arr[prior_active_mask] - prior_centers[prior_active_mask]) / prior_sigmas[
            prior_active_mask
        ]

    span = upper_bounds - lower_bounds
    finite_span = np.isfinite(span) & (span > 1e-12)
    fallback_scale = np.maximum(np.abs(x0_arr), 1.0)
    auto_scale = np.where(finite_span, span, fallback_scale)
    auto_scale = np.maximum(auto_scale, 1e-6)

    if x_scale_cfg:
        configured_scale = np.array(
            [float(x_scale_cfg.get(name, 1.0)) for name in var_names],
            dtype=float,
        )
        configured_scale = np.where(
            np.isfinite(configured_scale) & (configured_scale > 0.0),
            configured_scale,
            1.0,
        )
        # Treat all-ones as a "use defaults" sentinel and infer useful
        # per-parameter scales from bounds/initial magnitudes.
        if np.allclose(configured_scale, 1.0):
            x_scale = auto_scale
        else:
            x_scale = configured_scale
    else:
        x_scale = auto_scale

    def _build_bound_proximity_summary(
        x_trial: Sequence[float],
    ) -> Dict[str, object]:
        x_arr = np.asarray(x_trial, dtype=float).reshape(-1)
        threshold_fraction = 0.01
        entries: List[Dict[str, object]] = []
        hugging_parameters: List[str] = []
        for idx, name in enumerate(var_names):
            if idx >= x_arr.size:
                continue
            lo = float(lower_bounds[idx]) if idx < lower_bounds.size else float("nan")
            hi = float(upper_bounds[idx]) if idx < upper_bounds.size else float("nan")
            value = float(x_arr[idx])
            span_local = float(hi - lo) if np.isfinite(lo) and np.isfinite(hi) else float("nan")
            near_bound = False
            nearest_bound = ""
            distance_fraction = float("nan")
            if np.isfinite(span_local) and span_local > 1.0e-12:
                frac_lower = abs(float(value - lo)) / span_local
                frac_upper = abs(float(hi - value)) / span_local
                distance_fraction = min(frac_lower, frac_upper)
                if distance_fraction <= threshold_fraction:
                    near_bound = True
                    nearest_bound = "lower" if frac_lower <= frac_upper else "upper"
            entry = {
                "name": str(name),
                "value": float(value),
                "lower_bound": float(lo),
                "upper_bound": float(hi),
                "span": float(span_local),
                "distance_fraction": float(distance_fraction),
                "near_bound": bool(near_bound),
                "nearest_bound": str(nearest_bound),
            }
            entries.append(entry)
            if near_bound:
                hugging_parameters.append(str(name))
        return {
            "threshold_fraction": float(threshold_fraction),
            "entries": entries,
            "hugging_parameters": hugging_parameters,
            "has_hugging_parameters": bool(hugging_parameters),
        }

    residual_fn = pixel_cost_fn if point_match_mode else cost_fn
    final_metric_residual_fn = residual_fn
    final_metric_point_match_evaluator = point_match_evaluator

    class _ManualPointFitEarlyStop(RuntimeError):
        def __init__(
            self,
            message: str,
            *,
            x_trial: Sequence[float],
            residual_arr: Sequence[float],
            point_match_summary: Mapping[str, object] | None,
            bound_proximity_summary: Mapping[str, object] | None,
        ) -> None:
            super().__init__(message)
            self.x_trial = np.asarray(x_trial, dtype=float)
            self.residual_arr = np.asarray(residual_arr, dtype=float)
            self.point_match_summary = (
                dict(point_match_summary) if isinstance(point_match_summary, Mapping) else {}
            )
            self.bound_proximity_summary = (
                dict(bound_proximity_summary)
                if isinstance(bound_proximity_summary, Mapping)
                else {}
            )

    parameter_debug_entries: List[Dict[str, object]] = []
    for idx, name in enumerate(var_names):
        prior_enabled = bool(prior_active_mask[idx]) if idx < prior_active_mask.size else False
        parameter_debug_entries.append(
            {
                "name": str(name),
                "group": _parameter_group(str(name)),
                "start": float(requested_x0_arr[idx])
                if idx < requested_x0_arr.size
                else float("nan"),
                "lower_bound": float(lower_bounds[idx])
                if idx < lower_bounds.size
                else float("nan"),
                "upper_bound": float(upper_bounds[idx])
                if idx < upper_bounds.size
                else float("nan"),
                "scale": float(x_scale[idx]) if idx < x_scale.size else float("nan"),
                "prior_enabled": bool(prior_enabled),
                "prior_center": float(prior_centers[idx]) if prior_enabled else float("nan"),
                "prior_sigma": float(prior_sigmas[idx]) if prior_enabled else float("nan"),
            }
        )

    dataset_debug_entries: List[Dict[str, object]] = []
    for dataset_ctx in dataset_contexts:
        subset = dataset_ctx.subset
        dataset_debug_entries.append(
            {
                "dataset_index": int(dataset_ctx.dataset_index),
                "label": str(dataset_ctx.label),
                "theta_initial_deg": float(dataset_ctx.theta_initial),
                "measured_count": int(len(subset.measured_entries)),
                "subset_reflection_count": int(subset.miller.shape[0]),
                "total_reflection_count": int(subset.total_reflection_count),
                "fixed_source_reflection_count": int(subset.fixed_source_reflection_count),
                "fallback_hkl_count": int(subset.fallback_hkl_count),
                "subset_reduced": bool(subset.reduced),
            }
        )

    geometry_fit_debug_summary: Dict[str, object] = {
        "point_match_mode": bool(point_match_mode),
        "dynamic_point_geometry_fit": bool(dynamic_point_geometry_fit),
        "dataset_count": int(len(dataset_contexts)),
        "var_names": [str(name) for name in var_names],
        "solver": {
            "manual_point_fit_mode": bool(manual_point_fit_mode),
            "loss": str(solver_loss),
            "f_scale_px": float(solver_f_scale),
            "max_nfev": int(solver_max_nfev),
            "restarts": int(solver_restarts),
            "weighted_matching": bool(weighted_matching),
            "missing_pair_penalty_px": float(missing_pair_penalty),
            "missing_pair_penalty_deg": float(missing_pair_penalty_deg),
            "q_group_line_constraints": bool(q_group_line_constraints_enabled),
            "q_group_line_angle_weight": float(q_group_line_angle_weight),
            "q_group_line_offset_weight": float(q_group_line_offset_weight),
            "q_group_line_missing_penalty_scale": float(q_group_line_missing_penalty_scale),
            "hk0_peak_priority_weight": float(hk0_peak_priority_weight),
            "use_measurement_uncertainty": bool(use_measurement_uncertainty),
            "anisotropic_measurement_uncertainty": bool(anisotropic_uncertainty_enabled),
            "full_beam_polish_enabled": bool(full_beam_polish_enabled),
            "full_beam_polish_max_nfev": int(full_beam_polish_max_nfev),
            "full_beam_polish_match_radius_px": float(full_beam_polish_match_radius_px),
            "manual_fail_fast": bool(manual_fail_fast_enabled),
            "manual_fail_fast_eval_window": int(manual_fail_fast_eval_window),
            "manual_fail_fast_probe_period": int(manual_fail_fast_probe_period),
            "manual_fail_fast_peak_rms_px": float(manual_fail_fast_peak_rms_px),
            "manual_fail_fast_min_bound_parameters": int(manual_fail_fast_min_bound_parameters),
        },
        "parallelization": dict(parallelization_summary),
        "parameter_entries": parameter_debug_entries,
        "dataset_entries": dataset_debug_entries,
    }

    all_candidate_param_names = [
        str(name)
        for name in (candidate_param_names if candidate_param_names is not None else var_names)
    ]
    if not all_candidate_param_names:
        all_candidate_param_names = [str(name) for name in var_names]
    active_name_set = {str(name) for name in var_names}

    nominal_scale_by_group = {
        "center": max(float(image_size) * 0.05, 1.0),
        "tilt": 0.5,
        "distance": 0.01,
        "lattice": 0.01,
        "other": 1.0,
    }
    nominal_scale_by_name = {
        "zb": 5.0e-4,
        "zs": 5.0e-4,
        "theta_initial": 0.25,
        "theta_offset": 0.25,
        "psi_z": 0.5,
        "chi": 0.5,
        "cor_angle": 0.5,
        "gamma": 0.5,
        "Gamma": 0.5,
        "corto_detector": 0.01,
        "a": 0.01,
        "c": 0.01,
        "center_x": max(float(image_size) * 0.05, 1.0),
        "center_y": max(float(image_size) * 0.05, 1.0),
    }

    def _nominal_scale_for_name(name: str) -> float:
        if str(name) in nominal_scale_by_name:
            return float(nominal_scale_by_name[str(name)])
        return float(nominal_scale_by_group.get(_parameter_group(str(name)), 1.0))

    def _coerce_prior_entry(name: str, ref_value: float) -> tuple[float | None, float | None]:
        entry = prior_cfg.get(str(name))
        if entry is None:
            return None, None
        prior_mean: float | None = float(ref_value)
        prior_sigma: float | None = None
        enabled = True
        if isinstance(entry, (int, float)):
            try:
                prior_sigma = float(entry)
            except Exception:
                prior_sigma = None
        elif isinstance(entry, Mapping):
            enabled = bool(entry.get("enabled", True))
            try:
                prior_mean = float(entry.get("center", ref_value))
            except Exception:
                prior_mean = float(ref_value)
            sigma_value = entry.get("sigma", entry.get("stdev", entry.get("std")))
            try:
                prior_sigma = float(sigma_value)
            except Exception:
                prior_sigma = None
        if not enabled:
            return None, None
        if prior_mean is None or not np.isfinite(float(prior_mean)):
            prior_mean = float(ref_value)
        if prior_sigma is None or not np.isfinite(float(prior_sigma)) or float(prior_sigma) <= 0.0:
            return float(prior_mean), None
        return float(prior_mean), float(prior_sigma)

    def _build_param_spec_for_name(
        name: str,
        *,
        ref_value: float,
    ) -> ParamSpec:
        lower, upper = _cfg_bounds(str(name), float(ref_value))
        prior_mean, prior_sigma = _coerce_prior_entry(str(name), float(ref_value))
        scale = _default_scale(
            float(ref_value),
            float(lower),
            float(upper),
            prior_sigma=prior_sigma,
            nominal_scale=_nominal_scale_for_name(str(name)),
        )
        seed_group = "auto"
        prior_entry = prior_cfg.get(str(name))
        if isinstance(prior_entry, Mapping):
            explicit_group = str(prior_entry.get("seed_group", "") or "").strip().lower()
            if explicit_group in {"trusted", "uncertain"}:
                seed_group = explicit_group
        if seed_group not in {"trusted", "uncertain"}:
            span_value = (
                float(upper - lower) if np.isfinite(lower) and np.isfinite(upper) else float("nan")
            )
            if (
                prior_sigma is not None
                and np.isfinite(float(prior_sigma))
                and np.isfinite(span_value)
                and span_value > 0.0
                and float(prior_sigma) <= float(trusted_prior_fraction_of_span) * span_value
            ):
                seed_group = "trusted"
            else:
                seed_group = "uncertain"
        return ParamSpec(
            name=str(name),
            ref_value=float(ref_value),
            lower=float(lower),
            upper=float(upper),
            scale=float(scale),
            prior_mean=(None if prior_mean is None else float(prior_mean)),
            prior_sigma=(None if prior_sigma is None else float(prior_sigma)),
            seed_group=str(seed_group),
        )

    def _state_from_theta(
        theta_values: Sequence[float],
        *,
        names: Sequence[str],
        base_params: Optional[Mapping[str, object]] = None,
    ) -> Dict[str, object]:
        return _apply_trial_params(theta_values, names=names, base_params=base_params)

    def _state_from_u(
        specs: Sequence[ParamSpec],
        u_values: Sequence[float],
        *,
        base_params: Optional[Mapping[str, object]] = None,
    ) -> Dict[str, object]:
        return _state_from_theta(
            _physical_from_u(specs, u_values),
            names=[spec.name for spec in specs],
            base_params=base_params,
        )

    def _theta_vector_for_names(
        names: Sequence[str],
        *,
        base_params: Optional[Mapping[str, object]] = None,
    ) -> np.ndarray:
        return _vector_from_params(
            params if base_params is None else dict(base_params),
            names=names,
        )

    def _build_specs_from_params(
        names: Sequence[str],
        *,
        base_params: Optional[Mapping[str, object]] = None,
    ) -> List[ParamSpec]:
        theta_values = _theta_vector_for_names(names, base_params=base_params)
        return [
            _build_param_spec_for_name(str(name), ref_value=float(theta_values[idx]))
            for idx, name in enumerate(names)
        ]

    def _weighted_rms_px(residual_arr: Sequence[float]) -> float:
        residual_np = np.asarray(residual_arr, dtype=float).reshape(-1)
        finite_residual = residual_np[np.isfinite(residual_np)]
        if finite_residual.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(finite_residual * finite_residual)))

    def _status_float(value: object, digits: int = 4) -> str:
        try:
            numeric = float(value)
        except Exception:
            return "nan"
        if not np.isfinite(numeric):
            return "nan"
        return f"{numeric:.{int(digits)}f}"

    def _emit_geometry_fit_setup_status() -> None:
        if point_match_mode and dynamic_point_geometry_fit:
            mode_label = "dynamic-angle"
        else:
            mode_label = "point-match" if point_match_mode else "angle"
        _emit_status(
            "Geometry fit: setup "
            f"mode={mode_label} "
            f"datasets={int(len(dataset_contexts))} "
            f"vars={','.join(str(name) for name in var_names)} "
            f"loss={solver_loss} "
            f"f_scale={_status_float(solver_f_scale, 3)} "
            f"max_nfev={int(solver_max_nfev)} "
            f"restarts={int(solver_restarts)}"
        )
        for dataset_entry in dataset_debug_entries:
            _emit_status(
                "Geometry fit: dataset[{idx}] label={label} theta={theta}deg "
                "measured={measured} subset={subset}/{total} "
                "fixed_reflections={fixed} fallback_hkls={fallback} reduced={reduced}".format(
                    idx=int(dataset_entry.get("dataset_index", -1)),
                    label=str(dataset_entry.get("label", "")),
                    theta=_status_float(dataset_entry.get("theta_initial_deg", np.nan), 4),
                    measured=int(dataset_entry.get("measured_count", 0)),
                    subset=int(dataset_entry.get("subset_reflection_count", 0)),
                    total=int(dataset_entry.get("total_reflection_count", 0)),
                    fixed=int(dataset_entry.get("fixed_source_reflection_count", 0)),
                    fallback=int(dataset_entry.get("fallback_hkl_count", 0)),
                    reduced=bool(dataset_entry.get("subset_reduced", False)),
                )
            )

    _emit_geometry_fit_setup_status()

    def _build_single_dataset_refinement_context(
        local_params: Dict[str, object],
    ) -> Optional[Dict[str, object]]:
        if len(dataset_contexts) != 1:
            return None
        dataset_ctx = dataset_contexts[0]
        experimental_local = dataset_ctx.experimental_image
        if experimental_local is None:
            return None
        experimental_local = np.asarray(experimental_local, dtype=np.float64)
        if experimental_local.shape != (image_size, image_size):
            return None

        fit_miller = np.asarray(dataset_ctx.subset.miller, dtype=np.float64)
        fit_intensities = np.asarray(dataset_ctx.subset.intensities, dtype=np.float64)
        fit_measured = list(dataset_ctx.subset.measured_entries)
        if fit_miller.ndim != 2 or fit_miller.shape[0] <= 0 or not fit_measured:
            return None

        measured_dict = build_measured_dict(fit_measured)
        if not measured_dict:
            return None

        cache = SimulationCache(
            [
                "gamma",
                "Gamma",
                "corto_detector",
                "theta_initial",
                "cor_angle",
                "zs",
                "zb",
                "chi",
                "psi_z",
                "a",
                "c",
                "center",
            ]
        )

        def _refine_simulator(local_updates: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
            merged = dict(local_params)
            merged.update(local_updates)
            merged = _update_params(merged, (), ())
            return _simulate_with_cache(
                merged,
                fit_miller,
                fit_intensities,
                image_size,
                cache,
            )

        return {
            "dataset_ctx": dataset_ctx,
            "experimental_image": experimental_local,
            "fit_miller": fit_miller,
            "fit_intensities": fit_intensities,
            "fit_measured": fit_measured,
            "measured_dict": measured_dict,
            "simulator": _refine_simulator,
        }

    def _maybe_run_ridge_refinement(current_result: OptimizeResult) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "enabled": bool(ridge_refinement_enabled),
            "status": "skipped",
            "reason": "",
            "accepted": False,
        }
        if not point_match_mode:
            summary["reason"] = "point_match_mode_disabled"
            return summary
        if not ridge_refinement_enabled:
            summary["reason"] = "disabled_by_config"
            return summary
        if len(dataset_contexts) != 1:
            summary["reason"] = "requires_single_dataset"
            return summary
        if "theta_offset" in var_names:
            summary["reason"] = "theta_offset_not_supported"
            return summary
        if getattr(current_result, "x", None) is None:
            summary["reason"] = "missing_parameter_vector"
            return summary

        point_x = np.asarray(current_result.x, dtype=float)
        point_local = _apply_trial_params(point_x)
        point_residual_before, _, point_summary_before = _evaluate_pixel_matches(
            point_local,
            collect_diagnostics=True,
        )
        point_cost_before = _robust_cost(
            np.asarray(point_residual_before, dtype=float),
            loss=solver_loss,
            f_scale=solver_f_scale,
        )
        try:
            point_rms_before = float(point_summary_before.get("unweighted_peak_rms_px", np.nan))
        except Exception:
            point_rms_before = float("nan")
        matched_before = int(point_summary_before.get("matched_pair_count", 0))
        summary.update(
            {
                "point_cost_before": float(point_cost_before),
                "point_rms_before_px": float(point_rms_before),
                "matched_pair_count_before": int(matched_before),
            }
        )

        ctx = _build_single_dataset_refinement_context(point_local)
        if ctx is None:
            summary["reason"] = "single_dataset_context_unavailable"
            return summary

        stage_cfg = {
            "downsample_factor": int(ridge_refinement_cfg.get("downsample_factor", 4)),
            "max_nfev": int(ridge_refinement_cfg.get("max_nfev", 20)),
        }
        stage_cfg["downsample_factor"] = max(1, int(stage_cfg["downsample_factor"]))
        stage_cfg["max_nfev"] = max(10, int(stage_cfg["max_nfev"]))

        _emit_status("Geometry fit: running ridge refinement")
        try:
            updated_params, ridge_result = _stage_one_initialize(
                np.asarray(ctx["experimental_image"], dtype=np.float64),
                point_local,
                var_names,
                ctx["simulator"],
                downsample_factor=int(stage_cfg["downsample_factor"]),
                max_nfev=int(stage_cfg["max_nfev"]),
                bounds=(lower_bounds, upper_bounds),
                x_scale=x_scale,
            )
        except Exception as exc:
            summary["status"] = "failed"
            summary["reason"] = f"ridge_refinement_failed: {exc}"
            return summary

        refined_x = _vector_from_params(updated_params)
        refined_x = np.minimum(np.maximum(refined_x, lower_bounds), upper_bounds)
        refined_local = _apply_trial_params(refined_x)
        point_residual_after, _, point_summary_after = _evaluate_pixel_matches(
            refined_local,
            collect_diagnostics=True,
        )
        point_cost_after = _robust_cost(
            np.asarray(point_residual_after, dtype=float),
            loss=solver_loss,
            f_scale=solver_f_scale,
        )
        try:
            point_rms_after = float(point_summary_after.get("unweighted_peak_rms_px", np.nan))
        except Exception:
            point_rms_after = float("nan")
        matched_after = int(point_summary_after.get("matched_pair_count", 0))

        ridge_initial_cost = float(getattr(ridge_result, "initial_cost", np.nan))
        ridge_final_cost = float(getattr(ridge_result, "final_cost", np.nan))
        max_point_cost_increase_fraction = float(
            ridge_refinement_cfg.get("max_point_cost_increase_fraction", 0.03)
        )
        max_point_cost_increase_fraction = max(0.0, max_point_cost_increase_fraction)
        max_point_rms_increase_px = float(
            ridge_refinement_cfg.get("max_point_rms_increase_px", 0.35)
        )
        if not np.isfinite(max_point_rms_increase_px):
            max_point_rms_increase_px = 0.35
        max_point_rms_increase_px = max(0.0, max_point_rms_increase_px)
        min_ridge_cost_reduction = float(
            ridge_refinement_cfg.get("min_ridge_cost_reduction", 1.0e-6)
        )
        if not np.isfinite(min_ridge_cost_reduction):
            min_ridge_cost_reduction = 1.0e-6
        min_ridge_cost_reduction = max(0.0, min_ridge_cost_reduction)

        if np.isfinite(point_cost_before):
            if point_cost_before > 1.0e-12:
                point_cost_limit = point_cost_before * (1.0 + max_point_cost_increase_fraction)
            else:
                point_cost_limit = point_cost_before + max(
                    max_point_cost_increase_fraction,
                    1.0e-6,
                )
        else:
            point_cost_limit = float("inf")
        if np.isfinite(point_rms_before):
            point_rms_limit = point_rms_before + max_point_rms_increase_px
        else:
            point_rms_limit = float("inf")

        ridge_improved = (
            np.isfinite(ridge_initial_cost)
            and np.isfinite(ridge_final_cost)
            and ridge_final_cost + min_ridge_cost_reduction < ridge_initial_cost
        )
        point_cost_ok = not np.isfinite(point_cost_limit) or (
            np.isfinite(point_cost_after) and point_cost_after <= point_cost_limit + 1.0e-12
        )
        point_rms_ok = not np.isfinite(point_rms_limit) or (
            np.isfinite(point_rms_after) and point_rms_after <= point_rms_limit + 1.0e-12
        )
        matched_ok = matched_after >= matched_before
        accepted = bool(ridge_improved and point_cost_ok and point_rms_ok and matched_ok)

        summary.update(
            {
                "status": "accepted" if accepted else "rejected",
                "accepted": bool(accepted),
                "reason": (
                    "accepted"
                    if accepted
                    else ", ".join(
                        part
                        for ok, part in (
                            (ridge_improved, "ridge_cost_not_improved"),
                            (point_cost_ok, "point_cost_regressed"),
                            (point_rms_ok, "point_rms_regressed"),
                            (matched_ok, "matched_pairs_decreased"),
                        )
                        if not ok
                    )
                ),
                "stage_nfev": int(getattr(ridge_result, "nfev", 0)),
                "stage_success": bool(getattr(ridge_result, "success", False)),
                "stage_message": str(getattr(ridge_result, "message", "")),
                "ridge_cost_before": float(ridge_initial_cost),
                "ridge_cost_after": float(ridge_final_cost),
                "point_cost_after": float(point_cost_after),
                "point_rms_after_px": float(point_rms_after),
                "matched_pair_count_after": int(matched_after),
                "point_cost_limit": float(point_cost_limit),
                "point_rms_limit_px": float(point_rms_limit),
            }
        )
        if accepted:
            summary["x"] = refined_x
        return summary

    def _maybe_run_image_refinement(point_result: OptimizeResult) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "enabled": bool(image_refinement_enabled),
            "status": "skipped",
            "reason": "",
            "accepted": False,
        }
        if not point_match_mode:
            summary["reason"] = "point_match_mode_disabled"
            return summary
        if not image_refinement_enabled:
            summary["reason"] = "disabled_by_config"
            return summary
        if len(dataset_contexts) != 1:
            summary["reason"] = "requires_single_dataset"
            return summary
        if "theta_offset" in var_names:
            summary["reason"] = "theta_offset_not_supported"
            return summary
        if getattr(point_result, "x", None) is None:
            summary["reason"] = "missing_parameter_vector"
            return summary

        dataset_ctx = dataset_contexts[0]
        experimental_local = dataset_ctx.experimental_image
        if experimental_local is None:
            summary["reason"] = "missing_experimental_image"
            return summary
        experimental_local = np.asarray(experimental_local, dtype=np.float64)
        if experimental_local.shape != (image_size, image_size):
            summary["reason"] = "experimental_image_shape_mismatch"
            return summary

        point_x = np.asarray(point_result.x, dtype=float)
        point_local = _apply_trial_params(point_x)
        point_residual_before, _, point_summary_before = _evaluate_pixel_matches(
            point_local,
            collect_diagnostics=True,
        )
        point_cost_before = _robust_cost(
            np.asarray(point_residual_before, dtype=float),
            loss=solver_loss,
            f_scale=solver_f_scale,
        )
        try:
            point_rms_before = float(point_summary_before.get("unweighted_peak_rms_px", np.nan))
        except Exception:
            point_rms_before = float("nan")
        matched_before = int(point_summary_before.get("matched_pair_count", 0))
        summary.update(
            {
                "point_cost_before": float(point_cost_before),
                "point_rms_before_px": float(point_rms_before),
                "matched_pair_count_before": int(matched_before),
            }
        )

        min_rois_default = max(3, len(var_names) + 1)
        min_rois = int(image_refinement_cfg.get("min_rois", min_rois_default))
        min_rois = max(1, min_rois)
        if matched_before < min_rois:
            summary["reason"] = "insufficient_matched_pairs"
            summary["min_rois"] = int(min_rois)
            return summary

        ctx = _build_single_dataset_refinement_context(point_local)
        if ctx is None:
            summary["reason"] = "single_dataset_context_unavailable"
            return summary

        stage_cfg = {
            "downsample_factor": int(image_refinement_cfg.get("downsample_factor", 4)),
            "percentile": float(image_refinement_cfg.get("percentile", 90.0)),
            "huber_percentile": float(image_refinement_cfg.get("huber_percentile", 97.0)),
            "per_reflection_quota": int(image_refinement_cfg.get("per_reflection_quota", 200)),
            "off_tube_fraction": float(image_refinement_cfg.get("off_tube_fraction", 0.05)),
            "max_reflections": int(image_refinement_cfg.get("max_reflections", 12)),
            "equal_peak_weights": bool(image_refinement_cfg.get("equal_peak_weights", True)),
            "random_reflection_fraction": float(
                image_refinement_cfg.get("random_reflection_fraction", 0.15)
            ),
            "sampling_temperature": float(image_refinement_cfg.get("sampling_temperature", 1.0)),
            "explore_fraction": float(image_refinement_cfg.get("explore_fraction", 0.15)),
            "huber_delta": float(image_refinement_cfg.get("huber_delta", 2.5)),
            "outlier_mixture": float(image_refinement_cfg.get("outlier_mixture", 0.1)),
            "max_nfev": int(image_refinement_cfg.get("max_nfev", 25)),
            "roi_refresh_threshold": float(image_refinement_cfg.get("roi_refresh_threshold", 1.0)),
        }
        stage_cfg["downsample_factor"] = max(1, int(stage_cfg["downsample_factor"]))
        stage_cfg["per_reflection_quota"] = max(1, int(stage_cfg["per_reflection_quota"]))
        stage_cfg["max_reflections"] = max(1, int(stage_cfg["max_reflections"]))
        stage_cfg["max_nfev"] = max(10, int(stage_cfg["max_nfev"]))

        _emit_status("Geometry fit: running ROI/image refinement")
        try:
            _, preview_maxpos = ctx["simulator"](point_local)
            preview_rois = build_tube_rois(
                np.asarray(ctx["fit_miller"], dtype=np.float64),
                preview_maxpos,
                point_local,
                image_size,
                measured_dict=dict(ctx["measured_dict"]),
            )
        except Exception as exc:
            summary["status"] = "failed"
            summary["reason"] = f"roi_preview_failed: {exc}"
            return summary

        preview_roi_count = int(len(preview_rois))
        summary["preview_roi_count"] = preview_roi_count
        summary["min_rois"] = int(min_rois)
        if preview_roi_count < min_rois:
            summary["reason"] = "insufficient_rois"
            return summary

        try:
            updated_params, image_result, selected_rois, _final_sim, _image_residual = (
                _stage_two_refinement(
                    np.asarray(ctx["experimental_image"], dtype=np.float64),
                    np.asarray(ctx["fit_miller"], dtype=np.float64),
                    np.asarray(ctx["fit_intensities"], dtype=np.float64),
                    image_size,
                    point_local,
                    var_names,
                    ctx["simulator"],
                    dict(ctx["measured_dict"]),
                    cfg=stage_cfg,
                )
            )
        except Exception as exc:
            summary["status"] = "failed"
            summary["reason"] = f"image_refinement_failed: {exc}"
            return summary

        image_initial_cost = float(getattr(image_result, "initial_cost", np.nan))
        image_final_cost = float(getattr(image_result, "final_cost", np.nan))
        refined_x = _vector_from_params(updated_params)
        refined_x = np.minimum(np.maximum(refined_x, lower_bounds), upper_bounds)
        refined_local = _apply_trial_params(refined_x)
        point_residual_after, _, point_summary_after = _evaluate_pixel_matches(
            refined_local,
            collect_diagnostics=True,
        )
        point_cost_after = _robust_cost(
            np.asarray(point_residual_after, dtype=float),
            loss=solver_loss,
            f_scale=solver_f_scale,
        )
        try:
            point_rms_after = float(point_summary_after.get("unweighted_peak_rms_px", np.nan))
        except Exception:
            point_rms_after = float("nan")
        matched_after = int(point_summary_after.get("matched_pair_count", 0))

        max_point_cost_increase_fraction = float(
            image_refinement_cfg.get("max_point_cost_increase_fraction", 0.05)
        )
        max_point_cost_increase_fraction = max(
            0.0,
            max_point_cost_increase_fraction,
        )
        max_point_rms_increase_px = float(
            image_refinement_cfg.get("max_point_rms_increase_px", 0.5)
        )
        if not np.isfinite(max_point_rms_increase_px):
            max_point_rms_increase_px = 0.5
        max_point_rms_increase_px = max(0.0, max_point_rms_increase_px)
        min_image_cost_reduction = float(
            image_refinement_cfg.get("min_image_cost_reduction", 1.0e-6)
        )
        if not np.isfinite(min_image_cost_reduction):
            min_image_cost_reduction = 1.0e-6
        min_image_cost_reduction = max(0.0, min_image_cost_reduction)

        if np.isfinite(point_cost_before):
            if point_cost_before > 1.0e-12:
                point_cost_limit = point_cost_before * (1.0 + max_point_cost_increase_fraction)
            else:
                point_cost_limit = point_cost_before + max(max_point_cost_increase_fraction, 1.0e-6)
        else:
            point_cost_limit = float("inf")
        if np.isfinite(point_rms_before):
            point_rms_limit = point_rms_before + max_point_rms_increase_px
        else:
            point_rms_limit = float("inf")

        image_improved = (
            np.isfinite(image_initial_cost)
            and np.isfinite(image_final_cost)
            and image_final_cost + min_image_cost_reduction < image_initial_cost
        )
        point_cost_ok = not np.isfinite(point_cost_limit) or (
            np.isfinite(point_cost_after) and point_cost_after <= point_cost_limit + 1.0e-12
        )
        point_rms_ok = not np.isfinite(point_rms_limit) or (
            np.isfinite(point_rms_after) and point_rms_after <= point_rms_limit + 1.0e-12
        )
        matched_ok = matched_after >= matched_before
        accepted = bool(image_improved and point_cost_ok and point_rms_ok and matched_ok)

        summary.update(
            {
                "status": "accepted" if accepted else "rejected",
                "accepted": bool(accepted),
                "reason": (
                    "accepted"
                    if accepted
                    else ", ".join(
                        part
                        for ok, part in (
                            (image_improved, "image_cost_not_improved"),
                            (point_cost_ok, "point_cost_regressed"),
                            (point_rms_ok, "point_rms_regressed"),
                            (matched_ok, "matched_pairs_decreased"),
                        )
                        if not ok
                    )
                ),
                "preview_roi_count": int(preview_roi_count),
                "selected_roi_count": int(len(selected_rois)),
                "stage_nfev": int(getattr(image_result, "nfev", 0)),
                "stage_success": bool(getattr(image_result, "success", False)),
                "stage_message": str(getattr(image_result, "message", "")),
                "image_cost_before": float(image_initial_cost),
                "image_cost_after": float(image_final_cost),
                "point_cost_after": float(point_cost_after),
                "point_rms_after_px": float(point_rms_after),
                "matched_pair_count_after": int(matched_after),
                "point_cost_limit": float(point_cost_limit),
                "point_rms_limit_px": float(point_rms_limit),
            }
        )
        if accepted:
            summary["x"] = refined_x
        return summary

    def _diagnostic_match_key(entry: Dict[str, object]) -> Tuple[object, ...]:
        raw_hkl = entry.get("hkl")
        if isinstance(raw_hkl, tuple):
            hkl_key: object = tuple(int(v) for v in raw_hkl)
        else:
            hkl_key = str(raw_hkl)
        try:
            measured_x = round(float(entry.get("measured_x", np.nan)), 6)
        except Exception:
            measured_x = float("nan")
        try:
            measured_y = round(float(entry.get("measured_y", np.nan)), 6)
        except Exception:
            measured_y = float("nan")
        return (
            int(entry.get("dataset_index", -1)),
            str(entry.get("match_kind", "")),
            str(entry.get("match_status", "")),
            hkl_key,
            int(entry.get("overlay_match_index", -1)),
            int(entry.get("source_table_index", -1)),
            int(entry.get("source_row_index", -1)),
            measured_x,
            measured_y,
        )

    def _build_identifiability_summary(
        final_result: OptimizeResult,
        point_match_diagnostics: Optional[Sequence[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "enabled": bool(identifiability_enabled),
            "status": "skipped",
            "reason": "",
            "underconstrained": False,
        }
        if not identifiability_enabled:
            summary["reason"] = "disabled_by_config"
            return summary
        if getattr(final_result, "x", None) is None:
            summary["reason"] = "missing_parameter_vector"
            return summary

        x_ref = np.asarray(final_result.x, dtype=float)
        if x_ref.ndim != 1 or x_ref.size == 0:
            summary["reason"] = "empty_parameter_vector"
            return summary

        residual_ref = np.asarray(final_metric_residual_fn(x_ref), dtype=float)
        if residual_ref.ndim != 1 or residual_ref.size == 0:
            summary["reason"] = "empty_residual"
            return summary

        jacobian = np.full((residual_ref.size, x_ref.size), np.nan, dtype=np.float64)
        steps = np.full(x_ref.size, np.nan, dtype=np.float64)
        residual_plus_cache: Dict[int, np.ndarray] = {}
        residual_minus_cache: Dict[int, np.ndarray] = {}
        diagnostics_plus_cache: Dict[int, Dict[Tuple[object, ...], np.ndarray]] = {}
        diagnostics_minus_cache: Dict[int, Dict[Tuple[object, ...], np.ndarray]] = {}

        base_diag_map: Dict[Tuple[object, ...], np.ndarray] = {}
        if point_match_mode and point_match_diagnostics:
            for entry in point_match_diagnostics:
                if str(entry.get("match_status", "")).lower() != "matched":
                    continue
                try:
                    base_diag_map[_diagnostic_match_key(dict(entry))] = np.array(
                        [
                            float(entry.get("dx_px", np.nan)),
                            float(entry.get("dy_px", np.nan)),
                        ],
                        dtype=np.float64,
                    )
                except Exception:
                    continue

        def _evaluate_diag_map(x_trial: np.ndarray) -> Dict[Tuple[object, ...], np.ndarray]:
            local_trial = _apply_trial_params(x_trial)
            _, diagnostics_trial, _ = final_metric_point_match_evaluator(
                local_trial,
                collect_diagnostics=True,
            )
            out: Dict[Tuple[object, ...], np.ndarray] = {}
            for raw_entry in diagnostics_trial:
                entry = dict(raw_entry)
                if str(entry.get("match_status", "")).lower() != "matched":
                    continue
                try:
                    out[_diagnostic_match_key(entry)] = np.array(
                        [
                            float(entry.get("dx_px", np.nan)),
                            float(entry.get("dy_px", np.nan)),
                        ],
                        dtype=np.float64,
                    )
                except Exception:
                    continue
            return out

        for idx in range(x_ref.size):
            probe_scale = max(
                float(x_scale[idx]) if idx < len(x_scale) else 0.0,
                float(np.abs(x_ref[idx])),
                1.0,
            )
            if idx < len(span) and np.isfinite(span[idx]) and span[idx] > 1.0e-12:
                probe_scale = max(probe_scale, float(span[idx]))
            step = max(
                identifiability_fd_min_step,
                identifiability_fd_step_fraction * probe_scale,
            )
            if idx < len(span) and np.isfinite(span[idx]) and span[idx] > 1.0e-12:
                step = min(step, 0.25 * float(span[idx]))
            step = max(step, identifiability_fd_min_step)
            x_plus = np.asarray(x_ref, dtype=float).copy()
            x_minus = np.asarray(x_ref, dtype=float).copy()
            x_plus[idx] = min(float(upper_bounds[idx]), float(x_plus[idx] + step))
            x_minus[idx] = max(float(lower_bounds[idx]), float(x_minus[idx] - step))
            delta_plus = float(x_plus[idx] - x_ref[idx])
            delta_minus = float(x_ref[idx] - x_minus[idx])
            if delta_plus <= 0.0 and delta_minus <= 0.0:
                continue

            steps[idx] = max(delta_plus, delta_minus)
            use_central = delta_plus > 0.0 and delta_minus > 0.0
            try:
                residual_plus = np.asarray(final_metric_residual_fn(x_plus), dtype=float)
            except Exception:
                residual_plus = np.array([], dtype=float)
            if residual_plus.shape == residual_ref.shape:
                residual_plus_cache[idx] = residual_plus
            try:
                residual_minus = np.asarray(final_metric_residual_fn(x_minus), dtype=float)
            except Exception:
                residual_minus = np.array([], dtype=float)
            if residual_minus.shape == residual_ref.shape:
                residual_minus_cache[idx] = residual_minus

            if use_central and idx in residual_plus_cache and idx in residual_minus_cache:
                denom = float(delta_plus + delta_minus)
                if denom > 0.0:
                    jacobian[:, idx] = (
                        residual_plus_cache[idx] - residual_minus_cache[idx]
                    ) / denom
            elif idx in residual_plus_cache and delta_plus > 0.0:
                jacobian[:, idx] = (residual_plus_cache[idx] - residual_ref) / delta_plus
            elif idx in residual_minus_cache and delta_minus > 0.0:
                jacobian[:, idx] = (residual_ref - residual_minus_cache[idx]) / delta_minus

            if point_match_mode and base_diag_map:
                try:
                    diagnostics_plus_cache[idx] = _evaluate_diag_map(x_plus)
                except Exception:
                    diagnostics_plus_cache[idx] = {}
                try:
                    diagnostics_minus_cache[idx] = _evaluate_diag_map(x_minus)
                except Exception:
                    diagnostics_minus_cache[idx] = {}

        valid_columns = np.all(np.isfinite(jacobian), axis=0)
        if not np.any(valid_columns):
            summary["status"] = "failed"
            summary["reason"] = "no_valid_jacobian_columns"
            return summary

        jacobian_valid = jacobian[:, valid_columns]
        try:
            _, singular_values, _ = np.linalg.svd(jacobian_valid, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            summary["status"] = "failed"
            summary["reason"] = f"svd_failed: {exc}"
            return summary

        singular_values = np.asarray(singular_values, dtype=np.float64)
        max_sv = float(np.max(singular_values)) if singular_values.size else 0.0
        eps = np.finfo(np.float64).eps
        rank_tol = max_sv * max(jacobian_valid.shape) * eps if max_sv > 0.0 else eps
        rank = int(np.count_nonzero(singular_values > rank_tol))
        min_nonzero_sv = singular_values[singular_values > rank_tol]
        if min_nonzero_sv.size:
            condition_number = float(max_sv / np.min(min_nonzero_sv))
        else:
            condition_number = float("inf")

        column_norms = np.linalg.norm(jacobian_valid, axis=0)
        full_column_norms = np.full(x_ref.size, np.nan, dtype=np.float64)
        full_column_norms[valid_columns] = column_norms
        total_column_norm = float(np.nansum(full_column_norms))
        param_entries: List[Dict[str, object]] = []
        for idx, name in enumerate(var_names):
            column_norm = float(full_column_norms[idx])
            param_entries.append(
                {
                    "name": str(name),
                    "valid": bool(valid_columns[idx]),
                    "step": float(steps[idx]) if np.isfinite(steps[idx]) else float("nan"),
                    "column_norm": column_norm,
                    "relative_sensitivity": (
                        float(column_norm / total_column_norm)
                        if np.isfinite(column_norm) and total_column_norm > 1.0e-12
                        else float("nan")
                    ),
                }
            )

        residual_variance = float("nan")
        dof = max(int(residual_ref.size - rank), 1)
        if residual_ref.size:
            residual_variance = float(np.sum(residual_ref * residual_ref) / dof)
        covariance = np.full((x_ref.size, x_ref.size), np.nan, dtype=np.float64)
        correlation = np.full((x_ref.size, x_ref.size), np.nan, dtype=np.float64)
        if rank > 0:
            info_matrix = jacobian_valid.T @ jacobian_valid
            try:
                cov_valid = np.linalg.pinv(info_matrix, hermitian=True)
            except TypeError:
                cov_valid = np.linalg.pinv(info_matrix)
            if np.isfinite(residual_variance):
                cov_valid = cov_valid * residual_variance
            valid_indices = np.flatnonzero(valid_columns)
            for i_local, i_global in enumerate(valid_indices.tolist()):
                for j_local, j_global in enumerate(valid_indices.tolist()):
                    covariance[i_global, j_global] = float(cov_valid[i_local, j_local])
            std = np.sqrt(np.clip(np.diag(cov_valid), 0.0, None))
            for i_local, i_global in enumerate(valid_indices.tolist()):
                for j_local, j_global in enumerate(valid_indices.tolist()):
                    denom = float(std[i_local] * std[j_local])
                    if denom > 1.0e-12:
                        correlation[i_global, j_global] = float(cov_valid[i_local, j_local] / denom)

        group_sensitivity: Dict[str, float] = {}
        for entry in param_entries:
            group = _parameter_group(str(entry["name"]))
            column_norm = float(entry.get("column_norm", np.nan))
            if np.isfinite(column_norm):
                group_sensitivity[group] = float(group_sensitivity.get(group, 0.0) + column_norm)
        dominant_group = "mixed"
        underconstrained = bool(
            rank < int(np.count_nonzero(valid_columns))
            or not np.isfinite(condition_number)
            or condition_number >= identifiability_condition_warn
            or residual_ref.size < int(np.count_nonzero(valid_columns))
        )
        if underconstrained:
            dominant_group = "underconstrained"
        elif group_sensitivity:
            total_group_norm = sum(group_sensitivity.values())
            best_group, best_norm = max(
                group_sensitivity.items(),
                key=lambda item: float(item[1]),
            )
            if total_group_norm > 1.0e-12 and float(best_norm) / total_group_norm >= 0.55:
                dominant_group = str(best_group)

        top_peak_sensitivity: Dict[str, List[Dict[str, object]]] = {}
        if point_match_mode and base_diag_map:
            for idx, name in enumerate(var_names):
                if not bool(valid_columns[idx]):
                    continue
                peak_scores: List[Dict[str, object]] = []
                diag_plus_map = diagnostics_plus_cache.get(idx, {})
                diag_minus_map = diagnostics_minus_cache.get(idx, {})
                for key, base_vec in base_diag_map.items():
                    if key in diag_plus_map and key in diag_minus_map:
                        delta_vec = (diag_plus_map[key] - diag_minus_map[key]) / max(
                            float(steps[idx]) * 2.0, 1.0e-12
                        )
                    elif key in diag_plus_map:
                        delta_vec = (diag_plus_map[key] - base_vec) / max(
                            float(steps[idx]), 1.0e-12
                        )
                    elif key in diag_minus_map:
                        delta_vec = (base_vec - diag_minus_map[key]) / max(
                            float(steps[idx]), 1.0e-12
                        )
                    else:
                        continue
                    sensitivity_val = float(np.linalg.norm(delta_vec))
                    if not np.isfinite(sensitivity_val):
                        continue
                    peak_scores.append(
                        {
                            "key": list(key),
                            "sensitivity": float(sensitivity_val),
                        }
                    )
                peak_scores.sort(
                    key=lambda item: float(item.get("sensitivity", 0.0)),
                    reverse=True,
                )
                top_peak_sensitivity[str(name)] = peak_scores[
                    :identifiability_top_peaks_per_parameter
                ]

        max_column_norm = 0.0
        finite_column_norms = full_column_norms[np.isfinite(full_column_norms)]
        if finite_column_norms.size:
            max_column_norm = float(np.max(finite_column_norms))
        weak_norm_threshold = max(
            max_column_norm * identifiability_weak_norm_ratio,
            1.0e-12,
        )
        weak_parameters: List[Dict[str, object]] = []
        freeze_recommendation_map: Dict[int, Dict[str, object]] = {}

        def _ensure_freeze_recommendation(idx: int) -> Dict[str, object]:
            entry = freeze_recommendation_map.get(int(idx))
            if entry is not None:
                return entry
            param_entry = param_entries[int(idx)]
            entry = {
                "name": str(param_entry.get("name", var_names[int(idx)])),
                "index": int(idx),
                "reasons": [],
                "column_norm": float(param_entry.get("column_norm", np.nan)),
                "relative_sensitivity": float(param_entry.get("relative_sensitivity", np.nan)),
                "valid": bool(param_entry.get("valid", False)),
                "partners": [],
                "max_abs_correlation": float("nan"),
            }
            freeze_recommendation_map[int(idx)] = entry
            return entry

        for idx, entry in enumerate(param_entries):
            column_norm = float(entry.get("column_norm", np.nan))
            is_weak = not bool(entry.get("valid", False)) or (
                np.isfinite(column_norm) and column_norm <= weak_norm_threshold
            )
            if not is_weak:
                continue
            weak_reason = (
                "invalid_jacobian_column"
                if not bool(entry.get("valid", False))
                else "weak_sensitivity"
            )
            weak_parameters.append(
                {
                    "name": str(entry.get("name", var_names[idx])),
                    "index": int(idx),
                    "reason": weak_reason,
                    "column_norm": column_norm,
                    "relative_sensitivity": float(entry.get("relative_sensitivity", np.nan)),
                    "valid": bool(entry.get("valid", False)),
                }
            )
            recommendation = _ensure_freeze_recommendation(idx)
            if weak_reason not in recommendation["reasons"]:
                recommendation["reasons"].append(weak_reason)

        high_correlation_pairs: List[Dict[str, object]] = []
        for i in range(x_ref.size):
            for j in range(i + 1, x_ref.size):
                corr_val = float(correlation[i, j])
                if not np.isfinite(corr_val):
                    continue
                abs_corr = abs(corr_val)
                if abs_corr < identifiability_correlation_warn:
                    continue
                norm_i = float(param_entries[i].get("column_norm", np.nan))
                norm_j = float(param_entries[j].get("column_norm", np.nan))
                if not np.isfinite(norm_i):
                    norm_i = float("inf")
                if not np.isfinite(norm_j):
                    norm_j = float("inf")
                preferred_idx = int(i) if norm_i <= norm_j else int(j)
                partner_idx = int(j) if preferred_idx == int(i) else int(i)
                high_correlation_pairs.append(
                    {
                        "name_i": str(param_entries[i].get("name", var_names[i])),
                        "index_i": int(i),
                        "name_j": str(param_entries[j].get("name", var_names[j])),
                        "index_j": int(j),
                        "correlation": float(corr_val),
                        "abs_correlation": float(abs_corr),
                        "preferred_freeze": str(
                            param_entries[preferred_idx].get(
                                "name",
                                var_names[preferred_idx],
                            )
                        ),
                        "preferred_freeze_index": int(preferred_idx),
                    }
                )
                recommendation = _ensure_freeze_recommendation(preferred_idx)
                if "high_correlation" not in recommendation["reasons"]:
                    recommendation["reasons"].append("high_correlation")
                partner_name = str(param_entries[partner_idx].get("name", var_names[partner_idx]))
                if partner_name not in recommendation["partners"]:
                    recommendation["partners"].append(partner_name)
                current_max_corr = float(recommendation.get("max_abs_correlation", np.nan))
                if not np.isfinite(current_max_corr) or abs_corr > current_max_corr:
                    recommendation["max_abs_correlation"] = float(abs_corr)

        weak_parameters.sort(
            key=lambda item: (
                float(item.get("column_norm", np.inf))
                if np.isfinite(float(item.get("column_norm", np.nan)))
                else float("inf"),
                int(item.get("index", 0)),
            )
        )
        high_correlation_pairs.sort(
            key=lambda item: (
                -float(item.get("abs_correlation", 0.0)),
                int(item.get("index_i", 0)),
                int(item.get("index_j", 0)),
            )
        )
        freeze_recommendations = list(freeze_recommendation_map.values())
        freeze_recommendations.sort(
            key=lambda item: (
                0
                if (
                    "weak_sensitivity" in item.get("reasons", [])
                    or "invalid_jacobian_column" in item.get("reasons", [])
                )
                else 1,
                -float(item.get("max_abs_correlation", 0.0))
                if np.isfinite(float(item.get("max_abs_correlation", np.nan)))
                else 0.0,
                float(item.get("column_norm", np.inf))
                if np.isfinite(float(item.get("column_norm", np.nan)))
                else float("inf"),
                int(item.get("index", 0)),
            )
        )
        warning_flags: List[str] = []
        if underconstrained:
            warning_flags.append("underconstrained")
        if high_correlation_pairs:
            warning_flags.append("high_correlation")
        if weak_parameters:
            warning_flags.append("weak_sensitivity")

        summary.update(
            {
                "enabled": True,
                "status": "ok",
                "reason": "computed",
                "num_parameters": int(x_ref.size),
                "num_valid_parameters": int(np.count_nonzero(valid_columns)),
                "num_residuals": int(residual_ref.size),
                "rank": int(rank),
                "condition_number": float(condition_number),
                "condition_number_warn": float(identifiability_condition_warn),
                "correlation_warn": float(identifiability_correlation_warn),
                "singular_values": singular_values.tolist(),
                "residual_variance": float(residual_variance),
                "parameter_entries": param_entries,
                "covariance_matrix": covariance.tolist(),
                "correlation_matrix": correlation.tolist(),
                "group_sensitivity": {
                    str(key): float(val) for key, val in group_sensitivity.items()
                },
                "dominant_group": str(dominant_group),
                "underconstrained": bool(underconstrained),
                "top_peak_sensitivity": top_peak_sensitivity,
                "weak_column_norm_threshold": float(weak_norm_threshold),
                "weak_parameters": weak_parameters,
                "high_correlation_pairs": high_correlation_pairs,
                "freeze_recommendations": freeze_recommendations,
                "recommended_fixed_parameters": [
                    str(entry.get("name", "")) for entry in freeze_recommendations
                ],
                "recommended_fixed_indices": [
                    int(entry.get("index", 0)) for entry in freeze_recommendations
                ],
                "warning_flags": warning_flags,
            }
        )
        return summary

    def _emit_seed_status(
        status_label: str,
        seed_summary: Mapping[str, object] | None,
    ) -> None:
        summary = seed_summary if isinstance(seed_summary, Mapping) else {}
        if not summary:
            return
        point_summary = summary.get("point_match_summary", None)
        if isinstance(point_summary, Mapping):
            _emit_status(
                "Geometry fit: {label} seed "
                "cost={cost} weighted_rms={weighted_rms}px matched={matched} "
                "missing={missing} peak_rms={peak_rms}px peak_max={peak_max}px".format(
                    label=str(status_label),
                    cost=_status_float(summary.get("cost", np.nan), 6),
                    weighted_rms=_status_float(summary.get("weighted_rms_px", np.nan), 4),
                    matched=int(point_summary.get("matched_pair_count", 0)),
                    missing=int(point_summary.get("missing_pair_count", 0)),
                    peak_rms=_status_float(
                        point_summary.get("unweighted_peak_rms_px", np.nan),
                        4,
                    ),
                    peak_max=_status_float(
                        point_summary.get("unweighted_peak_max_px", np.nan),
                        4,
                    ),
                )
            )
            return
        _emit_status(
            "Geometry fit: {label} seed cost={cost} weighted_rms={weighted_rms}px".format(
                label=str(status_label),
                cost=_status_float(summary.get("cost", np.nan), 6),
                weighted_rms=_status_float(summary.get("weighted_rms_px", np.nan), 4),
            )
        )

    def _run_solver(
        x_start: np.ndarray,
        *,
        status_label: str | None = "main solve",
    ) -> OptimizeResult:
        return _run_solver_with_max_nfev(
            x_start,
            max_nfev=solver_max_nfev,
            status_label=status_label,
        )

    def _run_solver_with_max_nfev(
        x_start: np.ndarray,
        *,
        max_nfev: int,
        residual_callable: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        status_label: str | None = None,
    ) -> OptimizeResult:
        solver_residual = residual_fn if residual_callable is None else residual_callable
        trace_points: List[Dict[str, object]] = []
        progress_state: Dict[str, object] = {
            "label": (None if status_label is None else str(status_label)),
            "evaluation_count": 0,
            "status_emit_count": 0,
            "best_cost_seen": float("inf"),
            "best_weighted_rms_px": float("nan"),
            "last_cost_seen": float("nan"),
            "last_weighted_rms_px": float("nan"),
            "best_unweighted_peak_rms_px": float("inf"),
            "last_unweighted_peak_rms_px": float("nan"),
            "aborted_early": False,
            "trace": trace_points,
        }

        def _maybe_abort_manual_point_fit(
            x_trial: np.ndarray,
            residual_arr: np.ndarray,
            *,
            eval_count: int,
        ) -> None:
            if not manual_fail_fast_enabled or residual_callable is not None:
                return
            if eval_count < manual_fail_fast_eval_window:
                return
            last_probe_eval = int(progress_state.get("manual_fail_fast_last_probe_eval", 0))
            if (
                last_probe_eval > 0
                and (eval_count - last_probe_eval) < manual_fail_fast_probe_period
            ):
                return
            progress_state["manual_fail_fast_last_probe_eval"] = int(eval_count)

            local_trial = _apply_trial_params(np.asarray(x_trial, dtype=float))
            _, _, point_match_summary = point_match_evaluator(
                local_trial,
                collect_diagnostics=False,
            )
            probe_summary = (
                dict(point_match_summary) if isinstance(point_match_summary, Mapping) else {}
            )
            try:
                matched_pair_count = int(probe_summary.get("matched_pair_count", 0))
            except Exception:
                matched_pair_count = 0
            try:
                current_peak_rms = float(probe_summary.get("unweighted_peak_rms_px", np.nan))
            except Exception:
                current_peak_rms = float("nan")
            if matched_pair_count > 0 and np.isfinite(current_peak_rms):
                best_peak_rms = float(
                    progress_state.get(
                        "best_unweighted_peak_rms_px",
                        float("inf"),
                    )
                )
                if not np.isfinite(best_peak_rms) or current_peak_rms < best_peak_rms:
                    progress_state["best_unweighted_peak_rms_px"] = float(current_peak_rms)
                progress_state["last_unweighted_peak_rms_px"] = float(current_peak_rms)

            if matched_pair_count <= 0 or not np.isfinite(current_peak_rms):
                return
            best_peak_rms = float(
                progress_state.get(
                    "best_unweighted_peak_rms_px",
                    current_peak_rms,
                )
            )
            if current_peak_rms < manual_fail_fast_peak_rms_px or (
                np.isfinite(best_peak_rms) and best_peak_rms < manual_fail_fast_peak_rms_px
            ):
                return

            bound_summary = _build_bound_proximity_summary(np.asarray(x_trial, dtype=float))
            hugging_parameters = list(bound_summary.get("hugging_parameters", []))
            progress_state["last_bound_hugging_parameters"] = list(hugging_parameters)
            if len(hugging_parameters) < manual_fail_fast_min_bound_parameters:
                return

            reason = (
                "Manual point-fit guardrail stopped the solve after {evals} "
                "evaluations: unweighted peak RMS {rms:.2f} px remained above "
                "{limit:.2f} px with parameters near bounds ({params})."
            ).format(
                evals=int(eval_count),
                rms=float(current_peak_rms),
                limit=float(manual_fail_fast_peak_rms_px),
                params=", ".join(str(name) for name in hugging_parameters),
            )
            progress_state["aborted_early"] = True
            progress_state["early_stop_reason"] = str(reason)
            progress_state["early_stop_point_match_summary"] = dict(probe_summary)
            progress_state["early_stop_bound_proximity"] = dict(bound_summary)
            if status_label:
                _emit_status(str(reason))
            raise _ManualPointFitEarlyStop(
                str(reason),
                x_trial=np.asarray(x_trial, dtype=float),
                residual_arr=np.asarray(residual_arr, dtype=float),
                point_match_summary=probe_summary,
                bound_proximity_summary=bound_summary,
            )

        def _tracked_solver_residual(x_trial: np.ndarray) -> np.ndarray:
            residual_arr = np.asarray(
                solver_residual(np.asarray(x_trial, dtype=float)),
                dtype=float,
            )
            progress_state["evaluation_count"] = int(progress_state["evaluation_count"]) + 1
            eval_count = int(progress_state["evaluation_count"])
            current_cost = float(
                _robust_cost(
                    residual_arr,
                    loss=solver_loss,
                    f_scale=solver_f_scale,
                )
            )
            current_weighted_rms = float(_weighted_rms_px(residual_arr))
            progress_state["last_cost_seen"] = float(current_cost)
            progress_state["last_weighted_rms_px"] = float(current_weighted_rms)

            best_cost_seen = float(progress_state.get("best_cost_seen", float("inf")))
            improvement_tol = max(
                1.0e-9 * max(abs(best_cost_seen), 1.0),
                1.0e-12,
            )
            improved = bool(current_cost + improvement_tol < best_cost_seen)
            if improved:
                progress_state["best_cost_seen"] = float(current_cost)
                progress_state["best_weighted_rms_px"] = float(current_weighted_rms)
            elif not np.isfinite(best_cost_seen):
                progress_state["best_cost_seen"] = float(current_cost)
                progress_state["best_weighted_rms_px"] = float(current_weighted_rms)

            if point_match_mode and callable(live_update_callback):
                live_payload = (
                    dict(last_live_update_payload)
                    if isinstance(last_live_update_payload, Mapping)
                    else {}
                )
                live_payload.update(
                    {
                        "evaluation_count": int(eval_count),
                        "current_cost": float(current_cost),
                        "best_cost": float(progress_state.get("best_cost_seen", np.nan)),
                        "weighted_rms_px": float(current_weighted_rms),
                        "improved": bool(improved),
                        "var_names": [str(name) for name in var_names],
                        "x_trial": np.asarray(x_trial, dtype=float).tolist(),
                    }
                )
                try:
                    live_update_callback(live_payload)
                except Exception:
                    pass

            _maybe_abort_manual_point_fit(
                np.asarray(x_trial, dtype=float),
                residual_arr,
                eval_count=eval_count,
            )

            last_emit_eval = int(progress_state.get("last_emit_eval", 0))
            emit_reason = ""
            if eval_count in {1, 2, 3, 5, 10}:
                emit_reason = "milestone"
            elif improved and (eval_count - last_emit_eval) >= 5:
                emit_reason = "improved"
            elif (eval_count - last_emit_eval) >= 25:
                emit_reason = "periodic"

            if not emit_reason:
                return residual_arr

            trace_event = {
                "eval": int(eval_count),
                "current_cost": float(current_cost),
                "best_cost": float(progress_state.get("best_cost_seen", np.nan)),
                "weighted_rms_px": float(current_weighted_rms),
                "reason": str(emit_reason),
            }
            if len(trace_points) < 24:
                trace_points.append(trace_event)

            progress_state["last_emit_eval"] = int(eval_count)
            progress_state["status_emit_count"] = int(progress_state["status_emit_count"]) + 1
            if status_label:
                _emit_status(
                    "Geometry fit: {label} eval={eval} cost={cost} best_cost={best_cost} "
                    "weighted_rms={weighted_rms}px".format(
                        label=str(status_label),
                        eval=int(eval_count),
                        cost=_status_float(current_cost, 6),
                        best_cost=_status_float(
                            progress_state.get("best_cost_seen", np.nan),
                            6,
                        ),
                        weighted_rms=_status_float(current_weighted_rms, 4),
                    )
                )
            return residual_arr

        try:
            result = least_squares(
                _tracked_solver_residual,
                np.asarray(x_start, dtype=float),
                bounds=(lower_bounds, upper_bounds),
                x_scale=x_scale,
                loss=solver_loss,
                f_scale=solver_f_scale,
                max_nfev=int(max_nfev),
            )
        except _ManualPointFitEarlyStop as exc:
            x_abort = np.asarray(exc.x_trial, dtype=float)
            result = OptimizeResult(
                x=x_abort,
                fun=np.asarray(exc.residual_arr, dtype=float),
                success=False,
                status=0,
                message=str(exc),
                nfev=int(progress_state.get("evaluation_count", 0)),
                active_mask=np.zeros(x_abort.shape, dtype=int),
                optimality=float("nan"),
            )
            result.early_stop_reason = str(exc)
            result.point_match_summary = dict(exc.point_match_summary)
            result.bound_proximity_summary = dict(exc.bound_proximity_summary)
        progress_state["start_x"] = np.asarray(x_start, dtype=float).tolist()
        try:
            progress_state["end_x"] = np.asarray(result.x, dtype=float).tolist()
        except Exception:
            progress_state["end_x"] = []
        result.geometry_fit_progress = progress_state
        return result

    def _evaluate_cost_at(x_trial: np.ndarray) -> Tuple[np.ndarray, float]:
        residual = np.asarray(residual_fn(np.asarray(x_trial, dtype=float)), dtype=float)
        return residual, _robust_cost(
            residual,
            loss=solver_loss,
            f_scale=solver_f_scale,
        )

    active_specs: List[ParamSpec] = []

    def _data_residual_theta(
        theta_trial: Sequence[float],
        *,
        names: Optional[Sequence[str]] = None,
        base_params: Optional[Mapping[str, object]] = None,
    ) -> np.ndarray:
        local = _apply_trial_params(
            theta_trial,
            names=names,
            base_params=base_params,
        )
        if point_match_mode:
            residual_arr, _, _ = point_match_evaluator(
                local,
                collect_diagnostics=False,
            )
            return np.asarray(residual_arr, dtype=float)

        residual_blocks: List[np.ndarray] = []
        dataset_items = [(local, dataset_ctx) for dataset_ctx in dataset_contexts]
        if dataset_parallel_workers > 1 and len(dataset_items) > 1:
            residual_results = _threaded_map(
                _cost_fn_for_dataset,
                dataset_items,
                max_workers=dataset_parallel_workers,
                numba_threads=worker_numba_threads,
            )
        else:
            residual_results = [_cost_fn_for_dataset(item) for item in dataset_items]
        for residual_arr in residual_results:
            residual_np = np.asarray(residual_arr, dtype=float)
            if residual_np.size:
                residual_blocks.append(residual_np)
        if residual_blocks:
            return np.concatenate(residual_blocks)
        return np.array([], dtype=float)

    def _prior_residual_u(
        specs: Sequence[ParamSpec],
        u_trial: Sequence[float],
    ) -> np.ndarray:
        theta_values = _physical_from_u(specs, u_trial)
        terms: List[float] = []
        for idx, spec in enumerate(specs):
            prior_sigma = spec.prior_sigma
            if (
                prior_sigma is None
                or not np.isfinite(float(prior_sigma))
                or float(prior_sigma) <= 0.0
            ):
                continue
            prior_mean = spec.prior_mean
            if prior_mean is None or not np.isfinite(float(prior_mean)):
                prior_mean = float(spec.ref_value)
            terms.append(float((theta_values[idx] - float(prior_mean)) / float(prior_sigma)))
        return np.asarray(terms, dtype=float)

    def _residual_u(
        u_trial: Sequence[float],
        *,
        specs: Sequence[ParamSpec],
        include_priors: bool,
        base_params: Optional[Mapping[str, object]] = None,
    ) -> np.ndarray:
        theta_values = _physical_from_u(specs, u_trial)
        pieces = [
            _data_residual_theta(
                theta_values,
                names=[spec.name for spec in specs],
                base_params=base_params,
            )
        ]
        if include_priors:
            prior_residual = _prior_residual_u(specs, u_trial)
            if prior_residual.size:
                pieces.append(prior_residual)
        pieces = [piece.reshape(-1) for piece in pieces if np.asarray(piece).size]
        if pieces:
            return np.concatenate(pieces)
        return np.array([], dtype=float)

    def _evaluate_cost_at_u(
        u_trial: Sequence[float],
        *,
        specs: Sequence[ParamSpec],
        include_priors: bool,
        base_params: Optional[Mapping[str, object]] = None,
    ) -> Tuple[np.ndarray, float]:
        residual = _residual_u(
            u_trial,
            specs=specs,
            include_priors=include_priors,
            base_params=base_params,
        )
        return residual, _robust_cost(
            residual,
            loss=solver_loss,
            f_scale=solver_f_scale,
        )

    def _run_solver_u(
        u_start: np.ndarray,
        *,
        specs: Sequence[ParamSpec],
        lower: np.ndarray,
        upper: np.ndarray,
        include_priors: bool,
        max_nfev: int,
        status_label: str | None = None,
        base_params: Optional[Mapping[str, object]] = None,
    ) -> OptimizeResult:
        trace_points: List[Dict[str, object]] = []
        progress_state: Dict[str, object] = {
            "label": (None if status_label is None else str(status_label)),
            "evaluation_count": 0,
            "status_emit_count": 0,
            "best_cost_seen": float("inf"),
            "best_weighted_rms_px": float("nan"),
            "last_cost_seen": float("nan"),
            "last_weighted_rms_px": float("nan"),
            "best_unweighted_peak_rms_px": float("inf"),
            "last_unweighted_peak_rms_px": float("nan"),
            "aborted_early": False,
            "trace": trace_points,
        }

        def _maybe_abort_manual_point_fit_u(
            u_trial: np.ndarray,
            residual_arr: np.ndarray,
            *,
            eval_count: int,
        ) -> None:
            if not manual_fail_fast_enabled:
                return
            if eval_count < manual_fail_fast_eval_window:
                return
            last_probe_eval = int(progress_state.get("manual_fail_fast_last_probe_eval", 0))
            if (
                last_probe_eval > 0
                and (eval_count - last_probe_eval) < manual_fail_fast_probe_period
            ):
                return
            progress_state["manual_fail_fast_last_probe_eval"] = int(eval_count)

            x_trial = _physical_from_u(specs, np.asarray(u_trial, dtype=float))
            local_trial = _apply_trial_params(x_trial)
            _, _, point_match_summary = point_match_evaluator(
                local_trial,
                collect_diagnostics=False,
            )
            probe_summary = (
                dict(point_match_summary) if isinstance(point_match_summary, Mapping) else {}
            )
            try:
                matched_pair_count = int(probe_summary.get("matched_pair_count", 0))
            except Exception:
                matched_pair_count = 0
            try:
                current_peak_rms = float(probe_summary.get("unweighted_peak_rms_px", np.nan))
            except Exception:
                current_peak_rms = float("nan")
            if matched_pair_count > 0 and np.isfinite(current_peak_rms):
                best_peak_rms = float(
                    progress_state.get(
                        "best_unweighted_peak_rms_px",
                        float("inf"),
                    )
                )
                if not np.isfinite(best_peak_rms) or current_peak_rms < best_peak_rms:
                    progress_state["best_unweighted_peak_rms_px"] = float(current_peak_rms)
                progress_state["last_unweighted_peak_rms_px"] = float(current_peak_rms)

            if matched_pair_count <= 0 or not np.isfinite(current_peak_rms):
                return
            best_peak_rms = float(
                progress_state.get(
                    "best_unweighted_peak_rms_px",
                    current_peak_rms,
                )
            )
            if current_peak_rms < manual_fail_fast_peak_rms_px or (
                np.isfinite(best_peak_rms) and best_peak_rms < manual_fail_fast_peak_rms_px
            ):
                return

            bound_summary = _build_bound_proximity_summary(np.asarray(x_trial, dtype=float))
            hugging_parameters = list(bound_summary.get("hugging_parameters", []))
            progress_state["last_bound_hugging_parameters"] = list(hugging_parameters)
            if len(hugging_parameters) < manual_fail_fast_min_bound_parameters:
                return

            reason = (
                "Manual point-fit guardrail stopped the solve after {evals} "
                "evaluations: unweighted peak RMS {rms:.2f} px remained above "
                "{limit:.2f} px with parameters near bounds ({params})."
            ).format(
                evals=int(eval_count),
                rms=float(current_peak_rms),
                limit=float(manual_fail_fast_peak_rms_px),
                params=", ".join(str(name) for name in hugging_parameters),
            )
            progress_state["aborted_early"] = True
            progress_state["early_stop_reason"] = str(reason)
            progress_state["early_stop_point_match_summary"] = dict(probe_summary)
            progress_state["early_stop_bound_proximity"] = dict(bound_summary)
            if status_label:
                _emit_status(str(reason))
            raise _ManualPointFitEarlyStop(
                str(reason),
                x_trial=np.asarray(x_trial, dtype=float),
                residual_arr=np.asarray(residual_arr, dtype=float),
                point_match_summary=probe_summary,
                bound_proximity_summary=bound_summary,
            )

        def _tracked_solver_residual(u_trial: np.ndarray) -> np.ndarray:
            residual_arr = _residual_u(
                np.asarray(u_trial, dtype=float),
                specs=specs,
                include_priors=include_priors,
                base_params=base_params,
            )
            progress_state["evaluation_count"] = int(progress_state["evaluation_count"]) + 1
            eval_count = int(progress_state["evaluation_count"])
            current_cost = float(
                _robust_cost(
                    residual_arr,
                    loss=solver_loss,
                    f_scale=solver_f_scale,
                )
            )
            current_weighted_rms = float(_weighted_rms_px(residual_arr))
            progress_state["last_cost_seen"] = float(current_cost)
            progress_state["last_weighted_rms_px"] = float(current_weighted_rms)

            best_cost_seen = float(progress_state.get("best_cost_seen", float("inf")))
            improvement_tol = max(1.0e-9 * max(abs(best_cost_seen), 1.0), 1.0e-12)
            improved = bool(current_cost + improvement_tol < best_cost_seen)
            if improved or not np.isfinite(best_cost_seen):
                progress_state["best_cost_seen"] = float(current_cost)
                progress_state["best_weighted_rms_px"] = float(current_weighted_rms)

            _maybe_abort_manual_point_fit_u(
                np.asarray(u_trial, dtype=float),
                residual_arr,
                eval_count=eval_count,
            )

            last_emit_eval = int(progress_state.get("last_emit_eval", 0))
            emit_reason = ""
            if eval_count in {1, 2, 3, 5, 10}:
                emit_reason = "milestone"
            elif improved and (eval_count - last_emit_eval) >= 5:
                emit_reason = "improved"
            elif (eval_count - last_emit_eval) >= 25:
                emit_reason = "periodic"
            if not emit_reason:
                return residual_arr

            trace_event = {
                "eval": int(eval_count),
                "current_cost": float(current_cost),
                "best_cost": float(progress_state.get("best_cost_seen", np.nan)),
                "weighted_rms_px": float(current_weighted_rms),
                "reason": str(emit_reason),
            }
            if len(trace_points) < 24:
                trace_points.append(trace_event)
            progress_state["last_emit_eval"] = int(eval_count)
            progress_state["status_emit_count"] = int(progress_state["status_emit_count"]) + 1
            if status_label:
                _emit_status(
                    "Geometry fit: {label} eval={eval} cost={cost} best_cost={best_cost} "
                    "weighted_rms={weighted_rms}px".format(
                        label=str(status_label),
                        eval=int(eval_count),
                        cost=_status_float(current_cost, 6),
                        best_cost=_status_float(
                            progress_state.get("best_cost_seen", np.nan),
                            6,
                        ),
                        weighted_rms=_status_float(current_weighted_rms, 4),
                    )
                )
            return residual_arr

        try:
            result_u = least_squares(
                _tracked_solver_residual,
                np.asarray(u_start, dtype=float),
                bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
                method=solver_method,
                x_scale=1.0,
                loss=solver_loss,
                f_scale=solver_f_scale,
                diff_step=None,
                max_nfev=int(max_nfev),
            )
        except _ManualPointFitEarlyStop as exc:
            result_u = OptimizeResult(
                x=np.asarray(u_start, dtype=float),
                fun=np.asarray(exc.residual_arr, dtype=float),
                success=False,
                status=0,
                message=str(exc),
                nfev=int(progress_state.get("evaluation_count", 0)),
                active_mask=np.zeros(np.asarray(u_start, dtype=float).shape, dtype=int),
                optimality=float("nan"),
            )
            result_u.early_stop_reason = str(exc)
            result_u.point_match_summary = dict(exc.point_match_summary)
            result_u.bound_proximity_summary = dict(exc.bound_proximity_summary)
            try:
                result_u.x = _u_from_physical(specs, np.asarray(exc.x_trial, dtype=float))
            except Exception:
                result_u.x = np.asarray(u_start, dtype=float)
        progress_state["start_u"] = np.asarray(u_start, dtype=float).tolist()
        try:
            progress_state["end_u"] = np.asarray(result_u.x, dtype=float).tolist()
        except Exception:
            progress_state["end_u"] = []
        result_u.geometry_fit_progress = progress_state
        return result_u

    def _wrap_solver_result_from_u(
        result_u: OptimizeResult,
        *,
        specs: Sequence[ParamSpec],
    ) -> OptimizeResult:
        result = OptimizeResult(dict(result_u))
        result.x_u = np.asarray(getattr(result_u, "x", []), dtype=float)
        result.x = _physical_from_u(specs, result.x_u)
        if getattr(result, "geometry_fit_progress", None):
            try:
                result.geometry_fit_progress["start_x"] = _physical_from_u(
                    specs,
                    np.asarray(
                        result.geometry_fit_progress.get("start_u", result.x_u),
                        dtype=float,
                    ),
                ).tolist()
                result.geometry_fit_progress["end_x"] = np.asarray(
                    result.x,
                    dtype=float,
                ).tolist()
            except Exception:
                pass
        return result

    def _build_full_beam_seed_correspondence_map(
        seed_diagnostics: Sequence[Dict[str, object]],
    ) -> Dict[int, List[Dict[str, object]]]:
        dataset_table_signatures = {
            int(dataset_ctx.dataset_index): _local_table_signature(
                dataset_ctx.subset.original_indices
            )
            for dataset_ctx in dataset_contexts
        }

        def _frozen_locator_fields(
            entry: Mapping[str, object],
            *,
            dataset_index: int,
        ) -> Optional[Dict[str, object]]:
            if str(entry.get("match_status", "")).strip().lower() != "matched":
                return None
            trusted_full_reflection = _entry_trusted_full_reflection_identity(entry)
            source_branch_index, _source_branch_source = _measured_source_peak_index_with_source(
                entry
            )
            if trusted_full_reflection:
                reflection_idx = _nonnegative_index(entry.get("source_reflection_index"))
                if reflection_idx is None or source_branch_index not in {0, 1}:
                    return None
                return {
                    "frozen_locator_kind": "trusted_branch",
                    "frozen_table_namespace": "full_reflection",
                    "frozen_table_index": int(reflection_idx),
                    "frozen_branch_index": int(source_branch_index),
                }

            table_idx = _nonnegative_index(entry.get("resolved_table_index"))
            if table_idx is None:
                table_idx = _nonnegative_index(entry.get("source_table_index"))
            if table_idx is None:
                return None
            table_signature = dataset_table_signatures.get(int(dataset_index))
            if source_branch_index in {0, 1}:
                return {
                    "frozen_locator_kind": "local_branch",
                    "frozen_table_namespace": "current_full_local",
                    "frozen_table_index": int(table_idx),
                    "frozen_branch_index": int(source_branch_index),
                    "frozen_table_signature": table_signature,
                }
            row_idx = _nonnegative_index(entry.get("source_row_index"))
            if row_idx is None:
                return None
            return {
                "frozen_locator_kind": "local_row",
                "frozen_table_namespace": "current_full_local",
                "frozen_table_index": int(table_idx),
                "frozen_row_index": int(row_idx),
                "frozen_table_signature": table_signature,
            }

        grouped: Dict[int, Dict[int, Dict[str, object]]] = {}
        for fallback_index, raw_entry in enumerate(seed_diagnostics):
            if not isinstance(raw_entry, Mapping):
                continue
            entry = dict(raw_entry)
            try:
                dataset_index = int(entry.get("dataset_index", 0))
            except Exception:
                dataset_index = 0
            try:
                overlay_match_index = int(
                    entry.get(
                        "overlay_match_index",
                        entry.get("match_input_index", fallback_index),
                    )
                )
            except Exception:
                overlay_match_index = int(fallback_index)
            if overlay_match_index < 0:
                overlay_match_index = int(fallback_index)

            hkl_value = entry.get("hkl")
            if isinstance(hkl_value, tuple) and len(hkl_value) == 3:
                try:
                    normalized_hkl = tuple(int(v) for v in hkl_value)
                except Exception:
                    normalized_hkl = hkl_value
            else:
                normalized_hkl = hkl_value
            source_reflection_index = _nonnegative_index(
                entry.get("source_reflection_index")
            )
            source_branch_index, source_branch_source = _measured_source_peak_index_with_source(
                entry
            )
            source_reflection_namespace = entry.get("source_reflection_namespace")
            source_reflection_is_full = entry.get("source_reflection_is_full")
            resolved_table_index = entry.get("resolved_table_index")
            frozen_locator = _frozen_locator_fields(entry, dataset_index=int(dataset_index))
            if not isinstance(frozen_locator, dict):
                continue
            grouped.setdefault(int(dataset_index), {})[int(overlay_match_index)] = {
                "dataset_index": int(dataset_index),
                "overlay_match_index": int(overlay_match_index),
                "match_input_index": int(entry.get("match_input_index", overlay_match_index)),
                "pair_id": entry.get("pair_id"),
                "fit_run_id": entry.get("fit_run_id"),
                "label": str(entry.get("label", "")),
                "hkl": normalized_hkl,
                "q_group_key": entry.get("q_group_key"),
                "seed_match_status": str(entry.get("match_status", "")),
                "match_status": str(entry.get("match_status", "")),
                "match_kind": str(entry.get("match_kind", "")),
                "resolution_kind": str(entry.get("resolution_kind", "")),
                "resolution_reason": str(entry.get("resolution_reason", "")),
                "fit_source_resolution_kind": entry.get("fit_source_resolution_kind"),
                "source_table_index": _diagnostic_source_table_index(entry),
                "source_reflection_index": source_reflection_index,
                "source_reflection_namespace": source_reflection_namespace,
                "source_reflection_is_full": source_reflection_is_full,
                "source_row_index": entry.get("source_row_index"),
                "source_branch_index": source_branch_index,
                "source_branch_resolution_source": source_branch_source,
                "source_peak_index": entry.get(
                    "source_peak_index",
                    entry.get("resolved_peak_index"),
                ),
                "resolved_table_index": resolved_table_index,
                "resolved_peak_index": entry.get(
                    "resolved_peak_index",
                    entry.get("source_peak_index"),
                ),
                "measured_x": float(entry.get("measured_x", np.nan)),
                "measured_y": float(entry.get("measured_y", np.nan)),
                "placement_error_px": float(entry.get("placement_error_px", np.nan)),
                "sigma_is_custom": bool(entry.get("sigma_is_custom", False)),
                "seed_distance_px": float(entry.get("distance_px", np.nan)),
                **frozen_locator,
            }
            sigma_px = float(entry.get("measurement_sigma_px", np.nan))
            if np.isfinite(sigma_px) and sigma_px > 0.0:
                grouped[int(dataset_index)][int(overlay_match_index)][
                    "reported_measurement_sigma_px"
                ] = float(sigma_px)
            if bool(entry.get("sigma_is_custom", False)):
                grouped[int(dataset_index)][int(overlay_match_index)]["sigma_px"] = float(
                    entry.get("measurement_sigma_px", np.nan)
                )
            if bool(entry.get("anisotropic_sigma_used", False)):
                grouped[int(dataset_index)][int(overlay_match_index)]["sigma_radial_px"] = float(
                    entry.get("sigma_radial_px", np.nan)
                )
                grouped[int(dataset_index)][int(overlay_match_index)]["sigma_tangential_px"] = (
                    float(entry.get("sigma_tangential_px", np.nan))
                )

        out: Dict[int, List[Dict[str, object]]] = {}
        for dataset_ctx in dataset_contexts:
            dataset_index = int(dataset_ctx.dataset_index)
            records_by_overlay = grouped.get(dataset_index, {})
            out[dataset_index] = [
                dict(records_by_overlay[key]) for key in sorted(records_by_overlay)
            ]
        return out

    def _evaluate_full_beam_point_matches(
        local: Dict[str, object],
        fixed_correspondence_map: Mapping[int, Sequence[Mapping[str, object]]],
        *,
        collect_diagnostics: bool = False,
    ) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
        if not dataset_contexts:
            return (
                np.array([], dtype=float),
                [],
                {
                    "dataset_count": 0,
                    "measured_count": 0,
                    "fixed_source_resolved_count": 0,
                    "fallback_entry_count": 0,
                    "matched_pair_count": 0,
                    "missing_pair_count": 0,
                    "simulated_reflection_count": 0,
                    "total_reflection_count": 0,
                    "fixed_source_reflection_count": 0,
                    "subset_fallback_hkl_count": 0,
                    "subset_reduced": False,
                    "central_ray_mode": False,
                    "single_ray_enabled": False,
                    "single_ray_forced_count": 0,
                    "match_radius_px": float(full_beam_polish_match_radius_px),
                },
            )

        def _evaluate_full_beam_for_dataset(
            item: Tuple[Dict[str, object], GeometryFitDatasetContext],
        ) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
            local_item, dataset_ctx = item
            theta_value = _theta_initial_for_dataset(local_item, dataset_ctx)
            correspondence_entries = [
                dict(entry)
                for entry in fixed_correspondence_map.get(int(dataset_ctx.dataset_index), [])
                if isinstance(entry, Mapping)
            ]
            if not correspondence_entries:
                return (
                    np.array([], dtype=float),
                    [],
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "theta_initial_deg": float(theta_value),
                        "measured_count": 0,
                        "fixed_source_resolved_count": 0,
                        "fallback_entry_count": 0,
                        "matched_pair_count": 0,
                        "missing_pair_count": 0,
                        "simulated_reflection_count": int(dataset_ctx.subset.miller.shape[0]),
                        "total_reflection_count": int(dataset_ctx.subset.total_reflection_count),
                        "fixed_source_reflection_count": int(
                            dataset_ctx.subset.fixed_source_reflection_count
                        ),
                        "subset_fallback_hkl_count": int(dataset_ctx.subset.fallback_hkl_count),
                        "subset_reduced": bool(dataset_ctx.subset.reduced),
                        "central_ray_mode": False,
                        "single_ray_enabled": False,
                        "single_ray_forced_count": 0,
                        "match_radius_px": float(full_beam_polish_match_radius_px),
                    },
                )

            local_full = dict(local_item)
            local_full["mosaic_params"] = full_beam_mosaic_params
            try:
                hk0_peak_priority_weight = float(local_full.get("_hk0_peak_priority_weight", 1.0))
            except Exception:
                hk0_peak_priority_weight = 1.0
            if not np.isfinite(hk0_peak_priority_weight) or hk0_peak_priority_weight < 1.0:
                hk0_peak_priority_weight = 1.0
            fit_miller = dataset_ctx.subset.miller
            fit_intensities = dataset_ctx.subset.intensities
            mosaic = local_full["mosaic_params"]
            wavelength_array = mosaic.get("wavelength_array")
            if wavelength_array is None:
                wavelength_array = mosaic.get("wavelength_i_array")

            sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
            _, hit_tables, *_ = _process_peaks_parallel_safe(
                fit_miller,
                fit_intensities,
                image_size,
                local_full["a"],
                local_full["c"],
                wavelength_array,
                sim_buffer,
                local_full["corto_detector"],
                local_full["gamma"],
                local_full["Gamma"],
                local_full["chi"],
                local_full.get("psi", 0.0),
                local_full.get("psi_z", 0.0),
                local_full["zs"],
                local_full["zb"],
                local_full["n2"],
                mosaic["beam_x_array"],
                mosaic["beam_y_array"],
                mosaic["theta_array"],
                mosaic["phi_array"],
                mosaic["sigma_mosaic_deg"],
                mosaic["gamma_mosaic_deg"],
                mosaic["eta"],
                wavelength_array,
                local_full["debye_x"],
                local_full["debye_y"],
                local_full["center"],
                theta_value,
                local_full.get("cor_angle", 0.0),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                save_flag=0,
                **_simulation_kernel_kwargs(local_full, mosaic),
            )
            maxpos = hit_tables_to_max_positions(hit_tables)
            filtered_rows_cache: Dict[int, List[np.ndarray]] = {}
            current_local_table_signature = _local_table_signature(
                dataset_ctx.subset.original_indices
            )
            residual_components = np.zeros((len(correspondence_entries), 2), dtype=np.float64)
            diagnostics_local: List[Dict[str, object]] = []
            custom_sigma_values: List[float] = []
            matched_distances: List[float] = []
            anisotropic_sigma_count = 0
            matched_pair_count_local = 0
            missing_pair_count_local = 0

            fallback_hkls: set[Tuple[int, int, int]] = set()
            fixed_source_count_local = 0
            for entry in correspondence_entries:
                if str(entry.get("frozen_locator_kind", "")).strip().lower() in {
                    "trusted_branch",
                    "local_branch",
                    "local_row",
                }:
                    fixed_source_count_local += 1
                else:
                    raw_hkl = entry.get("hkl")
                    if isinstance(raw_hkl, tuple) and len(raw_hkl) == 3:
                        try:
                            fallback_hkls.add(tuple(int(v) for v in raw_hkl))
                        except Exception:
                            pass

            def _point_radius_px(point: Tuple[float, float]) -> float:
                try:
                    center_row = float(local_full["center"][0])
                    center_col = float(local_full["center"][1])
                except Exception:
                    return float("nan")
                if not (
                    np.isfinite(point[0])
                    and np.isfinite(point[1])
                    and np.isfinite(center_row)
                    and np.isfinite(center_col)
                ):
                    return float("nan")
                return float(math.hypot(float(point[0]) - center_col, float(point[1]) - center_row))

            def _weight_fields(
                entry: Dict[str, object],
                *,
                measured_point: Tuple[float, float],
                dx: float = 0.0,
                dy: float = 0.0,
                distance_weight: float = 1.0,
            ) -> Dict[str, float | bool]:
                nonlocal anisotropic_sigma_count
                fields = _weight_measurement_residual(
                    dx,
                    dy,
                    measured_point=measured_point,
                    center=local_full.get("center", []),
                    entry=entry,
                    distance_weight=distance_weight,
                    use_measurement_uncertainty=bool(use_measurement_uncertainty),
                    anisotropic_enabled=bool(anisotropic_uncertainty_enabled),
                    radial_scale=float(radial_sigma_scale),
                    tangential_scale=float(tangential_sigma_scale),
                )
                if bool(fields.get("sigma_is_custom", False)):
                    custom_sigma_values.append(float(fields.get("measurement_sigma_px", np.nan)))
                if bool(fields.get("anisotropic_sigma_used", False)):
                    anisotropic_sigma_count += 1
                priority_fields = _geometry_peak_priority_metadata(
                    entry,
                    hk0_peak_priority_weight=float(hk0_peak_priority_weight),
                )
                priority_weight = float(priority_fields["priority_weight"])
                for key in (
                    "weighted_dx_px",
                    "weighted_dy_px",
                    "weighted_radial_residual_px",
                    "weighted_tangential_residual_px",
                ):
                    try:
                        value = float(fields.get(key, np.nan))
                    except Exception:
                        continue
                    if np.isfinite(value):
                        fields[key] = float(priority_weight * value)
                fields.update(priority_fields)
                return fields

            def _add_diag(base_entry: Dict[str, object], payload: Dict[str, object]) -> None:
                if not collect_diagnostics:
                    return
                diag = dict(base_entry)
                diag.update(payload)
                diag["dataset_index"] = int(dataset_ctx.dataset_index)
                diag["dataset_label"] = str(dataset_ctx.label)
                diag["theta_initial_deg"] = float(theta_value)
                diagnostics_local.append(diag)

            def _resolve_simulated_point(
                entry: Dict[str, object],
            ) -> Tuple[Tuple[float, float] | None, Dict[str, object]]:
                if str(entry.get("seed_match_status", "")).lower() != "matched":
                    return None, {"resolution_reason": "seed_missing_pair"}
                return _resolve_geometry_fit_correspondence(
                    entry,
                    hit_tables=hit_tables,
                    max_positions=maxpos,
                    filtered_rows_cache=filtered_rows_cache,
                    current_local_table_signature=current_local_table_signature,
                )

            for slot, correspondence in enumerate(correspondence_entries):
                entry = dict(correspondence)
                source_branch_index, source_branch_source = _measured_source_peak_index_with_source(
                    entry
                )
                measured_point = (
                    float(entry.get("measured_x", np.nan)),
                    float(entry.get("measured_y", np.nan)),
                )
                base_diag = {
                    "match_input_index": int(entry.get("match_input_index", slot)),
                    "overlay_match_index": int(entry.get("overlay_match_index", slot)),
                    "pair_id": entry.get("pair_id"),
                    "fit_run_id": entry.get("fit_run_id"),
                    "label": str(entry.get("label", "")),
                "hkl": entry.get("hkl"),
                "source_table_index": _diagnostic_source_table_index(entry),
                "source_reflection_index": entry.get("source_reflection_index"),
                "source_reflection_namespace": entry.get(
                    "source_reflection_namespace"
                ),
                "source_reflection_is_full": entry.get(
                    "source_reflection_is_full"
                ),
                "source_row_index": entry.get("source_row_index"),
                "source_branch_index": source_branch_index,
                "source_branch_resolution_source": source_branch_source,
                "source_peak_index": entry.get(
                    "source_peak_index",
                    entry.get("resolved_peak_index"),
                ),
                "resolved_table_index": entry.get("resolved_table_index"),
                "resolved_peak_index": entry.get("resolved_peak_index"),
                "resolution_kind": str(entry.get("resolution_kind", "")),
                    "fit_source_resolution_kind": entry.get("fit_source_resolution_kind"),
                    "correspondence_resolution_reason": (
                        None
                        if entry.get("resolution_reason") in ("", None)
                        else str(entry.get("resolution_reason"))
                    ),
                }
                if not (np.isfinite(measured_point[0]) and np.isfinite(measured_point[1])):
                    priority_fields = _geometry_peak_priority_metadata(
                        entry,
                        hk0_peak_priority_weight=float(hk0_peak_priority_weight),
                    )
                    priority_weight = float(priority_fields["priority_weight"])
                    missing_pair_count_local += 1
                    residual_components[slot, 0] = float(missing_pair_penalty) * float(
                        priority_weight
                    )
                    _add_diag(
                        base_diag,
                        {
                            "match_kind": "full_beam_fixed",
                            "match_status": "missing_pair",
                            "resolution_reason": "invalid_measured_point",
                            "measured_x": float(measured_point[0]),
                            "measured_y": float(measured_point[1]),
                            "simulated_x": float("nan"),
                            "simulated_y": float("nan"),
                            "dx_px": float("nan"),
                            "dy_px": float("nan"),
                            "distance_px": float("nan"),
                            "measured_radius_px": float("nan"),
                            "simulated_radius_px": float("nan"),
                            "weighted_missing_penalty_px": float(missing_pair_penalty)
                            * float(priority_weight),
                            "weight": float(priority_weight),
                            **priority_fields,
                        },
                    )
                    continue

                sim_point, resolution_payload = _resolve_simulated_point(entry)
                if sim_point is not None:
                    dx = float(sim_point[0] - measured_point[0])
                    dy = float(sim_point[1] - measured_point[1])
                    pair_dist = float(math.hypot(dx, dy))
                    if pair_dist <= float(full_beam_polish_match_radius_px):
                        matched_pair_count_local += 1
                        matched_distances.append(float(pair_dist))
                        if weighted_matching:
                            distance_weight = 1.0 / math.sqrt(
                                1.0 + (pair_dist / solver_f_scale) ** 2
                            )
                        else:
                            distance_weight = 1.0
                        weight_fields = _weight_fields(
                            entry,
                            measured_point=measured_point,
                            dx=dx,
                            dy=dy,
                            distance_weight=float(distance_weight),
                        )
                        residual_components[slot, 0] = float(weight_fields["weighted_dx_px"])
                        residual_components[slot, 1] = float(weight_fields["weighted_dy_px"])
                        _add_diag(
                            base_diag,
                            {
                                "match_kind": "full_beam_fixed",
                                "match_status": "matched",
                                "resolution_reason": str(
                                    resolution_payload.get("resolution_reason", "resolved")
                                ),
                                "resolved_table_index": resolution_payload.get(
                                    "resolved_table_index"
                                ),
                                "resolved_peak_index": resolution_payload.get(
                                    "resolved_peak_index"
                                ),
                                "resolved_sim_hkl": resolution_payload.get("resolved_sim_hkl"),
                                "measured_x": float(measured_point[0]),
                                "measured_y": float(measured_point[1]),
                                "simulated_x": float(sim_point[0]),
                                "simulated_y": float(sim_point[1]),
                                "dx_px": float(dx),
                                "dy_px": float(dy),
                                "distance_px": float(pair_dist),
                                "placement_error_px": float(
                                    entry.get("placement_error_px", np.nan)
                                ),
                                "distance_weight": float(distance_weight),
                                "weight": float(
                                    float(distance_weight)
                                    * float(weight_fields.get("sigma_weight", 1.0))
                                    * float(weight_fields.get("priority_weight", 1.0))
                                ),
                                "measured_radius_px": _point_radius_px(measured_point),
                                "simulated_radius_px": _point_radius_px(sim_point),
                                **weight_fields,
                            },
                        )
                        continue
                    resolution_payload = dict(resolution_payload)
                    resolution_payload["resolution_reason"] = "outside_match_radius"

                missing_pair_count_local += 1
                weight_fields = _weight_fields(
                    entry,
                    measured_point=measured_point,
                    distance_weight=1.0,
                )
                penalty_weight = float(weight_fields.get("sigma_weight", 1.0))
                penalty_weight *= float(weight_fields.get("priority_weight", 1.0))
                residual_components[slot, 0] = float(missing_pair_penalty) * penalty_weight
                residual_components[slot, 1] = 0.0
                _add_diag(
                    base_diag,
                    {
                        "match_kind": "full_beam_fixed",
                        "match_status": "missing_pair",
                        "resolution_reason": str(
                            resolution_payload.get("resolution_reason", "missing_pair")
                        ),
                        "resolved_table_index": resolution_payload.get("resolved_table_index"),
                        "resolved_peak_index": resolution_payload.get("resolved_peak_index"),
                        "resolved_sim_hkl": resolution_payload.get("resolved_sim_hkl"),
                        "measured_x": float(measured_point[0]),
                        "measured_y": float(measured_point[1]),
                        "simulated_x": float("nan"),
                        "simulated_y": float("nan"),
                        "dx_px": float("nan"),
                        "dy_px": float("nan"),
                        "distance_px": float("nan"),
                        "placement_error_px": float(entry.get("placement_error_px", np.nan)),
                        "distance_weight": 1.0,
                        "weight": float(penalty_weight),
                        "weighted_missing_penalty_px": float(missing_pair_penalty)
                        * float(penalty_weight),
                        "measured_radius_px": _point_radius_px(measured_point),
                        "simulated_radius_px": float("nan"),
                        **weight_fields,
                    },
                )

            residual_arr = residual_components.reshape(-1)
            summary_local: Dict[str, object] = {
                "dataset_index": int(dataset_ctx.dataset_index),
                "dataset_label": str(dataset_ctx.label),
                "theta_initial_deg": float(theta_value),
                "measured_count": int(len(correspondence_entries)),
                "fixed_source_resolved_count": int(fixed_source_count_local),
                "fallback_entry_count": int(len(correspondence_entries) - fixed_source_count_local),
                "fallback_hkl_count": int(len(fallback_hkls)),
                "matched_pair_count": int(matched_pair_count_local),
                "missing_pair_count": int(missing_pair_count_local),
                "simulated_reflection_count": int(fit_miller.shape[0]),
                "total_reflection_count": int(dataset_ctx.subset.total_reflection_count),
                "fixed_source_reflection_count": int(
                    dataset_ctx.subset.fixed_source_reflection_count
                ),
                "subset_fallback_hkl_count": int(dataset_ctx.subset.fallback_hkl_count),
                "subset_reduced": bool(dataset_ctx.subset.reduced),
                "central_ray_mode": bool(_uses_central_geometry_ray(full_beam_mosaic_params)),
                "single_ray_enabled": False,
                "single_ray_forced_count": 0,
                "center_row": float(local_full["center"][0])
                if len(local_full.get("center", [])) >= 2
                else float("nan"),
                "center_col": float(local_full["center"][1])
                if len(local_full.get("center", [])) >= 2
                else float("nan"),
                "match_radius_px": float(full_beam_polish_match_radius_px),
                "anisotropic_sigma_count": int(anisotropic_sigma_count),
            }
            if custom_sigma_values:
                sigma_arr = np.asarray(custom_sigma_values, dtype=float)
                sigma_arr = sigma_arr[np.isfinite(sigma_arr) & (sigma_arr > 0.0)]
                if sigma_arr.size:
                    summary_local["custom_sigma_count"] = int(sigma_arr.size)
                    summary_local["measurement_sigma_median_px"] = float(np.median(sigma_arr))
                    summary_local["measurement_sigma_mean_px"] = float(np.mean(sigma_arr))
                    summary_local["measurement_sigma_max_px"] = float(np.max(sigma_arr))
                else:
                    summary_local["custom_sigma_count"] = 0
            else:
                summary_local["custom_sigma_count"] = 0
            if matched_distances:
                matched_dist_arr = np.asarray(matched_distances, dtype=float)
                matched_dist_arr = matched_dist_arr[np.isfinite(matched_dist_arr)]
                if matched_dist_arr.size:
                    summary_local["unweighted_peak_rms_px"] = float(
                        np.sqrt(np.mean(matched_dist_arr * matched_dist_arr))
                    )
                    summary_local["unweighted_peak_mean_px"] = float(np.mean(matched_dist_arr))
                    summary_local["unweighted_peak_max_px"] = float(np.max(matched_dist_arr))
                else:
                    summary_local["unweighted_peak_rms_px"] = float("nan")
                    summary_local["unweighted_peak_mean_px"] = float("nan")
                    summary_local["unweighted_peak_max_px"] = float("nan")
            else:
                summary_local["unweighted_peak_rms_px"] = float("nan")
                summary_local["unweighted_peak_mean_px"] = float("nan")
                summary_local["unweighted_peak_max_px"] = float("nan")
            if anisotropic_sigma_count > 0:
                summary_local["peak_weighting_mode"] = (
                    "measurement_covariance+distance"
                    if weighted_matching
                    else "measurement_covariance"
                )
            elif summary_local.get("custom_sigma_count", 0):
                summary_local["peak_weighting_mode"] = (
                    "measurement_sigma+distance" if weighted_matching else "measurement_sigma"
                )
            else:
                summary_local["peak_weighting_mode"] = "uniform"
            return residual_arr, diagnostics_local, summary_local

        residual_blocks: List[np.ndarray] = []
        diagnostics: List[Dict[str, object]] = []
        per_dataset_summaries: List[Dict[str, object]] = []
        summary: Dict[str, object] = {
            "dataset_count": int(len(dataset_contexts)),
            "measured_count": 0,
            "fixed_source_resolved_count": 0,
            "fallback_entry_count": 0,
            "matched_pair_count": 0,
            "missing_pair_count": 0,
            "simulated_reflection_count": 0,
            "total_reflection_count": 0,
            "fixed_source_reflection_count": 0,
            "subset_fallback_hkl_count": 0,
            "subset_reduced": False,
            "central_ray_mode": False,
            "single_ray_enabled": False,
            "single_ray_forced_count": 0,
            "center_row": float(local["center"][0])
            if len(local.get("center", [])) >= 2
            else float("nan"),
            "center_col": float(local["center"][1])
            if len(local.get("center", [])) >= 2
            else float("nan"),
            "match_radius_px": float(full_beam_polish_match_radius_px),
        }
        dataset_items = [(local, dataset_ctx) for dataset_ctx in dataset_contexts]
        # Keep full-beam polish sequential across datasets so the final metric
        # matches the conservative GUI multi-background safety profile.
        dataset_results = [_evaluate_full_beam_for_dataset(item) for item in dataset_items]
        for residual_i, diagnostics_i, summary_i in dataset_results:
            residual_i = np.asarray(residual_i, dtype=float)
            if residual_i.size:
                residual_blocks.append(residual_i)
            per_dataset_summaries.append(dict(summary_i))
            for key in (
                "measured_count",
                "fixed_source_resolved_count",
                "fallback_entry_count",
                "matched_pair_count",
                "missing_pair_count",
                "simulated_reflection_count",
                "total_reflection_count",
                "fixed_source_reflection_count",
                "subset_fallback_hkl_count",
                "single_ray_forced_count",
            ):
                summary[key] = int(summary.get(key, 0)) + int(summary_i.get(key, 0))
            summary["subset_reduced"] = bool(
                summary.get("subset_reduced", False) or bool(summary_i.get("subset_reduced", False))
            )
            summary["central_ray_mode"] = bool(
                summary.get("central_ray_mode", False)
                or bool(summary_i.get("central_ray_mode", False))
            )
            if collect_diagnostics and diagnostics_i:
                diagnostics.extend(diagnostics_i)
        residual_arr = (
            np.concatenate(residual_blocks) if residual_blocks else np.array([], dtype=float)
        )
        summary["per_dataset"] = per_dataset_summaries
        custom_sigma_values = np.asarray(
            [
                float(entry.get("measurement_sigma_px", np.nan))
                for entry in diagnostics
                if bool(entry.get("sigma_is_custom", False))
            ],
            dtype=float,
        )
        custom_sigma_values = custom_sigma_values[
            np.isfinite(custom_sigma_values) & (custom_sigma_values > 0.0)
        ]
        summary["custom_sigma_count"] = int(custom_sigma_values.size)
        anisotropic_count = int(
            sum(
                int(summary_i.get("anisotropic_sigma_count", 0))
                for summary_i in per_dataset_summaries
            )
        )
        summary["anisotropic_sigma_count"] = int(anisotropic_count)
        if anisotropic_count > 0:
            summary["peak_weighting_mode"] = (
                "measurement_covariance+distance"
                if bool(weighted_matching)
                else "measurement_covariance"
            )
        elif custom_sigma_values.size:
            summary["peak_weighting_mode"] = (
                "measurement_sigma+distance" if bool(weighted_matching) else "measurement_sigma"
            )
        else:
            summary["peak_weighting_mode"] = "uniform"
        if custom_sigma_values.size:
            summary["measurement_sigma_median_px"] = float(np.median(custom_sigma_values))
            summary["measurement_sigma_mean_px"] = float(np.mean(custom_sigma_values))
            summary["measurement_sigma_max_px"] = float(np.max(custom_sigma_values))

        matched_distances = np.asarray(
            [
                float(entry.get("distance_px", np.nan))
                for entry in diagnostics
                if str(entry.get("match_status", "")).lower() == "matched"
            ],
            dtype=float,
        )
        matched_distances = matched_distances[np.isfinite(matched_distances)]
        if matched_distances.size:
            summary["unweighted_peak_rms_px"] = float(
                np.sqrt(np.mean(matched_distances * matched_distances))
            )
            summary["unweighted_peak_mean_px"] = float(np.mean(matched_distances))
            summary["unweighted_peak_max_px"] = float(np.max(matched_distances))
        else:
            matched_sq_sum = 0.0
            matched_sum = 0.0
            matched_max = float("nan")
            matched_count = int(summary.get("matched_pair_count", 0))
            has_sq_stat = False
            has_mean_stat = False
            for summary_i in per_dataset_summaries:
                try:
                    dataset_rms = float(summary_i.get("unweighted_peak_rms_px", np.nan))
                except Exception:
                    dataset_rms = float("nan")
                try:
                    dataset_mean = float(summary_i.get("unweighted_peak_mean_px", np.nan))
                except Exception:
                    dataset_mean = float("nan")
                try:
                    dataset_max = float(summary_i.get("unweighted_peak_max_px", np.nan))
                except Exception:
                    dataset_max = float("nan")
                dataset_count = int(summary_i.get("matched_pair_count", 0))
                if dataset_count <= 0:
                    continue
                if np.isfinite(dataset_rms):
                    matched_sq_sum += float(dataset_count) * float(dataset_rms) * float(dataset_rms)
                    has_sq_stat = True
                if np.isfinite(dataset_mean):
                    matched_sum += float(dataset_count) * float(dataset_mean)
                    has_mean_stat = True
                if np.isfinite(dataset_max):
                    if not np.isfinite(matched_max) or dataset_max > matched_max:
                        matched_max = float(dataset_max)
            if matched_count > 0 and has_sq_stat:
                summary["unweighted_peak_rms_px"] = float(
                    np.sqrt(matched_sq_sum / float(matched_count))
                )
            else:
                summary["unweighted_peak_rms_px"] = float("nan")
            if matched_count > 0 and has_mean_stat:
                summary["unweighted_peak_mean_px"] = float(matched_sum / float(matched_count))
            else:
                summary["unweighted_peak_mean_px"] = float("nan")
            summary["unweighted_peak_max_px"] = float(matched_max)
        return residual_arr, diagnostics, summary

    def _maybe_run_full_beam_polish(
        current_result: OptimizeResult,
    ) -> Dict[str, object]:
        nonlocal final_metric_residual_fn
        nonlocal final_metric_point_match_evaluator
        summary: Dict[str, object] = {
            "enabled": bool(full_beam_polish_enabled),
            "status": "skipped",
            "reason": "",
            "accepted": False,
            "match_radius_px": float(full_beam_polish_match_radius_px),
            "max_nfev": int(full_beam_polish_max_nfev),
        }
        if not point_match_mode:
            summary["reason"] = "point_match_mode_disabled"
            return summary
        if not dataset_contexts:
            summary["reason"] = "no_datasets"
            return summary
        if getattr(current_result, "x", None) is None:
            summary["reason"] = "missing_parameter_vector"
            return summary
        if not full_beam_polish_enabled:
            summary["reason"] = "disabled_by_config"
            return summary
        if not isinstance(full_beam_mosaic_params, Mapping) or not full_beam_mosaic_params:
            summary["reason"] = "missing_full_beam_mosaic"
            return summary

        start_x = np.asarray(current_result.x, dtype=float)
        start_local = _apply_trial_params(start_x)
        _, seed_diagnostics, seed_summary = _evaluate_pixel_matches(
            start_local,
            collect_diagnostics=True,
        )
        fixed_correspondence_map = _build_full_beam_seed_correspondence_map(seed_diagnostics)
        correspondence_count = int(
            sum(len(entries) for entries in fixed_correspondence_map.values())
        )
        summary["seed_correspondence_count"] = int(correspondence_count)
        summary["seed_correspondence_records"] = [
            dict(entry)
            for dataset_entries in fixed_correspondence_map.values()
            for entry in dataset_entries
            if isinstance(entry, Mapping)
        ]
        summary["seed_matched_pair_count"] = int(seed_summary.get("matched_pair_count", 0))
        summary["seed_rms_px"] = float(seed_summary.get("unweighted_peak_rms_px", np.nan))
        if correspondence_count <= 0:
            summary["reason"] = "no_seed_correspondences"
            return summary

        def _full_beam_point_match_evaluator(
            local_trial: Dict[str, object],
            *,
            collect_diagnostics: bool = False,
        ) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
            return _evaluate_full_beam_point_matches(
                local_trial,
                fixed_correspondence_map,
                collect_diagnostics=collect_diagnostics,
            )

        def _full_beam_residual(x_trial: np.ndarray) -> np.ndarray:
            local_trial = _apply_trial_params(np.asarray(x_trial, dtype=float))
            residual_arr, _, _ = _full_beam_point_match_evaluator(
                local_trial,
                collect_diagnostics=False,
            )
            prior_residual = _parameter_prior_residuals(np.asarray(x_trial, dtype=float))
            if prior_residual.size:
                if residual_arr.size:
                    return np.concatenate(
                        [
                            np.asarray(residual_arr, dtype=float),
                            np.asarray(prior_residual, dtype=float),
                        ]
                    )
                return np.asarray(prior_residual, dtype=float)
            return np.asarray(residual_arr, dtype=float)

        start_residual, start_pm_diagnostics, start_pm_summary = _full_beam_point_match_evaluator(
            start_local,
            collect_diagnostics=True,
        )
        start_prior = _parameter_prior_residuals(start_x)
        if start_prior.size:
            if start_residual.size:
                start_fun = np.concatenate(
                    [np.asarray(start_residual, dtype=float), np.asarray(start_prior, dtype=float)]
                )
            else:
                start_fun = np.asarray(start_prior, dtype=float)
        else:
            start_fun = np.asarray(start_residual, dtype=float)
        start_cost = float(_robust_cost(start_fun, loss=solver_loss, f_scale=solver_f_scale))
        selected_x = np.asarray(start_x, dtype=float)
        selected_fun = np.asarray(start_fun, dtype=float)
        selected_pm_diagnostics = list(start_pm_diagnostics)
        selected_pm_summary = dict(start_pm_summary)
        selected_cost = float(start_cost)
        start_matched = int(start_pm_summary.get("matched_pair_count", 0))
        summary["matched_pair_count_before"] = int(start_matched)
        summary["fixed_source_resolved_count"] = int(
            start_pm_summary.get("fixed_source_resolved_count", 0)
        )
        try:
            start_point_rms = float(start_pm_summary.get("unweighted_peak_rms_px", np.nan))
        except Exception:
            start_point_rms = float("nan")
        try:
            start_peak_max = float(start_pm_summary.get("unweighted_peak_max_px", np.nan))
        except Exception:
            start_peak_max = float("nan")
        summary["start_rms_px"] = float(start_point_rms)
        summary["start_peak_max_px"] = float(start_peak_max)

        if int(start_matched) <= 0:
            summary.update(
                {
                    "status": "rejected",
                    "reason": "no_seed_matches",
                    "start_cost": float(start_cost),
                    "final_cost": float(start_cost),
                    "matched_pair_count_after": int(
                        selected_pm_summary.get("matched_pair_count", 0)
                    ),
                    "point_match_diagnostics": selected_pm_diagnostics,
                    "point_match_summary": selected_pm_summary,
                    "fun": selected_fun,
                }
            )
            return summary

        _emit_status("Geometry fit: running full-beam polish")
        try:
            polish_result = _run_solver_with_max_nfev(
                start_x,
                max_nfev=full_beam_polish_max_nfev,
                residual_callable=_full_beam_residual,
                status_label="full-beam polish",
            )
            trial_x = np.minimum(
                np.maximum(np.asarray(polish_result.x, dtype=float), lower_bounds),
                upper_bounds,
            )
            trial_local = _apply_trial_params(trial_x)
            trial_residual, trial_pm_diagnostics, trial_pm_summary = (
                _full_beam_point_match_evaluator(
                    trial_local,
                    collect_diagnostics=True,
                )
            )
            trial_prior = _parameter_prior_residuals(trial_x)
            if trial_prior.size:
                if trial_residual.size:
                    trial_fun = np.concatenate(
                        [
                            np.asarray(trial_residual, dtype=float),
                            np.asarray(trial_prior, dtype=float),
                        ]
                    )
                else:
                    trial_fun = np.asarray(trial_prior, dtype=float)
            else:
                trial_fun = np.asarray(trial_residual, dtype=float)
            trial_cost = float(_robust_cost(trial_fun, loss=solver_loss, f_scale=solver_f_scale))
        except Exception as exc:
            summary.update(
                {
                    "status": "failed",
                    "reason": f"unexpected_exception: {exc}",
                    "start_cost": float(start_cost),
                    "final_cost": float(start_cost),
                    "matched_pair_count_after": int(
                        selected_pm_summary.get("matched_pair_count", 0)
                    ),
                    "point_match_diagnostics": selected_pm_diagnostics,
                    "point_match_summary": selected_pm_summary,
                    "fun": selected_fun,
                }
            )
            return summary

        improvement_tol = max(1.0e-9 * max(abs(float(start_cost)), 1.0), 1.0e-12)
        trial_matched = int(trial_pm_summary.get("matched_pair_count", 0))
        try:
            trial_point_rms = float(trial_pm_summary.get("unweighted_peak_rms_px", np.nan))
        except Exception:
            trial_point_rms = float("nan")
        try:
            trial_peak_max = float(trial_pm_summary.get("unweighted_peak_max_px", np.nan))
        except Exception:
            trial_peak_max = float("nan")
        point_rms_tol = max(
            1.0e-9 * max(abs(float(start_point_rms)), 1.0),
            1.0e-12,
        )
        peak_max_tol = max(
            1.0e-9 * max(abs(float(start_peak_max)), 1.0),
            1.0e-12,
        )
        cost_improved = bool(
            np.isfinite(trial_cost) and trial_cost <= float(start_cost) + improvement_tol
        )
        matched_ok = bool(trial_matched >= start_matched)
        point_rms_ok = bool(
            not np.isfinite(start_point_rms)
            or (
                np.isfinite(trial_point_rms)
                and trial_point_rms <= float(start_point_rms) + point_rms_tol
            )
        )
        peak_max_ok = bool(
            not np.isfinite(start_peak_max)
            or (
                np.isfinite(trial_peak_max)
                and trial_peak_max <= float(start_peak_max) + peak_max_tol
            )
        )
        accepted = bool(cost_improved and matched_ok and point_rms_ok and peak_max_ok)
        if accepted:
            selected_x = np.asarray(trial_x, dtype=float)
            selected_fun = np.asarray(trial_fun, dtype=float)
            selected_pm_diagnostics = list(trial_pm_diagnostics)
            selected_pm_summary = dict(trial_pm_summary)
            selected_cost = float(trial_cost)
            final_metric_residual_fn = _full_beam_residual
            final_metric_point_match_evaluator = _full_beam_point_match_evaluator
        summary.update(
            {
                "status": "accepted" if accepted else "rejected",
                "reason": (
                    "accepted"
                    if accepted
                    else ", ".join(
                        part
                        for ok, part in (
                            (cost_improved, "cost_regressed"),
                            (matched_ok, "matched_pairs_decreased"),
                            (point_rms_ok, "point_rms_regressed"),
                            (peak_max_ok, "peak_offset_regressed"),
                        )
                        if not ok
                    )
                ),
                "accepted": bool(accepted),
                "start_cost": float(start_cost),
                "final_cost": float(selected_cost),
                "candidate_cost": float(trial_cost),
                "start_rms_px": float(start_point_rms),
                "final_rms_px": float(trial_point_rms if accepted else start_point_rms),
                "start_peak_max_px": float(start_peak_max),
                "final_peak_max_px": float(trial_peak_max if accepted else start_peak_max),
                "nfev": int(getattr(polish_result, "nfev", 0)),
                "success": bool(getattr(polish_result, "success", False)),
                "message": str(getattr(polish_result, "message", "")),
                "matched_pair_count_after": int(selected_pm_summary.get("matched_pair_count", 0)),
                "candidate_matched_pair_count": int(trial_matched),
                "candidate_rms_px": float(trial_point_rms),
                "candidate_peak_max_px": float(trial_peak_max),
                "point_match_diagnostics": selected_pm_diagnostics,
                "point_match_summary": selected_pm_summary,
                "fun": selected_fun,
            }
        )
        if accepted:
            summary["x"] = selected_x
        return summary

    def _build_identifiability_summary_at_x(
        x_trial: np.ndarray,
        current_result: Optional[OptimizeResult] = None,
    ) -> Dict[str, object]:
        x_arr = np.asarray(x_trial, dtype=float)
        residual_arr = np.asarray(final_metric_residual_fn(x_arr), dtype=float)
        template_result = current_result or OptimizeResult()
        return _build_identifiability_summary(
            OptimizeResult(
                x=x_arr,
                fun=residual_arr,
                success=bool(getattr(template_result, "success", False)),
                status=int(getattr(template_result, "status", 0)),
                message=str(getattr(template_result, "message", "")),
                nfev=int(getattr(template_result, "nfev", 0)),
                active_mask=np.zeros_like(x_arr, dtype=int),
                optimality=float(getattr(template_result, "optimality", np.nan)),
            ),
            getattr(template_result, "point_match_diagnostics", None),
        )

    def _identifiability_scale_for_index(idx: int, x_ref: np.ndarray) -> float:
        scale_candidates = [
            float(x_scale[idx]) if idx < len(x_scale) else float("nan"),
            float(span[idx]) if idx < len(span) else float("nan"),
            abs(float(x_ref[idx])) if idx < len(x_ref) else float("nan"),
            1.0,
        ]
        finite_candidates = [
            value for value in scale_candidates if np.isfinite(value) and value > 0.0
        ]
        if not finite_candidates:
            return 1.0
        return max(float(max(finite_candidates)), 1.0e-6)

    def _condition_improved(
        start_condition: float,
        trial_condition: float,
    ) -> bool:
        if np.isfinite(trial_condition):
            if not np.isfinite(start_condition):
                return True
            if start_condition <= 1.0:
                return True
            return bool(trial_condition <= 0.8 * start_condition)
        return False

    def _identifiability_progressed(
        start_summary: Dict[str, object],
        trial_summary: Dict[str, object],
        *,
        start_cost: float,
        trial_cost: float,
        cost_tol: float,
    ) -> bool:
        if np.isfinite(trial_cost) and trial_cost + cost_tol < float(start_cost):
            return True
        if bool(start_summary.get("underconstrained", False)) and not bool(
            trial_summary.get("underconstrained", False)
        ):
            return True
        if _condition_improved(
            float(start_summary.get("condition_number", np.inf)),
            float(trial_summary.get("condition_number", np.inf)),
        ):
            return True
        start_recommended = len(start_summary.get("recommended_fixed_indices", []))
        trial_recommended = len(trial_summary.get("recommended_fixed_indices", []))
        if int(trial_recommended) < int(start_recommended):
            return True
        return False

    def _build_probe_result(
        x_trial: np.ndarray,
        residual: np.ndarray,
        *,
        message: str,
    ) -> OptimizeResult:
        trial_x = np.asarray(x_trial, dtype=float)
        return OptimizeResult(
            x=trial_x,
            fun=np.asarray(residual, dtype=float),
            success=False,
            status=0,
            message=message,
            nfev=0,
            active_mask=np.zeros_like(trial_x, dtype=int),
            optimality=float("nan"),
        )

    def _resolve_reparameterization_pairs() -> List[Dict[str, object]]:
        if not reparameterize_enabled or len(var_names) <= 1:
            return []

        if reparameterize_pairs_cfg is None:
            raw_pairs = list(reparameterize_default_pairs)
        elif isinstance(reparameterize_pairs_cfg, (list, tuple)):
            raw_pairs = list(reparameterize_pairs_cfg)
        else:
            raw_pairs = []

        if not raw_pairs:
            return []

        name_to_index: Dict[str, int] = {}
        for idx, name in enumerate(var_names):
            key = str(name).strip()
            if key and key not in name_to_index:
                name_to_index[key] = int(idx)
            lower_key = key.lower()
            if lower_key and lower_key not in name_to_index:
                name_to_index[lower_key] = int(idx)

        pair_specs: List[Dict[str, object]] = []
        used_indices: set[int] = set()
        for raw_pair in raw_pairs:
            pair_names = raw_pair
            if isinstance(raw_pair, dict):
                pair_names = raw_pair.get("pair", raw_pair.get("names"))
            if not isinstance(pair_names, (list, tuple)) or len(pair_names) < 2:
                continue
            name_i = str(pair_names[0]).strip()
            name_j = str(pair_names[1]).strip()
            idx_i = name_to_index.get(name_i, name_to_index.get(name_i.lower(), -1))
            idx_j = name_to_index.get(name_j, name_to_index.get(name_j.lower(), -1))
            if (
                idx_i is None
                or idx_j is None
                or int(idx_i) < 0
                or int(idx_j) < 0
                or int(idx_i) == int(idx_j)
            ):
                continue
            if int(idx_i) in used_indices or int(idx_j) in used_indices:
                continue
            used_indices.add(int(idx_i))
            used_indices.add(int(idx_j))
            pair_specs.append(
                {
                    "index_i": int(idx_i),
                    "index_j": int(idx_j),
                    "name_i": str(var_names[int(idx_i)]),
                    "name_j": str(var_names[int(idx_j)]),
                    "mean_name": (f"{str(var_names[int(idx_i)])}+{str(var_names[int(idx_j)])}/2"),
                    "half_delta_name": (
                        f"{str(var_names[int(idx_i)])}-{str(var_names[int(idx_j)])}/2"
                    ),
                }
            )
        return pair_specs

    def _physical_to_reparameterized_vector(
        x_physical: np.ndarray,
        pair_specs: Sequence[Dict[str, object]],
    ) -> np.ndarray:
        out = np.asarray(x_physical, dtype=float).copy()
        for pair_spec in pair_specs:
            idx_i = int(pair_spec["index_i"])
            idx_j = int(pair_spec["index_j"])
            val_i = float(out[idx_i])
            val_j = float(out[idx_j])
            out[idx_i] = 0.5 * (val_i + val_j)
            out[idx_j] = 0.5 * (val_i - val_j)
        return out

    def _reparameterized_to_physical_vector(
        x_reparameterized: np.ndarray,
        pair_specs: Sequence[Dict[str, object]],
    ) -> np.ndarray:
        out = np.asarray(x_reparameterized, dtype=float).copy()
        for pair_spec in pair_specs:
            idx_i = int(pair_spec["index_i"])
            idx_j = int(pair_spec["index_j"])
            mean_val = float(out[idx_i])
            half_delta_val = float(out[idx_j])
            out[idx_i] = mean_val + half_delta_val
            out[idx_j] = mean_val - half_delta_val
        return out

    def _reparameterized_bound_residuals(x_physical: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_physical, dtype=float)
        scale = np.maximum(np.asarray(x_scale, dtype=float), 1.0e-6)
        penalties: List[np.ndarray] = []
        finite_lower = np.isfinite(lower_bounds)
        if np.any(finite_lower):
            penalties.append(
                np.maximum(lower_bounds[finite_lower] - x_arr[finite_lower], 0.0)
                / scale[finite_lower]
            )
        finite_upper = np.isfinite(upper_bounds)
        if np.any(finite_upper):
            penalties.append(
                np.maximum(x_arr[finite_upper] - upper_bounds[finite_upper], 0.0)
                / scale[finite_upper]
            )
        if not penalties:
            return np.array([], dtype=float)
        return np.concatenate([np.asarray(entry, dtype=float) for entry in penalties])

    def _build_reparameterized_geometry(
        pair_specs: Sequence[Dict[str, object]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower = np.asarray(lower_bounds, dtype=float).copy()
        upper = np.asarray(upper_bounds, dtype=float).copy()
        scale = np.maximum(np.asarray(x_scale, dtype=float).copy(), 1.0e-6)
        for pair_spec in pair_specs:
            idx_i = int(pair_spec["index_i"])
            idx_j = int(pair_spec["index_j"])
            lo_i = float(lower_bounds[idx_i])
            hi_i = float(upper_bounds[idx_i])
            lo_j = float(lower_bounds[idx_j])
            hi_j = float(upper_bounds[idx_j])

            if np.isfinite(lo_i) and np.isfinite(lo_j):
                lower[idx_i] = 0.5 * (lo_i + lo_j)
            else:
                lower[idx_i] = -np.inf
            if np.isfinite(hi_i) and np.isfinite(hi_j):
                upper[idx_i] = 0.5 * (hi_i + hi_j)
            else:
                upper[idx_i] = np.inf

            if np.isfinite(lo_i) and np.isfinite(hi_j):
                lower[idx_j] = 0.5 * (lo_i - hi_j)
            else:
                lower[idx_j] = -np.inf
            if np.isfinite(hi_i) and np.isfinite(lo_j):
                upper[idx_j] = 0.5 * (hi_i - lo_j)
            else:
                upper[idx_j] = np.inf

            mean_scale = max(
                0.5 * (float(scale[idx_i]) + float(scale[idx_j])),
                1.0e-6,
            )
            delta_scale = max(
                0.5 * (float(scale[idx_i]) + float(scale[idx_j])),
                1.0e-6,
            )
            scale[idx_i] = mean_scale
            scale[idx_j] = delta_scale
        return lower, upper, scale

    def _maybe_run_reparameterization_seed(
        x_start: np.ndarray,
    ) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "enabled": bool(reparameterize_enabled),
            "status": "skipped",
            "reason": "",
            "accepted": False,
            "pairs": [],
        }
        if not reparameterize_enabled:
            summary["reason"] = "disabled_by_config"
            return summary

        start_x = np.asarray(x_start, dtype=float)
        if start_x.ndim != 1 or start_x.size <= 1:
            summary["reason"] = "insufficient_parameter_count"
            return summary

        pair_specs = _resolve_reparameterization_pairs()
        if not pair_specs:
            summary["reason"] = "no_supported_pairs"
            return summary

        summary["pairs"] = [
            [str(pair_spec["name_i"]), str(pair_spec["name_j"])] for pair_spec in pair_specs
        ]
        summary["transformed_parameters"] = [
            str(pair_spec["mean_name"]) for pair_spec in pair_specs
        ] + [str(pair_spec["half_delta_name"]) for pair_spec in pair_specs]

        start_residual = np.asarray(residual_fn(start_x), dtype=float)
        if start_residual.ndim != 1 or start_residual.size == 0:
            summary["reason"] = "empty_residual"
            return summary
        start_cost = _robust_cost(
            start_residual,
            loss=solver_loss,
            f_scale=solver_f_scale,
        )
        summary["start_cost"] = float(start_cost)

        transformed_start = _physical_to_reparameterized_vector(start_x, pair_specs)
        reparam_lower_bounds, reparam_upper_bounds, reparam_x_scale = (
            _build_reparameterized_geometry(pair_specs)
        )
        _emit_status(f"Geometry fit: reparameterizing pairs {summary['pairs']}")

        def _reparameterized_residual(x_trial: np.ndarray) -> np.ndarray:
            physical_trial = _reparameterized_to_physical_vector(
                np.asarray(x_trial, dtype=float),
                pair_specs,
            )
            clipped_trial = np.minimum(
                np.maximum(physical_trial, lower_bounds),
                upper_bounds,
            )
            residual = np.asarray(
                residual_fn(np.asarray(clipped_trial, dtype=float)),
                dtype=float,
            )
            penalty = _reparameterized_bound_residuals(physical_trial)
            if penalty.size:
                if residual.size:
                    return np.concatenate((residual, penalty))
                return np.asarray(penalty, dtype=float)
            return residual

        try:
            stage_result = least_squares(
                _reparameterized_residual,
                np.asarray(transformed_start, dtype=float),
                bounds=(
                    np.asarray(reparam_lower_bounds, dtype=float),
                    np.asarray(reparam_upper_bounds, dtype=float),
                ),
                x_scale=np.asarray(reparam_x_scale, dtype=float),
                loss=solver_loss,
                f_scale=solver_f_scale,
                max_nfev=reparameterize_max_nfev,
            )
        except Exception as exc:
            summary["status"] = "failed"
            summary["reason"] = f"reparameterization_failed: {exc}"
            _emit_status(f"Geometry fit: pair reparameterization failed ({exc})")
            return summary

        final_physical = _reparameterized_to_physical_vector(
            np.asarray(stage_result.x, dtype=float),
            pair_specs,
        )
        final_physical = np.minimum(np.maximum(final_physical, lower_bounds), upper_bounds)
        final_residual = np.asarray(
            residual_fn(np.asarray(final_physical, dtype=float)),
            dtype=float,
        )
        final_cost = _robust_cost(
            final_residual,
            loss=solver_loss,
            f_scale=solver_f_scale,
        )
        cost_tol = max(
            1.0e-9 * max(abs(float(start_cost)), 1.0),
            1.0e-12,
        )
        if np.isfinite(start_cost) and start_cost > 1.0e-12:
            cost_limit = float(start_cost * (1.0 + reparameterize_max_cost_increase_fraction))
        else:
            cost_limit = float(start_cost + cost_tol)
        accepted = bool(np.isfinite(final_cost) and final_cost <= cost_limit + cost_tol)
        summary.update(
            {
                "status": "accepted" if accepted else "rejected",
                "reason": "accepted" if accepted else "cost_regressed",
                "accepted": bool(accepted),
                "final_cost": float(final_cost),
                "cost_limit": float(cost_limit),
                "nfev": int(getattr(stage_result, "nfev", 0)),
                "success": bool(getattr(stage_result, "success", False)),
                "message": str(getattr(stage_result, "message", "")),
                "max_nfev": int(reparameterize_max_nfev),
                "start_x": np.asarray(start_x, dtype=float).tolist(),
                "end_x": np.asarray(final_physical, dtype=float).tolist(),
            }
        )
        if accepted:
            summary["x"] = np.asarray(final_physical, dtype=float)
            _emit_status(
                f"Geometry fit: pair reparameterization accepted (cost={float(final_cost):.6f})"
            )
        else:
            _emit_status(
                f"Geometry fit: pair reparameterization rejected (cost={float(final_cost):.6f})"
            )
        return summary

    def _resolve_staged_release_blocks() -> List[Dict[str, object]]:
        if not staged_release_enabled or len(var_names) <= 1:
            return []

        if staged_release_blocks_cfg is None:
            raw_blocks = list(staged_release_default_blocks)
        elif isinstance(staged_release_blocks_cfg, (list, tuple)):
            raw_blocks = list(staged_release_blocks_cfg)
        else:
            raw_blocks = []

        if not raw_blocks:
            return []

        name_to_indices: Dict[str, List[int]] = {}
        for idx, name in enumerate(var_names):
            key = str(name)
            name_to_indices.setdefault(key, []).append(int(idx))
            name_to_indices.setdefault(key.lower(), []).append(int(idx))

        valid_group_tokens = {"center", "tilt", "distance", "lattice", "other"}
        stage_specs: List[Dict[str, object]] = []
        prior_key: Optional[Tuple[int, ...]] = None
        cumulative_indices: set[int] = set()

        for raw_block in raw_blocks:
            if isinstance(raw_block, str):
                tokens = [raw_block]
            elif isinstance(raw_block, (list, tuple, set)):
                tokens = list(raw_block)
            else:
                continue

            active_set = set(cumulative_indices)
            label_tokens: List[str] = []
            for raw_token in tokens:
                token = str(raw_token).strip()
                if not token:
                    continue
                label_tokens.append(token)
                token_key = token.lower()
                if token_key in {"remaining", "rest", "all"}:
                    active_set.update(range(len(var_names)))
                    continue
                if token_key in valid_group_tokens:
                    for idx, name in enumerate(var_names):
                        if _parameter_group(str(name)) == token_key:
                            active_set.add(int(idx))
                    continue
                active_set.update(name_to_indices.get(token, []))
                active_set.update(name_to_indices.get(token_key, []))

            cumulative_indices = set(active_set)
            if not active_set:
                continue

            active_key = tuple(sorted(int(idx) for idx in active_set))
            if prior_key is not None and active_key == prior_key:
                continue
            prior_key = active_key
            if len(active_key) >= len(var_names):
                continue

            active_indices = np.asarray(active_key, dtype=np.int64)
            fixed_indices = np.asarray(
                [idx for idx in range(len(var_names)) if idx not in active_set],
                dtype=np.int64,
            )
            stage_specs.append(
                {
                    "active_indices": active_indices,
                    "fixed_indices": fixed_indices,
                    "active_parameters": [str(var_names[idx]) for idx in active_indices.tolist()],
                    "fixed_parameters": [str(var_names[idx]) for idx in fixed_indices.tolist()],
                    "label": ", ".join(label_tokens) if label_tokens else "auto",
                }
            )

        return stage_specs

    def _maybe_run_staged_release(
        x_start: np.ndarray,
    ) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "enabled": bool(staged_release_enabled),
            "status": "skipped",
            "reason": "",
            "accepted": False,
            "stages": [],
        }
        if not staged_release_enabled:
            summary["reason"] = "disabled_by_config"
            return summary

        x_seed = np.asarray(x_start, dtype=float)
        if x_seed.ndim != 1 or x_seed.size <= 1:
            summary["reason"] = "insufficient_parameter_count"
            return summary

        stage_specs = _resolve_staged_release_blocks()
        summary["planned_stage_count"] = int(len(stage_specs))
        if not stage_specs:
            summary["reason"] = "no_reduced_stage_blocks"
            return summary

        _emit_status(f"Geometry fit: staged release enabled ({len(stage_specs)} stage(s))")
        current_x = np.asarray(x_seed, dtype=float).copy()
        accepted_stage_count = 0
        any_failed = False

        for stage_number, stage_spec in enumerate(stage_specs, start=1):
            active_indices = np.asarray(stage_spec["active_indices"], dtype=np.int64)
            fixed_indices = np.asarray(stage_spec["fixed_indices"], dtype=np.int64)
            stage_start = np.asarray(current_x, dtype=float).copy()
            stage_start_residual = np.asarray(residual_fn(stage_start), dtype=float)
            _emit_status(
                "Geometry fit: staged stage "
                f"{stage_number}/{len(stage_specs)} "
                f"active={list(stage_spec.get('active_parameters', []))}"
            )
            if stage_start_residual.ndim != 1 or stage_start_residual.size == 0:
                summary["stages"].append(
                    {
                        "stage": int(stage_number),
                        "label": str(stage_spec.get("label", "auto")),
                        "status": "failed",
                        "reason": "empty_residual",
                        "accepted": False,
                        "active_parameters": list(stage_spec.get("active_parameters", [])),
                        "fixed_parameters": list(stage_spec.get("fixed_parameters", [])),
                    }
                )
                any_failed = True
                _emit_status(f"Geometry fit: staged stage {stage_number} failed (empty residual)")
                continue

            start_cost = _robust_cost(
                stage_start_residual,
                loss=solver_loss,
                f_scale=solver_f_scale,
            )

            def _stage_residual(x_active: np.ndarray) -> np.ndarray:
                trial_x = np.asarray(stage_start, dtype=float).copy()
                trial_x[active_indices] = np.asarray(x_active, dtype=float)
                return np.asarray(residual_fn(trial_x), dtype=float)

            try:
                stage_result = least_squares(
                    _stage_residual,
                    np.asarray(stage_start[active_indices], dtype=float),
                    bounds=(
                        np.asarray(lower_bounds[active_indices], dtype=float),
                        np.asarray(upper_bounds[active_indices], dtype=float),
                    ),
                    x_scale=np.asarray(x_scale[active_indices], dtype=float),
                    loss=solver_loss,
                    f_scale=solver_f_scale,
                    max_nfev=staged_release_max_nfev,
                )
            except Exception as exc:
                summary["stages"].append(
                    {
                        "stage": int(stage_number),
                        "label": str(stage_spec.get("label", "auto")),
                        "status": "failed",
                        "reason": f"stage_failed: {exc}",
                        "accepted": False,
                        "active_parameters": list(stage_spec.get("active_parameters", [])),
                        "fixed_parameters": list(stage_spec.get("fixed_parameters", [])),
                    }
                )
                any_failed = True
                _emit_status(f"Geometry fit: staged stage {stage_number} failed ({exc})")
                continue

            trial_x = np.asarray(stage_start, dtype=float).copy()
            trial_x[active_indices] = np.asarray(stage_result.x, dtype=float)
            trial_x = np.minimum(np.maximum(trial_x, lower_bounds), upper_bounds)
            final_residual = np.asarray(residual_fn(trial_x), dtype=float)
            final_cost = _robust_cost(
                final_residual,
                loss=solver_loss,
                f_scale=solver_f_scale,
            )
            cost_tol = max(
                1.0e-9 * max(abs(float(start_cost)), 1.0),
                1.0e-12,
            )
            if np.isfinite(start_cost) and start_cost > 1.0e-12:
                cost_limit = float(start_cost * (1.0 + staged_release_max_cost_increase_fraction))
            else:
                cost_limit = float(start_cost + cost_tol)
            accepted = bool(np.isfinite(final_cost) and final_cost <= cost_limit + cost_tol)
            stage_status = "accepted" if accepted else "rejected"
            stage_reason = "accepted"
            if accepted and final_cost + cost_tol < start_cost:
                stage_reason = "accepted_improved"
            elif accepted:
                stage_reason = "accepted_no_regression"
            else:
                stage_reason = "cost_regressed"

            summary["stages"].append(
                {
                    "stage": int(stage_number),
                    "label": str(stage_spec.get("label", "auto")),
                    "status": stage_status,
                    "reason": stage_reason,
                    "accepted": bool(accepted),
                    "active_parameters": list(stage_spec.get("active_parameters", [])),
                    "fixed_parameters": list(stage_spec.get("fixed_parameters", [])),
                    "active_count": int(active_indices.size),
                    "fixed_count": int(fixed_indices.size),
                    "start_cost": float(start_cost),
                    "final_cost": float(final_cost),
                    "cost_limit": float(cost_limit),
                    "nfev": int(getattr(stage_result, "nfev", 0)),
                    "success": bool(getattr(stage_result, "success", False)),
                    "message": str(getattr(stage_result, "message", "")),
                    "start_x": stage_start.tolist(),
                    "end_x": trial_x.tolist(),
                }
            )
            if accepted:
                current_x = np.asarray(trial_x, dtype=float)
                accepted_stage_count += 1
                _emit_status(
                    f"Geometry fit: staged stage {stage_number} accepted "
                    f"(cost={float(final_cost):.6f})"
                )
            else:
                _emit_status(
                    f"Geometry fit: staged stage {stage_number} rejected "
                    f"(cost={float(final_cost):.6f})"
                )

        summary["accepted_stage_count"] = int(accepted_stage_count)
        if accepted_stage_count > 0:
            summary["status"] = "accepted"
            summary["reason"] = "accepted"
            summary["accepted"] = True
            summary["x"] = np.asarray(current_x, dtype=float)
            _emit_status(
                "Geometry fit: staged release accepted "
                f"({accepted_stage_count}/{len(stage_specs)} stage(s))"
            )
        elif any_failed:
            summary["status"] = "failed"
            summary["reason"] = "no_stage_accepted"
            _emit_status("Geometry fit: staged release failed")
        else:
            summary["status"] = "rejected"
            summary["reason"] = "no_stage_accepted"
            _emit_status("Geometry fit: staged release rejected")
        return summary

    def _maybe_run_adaptive_regularization_seed(
        x_start: np.ndarray,
    ) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "enabled": bool(adaptive_regularization_enabled),
            "status": "skipped",
            "reason": "",
            "accepted": False,
            "applied_parameters": [],
            "prior_entries": [],
            "release_attempted": False,
            "release_accepted": False,
            "candidates": [],
        }
        if not adaptive_regularization_enabled:
            summary["reason"] = "disabled_by_config"
            return summary
        if not identifiability_enabled:
            summary["reason"] = "identifiability_disabled"
            return summary
        if adaptive_regularization_max_parameters <= 0:
            summary["reason"] = "max_regularized_parameters_zero"
            return summary

        start_x = np.asarray(x_start, dtype=float)
        if start_x.ndim != 1 or start_x.size == 0:
            summary["reason"] = "empty_parameter_vector"
            return summary
        if len(var_names) <= 0:
            summary["reason"] = "insufficient_parameter_count"
            return summary

        start_residual = np.asarray(residual_fn(start_x), dtype=float)
        if start_residual.ndim != 1 or start_residual.size == 0:
            summary["reason"] = "empty_residual"
            return summary
        start_cost = _robust_cost(
            start_residual,
            loss=solver_loss,
            f_scale=solver_f_scale,
        )
        cost_tol = max(
            1.0e-9 * max(abs(float(start_cost)), 1.0),
            1.0e-12,
        )
        if np.isfinite(start_cost) and start_cost > 1.0e-12:
            cost_limit = float(
                start_cost * (1.0 + adaptive_regularization_max_cost_increase_fraction)
            )
        else:
            cost_limit = float(start_cost + cost_tol)
        summary["start_cost"] = float(start_cost)
        summary["cost_limit"] = float(cost_limit)

        start_ident_summary = _build_identifiability_summary_at_x(start_x, None)
        summary["condition_number_before"] = float(
            start_ident_summary.get("condition_number", np.nan)
        )
        summary["underconstrained_before"] = bool(
            start_ident_summary.get("underconstrained", False)
        )
        summary["warning_flags_before"] = list(start_ident_summary.get("warning_flags", []))
        summary["recommended_fixed_parameters_before"] = list(
            start_ident_summary.get("recommended_fixed_parameters", [])
        )
        if str(start_ident_summary.get("status", "")) != "ok":
            summary["reason"] = f"identifiability_{start_ident_summary.get('reason', 'failed')}"
            return summary

        candidate_entries: List[Dict[str, object]] = []
        has_high_correlation_candidate = False
        for entry in start_ident_summary.get("freeze_recommendations", []):
            if not isinstance(entry, dict):
                continue
            reasons = {str(reason) for reason in entry.get("reasons", [])}
            max_abs_correlation = float(entry.get("max_abs_correlation", np.nan))
            is_weak = bool(reasons.intersection({"weak_sensitivity", "invalid_jacobian_column"}))
            is_high_corr = bool(
                np.isfinite(max_abs_correlation)
                and max_abs_correlation >= adaptive_regularization_correlation
            )
            if is_high_corr:
                has_high_correlation_candidate = True
            if not (is_weak or is_high_corr):
                continue
            candidate_entries.append(dict(entry))

        if not candidate_entries:
            summary["reason"] = "no_regularization_candidates"
            return summary

        condition_triggered = bool(
            summary["underconstrained_before"]
            or not np.isfinite(summary["condition_number_before"])
            or float(summary["condition_number_before"]) >= adaptive_regularization_condition_number
            or has_high_correlation_candidate
        )
        if not condition_triggered:
            summary["reason"] = "condition_threshold_not_triggered"
            return summary

        selected_entries = candidate_entries[:adaptive_regularization_max_parameters]
        if not selected_entries:
            summary["reason"] = "max_regularized_parameters_zero"
            return summary

        prior_centers = np.full(start_x.shape, np.nan, dtype=float)
        prior_sigmas = np.full(start_x.shape, np.nan, dtype=float)
        applied_parameters: List[str] = []
        prior_entries: List[Dict[str, object]] = []
        for entry in selected_entries:
            idx = int(entry.get("index", -1))
            if idx < 0 or idx >= len(var_names):
                continue
            sigma = max(
                adaptive_regularization_min_sigma,
                adaptive_regularization_sigma_scale
                * _identifiability_scale_for_index(idx, start_x),
            )
            prior_centers[idx] = float(start_x[idx])
            prior_sigmas[idx] = float(sigma)
            applied_parameters.append(str(var_names[idx]))
            prior_entries.append(
                {
                    "name": str(var_names[idx]),
                    "index": int(idx),
                    "center": float(start_x[idx]),
                    "sigma": float(sigma),
                    "reasons": [str(reason) for reason in entry.get("reasons", [])],
                    "partners": [str(name) for name in entry.get("partners", [])],
                    "max_abs_correlation": float(entry.get("max_abs_correlation", np.nan)),
                }
            )

        adaptive_mask = (
            np.isfinite(prior_centers) & np.isfinite(prior_sigmas) & (prior_sigmas > 0.0)
        )
        if not np.any(adaptive_mask):
            summary["reason"] = "no_valid_regularization_entries"
            return summary
        summary["applied_parameters"] = applied_parameters
        summary["prior_entries"] = prior_entries
        _emit_status(f"Geometry fit: adaptive regularization evaluating {applied_parameters}")

        def _adaptive_prior_residuals(x_trial: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x_trial, dtype=float)
            return (x_arr[adaptive_mask] - prior_centers[adaptive_mask]) / prior_sigmas[
                adaptive_mask
            ]

        def _regularized_residual(x_trial: np.ndarray) -> np.ndarray:
            base_residual = np.asarray(residual_fn(np.asarray(x_trial, dtype=float)), dtype=float)
            adaptive_residual = _adaptive_prior_residuals(np.asarray(x_trial, dtype=float))
            if adaptive_residual.size:
                if base_residual.size:
                    return np.concatenate((base_residual, adaptive_residual))
                return np.asarray(adaptive_residual, dtype=float)
            return base_residual

        def _candidate_record(
            label: str,
            trial_x: np.ndarray,
            trial_cost: float,
            trial_ident_summary: Dict[str, object],
            trial_result: OptimizeResult,
        ) -> Dict[str, object]:
            ident_ok = str(trial_ident_summary.get("status", "")) == "ok"
            progress = bool(
                ident_ok
                and _identifiability_progressed(
                    start_ident_summary,
                    trial_ident_summary,
                    start_cost=float(start_cost),
                    trial_cost=float(trial_cost),
                    cost_tol=float(cost_tol),
                )
            )
            cost_ok = bool(
                np.isfinite(trial_cost) and float(trial_cost) <= float(cost_limit) + float(cost_tol)
            )
            return {
                "label": str(label),
                "x": np.asarray(trial_x, dtype=float),
                "cost": float(trial_cost),
                "condition_number": float(trial_ident_summary.get("condition_number", np.nan)),
                "underconstrained": bool(trial_ident_summary.get("underconstrained", False)),
                "recommended_fixed_parameters": list(
                    trial_ident_summary.get("recommended_fixed_parameters", [])
                ),
                "identifiability_status": str(trial_ident_summary.get("status", "unknown")),
                "progress": bool(progress),
                "cost_ok": bool(cost_ok),
                "accepted": bool(progress and cost_ok),
                "nfev": int(getattr(trial_result, "nfev", 0)),
                "success": bool(getattr(trial_result, "success", False)),
                "message": str(getattr(trial_result, "message", "")),
            }

        try:
            regularized_result = _run_solver_with_max_nfev(
                start_x,
                max_nfev=adaptive_regularization_max_nfev,
                residual_callable=_regularized_residual,
                status_label="adaptive regularization",
            )
        except Exception as exc:
            summary["status"] = "failed"
            summary["reason"] = f"adaptive_regularization_failed: {exc}"
            _emit_status(f"Geometry fit: adaptive regularization failed ({exc})")
            return summary

        regularized_x = np.minimum(
            np.maximum(np.asarray(regularized_result.x, dtype=float), lower_bounds),
            upper_bounds,
        )
        _, regularized_cost = _evaluate_cost_at(regularized_x)
        regularized_ident_summary = _build_identifiability_summary_at_x(
            regularized_x,
            regularized_result,
        )
        regularized_candidate = _candidate_record(
            "regularized",
            regularized_x,
            regularized_cost,
            regularized_ident_summary,
            regularized_result,
        )
        summary["regularized_cost"] = float(regularized_cost)
        summary["regularized_condition_number"] = float(regularized_candidate["condition_number"])
        summary["candidates"].append(
            {key: value for key, value in regularized_candidate.items() if key != "x"}
        )

        accepted_candidates: List[Dict[str, object]] = []
        if bool(regularized_candidate.get("accepted", False)):
            accepted_candidates.append(regularized_candidate)

        summary["release_attempted"] = True
        _emit_status("Geometry fit: adaptive regularization releasing seed")
        try:
            release_result = _run_solver_with_max_nfev(
                regularized_x,
                max_nfev=adaptive_regularization_release_max_nfev,
                status_label="adaptive regularization release",
            )
        except Exception as exc:
            summary["release_reason"] = f"release_failed: {exc}"
        else:
            release_x = np.minimum(
                np.maximum(np.asarray(release_result.x, dtype=float), lower_bounds),
                upper_bounds,
            )
            _, release_cost = _evaluate_cost_at(release_x)
            release_ident_summary = _build_identifiability_summary_at_x(
                release_x,
                release_result,
            )
            release_candidate = _candidate_record(
                "released",
                release_x,
                release_cost,
                release_ident_summary,
                release_result,
            )
            summary["released_cost"] = float(release_cost)
            summary["released_condition_number"] = float(release_candidate["condition_number"])
            summary["candidates"].append(
                {key: value for key, value in release_candidate.items() if key != "x"}
            )
            if bool(release_candidate.get("accepted", False)):
                accepted_candidates.append(release_candidate)

        if not accepted_candidates:
            summary["status"] = "rejected"
            summary["reason"] = "no_acceptable_regularized_seed"
            _emit_status("Geometry fit: adaptive regularization rejected")
            return summary

        accepted_candidates.sort(
            key=lambda item: (
                float(item.get("cost", np.inf)),
                float(item.get("condition_number", np.inf))
                if np.isfinite(float(item.get("condition_number", np.nan)))
                else float("inf"),
                0 if str(item.get("label", "")) == "released" else 1,
            )
        )
        best_candidate = accepted_candidates[0]
        summary["accepted"] = True
        summary["status"] = "accepted"
        summary["reason"] = f"{best_candidate['label']}_seed_accepted"
        summary["release_accepted"] = bool(str(best_candidate.get("label", "")) == "released")
        summary["final_cost"] = float(best_candidate["cost"])
        summary["final_condition_number"] = float(best_candidate["condition_number"])
        summary["underconstrained_after"] = bool(best_candidate.get("underconstrained", False))
        summary["recommended_fixed_parameters_after"] = list(
            best_candidate.get("recommended_fixed_parameters", [])
        )
        summary["x"] = np.asarray(best_candidate["x"], dtype=float)
        if summary["release_accepted"]:
            summary["released_x"] = np.asarray(best_candidate["x"], dtype=float)
        else:
            summary["regularized_x"] = np.asarray(best_candidate["x"], dtype=float)
        _emit_status(
            "Geometry fit: adaptive regularization accepted "
            f"(cost={float(best_candidate['cost']):.6f})"
        )
        return summary

    def _vector_key(x_trial: Sequence[float]) -> Tuple[float, ...]:
        return tuple(np.round(np.asarray(x_trial, dtype=float), 12).tolist())

    def _build_restart_candidates() -> List[Tuple[np.ndarray, str, str]]:
        if solver_restarts <= 0 or x0_arr.size == 0:
            return []

        candidate_budget = min(max(solver_restarts * 3, solver_restarts + 4), 16)
        local_scale = np.where(finite_span, span, fallback_scale)
        local_scale = np.maximum(local_scale, 1e-6)
        clipped_initial = np.minimum(np.maximum(x0_arr, lower_bounds), upper_bounds)
        finite_indices = np.flatnonzero(finite_span)
        candidates: List[Tuple[np.ndarray, str, str]] = []
        seen = {_vector_key(x0_arr)}

        def _append_candidate(
            x_trial: Sequence[float],
            *,
            seed_kind: str,
            seed_label: str,
        ) -> None:
            trial_x = np.asarray(x_trial, dtype=float)
            if trial_x.shape != x0_arr.shape:
                return
            trial_x = np.minimum(np.maximum(trial_x, lower_bounds), upper_bounds)
            if not np.all(np.isfinite(trial_x)):
                return
            key = _vector_key(trial_x)
            if key in seen:
                return
            seen.add(key)
            candidates.append((trial_x, str(seed_kind), str(seed_label)))

        if finite_indices.size > 0:
            midpoint = np.asarray(clipped_initial, dtype=float)
            midpoint[finite_indices] = lower_bounds[finite_indices] + 0.5 * span[finite_indices]
            relative = np.clip(
                (clipped_initial[finite_indices] - lower_bounds[finite_indices])
                / np.maximum(span[finite_indices], 1e-12),
                0.0,
                1.0,
            )
            opposite_corner = np.asarray(clipped_initial, dtype=float)
            opposite_corner[finite_indices] = np.where(
                relative <= 0.5,
                upper_bounds[finite_indices],
                lower_bounds[finite_indices],
            )
            _append_candidate(
                opposite_corner,
                seed_kind="restart corner seed",
                seed_label="opposite-corner",
            )
            _append_candidate(
                midpoint,
                seed_kind="restart center seed",
                seed_label="box-center",
            )

            ranked_finite = finite_indices[np.argsort(local_scale[finite_indices])[::-1]]
            axis_limit = min(max(2, solver_restarts), ranked_finite.size)
            for idx in ranked_finite[:axis_limit]:
                lower_seed = np.asarray(clipped_initial, dtype=float)
                lower_seed[idx] = lower_bounds[idx]
                _append_candidate(
                    lower_seed,
                    seed_kind="restart axis seed",
                    seed_label=f"{var_names[int(idx)]}-lower",
                )

                upper_seed = np.asarray(clipped_initial, dtype=float)
                upper_seed[idx] = upper_bounds[idx]
                _append_candidate(
                    upper_seed,
                    seed_kind="restart axis seed",
                    seed_label=f"{var_names[int(idx)]}-upper",
                )

        irrational_steps = np.modf(np.sqrt(np.arange(2, x0_arr.size + 2, dtype=float)))[0]
        irrational_steps = np.where(
            np.abs(irrational_steps) > 1e-12,
            irrational_steps,
            0.5,
        )
        for sample_idx in range(candidate_budget):
            unit = np.mod(0.5 + (sample_idx + 1) * irrational_steps, 1.0)
            sample = np.asarray(clipped_initial, dtype=float)
            if finite_indices.size > 0:
                sample[finite_indices] = (
                    lower_bounds[finite_indices] + unit[finite_indices] * span[finite_indices]
                )
            infinite_indices = np.flatnonzero(~finite_span)
            if infinite_indices.size > 0:
                sample[infinite_indices] = (
                    x0_arr[infinite_indices]
                    + (2.0 * unit[infinite_indices] - 1.0)
                    * solver_restart_jitter
                    * local_scale[infinite_indices]
                )
            _append_candidate(
                sample,
                seed_kind="restart global sample",
                seed_label=f"qmc#{sample_idx + 1}",
            )
            if len(candidates) >= candidate_budget:
                break

        restart_rng = np.random.default_rng(20260214)
        attempts = 0
        while len(candidates) < candidate_budget and attempts < candidate_budget * 8:
            perturb = restart_rng.uniform(-1.0, 1.0, size=x0_arr.shape)
            sample = x0_arr + solver_restart_jitter * local_scale * perturb
            _append_candidate(
                sample,
                seed_kind="restart local jitter",
                seed_label=f"local#{attempts + 1}",
            )
            attempts += 1

        return candidates

    def _evaluate_restart_seed(
        item: Tuple[int, np.ndarray, str, str],
    ) -> Tuple[float, int, np.ndarray, np.ndarray, str, str]:
        candidate_idx, trial_start, seed_kind, seed_label = item
        seed_residual, seed_cost = _evaluate_cost_at(np.asarray(trial_start, dtype=float))
        return (
            float(seed_cost),
            int(candidate_idx),
            np.asarray(trial_start, dtype=float),
            np.asarray(seed_residual, dtype=float),
            str(seed_kind),
            str(seed_label),
        )

    def _solve_restart_seed(
        item: Tuple[float, int, np.ndarray, np.ndarray, str, str],
    ) -> Tuple[np.ndarray, OptimizeResult, float, float, str, str]:
        (
            _seed_cost_sort,
            _candidate_idx,
            trial_start,
            seed_residual,
            seed_kind,
            seed_label,
        ) = item
        trial_start = np.asarray(trial_start, dtype=float)
        seed_residual = np.asarray(seed_residual, dtype=float)
        seed_cost = _robust_cost(
            seed_residual,
            loss=solver_loss,
            f_scale=solver_f_scale,
        )
        trial = _run_solver(trial_start)
        trial_cost = _robust_cost(
            np.asarray(trial.fun, dtype=float),
            loss=solver_loss,
            f_scale=solver_f_scale,
        )

        if seed_cost + restart_selection_tol < trial_cost:
            trial = _build_probe_result(
                trial_start,
                seed_residual,
                message=(
                    f"{seed_kind} {seed_label} improved the fit; "
                    "local least-squares did not improve further."
                ),
            )
            trial_cost = float(seed_cost)

        result_message = str(getattr(trial, "message", "") or "").strip()
        if result_message:
            result_message = f"{seed_kind} {seed_label}: {result_message}"
        else:
            result_message = f"{seed_kind} {seed_label}"
        trial.message = result_message

        return (
            trial_start,
            trial,
            float(trial_cost),
            float(seed_cost),
            str(seed_kind),
            str(seed_label),
        )

    def _inactive_stage_summary(
        *,
        enabled: bool,
        stage_name: str,
    ) -> Dict[str, object]:
        return {
            "enabled": bool(enabled),
            "status": "skipped",
            "reason": f"{stage_name}_retired_by_normalized_u_solver",
            "accepted": False,
        }

    reparameterization_summary = _inactive_stage_summary(
        enabled=bool(reparameterize_enabled),
        stage_name="reparameterization",
    )
    staged_release_summary = _inactive_stage_summary(
        enabled=bool(staged_release_enabled),
        stage_name="staged_release",
    )
    adaptive_regularization_summary = _inactive_stage_summary(
        enabled=bool(adaptive_regularization_enabled),
        stage_name="adaptive_regularization",
    )

    active_specs = _build_specs_from_params([str(name) for name in var_names])
    for idx, spec in enumerate(active_specs):
        if idx >= len(parameter_debug_entries):
            continue
        parameter_debug_entries[idx]["scale"] = float(spec.scale)
        parameter_debug_entries[idx]["prior_enabled"] = bool(
            spec.prior_sigma is not None
            and np.isfinite(float(spec.prior_sigma))
            and float(spec.prior_sigma) > 0.0
        )
        parameter_debug_entries[idx]["prior_center"] = (
            float(spec.prior_mean)
            if spec.prior_mean is not None and np.isfinite(float(spec.prior_mean))
            else float("nan")
        )
        parameter_debug_entries[idx]["prior_sigma"] = (
            float(spec.prior_sigma)
            if spec.prior_sigma is not None and np.isfinite(float(spec.prior_sigma))
            else float("nan")
        )
        parameter_debug_entries[idx]["seed_group"] = str(spec.seed_group)

    u0_arr = np.zeros(len(active_specs), dtype=float)
    u_lower_bounds, u_upper_bounds = _build_u_bounds(active_specs)
    base_dataset_contexts = list(dataset_contexts)
    discrete_modes = _iter_discrete_mode_specs(discrete_modes_cfg)
    geometry_fit_debug_summary["seed_search"] = {
        "prescore_top_k": int(seed_prescore_top_k),
        "n_global": int(seed_n_global),
        "n_jitter": int(seed_n_jitter),
        "jitter_sigma_u": float(seed_jitter_sigma_u),
        "min_seed_separation_u": float(seed_min_separation_u),
        "trusted_prior_fraction_of_span": float(trusted_prior_fraction_of_span),
    }
    geometry_fit_debug_summary["discrete_modes"] = [dict(mode) for mode in discrete_modes]

    def _seed_entry_distance(
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        return float(
            np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float))
        )

    def _build_mode_seeds(
        specs: Sequence[ParamSpec],
        lb_u: np.ndarray,
        ub_u: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> List[Dict[str, object]]:
        seed_entries: List[Dict[str, object]] = []
        seen_exact: Set[Tuple[float, ...]] = set()

        def _append_seed(u_seed: Sequence[float], seed_kind: str, seed_label: str) -> None:
            seed = np.asarray(u_seed, dtype=float).reshape(-1)
            if seed.shape != lb_u.shape:
                return
            seed = np.minimum(np.maximum(seed, lb_u), ub_u)
            if not np.all(np.isfinite(seed)):
                return
            key = tuple(np.round(seed, 12).tolist())
            if key in seen_exact:
                return
            if any(
                _seed_entry_distance(seed, np.asarray(entry["u"], dtype=float))
                < float(seed_min_separation_u)
                for entry in seed_entries
            ):
                return
            seen_exact.add(key)
            seed_entries.append(
                {
                    "u": np.asarray(seed, dtype=float),
                    "seed_kind": str(seed_kind),
                    "seed_label": str(seed_label),
                }
            )

        _append_seed(np.zeros(len(specs), dtype=float), "zero", "u=0")
        uncertain_indices = [
            idx for idx, spec in enumerate(specs) if str(spec.seed_group) == "uncertain"
        ]
        for idx in uncertain_indices:
            for amplitude, suffix in ((-1.0, "-1"), (1.0, "+1")):
                seed = np.zeros(len(specs), dtype=float)
                seed[idx] = float(np.clip(amplitude, lb_u[idx], ub_u[idx]))
                _append_seed(seed, "axis", f"{specs[idx].name}{suffix}")
        for jitter_idx in range(seed_n_jitter):
            seed = np.zeros(len(specs), dtype=float)
            if uncertain_indices:
                seed[uncertain_indices] = rng.normal(
                    0.0,
                    float(seed_jitter_sigma_u),
                    size=len(uncertain_indices),
                )
            _append_seed(seed, "jitter", f"jitter#{jitter_idx + 1}")
        finite_global_indices = [
            idx
            for idx in uncertain_indices
            if np.isfinite(lb_u[idx]) and np.isfinite(ub_u[idx]) and ub_u[idx] > lb_u[idx]
        ]
        if finite_global_indices:
            box_points = _low_discrepancy_points(seed_n_global, len(finite_global_indices))
            mapped = _map_box_points(
                box_points,
                lb_u[finite_global_indices],
                ub_u[finite_global_indices],
            )
            for global_idx, values in enumerate(mapped, start=1):
                seed = np.zeros(len(specs), dtype=float)
                seed[finite_global_indices] = np.asarray(values, dtype=float)
                _append_seed(seed, "global", f"global#{global_idx}")
        return seed_entries

    rng = np.random.default_rng(20260401)
    restart_history: List[Dict[str, object]] = []
    best_result: OptimizeResult | None = None
    best_cost = float("inf")
    best_mode: Dict[str, object] = (
        dict(discrete_modes[0]) if discrete_modes else {"label": "identity"}
    )
    best_mode_contexts = list(base_dataset_contexts)
    solve_record_count = 0
    prescore_record_count = 0

    if not active_specs:
        mode_results: List[
            Tuple[Dict[str, object], OptimizeResult, float, List[GeometryFitDatasetContext]]
        ] = []
        for mode in discrete_modes:
            dataset_contexts = _apply_discrete_mode_to_dataset_contexts(
                base_dataset_contexts,
                image_size=int(image_size),
                mode=mode,
            )
            residual = _data_residual_theta([], names=[], base_params=params)
            trial = _build_probe_result(
                np.array([], dtype=float),
                residual,
                message=f"fixed-parameter mode {_format_discrete_mode_label(mode)}",
            )
            trial.x_u = np.array([], dtype=float)
            trial_cost = _robust_cost(
                np.asarray(trial.fun, dtype=float),
                loss=solver_loss,
                f_scale=solver_f_scale,
            )
            mode_results.append((dict(mode), trial, float(trial_cost), list(dataset_contexts)))
        best_mode, best_result, best_cost, best_mode_contexts = min(
            mode_results,
            key=lambda item: float(item[2]),
        )
    else:
        _emit_status(
            "Geometry fit: running normalized-u multistart solve "
            f"(modes={len(discrete_modes)}, vars={len(active_specs)})"
        )
        for mode_index, mode in enumerate(discrete_modes):
            mode_label = _format_discrete_mode_label(mode)
            dataset_contexts = _apply_discrete_mode_to_dataset_contexts(
                base_dataset_contexts,
                image_size=int(image_size),
                mode=mode,
            )
            mode_seeds = _build_mode_seeds(
                active_specs,
                u_lower_bounds,
                u_upper_bounds,
                rng=rng,
            )
            if zero_seed_only_solver:
                mode_seeds = [
                    dict(entry) for entry in mode_seeds if str(entry.get("seed_kind", "")) == "zero"
                ] or mode_seeds[:1]
            scored_seeds: List[Dict[str, object]] = []
            for seed_entry in mode_seeds:
                seed_u = np.asarray(seed_entry["u"], dtype=float)
                seed_residual, seed_cost = _evaluate_cost_at_u(
                    seed_u,
                    specs=active_specs,
                    include_priors=True,
                )
                scored_seeds.append(
                    {
                        "u": seed_u,
                        "seed_kind": str(seed_entry.get("seed_kind", "")),
                        "seed_label": str(seed_entry.get("seed_label", "")),
                        "residual": np.asarray(seed_residual, dtype=float),
                        "cost": float(seed_cost),
                    }
                )
            scored_seeds.sort(
                key=lambda item: (
                    float(item.get("cost", np.inf)),
                    str(item.get("seed_kind", "")),
                    str(item.get("seed_label", "")),
                )
            )
            selected_seeds: List[Dict[str, object]] = []
            for seed_entry in scored_seeds:
                if len(selected_seeds) >= seed_prescore_top_k:
                    break
                seed_u = np.asarray(seed_entry["u"], dtype=float)
                if any(
                    _seed_entry_distance(seed_u, np.asarray(existing["u"], dtype=float))
                    < float(seed_min_separation_u)
                    for existing in selected_seeds
                ):
                    continue
                selected_seeds.append(dict(seed_entry))

            geometry_fit_debug_summary.setdefault("mode_seed_summary", []).append(
                {
                    "mode": dict(mode),
                    "seed_count": int(len(mode_seeds)),
                    "selected_seed_count": int(len(selected_seeds)),
                    "selected_seeds": [
                        {
                            "seed_kind": str(entry.get("seed_kind", "")),
                            "seed_label": str(entry.get("seed_label", "")),
                            "cost": float(entry.get("cost", np.nan)),
                            "u": np.asarray(entry.get("u", []), dtype=float).tolist(),
                        }
                        for entry in selected_seeds
                    ],
                }
            )
            _emit_status(
                "Geometry fit: mode {label} prescore total_seeds={total} "
                "selected={selected}".format(
                    label=str(mode_label),
                    total=int(len(mode_seeds)),
                    selected=int(len(selected_seeds)),
                )
            )

            for seed_entry in selected_seeds:
                prescore_record_count += 1
                seed_u = np.asarray(seed_entry["u"], dtype=float)
                seed_x = _physical_from_u(active_specs, seed_u)
                restart_history.append(
                    {
                        "restart": len(restart_history),
                        "mode_index": int(mode_index),
                        "mode_label": str(mode_label),
                        "start_u": seed_u.tolist(),
                        "start_x": seed_x.tolist(),
                        "end_u": seed_u.tolist(),
                        "end_x": seed_x.tolist(),
                        "cost": float(seed_entry.get("cost", np.nan)),
                        "success": False,
                        "seed_kind": str(seed_entry.get("seed_kind", "")),
                        "seed_label": str(seed_entry.get("seed_label", "")),
                        "message": "prescore",
                    }
                )

            for solve_idx, seed_entry in enumerate(selected_seeds, start=1):
                solve_record_count += 1
                seed_u = np.asarray(seed_entry["u"], dtype=float)
                trial_u_result = _run_solver_u(
                    seed_u,
                    specs=active_specs,
                    lower=u_lower_bounds,
                    upper=u_upper_bounds,
                    include_priors=True,
                    max_nfev=solver_max_nfev,
                    status_label=f"{mode_label} seed {solve_idx}",
                )
                trial_result = _wrap_solver_result_from_u(
                    trial_u_result,
                    specs=active_specs,
                )
                trial_cost = _robust_cost(
                    np.asarray(trial_result.fun, dtype=float),
                    loss=solver_loss,
                    f_scale=solver_f_scale,
                )
                seed_cost = float(seed_entry.get("cost", np.inf))
                selection_tol = max(1.0e-9 * max(abs(seed_cost), 1.0), 1.0e-12)
                if seed_cost + selection_tol < trial_cost:
                    trial_result = _build_probe_result(
                        _physical_from_u(active_specs, seed_u),
                        np.asarray(seed_entry.get("residual", []), dtype=float),
                        message=(
                            f"{seed_entry.get('seed_kind', 'seed')} "
                            f"{seed_entry.get('seed_label', '')} prescore preferred"
                        ),
                    )
                    trial_result.x_u = np.asarray(seed_u, dtype=float)
                    trial_cost = seed_cost
                restart_history.append(
                    {
                        "restart": len(restart_history),
                        "mode_index": int(mode_index),
                        "mode_label": str(mode_label),
                        "start_u": seed_u.tolist(),
                        "start_x": _physical_from_u(active_specs, seed_u).tolist(),
                        "end_u": np.asarray(getattr(trial_result, "x_u", []), dtype=float).tolist(),
                        "end_x": np.asarray(trial_result.x, dtype=float).tolist(),
                        "cost": float(trial_cost),
                        "seed_cost": float(seed_cost),
                        "success": bool(getattr(trial_result, "success", False)),
                        "seed_kind": str(seed_entry.get("seed_kind", "")),
                        "seed_label": str(seed_entry.get("seed_label", "")),
                        "message": str(getattr(trial_result, "message", "")),
                    }
                )
                if best_result is None or float(trial_cost) < float(best_cost):
                    best_result = trial_result
                    best_cost = float(trial_cost)
                    best_mode = dict(mode)
                    best_mode_contexts = list(dataset_contexts)

    if best_result is None:
        best_result = _build_probe_result(
            np.asarray(x0_arr, dtype=float),
            np.asarray(residual_fn(np.asarray(x0_arr, dtype=float)), dtype=float),
            message="normalized_u_solver_failed_to_generate_result",
        )
        best_result.x_u = np.asarray(u0_arr, dtype=float)
        best_cost = _robust_cost(
            np.asarray(best_result.fun, dtype=float),
            loss=solver_loss,
            f_scale=solver_f_scale,
        )

    dataset_contexts = list(best_mode_contexts)
    result = best_result
    geometry_fit_debug_summary["main_solve_seed"] = {
        "seed_kind": "u=0",
        "seed_label": "normalized-zero",
        "u": np.asarray(u0_arr, dtype=float).tolist(),
        "x": np.asarray(x0_arr, dtype=float).tolist(),
    }
    geometry_fit_debug_summary["chosen_discrete_mode"] = dict(best_mode)
    geometry_fit_debug_summary["selected_mode_label"] = _format_discrete_mode_label(best_mode)
    geometry_fit_debug_summary["solve_counts"] = {
        "prescored": int(prescore_record_count),
        "solved": int(solve_record_count),
    }
    _emit_status(
        "Geometry fit: multistart summary selected_mode={mode} "
        "prescored={prescored} solved={solved}".format(
            mode=str(_format_discrete_mode_label(best_mode)),
            prescored=int(prescore_record_count),
            solved=int(solve_record_count),
        )
    )

    def _run_reduced_solver(
        reference_x: np.ndarray,
        active_indices: Sequence[int],
        *,
        max_nfev: int,
    ) -> Dict[str, object]:
        reference_arr = np.asarray(reference_x, dtype=float).copy()
        active_arr = np.asarray(active_indices, dtype=np.int64).reshape(-1)
        if reference_arr.ndim != 1 or reference_arr.size != len(var_names):
            raise ValueError("reference_x shape mismatch")
        if active_arr.size == 0:
            raise ValueError("active_indices must not be empty")
        active_arr = np.unique(active_arr)
        active_arr = active_arr[(active_arr >= 0) & (active_arr < len(var_names))]
        if active_arr.size == 0:
            raise ValueError("active_indices must include at least one valid index")

        active_set = set(active_arr.tolist())
        fixed_arr = np.asarray(
            [idx for idx in range(len(var_names)) if idx not in active_set],
            dtype=np.int64,
        )

        def _reduced_residual(x_active: np.ndarray) -> np.ndarray:
            x_trial = np.asarray(reference_arr, dtype=float).copy()
            x_trial[active_arr] = np.asarray(x_active, dtype=float)
            return np.asarray(residual_fn(x_trial), dtype=float)

        reduced_result = least_squares(
            _reduced_residual,
            np.asarray(reference_arr[active_arr], dtype=float),
            bounds=(
                np.asarray(lower_bounds[active_arr], dtype=float),
                np.asarray(upper_bounds[active_arr], dtype=float),
            ),
            x_scale=np.asarray(x_scale[active_arr], dtype=float),
            loss=solver_loss,
            f_scale=solver_f_scale,
            max_nfev=int(max_nfev),
        )

        reduced_x = np.asarray(reference_arr, dtype=float).copy()
        reduced_x[active_arr] = np.asarray(reduced_result.x, dtype=float)
        reduced_x = np.minimum(np.maximum(reduced_x, lower_bounds), upper_bounds)
        reduced_residual = np.asarray(
            residual_fn(np.asarray(reduced_x, dtype=float)),
            dtype=float,
        )
        final_cost = _robust_cost(
            reduced_residual,
            loss=solver_loss,
            f_scale=solver_f_scale,
        )
        return {
            "x": np.asarray(reduced_x, dtype=float),
            "residual": np.asarray(reduced_residual, dtype=float),
            "cost": float(final_cost),
            "result": reduced_result,
            "active_indices": np.asarray(active_arr, dtype=np.int64),
            "fixed_indices": np.asarray(fixed_arr, dtype=np.int64),
        }

    def _build_identifiability_summary_for_x(
        x_trial: np.ndarray,
        current_result: Optional[OptimizeResult] = None,
    ) -> Dict[str, object]:
        return _build_identifiability_summary_at_x(x_trial, current_result)

    def _append_result_message(current_result: OptimizeResult, note: str) -> None:
        message_text = str(getattr(current_result, "message", "") or "").strip()
        if not message_text:
            current_result.message = note
            return
        existing_parts = [part.strip() for part in message_text.split(";") if part.strip()]
        if note not in existing_parts:
            current_result.message = f"{message_text}; {note}"

    def _maybe_run_auto_freeze(
        current_result: OptimizeResult,
    ) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "enabled": bool(auto_freeze_enabled),
            "status": "skipped",
            "reason": "",
            "accepted": False,
            "fixed_parameters": [],
            "active_parameters": [],
        }
        if not auto_freeze_enabled:
            summary["reason"] = "disabled_by_config"
            return summary
        if not identifiability_enabled:
            summary["reason"] = "identifiability_disabled"
            return summary
        if getattr(current_result, "x", None) is None:
            summary["reason"] = "missing_parameter_vector"
            return summary
        if len(var_names) <= 1:
            summary["reason"] = "insufficient_parameter_count"
            return summary

        start_x = np.asarray(current_result.x, dtype=float)
        if start_x.ndim != 1 or start_x.size != len(var_names):
            summary["reason"] = "parameter_vector_shape_mismatch"
            return summary
        start_residual = np.asarray(residual_fn(start_x), dtype=float)
        if start_residual.ndim != 1 or start_residual.size == 0:
            summary["reason"] = "empty_residual"
            return summary
        start_cost = _robust_cost(
            start_residual,
            loss=solver_loss,
            f_scale=solver_f_scale,
        )
        ident_summary = _build_identifiability_summary_for_x(start_x, current_result)
        summary["condition_number"] = float(ident_summary.get("condition_number", np.nan))
        summary["underconstrained"] = bool(ident_summary.get("underconstrained", False))
        summary["warning_flags"] = list(ident_summary.get("warning_flags", []))
        summary["freeze_recommendations"] = list(ident_summary.get("freeze_recommendations", []))
        if str(ident_summary.get("status", "")) != "ok":
            summary["reason"] = f"identifiability_{ident_summary.get('reason', 'failed')}"
            return summary

        candidate_entries: List[Dict[str, object]] = []
        for entry in ident_summary.get("freeze_recommendations", []):
            if not isinstance(entry, dict):
                continue
            reasons = {str(reason) for reason in entry.get("reasons", [])}
            max_abs_correlation = float(entry.get("max_abs_correlation", np.nan))
            is_weak = bool(reasons.intersection({"weak_sensitivity", "invalid_jacobian_column"}))
            is_high_corr = (
                np.isfinite(max_abs_correlation) and max_abs_correlation >= auto_freeze_correlation
            )
            if not (is_weak or is_high_corr):
                continue
            candidate_entries.append(dict(entry))

        condition_triggered = bool(
            summary["underconstrained"]
            or not np.isfinite(summary["condition_number"])
            or float(summary["condition_number"]) >= auto_freeze_condition_number
        )
        if not candidate_entries:
            summary["reason"] = "no_freeze_candidates"
            return summary
        if not condition_triggered:
            has_weak_candidate = any(
                {str(reason) for reason in entry.get("reasons", [])}.intersection(
                    {"weak_sensitivity", "invalid_jacobian_column"}
                )
                for entry in candidate_entries
            )
            if not has_weak_candidate:
                summary["reason"] = "condition_threshold_not_triggered"
                return summary

        selected_entries = candidate_entries[:auto_freeze_max_parameters]
        if not selected_entries or auto_freeze_max_parameters <= 0:
            summary["reason"] = "max_freeze_parameters_zero"
            return summary

        fixed_indices = np.array(
            [int(entry.get("index", -1)) for entry in selected_entries],
            dtype=np.int64,
        )
        fixed_indices = fixed_indices[(fixed_indices >= 0) & (fixed_indices < len(var_names))]
        if fixed_indices.size == 0:
            summary["reason"] = "no_valid_fixed_indices"
            return summary
        fixed_indices = np.unique(fixed_indices)
        fixed_mask = np.zeros(len(var_names), dtype=bool)
        fixed_mask[fixed_indices] = True
        active_indices = np.flatnonzero(~fixed_mask)
        if active_indices.size == 0:
            summary["reason"] = "all_parameters_would_be_fixed"
            return summary

        summary["fixed_parameters"] = [str(var_names[idx]) for idx in fixed_indices.tolist()]
        summary["active_parameters"] = [str(var_names[idx]) for idx in active_indices.tolist()]
        summary["fixed_indices"] = fixed_indices.tolist()
        summary["active_indices"] = active_indices.tolist()
        summary["candidate_count"] = int(len(candidate_entries))
        summary["selected_count"] = int(fixed_indices.size)
        summary["truncated"] = bool(len(candidate_entries) > fixed_indices.size)
        summary["start_cost"] = float(start_cost)
        _emit_status(f"Geometry fit: auto-freeze evaluating {summary['fixed_parameters']}")

        try:
            reduced_summary = _run_reduced_solver(
                start_x,
                active_indices,
                max_nfev=auto_freeze_max_nfev,
            )
        except Exception as exc:
            summary["status"] = "failed"
            summary["reason"] = f"auto_freeze_failed: {exc}"
            return summary

        reduced_result = reduced_summary["result"]
        reduced_x = np.asarray(reduced_summary["x"], dtype=float)
        final_cost = float(reduced_summary["cost"])
        cost_tol = max(
            1.0e-9 * max(abs(float(start_cost)), 1.0),
            1.0e-12,
        )
        if np.isfinite(start_cost) and start_cost > 1.0e-12:
            cost_limit = float(start_cost * (1.0 + auto_freeze_max_cost_increase_fraction))
        else:
            cost_limit = float(start_cost + cost_tol)
        accepted = bool(final_cost <= cost_limit + cost_tol)
        summary.update(
            {
                "status": "accepted" if accepted else "rejected",
                "accepted": bool(accepted),
                "reason": "accepted" if accepted else "cost_regressed",
                "final_cost": float(final_cost),
                "cost_limit": float(cost_limit),
                "nfev": int(getattr(reduced_result, "nfev", 0)),
                "success": bool(getattr(reduced_result, "success", False)),
                "message": str(getattr(reduced_result, "message", "")),
                "max_nfev": int(auto_freeze_max_nfev),
            }
        )
        if accepted:
            summary["x"] = reduced_x
        return summary

    def _maybe_run_selective_thaw(
        current_result: OptimizeResult,
        freeze_summary: Dict[str, object],
    ) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "enabled": bool(selective_thaw_enabled),
            "status": "skipped",
            "reason": "",
            "accepted": False,
            "thawed_parameters": [],
            "thawed_groups": [],
            "remaining_fixed_parameters": [],
            "remaining_fixed_indices": [],
            "steps": [],
            "max_parameters": int(selective_thaw_max_parameters),
            "max_nfev": int(selective_thaw_max_nfev),
            "max_condition_number": float(selective_thaw_condition_number),
        }
        if not selective_thaw_enabled:
            summary["reason"] = "disabled_by_config"
            return summary
        if not identifiability_enabled:
            summary["reason"] = "identifiability_disabled"
            return summary
        if not isinstance(freeze_summary, dict) or not freeze_summary.get("accepted"):
            summary["reason"] = "auto_freeze_not_accepted"
            return summary
        if selective_thaw_max_parameters <= 0:
            summary["reason"] = "max_thaw_parameters_zero"
            return summary
        if getattr(current_result, "x", None) is None:
            summary["reason"] = "missing_parameter_vector"
            return summary

        current_x = np.asarray(current_result.x, dtype=float)
        if current_x.ndim != 1 or current_x.size != len(var_names):
            summary["reason"] = "parameter_vector_shape_mismatch"
            return summary

        fixed_indices_raw = freeze_summary.get("fixed_indices", [])
        fixed_indices = np.asarray(fixed_indices_raw, dtype=np.int64).reshape(-1)
        fixed_indices = fixed_indices[(fixed_indices >= 0) & (fixed_indices < len(var_names))]
        if fixed_indices.size == 0:
            fixed_names = [str(name) for name in freeze_summary.get("fixed_parameters", [])]
            fixed_lookup = {str(name): idx for idx, name in enumerate(var_names)}
            fixed_indices = np.asarray(
                [fixed_lookup[name] for name in fixed_names if name in fixed_lookup],
                dtype=np.int64,
            )
        if fixed_indices.size == 0:
            summary["reason"] = "no_fixed_parameters"
            return summary

        current_fixed = np.unique(fixed_indices)
        current_residual = np.asarray(residual_fn(current_x), dtype=float)
        if current_residual.ndim != 1 or current_residual.size == 0:
            summary["reason"] = "empty_residual"
            return summary
        current_cost = _robust_cost(
            current_residual,
            loss=solver_loss,
            f_scale=solver_f_scale,
        )
        start_cost = float(current_cost)
        summary["start_cost"] = start_cost
        thawed_parameters: List[str] = []
        thawed_groups: List[List[str]] = []
        thawed_count = 0
        accepted_any = False
        current_template = current_result

        while current_fixed.size > 0 and thawed_count < selective_thaw_max_parameters:
            remaining_capacity = int(selective_thaw_max_parameters - thawed_count)
            current_fixed_set = set(current_fixed.tolist())
            current_ident_summary = _build_identifiability_summary_for_x(
                current_x,
                current_template,
            )
            if str(current_ident_summary.get("status", "")) != "ok":
                summary["reason"] = (
                    f"identifiability_{current_ident_summary.get('reason', 'failed')}"
                )
                break

            parameter_entries = list(current_ident_summary.get("parameter_entries", []))
            recommendation_map = {
                int(entry.get("index", -1)): dict(entry)
                for entry in current_ident_summary.get("freeze_recommendations", [])
                if isinstance(entry, dict)
            }
            recommended_indices = {
                int(idx)
                for idx in current_ident_summary.get("recommended_fixed_indices", [])
                if 0 <= int(idx) < len(var_names)
            }

            candidate_groups: List[Dict[str, object]] = []
            for idx in current_fixed.tolist():
                param_entry = (
                    dict(parameter_entries[idx])
                    if 0 <= idx < len(parameter_entries)
                    and isinstance(parameter_entries[idx], dict)
                    else {}
                )
                freeze_entry = recommendation_map.get(int(idx), {})
                candidate_groups.append(
                    {
                        "indices": [int(idx)],
                        "parameters": [str(var_names[int(idx)])],
                        "kind": "single",
                        "still_recommended": bool(int(idx) in recommended_indices),
                        "max_abs_correlation": float(
                            freeze_entry.get("max_abs_correlation", np.nan)
                        ),
                        "column_norm": float(param_entry.get("column_norm", np.nan)),
                    }
                )

            if remaining_capacity >= 2 and current_fixed.size >= 2:
                seen_pairs: Set[Tuple[int, int]] = set()
                for pair_entry in current_ident_summary.get("high_correlation_pairs", []):
                    if not isinstance(pair_entry, dict):
                        continue
                    i_idx = int(pair_entry.get("index_i", -1))
                    j_idx = int(pair_entry.get("index_j", -1))
                    if i_idx == j_idx:
                        continue
                    if i_idx not in current_fixed_set or j_idx not in current_fixed_set:
                        continue
                    pair_key = tuple(sorted((i_idx, j_idx)))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    candidate_groups.append(
                        {
                            "indices": [int(pair_key[0]), int(pair_key[1])],
                            "parameters": [
                                str(var_names[int(pair_key[0])]),
                                str(var_names[int(pair_key[1])]),
                            ],
                            "kind": "pair",
                            "still_recommended": bool(
                                int(pair_key[0]) in recommended_indices
                                or int(pair_key[1]) in recommended_indices
                            ),
                            "max_abs_correlation": float(pair_entry.get("abs_correlation", np.nan)),
                            "column_norm": float(
                                sum(
                                    float(
                                        parameter_entries[idx].get(
                                            "column_norm",
                                            0.0,
                                        )
                                    )
                                    if (
                                        0 <= idx < len(parameter_entries)
                                        and isinstance(parameter_entries[idx], dict)
                                        and np.isfinite(
                                            float(
                                                parameter_entries[idx].get(
                                                    "column_norm",
                                                    np.nan,
                                                )
                                            )
                                        )
                                    )
                                    else 0.0
                                    for idx in pair_key
                                )
                            ),
                        }
                    )

            if not candidate_groups:
                summary["reason"] = "no_thaw_candidates"
                break

            candidate_groups.sort(
                key=lambda item: (
                    0 if not bool(item.get("still_recommended", False)) else 1,
                    len(item.get("indices", [])),
                    float(item.get("max_abs_correlation", np.inf))
                    if np.isfinite(float(item.get("max_abs_correlation", np.nan)))
                    else float("inf"),
                    -float(item.get("column_norm", 0.0))
                    if np.isfinite(float(item.get("column_norm", np.nan)))
                    else 0.0,
                    tuple(int(idx) for idx in item.get("indices", [])),
                )
            )
            candidate_labels = [list(item.get("parameters", [])) for item in candidate_groups]
            _emit_status(
                "Geometry fit: selective thaw step "
                f"{len(summary['steps']) + 1} evaluating {candidate_labels}"
            )

            step_summary: Dict[str, object] = {
                "step": int(len(summary["steps"]) + 1),
                "start_cost": float(current_cost),
                "fixed_parameters": [str(var_names[idx]) for idx in current_fixed.tolist()],
                "candidate_attempts": [],
            }
            best_accept: Optional[Dict[str, object]] = None
            cost_tol = max(
                1.0e-9 * max(abs(float(current_cost)), 1.0),
                1.0e-12,
            )

            for candidate in candidate_groups:
                candidate_indices = np.asarray(
                    candidate.get("indices", []),
                    dtype=np.int64,
                ).reshape(-1)
                candidate_indices = candidate_indices[
                    (candidate_indices >= 0) & (candidate_indices < len(var_names))
                ]
                if candidate_indices.size == 0:
                    continue
                if candidate_indices.size > remaining_capacity:
                    continue
                trial_fixed = np.asarray(
                    [
                        idx
                        for idx in current_fixed.tolist()
                        if idx not in set(candidate_indices.tolist())
                    ],
                    dtype=np.int64,
                )
                if trial_fixed.size == len(var_names):
                    continue
                trial_active = np.asarray(
                    [idx for idx in range(len(var_names)) if idx not in set(trial_fixed.tolist())],
                    dtype=np.int64,
                )
                attempt_entry: Dict[str, object] = {
                    "kind": str(candidate.get("kind", "single")),
                    "candidate_parameters": list(candidate.get("parameters", [])),
                    "candidate_indices": candidate_indices.tolist(),
                    "still_recommended": bool(candidate.get("still_recommended", False)),
                    "start_cost": float(current_cost),
                }
                try:
                    reduced_summary = _run_reduced_solver(
                        current_x,
                        trial_active,
                        max_nfev=selective_thaw_max_nfev,
                    )
                except Exception as exc:
                    attempt_entry.update(
                        {
                            "status": "failed",
                            "reason": f"thaw_failed: {exc}",
                        }
                    )
                    step_summary["candidate_attempts"].append(attempt_entry)
                    continue

                trial_result = reduced_summary["result"]
                trial_x = np.asarray(reduced_summary["x"], dtype=float)
                trial_cost = float(reduced_summary["cost"])
                trial_ident_summary = _build_identifiability_summary_for_x(
                    trial_x,
                    trial_result,
                )
                trial_condition = float(trial_ident_summary.get("condition_number", np.nan))
                condition_ok = bool(
                    np.isfinite(trial_condition)
                    and trial_condition <= selective_thaw_condition_number
                )
                ident_ok = str(trial_ident_summary.get("status", "")) == "ok"
                cost_improved = bool(
                    np.isfinite(trial_cost) and trial_cost + cost_tol < float(current_cost)
                )
                accepted = bool(ident_ok and condition_ok and cost_improved)
                attempt_entry.update(
                    {
                        "status": "accepted" if accepted else "rejected",
                        "reason": (
                            "accepted"
                            if accepted
                            else (
                                f"identifiability_{trial_ident_summary.get('reason', 'failed')}"
                                if not ident_ok
                                else (
                                    "condition_number_too_high"
                                    if not condition_ok
                                    else "cost_not_improved"
                                )
                            )
                        ),
                        "final_cost": float(trial_cost),
                        "cost_improved": bool(cost_improved),
                        "condition_number": float(trial_condition),
                        "nfev": int(getattr(trial_result, "nfev", 0)),
                        "success": bool(getattr(trial_result, "success", False)),
                        "message": str(getattr(trial_result, "message", "")),
                        "remaining_fixed_parameters": [
                            str(var_names[idx]) for idx in trial_fixed.tolist()
                        ],
                    }
                )
                step_summary["candidate_attempts"].append(attempt_entry)
                if not accepted:
                    continue
                if best_accept is None or trial_cost + cost_tol < float(best_accept["cost"]):
                    best_accept = {
                        "x": trial_x,
                        "cost": float(trial_cost),
                        "result": trial_result,
                        "candidate_parameters": list(candidate.get("parameters", [])),
                        "candidate_indices": candidate_indices.tolist(),
                        "trial_fixed": np.asarray(trial_fixed, dtype=np.int64),
                        "condition_number": float(trial_condition),
                    }

            if best_accept is None:
                summary["steps"].append(step_summary)
                if accepted_any:
                    summary["reason"] = "no_additional_acceptable_candidate"
                    _emit_status(
                        "Geometry fit: selective thaw stopped (no additional acceptable candidate)"
                    )
                else:
                    summary["reason"] = "no_acceptable_candidate"
                    _emit_status("Geometry fit: selective thaw rejected")
                break

            accepted_any = True
            accepted_group = [str(name) for name in best_accept["candidate_parameters"]]
            thawed_groups.append(accepted_group)
            thawed_parameters.extend(accepted_group)
            thawed_count += len(accepted_group)
            current_x = np.asarray(best_accept["x"], dtype=float)
            current_cost = float(best_accept["cost"])
            current_fixed = np.asarray(best_accept["trial_fixed"], dtype=np.int64)
            current_template = OptimizeResult(
                x=np.asarray(current_x, dtype=float),
                fun=np.asarray(residual_fn(np.asarray(current_x, dtype=float)), dtype=float),
                success=bool(getattr(best_accept["result"], "success", False)),
                status=int(getattr(best_accept["result"], "status", 0)),
                message=str(getattr(best_accept["result"], "message", "")),
                nfev=int(getattr(best_accept["result"], "nfev", 0)),
                active_mask=np.zeros_like(current_x, dtype=int),
                optimality=float(getattr(best_accept["result"], "optimality", np.nan)),
            )
            step_summary["accepted_parameters"] = accepted_group
            step_summary["reason"] = "accepted"
            step_summary["final_cost"] = float(current_cost)
            step_summary["condition_number"] = float(best_accept["condition_number"])
            step_summary["remaining_fixed_parameters"] = [
                str(var_names[idx]) for idx in current_fixed.tolist()
            ]
            summary["steps"].append(step_summary)
            _emit_status(
                "Geometry fit: selective thaw accepted "
                f"{accepted_group} "
                f"(cost={float(current_cost):.6f})"
            )

        summary["thawed_parameters"] = thawed_parameters
        summary["thawed_groups"] = thawed_groups
        summary["remaining_fixed_indices"] = current_fixed.tolist()
        summary["remaining_fixed_parameters"] = [
            str(var_names[idx]) for idx in current_fixed.tolist()
        ]
        summary["final_cost"] = float(current_cost)
        summary["accepted_count"] = int(len(thawed_parameters))
        summary["accepted"] = bool(accepted_any)
        if accepted_any:
            summary["status"] = "accepted"
            if not summary.get("reason"):
                if current_fixed.size == 0:
                    summary["reason"] = "all_candidates_thawed"
                elif thawed_count >= selective_thaw_max_parameters:
                    summary["reason"] = "max_thaw_parameters_reached"
                else:
                    summary["reason"] = "accepted"
            summary["x"] = np.asarray(current_x, dtype=float)
        elif not summary.get("reason"):
            summary["status"] = "rejected"
            summary["reason"] = "no_acceptable_candidate"
        else:
            summary["status"] = "rejected"
        return summary

    result = best_result
    if reparameterization_summary.get("accepted"):
        reparam_pairs = reparameterization_summary.get("pairs", [])
        if reparam_pairs:
            _append_result_message(
                result,
                f"Pair reparameterization seed accepted ({reparam_pairs})",
            )
    if staged_release_summary.get("accepted"):
        accepted_stage_count = int(staged_release_summary.get("accepted_stage_count", 0))
        if accepted_stage_count > 0:
            _append_result_message(
                result,
                f"Staged release seed accepted ({accepted_stage_count} stage(s))",
            )
    if adaptive_regularization_summary.get("accepted"):
        adaptive_names = adaptive_regularization_summary.get("applied_parameters", [])
        if adaptive_names:
            joined = ", ".join(str(name) for name in adaptive_names)
            _append_result_message(
                result,
                f"Adaptive regularization seed accepted ({joined})",
            )
        else:
            _append_result_message(
                result,
                "Adaptive regularization seed accepted",
            )
    full_beam_polish_summary: Dict[str, object] = {
        "enabled": bool(full_beam_polish_enabled),
        "status": "skipped",
        "reason": "not_evaluated",
        "accepted": False,
        "match_radius_px": float(full_beam_polish_match_radius_px),
    }
    if point_match_mode and getattr(result, "x", None) is not None:
        try:
            full_beam_polish_summary = _maybe_run_full_beam_polish(result)
        except Exception as exc:
            full_beam_polish_summary = {
                "enabled": bool(full_beam_polish_enabled),
                "status": "failed",
                "reason": f"unexpected_exception: {exc}",
                "accepted": False,
                "match_radius_px": float(full_beam_polish_match_radius_px),
            }
        refined_x = full_beam_polish_summary.get("x")
        if full_beam_polish_summary.get("accepted") and refined_x is not None:
            result.x = np.asarray(refined_x, dtype=float)
            _append_result_message(result, "Full-beam polish accepted")
    if point_match_mode:
        residual_fn = final_metric_residual_fn
    ridge_refinement_summary: Dict[str, object] = {
        "enabled": bool(ridge_refinement_enabled),
        "status": "skipped",
        "reason": "not_evaluated",
        "accepted": False,
    }
    if point_match_mode and getattr(result, "x", None) is not None:
        try:
            ridge_refinement_summary = _maybe_run_ridge_refinement(result)
        except Exception as exc:
            ridge_refinement_summary = {
                "enabled": bool(ridge_refinement_enabled),
                "status": "failed",
                "reason": f"unexpected_exception: {exc}",
                "accepted": False,
            }
        refined_x = ridge_refinement_summary.get("x")
        if ridge_refinement_summary.get("accepted") and refined_x is not None:
            result.x = np.asarray(refined_x, dtype=float)
            _append_result_message(result, "Ridge refinement accepted")

    image_refinement_summary: Dict[str, object] = {
        "enabled": bool(image_refinement_enabled),
        "status": "skipped",
        "reason": "not_evaluated",
        "accepted": False,
    }
    if point_match_mode and getattr(result, "x", None) is not None:
        try:
            image_refinement_summary = _maybe_run_image_refinement(result)
        except Exception as exc:
            image_refinement_summary = {
                "enabled": bool(image_refinement_enabled),
                "status": "failed",
                "reason": f"unexpected_exception: {exc}",
                "accepted": False,
            }
        refined_x = image_refinement_summary.get("x")
        if image_refinement_summary.get("accepted") and refined_x is not None:
            result.x = np.asarray(refined_x, dtype=float)
            _append_result_message(result, "ROI/image refinement accepted")

    auto_freeze_summary = _inactive_stage_summary(
        enabled=bool(auto_freeze_enabled),
        stage_name="auto_freeze",
    )
    selective_thaw_summary = _inactive_stage_summary(
        enabled=bool(selective_thaw_enabled),
        stage_name="selective_thaw",
    )

    result.restart_history = restart_history
    result.reparameterization_summary = reparameterization_summary
    result.staged_release_summary = staged_release_summary
    result.adaptive_regularization_summary = adaptive_regularization_summary
    result.ridge_refinement_summary = ridge_refinement_summary
    result.image_refinement_summary = image_refinement_summary
    result.auto_freeze_summary = auto_freeze_summary
    result.selective_thaw_summary = selective_thaw_summary
    result.full_beam_polish_summary = full_beam_polish_summary
    result.optimizer_method = str(solver_method)
    result.solver_loss = solver_loss
    result.solver_f_scale = float(solver_f_scale)
    result.parameter_prior_summary = parameter_prior_summary
    result.parallelization_summary = dict(parallelization_summary)
    result.chosen_discrete_mode = dict(best_mode)
    result.discrete_mode_summary = {
        "enabled": bool(discrete_modes_cfg.get("enabled", False)),
        "mode_count": int(len(discrete_modes)),
        "selected_mode": dict(best_mode),
        "selected_label": _format_discrete_mode_label(best_mode),
    }
    if point_match_mode and bool(full_beam_polish_summary.get("accepted", False)):
        result.final_metric_name = "full_beam_fixed_correspondence"
    elif point_match_mode and dynamic_point_geometry_fit:
        result.final_metric_name = "dynamic_angular_point_match"
    else:
        result.final_metric_name = "central_point_match" if point_match_mode else "angle"

    if getattr(result, "x", None) is not None:
        result.x = np.minimum(np.maximum(result.x, lower_bounds), upper_bounds)
        final_metric_fun = full_beam_polish_summary.get("fun")
        final_metric_x = full_beam_polish_summary.get("x")
        if (
            isinstance(final_metric_fun, np.ndarray)
            and isinstance(final_metric_x, np.ndarray)
            and np.asarray(final_metric_x, dtype=float).shape
            == np.asarray(result.x, dtype=float).shape
            and np.allclose(
                np.asarray(final_metric_x, dtype=float),
                np.asarray(result.x, dtype=float),
                atol=1.0e-12,
                rtol=0.0,
            )
        ):
            result.fun = np.asarray(final_metric_fun, dtype=float)
        else:
            result.fun = np.asarray(
                final_metric_residual_fn(np.asarray(result.x, dtype=float)),
                dtype=float,
            )
    elif getattr(result, "fun", None) is not None:
        result.fun = np.asarray(result.fun, dtype=float)
    else:
        result.fun = np.array([], dtype=float)

    result.robust_cost = _robust_cost(
        np.asarray(result.fun, dtype=float),
        loss=solver_loss,
        f_scale=solver_f_scale,
    )
    result.cost = 0.5 * float(np.sum(np.asarray(result.fun, dtype=float) ** 2))

    weighted_residual_rms = float("nan")
    result_fun = np.asarray(result.fun, dtype=float)
    if result_fun.size:
        weighted_residual_rms = float(np.sqrt(np.mean(result_fun * result_fun)))
    result.weighted_residual_rms_px = float(weighted_residual_rms)
    result.rms_px = float(weighted_residual_rms)

    if point_match_mode and getattr(result, "x", None) is not None:
        try:
            final_local = _apply_trial_params(np.asarray(result.x, dtype=float))
            _, point_match_diagnostics, point_match_summary = final_metric_point_match_evaluator(
                final_local,
                collect_diagnostics=True,
            )
            point_match_summary = _public_point_match_summary(point_match_summary)
            point_match_summary["metric_name"] = str(result.final_metric_name)
            point_match_summary["single_ray_coarse_enabled"] = bool(use_single_ray)
            point_match_summary["full_beam_polish_enabled"] = bool(
                full_beam_polish_summary.get("enabled", False)
            )
            point_match_summary["full_beam_polish_accepted"] = bool(
                full_beam_polish_summary.get("accepted", False)
            )
            point_match_summary["seed_correspondence_count"] = int(
                full_beam_polish_summary.get("seed_correspondence_count", 0)
            )
            point_match_summary["seed_rms_px"] = float(
                full_beam_polish_summary.get("seed_rms_px", np.nan)
            )
            point_match_summary["final_full_beam_rms_px"] = float(
                point_match_summary.get("unweighted_peak_rms_px", np.nan)
            )
            result.point_match_diagnostics = point_match_diagnostics
            result.point_match_summary = point_match_summary
            try:
                peak_rms = float(point_match_summary.get("unweighted_peak_rms_px", np.nan))
            except Exception:
                peak_rms = float("nan")
            if np.isfinite(peak_rms):
                result.rms_px = float(peak_rms)
        except Exception as exc:
            result.point_match_diagnostics = [
                {
                    "match_kind": "diagnostics",
                    "match_status": "failed",
                    "resolution_reason": str(exc),
                }
            ]
            result.point_match_summary = {
                "diagnostics_failed": True,
                "error": str(exc),
            }
            result.rms_px = float(weighted_residual_rms)

    bound_threshold_fraction = 0.01
    bound_entries: List[Dict[str, object]] = []
    if getattr(result, "x", None) is not None:
        x_result = np.asarray(result.x, dtype=float).reshape(-1)
        for idx, name in enumerate(var_names):
            if idx >= x_result.size or idx >= lower_bounds.size or idx >= upper_bounds.size:
                continue
            lower = float(lower_bounds[idx])
            upper = float(upper_bounds[idx])
            if not (np.isfinite(lower) and np.isfinite(upper) and upper > lower):
                continue
            value = float(x_result[idx])
            span_value = float(upper - lower)
            threshold_value = float(span_value * bound_threshold_fraction)
            lower_gap = float(abs(value - lower))
            upper_gap = float(abs(upper - value))
            if lower_gap <= threshold_value + 1.0e-12:
                bound_entries.append(
                    {
                        "name": str(name),
                        "side": "lower",
                        "value": float(value),
                        "bound": float(lower),
                        "gap": float(lower_gap),
                        "span": float(span_value),
                        "gap_fraction": float(lower_gap / span_value),
                    }
                )
            elif upper_gap <= threshold_value + 1.0e-12:
                bound_entries.append(
                    {
                        "name": str(name),
                        "side": "upper",
                        "value": float(value),
                        "bound": float(upper),
                        "gap": float(upper_gap),
                        "span": float(span_value),
                        "gap_fraction": float(upper_gap / span_value),
                    }
                )
    result.bound_proximity_summary = {
        "threshold_fraction": float(bound_threshold_fraction),
        "near_bound_parameters": bound_entries,
    }
    result.bound_hits = [str(entry.get("name", "")) for entry in bound_entries]
    result.boundary_warning = (
        "Possible identifiability issue: parameters finished near bounds ("
        + ", ".join(
            f"{entry['name']}={entry['side']}" for entry in bound_entries if entry.get("name")
        )
        + ")."
        if bound_entries
        else None
    )

    def _augment_identifiability_summary(
        summary: Dict[str, object],
        *,
        names: Sequence[str],
        weak_column_rel: float,
        correlation_warn: float,
    ) -> Dict[str, object]:
        if str(summary.get("status", "")) != "ok":
            return summary
        param_entries = list(summary.get("parameter_entries", []))
        column_norms = np.array(
            [
                float(entry.get("column_norm", np.nan))
                if isinstance(entry, Mapping)
                else float("nan")
                for entry in param_entries
            ],
            dtype=float,
        )
        finite_norms = column_norms[np.isfinite(column_norms) & (column_norms > 0.0)]
        median_norm = float(np.median(finite_norms)) if finite_norms.size else 0.0
        weak_threshold = max(float(weak_column_rel) * median_norm, 0.0)
        weak_parameters: List[Dict[str, object]] = []
        for idx, name in enumerate(names):
            column_norm = float(column_norms[idx]) if idx < column_norms.size else float("nan")
            if not np.isfinite(column_norm) or column_norm > weak_threshold + 1.0e-12:
                continue
            weak_parameters.append(
                {
                    "name": str(name),
                    "index": int(idx),
                    "column_norm": float(column_norm),
                    "reason": "weak_sensitivity",
                }
            )

        high_correlation_pairs: List[Dict[str, object]] = []
        corr_u = np.asarray(summary.get("correlation_u", []), dtype=float)
        if corr_u.ndim == 2 and corr_u.shape[0] == len(names):
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    corr_val = float(corr_u[i, j])
                    if not np.isfinite(corr_val):
                        continue
                    if abs(corr_val) < float(correlation_warn):
                        continue
                    high_correlation_pairs.append(
                        {
                            "name_i": str(names[i]),
                            "index_i": int(i),
                            "name_j": str(names[j]),
                            "index_j": int(j),
                            "correlation": float(corr_val),
                            "abs_correlation": float(abs(corr_val)),
                        }
                    )
        high_correlation_pairs.sort(
            key=lambda item: (
                -float(item.get("abs_correlation", 0.0)),
                str(item.get("name_i", "")),
                str(item.get("name_j", "")),
            )
        )

        warning_flags = [str(flag) for flag in summary.get("warning_flags", [])]
        if weak_parameters and "weak_sensitivity" not in warning_flags:
            warning_flags.append("weak_sensitivity")
        if high_correlation_pairs and "high_correlation" not in warning_flags:
            warning_flags.append("high_correlation")
        summary["weak_column_norm_threshold"] = float(weak_threshold)
        summary["weak_parameters"] = weak_parameters
        summary["high_correlation_pairs"] = high_correlation_pairs
        summary["recommended_fixed_parameters"] = []
        summary["recommended_fixed_indices"] = []
        summary["warning_flags"] = warning_flags
        return summary

    def _build_jacobian_for_specs(
        specs: Sequence[ParamSpec],
        *,
        u_ref: np.ndarray,
        include_priors: bool,
        base_params: Optional[Mapping[str, object]] = None,
        robust_weight: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not specs:
            return np.empty((0, 0), dtype=float), np.array([], dtype=float)
        lb_u_local, ub_u_local = _build_u_bounds(specs)
        fun = lambda u: _residual_u(
            u,
            specs=specs,
            include_priors=include_priors,
            base_params=base_params,
        )
        residual_ref = np.asarray(fun(np.asarray(u_ref, dtype=float)), dtype=float)
        jacobian = _finite_diff_jacobian_abs(
            fun,
            np.asarray(u_ref, dtype=float),
            np.full_like(np.asarray(u_ref, dtype=float), identifiability_fd_abs_step_u),
            lower=lb_u_local,
            upper=ub_u_local,
        )
        if robust_weight and solver_loss == "soft_l1" and residual_ref.size:
            row_scale = _soft_l1_row_scale(residual_ref, solver_f_scale)
            jacobian = row_scale[:, None] * jacobian
        return jacobian, residual_ref

    def _build_solver_conditioned_identifiability_summary() -> Dict[str, object]:
        if not identifiability_enabled:
            return {
                "enabled": False,
                "status": "skipped",
                "reason": "disabled_by_config",
                "underconstrained": False,
            }
        if getattr(result, "x", None) is None:
            return {
                "enabled": True,
                "status": "failed",
                "reason": "missing_parameter_vector",
                "underconstrained": False,
            }
        specs = list(active_specs)
        if not specs:
            return {
                "enabled": True,
                "status": "failed",
                "reason": "empty_parameter_vector",
                "underconstrained": False,
            }
        u_ref = _u_from_physical(specs, np.asarray(result.x, dtype=float))
        J_u, residual_ref = _build_jacobian_for_specs(
            specs,
            u_ref=np.asarray(u_ref, dtype=float),
            include_priors=True,
            robust_weight=True,
        )
        summary = _svd_summary(
            J_u,
            [spec.name for spec in specs],
            scales=[spec.scale for spec in specs],
            rcond=identifiability_sv_rcond,
            weak_singular_rel=identifiability_weak_singular_rel,
            combo_thresh=identifiability_combo_thresh,
        )
        summary["enabled"] = True
        summary["diagnostic_scope"] = "solver_conditioned_active"
        summary["includes_priors"] = True
        summary["residual_count"] = int(residual_ref.size)
        return _augment_identifiability_summary(
            summary,
            names=[spec.name for spec in specs],
            weak_column_rel=identifiability_weak_column_rel,
            correlation_warn=identifiability_block_corr_abs,
        )

    def _build_data_only_identifiability_summary() -> Dict[str, object]:
        if not identifiability_enabled:
            return {
                "enabled": False,
                "status": "skipped",
                "reason": "disabled_by_config",
                "underconstrained": False,
            }
        if getattr(result, "x", None) is None:
            return {
                "enabled": True,
                "status": "failed",
                "reason": "missing_parameter_vector",
                "underconstrained": False,
            }
        final_local = _apply_trial_params(np.asarray(result.x, dtype=float))
        candidate_specs = _build_specs_from_params(
            all_candidate_param_names,
            base_params=final_local,
        )
        if not candidate_specs:
            return {
                "enabled": True,
                "status": "failed",
                "reason": "no_candidate_parameters",
                "underconstrained": False,
            }
        u_ref = np.zeros(len(candidate_specs), dtype=float)
        J_u, residual_ref = _build_jacobian_for_specs(
            candidate_specs,
            u_ref=u_ref,
            include_priors=False,
            base_params=final_local,
            robust_weight=True,
        )
        summary = _svd_summary(
            J_u,
            [spec.name for spec in candidate_specs],
            scales=[spec.scale for spec in candidate_specs],
            rcond=identifiability_sv_rcond,
            weak_singular_rel=identifiability_weak_singular_rel,
            combo_thresh=identifiability_combo_thresh,
        )
        summary["enabled"] = True
        summary["diagnostic_scope"] = "data_only_all_selectable"
        summary["includes_priors"] = False
        summary["residual_count"] = int(residual_ref.size)
        summary = _augment_identifiability_summary(
            summary,
            names=[spec.name for spec in candidate_specs],
            weak_column_rel=identifiability_weak_column_rel,
            correlation_warn=identifiability_block_corr_abs,
        )
        summary["next_stage_recommendations"] = recommend_next_stage(
            J_u,
            param_names=[spec.name for spec in candidate_specs],
            active_names=[str(name) for name in var_names],
            rcond=identifiability_sv_rcond,
            block_corr_abs=identifiability_block_corr_abs,
            weak_column_rel=identifiability_weak_column_rel,
            single_release_min_indep=identifiability_single_release_min_indep,
            block_release_min_indep=identifiability_block_release_min_indep,
        )
        return summary

    debug_summary_result = copy.deepcopy(geometry_fit_debug_summary)
    final_x_arr = np.asarray(getattr(result, "x", []), dtype=float).reshape(-1)
    if final_x_arr.size == len(debug_summary_result.get("parameter_entries", [])):
        for idx, entry in enumerate(debug_summary_result.get("parameter_entries", [])):
            if not isinstance(entry, dict):
                continue
            try:
                start_value = float(entry.get("start", np.nan))
            except Exception:
                start_value = float("nan")
            final_value = float(final_x_arr[idx])
            entry["final"] = float(final_value)
            entry["delta"] = (
                float(final_value - start_value) if np.isfinite(start_value) else float("nan")
            )
    debug_summary_result["final"] = {
        "cost": float(getattr(result, "cost", np.nan)),
        "robust_cost": float(getattr(result, "robust_cost", np.nan)),
        "weighted_rms_px": float(weighted_residual_rms),
        "display_rms_px": float(getattr(result, "rms_px", np.nan)),
        "seed_rms_px": float(full_beam_polish_summary.get("seed_rms_px", np.nan)),
        "final_full_beam_rms_px": float(getattr(result, "rms_px", np.nan)),
        "metric_name": str(getattr(result, "final_metric_name", "")),
    }
    if isinstance(getattr(result, "point_match_summary", None), Mapping):
        debug_summary_result["final_point_match_summary"] = copy.deepcopy(
            getattr(result, "point_match_summary")
        )
    if isinstance(getattr(result, "geometry_fit_progress", None), Mapping):
        debug_summary_result["solve_progress"] = copy.deepcopy(
            getattr(result, "geometry_fit_progress")
        )
    result.geometry_fit_debug_summary = debug_summary_result

    identifiability_summary = _build_solver_conditioned_identifiability_summary()
    data_only_identifiability_summary = _build_data_only_identifiability_summary()
    result.identifiability_summary = identifiability_summary
    result.data_only_identifiability_summary = data_only_identifiability_summary
    result.next_stage_recommendations = list(
        data_only_identifiability_summary.get("next_stage_recommendations", [])
        if isinstance(data_only_identifiability_summary, Mapping)
        else []
    )
    result.next_stage_recommendation = (
        dict(result.next_stage_recommendations[0]) if result.next_stage_recommendations else None
    )
    point_match_summary = getattr(result, "point_match_summary", None)
    matched_pair_count_text = ""
    if isinstance(point_match_summary, Mapping):
        try:
            matched_pair_count_text = (
                f", matched={int(point_match_summary.get('matched_pair_count', 0))}"
            )
        except Exception:
            matched_pair_count_text = ""
    metric_name_text = str(getattr(result, "final_metric_name", "") or "").strip()
    metric_part = f", metric={metric_name_text}" if metric_name_text else ""
    _emit_status(
        "Geometry fit: complete "
        f"(cost={float(getattr(result, 'cost', np.nan)):.6f}, "
        f"rms={float(getattr(result, 'rms_px', np.nan)):.4f}px"
        f"{metric_part}{matched_pair_count_text})"
    )

    return result


def run_optimization_positions_geometry_local(
    miller, intensities, image_size, initial_params, bounds, measured_peaks
):
    """
    Global optimization (Differential Evolution) over geometry + beam center.
    bounds is list of (min,max) for [gamma, Gamma, dist, theta_i, cor_angle,
    zs, zb, chi, a, c, center_x, center_y].
    """

    def obj_glob(x):
        gamma, Gamma, dist, theta_initial, cor_angle, zs, zb, chi, a, c, center_x, center_y = x
        return np.sum(
            compute_peak_position_error_geometry_local(
                gamma,
                Gamma,
                dist,
                theta_initial,
                cor_angle,
                zs,
                zb,
                chi,
                a,
                c,
                center_x,
                center_y,
                measured_peaks,
                miller,
                intensities,
                image_size,
                initial_params["mosaic_params"],
                initial_params["n2"],
                initial_params.get("psi", 0.0),
                initial_params.get("psi_z", 0.0),
                initial_params["debye_x"],
                initial_params["debye_y"],
                initial_params["lambda"],
                pixel_tol=np.inf,
            )
        )

    res = differential_evolution(obj_glob, bounds, maxiter=200, popsize=15)
    return res
