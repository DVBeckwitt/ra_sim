"""Optimization routines for fitting simulated data to experiments."""

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares, differential_evolution, OptimizeResult
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel, zoom
from scipy.spatial import cKDTree

from ra_sim.simulation.diffraction import (
    hit_tables_to_max_positions,
    process_peaks_parallel,
)
from ra_sim.utils.calculations import d_spacing, two_theta

RNG = np.random.default_rng(42)

_USE_NUMBA_PROCESS_PEAKS = True


def _set_numba_usage_from_config(
    refinement_config: Optional[Dict[str, Dict[str, float]]],
) -> None:
    """Apply per-fit config flags (e.g., disable numba for fitting)."""

    global _USE_NUMBA_PROCESS_PEAKS
    if isinstance(refinement_config, dict):
        use_numba = refinement_config.get("use_numba", True)
        _USE_NUMBA_PROCESS_PEAKS = bool(use_numba)


def _process_peaks_parallel_safe(*args, **kwargs):
    """Call the numba-compiled process_peaks_parallel with a python fallback."""

    global _USE_NUMBA_PROCESS_PEAKS

    def _invoke(fn):
        try:
            return fn(*args, **kwargs)
        except TypeError as exc:
            # Test doubles and older callsites may not accept the new keyword.
            if (
                (
                    "solve_q_steps" in kwargs
                    or "solve_q_rel_tol" in kwargs
                    or "solve_q_mode" in kwargs
                )
                and "unexpected keyword" in str(exc)
            ):
                reduced_kwargs = dict(kwargs)
                reduced_kwargs.pop("solve_q_steps", None)
                reduced_kwargs.pop("solve_q_rel_tol", None)
                reduced_kwargs.pop("solve_q_mode", None)
                return fn(*args, **reduced_kwargs)
            raise

    if _USE_NUMBA_PROCESS_PEAKS:
        try:
            return _invoke(process_peaks_parallel)
        except Exception:
            _USE_NUMBA_PROCESS_PEAKS = False
    py_runner = getattr(process_peaks_parallel, "py_func", None)
    if callable(py_runner):
        return _invoke(py_runner)
    return _invoke(process_peaks_parallel)


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
class SimulationCache:
    """Simple cache for simulated detector images keyed by parameter vectors."""

    keys: Sequence[str]
    images: Dict[Tuple[float, ...], np.ndarray] = field(default_factory=dict)
    max_positions: Dict[Tuple[float, ...], np.ndarray] = field(default_factory=dict)

    def _flatten_value(self, value: np.ndarray) -> Iterable[float]:
        if isinstance(value, np.ndarray):
            return value.ravel()
        if isinstance(value, (list, tuple)):
            return np.asarray(value, dtype=float).ravel()
        return (float(value),)

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
        key = self.key_for(params)
        self.images[key] = image
        self.max_positions[key] = max_positions


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
    for name, val in zip(var_names, values):
        updated[name] = float(val)
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
    if not measured_peaks:
        return normalized

    for entry in measured_peaks:
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
        normalized.append(normalized_entry)
    return normalized


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


def _miller_key_from_row(row: Sequence[float]) -> Optional[Tuple[int, int, int]]:
    """Return an integer HKL tuple from a Miller-array row."""

    try:
        return (
            int(round(float(row[0]))),
            int(round(float(row[1]))),
            int(round(float(row[2]))),
        )
    except Exception:
        return None


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

    selected_original_indices: List[int] = []
    selected_lookup: set[int] = set()

    for entry in normalized_measured:
        source_key = _measured_source_indices(entry)
        if source_key is None:
            continue
        table_idx, _ = source_key
        if table_idx < 0 or table_idx >= total_reflections or table_idx in selected_lookup:
            continue
        selected_lookup.add(table_idx)
        selected_original_indices.append(int(table_idx))

    fixed_source_reflection_count = int(len(selected_original_indices))

    fallback_hkl_keys: set[Tuple[int, int, int]] = set()
    for entry in normalized_measured:
        source_key = _measured_source_indices(entry)
        if source_key is not None:
            table_idx, _ = source_key
            if 0 <= table_idx < total_reflections:
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

    remapped_measured: List[Dict[str, object]] = []
    for entry in normalized_measured:
        remapped_entry = dict(entry)
        source_key = _measured_source_indices(entry)
        if source_key is not None:
            table_idx, row_idx = source_key
            local_idx = local_index_map.get(int(table_idx))
            if local_idx is not None:
                remapped_entry["source_table_index"] = int(local_idx)
                remapped_entry["source_row_index"] = int(row_idx)
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

    if dataset_specs:
        for dataset_index, raw_entry in enumerate(dataset_specs):
            if not isinstance(raw_entry, dict):
                raise TypeError(
                    "geometry fit dataset_specs entries must be dictionaries"
                )
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
            )
        )
    return contexts


def _build_greedy_point_matches(
    simulated_points: Sequence[Tuple[float, float]],
    measured_points: Sequence[Tuple[float, float]],
    *,
    max_distance: float = np.inf,
) -> List[Tuple[np.ndarray, np.ndarray, float, int, int]]:
    """Greedy one-to-one matching between measured and simulated point sets."""

    if not simulated_points or not measured_points:
        return []

    sim = np.asarray(simulated_points, dtype=float)
    meas = np.asarray(measured_points, dtype=float)
    if sim.ndim != 2 or meas.ndim != 2 or sim.shape[1] != 2 or meas.shape[1] != 2:
        return []

    dist = np.linalg.norm(meas[:, None, :] - sim[None, :, :], axis=2)
    if not np.isfinite(max_distance):
        mask = np.isfinite(dist)
    else:
        mask = np.isfinite(dist) & (dist <= float(max_distance))
    candidate_pairs = np.argwhere(mask)
    if candidate_pairs.size == 0:
        return []

    candidate_dist = dist[candidate_pairs[:, 0], candidate_pairs[:, 1]]
    order = np.argsort(candidate_dist)
    used_meas: set[int] = set()
    used_sim: set[int] = set()
    matches: List[Tuple[np.ndarray, np.ndarray, float, int, int]] = []
    for idx in order:
        m_idx = int(candidate_pairs[idx, 0])
        s_idx = int(candidate_pairs[idx, 1])
        if m_idx in used_meas or s_idx in used_sim:
            continue
        d = float(candidate_dist[idx])
        matches.append((sim[s_idx], meas[m_idx], d, s_idx, m_idx))
        used_meas.add(m_idx)
        used_sim.add(s_idx)
    return matches


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
    collect_diagnostics: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
    """Evaluate one measured dataset against one simulated geometry state."""

    simulation_subset = dataset_ctx.subset
    fit_miller = simulation_subset.miller
    fit_intensities = simulation_subset.intensities
    normalized_measured = simulation_subset.measured_entries
    single_ray_indices = dataset_ctx.single_ray_indices

    if not normalized_measured:
        return np.array([], dtype=float), [], {
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
            "single_ray_enabled": bool(use_single_ray),
            "single_ray_forced_count": 0,
        }

    mosaic = local["mosaic_params"]
    wavelength_array = mosaic.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic.get("wavelength_i_array")

    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
    _, hit_tables, *_ = _process_peaks_parallel_safe(
        fit_miller, fit_intensities, image_size,
        local["a"], local["c"], wavelength_array,
        sim_buffer, local["corto_detector"],
        local["gamma"], local["Gamma"], local["chi"], local.get("psi", 0.0), local.get("psi_z", 0.0),
        local["zs"], local["zb"], local["n2"],
        mosaic["beam_x_array"],
        mosaic["beam_y_array"],
        mosaic["theta_array"],
        mosaic["phi_array"],
        mosaic["sigma_mosaic_deg"],
        mosaic["gamma_mosaic_deg"],
        mosaic["eta"],
        wavelength_array,
        local["debye_x"], local["debye_y"],
        local["center"], theta_value, local.get("cor_angle", 0.0),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        save_flag=0,
        optics_mode=int(local.get("optics_mode", 0)),
        solve_q_steps=int(mosaic.get("solve_q_steps", 1000)),
        solve_q_rel_tol=float(mosaic.get("solve_q_rel_tol", 5.0e-4)),
        solve_q_mode=int(mosaic.get("solve_q_mode", 1)),
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
    residuals: List[float] = []
    diagnostics: List[Dict[str, object]] = []
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

    for measured_entry, sim_pt, _sim_hkl in fixed_matches:
        try:
            meas_pt = (float(measured_entry["x"]), float(measured_entry["y"]))
        except Exception:
            continue
        if not (
            np.isfinite(sim_pt[0]) and np.isfinite(sim_pt[1])
            and np.isfinite(meas_pt[0]) and np.isfinite(meas_pt[1])
        ):
            continue
        dx = float(sim_pt[0] - meas_pt[0])
        dy = float(sim_pt[1] - meas_pt[1])
        pair_dist = math.hypot(dx, dy)
        if weighted_matching:
            w = 1.0 / math.sqrt(1.0 + (pair_dist / solver_f_scale) ** 2)
            dx *= w
            dy *= w
        else:
            w = 1.0
        residuals.extend([dx, dy])
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
                "weight": float(w),
                "weighted_dx_px": float(dx),
                "weighted_dy_px": float(dy),
                "measured_radius_px": _point_radius_px(meas_pt),
                "simulated_radius_px": _point_radius_px(sim_pt),
            },
        )

    measured_dict = build_measured_dict(fallback_measured)
    simulated_by_hkl: Dict[Tuple[int, int, int], List[Tuple[float, float]]] = {}
    for idx, (H, K, L) in enumerate(fit_miller):
        key = (int(round(H)), int(round(K)), int(round(L)))
        if key not in measured_dict:
            continue
        _, x0, y0, _, x1, y1 = maxpos[idx]
        for col, row in ((x0, y0), (x1, y1)):
            if np.isfinite(col) and np.isfinite(row):
                simulated_by_hkl.setdefault(key, []).append((float(col), float(row)))

    missing_pairs = 0
    for hkl_key in measured_dict:
        sim_list = simulated_by_hkl.get(hkl_key, [])
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
                _add_diag(
                    dict(resolution_lookup.get(id(entry), {})),
                    {
                        "match_kind": "hkl_fallback",
                        "match_status": "missing_pair",
                        "hkl": tuple(int(v) for v in hkl_key),
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
                        "weight": 0.0,
                        "weighted_dx_px": float("nan"),
                        "weighted_dy_px": float("nan"),
                        "measured_radius_px": _point_radius_px((float(entry["x"]), float(entry["y"]))),
                        "simulated_radius_px": float("nan"),
                    },
                )
            continue

        matches = _build_greedy_point_matches(sim_list, measured_points, max_distance=pixel_tol)
        matched_meas_indices: set[int] = set()
        for sim_pt, meas_pt, pair_dist, sim_idx, meas_idx in matches:
            matched_meas_indices.add(int(meas_idx))
            dx = float(sim_pt[0] - meas_pt[0])
            dy = float(sim_pt[1] - meas_pt[1])
            if weighted_matching:
                w = 1.0 / math.sqrt(1.0 + (pair_dist / solver_f_scale) ** 2)
                dx *= w
                dy *= w
            else:
                w = 1.0
            residuals.extend([dx, dy])
            if 0 <= meas_idx < len(valid_entries_hkl):
                entry = valid_entries_hkl[meas_idx]
                _add_diag(
                    dict(resolution_lookup.get(id(entry), {})),
                    {
                        "match_kind": "hkl_fallback",
                        "match_status": "matched",
                        "hkl": tuple(int(v) for v in hkl_key),
                        "sim_list_index": int(sim_idx),
                        "meas_list_index": int(meas_idx),
                        "measured_x": float(meas_pt[0]),
                        "measured_y": float(meas_pt[1]),
                        "simulated_x": float(sim_pt[0]),
                        "simulated_y": float(sim_pt[1]),
                        "dx_px": float(sim_pt[0] - meas_pt[0]),
                        "dy_px": float(sim_pt[1] - meas_pt[1]),
                        "distance_px": float(pair_dist),
                        "weight": float(w),
                        "weighted_dx_px": float(dx),
                        "weighted_dy_px": float(dy),
                        "measured_radius_px": _point_radius_px((float(meas_pt[0]), float(meas_pt[1]))),
                        "simulated_radius_px": _point_radius_px((float(sim_pt[0]), float(sim_pt[1]))),
                    },
                )

        unmatched_meas_indices = [idx for idx in range(len(valid_entries_hkl)) if idx not in matched_meas_indices]
        missing_pairs += len(unmatched_meas_indices)
        for meas_idx in unmatched_meas_indices:
            entry = valid_entries_hkl[meas_idx]
            _add_diag(
                dict(resolution_lookup.get(id(entry), {})),
                {
                    "match_kind": "hkl_fallback",
                    "match_status": "missing_pair",
                    "hkl": tuple(int(v) for v in hkl_key),
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
                    "weight": 0.0,
                    "weighted_dx_px": float("nan"),
                    "weighted_dy_px": float("nan"),
                    "measured_radius_px": _point_radius_px((float(entry["x"]), float(entry["y"]))),
                    "simulated_radius_px": float("nan"),
                },
            )

    if missing_pairs > 0 and missing_pair_penalty > 0.0:
        residuals.extend([missing_pair_penalty] * missing_pairs)

    residual_arr = (
        np.asarray(residuals, dtype=float)
        if residuals
        else np.array([max(1.0, missing_pair_penalty)], dtype=float)
    )
    summary: Dict[str, object] = {
        "dataset_index": int(dataset_ctx.dataset_index),
        "dataset_label": str(dataset_ctx.label),
        "theta_initial_deg": float(theta_value),
        "measured_count": int(len(normalized_measured)),
        "fixed_source_resolved_count": int(len(fixed_matches)),
        "fallback_entry_count": int(len(fallback_measured)),
        "fallback_hkl_count": int(len(measured_dict)),
        "missing_pair_count": int(missing_pairs),
        "simulated_reflection_count": int(fit_miller.shape[0]),
        "total_reflection_count": int(simulation_subset.total_reflection_count),
        "fixed_source_reflection_count": int(simulation_subset.fixed_source_reflection_count),
        "subset_fallback_hkl_count": int(simulation_subset.fallback_hkl_count),
        "subset_reduced": bool(simulation_subset.reduced),
        "single_ray_enabled": bool(use_single_ray),
        "single_ray_forced_count": int(np.count_nonzero(single_ray_indices >= 0))
        if isinstance(single_ray_indices, np.ndarray)
        else 0,
        "center_row": float(local["center"][0]) if len(local.get("center", [])) >= 2 else float("nan"),
        "center_col": float(local["center"][1]) if len(local.get("center", [])) >= 2 else float("nan"),
    }
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

    mosaic = params['mosaic_params']
    wavelength_array = mosaic.get('wavelength_array')
    if wavelength_array is None:
        wavelength_array = mosaic.get('wavelength_i_array')
    if wavelength_array is None:
        wavelength_array = params.get('lambda')

    image, hit_tables, *_ = _process_peaks_parallel_safe(
        miller, intensities, image_size,
        params['a'], params['c'], wavelength_array,
        buffer, params['corto_detector'],
        params['gamma'], params['Gamma'], params['chi'], params.get('psi', 0.0), params.get('psi_z', 0.0),
        params['zs'], params['zb'], params['n2'],
        mosaic['beam_x_array'],
        mosaic['beam_y_array'],
        mosaic['theta_array'],
        mosaic['phi_array'],
        mosaic['sigma_mosaic_deg'],
        mosaic['gamma_mosaic_deg'],
        mosaic['eta'],
        wavelength_array,
        params['debye_x'], params['debye_y'],
        params['center'], params['theta_initial'], params.get('cor_angle', 0.0),
        params.get('uv1', np.array([1.0, 0.0, 0.0])),
        params.get('uv2', np.array([0.0, 1.0, 0.0])),
        save_flag=0,
        optics_mode=int(params.get('optics_mode', 0)),
        solve_q_steps=int(mosaic.get('solve_q_steps', 1000)),
        solve_q_rel_tol=float(mosaic.get('solve_q_rel_tol', 5.0e-4)),
        solve_q_mode=int(mosaic.get('solve_q_mode', 1)),
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
        raise ValueError(
            "experimental_image shape must match the provided image_size"
        )

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
        raise RuntimeError(
            "No reflections satisfy 2h + k ≡ 0 (mod 3) for mosaic-width fitting"
        )

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
            solve_q_steps=solve_q_steps,
            solve_q_rel_tol=solve_q_rel_tol,
            solve_q_mode=solve_q_mode,
        )
        image = np.asarray(image, dtype=np.float64)
        if not record_hits:
            return image, None
        hits_py = [np.asarray(tbl) for tbl in hit_tables]
        return image, hits_py

    allowed_miller = np.ascontiguousarray(miller[allowed_indices], dtype=np.float64)
    allowed_intensities = np.ascontiguousarray(
        intensities[allowed_indices], dtype=np.float64
    )

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
        raise RuntimeError(
            "No reflections satisfy the 2h + k ≡ 0 constraint within 65° 2θ"
        )

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
        trial_cost = _robust_cost(np.asarray(trial.fun, dtype=np.float64), loss=loss, f_scale=f_scale)
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


def _estimate_pixel_size(params: Dict[str, float]) -> float:
    pixel_size = params.get('pixel_size')
    if pixel_size is None:
        # Fall back to detector distance divided by nominal pixels for 4k detector
        pixel_size = params.get('corto_detector', 1.0) / 4096.0
    return max(float(pixel_size), 1e-6)


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
    mosaic = params['mosaic_params']
    sigma_deg = float(mosaic.get('sigma_mosaic_deg', 0.3))
    mosaic_fwhm_rad = math.radians(sigma_deg) * 2.0 * math.sqrt(2.0 * math.log(2.0))
    divergence_rad = math.radians(float(mosaic.get('gamma_mosaic_deg', 0.3)))
    bandwidth_rad = float(params.get('bandwidth_rad', 0.0))
    detector_length = float(params.get('corto_detector', 1.0))

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
        grid_y, grid_x = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]
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
        weights_map = upsampled[min_y:max_y + 1, min_x:max_x + 1]
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

        roi.active_pixels = np.stack((global_y, global_x), axis=1)
        roi.weights = weights_map[ys, xs]
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
        patch = experimental_image[min_y:max_y + 1, min_x:max_x + 1]
        mask = roi.mask
        center_mask = roi.centerline_mask if roi.centerline_mask is not None else np.zeros_like(mask, dtype=bool)
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
        A = np.column_stack([
            np.ones_like(xx),
            xx,
            yy,
            xx * xx,
            xx * yy,
            yy * yy,
        ])
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

        background[min_y:max_y + 1, min_x:max_x + 1] += fitted
        coverage[min_y:max_y + 1, min_x:max_x + 1] += 1.0
        global_mask[min_y:max_y + 1, min_x:max_x + 1] |= mask

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
    mixture_weight = (1.0 - mixture) + mixture * np.exp(-0.5 * (ans_res / (huber_delta + 1e-6)) ** 2)
    combined = 0.5 * np.sqrt(np.abs(dev)) + 0.5 * huber_res
    scaling = np.sqrt(np.maximum(weights, 1e-8)) / max(sampling_probability, 1e-3)
    return combined * scaling * np.sqrt(mixture_weight)


def _select_active_reflections(
    rois: Sequence[TubeROI],
    *,
    max_reflections: int,
    random_fraction: float,
) -> List[TubeROI]:
    scored = [
        (float(np.sum(roi.weights)) if roi.weights is not None else 0.0, roi)
        for roi in rois
    ]
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
                miller[idx:idx + 1],
                max_positions[idx:idx + 1],
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

    result = least_squares(residual, x0, max_nfev=max_nfev)
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
) -> Tuple[Dict[str, float], OptimizeResult, List[TubeROI], np.ndarray, Callable[[np.ndarray], np.ndarray]]:
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
        downsample_factor=int(cfg.get('downsample_factor', 4)),
        percentile=float(cfg.get('percentile', 90.0)),
        huber_percentile=float(cfg.get('huber_percentile', 97.0)),
        per_reflection_quota=int(cfg.get('per_reflection_quota', 200)),
        off_tube_fraction=float(cfg.get('off_tube_fraction', 0.05)),
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
            threshold=float(cfg.get('roi_refresh_threshold', 1.0)),
        )

        missing = [roi for roi in rois_state if roi.weights is None]
        if missing:
            compute_sensitivity_weights(
                sim_img,
                trial,
                var_names,
                rois_state,
                simulator,
                downsample_factor=int(cfg.get('downsample_factor', 4)),
                percentile=float(cfg.get('percentile', 90.0)),
                huber_percentile=float(cfg.get('huber_percentile', 97.0)),
                per_reflection_quota=int(cfg.get('per_reflection_quota', 200)),
                off_tube_fraction=float(cfg.get('off_tube_fraction', 0.05)),
            )

        active = _select_active_reflections(
            rois_state,
            max_reflections=int(cfg.get('max_reflections', 12)),
            random_fraction=float(cfg.get('random_reflection_fraction', 0.15)),
        )

        background = fit_local_background(experimental_image, rois_state)
        sampled = sample_tiles(
            active,
            temperature=float(cfg.get('sampling_temperature', 1.0)),
            explore_fraction=float(cfg.get('explore_fraction', 0.15)),
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
                huber_delta=float(cfg.get('huber_delta', 2.5)),
                mixture=float(cfg.get('outlier_mixture', 0.1)),
            )
            residuals_list.append(res)

        if not residuals_list:
            return np.zeros(1, dtype=float)
        return np.concatenate(residuals_list)

    x0 = np.array([params[name] for name in var_names], dtype=float)
    result = least_squares(residual, x0, max_nfev=int(cfg.get('max_nfev', 25)))
    updated_params = _update_params(params, var_names, result.x)
    final_sim, _ = simulator(updated_params)
    rois_state = _refresh_rois_if_needed(
        rois_state,
        miller,
        maxpos,
        updated_params,
        image_size,
        measured_dict=measured_dict,
        threshold=float(cfg.get('roi_refresh_threshold', 1.0)),
    )
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
        downsample_factor=int(cfg.get('downsample_factor', 1)),
        percentile=float(cfg.get('percentile', 85.0)),
        huber_percentile=float(cfg.get('huber_percentile', 98.0)),
        per_reflection_quota=int(cfg.get('per_reflection_quota', 400)),
        off_tube_fraction=float(cfg.get('off_tube_fraction', 0.1)),
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
            threshold=float(cfg.get('roi_refresh_threshold', 0.5)),
        )

        compute_sensitivity_weights(
            sim_img,
            trial,
            var_names,
            rois_state,
            simulator,
            downsample_factor=int(cfg.get('downsample_factor', 1)),
            percentile=float(cfg.get('percentile', 85.0)),
            huber_percentile=float(cfg.get('huber_percentile', 98.0)),
            per_reflection_quota=int(cfg.get('per_reflection_quota', 400)),
            off_tube_fraction=float(cfg.get('off_tube_fraction', 0.1)),
        )

        active = _select_active_reflections(
            rois_state,
            max_reflections=int(cfg.get('max_reflections', len(rois_state))),
            random_fraction=float(cfg.get('random_reflection_fraction', 0.1)),
        )

        background = fit_local_background(experimental_image, rois_state)
        sampled = sample_tiles(
            active,
            temperature=float(cfg.get('sampling_temperature', 0.7)),
            explore_fraction=float(cfg.get('explore_fraction', 0.1)),
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
                huber_delta=float(cfg.get('huber_delta', 2.0)),
                mixture=float(cfg.get('outlier_mixture', 0.05)),
            )
            residuals_list.append(res)

        if not residuals_list:
            return np.zeros(1, dtype=float)
        return np.concatenate(residuals_list)

    x0 = np.array([params[name] for name in var_names], dtype=float)
    result = least_squares(residual, x0, max_nfev=int(cfg.get('max_nfev', 35)))
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
        var_names = ('zb', 'zs', 'theta_initial', 'chi')
    var_names = list(var_names)

    measured_dict = build_measured_dict(measured_peaks or [])

    cache_keys = [
        'gamma', 'Gamma', 'corto_detector', 'theta_initial',
        'zs', 'zb', 'chi', 'a', 'c', 'center'
    ]
    cache = SimulationCache(cache_keys)

    def simulator(local_params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        merged = dict(params)
        merged.update(local_params)
        return _simulate_with_cache(merged, miller, intensities, image_size, cache)

    config = config or {}
    stage1_cfg = {'downsample_factor': 8, 'max_nfev': 20}
    stage1_cfg.update(config.get('stage1', {}))
    stage2_cfg = {
        'downsample_factor': 4,
        'percentile': 90.0,
        'huber_percentile': 97.0,
        'per_reflection_quota': 200,
        'off_tube_fraction': 0.05,
        'max_reflections': 12,
        'random_reflection_fraction': 0.15,
        'sampling_temperature': 1.0,
        'explore_fraction': 0.15,
        'huber_delta': 2.5,
        'outlier_mixture': 0.1,
        'max_nfev': 25,
        'roi_refresh_threshold': 1.0,
    }
    stage2_cfg.update(config.get('stage2', {}))
    stage3_cfg = {
        'downsample_factor': 1,
        'percentile': 85.0,
        'huber_percentile': 98.0,
        'per_reflection_quota': 400,
        'off_tube_fraction': 0.1,
        'max_reflections': len(miller),
        'random_reflection_fraction': 0.1,
        'sampling_temperature': 0.7,
        'explore_fraction': 0.1,
        'huber_delta': 2.0,
        'outlier_mixture': 0.05,
        'max_nfev': 35,
        'roi_refresh_threshold': 0.5,
    }
    stage3_cfg.update(config.get('stage3', {}))

    history: List[Dict[str, float]] = []
    stage_summaries: List[Dict[str, float]] = []
    current_params = dict(params)

    current_params, stage1_result = _stage_one_initialize(
        experimental_image,
        current_params,
        var_names,
        simulator,
        downsample_factor=stage1_cfg['downsample_factor'],
        max_nfev=stage1_cfg['max_nfev'],
    )
    history.append({name: current_params[name] for name in var_names})
    stage_summaries.append({
        'stage': 'level1',
        'cost': float(np.sum(stage1_result.fun ** 2)),
        'nfev': stage1_result.nfev,
    })

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
    stage_summaries.append({
        'stage': 'level2',
        'cost': float(np.sum(stage2_result.fun ** 2)),
        'nfev': stage2_result.nfev,
    })

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
    stage_summaries.append({
        'stage': 'level3',
        'cost': float(np.sum(stage3_result.fun ** 2)),
        'nfev': stage3_result.nfev,
    })

    final_residual = stage3_residual(np.array([current_params[name] for name in var_names]))

    return IterativeRefinementResult(
        x=np.array([current_params[name] for name in var_names], dtype=float),
        fun=final_residual,
        success=bool(stage1_result.success and stage2_result.success and stage3_result.success),
        message='; '.join(filter(None, [stage1_result.message, stage2_result.message, stage3_result.message])),
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


def _measured_source_indices(
    entry: Dict[str, object],
) -> Optional[Tuple[int, int]]:
    """Return the preview-style hit-table indices for a measured entry when present."""

    try:
        table_idx = int(entry.get("source_table_index"))  # type: ignore[arg-type]
        row_idx = int(entry.get("source_row_index"))  # type: ignore[arg-type]
    except Exception:
        return None
    if table_idx < 0 or row_idx < 0:
        return None
    return table_idx, row_idx


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
    resolved: List[
        Tuple[Dict[str, object], Tuple[float, float], Tuple[int, int, int]]
    ] = []
    fallback_entries: List[Dict[str, object]] = []
    resolution_lookup: Dict[int, Dict[str, object]] = {}
    used_source_keys: set[Tuple[int, int]] = set()

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
            "hkl": tuple(entry.get("hkl", ())) if isinstance(entry.get("hkl"), tuple) else entry.get("hkl"),
            "source_table_index": entry.get("source_table_index"),
            "source_row_index": entry.get("source_row_index"),
        }
        source_key = _measured_source_indices(entry)
        if source_key is None:
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
                "resolution_kind": "hkl_fallback",
                "resolution_reason": "duplicate_source_key",
            }
            continue

        table_idx, row_idx = source_key
        if table_idx not in filtered_rows_cache:
            if table_idx < 0 or table_idx >= len(hit_tables):
                filtered_rows_cache[table_idx] = []
            else:
                filtered_rows_cache[table_idx] = _valid_hit_rows(hit_tables[table_idx])

        rows = filtered_rows_cache.get(table_idx, [])
        if row_idx < 0 or row_idx >= len(rows):
            fallback_entries.append(entry)
            resolution_lookup[id(entry)] = {
                **base_diag,
                "resolution_kind": "hkl_fallback",
                "resolution_reason": "source_row_out_of_range",
                "source_row_count": int(len(rows)),
            }
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
            fallback_entries.append(entry)
            resolution_lookup[id(entry)] = {
                **base_diag,
                "resolution_kind": "hkl_fallback",
                "resolution_reason": "source_row_parse_failed",
            }
            continue
        if not (np.isfinite(sim_col) and np.isfinite(sim_row)):
            fallback_entries.append(entry)
            resolution_lookup[id(entry)] = {
                **base_diag,
                "resolution_kind": "hkl_fallback",
                "resolution_reason": "invalid_simulated_point",
            }
            continue

        resolved.append((entry, (sim_col, sim_row), sim_hkl))
        resolution_lookup[id(entry)] = {
            **base_diag,
            "resolution_kind": "fixed_source",
            "resolution_reason": "resolved",
            "sim_x": float(sim_col),
            "sim_y": float(sim_row),
            "sim_hkl": tuple(int(v) for v in sim_hkl),
        }
        used_source_keys.add(source_key)

    return resolved, fallback_entries, resolution_lookup


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

    if center is None or len(center) < 2:
        return None, None
    if not np.isfinite(detector_distance) or detector_distance <= 0:
        return None, None
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        return None, None

    try:
        centre_row = float(center[0])
        centre_col = float(center[1])
    except (TypeError, ValueError, IndexError):
        return None, None

    if not (np.isfinite(col) and np.isfinite(row)):
        return None, None

    dx = (float(col) - centre_col) * pixel_size
    dy = (centre_row - float(row)) * pixel_size
    radius = math.hypot(dx, dy)

    two_theta = math.degrees(math.atan2(radius, detector_distance))
    phi = math.degrees(math.atan2(dx, dy))
    return two_theta, phi


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
    a = params['a']; c = params['c']
    dist = params['corto_detector']
    gamma = params['gamma']; Gamma = params['Gamma']
    chi   = params['chi']; psi = params.get('psi', 0.0); psi_z = params.get('psi_z', 0.0)
    zs    = params['zs']; zb    = params['zb']
    debye_x = params['debye_x']; debye_y = params['debye_y']
    n2    = params['n2']
    center = params['center']
    theta_initial = params['theta_initial']
    cor_angle = params.get('cor_angle', 0.0)

    mosaic = params['mosaic_params']
    wavelength_array = mosaic.get('wavelength_array')
    if wavelength_array is None:
        wavelength_array = mosaic.get('wavelength_i_array')

    # Full-pattern simulation
    updated_image, hit_tables, *_ = _process_peaks_parallel_safe(
        sim_miller, sim_intensities, image_size,
        a, c, wavelength_array,
        sim_buffer, dist,
        gamma, Gamma, chi, psi, psi_z,
        zs, zb, n2,
        mosaic['beam_x_array'],
        mosaic['beam_y_array'],
        mosaic['theta_array'],
        mosaic['phi_array'],
        mosaic['sigma_mosaic_deg'],
        mosaic['gamma_mosaic_deg'],
        mosaic['eta'],
        wavelength_array,
        debye_x, debye_y,
        center, theta_initial, cor_angle,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        save_flag=0,
        optics_mode=int(params.get('optics_mode', 0)),
        solve_q_steps=int(mosaic.get('solve_q_steps', 1000)),
        solve_q_rel_tol=float(mosaic.get('solve_q_rel_tol', 5.0e-4)),
        solve_q_mode=int(mosaic.get('solve_q_mode', 1)),
    )
    maxpos = hit_tables_to_max_positions(hit_tables)

    distances: list[float] = []
    sim_coords: list[tuple[float, float]] = []
    meas_coords: list[tuple[float, float]] = []
    sim_millers: list[tuple[int, int, int]] = []
    meas_millers: list[tuple[int, int, int]] = []

    detector_distance = float(params.get('corto_detector', 0.0))
    pixel_size = _estimate_pixel_size(params)
    centre = params.get('center')

    def _match_points(
        simulated_points: list[tuple[float, float]],
        measured_points: list[tuple[float, float]],
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        if not simulated_points or not measured_points:
            return []

        sim_arr = np.asarray(simulated_points, dtype=float)
        meas_arr = np.asarray(measured_points, dtype=float)
        if sim_arr.ndim != 2 or meas_arr.ndim != 2 or sim_arr.shape[1] != 2 or meas_arr.shape[1] != 2:
            return []

        dist = np.linalg.norm(meas_arr[:, None, :] - sim_arr[None, :, :], axis=2)
        candidate_pairs = np.argwhere(np.isfinite(dist))
        if candidate_pairs.size == 0:
            return []

        candidate_dist = dist[candidate_pairs[:, 0], candidate_pairs[:, 1]]
        order = np.argsort(candidate_dist)
        used_meas: set[int] = set()
        used_sim: set[int] = set()
        matches: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for idx in order:
            meas_idx = int(candidate_pairs[idx, 0])
            sim_idx = int(candidate_pairs[idx, 1])
            if meas_idx in used_meas or sim_idx in used_sim:
                continue
            sim_pt = (float(sim_arr[sim_idx, 0]), float(sim_arr[sim_idx, 1]))
            meas_pt = (float(meas_arr[meas_idx, 0]), float(meas_arr[meas_idx, 1]))
            matches.append((sim_pt, meas_pt))
            used_meas.add(meas_idx)
            used_sim.add(sim_idx)
        return matches

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
            if sim_two_theta is None or sim_phi is None or meas_two_theta is None or meas_phi is None:
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
    gamma, Gamma, dist, theta_initial, cor_angle, zs, zb, chi, a, c,
    center_x, center_y, measured_peaks,
    miller, intensities, image_size, mosaic_params, n2,
    psi, psi_z, debye_x, debye_y, wavelength, pixel_tol=np.inf, optics_mode=0
):
    """
    Objective for DE: returns the 1D array of distances for all matched peaks.
    """
    params = {
        'gamma': gamma,
        'Gamma': Gamma,
        'corto_detector': dist,
        'theta_initial': theta_initial,
        'cor_angle': cor_angle,
        'zs': zs,
        'zb': zb,
        'chi': chi,
        'a': a,
        'c': c,
        'center': (center_x, center_y),
        'lambda': wavelength,
        'n2': n2,
        'psi': psi,
        'psi_z': psi_z,
        'debye_x': debye_x,
        'debye_y': debye_y,
        'mosaic_params': mosaic_params,
        'optics_mode': int(optics_mode),
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
    miller, intensities, image_size,
    params, measured_peaks, var_names, pixel_tol=np.inf,
    *, experimental_image: Optional[np.ndarray] = None,
    dataset_specs: Optional[Sequence[Dict[str, object]]] = None,
    refinement_config: Optional[Dict[str, Dict[str, float]]] = None,
):
    """
    Least-squares fit for a subset of geometry parameters.
    var_names is a list of keys in `params` to optimize.
    """

    _set_numba_usage_from_config(refinement_config)

    point_match_mode = experimental_image is not None or bool(dataset_specs)

    single_ray_cfg: Dict[str, float] = {}
    use_single_ray = False
    if point_match_mode:
        if isinstance(refinement_config, dict):
            single_ray_cfg = refinement_config.get("single_ray", {}) or {}
        if not isinstance(single_ray_cfg, dict):
            single_ray_cfg = {}
        use_single_ray = bool(single_ray_cfg.get("enabled", True))

    solver_cfg: Dict[str, float] = {}
    if isinstance(refinement_config, dict):
        solver_cfg = refinement_config.get("solver", {}) or {}
    if not isinstance(solver_cfg, dict):
        solver_cfg = {}

    solver_loss = str(
        solver_cfg.get("loss", "soft_l1" if point_match_mode else "linear")
    ).strip().lower()
    if solver_loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
        solver_loss = "soft_l1" if point_match_mode else "linear"

    solver_f_scale = float(
        solver_cfg.get("f_scale_px", 6.0 if point_match_mode else 1.0)
    )
    solver_f_scale = max(solver_f_scale, 1e-6)

    solver_max_nfev = int(solver_cfg.get("max_nfev", 120 if point_match_mode else 60))
    solver_max_nfev = max(20, solver_max_nfev)

    solver_restarts = int(solver_cfg.get("restarts", 4 if point_match_mode else 0))
    solver_restarts = max(0, solver_restarts)

    solver_restart_jitter = float(solver_cfg.get("restart_jitter", 0.15))
    solver_restart_jitter = max(0.0, solver_restart_jitter)

    missing_pair_penalty = float(solver_cfg.get("missing_pair_penalty_px", 20.0))
    missing_pair_penalty = max(0.0, missing_pair_penalty)

    weighted_matching = bool(solver_cfg.get("weighted_matching", True))
    stagnation_probe_enabled = bool(
        solver_cfg.get("stagnation_probe", point_match_mode)
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
        dataset_specs=dataset_specs,
    )
    multi_dataset_mode = ("theta_offset" in var_names) and len(dataset_contexts) > 0

    def _safe_float(value: object, fallback: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        if not np.isfinite(out):
            return float(fallback)
        return out

    def _apply_trial_params(x: Sequence[float]) -> Dict[str, object]:
        local = params.copy()
        center_pair = local.get("center", [center_row_default, center_col_default])
        try:
            center_row = _safe_float(center_pair[0], center_row_default)
            center_col = _safe_float(center_pair[1], center_col_default)
        except Exception:
            center_row = center_row_default
            center_col = center_col_default

        center_row = _safe_float(local.get("center_x", center_row), center_row)
        center_col = _safe_float(local.get("center_y", center_col), center_col)

        for name, v in zip(var_names, x):
            val = float(v)
            if name == "center_x":
                center_row = val
                local["center_x"] = val
            elif name == "center_y":
                center_col = val
                local["center_y"] = val
            else:
                local[name] = val

        local["center"] = [float(center_row), float(center_col)]
        local["center_x"] = float(center_row)
        local["center_y"] = float(center_col)
        return local

    def _build_point_matches(
        simulated_points: Sequence[Tuple[float, float]],
        measured_points: Sequence[Tuple[float, float]],
        *,
        max_distance: float = np.inf,
    ) -> List[Tuple[np.ndarray, np.ndarray, float, int, int]]:
        """Greedy one-to-one matching between measured and simulated peak points."""

        if not simulated_points or not measured_points:
            return []

        sim = np.asarray(simulated_points, dtype=float)
        meas = np.asarray(measured_points, dtype=float)
        if sim.ndim != 2 or meas.ndim != 2 or sim.shape[1] != 2 or meas.shape[1] != 2:
            return []

        dist = np.linalg.norm(meas[:, None, :] - sim[None, :, :], axis=2)
        if not np.isfinite(max_distance):
            mask = np.isfinite(dist)
        else:
            mask = np.isfinite(dist) & (dist <= float(max_distance))
        candidate_pairs = np.argwhere(mask)
        if candidate_pairs.size == 0:
            return []

        candidate_dist = dist[candidate_pairs[:, 0], candidate_pairs[:, 1]]
        order = np.argsort(candidate_dist)

        used_meas: set[int] = set()
        used_sim: set[int] = set()
        matches: List[Tuple[np.ndarray, np.ndarray, float, int, int]] = []
        for idx in order:
            m_idx = int(candidate_pairs[idx, 0])
            s_idx = int(candidate_pairs[idx, 1])
            if m_idx in used_meas or s_idx in used_sim:
                continue
            d = float(candidate_dist[idx])
            matches.append((sim[s_idx], meas[m_idx], d, s_idx, m_idx))
            used_meas.add(m_idx)
            used_sim.add(s_idx)
        return matches

    def _legacy_cost_fn_unused(x):
        local = _apply_trial_params(x)
        args = [
            local['gamma'], local['Gamma'], local['corto_detector'],
            local['theta_initial'], local.get('cor_angle', 0.0), local['zs'], local['zb'],
            local['chi'], local['a'], local['c'],
            local['center'][0], local['center'][1]
        ]
        D = compute_peak_position_error_geometry_local(
            *args,
            measured_peaks=measured_peaks,
            miller=miller,
            intensities=intensities,
            image_size=image_size,
            mosaic_params=local['mosaic_params'],
            n2=local['n2'],
            psi=local.get('psi', 0.0),
            psi_z=local.get('psi_z', 0.0),
            debye_x=local['debye_x'],
            debye_y=local['debye_y'],
            wavelength=local['lambda'],
            pixel_tol=pixel_tol,
            optics_mode=local.get('optics_mode', 0),
        )
        return D

    def _legacy_evaluate_pixel_matches_unused(
        local: Dict[str, object],
        *,
        collect_diagnostics: bool = False,
    ) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
        normalized_measured = fit_measured_peaks
        if not normalized_measured:
            return np.array([], dtype=float), [], {
                "measured_count": 0,
                "fixed_source_resolved_count": 0,
                "fallback_entry_count": 0,
                "missing_pair_count": 0,
                "simulated_reflection_count": int(fit_miller.shape[0]),
                "total_reflection_count": int(simulation_subset.total_reflection_count),
                "subset_reduced": bool(simulation_subset.reduced),
                "single_ray_enabled": bool(use_single_ray),
            }

        mosaic = local['mosaic_params']
        wavelength_array = mosaic.get('wavelength_array')
        if wavelength_array is None:
            wavelength_array = mosaic.get('wavelength_i_array')

        sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
        _, hit_tables, *_ = _process_peaks_parallel_safe(
            fit_miller, fit_intensities, image_size,
            local['a'], local['c'], wavelength_array,
            sim_buffer, local['corto_detector'],
            local['gamma'], local['Gamma'], local['chi'], local.get('psi', 0.0), local.get('psi_z', 0.0),
            local['zs'], local['zb'], local['n2'],
            mosaic['beam_x_array'],
            mosaic['beam_y_array'],
            mosaic['theta_array'],
            mosaic['phi_array'],
            mosaic['sigma_mosaic_deg'],
            mosaic['gamma_mosaic_deg'],
            mosaic['eta'],
            wavelength_array,
            local['debye_x'], local['debye_y'],
            local['center'], local['theta_initial'], local.get('cor_angle', 0.0),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            save_flag=0,
            optics_mode=int(local.get('optics_mode', 0)),
            solve_q_steps=int(mosaic.get('solve_q_steps', 1000)),
            solve_q_rel_tol=float(mosaic.get('solve_q_rel_tol', 5.0e-4)),
            solve_q_mode=int(mosaic.get('solve_q_mode', 1)),
            single_sample_indices=single_ray_indices,
        )

        def _point_radius_px(point: Tuple[float, float]) -> float:
            try:
                center_row = float(local['center'][0])
                center_col = float(local['center'][1])
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
        residuals: list[float] = []
        diagnostics: List[Dict[str, object]] = []
        fixed_matches, fallback_measured, resolution_lookup = _resolve_fixed_source_matches(
            normalized_measured,
            hit_tables,
        )
        fallback_entries_by_hkl: Dict[Tuple[int, int, int], List[Dict[str, object]]] = {}
        for entry in fallback_measured:
            raw_hkl = entry.get("hkl")
            if not isinstance(raw_hkl, tuple) or len(raw_hkl) != 3:
                continue
            hkl_key = (
                int(raw_hkl[0]),
                int(raw_hkl[1]),
                int(raw_hkl[2]),
            )
            fallback_entries_by_hkl.setdefault(hkl_key, []).append(entry)

        for measured_entry, sim_pt, _sim_hkl in fixed_matches:
            try:
                meas_pt = (
                    float(measured_entry["x"]),
                    float(measured_entry["y"]),
                )
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
            if weighted_matching:
                w = 1.0 / math.sqrt(1.0 + (pair_dist / solver_f_scale) ** 2)
                dx *= w
                dy *= w
            else:
                w = 1.0
            residuals.extend([dx, dy])
            if collect_diagnostics:
                entry_diag = dict(resolution_lookup.get(id(measured_entry), {}))
                entry_diag.update(
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
                        "weight": float(w),
                        "weighted_dx_px": float(dx),
                        "weighted_dy_px": float(dy),
                        "measured_radius_px": _point_radius_px(meas_pt),
                        "simulated_radius_px": _point_radius_px(sim_pt),
                    }
                )
                diagnostics.append(entry_diag)

        measured_dict = build_measured_dict(fallback_measured)

        simulated_by_hkl: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
        for idx, (H, K, L) in enumerate(fit_miller):
            key = (int(round(H)), int(round(K)), int(round(L)))
            if key not in measured_dict:
                continue
            _, x0, y0, _, x1, y1 = maxpos[idx]
            for col, row in ((x0, y0), (x1, y1)):
                if np.isfinite(col) and np.isfinite(row):
                    simulated_by_hkl.setdefault(key, []).append((float(col), float(row)))

        missing_pairs = 0
        for hkl_key, measured_list in measured_dict.items():
            sim_list = simulated_by_hkl.get(hkl_key, [])
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
                if collect_diagnostics:
                    for entry in valid_entries_hkl:
                        entry_diag = dict(resolution_lookup.get(id(entry), {}))
                        entry_diag.update(
                            {
                                "match_kind": "hkl_fallback",
                                "match_status": "missing_pair",
                                "hkl": tuple(int(v) for v in hkl_key),
                                "resolution_kind": str(
                                    entry_diag.get("resolution_kind", "hkl_fallback")
                                ),
                                "resolution_reason": str(
                                    entry_diag.get("resolution_reason", "no_simulated_candidates")
                                ),
                                "measured_x": float(entry["x"]),
                                "measured_y": float(entry["y"]),
                                "simulated_x": float("nan"),
                                "simulated_y": float("nan"),
                                "dx_px": float("nan"),
                                "dy_px": float("nan"),
                                "distance_px": float("nan"),
                                "weight": 0.0,
                                "weighted_dx_px": float("nan"),
                                "weighted_dy_px": float("nan"),
                                "measured_radius_px": _point_radius_px(
                                    (float(entry["x"]), float(entry["y"]))
                                ),
                                "simulated_radius_px": float("nan"),
                            }
                        )
                        diagnostics.append(entry_diag)
                continue

            matches = _build_point_matches(
                sim_list,
                measured_points,
                max_distance=pixel_tol,
            )
            matched_meas_indices: set[int] = set()
            for sim_pt, meas_pt, pair_dist, sim_idx, meas_idx in matches:
                matched_meas_indices.add(int(meas_idx))
                dx = float(sim_pt[0] - meas_pt[0])
                dy = float(sim_pt[1] - meas_pt[1])
                if weighted_matching:
                    w = 1.0 / math.sqrt(1.0 + (pair_dist / solver_f_scale) ** 2)
                    dx *= w
                    dy *= w
                else:
                    w = 1.0
                residuals.extend([dx, dy])
                if collect_diagnostics and 0 <= meas_idx < len(valid_entries_hkl):
                    entry = valid_entries_hkl[meas_idx]
                    entry_diag = dict(resolution_lookup.get(id(entry), {}))
                    entry_diag.update(
                        {
                            "match_kind": "hkl_fallback",
                            "match_status": "matched",
                            "hkl": tuple(int(v) for v in hkl_key),
                            "sim_list_index": int(sim_idx),
                            "meas_list_index": int(meas_idx),
                            "measured_x": float(meas_pt[0]),
                            "measured_y": float(meas_pt[1]),
                            "simulated_x": float(sim_pt[0]),
                            "simulated_y": float(sim_pt[1]),
                            "dx_px": float(sim_pt[0] - meas_pt[0]),
                            "dy_px": float(sim_pt[1] - meas_pt[1]),
                            "distance_px": float(pair_dist),
                            "weight": float(w),
                            "weighted_dx_px": float(dx),
                            "weighted_dy_px": float(dy),
                            "measured_radius_px": _point_radius_px(
                                (float(meas_pt[0]), float(meas_pt[1]))
                            ),
                            "simulated_radius_px": _point_radius_px(
                                (float(sim_pt[0]), float(sim_pt[1]))
                            ),
                        }
                    )
                    diagnostics.append(entry_diag)

            unmatched_meas_indices = [
                idx for idx in range(len(valid_entries_hkl)) if idx not in matched_meas_indices
            ]
            missing_pairs += len(unmatched_meas_indices)
            if collect_diagnostics:
                for meas_idx in unmatched_meas_indices:
                    entry = valid_entries_hkl[meas_idx]
                    entry_diag = dict(resolution_lookup.get(id(entry), {}))
                    entry_diag.update(
                        {
                            "match_kind": "hkl_fallback",
                            "match_status": "missing_pair",
                            "hkl": tuple(int(v) for v in hkl_key),
                            "resolution_kind": str(
                                entry_diag.get("resolution_kind", "hkl_fallback")
                            ),
                            "resolution_reason": str(
                                entry_diag.get("resolution_reason", "unmatched_after_assignment")
                            ),
                            "measured_x": float(entry["x"]),
                            "measured_y": float(entry["y"]),
                            "simulated_x": float("nan"),
                            "simulated_y": float("nan"),
                            "dx_px": float("nan"),
                            "dy_px": float("nan"),
                            "distance_px": float("nan"),
                            "weight": 0.0,
                            "weighted_dx_px": float("nan"),
                            "weighted_dy_px": float("nan"),
                            "measured_radius_px": _point_radius_px(
                                (float(entry["x"]), float(entry["y"]))
                            ),
                            "simulated_radius_px": float("nan"),
                        }
                    )
                    diagnostics.append(entry_diag)

        if missing_pairs > 0 and missing_pair_penalty > 0.0:
            residuals.extend([missing_pair_penalty] * missing_pairs)

        residual_arr = (
            np.asarray(residuals, dtype=float)
            if residuals
            else np.array([max(1.0, missing_pair_penalty)], dtype=float)
        )
        summary: Dict[str, object] = {
            "measured_count": int(len(normalized_measured)),
            "fixed_source_resolved_count": int(len(fixed_matches)),
            "fallback_entry_count": int(len(fallback_measured)),
            "fallback_hkl_count": int(len(measured_dict)),
            "missing_pair_count": int(missing_pairs),
            "simulated_reflection_count": int(fit_miller.shape[0]),
            "total_reflection_count": int(simulation_subset.total_reflection_count),
            "fixed_source_reflection_count": int(
                simulation_subset.fixed_source_reflection_count
            ),
            "subset_fallback_hkl_count": int(simulation_subset.fallback_hkl_count),
            "subset_reduced": bool(simulation_subset.reduced),
            "single_ray_enabled": bool(use_single_ray),
            "single_ray_forced_count": int(
                np.count_nonzero(single_ray_indices >= 0)
            )
            if isinstance(single_ray_indices, np.ndarray)
            else 0,
            "center_row": float(local['center'][0]) if len(local.get('center', [])) >= 2 else float("nan"),
            "center_col": float(local['center'][1]) if len(local.get('center', [])) >= 2 else float("nan"),
        }
        if diagnostics:
            sim_radius = np.asarray(
                [
                    float(entry.get("simulated_radius_px", np.nan))
                    for entry in diagnostics
                ],
                dtype=float,
            )
            meas_radius = np.asarray(
                [
                    float(entry.get("measured_radius_px", np.nan))
                    for entry in diagnostics
                ],
                dtype=float,
            )
            finite_sim_radius = sim_radius[np.isfinite(sim_radius)]
            finite_meas_radius = meas_radius[np.isfinite(meas_radius)]
            if finite_sim_radius.size:
                summary.update(
                    {
                        "sim_radius_min_px": float(np.min(finite_sim_radius)),
                        "sim_radius_median_px": float(np.median(finite_sim_radius)),
                        "sim_radius_lt_10px": int(np.count_nonzero(finite_sim_radius < 10.0)),
                        "sim_radius_lt_25px": int(np.count_nonzero(finite_sim_radius < 25.0)),
                    }
                )
            if finite_meas_radius.size:
                summary.update(
                    {
                        "meas_radius_min_px": float(np.min(finite_meas_radius)),
                        "meas_radius_median_px": float(np.median(finite_meas_radius)),
                        "meas_radius_lt_10px": int(np.count_nonzero(finite_meas_radius < 10.0)),
                        "meas_radius_lt_25px": int(np.count_nonzero(finite_meas_radius < 25.0)),
                    }
                )
            matched_distances = np.asarray(
                [
                    float(entry.get("distance_px", np.nan))
                    for entry in diagnostics
                    if str(entry.get("match_status", "")).lower() == "matched"
                ],
                dtype=float,
            )
            matched_distances = matched_distances[np.isfinite(matched_distances)]
            summary["matched_pair_count"] = int(matched_distances.size)
            if matched_distances.size:
                summary.update(
                    {
                        "unweighted_peak_rms_px": float(
                            np.sqrt(np.mean(matched_distances * matched_distances))
                        ),
                        "unweighted_peak_mean_px": float(np.mean(matched_distances)),
                        "unweighted_peak_max_px": float(np.max(matched_distances)),
                        "peak_weighting_mode": "uniform",
                    }
                )
            else:
                summary.update(
                    {
                        "unweighted_peak_rms_px": float("nan"),
                        "unweighted_peak_mean_px": float("nan"),
                        "unweighted_peak_max_px": float("nan"),
                        "peak_weighting_mode": "uniform",
                    }
                )
        else:
            summary.update(
                {
                    "matched_pair_count": 0,
                    "unweighted_peak_rms_px": float("nan"),
                    "unweighted_peak_mean_px": float("nan"),
                    "unweighted_peak_max_px": float("nan"),
                    "peak_weighting_mode": "uniform",
                }
            )
        return residual_arr, diagnostics, summary

    def _theta_initial_for_dataset(
        local: Dict[str, object],
        dataset_ctx: GeometryFitDatasetContext,
    ) -> float:
        theta_base = _safe_float(dataset_ctx.theta_initial, 0.0)
        if multi_dataset_mode:
            theta_offset = _safe_float(local.get("theta_offset", 0.0), 0.0)
            return float(theta_base + theta_offset)
        return _safe_float(local.get("theta_initial", theta_base), theta_base)

    def cost_fn(x):
        local = _apply_trial_params(x)
        residual_blocks: List[np.ndarray] = []
        for dataset_ctx in dataset_contexts:
            theta_value = _theta_initial_for_dataset(local, dataset_ctx)
            args = [
                local['gamma'], local['Gamma'], local['corto_detector'],
                theta_value, local.get('cor_angle', 0.0), local['zs'], local['zb'],
                local['chi'], local['a'], local['c'],
                local['center'][0], local['center'][1]
            ]
            residual = compute_peak_position_error_geometry_local(
                *args,
                measured_peaks=dataset_ctx.subset.measured_entries,
                miller=dataset_ctx.subset.miller,
                intensities=dataset_ctx.subset.intensities,
                image_size=image_size,
                mosaic_params=local['mosaic_params'],
                n2=local['n2'],
                psi=local.get('psi', 0.0),
                psi_z=local.get('psi_z', 0.0),
                debye_x=local['debye_x'],
                debye_y=local['debye_y'],
                wavelength=local['lambda'],
                pixel_tol=pixel_tol,
                optics_mode=local.get('optics_mode', 0),
            )
            residual_arr = np.asarray(residual, dtype=float)
            if residual_arr.size:
                residual_blocks.append(residual_arr)
        return (
            np.concatenate(residual_blocks)
            if residual_blocks
            else np.array([], dtype=float)
        )

    def _evaluate_pixel_matches(
        local: Dict[str, object],
        *,
        collect_diagnostics: bool = False,
    ) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
        if not dataset_contexts:
            return np.array([], dtype=float), [], {
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
                "single_ray_enabled": bool(use_single_ray),
                "single_ray_forced_count": 0,
            }

        residual_blocks: List[np.ndarray] = []
        diagnostics: List[Dict[str, object]] = []
        per_dataset_summaries: List[Dict[str, object]] = []
        summary: Dict[str, object] = {
            "dataset_count": int(len(dataset_contexts)),
            "measured_count": 0,
            "fixed_source_resolved_count": 0,
            "fallback_entry_count": 0,
            "missing_pair_count": 0,
            "simulated_reflection_count": 0,
            "total_reflection_count": 0,
            "fixed_source_reflection_count": 0,
            "subset_fallback_hkl_count": 0,
            "subset_reduced": False,
            "single_ray_enabled": bool(use_single_ray),
            "single_ray_forced_count": 0,
            "center_row": float(local['center'][0]) if len(local.get('center', [])) >= 2 else float("nan"),
            "center_col": float(local['center'][1]) if len(local.get('center', [])) >= 2 else float("nan"),
        }

        for dataset_ctx in dataset_contexts:
            theta_value = _theta_initial_for_dataset(local, dataset_ctx)
            residual_i, diagnostics_i, summary_i = _evaluate_geometry_fit_dataset_point_matches(
                local,
                dataset_ctx,
                image_size=image_size,
                pixel_tol=float(pixel_tol),
                weighted_matching=bool(weighted_matching),
                solver_f_scale=float(solver_f_scale),
                missing_pair_penalty=float(missing_pair_penalty),
                use_single_ray=bool(use_single_ray),
                theta_value=float(theta_value),
                collect_diagnostics=collect_diagnostics,
            )
            residual_i = np.asarray(residual_i, dtype=float)
            if residual_i.size:
                residual_blocks.append(residual_i)
            per_dataset_summaries.append(dict(summary_i))
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
            ):
                summary[key] = int(summary.get(key, 0)) + int(summary_i.get(key, 0))
            summary["subset_reduced"] = bool(
                summary.get("subset_reduced", False) or bool(summary_i.get("subset_reduced", False))
            )
            if collect_diagnostics and diagnostics_i:
                diagnostics.extend(diagnostics_i)

        residual_arr = (
            np.concatenate(residual_blocks)
            if residual_blocks
            else np.array([], dtype=float)
        )
        summary["per_dataset"] = per_dataset_summaries

        matched_distances = np.asarray(
            [
                float(entry.get("distance_px", np.nan))
                for entry in diagnostics
                if str(entry.get("match_status", "")).lower() == "matched"
            ],
            dtype=float,
        )
        matched_distances = matched_distances[np.isfinite(matched_distances)]
        summary["matched_pair_count"] = int(matched_distances.size)
        summary["peak_weighting_mode"] = "uniform"
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

    def pixel_cost_fn(x):
        local = _apply_trial_params(x)
        residual_arr, _, _ = _evaluate_pixel_matches(local, collect_diagnostics=False)
        return residual_arr

    def _initial_value(name: str) -> float:
        if name == "center_x":
            return _safe_float(
                params.get("center_x", params.get("center", [center_row_default, center_col_default])[0]),
                center_row_default,
            )
        if name == "center_y":
            return _safe_float(
                params.get("center_y", params.get("center", [center_row_default, center_col_default])[1]),
                center_col_default,
            )
        return _safe_float(params.get(name, 0.0), 0.0)

    x0 = [_initial_value(name) for name in var_names]

    if point_match_mode and use_single_ray and dataset_contexts:
        local = _apply_trial_params(np.asarray(x0, dtype=float))
        mosaic = local['mosaic_params']
        wavelength_array = mosaic.get('wavelength_array')
        if wavelength_array is None:
            wavelength_array = mosaic.get('wavelength_i_array')

        for dataset_ctx in dataset_contexts:
            fit_miller = dataset_ctx.subset.miller
            fit_measured_peaks = dataset_ctx.subset.measured_entries
            if fit_miller.shape[0] <= 0 or not fit_measured_peaks:
                dataset_ctx.single_ray_indices = None
                continue
            try:
                theta_value = _theta_initial_for_dataset(local, dataset_ctx)
                best_indices = np.full(fit_miller.shape[0], -1, dtype=np.int64)
                sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
                _process_peaks_parallel_safe(
                    fit_miller, dataset_ctx.subset.intensities, image_size,
                    local['a'], local['c'], wavelength_array,
                    sim_buffer, local['corto_detector'],
                    local['gamma'], local['Gamma'], local['chi'], local.get('psi', 0.0), local.get('psi_z', 0.0),
                    local['zs'], local['zb'], local['n2'],
                    mosaic['beam_x_array'],
                    mosaic['beam_y_array'],
                    mosaic['theta_array'],
                    mosaic['phi_array'],
                    mosaic['sigma_mosaic_deg'],
                    mosaic['gamma_mosaic_deg'],
                    mosaic['eta'],
                    wavelength_array,
                    local['debye_x'], local['debye_y'],
                    local['center'], theta_value, local.get('cor_angle', 0.0),
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    save_flag=0,
                    optics_mode=int(local.get('optics_mode', 0)),
                    solve_q_steps=int(mosaic.get('solve_q_steps', 1000)),
                    solve_q_rel_tol=float(mosaic.get('solve_q_rel_tol', 5.0e-4)),
                    solve_q_mode=int(mosaic.get('solve_q_mode', 1)),
                    best_sample_indices_out=best_indices,
                )
                dataset_ctx.single_ray_indices = best_indices
            except Exception:
                dataset_ctx.single_ray_indices = None

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

    residual_fn = pixel_cost_fn if point_match_mode else cost_fn

    def _run_solver(x_start: np.ndarray) -> OptimizeResult:
        return least_squares(
            residual_fn,
            np.asarray(x_start, dtype=float),
            bounds=(lower_bounds, upper_bounds),
            x_scale=x_scale,
            loss=solver_loss,
            f_scale=solver_f_scale,
            max_nfev=solver_max_nfev,
        )

    def _evaluate_cost_at(x_trial: np.ndarray) -> Tuple[np.ndarray, float]:
        residual = np.asarray(residual_fn(np.asarray(x_trial, dtype=float)), dtype=float)
        return residual, _robust_cost(
            residual,
            loss=solver_loss,
            f_scale=solver_f_scale,
        )

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

    result = _run_solver(x0_arr)
    best_result = result
    initial_cost = _robust_cost(
        np.asarray(result.fun, dtype=float),
        loss=solver_loss,
        f_scale=solver_f_scale,
    )
    best_cost = float(initial_cost)
    restart_history: List[Dict[str, object]] = [
        {
            "restart": 0,
            "start_x": x0_arr.tolist(),
            "end_x": np.asarray(result.x, dtype=float).tolist(),
            "cost": float(best_cost),
            "success": bool(result.success),
            "message": str(result.message),
        }
    ]

    if solver_restarts > 0 and x0_arr.size > 0:
        restart_rng = np.random.default_rng(20260214)
        restart_scale = np.where(finite_span, span, fallback_scale)
        restart_scale = np.maximum(restart_scale, 1e-6)

        for restart_idx in range(solver_restarts):
            anchor = np.asarray(best_result.x, dtype=float)
            perturb = restart_rng.uniform(-1.0, 1.0, size=anchor.shape)
            trial_start = anchor + solver_restart_jitter * restart_scale * perturb
            trial_start = np.clip(trial_start, lower_bounds, upper_bounds)
            if np.allclose(trial_start, anchor, atol=1e-12, rtol=0.0):
                continue

            trial = _run_solver(trial_start)
            trial_cost = _robust_cost(
                np.asarray(trial.fun, dtype=float),
                loss=solver_loss,
                f_scale=solver_f_scale,
            )
            restart_history.append(
                {
                    "restart": restart_idx + 1,
                    "start_x": np.asarray(trial_start, dtype=float).tolist(),
                    "end_x": np.asarray(trial.x, dtype=float).tolist(),
                    "cost": float(trial_cost),
                    "success": bool(trial.success),
                    "message": str(trial.message),
                }
            )
            if trial_cost < best_cost:
                best_result = trial
                best_cost = trial_cost

    stagnation_tol = max(
        stagnation_probe_min_improvement,
        1e-9 * max(abs(float(initial_cost)), 1.0),
    )
    best_x = np.asarray(getattr(best_result, "x", x0_arr), dtype=float)
    best_is_initial = (
        best_x.shape == x0_arr.shape
        and np.allclose(best_x, x0_arr, atol=1e-12, rtol=0.0)
    )
    best_improvement = float(initial_cost - best_cost)
    if (
        stagnation_probe_enabled
        and stagnation_probe_fraction > 0.0
        and x0_arr.size > 0
        and (best_is_initial or best_improvement <= stagnation_tol)
    ):
        probe_anchor = np.asarray(best_x, dtype=float)
        probe_scale = np.where(
            finite_span,
            span,
            np.maximum(fallback_scale, x_scale),
        )
        probe_scale = np.maximum(probe_scale, 1e-6)
        visited = {
            tuple(np.round(np.asarray(probe_anchor, dtype=float), 12).tolist()),
        }
        best_probe_x: Optional[np.ndarray] = None
        best_probe_residual: Optional[np.ndarray] = None
        best_probe_cost = float("inf")
        best_probe_label = ""
        probe_candidates: List[Tuple[np.ndarray, str, str]] = []
        for idx, name in enumerate(var_names):
            step = float(stagnation_probe_fraction * probe_scale[idx])
            if not np.isfinite(step) or step <= 0.0:
                continue
            for direction, direction_label in ((-1.0, "-"), (1.0, "+")):
                direction_vec = np.zeros_like(probe_anchor, dtype=float)
                direction_vec[idx] = float(direction)
                probe_candidates.append(
                    (direction_vec, "directional probe", f"{name}{direction_label}")
                )

        if (
            stagnation_probe_pairwise
            and probe_anchor.size > 1
            and stagnation_probe_pair_limit > 1
        ):
            ranked_indices = np.argsort(probe_scale)[::-1]
            ranked_indices = ranked_indices[: min(stagnation_probe_pair_limit, ranked_indices.size)]
            for first_pos, first_idx in enumerate(ranked_indices):
                idx_i = int(first_idx)
                for second_idx in ranked_indices[first_pos + 1 :]:
                    idx_j = int(second_idx)
                    for dir_i, label_i in ((-1.0, "-"), (1.0, "+")):
                        for dir_j, label_j in ((-1.0, "-"), (1.0, "+")):
                            direction_vec = np.zeros_like(probe_anchor, dtype=float)
                            direction_vec[idx_i] = float(dir_i)
                            direction_vec[idx_j] = float(dir_j)
                            probe_candidates.append(
                                (
                                    direction_vec / math.sqrt(2.0),
                                    "pairwise probe",
                                    (
                                        f"{var_names[idx_i]}{label_i},"
                                        f"{var_names[idx_j]}{label_j}"
                                    ),
                                )
                            )

        if stagnation_probe_random_directions > 0 and probe_anchor.size > 1:
            probe_rng = np.random.default_rng(20260310)
            for probe_idx in range(stagnation_probe_random_directions):
                direction_vec = probe_rng.normal(size=probe_anchor.shape[0])
                if not np.all(np.isfinite(direction_vec)):
                    continue
                norm = float(np.linalg.norm(direction_vec))
                if norm <= 1.0e-12:
                    continue
                probe_candidates.append(
                    (
                        direction_vec / norm,
                        "random probe",
                        f"rand#{probe_idx + 1}",
                    )
                )

        for direction_vec, probe_kind, probe_label in probe_candidates:
            trial_x = np.asarray(probe_anchor, dtype=float) + (
                stagnation_probe_fraction * probe_scale * np.asarray(direction_vec, dtype=float)
            )
            trial_x = np.minimum(np.maximum(trial_x, lower_bounds), upper_bounds)
            if np.allclose(trial_x, probe_anchor, atol=1e-12, rtol=0.0):
                continue
            trial_key = tuple(np.round(np.asarray(trial_x, dtype=float), 12).tolist())
            if trial_key in visited:
                continue
            visited.add(trial_key)
            trial_residual, trial_cost = _evaluate_cost_at(trial_x)
            restart_history.append(
                {
                    "restart": len(restart_history),
                    "start_x": np.asarray(trial_x, dtype=float).tolist(),
                    "end_x": np.asarray(trial_x, dtype=float).tolist(),
                    "cost": float(trial_cost),
                    "success": False,
                    "message": (
                        f"{probe_kind} {probe_label} "
                        "(cost-only seed evaluation)"
                    ),
                }
            )
            if trial_cost + stagnation_tol < best_probe_cost:
                best_probe_x = np.asarray(trial_x, dtype=float)
                best_probe_residual = np.asarray(trial_residual, dtype=float)
                best_probe_cost = float(trial_cost)
                best_probe_label = f"{probe_kind} {probe_label}"

        if (
            best_probe_x is not None
            and best_probe_residual is not None
            and best_probe_cost + stagnation_tol < best_cost
        ):
            trial = _run_solver(best_probe_x)
            trial_cost = _robust_cost(
                np.asarray(trial.fun, dtype=float),
                loss=solver_loss,
                f_scale=solver_f_scale,
            )
            restart_history.append(
                {
                    "restart": len(restart_history),
                    "start_x": np.asarray(best_probe_x, dtype=float).tolist(),
                    "end_x": np.asarray(trial.x, dtype=float).tolist(),
                    "cost": float(trial_cost),
                    "success": bool(trial.success),
                    "message": (
                        f"{best_probe_label} refine: "
                        f"{str(trial.message).strip()}"
                    ),
                }
            )

            if best_probe_cost + stagnation_tol < trial_cost:
                trial = _build_probe_result(
                    best_probe_x,
                    best_probe_residual,
                    message=(
                        f"{best_probe_label} improved the fit; "
                        "local least-squares did not improve further."
                    ),
                )
                trial_cost = float(best_probe_cost)

            if trial_cost < best_cost:
                best_result = trial
                best_cost = trial_cost

    result = best_result
    result.restart_history = restart_history
    result.robust_cost = float(best_cost)
    result.solver_loss = solver_loss
    result.solver_f_scale = float(solver_f_scale)
    weighted_residual_rms = float("nan")
    if getattr(result, "fun", None) is not None:
        result_fun = np.asarray(result.fun, dtype=float)
        if result_fun.size:
            weighted_residual_rms = float(np.sqrt(np.mean(result_fun * result_fun)))
    result.weighted_residual_rms_px = float(weighted_residual_rms)
    result.rms_px = float(weighted_residual_rms)

    if getattr(result, "x", None) is not None:
        result.x = np.minimum(np.maximum(result.x, lower_bounds), upper_bounds)

    if point_match_mode and getattr(result, "x", None) is not None:
        try:
            final_local = _apply_trial_params(np.asarray(result.x, dtype=float))
            _, point_match_diagnostics, point_match_summary = _evaluate_pixel_matches(
                final_local,
                collect_diagnostics=True,
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

    return result


def run_optimization_positions_geometry_local(
    miller, intensities, image_size,
    initial_params, bounds, measured_peaks
):
    """
    Global optimization (Differential Evolution) over geometry + beam center.
    bounds is list of (min,max) for [gamma, Gamma, dist, theta_i, cor_angle,
    zs, zb, chi, a, c, center_x, center_y].
    """
    def obj_glob(x):
        gamma, Gamma, dist, theta_initial, cor_angle, zs, zb, chi, a, c, center_x, center_y = x
        return np.sum(compute_peak_position_error_geometry_local(
            gamma, Gamma, dist, theta_initial, cor_angle, zs, zb, chi, a, c,
            center_x, center_y,
            measured_peaks,
            miller, intensities, image_size,
            initial_params['mosaic_params'],
            initial_params['n2'],
            initial_params.get('psi', 0.0),
            initial_params.get('psi_z', 0.0),
            initial_params['debye_x'],
            initial_params['debye_y'],
            initial_params['lambda'],
            pixel_tol=np.inf
        ))
    res = differential_evolution(obj_glob, bounds, maxiter=200, popsize=15)
    return res
