"""Importable peak-fit worker helpers for all-background diagnostics.

The notebook keeps plotting/orchestration code local, but process pools on
Windows need the CPU-bound worker function to live in an importable module.
Large caked arrays are passed by ``.npy`` memmap paths instead of pickling one
copy per peak.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.optimize import least_squares

try:
    from numba import njit, set_num_threads

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency fallback
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[override]
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    def set_num_threads(_n: int) -> None:  # type: ignore[override]
        return None


DEFAULT_PEAK_FIT_SETTINGS: dict[str, object] = {
    "THETA_HALF_WINDOW_DEG": 1.8,
    "PHI_HALF_WINDOW_DEG": 6.0,
    "CENTER_THETA_BOUND_DEG": 0.55,
    "CENTER_PHI_BOUND_DEG": 1.6,
    "GAUSSIAN_TAIL_DISTANCE_WEIGHT": 1.25,
    "GAUSSIAN_CORE_SIGNAL_DOWNSCALE": 0.06,
    "GAUSSIAN_TAIL_OVERPREDICTION_START": 0.55,
    "GAUSSIAN_TAIL_OVERPREDICTION_WEIGHT": 1.75,
    "PEAK_FIT_BACKGROUND_QUANTILE": 45.0,
    "PEAK_FIT_BACKGROUND_EXCLUSION_RADIUS": 1.15,
    "PEAK_FIT_BACKGROUND_IRLS_STEPS": 6,
    "PEAK_FIT_BACKGROUND_HUBER_K": 1.5,
    "PEAK_FIT_MULTISTART_WIDTH_FACTORS": (1.0, 1.8, 2.8),
    "PEAK_FIT_MAX_NFEV": 1400,
}

_PEAK_FIT_SETTINGS = dict(DEFAULT_PEAK_FIT_SETTINGS)
_ARRAY_CACHE: dict[str, np.ndarray] = {}


def clear_peak_fit_worker_array_cache() -> None:
    """Close cached memmaps held by the current process."""

    for array in list(_ARRAY_CACHE.values()):
        mmap_obj = getattr(array, "_mmap", None)
        if mmap_obj is not None:
            try:
                mmap_obj.close()
            except Exception:
                pass
    _ARRAY_CACHE.clear()


def configure_peak_fit_worker(
    settings: dict[str, object] | None = None,
    numba_threads: int | None = 1,
    *,
    limit_native_threads: bool = True,
) -> None:
    """Configure peak-fit settings and optional process-worker thread caps."""

    if limit_native_threads:
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[name] = "1"
    if settings:
        _PEAK_FIT_SETTINGS.update(settings)
    if numba_threads is not None:
        try:
            set_num_threads(max(int(numba_threads), 1))
        except Exception:
            pass


def peak_fit_settings_from_values(**values: object) -> dict[str, object]:
    """Return settings payload filtered to known peak-fit keys."""

    settings = dict(DEFAULT_PEAK_FIT_SETTINGS)
    for key in settings:
        if key in values:
            settings[key] = values[key]
    return settings


def save_peak_fit_background_arrays(
    directory: str | os.PathLike[str],
    *,
    background_index: int,
    caked_image: np.ndarray,
    theta_axis: np.ndarray,
    phi_axis: np.ndarray,
) -> dict[str, str]:
    """Save one background's fit arrays as memmap-loadable ``.npy`` files."""

    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    prefix = base / f"background_{int(background_index):02d}"
    caked_path = f"{prefix}_caked_image.npy"
    theta_path = f"{prefix}_theta_axis.npy"
    phi_path = f"{prefix}_phi_axis.npy"
    np.save(caked_path, np.ascontiguousarray(caked_image, dtype=np.float64))
    np.save(theta_path, np.ascontiguousarray(theta_axis, dtype=np.float64))
    np.save(phi_path, np.ascontiguousarray(phi_axis, dtype=np.float64))
    return {
        "caked_image_path": str(caked_path),
        "theta_axis_path": str(theta_path),
        "phi_axis_path": str(phi_path),
    }


def make_peak_fit_job(
    *,
    background_index: int,
    local_index: int,
    entry: dict[str, object],
    caked_image_path: str,
    theta_axis_path: str,
    phi_axis_path: str,
) -> dict[str, object]:
    """Build one pickle-small process-pool job payload."""

    return {
        "background_index": int(background_index),
        "local_index": int(local_index),
        "entry": dict(entry),
        "caked_image_path": str(caked_image_path),
        "theta_axis_path": str(theta_axis_path),
        "phi_axis_path": str(phi_axis_path),
    }


def _setting_float(name: str) -> float:
    return float(_PEAK_FIT_SETTINGS[name])


def _setting_int(name: str) -> int:
    return int(_PEAK_FIT_SETTINGS[name])


def _setting_tuple(name: str) -> tuple[float, ...]:
    raw = _PEAK_FIT_SETTINGS[name]
    return tuple(float(value) for value in raw)  # type: ignore[union-attr]


def _load_array(path: object) -> np.ndarray:
    key = str(path)
    cached = _ARRAY_CACHE.get(key)
    if cached is None:
        cached = np.load(key, mmap_mode="r")
        _ARRAY_CACHE[key] = cached
    return cached


def as_float(value: object, fallback: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fallback)
    return out if np.isfinite(out) else float(fallback)


@njit(fastmath=True, nogil=True)
def _wrapped_delta_deg_scalar_numba(value: float, center: float) -> float:
    return ((value - center + 180.0) % 360.0) - 180.0


def wrapped_delta_deg(values: object, center: float) -> np.ndarray:
    return ((np.asarray(values, dtype=np.float64) - float(center) + 180.0) % 360.0) - 180.0


GAUSSIAN_FWHM_TO_SIGMA = 2.3548200450309493
FIT_MODEL_NAME = "rotated_gaussian_core_lorentzian_tail_shared_center"


@njit(fastmath=True, nogil=True)
def _rotated_gaussian_value_numba(
    params: np.ndarray,
    theta: float,
    phi: float,
    peak_only: bool,
) -> float:
    amp = params[0]
    theta0 = params[1]
    phi0 = params[2]
    sigma_g_u = max(params[3] / GAUSSIAN_FWHM_TO_SIGMA, 1.0e-12)
    sigma_g_v = max(params[4] / GAUSSIAN_FWHM_TO_SIGMA, 1.0e-12)
    angle = params[5]
    dt = theta - theta0
    dp = _wrapped_delta_deg_scalar_numba(phi, phi0)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    u = cos_a * dt + sin_a * dp
    v = -sin_a * dt + cos_a * dp
    gaussian_r2 = (u / sigma_g_u) * (u / sigma_g_u) + (v / sigma_g_v) * (v / sigma_g_v)
    gaussian_core = math.exp(-0.5 * gaussian_r2)
    gamma_l_u = max(0.5 * params[9], 1.0e-12)
    gamma_l_v = max(0.5 * params[10], 1.0e-12)
    lorentzian_tail = 1.0 / (
        1.0 + (u / gamma_l_u) * (u / gamma_l_u) + (v / gamma_l_v) * (v / gamma_l_v)
    )
    eta_tail = params[11]
    if eta_tail < 0.0:
        eta_tail = 0.0
    elif eta_tail > 0.95:
        eta_tail = 0.95
    peak = amp * ((1.0 - eta_tail) * gaussian_core + eta_tail * lorentzian_tail)
    if peak_only:
        return peak
    return params[6] + params[7] * dt + params[8] * dp + peak


@njit(fastmath=True, nogil=True)
def _rotated_gaussian_plane_points_numba(
    params: np.ndarray,
    theta_values: np.ndarray,
    phi_values: np.ndarray,
) -> np.ndarray:
    n = theta_values.size
    out = np.empty(n, dtype=np.float64)
    for idx in range(n):
        out[idx] = _rotated_gaussian_value_numba(params, theta_values[idx], phi_values[idx], False)
    return out


@njit(fastmath=True, nogil=True)
def _rotated_gaussian_residual_points_numba(
    params: np.ndarray,
    theta_values: np.ndarray,
    phi_values: np.ndarray,
    y_values: np.ndarray,
    tail_weight: np.ndarray,
    denominator: np.ndarray,
    tail_overprediction_weight: np.ndarray,
) -> np.ndarray:
    n = theta_values.size
    out = np.empty(n, dtype=np.float64)
    for idx in range(n):
        model = _rotated_gaussian_value_numba(params, theta_values[idx], phi_values[idx], False)
        residual = model - y_values[idx]
        if residual > 0.0:
            residual *= tail_overprediction_weight[idx]
        out[idx] = tail_weight[idx] * residual / denominator[idx]
    return out


@njit(fastmath=True, nogil=True)
def _rotated_gaussian_grid_numba(
    params: np.ndarray,
    theta_axis: np.ndarray,
    phi_axis: np.ndarray,
    peak_only: bool,
) -> np.ndarray:
    n_phi = phi_axis.size
    n_theta = theta_axis.size
    out = np.empty((n_phi, n_theta), dtype=np.float64)
    for row in range(n_phi):
        phi = phi_axis[row]
        for col in range(n_theta):
            out[row, col] = _rotated_gaussian_value_numba(
                params,
                theta_axis[col],
                phi,
                peak_only,
            )
    return out


def robust_peak_background_plane(
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    roi: np.ndarray,
    finite: np.ndarray,
    *,
    theta_ref: float,
    phi_ref: float,
) -> np.ndarray:
    """Robust local 2D baseline plane from pixels outside the peak core."""

    if np.count_nonzero(finite) < 3:
        fill = float(np.nanmedian(roi[finite])) if np.any(finite) else 0.0
        return np.asarray([fill, 0.0, 0.0], dtype=np.float64)

    theta_delta_grid = np.asarray(theta_grid, dtype=np.float64) - float(theta_ref)
    phi_delta_grid = wrapped_delta_deg(phi_grid, float(phi_ref))
    center_theta_bound = _setting_float("CENTER_THETA_BOUND_DEG")
    center_phi_bound = _setting_float("CENTER_PHI_BOUND_DEG")
    core_radius = np.sqrt(
        (theta_delta_grid / max(center_theta_bound, 1.0e-9)) ** 2
        + (phi_delta_grid / max(center_phi_bound, 1.0e-9)) ** 2
    )
    off_peak = finite & (core_radius >= _setting_float("PEAK_FIT_BACKGROUND_EXCLUSION_RADIUS"))
    minimum = max(12, int(0.20 * np.count_nonzero(finite)))
    if np.count_nonzero(off_peak) < minimum:
        cutoff = float(
            np.nanpercentile(roi[finite], _setting_float("PEAK_FIT_BACKGROUND_QUANTILE"))
        )
        off_peak = finite & (roi <= cutoff)
    if np.count_nonzero(off_peak) < 3:
        off_peak = finite

    theta_fit = np.asarray(theta_delta_grid[off_peak], dtype=np.float64)
    phi_fit = np.asarray(phi_delta_grid[off_peak], dtype=np.float64)
    y_fit = np.asarray(roi[off_peak], dtype=np.float64)
    design = np.column_stack([np.ones_like(theta_fit), theta_fit, phi_fit])

    try:
        coef, *_ = np.linalg.lstsq(design, y_fit, rcond=None)
    except Exception:
        coef = np.asarray([float(np.nanmedian(y_fit)), 0.0, 0.0], dtype=np.float64)

    coef = np.asarray(coef, dtype=np.float64)
    for _ in range(_setting_int("PEAK_FIT_BACKGROUND_IRLS_STEPS")):
        residual = y_fit - design @ coef[:3]
        sigma = 1.4826 * float(np.nanmedian(np.abs(residual - np.nanmedian(residual))))
        if not np.isfinite(sigma) or sigma <= 0.0:
            break
        threshold = _setting_float("PEAK_FIT_BACKGROUND_HUBER_K") * sigma
        weights = np.ones_like(residual, dtype=np.float64)
        large = np.abs(residual) > threshold
        weights[large] = threshold / np.maximum(np.abs(residual[large]), 1.0e-12)
        root_w = np.sqrt(weights)
        try:
            coef, *_ = np.linalg.lstsq(design * root_w[:, None], y_fit * root_w, rcond=None)
        except Exception:
            break
        coef = np.asarray(coef, dtype=np.float64)

    if coef.size < 3 or not np.all(np.isfinite(coef[:3])):
        coef = np.asarray([float(np.nanmedian(y_fit)), 0.0, 0.0], dtype=np.float64)
    return np.asarray([float(coef[0]), float(coef[1]), float(coef[2])], dtype=np.float64)


def local_plane_from_coefficients(
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    *,
    theta_ref: float,
    phi_ref: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    coeff = np.asarray(coefficients, dtype=np.float64)
    intercept = float(coeff[0]) if coeff.size > 0 and np.isfinite(coeff[0]) else 0.0
    theta_slope = float(coeff[1]) if coeff.size > 1 and np.isfinite(coeff[1]) else 0.0
    phi_slope = float(coeff[2]) if coeff.size > 2 and np.isfinite(coeff[2]) else 0.0
    return (
        intercept
        + theta_slope * (np.asarray(theta_grid, dtype=np.float64) - float(theta_ref))
        + phi_slope * wrapped_delta_deg(phi_grid, float(phi_ref))
    )


def deduplicate_peak_starts(
    candidates: list[tuple[float, float]],
    *,
    theta_step: float,
    phi_step: float,
) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    theta_tol = max(abs(theta_step), 1.0e-6)
    phi_tol = max(abs(phi_step), 1.0e-6)
    for theta0, phi0 in candidates:
        if not np.isfinite(theta0) or not np.isfinite(phi0):
            continue
        duplicate = False
        for theta_prev, phi_prev in kept:
            phi_distance = abs(float(wrapped_delta_deg(float(phi0), float(phi_prev))))
            if abs(float(theta0) - float(theta_prev)) <= theta_tol and phi_distance <= phi_tol:
                duplicate = True
                break
        if not duplicate:
            kept.append((float(theta0), float(phi0)))
    return kept


def _fit_boundary_warnings(
    params: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    optimizer_success: bool,
) -> list[str]:
    warnings: list[str] = []
    if params.size < 12:
        warnings.append("unexpected_parameter_count")
        return warnings
    if not optimizer_success:
        warnings.append("optimizer_not_converged")
    if float(params[11]) >= float(upper[11]) - 0.02:
        warnings.append("eta_tail_near_upper_bound")
    for index, name in ((9, "lorentzian_fwhm_u"), (10, "lorentzian_fwhm_v")):
        span = max(float(upper[index]) - float(lower[index]), 1.0e-12)
        if float(params[index]) >= float(upper[index]) - 0.02 * span:
            warnings.append(f"{name}_near_upper_bound")
    for index, name in ((3, "gaussian_fwhm_u"), (4, "gaussian_fwhm_v")):
        span = max(float(upper[index]) - float(lower[index]), 1.0e-12)
        if float(params[index]) <= float(lower[index]) + 0.02 * span:
            warnings.append(f"{name}_near_lower_bound")
    theta_span = max(float(upper[1]) - float(lower[1]), 1.0e-12)
    phi_span = max(float(upper[2]) - float(lower[2]), 1.0e-12)
    if (
        float(params[1]) <= float(lower[1]) + 0.02 * theta_span
        or float(params[1]) >= float(upper[1]) - 0.02 * theta_span
        or float(params[2]) <= float(lower[2]) + 0.02 * phi_span
        or float(params[2]) >= float(upper[2]) - 0.02 * phi_span
    ):
        warnings.append("center_near_allowed_bound_edge")
    return warnings


def _fit_boundary_penalty(params: np.ndarray, lower: np.ndarray, upper: np.ndarray, *, optimizer_success: bool) -> float:
    warnings = _fit_boundary_warnings(params, lower, upper, optimizer_success=optimizer_success)
    if not warnings:
        return 0.0
    return 0.015 * float(len(warnings))


def fit_one_peak(
    entry: dict[str, object],
    caked_image: np.ndarray,
    theta_axis: np.ndarray,
    phi_axis: np.ndarray,
) -> dict[str, object]:
    """Fit one caked peak ROI and return the notebook-compatible payload."""

    theta_half_window = _setting_float("THETA_HALF_WINDOW_DEG")
    phi_half_window = _setting_float("PHI_HALF_WINDOW_DEG")
    center_theta_bound = _setting_float("CENTER_THETA_BOUND_DEG")
    center_phi_bound = _setting_float("CENTER_PHI_BOUND_DEG")

    theta_seed = as_float(
        entry.get("_theta_seed_deg"), as_float(entry.get("background_two_theta_deg"))
    )
    phi_seed = as_float(entry.get("_phi_seed_deg"), as_float(entry.get("background_phi_deg")))
    if not np.isfinite(theta_seed) or not np.isfinite(phi_seed):
        raise ValueError("missing peak seed angles")

    theta_mask = np.abs(theta_axis - theta_seed) <= theta_half_window
    phi_mask = np.abs(wrapped_delta_deg(phi_axis, phi_seed)) <= phi_half_window
    theta_idx = np.flatnonzero(theta_mask)
    phi_idx = np.flatnonzero(phi_mask)
    if theta_idx.size < 6 or phi_idx.size < 6:
        raise ValueError("ROI too small")

    roi = np.ascontiguousarray(caked_image[np.ix_(phi_idx, theta_idx)], dtype=np.float64)
    theta_vals = np.ascontiguousarray(theta_axis[theta_idx], dtype=np.float64)
    phi_vals = np.ascontiguousarray(phi_axis[phi_idx], dtype=np.float64)
    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
    finite = np.isfinite(roi)
    if np.count_nonzero(finite) < 20:
        raise ValueError("too few finite ROI pixels")

    theta_step = float(np.nanmedian(np.diff(theta_axis))) if theta_axis.size > 1 else 0.05
    phi_step = float(np.nanmedian(np.diff(phi_axis))) if phi_axis.size > 1 else 0.5
    background_coeff = robust_peak_background_plane(
        theta_grid,
        phi_grid,
        roi,
        finite,
        theta_ref=float(theta_seed),
        phi_ref=float(phi_seed),
    )
    background_plane_seed = local_plane_from_coefficients(
        theta_grid,
        phi_grid,
        theta_ref=float(theta_seed),
        phi_ref=float(phi_seed),
        coefficients=background_coeff,
    )
    net_roi = roi - background_plane_seed

    center_candidate = (
        finite
        & (np.abs(theta_grid - theta_seed) <= center_theta_bound)
        & (np.abs(wrapped_delta_deg(phi_grid, phi_seed)) <= center_phi_bound)
    )
    candidate_centers: list[tuple[float, float]] = []
    if np.any(center_candidate):
        peak_flat_index = int(np.nanargmax(np.where(center_candidate, net_roi, -np.inf)))
        peak_row, peak_col = np.unravel_index(peak_flat_index, roi.shape)
        candidate_centers.append((float(theta_vals[peak_col]), float(phi_vals[peak_row])))

        positive_core = (
            center_candidate
            & np.isfinite(net_roi)
            & (net_roi > np.nanpercentile(net_roi[center_candidate], 60.0))
        )
        if np.count_nonzero(positive_core) >= 4:
            weights = np.clip(net_roi[positive_core], 0.0, None)
            if np.sum(weights) > 0.0:
                theta_centroid = float(np.average(theta_grid[positive_core], weights=weights))
                phi_centroid = float(
                    phi_seed
                    + np.average(
                        wrapped_delta_deg(phi_grid[positive_core], phi_seed), weights=weights
                    )
                )
                candidate_centers.append((theta_centroid, phi_centroid))
    else:
        seed_distance = np.abs(theta_grid - theta_seed) + np.abs(
            wrapped_delta_deg(phi_grid, phi_seed)
        )
        nearest_flat_index = int(np.nanargmin(np.where(finite, seed_distance, np.inf)))
        peak_row, peak_col = np.unravel_index(nearest_flat_index, roi.shape)
        candidate_centers.append((float(theta_vals[peak_col]), float(phi_vals[peak_row])))
    candidate_centers.append((float(theta_seed), float(phi_seed)))
    candidate_centers = deduplicate_peak_starts(
        candidate_centers,
        theta_step=theta_step,
        phi_step=phi_step,
    )

    y = np.ascontiguousarray(roi[finite], dtype=np.float64)
    theta_fit = np.ascontiguousarray(theta_grid[finite], dtype=np.float64)
    phi_fit = np.ascontiguousarray(phi_grid[finite], dtype=np.float64)
    robust_scale = max(float(np.nanpercentile(np.abs(y - np.nanmedian(y)), 75.0)), 1.0)
    intensity_span = max(float(np.nanpercentile(y, 99.5) - np.nanpercentile(y, 5.0)), 1.0)
    theta_step_abs = max(abs(theta_step), 1.0e-12)
    phi_step_abs = max(abs(phi_step), 1.0e-12)
    fwhm_g_u_min = max(1.2 * theta_step_abs, 1.0e-6)
    fwhm_g_v_min = max(1.2 * phi_step_abs, 1.0e-6)
    fwhm_l_u_min = fwhm_g_u_min
    fwhm_l_v_min = fwhm_g_v_min
    fwhm_g_u_max = max(2.0 * theta_half_window, 4.0 * theta_step_abs)
    fwhm_g_v_max = max(2.0 * phi_half_window, 4.0 * phi_step_abs)
    fwhm_l_u_max = max(4.0 * theta_half_window, fwhm_g_u_max)
    fwhm_l_v_max = max(4.0 * phi_half_window, fwhm_g_v_max)
    theta_slope_bound = max(8.0 * intensity_span / max(theta_half_window, 1.0e-6), 1.0e-6)
    phi_slope_bound = max(8.0 * intensity_span / max(phi_half_window, 1.0e-6), 1.0e-6)

    lower = np.array(
        [
            0.0,
            theta_seed - center_theta_bound,
            phi_seed - center_phi_bound,
            fwhm_g_u_min,
            fwhm_g_v_min,
            -math.pi / 2.0,
            -np.inf,
            -theta_slope_bound,
            -phi_slope_bound,
            fwhm_l_u_min,
            fwhm_l_v_min,
            0.0,
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            max(intensity_span * 20.0, 1.0),
            theta_seed + center_theta_bound,
            phi_seed + center_phi_bound,
            fwhm_g_u_max,
            fwhm_g_v_max,
            math.pi / 2.0,
            np.inf,
            theta_slope_bound,
            phi_slope_bound,
            fwhm_l_u_max,
            fwhm_l_v_max,
            0.95,
        ],
        dtype=np.float64,
    )

    fit_attempts: list[tuple[float, Any, np.ndarray]] = []
    for theta0, phi0 in candidate_centers:
        baseline_at_center = float(
            background_coeff[0]
            + background_coeff[1] * (float(theta0) - float(theta_seed))
            + background_coeff[2] * float(wrapped_delta_deg(float(phi0), float(phi_seed)))
        )
        peak_distance = np.abs(theta_grid - float(theta0)) + np.abs(
            wrapped_delta_deg(phi_grid, float(phi0))
        )
        nearest_flat_index = int(np.nanargmin(np.where(finite, peak_distance, np.inf)))
        peak_row, peak_col = np.unravel_index(nearest_flat_index, roi.shape)
        local_net = float(roi[peak_row, peak_col] - baseline_at_center)
        if np.any(center_candidate):
            local_net = max(
                local_net, float(np.nanmax(np.where(center_candidate, net_roi, -np.inf)))
            )
        amp0 = max(local_net, 1.0)

        tail_distance = np.sqrt(
            ((theta_fit - float(theta0)) / max(center_theta_bound, 1.0e-6)) ** 2
            + (wrapped_delta_deg(phi_fit, float(phi0)) / max(center_phi_bound, 1.0e-6)) ** 2
        )
        tail_weight = np.ascontiguousarray(
            1.0
            + _setting_float("GAUSSIAN_TAIL_DISTANCE_WEIGHT") * np.clip(tail_distance, 0.0, 2.0),
            dtype=np.float64,
        )
        tail_excess = np.clip(
            (tail_distance - _setting_float("GAUSSIAN_TAIL_OVERPREDICTION_START"))
            / max(2.0 - _setting_float("GAUSSIAN_TAIL_OVERPREDICTION_START"), 1.0e-6),
            0.0,
            1.0,
        )
        tail_overprediction_weight = np.ascontiguousarray(
            1.0 + _setting_float("GAUSSIAN_TAIL_OVERPREDICTION_WEIGHT") * tail_excess,
            dtype=np.float64,
        )
        signal = np.clip(y - baseline_at_center, 0.0, None)
        tail_scale = np.ascontiguousarray(
            robust_scale + _setting_float("GAUSSIAN_CORE_SIGNAL_DOWNSCALE") * signal,
            dtype=np.float64,
        )

        fwhm_g_u0 = max(3.0 * theta_step_abs, 0.08)
        fwhm_g_v0 = max(3.0 * phi_step_abs, 0.35)
        angle_seeds = (0.0, 0.25, -0.25)
        eta_tail_seeds = (0.0, 0.15, 0.35, 0.60)
        lorentzian_width_multipliers = (1.5, 2.5, 4.0, 7.0)

        def residual(params: np.ndarray) -> np.ndarray:
            return _rotated_gaussian_residual_points_numba(
                np.asarray(params, dtype=np.float64),
                theta_fit,
                phi_fit,
                y,
                tail_weight,
                tail_scale,
                tail_overprediction_weight,
            )

        for eta_tail0 in eta_tail_seeds:
            for lorentzian_width_multiplier in lorentzian_width_multipliers:
                for angle0 in angle_seeds:
                    x0 = np.array(
                        [
                            amp0,
                            float(theta0),
                            float(phi0),
                            fwhm_g_u0,
                            fwhm_g_v0,
                            angle0,
                            baseline_at_center,
                            float(background_coeff[1]),
                            float(background_coeff[2]),
                            max(float(lorentzian_width_multiplier) * fwhm_g_u0, 0.20),
                            max(float(lorentzian_width_multiplier) * fwhm_g_v0, 0.80),
                            eta_tail0,
                        ],
                        dtype=np.float64,
                    )
                    dynamic_upper = upper.copy()
                    dynamic_upper[0] = max(dynamic_upper[0], amp0 * 80.0, amp0 + 1.0)
                    x0 = np.minimum(np.maximum(x0, lower + 1.0e-9), dynamic_upper - 1.0e-9)

                    try:
                        result = least_squares(
                            residual,
                            x0,
                            bounds=(lower, dynamic_upper),
                            loss="soft_l1",
                            f_scale=2.0,
                            max_nfev=_setting_int("PEAK_FIT_MAX_NFEV"),
                            x_scale=[
                                max(amp0, 1.0),
                                0.2,
                                1.0,
                                0.2,
                                1.0,
                                1.0,
                                max(abs(baseline_at_center), 1.0),
                                max(theta_slope_bound * 0.25, 1.0),
                                max(phi_slope_bound * 0.25, 1.0),
                                0.5,
                                2.0,
                                0.25,
                            ],
                        )
                    except Exception:
                        continue

                    params_candidate = np.asarray(result.x, dtype=np.float64)
                    weighted_score = float(np.nanmean(residual(params_candidate) ** 2))
                    model_candidate = _rotated_gaussian_plane_points_numba(
                        params_candidate,
                        theta_fit,
                        phi_fit,
                    )
                    raw_residual = model_candidate - y
                    raw_score = float(
                        np.nanmean((raw_residual / max(robust_scale, 1.0e-12)) ** 2)
                    )
                    score = (
                        0.65 * weighted_score
                        + 0.35 * raw_score
                        + _fit_boundary_penalty(
                            params_candidate,
                            lower,
                            dynamic_upper,
                            optimizer_success=bool(result.success),
                        )
                    )
                    if np.isfinite(score):
                        fit_attempts.append((score, result, params_candidate))

    if not fit_attempts:
        raise RuntimeError("all peak fit starts failed")
    fit_attempts.sort(key=lambda item: item[0])
    best_score, result, params = fit_attempts[0]
    fit_parameter_warnings = _fit_boundary_warnings(
        params,
        lower,
        upper,
        optimizer_success=bool(result.success),
    )

    full_fit = _rotated_gaussian_grid_numba(params, theta_vals, phi_vals, False)
    peak_fit = _rotated_gaussian_grid_numba(params, theta_vals, phi_vals, True)
    peak_subtracted_image = roi - peak_fit
    optimization_residual_image = roi - full_fit
    finite_resid = optimization_residual_image[finite]
    return {
        "entry": dict(entry),
        "background_index": int(entry["_background_index"]),
        "background_name": str(entry["_background_name"]),
        "label": str(entry["_label"]),
        "branch": str(entry["_branch"]),
        "theta_seed_deg": float(theta_seed),
        "phi_seed_deg": float(phi_seed),
        "params": params,
        "theta_idx": theta_idx,
        "phi_idx": phi_idx,
        "theta_vals": theta_vals,
        "phi_vals": phi_vals,
        "roi": roi,
        "fit": peak_fit,
        "fit_with_nuisance_background": full_fit,
        "peak_fit": peak_fit,
        "peak_subtracted_roi": peak_subtracted_image,
        "residual": peak_subtracted_image,
        "optimization_residual": optimization_residual_image,
        "success": True,
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "rmse": float(np.sqrt(np.nanmean(finite_resid**2))),
        "fit_detector_col": None,
        "fit_detector_row": None,
        "fit_model": FIT_MODEL_NAME,
        "fit_score": float(best_score),
        "fit_start_count": int(len(fit_attempts)),
        "fit_parameter_warnings": fit_parameter_warnings,
        "fit_has_parameter_warning": bool(fit_parameter_warnings),
        "background_plane_coefficients": np.asarray(background_coeff, dtype=np.float64),
        "baseline_equation": "density = baseline + theta_slope*(two_theta_deg - fit_two_theta_deg) + phi_slope*wrapped_delta_deg(phi_deg, fit_phi_deg)",
    }


def fit_peak_from_job(job: dict[str, object]) -> dict[str, object]:
    """Process-pool entrypoint for one peak-fit job."""

    start = time.perf_counter()
    item = fit_one_peak(
        dict(job["entry"]),  # type: ignore[arg-type]
        _load_array(job["caked_image_path"]),
        _load_array(job["theta_axis_path"]),
        _load_array(job["phi_axis_path"]),
    )
    return {
        "background_index": int(job["background_index"]),
        "local_index": int(job["local_index"]),
        "item": item,
        "elapsed_s": float(time.perf_counter() - start),
        "pid": int(os.getpid()),
    }
