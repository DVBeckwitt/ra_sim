"""Helpers for caked-view peak picking and 1D profile fitting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares


PROFILE_GAUSSIAN = "gaussian"
PROFILE_LORENTZIAN = "lorentzian"
PROFILE_PSEUDO_VOIGT = "pseudo_voigt"
SUPPORTED_PROFILE_MODELS = (
    PROFILE_GAUSSIAN,
    PROFILE_LORENTZIAN,
    PROFILE_PSEUDO_VOIGT,
)


def wrap_angle_degrees(value: float) -> float:
    """Wrap one angle into ``[-180, 180)`` degrees."""

    return float(((float(value) + 180.0) % 360.0) - 180.0)


def align_angle_to_axis(
    angle_deg: float,
    axis_values: Sequence[float] | None,
) -> float:
    """Return the wrapped representation of one angle nearest the axis domain."""

    axis_arr = np.asarray(axis_values, dtype=float)
    finite_axis = axis_arr[np.isfinite(axis_arr)]
    if finite_axis.size <= 0:
        return wrap_angle_degrees(angle_deg)

    finite_axis = np.sort(finite_axis)
    if finite_axis.size >= 2:
        step = float(np.nanmedian(np.abs(np.diff(finite_axis))))
    else:
        step = 1.0
    pad = max(step, 1.0)
    lower = float(finite_axis[0]) - pad
    upper = float(finite_axis[-1]) + pad
    target = float(np.nanmedian(finite_axis))
    raw_angle = float(angle_deg)
    candidates = (raw_angle - 360.0, raw_angle, raw_angle + 360.0)
    in_range = [value for value in candidates if lower <= float(value) <= upper]
    source = in_range if in_range else list(candidates)
    return float(min(source, key=lambda value: abs(float(value) - target)))


def integration_region_contains(
    two_theta_deg: float,
    phi_deg: float,
    *,
    tth_min: float,
    tth_max: float,
    phi_min: float,
    phi_max: float,
) -> bool:
    """Return whether one caked-space point falls inside the active 1D region."""

    tth_lo, tth_hi = sorted((float(tth_min), float(tth_max)))
    tth_value = float(two_theta_deg)
    if not (tth_lo <= tth_value <= tth_hi):
        return False

    phi_value = wrap_angle_degrees(phi_deg)
    phi_start = float(phi_min)
    phi_end = float(phi_max)
    if phi_end >= phi_start:
        return bool(phi_start <= phi_value <= phi_end)
    return bool(phi_value >= phi_start or phi_value <= phi_end)


def sample_curve_value(
    axis_values: Sequence[float] | None,
    curve_values: Sequence[float] | None,
    position: float,
) -> float:
    """Interpolate one y value from a finite monotonic 1D curve."""

    axis_arr, curve_arr = _prepare_curve_samples(axis_values, curve_values)
    if axis_arr.size <= 0:
        return float("nan")
    position_value = float(position)
    if position_value < float(axis_arr[0]) or position_value > float(axis_arr[-1]):
        return float("nan")
    return float(np.interp(position_value, axis_arr, curve_arr))


def match_selected_peak_index(
    peaks: Sequence[dict[str, object]] | None,
    *,
    two_theta_deg: float,
    phi_deg: float,
    radial_tolerance_deg: float,
    azimuth_tolerance_deg: float,
) -> int | None:
    """Return the index of one existing selected peak within the given tolerances."""

    target_tth = float(two_theta_deg)
    target_phi = wrap_angle_degrees(phi_deg)
    radial_tol = max(float(radial_tolerance_deg), 1.0e-6)
    azimuth_tol = max(float(azimuth_tolerance_deg), 1.0e-6)

    for idx, entry in enumerate(peaks or ()):
        if not isinstance(entry, dict):
            continue
        try:
            peak_tth = float(entry.get("two_theta_deg"))
            peak_phi = wrap_angle_degrees(float(entry.get("phi_deg")))
        except Exception:
            continue
        if abs(peak_tth - target_tth) <= radial_tol and abs(peak_phi - target_phi) <= azimuth_tol:
            return int(idx)
    return None


def profile_model_label(model: str) -> str:
    """Return the user-facing label for one supported profile family."""

    normalized = str(model or "").strip().lower()
    if normalized == PROFILE_GAUSSIAN:
        return "Gaussian"
    if normalized == PROFILE_LORENTZIAN:
        return "Lorentzian"
    if normalized == PROFILE_PSEUDO_VOIGT:
        return "Pseudo-Voigt"
    return str(model)


def format_peak_fit_axis_summary(
    axis_label: str,
    entries: Sequence[Mapping[str, object]] | None,
) -> str:
    """Return one compact fit-summary line for a radial or azimuth axis."""

    normalized_entries = [entry for entry in (entries or ()) if isinstance(entry, Mapping)]
    if not normalized_entries:
        return ""

    successful_entries = [
        entry for entry in normalized_entries if bool(entry.get("success", False))
    ]
    prefix = f"{str(axis_label)}: {len(successful_entries)}/{len(normalized_entries)} fits"
    if successful_entries:
        best_entry = min(
            successful_entries,
            key=lambda entry: _fit_entry_rmse_sort_key(entry),
        )
        best_text = _format_peak_fit_entry_detail(best_entry)
        if best_text:
            return f"{prefix}; best {best_text}"
        return prefix

    failure_text = _format_peak_fit_entry_failure(normalized_entries[-1])
    if failure_text:
        return f"{prefix}; last failure {failure_text}"
    return prefix


def recommended_peak_window_half_width(
    peak_centers: Sequence[float] | None,
    peak_index: int,
    *,
    axis_values: Sequence[float] | None,
    axis_kind: str,
    region_bounds: tuple[float, float] | None = None,
) -> float:
    """Return a stable local fit window half-width for one selected peak."""

    axis_arr = np.asarray(axis_values, dtype=float)
    finite_axis = axis_arr[np.isfinite(axis_arr)]
    if finite_axis.size >= 2:
        step = float(np.nanmedian(np.abs(np.diff(np.sort(finite_axis)))))
    else:
        step = 0.0

    normalized_kind = str(axis_kind or "").strip().lower()
    min_half_width = max(step * 6.0, 0.08 if normalized_kind == "radial" else 0.5)
    max_half_width = 2.5 if normalized_kind == "radial" else 18.0

    if region_bounds is not None and len(region_bounds) >= 2:
        region_span = abs(float(region_bounds[1]) - float(region_bounds[0]))
    elif finite_axis.size >= 2:
        region_span = abs(float(finite_axis[-1]) - float(finite_axis[0]))
    else:
        region_span = 0.0
    default_half_width = max(min_half_width, region_span / 14.0 if region_span > 0.0 else min_half_width)
    half_width = min(default_half_width, max_half_width)

    centers = np.asarray(peak_centers, dtype=float)
    finite_centers = centers[np.isfinite(centers)]
    if finite_centers.size <= 1 or peak_index < 0 or peak_index >= centers.size:
        return float(max(min_half_width, min(half_width, max_half_width)))

    target = float(centers[int(peak_index)])
    finite_centers = np.sort(finite_centers)
    left_neighbors = finite_centers[finite_centers < target]
    right_neighbors = finite_centers[finite_centers > target]
    neighbor_spacings = []
    if left_neighbors.size:
        neighbor_spacings.append(abs(target - float(left_neighbors[-1])))
    if right_neighbors.size:
        neighbor_spacings.append(abs(float(right_neighbors[0]) - target))
    if neighbor_spacings:
        half_width = min(float(half_width), 0.45 * min(neighbor_spacings))
    return float(max(min_half_width, min(float(half_width), max_half_width)))


def gaussian_profile(
    x_values: Sequence[float] | np.ndarray,
    baseline: float,
    amplitude: float,
    center: float,
    fwhm: float,
) -> np.ndarray:
    """Return one Gaussian peak parameterized directly by FWHM."""

    x_arr = np.asarray(x_values, dtype=float)
    fwhm_value = max(abs(float(fwhm)), 1.0e-9)
    exponent = -4.0 * math.log(2.0) * ((x_arr - float(center)) / fwhm_value) ** 2
    return float(baseline) + float(amplitude) * np.exp(exponent)


def lorentzian_profile(
    x_values: Sequence[float] | np.ndarray,
    baseline: float,
    amplitude: float,
    center: float,
    fwhm: float,
) -> np.ndarray:
    """Return one Lorentzian peak parameterized directly by FWHM."""

    x_arr = np.asarray(x_values, dtype=float)
    fwhm_value = max(abs(float(fwhm)), 1.0e-9)
    denom = 1.0 + 4.0 * ((x_arr - float(center)) / fwhm_value) ** 2
    return float(baseline) + float(amplitude) / denom


def pseudo_voigt_profile(
    x_values: Sequence[float] | np.ndarray,
    baseline: float,
    amplitude: float,
    center: float,
    fwhm: float,
    eta: float,
) -> np.ndarray:
    """Return one pseudo-Voigt peak with FWHM and mixing fraction ``eta``."""

    eta_value = float(np.clip(float(eta), 0.0, 1.0))
    gaussian = gaussian_profile(x_values, 0.0, 1.0, center, fwhm)
    lorentzian = lorentzian_profile(x_values, 0.0, 1.0, center, fwhm)
    return float(baseline) + float(amplitude) * (
        (1.0 - eta_value) * gaussian + eta_value * lorentzian
    )


def evaluate_profile(
    model: str,
    x_values: Sequence[float] | np.ndarray,
    parameters: Sequence[float],
) -> np.ndarray:
    """Evaluate one supported peak profile family."""

    normalized = str(model or "").strip().lower()
    params = np.asarray(parameters, dtype=float)
    if normalized == PROFILE_GAUSSIAN and params.size >= 4:
        return gaussian_profile(x_values, params[0], params[1], params[2], params[3])
    if normalized == PROFILE_LORENTZIAN and params.size >= 4:
        return lorentzian_profile(x_values, params[0], params[1], params[2], params[3])
    if normalized == PROFILE_PSEUDO_VOIGT and params.size >= 5:
        return pseudo_voigt_profile(
            x_values,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
        )
    raise ValueError(f"Unsupported peak profile model: {model!r}")


def fit_peak_profile(
    x_values: Sequence[float] | None,
    y_values: Sequence[float] | None,
    *,
    center_guess: float,
    model: str,
    window_half_width: float,
    max_nfev: int = 400,
) -> dict[str, object]:
    """Fit one local 1D peak with the chosen model family."""

    normalized_model = str(model or "").strip().lower()
    if normalized_model not in SUPPORTED_PROFILE_MODELS:
        raise ValueError(f"Unsupported peak profile model: {model!r}")

    axis_arr, curve_arr = _prepare_curve_samples(x_values, y_values)
    if axis_arr.size < 7:
        return {
            "success": False,
            "model": normalized_model,
            "label": profile_model_label(normalized_model),
            "error": "Not enough finite samples are available for fitting.",
        }

    center_value = float(center_guess)
    half_width = max(abs(float(window_half_width)), 1.0e-9)
    lower_bound = center_value - half_width
    upper_bound = center_value + half_width
    window_mask = (axis_arr >= lower_bound) & (axis_arr <= upper_bound)
    if int(np.count_nonzero(window_mask)) < 7:
        nearest_order = np.argsort(np.abs(axis_arr - center_value))
        window_mask = np.zeros(axis_arr.shape, dtype=bool)
        window_mask[nearest_order[: min(21, axis_arr.size)]] = True

    x_window = axis_arr[window_mask]
    y_window = curve_arr[window_mask]
    if x_window.size < 7:
        return {
            "success": False,
            "model": normalized_model,
            "label": profile_model_label(normalized_model),
            "error": "The local fit window does not contain enough data.",
        }

    peak_index = int(np.argmax(y_window))
    y_min = float(np.min(y_window))
    y_max = float(np.max(y_window))
    y_span = max(y_max - y_min, 1.0e-6)
    baseline0 = float(np.percentile(y_window, 15.0))
    amplitude0 = max(float(y_window[peak_index]) - baseline0, 0.25 * y_span, 1.0e-6)
    center0 = float(np.clip(float(x_window[peak_index]), lower_bound, upper_bound))

    unique_x = np.unique(np.asarray(x_window, dtype=float))
    if unique_x.size >= 2:
        step = float(np.nanmedian(np.diff(unique_x)))
    else:
        step = abs(float(x_window[-1]) - float(x_window[0])) / max(int(x_window.size) - 1, 1)
    if not np.isfinite(step) or step <= 0.0:
        step = max(abs(float(x_window[-1]) - float(x_window[0])) / max(int(x_window.size) - 1, 1), 1.0e-4)
    span = max(abs(float(x_window[-1]) - float(x_window[0])), abs(upper_bound - lower_bound), step)
    min_fwhm = max(step * 2.0, span / max(4.0 * float(x_window.size), 1.0), 1.0e-6)
    max_fwhm = max(min_fwhm * 4.0, min(max(span * 1.25, min_fwhm * 4.0), 2.0 * half_width))
    fwhm0 = float(np.clip(max(span / 4.0, min_fwhm * 1.5), min_fwhm, max_fwhm))

    initial = [baseline0, amplitude0, center0, fwhm0]
    lower = [y_min - y_span, 0.0, lower_bound, min_fwhm]
    upper = [y_max + y_span, max(4.0 * y_span, amplitude0 * 4.0), upper_bound, max_fwhm]
    if normalized_model == PROFILE_PSEUDO_VOIGT:
        initial.append(0.5)
        lower.append(0.0)
        upper.append(1.0)

    def _residual(parameters: np.ndarray) -> np.ndarray:
        return evaluate_profile(normalized_model, x_window, parameters) - y_window

    try:
        result = least_squares(
            _residual,
            np.asarray(initial, dtype=float),
            bounds=(
                np.asarray(lower, dtype=float),
                np.asarray(upper, dtype=float),
            ),
            max_nfev=max(25, int(max_nfev)),
        )
    except Exception as exc:
        return {
            "success": False,
            "model": normalized_model,
            "label": profile_model_label(normalized_model),
            "error": str(exc),
        }

    params = np.asarray(result.x, dtype=float)
    fitted = evaluate_profile(normalized_model, x_window, params)
    residual = fitted - y_window
    fwhm_value = float(abs(params[3]))
    fit_result: dict[str, object] = {
        "success": bool(result.success),
        "model": normalized_model,
        "label": profile_model_label(normalized_model),
        "baseline": float(params[0]),
        "amplitude": float(params[1]),
        "center": float(params[2]),
        "fwhm": fwhm_value,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "rss": float(np.sum(residual**2)),
        "window_half_width": float(half_width),
        "x_window": np.asarray(x_window, dtype=float),
        "y_window": np.asarray(y_window, dtype=float),
        "y_fit": np.asarray(fitted, dtype=float),
        "nfev": int(getattr(result, "nfev", 0)),
    }
    if normalized_model == PROFILE_GAUSSIAN:
        fit_result["sigma"] = float(fwhm_value / (2.0 * math.sqrt(2.0 * math.log(2.0))))
    elif normalized_model == PROFILE_LORENTZIAN:
        fit_result["gamma"] = float(0.5 * fwhm_value)
    else:
        fit_result["eta"] = float(np.clip(float(params[4]), 0.0, 1.0))
    return fit_result


def _prepare_curve_samples(
    x_values: Sequence[float] | None,
    y_values: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one finite, sorted x/y curve pair."""

    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.size != y_arr.size:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(finite_mask):
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    x_arr = x_arr[finite_mask]
    y_arr = y_arr[finite_mask]
    order = np.argsort(x_arr)
    return np.asarray(x_arr[order], dtype=float), np.asarray(y_arr[order], dtype=float)


def _fit_entry_rmse_sort_key(entry: Mapping[str, object]) -> tuple[float, int]:
    try:
        rmse = float(entry.get("rmse", float("inf")))
    except Exception:
        rmse = float("inf")
    if not np.isfinite(rmse):
        rmse = float("inf")

    try:
        peak_index = int(entry.get("peak_index", 0))
    except Exception:
        peak_index = 0
    return float(rmse), int(peak_index)


def _fit_entry_prefix(entry: Mapping[str, object]) -> str:
    model_label = str(
        entry.get("label")
        or profile_model_label(str(entry.get("model", "Fit")))
    ).strip()

    prefix_parts: list[str] = []
    try:
        peak_index = int(entry.get("peak_index", 0))
    except Exception:
        peak_index = 0
    if peak_index > 0:
        prefix_parts.append(f"P{peak_index}")
    if model_label:
        prefix_parts.append(model_label)
    return " ".join(prefix_parts)


def _fit_entry_axis_position_text(entry: Mapping[str, object]) -> str:
    try:
        axis_value = float(entry.get("selected_axis_value", np.nan))
    except Exception:
        axis_value = float("nan")
    if np.isfinite(axis_value):
        return f"@ {axis_value:.4f} deg"
    return ""


def _format_peak_fit_entry_detail(entry: Mapping[str, object]) -> str:
    prefix = _fit_entry_prefix(entry)
    axis_text = _fit_entry_axis_position_text(entry)

    detail_parts: list[str] = []
    for key, label_text, precision in (
        ("center", "center", ".4f"),
        ("fwhm", "FWHM", ".4f"),
        ("rmse", "RMSE", ".4g"),
    ):
        try:
            value = float(entry.get(key, np.nan))
        except Exception:
            value = float("nan")
        if np.isfinite(value):
            detail_parts.append(f"{label_text} {value:{precision}}")

    detail = ", ".join(detail_parts)
    text = " ".join(part for part in (prefix, axis_text) if part).strip()
    if detail:
        return f"{text}; {detail}" if text else detail
    return text


def _format_peak_fit_entry_failure(entry: Mapping[str, object]) -> str:
    prefix = _fit_entry_prefix(entry)
    axis_text = _fit_entry_axis_position_text(entry)
    error_text = " ".join(str(entry.get("error", "fit failed") or "fit failed").split())
    text = " ".join(part for part in (prefix, axis_text) if part).strip()
    if text:
        return f"{text}: {error_text}"
    return error_text
