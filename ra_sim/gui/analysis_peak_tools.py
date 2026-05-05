"""Helpers for caked-view peak picking and 1D profile fitting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares


PROFILE_GAUSSIAN = "gaussian"
PROFILE_LORENTZIAN = "lorentzian"
PROFILE_MOSAIC_MIX = "mosaic_mix"
SUPPORTED_PROFILE_MODELS = (
    PROFILE_MOSAIC_MIX,
)
_GAUSSIAN_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


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


def subtract_linear_background_plane(
    theta_values: Sequence[float] | None,
    phi_values: Sequence[float] | None,
    roi_values: Sequence[Sequence[float]] | np.ndarray | None,
    peak_centers: Sequence[Mapping[str, object]] | None = None,
    *,
    theta_exclusion_half_width: float | None = None,
    phi_exclusion_half_width: float | None = None,
) -> dict[str, object]:
    """Fit and subtract one local 2D linear background plane from a caked ROI."""

    theta_axis = np.asarray(theta_values, dtype=float).reshape(-1)
    phi_axis = np.asarray(phi_values, dtype=float).reshape(-1)
    roi = np.asarray(roi_values, dtype=float)
    if (
        roi.ndim != 2
        or theta_axis.size != int(roi.shape[1])
        or phi_axis.size != int(roi.shape[0])
        or theta_axis.size <= 0
        or phi_axis.size <= 0
    ):
        return {
            "success": False,
            "error": "ROI shape does not match caked axes.",
        }

    theta_grid, phi_grid = np.meshgrid(theta_axis, phi_axis)
    finite = np.isfinite(roi) & np.isfinite(theta_grid) & np.isfinite(phi_grid)
    finite_count = int(np.count_nonzero(finite))
    if finite_count < 3:
        return {
            "success": False,
            "error": "Not enough finite ROI samples for linear background.",
        }

    def _axis_step(values: np.ndarray, fallback: float) -> float:
        finite_values = np.sort(values[np.isfinite(values)])
        if finite_values.size >= 2:
            diffs = np.diff(finite_values)
            diffs = diffs[np.isfinite(diffs) & (np.abs(diffs) > 0.0)]
            if diffs.size:
                return float(np.nanmedian(np.abs(diffs)))
        return float(fallback)

    def _wrapped_delta(values: np.ndarray, center: float) -> np.ndarray:
        return ((np.asarray(values, dtype=float) - float(center) + 180.0) % 360.0) - 180.0

    finite_theta_axis = theta_axis[np.isfinite(theta_axis)]
    finite_phi_axis = phi_axis[np.isfinite(phi_axis)]
    theta_span = (
        float(np.nanmax(finite_theta_axis) - np.nanmin(finite_theta_axis))
        if finite_theta_axis.size >= 2
        else 0.0
    )
    phi_span = (
        float(np.nanmax(finite_phi_axis) - np.nanmin(finite_phi_axis))
        if finite_phi_axis.size >= 2
        else 0.0
    )
    theta_step = _axis_step(theta_axis, 0.03)
    phi_step = _axis_step(phi_axis, 0.25)

    def _default_half_width(
        *,
        axis_step: float,
        axis_span: float,
        floor: float,
        span_fraction: float,
    ) -> float:
        half_width = max(8.0 * float(axis_step), float(floor))
        if np.isfinite(axis_span) and axis_span > 0.0:
            half_width = max(half_width, float(span_fraction) * float(axis_span))
            half_width = min(half_width, 0.48 * float(axis_span))
        return float(max(half_width, 0.0))

    theta_half = (
        _default_half_width(
            axis_step=theta_step,
            axis_span=theta_span,
            floor=0.12,
            span_fraction=0.20,
        )
        if theta_exclusion_half_width is None
        else max(float(theta_exclusion_half_width), 0.0)
    )
    phi_half = (
        _default_half_width(
            axis_step=phi_step,
            axis_span=phi_span,
            floor=1.0,
            span_fraction=0.25,
        )
        if phi_exclusion_half_width is None
        else max(float(phi_exclusion_half_width), 0.0)
    )

    peak_mask = np.zeros(roi.shape, dtype=bool)
    for entry in peak_centers or ():
        if not isinstance(entry, Mapping):
            continue
        try:
            theta_center = float(entry.get("two_theta_deg"))
            phi_center = float(entry.get("phi_deg"))
        except Exception:
            continue
        if not (np.isfinite(theta_center) and np.isfinite(phi_center)):
            continue
        peak_mask |= (
            (np.abs(theta_grid - theta_center) <= theta_half)
            & (np.abs(_wrapped_delta(phi_grid, phi_center)) <= phi_half)
        )

    theta_ref = float(np.nanmedian(theta_grid[finite]))
    phi_ref = float(wrap_angle_degrees(np.nanmedian(phi_grid[finite])))
    minimum_fit_count = max(6, min(finite_count, int(math.ceil(0.15 * finite_count))))

    def _edge_band_mask() -> np.ndarray:
        row_count, col_count = roi.shape
        row_width = max(1, min(row_count, int(math.ceil(0.15 * row_count))))
        col_width = max(1, min(col_count, int(math.ceil(0.15 * col_count))))
        row_indices = np.arange(row_count, dtype=int)[:, None]
        col_indices = np.arange(col_count, dtype=int)[None, :]
        return (
            (row_indices < row_width)
            | (row_indices >= row_count - row_width)
            | (col_indices < col_width)
            | (col_indices >= col_count - col_width)
        )

    def _failure_payload(error: str, *, fit_mask: np.ndarray, used_edge: bool) -> dict[str, object]:
        return {
            "success": False,
            "error": str(error),
            "fit_mask": np.asarray(fit_mask, dtype=bool),
            "peak_mask": np.asarray(peak_mask, dtype=bool),
            "fit_sample_count": int(np.count_nonzero(fit_mask)),
            "initial_fit_sample_count": int(np.count_nonzero(fit_mask)),
            "finite_sample_count": int(finite_count),
            "peak_mask_sample_count": int(np.count_nonzero(finite & peak_mask)),
            "used_fallback_fit_mask": bool(used_edge),
            "used_edge_fit_mask": bool(used_edge),
            "fallback_mode": "edge_background" if used_edge else "none",
            "clipped_sample_count": 0,
            "theta_exclusion_half_width": float(theta_half),
            "phi_exclusion_half_width": float(phi_half),
        }

    background_mask = finite & ~peak_mask
    edge_background_mask = background_mask & _edge_band_mask()
    if int(np.count_nonzero(background_mask)) < minimum_fit_count:
        return _failure_payload(
            "Not enough background samples remain outside excluded peak cores.",
            fit_mask=background_mask,
            used_edge=False,
        )

    def _fit_plane_coefficients(mask: np.ndarray) -> np.ndarray:
        sample_count = int(np.count_nonzero(mask))
        if sample_count < 3:
            raise ValueError("Not enough samples are available for a 2D background plane.")
        design = np.column_stack(
            [
                np.ones(sample_count, dtype=float),
                np.asarray(theta_grid[mask] - theta_ref, dtype=float),
                np.asarray(_wrapped_delta(phi_grid[mask], phi_ref), dtype=float),
            ]
        )
        y_fit = np.asarray(roi[mask], dtype=float)
        coeff, *_ = np.linalg.lstsq(design, y_fit, rcond=None)
        coeff = np.asarray(coeff, dtype=float)
        if coeff.size < 3 or not np.all(np.isfinite(coeff[:3])):
            raise ValueError("Linear background plane fit produced non-finite coefficients.")
        return coeff[:3]

    def _evaluate_plane(coeff: np.ndarray) -> np.ndarray:
        return (
            float(coeff[0])
            + float(coeff[1]) * (theta_grid - theta_ref)
            + float(coeff[2]) * _wrapped_delta(phi_grid, phi_ref)
        )

    def _robust_fit_plane(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
        fit_mask = np.asarray(mask, dtype=bool)
        initial_fit_sample_count = int(np.count_nonzero(fit_mask))
        if initial_fit_sample_count < minimum_fit_count:
            raise ValueError("Not enough background samples are available for clipping.")
        clipped_sample_count = 0
        for _iteration in range(3):
            coeff = _fit_plane_coefficients(fit_mask)
            residual = np.asarray(roi - _evaluate_plane(coeff), dtype=float)
            sample_residual = np.asarray(residual[fit_mask], dtype=float)
            sample_residual = sample_residual[np.isfinite(sample_residual)]
            if sample_residual.size < minimum_fit_count:
                break
            residual_center = float(np.nanmedian(sample_residual))
            mad = float(np.nanmedian(np.abs(sample_residual - residual_center)))
            residual_scale = 1.4826 * mad
            if not np.isfinite(residual_scale) or residual_scale <= 1.0e-12:
                residual_scale = float(np.nanstd(sample_residual))
            if not np.isfinite(residual_scale) or residual_scale <= 1.0e-12:
                break
            threshold = residual_center + 3.0 * residual_scale
            high_residual = fit_mask & np.isfinite(residual) & (residual > threshold)
            high_count = int(np.count_nonzero(high_residual))
            if high_count <= 0:
                break
            proposed_mask = fit_mask & ~high_residual
            if int(np.count_nonzero(proposed_mask)) < minimum_fit_count:
                break
            fit_mask = proposed_mask
            clipped_sample_count += high_count
        coeff = _fit_plane_coefficients(fit_mask)
        return coeff, fit_mask, int(clipped_sample_count), int(initial_fit_sample_count)

    used_edge = False
    try:
        coeff, fit_mask, clipped_sample_count, initial_fit_sample_count = _robust_fit_plane(
            background_mask
        )
    except Exception as primary_exc:
        if int(np.count_nonzero(edge_background_mask)) >= minimum_fit_count:
            try:
                (
                    coeff,
                    fit_mask,
                    clipped_sample_count,
                    initial_fit_sample_count,
                ) = _robust_fit_plane(edge_background_mask)
                used_edge = True
            except Exception as edge_exc:
                return _failure_payload(
                    f"{primary_exc}; edge-band retry failed: {edge_exc}",
                    fit_mask=edge_background_mask,
                    used_edge=True,
                )
        else:
            return _failure_payload(
                str(primary_exc),
                fit_mask=background_mask,
                used_edge=False,
            )

    plane = _evaluate_plane(coeff)
    corrected = np.asarray(roi - plane, dtype=float)
    return {
        "success": True,
        "corrected": corrected,
        "background_plane": np.asarray(plane, dtype=float),
        "fit_mask": np.asarray(fit_mask, dtype=bool),
        "peak_mask": np.asarray(peak_mask, dtype=bool),
        "coefficients": [float(coeff[0]), float(coeff[1]), float(coeff[2])],
        "theta_ref": float(theta_ref),
        "phi_ref": float(phi_ref),
        "fit_sample_count": int(np.count_nonzero(fit_mask)),
        "initial_fit_sample_count": int(initial_fit_sample_count),
        "finite_sample_count": int(finite_count),
        "peak_mask_sample_count": int(np.count_nonzero(finite & peak_mask)),
        "used_fallback_fit_mask": bool(used_edge),
        "used_edge_fit_mask": bool(used_edge),
        "fallback_mode": "edge_background" if used_edge else "none",
        "clipped_sample_count": int(clipped_sample_count),
        "theta_exclusion_half_width": float(theta_half),
        "phi_exclusion_half_width": float(phi_half),
    }


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
    if normalized == PROFILE_MOSAIC_MIX:
        return "Mosaic mix"
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


def format_peak_fit_axis_table(
    axis_label: str,
    entries: Sequence[Mapping[str, object]] | None,
) -> str:
    """Return a monospaced table of peak-fit widths and mixture percentages."""

    normalized_entries = [entry for entry in (entries or ()) if isinstance(entry, Mapping)]
    if not normalized_entries:
        return ""

    successful_entries = [
        entry for entry in normalized_entries if bool(entry.get("success", False))
    ]
    prefix = f"{str(axis_label)}: {len(successful_entries)}/{len(normalized_entries)} fits"
    best_text = ""
    if successful_entries:
        best_entry = min(
            successful_entries,
            key=lambda entry: _fit_entry_rmse_sort_key(entry),
        )
        best_text = _format_peak_fit_best_table_note(best_entry)
    if best_text:
        prefix = f"{prefix}; best {best_text}"

    lines = [
        prefix,
        "ID   Model    At deg   Center   G-FWHM  L-FWHM  G/L%       RMSE",
    ]
    failure_lines: list[str] = []
    for entry in sorted(normalized_entries, key=_fit_entry_table_sort_key):
        lines.append(_format_peak_fit_table_row(entry))
        if not bool(entry.get("success", False)):
            failure_text = _format_peak_fit_entry_failure(entry)
            if failure_text:
                failure_lines.append(failure_text)

    if failure_lines:
        lines.append("Failures:")
        lines.extend(failure_lines)
    return "\n".join(lines)


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
    default_half_width = max(
        min_half_width, region_span / 14.0 if region_span > 0.0 else min_half_width
    )
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
    return float(baseline) + float(amplitude) * _gaussian_unit_profile(
        x_arr,
        center=float(center),
        fwhm=float(fwhm),
    )


def lorentzian_profile(
    x_values: Sequence[float] | np.ndarray,
    baseline: float,
    amplitude: float,
    center: float,
    fwhm: float,
) -> np.ndarray:
    """Return one Lorentzian peak parameterized directly by FWHM."""

    x_arr = np.asarray(x_values, dtype=float)
    return float(baseline) + float(amplitude) * _lorentzian_unit_profile(
        x_arr,
        center=float(center),
        fwhm=float(fwhm),
    )


def mosaic_mix_profile(
    x_values: Sequence[float] | np.ndarray,
    baseline: float,
    amplitude: float,
    center: float,
    sigma: float,
    gamma: float,
    eta: float,
) -> np.ndarray:
    """Return one area-normalized independent Gaussian/Lorentzian mosaic mix.

    ``amplitude`` is the integrated peak area, and ``eta`` is the Lorentzian
    area fraction.
    """

    x_arr = np.asarray(x_values, dtype=float)
    return float(baseline) + float(amplitude) * _mosaic_mix_unit_profile(
        x_arr,
        center=float(center),
        sigma=float(sigma),
        gamma=float(gamma),
        eta=float(eta),
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
    if normalized == PROFILE_MOSAIC_MIX and params.size >= 6:
        return mosaic_mix_profile(
            x_values,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
        )
    raise ValueError(f"Unsupported peak profile model: {model!r}")


def _gaussian_unit_profile(
    x_values: Sequence[float] | np.ndarray,
    *,
    center: float,
    fwhm: float,
) -> np.ndarray:
    """Return one baseline-free, unit-amplitude Gaussian profile."""

    x_arr = np.asarray(x_values, dtype=float)
    fwhm_value = max(abs(float(fwhm)), 1.0e-9)
    exponent = -4.0 * math.log(2.0) * ((x_arr - float(center)) / fwhm_value) ** 2
    return np.asarray(np.exp(exponent), dtype=float)


def _lorentzian_unit_profile(
    x_values: Sequence[float] | np.ndarray,
    *,
    center: float,
    fwhm: float,
) -> np.ndarray:
    """Return one baseline-free, unit-amplitude Lorentzian profile."""

    x_arr = np.asarray(x_values, dtype=float)
    fwhm_value = max(abs(float(fwhm)), 1.0e-9)
    denom = 1.0 + 4.0 * ((x_arr - float(center)) / fwhm_value) ** 2
    return np.asarray(1.0 / denom, dtype=float)


def _gaussian_unit_area_profile(
    x_values: Sequence[float] | np.ndarray,
    *,
    center: float,
    fwhm: float,
) -> np.ndarray:
    """Return one baseline-free, unit-area Gaussian profile."""

    fwhm_value = max(abs(float(fwhm)), 1.0e-9)
    coefficient = 2.0 * math.sqrt(math.log(2.0)) / (math.sqrt(math.pi) * fwhm_value)
    return coefficient * _gaussian_unit_profile(x_values, center=center, fwhm=fwhm_value)


def _gaussian_sigma_unit_area_profile(
    x_values: Sequence[float] | np.ndarray,
    *,
    center: float,
    sigma: float,
) -> np.ndarray:
    """Return one baseline-free, unit-area Gaussian parameterized by sigma."""

    x_arr = np.asarray(x_values, dtype=float)
    sigma_value = max(abs(float(sigma)), 1.0e-9)
    delta = x_arr - float(center)
    coefficient = 1.0 / (sigma_value * math.sqrt(2.0 * math.pi))
    return np.asarray(coefficient * np.exp(-0.5 * (delta / sigma_value) ** 2), dtype=float)


def _lorentzian_unit_area_profile(
    x_values: Sequence[float] | np.ndarray,
    *,
    center: float,
    fwhm: float,
) -> np.ndarray:
    """Return one baseline-free, unit-area Lorentzian profile."""

    fwhm_value = max(abs(float(fwhm)), 1.0e-9)
    coefficient = 2.0 / (math.pi * fwhm_value)
    return coefficient * _lorentzian_unit_profile(x_values, center=center, fwhm=fwhm_value)


def _lorentzian_gamma_unit_area_profile(
    x_values: Sequence[float] | np.ndarray,
    *,
    center: float,
    gamma: float,
) -> np.ndarray:
    """Return one baseline-free, unit-area Lorentzian parameterized by HWHM gamma."""

    x_arr = np.asarray(x_values, dtype=float)
    gamma_value = max(abs(float(gamma)), 1.0e-9)
    delta = x_arr - float(center)
    return np.asarray(
        (gamma_value / math.pi) / (delta**2 + gamma_value**2),
        dtype=float,
    )


def _mosaic_mix_unit_profile(
    x_values: Sequence[float] | np.ndarray,
    *,
    center: float,
    sigma: float,
    gamma: float,
    eta: float,
) -> np.ndarray:
    """Return one baseline-free, unit-area independent-width mosaic profile."""

    eta_value = float(np.clip(float(eta), 0.0, 1.0))
    gaussian = _gaussian_sigma_unit_area_profile(x_values, center=center, sigma=sigma)
    lorentzian = _lorentzian_gamma_unit_area_profile(x_values, center=center, gamma=gamma)
    return np.asarray((1.0 - eta_value) * gaussian + eta_value * lorentzian, dtype=float)


def _profile_unit_peak_height(
    model: str,
    *,
    fwhm: float,
    sigma: float | None = None,
    gamma: float | None = None,
    eta: float = 0.5,
) -> float:
    """Return model value at peak center for one unit-amplitude/unit-area peak."""

    normalized_model = str(model or "").strip().lower()
    if normalized_model != PROFILE_MOSAIC_MIX:
        return 1.0
    if sigma is None:
        sigma_value = max(abs(float(fwhm)) / _GAUSSIAN_FWHM_PER_SIGMA, 1.0e-9)
    else:
        sigma_value = max(abs(float(sigma)), 1.0e-9)
    if gamma is None:
        gamma_value = max(0.5 * abs(float(fwhm)), 1.0e-9)
    else:
        gamma_value = max(abs(float(gamma)), 1.0e-9)
    eta_value = float(np.clip(float(eta), 0.0, 1.0))
    gaussian_peak = 1.0 / (sigma_value * math.sqrt(2.0 * math.pi))
    lorentzian_peak = 1.0 / (math.pi * gamma_value)
    return float((1.0 - eta_value) * gaussian_peak + eta_value * lorentzian_peak)


def _profile_amplitude_from_peak_height(
    model: str,
    peak_height: float,
    *,
    fwhm: float,
    sigma: float | None = None,
    gamma: float | None = None,
    eta: float = 0.5,
) -> float:
    """Convert desired peak height into the fitted amplitude parameter."""

    unit_peak_height = max(
        _profile_unit_peak_height(model, fwhm=fwhm, sigma=sigma, gamma=gamma, eta=eta),
        1.0e-12,
    )
    return float(max(float(peak_height), 1.0e-12) / unit_peak_height)


def _profile_unit_values(
    model: str,
    x_values: Sequence[float] | np.ndarray,
    *,
    center: float,
    fwhm: float,
    sigma: float | None = None,
    gamma: float | None = None,
    eta: float = 0.5,
) -> np.ndarray:
    """Return one baseline-free profile for the chosen family."""

    normalized_model = str(model or "").strip().lower()
    if normalized_model == PROFILE_GAUSSIAN:
        return _gaussian_unit_profile(x_values, center=center, fwhm=fwhm)
    if normalized_model == PROFILE_LORENTZIAN:
        return _lorentzian_unit_profile(x_values, center=center, fwhm=fwhm)
    if normalized_model == PROFILE_MOSAIC_MIX:
        sigma_value = max(
            abs(float(sigma)) if sigma is not None else abs(float(fwhm)) / _GAUSSIAN_FWHM_PER_SIGMA,
            1.0e-9,
        )
        gamma_value = max(
            abs(float(gamma)) if gamma is not None else 0.5 * abs(float(fwhm)),
            1.0e-9,
        )
        return _mosaic_mix_unit_profile(
            x_values,
            center=center,
            sigma=sigma_value,
            gamma=gamma_value,
            eta=eta,
        )
    raise ValueError(f"Unsupported peak profile model: {model!r}")


def _axis_step_from_values(axis_values: Sequence[float] | np.ndarray) -> float:
    """Return one robust positive sample spacing for a sorted axis."""

    axis_arr = np.asarray(axis_values, dtype=float)
    finite_axis = np.unique(axis_arr[np.isfinite(axis_arr)])
    if finite_axis.size < 2:
        return 0.0
    positive_diffs = np.diff(np.sort(finite_axis))
    positive_diffs = positive_diffs[np.isfinite(positive_diffs) & (positive_diffs > 0.0)]
    if positive_diffs.size <= 0:
        return 0.0
    return float(np.nanmedian(positive_diffs))


def _component_parameter_count(model: str, component_count: int) -> int:
    """Return one flattened parameter count for a composite fit."""

    per_component = 5 if str(model or "").strip().lower() == PROFILE_MOSAIC_MIX else 3
    return int(1 + per_component * max(int(component_count), 0))


def _collapse_center_guess_groups(
    center_guesses: Sequence[float] | np.ndarray,
    *,
    duplicate_center_tolerance: float,
) -> list[dict[str, object]]:
    """Collapse near-duplicate center guesses into one ordered component map."""

    centers_arr = np.asarray(center_guesses, dtype=float)
    finite_pairs = [
        (int(index), float(value)) for index, value in enumerate(centers_arr) if np.isfinite(value)
    ]
    if not finite_pairs:
        return []

    tolerance = max(abs(float(duplicate_center_tolerance)), 1.0e-9)
    ordered_pairs = sorted(finite_pairs, key=lambda item: float(item[1]))
    grouped_pairs: list[list[tuple[int, float]]] = [[ordered_pairs[0]]]
    for index, value in ordered_pairs[1:]:
        current_group = grouped_pairs[-1]
        current_values = np.asarray([candidate[1] for candidate in current_group], dtype=float)
        group_center = float(np.mean(current_values))
        if abs(float(value) - group_center) <= tolerance:
            current_group.append((int(index), float(value)))
            continue
        grouped_pairs.append([(int(index), float(value))])

    collapsed_groups: list[dict[str, object]] = []
    for component_index, group in enumerate(grouped_pairs):
        group_values = np.asarray([value for _index, value in group], dtype=float)
        collapsed_groups.append(
            {
                "component_index": int(component_index),
                "center_guess_indices": [int(index) for index, _value in group],
                "selected_axis_value": float(np.mean(group_values)),
            }
        )
    return collapsed_groups


def _selected_window_fit_coordinate_context(
    axis_values: Sequence[float] | np.ndarray,
    center_guesses: Sequence[float] | np.ndarray,
) -> dict[str, object]:
    """Return one coordinate transform that makes wrapped windows contiguous."""

    axis_arr = np.asarray(axis_values, dtype=float)
    center_arr = np.asarray(center_guesses, dtype=float)
    fit_axis_unsorted = np.asarray(axis_arr, dtype=float)
    fit_centers = np.asarray(center_arr, dtype=float)
    wrap_period: float | None = None

    unique_axis = np.unique(axis_arr[np.isfinite(axis_arr)])
    axis_step = _axis_step_from_values(unique_axis)
    axis_span = (
        abs(float(unique_axis[-1]) - float(unique_axis[0])) if unique_axis.size >= 2 else 0.0
    )
    if unique_axis.size >= 3 and axis_span > 180.0:
        diffs = np.diff(unique_axis)
        positive_diffs = diffs[np.isfinite(diffs)]
        if positive_diffs.size > 0:
            gap_index = int(np.argmax(positive_diffs))
            max_gap = float(positive_diffs[gap_index])
            if max_gap > max(180.0, 10.0 * max(axis_step, 1.0e-9)):
                wrap_period = 360.0
                gap_left = float(unique_axis[gap_index])
                shift_limit = gap_left + 0.5 * max(axis_step, 1.0e-9)
                fit_axis_unsorted = np.where(
                    axis_arr <= shift_limit,
                    axis_arr + wrap_period,
                    axis_arr,
                )
                fit_centers = np.where(
                    np.isfinite(center_arr) & (center_arr <= shift_limit),
                    center_arr + wrap_period,
                    center_arr,
                )

    fit_order = np.argsort(fit_axis_unsorted, kind="stable")
    return {
        "fit_axis_unsorted": np.asarray(fit_axis_unsorted, dtype=float),
        "fit_axis_sorted": np.asarray(fit_axis_unsorted[fit_order], dtype=float),
        "fit_order": np.asarray(fit_order, dtype=int),
        "fit_center_guesses": np.asarray(fit_centers, dtype=float),
        "wrap_period": wrap_period,
    }


def _display_coordinate_from_fit_value(
    fit_value: float,
    *,
    axis_values: Sequence[float] | np.ndarray | None = None,
    reference_value: float,
    wrap_period: float | None,
) -> float:
    """Map one internal fit coordinate back onto the selected-window axis domain."""

    if not np.isfinite(fit_value):
        return float("nan")
    if wrap_period is None or not np.isfinite(float(wrap_period)):
        return float(fit_value)
    axis_arr = np.asarray(axis_values, dtype=float)
    if axis_arr.size > 0 and np.any(np.isfinite(axis_arr)):
        return align_angle_to_axis(float(fit_value), axis_arr)
    reference = float(reference_value) if np.isfinite(reference_value) else float(fit_value)
    candidates = np.asarray(
        [
            float(fit_value),
            float(fit_value) - float(wrap_period),
            float(fit_value) + float(wrap_period),
        ],
        dtype=float,
    )
    candidate_index = int(np.argmin(np.abs(candidates - reference)))
    return float(candidates[candidate_index])


def _evaluate_composite_profile(
    model: str,
    x_values: Sequence[float] | np.ndarray,
    parameters: Sequence[float] | np.ndarray,
    *,
    component_count: int,
) -> np.ndarray:
    """Evaluate one composite peak profile with a shared baseline."""

    normalized_model = str(model or "").strip().lower()
    params = np.asarray(parameters, dtype=float)
    x_arr = np.asarray(x_values, dtype=float)
    profile = np.full(x_arr.shape, float(params[0]), dtype=float)
    stride = 5 if normalized_model == PROFILE_MOSAIC_MIX else 3
    for component_index in range(max(int(component_count), 0)):
        offset = 1 + component_index * stride
        amplitude = float(params[offset])
        center = float(params[offset + 1])
        if normalized_model == PROFILE_MOSAIC_MIX:
            sigma = float(params[offset + 2])
            gamma = float(params[offset + 3])
            eta = float(params[offset + 4])
            profile += amplitude * _profile_unit_values(
                normalized_model,
                x_arr,
                center=center,
                fwhm=float(sigma) * _GAUSSIAN_FWHM_PER_SIGMA,
                sigma=sigma,
                gamma=gamma,
                eta=eta,
            )
        else:
            fwhm = float(params[offset + 2])
            profile += amplitude * _profile_unit_values(
                normalized_model,
                x_arr,
                center=center,
                fwhm=fwhm,
                eta=0.5,
            )
    return np.asarray(profile, dtype=float)


def _tail_weight_floor(data_y: Sequence[float] | np.ndarray, baseline: float) -> float:
    """Return a positive variance floor for tail-aware profile residuals."""

    signal = np.abs(np.asarray(data_y, dtype=float) - float(baseline))
    finite_signal = signal[np.isfinite(signal)]
    if finite_signal.size <= 0:
        return 1.0e-6
    percentile_floor = float(np.nanpercentile(finite_signal, 25.0))
    max_floor = 0.02 * float(np.nanmax(finite_signal))
    return float(max(1.0e-6, percentile_floor, max_floor))


def _tail_weighted_profile_residual(
    model_y: Sequence[float] | np.ndarray,
    data_y: Sequence[float] | np.ndarray,
    *,
    baseline: float,
    floor: float,
) -> np.ndarray:
    """Return residuals weighted to keep low-intensity tails visible."""

    model_arr = np.asarray(model_y, dtype=float)
    data_arr = np.asarray(data_y, dtype=float)
    denom = np.sqrt(np.abs(data_arr - float(baseline)) + max(float(floor), 1.0e-12))
    return np.asarray((model_arr - data_arr) / denom, dtype=float)


def fit_composite_peak_profile(
    x_values: Sequence[float] | None,
    y_values: Sequence[float] | None,
    center_guesses: Sequence[float] | None,
    *,
    model: str,
    max_nfev: int = 1000,
) -> dict[str, object]:
    """Fit one composite 1D peak model over the full selected-window curve."""

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

    axis_step = _axis_step_from_values(axis_arr)
    if not np.isfinite(axis_step) or axis_step <= 0.0:
        if axis_arr.size >= 2:
            axis_step = abs(float(axis_arr[-1]) - float(axis_arr[0])) / max(
                int(axis_arr.size) - 1, 1
            )
        else:
            axis_step = 0.0
    if not np.isfinite(axis_step) or axis_step <= 0.0:
        axis_step = 1.0e-6

    center_guess_arr = np.asarray(
        () if center_guesses is None else center_guesses,
        dtype=float,
    )
    duplicate_center_tolerance = max(float(axis_step), 1.0e-9)
    collapsed_groups = _collapse_center_guess_groups(
        center_guess_arr,
        duplicate_center_tolerance=duplicate_center_tolerance,
    )
    if not collapsed_groups:
        return {
            "success": False,
            "model": normalized_model,
            "label": profile_model_label(normalized_model),
            "error": "No finite center guesses are available for fitting.",
        }

    fit_coordinate_context = _selected_window_fit_coordinate_context(axis_arr, center_guess_arr)
    fit_axis_unsorted = np.asarray(fit_coordinate_context["fit_axis_unsorted"], dtype=float)
    fit_axis_arr = np.asarray(fit_coordinate_context["fit_axis_sorted"], dtype=float)
    fit_order = np.asarray(fit_coordinate_context["fit_order"], dtype=int)
    fit_curve_arr = np.asarray(curve_arr[fit_order], dtype=float)
    fit_center_guesses = np.asarray(
        fit_coordinate_context["fit_center_guesses"],
        dtype=float,
    )
    wrap_period = fit_coordinate_context["wrap_period"]

    collapsed_fit_groups = _collapse_center_guess_groups(
        fit_center_guesses,
        duplicate_center_tolerance=duplicate_center_tolerance,
    )
    collapsed_groups = []
    for raw_group in collapsed_fit_groups:
        guess_indices = [int(index) for index in raw_group.get("center_guess_indices", [])]
        fit_selected_axis_value = float(raw_group.get("selected_axis_value", np.nan))
        if guess_indices:
            guess_fit_values = np.asarray(fit_center_guesses[guess_indices], dtype=float)
            closest_guess_offset = int(
                np.argmin(np.abs(guess_fit_values - fit_selected_axis_value))
            )
            display_reference_index = int(guess_indices[closest_guess_offset])
            selected_axis_value = float(center_guess_arr[display_reference_index])
        else:
            selected_axis_value = _display_coordinate_from_fit_value(
                fit_selected_axis_value,
                axis_values=axis_arr,
                reference_value=float("nan"),
                wrap_period=wrap_period,
            )
        collapsed_groups.append(
            {
                "component_index": int(raw_group["component_index"]),
                "center_guess_indices": guess_indices,
                "fit_selected_axis_value": fit_selected_axis_value,
                "selected_axis_value": selected_axis_value,
            }
        )

    component_count = len(collapsed_groups)
    parameter_count = _component_parameter_count(normalized_model, component_count)
    if fit_axis_arr.size <= parameter_count:
        return {
            "success": False,
            "model": normalized_model,
            "label": profile_model_label(normalized_model),
            "error": "Not enough finite samples are available for the requested composite fit.",
        }

    y_min = float(np.min(fit_curve_arr))
    y_max = float(np.max(fit_curve_arr))
    y_span = max(y_max - y_min, 1.0e-6)
    baseline0 = float(np.percentile(fit_curve_arr, 15.0))
    full_span = max(abs(float(fit_axis_arr[-1]) - float(fit_axis_arr[0])), float(axis_step))
    collapsed_centers = np.asarray(
        [float(group["fit_selected_axis_value"]) for group in collapsed_groups],
        dtype=float,
    )

    initial = [baseline0]
    lower = [y_min - y_span]
    upper = [y_max + y_span]
    stride = 5 if normalized_model == PROFILE_MOSAIC_MIX else 3
    min_fwhm = max(2.0 * float(axis_step), 1.0e-6)
    min_sigma = min_fwhm / _GAUSSIAN_FWHM_PER_SIGMA
    min_gamma = 0.5 * min_fwhm

    for component_index, selected_axis_value in enumerate(collapsed_centers):
        center_lower = float(fit_axis_arr[0])
        center_upper = float(fit_axis_arr[-1])
        nearest_neighbor_spacing = float(full_span)
        if component_index > 0:
            left_neighbor = float(collapsed_centers[component_index - 1])
            center_lower = max(center_lower, 0.5 * (left_neighbor + float(selected_axis_value)))
            nearest_neighbor_spacing = min(
                nearest_neighbor_spacing,
                abs(float(selected_axis_value) - left_neighbor),
            )
        if component_index < component_count - 1:
            right_neighbor = float(collapsed_centers[component_index + 1])
            center_upper = min(center_upper, 0.5 * (float(selected_axis_value) + right_neighbor))
            nearest_neighbor_spacing = min(
                nearest_neighbor_spacing,
                abs(right_neighbor - float(selected_axis_value)),
            )
        if not np.isfinite(nearest_neighbor_spacing) or nearest_neighbor_spacing <= 0.0:
            nearest_neighbor_spacing = float(full_span)
        if center_upper <= center_lower:
            center_pad = max(float(axis_step), 1.0e-6)
            center_lower = float(selected_axis_value) - center_pad
            center_upper = float(selected_axis_value) + center_pad

        if component_count > 1:
            max_candidate = max(
                10.0 * float(axis_step),
                0.15 * float(full_span),
                2.5 * float(nearest_neighbor_spacing),
            )
        else:
            max_candidate = max(10.0 * float(axis_step), 0.35 * float(full_span))
        fwhm_max = min(float(full_span), float(max_candidate))
        if not np.isfinite(fwhm_max) or fwhm_max <= min_fwhm:
            fwhm_max = max(min_fwhm * 4.0, float(full_span), 10.0 * float(axis_step))
        sigma_max = max(float(fwhm_max) / _GAUSSIAN_FWHM_PER_SIGMA, min_sigma * 4.0)
        if component_count > 1:
            gamma_candidate = max(
                10.0 * float(axis_step),
                0.5 * float(full_span),
                1.5 * float(nearest_neighbor_spacing),
            )
            gamma_max = min(float(full_span), float(gamma_candidate))
        else:
            gamma_max = float(full_span)
        if not np.isfinite(gamma_max) or gamma_max <= min_gamma:
            gamma_max = max(min_gamma * 4.0, float(full_span), 10.0 * float(axis_step))

        center0 = float(np.clip(float(selected_axis_value), center_lower, center_upper))
        sample_y = float(np.interp(center0, fit_axis_arr, fit_curve_arr))
        peak_height0 = max(sample_y - baseline0, 0.05 * y_span, 1.0e-6)
        width_seed = max(
            4.0 * float(axis_step),
            min(
                float(nearest_neighbor_spacing),
                0.35 * float(full_span),
            ),
        )
        fwhm0 = float(np.clip(width_seed, min_fwhm, fwhm_max))
        sigma0 = float(np.clip(fwhm0 / _GAUSSIAN_FWHM_PER_SIGMA, min_sigma, sigma_max))
        gamma0 = float(np.clip(0.5 * fwhm0, min_gamma, gamma_max))
        amplitude0 = _profile_amplitude_from_peak_height(
            normalized_model,
            peak_height0,
            fwhm=fwhm0,
            sigma=sigma0,
            gamma=gamma0,
            eta=0.5,
        )
        widest_lorentzian_area_scale = 1.0 / max(
            _profile_unit_peak_height(
                normalized_model,
                fwhm=fwhm_max,
                sigma=sigma_max,
                gamma=gamma_max,
                eta=1.0,
            ),
            1.0e-12,
        )
        amplitude_upper = max(
            4.0 * y_span * widest_lorentzian_area_scale,
            4.0 * amplitude0,
            1.0e-6,
        )

        if normalized_model == PROFILE_MOSAIC_MIX:
            initial.extend([amplitude0, center0, sigma0, gamma0, 0.5])
            lower.extend([0.0, center_lower, min_sigma, min_gamma, 0.0])
            upper.extend([amplitude_upper, center_upper, sigma_max, gamma_max, 1.0])
        else:
            initial.extend([amplitude0, center0, fwhm0])
            lower.extend([0.0, center_lower, min_fwhm])
            upper.extend([amplitude_upper, center_upper, fwhm_max])

    residual_floor = _tail_weight_floor(fit_curve_arr, baseline0)

    def _residual(parameters: np.ndarray) -> np.ndarray:
        model_y = _evaluate_composite_profile(
            normalized_model,
            fit_axis_arr,
            parameters,
            component_count=component_count,
        )
        if normalized_model == PROFILE_MOSAIC_MIX:
            return _tail_weighted_profile_residual(
                model_y,
                fit_curve_arr,
                baseline=baseline0,
                floor=residual_floor,
            )
        return model_y - fit_curve_arr

    try:
        result = least_squares(
            _residual,
            np.asarray(initial, dtype=float),
            bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
            max_nfev=max(25, int(max_nfev)),
            loss=("soft_l1" if normalized_model == PROFILE_MOSAIC_MIX else "linear"),
            f_scale=1.0,
        )
    except Exception as exc:
        return {
            "success": False,
            "model": normalized_model,
            "label": profile_model_label(normalized_model),
            "error": str(exc),
        }

    params = np.asarray(result.x, dtype=float)
    fitted = _evaluate_composite_profile(
        normalized_model,
        fit_axis_arr,
        params,
        component_count=component_count,
    )
    residual = fitted - fit_curve_arr
    weighted_residual = _tail_weighted_profile_residual(
        fitted,
        fit_curve_arr,
        baseline=baseline0,
        floor=residual_floor,
    )
    fitted_output = _evaluate_composite_profile(
        normalized_model,
        fit_axis_unsorted,
        params,
        component_count=component_count,
    )

    components: list[dict[str, object]] = []
    component_groups = [
        {
            "component_index": int(group["component_index"]),
            "center_guess_indices": [int(index) for index in group.get("center_guess_indices", [])],
        }
        for group in collapsed_groups
    ]
    for component_index, group in enumerate(collapsed_groups):
        offset = 1 + component_index * stride
        display_selected_axis_value = float(group["selected_axis_value"])
        display_center = _display_coordinate_from_fit_value(
            float(params[offset + 1]),
            axis_values=axis_arr,
            reference_value=display_selected_axis_value,
            wrap_period=wrap_period,
        )
        component: dict[str, object] = {
            "component_index": int(component_index),
            "selected_axis_value": display_selected_axis_value,
            "amplitude": float(params[offset]),
            "center": display_center,
        }
        if normalized_model == PROFILE_MOSAIC_MIX:
            sigma_value = float(abs(params[offset + 2]))
            gamma_value = float(abs(params[offset + 3]))
            gaussian_fwhm = float(_GAUSSIAN_FWHM_PER_SIGMA * sigma_value)
            lorentzian_fwhm = float(2.0 * gamma_value)
            component["sigma"] = sigma_value
            component["gamma"] = gamma_value
            component["eta"] = float(np.clip(float(params[offset + 4]), 0.0, 1.0))
            component["gaussian_fwhm"] = gaussian_fwhm
            component["lorentzian_fwhm"] = lorentzian_fwhm
            component["fwhm"] = gaussian_fwhm
        else:
            fwhm_value = float(abs(params[offset + 2]))
            component["fwhm"] = fwhm_value
        if normalized_model == PROFILE_GAUSSIAN:
            component["sigma"] = float(fwhm_value / (2.0 * math.sqrt(2.0 * math.log(2.0))))
        elif normalized_model == PROFILE_LORENTZIAN:
            component["gamma"] = float(0.5 * fwhm_value)
        components.append(component)

    fit_result: dict[str, object] = {
        "success": bool(result.success),
        "model": normalized_model,
        "label": profile_model_label(normalized_model),
        "baseline": float(params[0]),
        "components": components,
        "component_groups": component_groups,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "rss": float(np.sum(residual**2)),
        "weighted_rmse": float(np.sqrt(np.mean(weighted_residual**2))),
        "residual_mode": "hybrid_tail"
        if normalized_model == PROFILE_MOSAIC_MIX
        else "linear",
        "x_fit": np.asarray(axis_arr, dtype=float),
        "y_fit": np.asarray(fitted_output, dtype=float),
        "x_window": np.asarray(axis_arr, dtype=float),
        "y_window": np.asarray(curve_arr, dtype=float),
        "nfev": int(getattr(result, "nfev", 0)),
    }
    if not bool(result.success):
        fit_result["error"] = " ".join(str(getattr(result, "message", "fit failed")).split())
    return fit_result


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
    peak_height0 = max(float(y_window[peak_index]) - baseline0, 0.25 * y_span, 1.0e-6)
    center0 = float(np.clip(float(x_window[peak_index]), lower_bound, upper_bound))

    unique_x = np.unique(np.asarray(x_window, dtype=float))
    if unique_x.size >= 2:
        step = float(np.nanmedian(np.diff(unique_x)))
    else:
        step = abs(float(x_window[-1]) - float(x_window[0])) / max(int(x_window.size) - 1, 1)
    if not np.isfinite(step) or step <= 0.0:
        step = max(
            abs(float(x_window[-1]) - float(x_window[0])) / max(int(x_window.size) - 1, 1), 1.0e-4
        )
    span = max(abs(float(x_window[-1]) - float(x_window[0])), abs(upper_bound - lower_bound), step)
    min_fwhm = max(step * 2.0, span / max(4.0 * float(x_window.size), 1.0), 1.0e-6)
    max_fwhm = max(min_fwhm * 4.0, min(max(span * 1.25, min_fwhm * 4.0), 2.0 * half_width))
    fwhm0 = float(np.clip(max(span / 4.0, min_fwhm * 1.5), min_fwhm, max_fwhm))
    min_sigma = min_fwhm / _GAUSSIAN_FWHM_PER_SIGMA
    sigma_max = max(max_fwhm / _GAUSSIAN_FWHM_PER_SIGMA, min_sigma * 4.0)
    sigma0 = float(np.clip(fwhm0 / _GAUSSIAN_FWHM_PER_SIGMA, min_sigma, sigma_max))
    min_gamma = 0.5 * min_fwhm
    gamma_max = max(min_gamma * 4.0, span)
    gamma0 = float(np.clip(0.5 * fwhm0, min_gamma, gamma_max))
    amplitude0 = _profile_amplitude_from_peak_height(
        normalized_model,
        peak_height0,
        fwhm=fwhm0,
        sigma=sigma0,
        gamma=gamma0,
        eta=0.5,
    )
    widest_lorentzian_area_scale = 1.0 / max(
        _profile_unit_peak_height(
            normalized_model,
            fwhm=max_fwhm,
            sigma=sigma_max,
            gamma=gamma_max,
            eta=1.0,
        ),
        1.0e-12,
    )

    initial = [baseline0, amplitude0, center0, sigma0, gamma0, 0.5]
    lower = [y_min - y_span, 0.0, lower_bound, min_sigma, min_gamma, 0.0]
    upper = [
        y_max + y_span,
        max(4.0 * y_span * widest_lorentzian_area_scale, amplitude0 * 4.0),
        upper_bound,
        sigma_max,
        gamma_max,
        1.0,
    ]
    residual_floor = _tail_weight_floor(y_window, baseline0)

    def _residual(parameters: np.ndarray) -> np.ndarray:
        model_y = evaluate_profile(normalized_model, x_window, parameters)
        return _tail_weighted_profile_residual(
            model_y,
            y_window,
            baseline=baseline0,
            floor=residual_floor,
        )

    try:
        result = least_squares(
            _residual,
            np.asarray(initial, dtype=float),
            bounds=(
                np.asarray(lower, dtype=float),
                np.asarray(upper, dtype=float),
            ),
            max_nfev=max(25, int(max_nfev)),
            loss="soft_l1",
            f_scale=1.0,
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
    weighted_residual = _tail_weighted_profile_residual(
        fitted,
        y_window,
        baseline=baseline0,
        floor=residual_floor,
    )
    sigma_value = float(abs(params[3]))
    gamma_value = float(abs(params[4]))
    eta_value = float(np.clip(float(params[5]), 0.0, 1.0))
    gaussian_fwhm = float(_GAUSSIAN_FWHM_PER_SIGMA * sigma_value)
    lorentzian_fwhm = float(2.0 * gamma_value)
    fit_result: dict[str, object] = {
        "success": bool(result.success),
        "model": normalized_model,
        "label": profile_model_label(normalized_model),
        "baseline": float(params[0]),
        "amplitude": float(params[1]),
        "center": float(params[2]),
        "fwhm": gaussian_fwhm,
        "sigma": sigma_value,
        "gamma": gamma_value,
        "eta": eta_value,
        "gaussian_fwhm": gaussian_fwhm,
        "lorentzian_fwhm": lorentzian_fwhm,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "rss": float(np.sum(residual**2)),
        "weighted_rmse": float(np.sqrt(np.mean(weighted_residual**2))),
        "residual_mode": "hybrid_tail",
        "window_half_width": float(half_width),
        "x_window": np.asarray(x_window, dtype=float),
        "y_window": np.asarray(y_window, dtype=float),
        "y_fit": np.asarray(fitted, dtype=float),
        "nfev": int(getattr(result, "nfev", 0)),
    }
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
        entry.get("label") or profile_model_label(str(entry.get("model", "Fit")))
    ).strip()

    prefix_parts: list[str] = []
    source_name = str(entry.get("source", "") or "").strip().lower()
    try:
        source_peak_index = int(entry.get("source_peak_index", 0))
    except Exception:
        source_peak_index = 0
    if source_peak_index > 0:
        if source_name == "background":
            prefix_parts.append(f"B{source_peak_index}")
        elif source_name == "simulated":
            prefix_parts.append(f"S{source_peak_index}")
        elif source_name == "unknown":
            prefix_parts.append(f"U{source_peak_index}")
    try:
        peak_index = int(entry.get("peak_index", 0))
    except Exception:
        peak_index = 0
    if peak_index > 0 and not prefix_parts:
        prefix_parts.append(f"P{peak_index}")
    if source_name == "unknown" and not any(part.startswith("U") for part in prefix_parts):
        prefix_parts.append("[unknown]")
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


def _fit_entry_id_text(entry: Mapping[str, object]) -> str:
    source_name = str(entry.get("source", "") or "").strip().lower()
    try:
        source_peak_index = int(entry.get("source_peak_index", 0))
    except Exception:
        source_peak_index = 0
    if source_peak_index > 0:
        if source_name == "background":
            return f"B{source_peak_index}"
        if source_name == "simulated":
            return f"S{source_peak_index}"
        if source_name == "unknown":
            return f"U{source_peak_index}"

    try:
        peak_index = int(entry.get("peak_index", 0))
    except Exception:
        peak_index = 0
    if peak_index > 0:
        return f"P{peak_index}"
    if source_name == "unknown":
        return "U"
    return "-"


def _fit_entry_model(entry: Mapping[str, object]) -> str:
    model = str(entry.get("model", "") or "").strip().lower()
    if model in SUPPORTED_PROFILE_MODELS:
        return model
    label = str(entry.get("label", "") or "").strip().lower()
    if "mosaic" in label:
        return PROFILE_MOSAIC_MIX
    if "lorentz" in label:
        return PROFILE_LORENTZIAN
    if "gauss" in label:
        return PROFILE_GAUSSIAN
    return model


def _fit_entry_short_model_label(entry: Mapping[str, object]) -> str:
    model = _fit_entry_model(entry)
    if model == PROFILE_GAUSSIAN:
        return "Gaussian"
    if model == PROFILE_LORENTZIAN:
        return "Lorentz"
    if model == PROFILE_MOSAIC_MIX:
        return "Mosaic"
    label = str(entry.get("label", "") or profile_model_label(model)).strip()
    return label[:8] if label else "-"


def _fit_entry_table_sort_key(entry: Mapping[str, object]) -> tuple[object, ...]:
    source_name = str(entry.get("source", "") or "").strip().lower()
    source_order = {
        "background": 0,
        "simulated": 1,
        "unknown": 2,
    }.get(source_name, 3)
    model_order = {
        PROFILE_MOSAIC_MIX: 0,
        PROFILE_GAUSSIAN: 1,
        PROFILE_LORENTZIAN: 2,
    }.get(_fit_entry_model(entry), 9)
    try:
        source_peak_index = int(entry.get("source_peak_index", 0))
    except Exception:
        source_peak_index = 0
    try:
        peak_index = int(entry.get("peak_index", 0))
    except Exception:
        peak_index = 0
    try:
        axis_value = float(entry.get("selected_axis_value", np.nan))
    except Exception:
        axis_value = float("nan")
    if not np.isfinite(axis_value):
        axis_value = float("inf")
    return (
        int(source_order),
        int(source_peak_index) if source_peak_index > 0 else int(peak_index),
        float(axis_value),
        int(peak_index),
        int(model_order),
    )


def _format_peak_fit_numeric_cell(
    value: object,
    *,
    precision: str = ".4f",
    width: int = 7,
) -> str:
    try:
        numeric_value = float(value)
    except Exception:
        numeric_value = float("nan")
    if not np.isfinite(numeric_value):
        return "-".rjust(width)
    return f"{numeric_value:{width}{precision}}"


def _format_peak_fit_percent_pair(
    gaussian_percent: float,
    lorentzian_percent: float,
) -> str:
    if not (np.isfinite(gaussian_percent) and np.isfinite(lorentzian_percent)):
        return "-".rjust(10)
    return f"{gaussian_percent:4.1f}/{lorentzian_percent:4.1f}"


def _fit_entry_width_mix_cells(entry: Mapping[str, object]) -> tuple[str, str, str]:
    if not bool(entry.get("success", False)):
        return "-".rjust(7), "-".rjust(7), "-".rjust(10)

    model = _fit_entry_model(entry)
    if model == PROFILE_MOSAIC_MIX:
        try:
            gaussian_fwhm = float(entry.get("gaussian_fwhm", np.nan))
        except Exception:
            gaussian_fwhm = float("nan")
        if not np.isfinite(gaussian_fwhm):
            try:
                gaussian_fwhm = _GAUSSIAN_FWHM_PER_SIGMA * float(entry.get("sigma", np.nan))
            except Exception:
                gaussian_fwhm = float("nan")
        if not np.isfinite(gaussian_fwhm):
            try:
                gaussian_fwhm = float(entry.get("fwhm", np.nan))
            except Exception:
                gaussian_fwhm = float("nan")

        try:
            lorentzian_fwhm = float(entry.get("lorentzian_fwhm", np.nan))
        except Exception:
            lorentzian_fwhm = float("nan")
        if not np.isfinite(lorentzian_fwhm):
            try:
                lorentzian_fwhm = 2.0 * float(entry.get("gamma", np.nan))
            except Exception:
                lorentzian_fwhm = float("nan")

        try:
            eta = float(entry.get("eta", np.nan))
        except Exception:
            eta = float("nan")
        if np.isfinite(eta):
            eta = float(np.clip(eta, 0.0, 1.0))
            mix_text = _format_peak_fit_percent_pair((1.0 - eta) * 100.0, eta * 100.0)
        else:
            mix_text = "-".rjust(10)
        return (
            _format_peak_fit_numeric_cell(gaussian_fwhm),
            _format_peak_fit_numeric_cell(lorentzian_fwhm),
            mix_text,
        )

    try:
        fwhm = float(entry.get("fwhm", np.nan))
    except Exception:
        fwhm = float("nan")
    if not np.isfinite(fwhm):
        return "-".rjust(7), "-".rjust(7), "-".rjust(10)

    if model == PROFILE_GAUSSIAN:
        return (
            _format_peak_fit_numeric_cell(fwhm),
            "-".rjust(7),
            _format_peak_fit_percent_pair(100.0, 0.0),
        )
    if model == PROFILE_LORENTZIAN:
        return (
            "-".rjust(7),
            _format_peak_fit_numeric_cell(fwhm),
            _format_peak_fit_percent_pair(0.0, 100.0),
        )
    return _format_peak_fit_numeric_cell(fwhm), "-".rjust(7), "-".rjust(10)


def _format_peak_fit_best_table_note(entry: Mapping[str, object]) -> str:
    entry_id = _fit_entry_id_text(entry)
    model_label = _fit_entry_short_model_label(entry)
    try:
        rmse = float(entry.get("rmse", np.nan))
    except Exception:
        rmse = float("nan")
    rmse_text = f", RMSE {rmse:.4g}" if np.isfinite(rmse) else ""
    return " ".join(part for part in (entry_id, model_label) if part and part != "-") + rmse_text


def _format_peak_fit_table_row(entry: Mapping[str, object]) -> str:
    entry_id = _fit_entry_id_text(entry)[:4]
    model_text = _fit_entry_short_model_label(entry)[:8]
    try:
        axis_value = float(entry.get("selected_axis_value", np.nan))
    except Exception:
        axis_value = float("nan")
    try:
        center_value = float(entry.get("center", np.nan))
    except Exception:
        center_value = float("nan")

    gaussian_fwhm, lorentzian_fwhm, mix_text = _fit_entry_width_mix_cells(entry)
    rmse_text = _format_peak_fit_numeric_cell(entry.get("rmse", np.nan), precision=".4g")
    if not bool(entry.get("success", False)):
        rmse_text = "fail".rjust(7)
    return (
        f"{entry_id:<4} "
        f"{model_text:<8} "
        f"{_format_peak_fit_numeric_cell(axis_value)} "
        f"{_format_peak_fit_numeric_cell(center_value)} "
        f"{gaussian_fwhm} "
        f"{lorentzian_fwhm} "
        f"{mix_text} "
        f"{rmse_text}"
    ).rstrip()


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
