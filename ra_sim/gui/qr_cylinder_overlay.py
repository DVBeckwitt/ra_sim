from __future__ import annotations

import math

import numpy as np


def wrap_caked_phi_degrees(phi_values: np.ndarray | float) -> np.ndarray:
    """Wrap azimuth values into the ``[-180, 180)`` interval."""

    return ((np.asarray(phi_values, dtype=np.float64) + 180.0) % 360.0) - 180.0


def _bilinear_sample(
    image: np.ndarray,
    cols: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    """Sample one 2D image at fractional detector coordinates."""

    arr = np.asarray(image, dtype=np.float64)
    col_arr = np.asarray(cols, dtype=np.float64)
    row_arr = np.asarray(rows, dtype=np.float64)
    out = np.full(col_arr.shape, np.nan, dtype=np.float64)

    if arr.ndim != 2 or col_arr.shape != row_arr.shape or arr.size == 0:
        return out

    height, width = arr.shape
    if height <= 0 or width <= 0:
        return out

    valid = (
        np.isfinite(col_arr)
        & np.isfinite(row_arr)
        & (col_arr >= 0.0)
        & (row_arr >= 0.0)
        & (col_arr <= float(width - 1))
        & (row_arr <= float(height - 1))
    )
    if not np.any(valid):
        return out

    cols_valid = col_arr[valid]
    rows_valid = row_arr[valid]

    col0 = np.floor(cols_valid).astype(np.int64, copy=False)
    row0 = np.floor(rows_valid).astype(np.int64, copy=False)
    col1 = np.clip(col0 + 1, 0, width - 1)
    row1 = np.clip(row0 + 1, 0, height - 1)

    dcol = cols_valid - col0
    drow = rows_valid - row0

    top = (1.0 - dcol) * arr[row0, col0] + dcol * arr[row0, col1]
    bottom = (1.0 - dcol) * arr[row1, col0] + dcol * arr[row1, col1]
    out[valid] = (1.0 - drow) * top + drow * bottom
    return out


def _sample_wrapped_phi_degrees(
    phi_map_deg: np.ndarray,
    cols: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    """Interpolate wrapped azimuths without introducing 180-degree artifacts."""

    phi_wrapped = wrap_caked_phi_degrees(phi_map_deg)
    phi_rad = np.deg2rad(phi_wrapped)
    sin_sample = _bilinear_sample(np.sin(phi_rad), cols, rows)
    cos_sample = _bilinear_sample(np.cos(phi_rad), cols, rows)

    phi_deg = np.full_like(sin_sample, np.nan, dtype=np.float64)
    finite = np.isfinite(sin_sample) & np.isfinite(cos_sample)
    if not np.any(finite):
        return phi_deg

    magnitude = np.hypot(sin_sample[finite], cos_sample[finite])
    good = magnitude > 1e-12
    if not np.any(good):
        return phi_deg

    finite_idx = np.flatnonzero(finite)
    good_idx = finite_idx[good]
    phi_deg[good_idx] = wrap_caked_phi_degrees(
        np.rad2deg(np.arctan2(sin_sample[good_idx], cos_sample[good_idx]))
    )
    return phi_deg


def interpolate_trace_to_caked_coords(
    *,
    detector_cols: np.ndarray,
    detector_rows: np.ndarray,
    valid_mask: np.ndarray | None,
    two_theta_map: np.ndarray,
    phi_map_deg: np.ndarray,
    two_theta_limits: tuple[float, float] = (0.0, 90.0),
    discontinuity_threshold_deg: float = 180.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one detector-space trace into caked ``(2theta, phi)`` coordinates."""

    cols = np.asarray(detector_cols, dtype=np.float64)
    rows = np.asarray(detector_rows, dtype=np.float64)
    tth = np.full(cols.shape, np.nan, dtype=np.float64)
    phi = np.full(cols.shape, np.nan, dtype=np.float64)

    if cols.shape != rows.shape:
        return tth, phi

    base_valid = np.isfinite(cols) & np.isfinite(rows)
    if valid_mask is not None:
        mask_arr = np.asarray(valid_mask, dtype=bool)
        if mask_arr.shape != cols.shape:
            return tth, phi
        base_valid &= mask_arr
    if not np.any(base_valid):
        return tth, phi

    tth_sample = _bilinear_sample(two_theta_map, cols, rows)
    phi_sample = _sample_wrapped_phi_degrees(phi_map_deg, cols, rows)

    visible = base_valid & np.isfinite(tth_sample) & np.isfinite(phi_sample)
    tth_min, tth_max = sorted((float(two_theta_limits[0]), float(two_theta_limits[1])))
    visible &= (tth_sample >= tth_min) & (tth_sample <= tth_max)
    if not np.any(visible):
        return tth, phi

    tth[visible] = tth_sample[visible]
    phi[visible] = phi_sample[visible]

    jump_limit = float(discontinuity_threshold_deg)
    if math.isfinite(jump_limit) and jump_limit > 0.0 and phi.size > 1:
        finite_pairs = (
            np.isfinite(tth[:-1])
            & np.isfinite(tth[1:])
            & np.isfinite(phi[:-1])
            & np.isfinite(phi[1:])
        )
        jumps = finite_pairs & (np.abs(phi[1:] - phi[:-1]) > jump_limit)
        if np.any(jumps):
            jump_idx = np.flatnonzero(jumps) + 1
            tth[jump_idx] = np.nan
            phi[jump_idx] = np.nan

    return tth, phi
