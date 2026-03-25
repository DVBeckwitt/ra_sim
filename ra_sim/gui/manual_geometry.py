"""Pure helpers for manual geometry selection and serialization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

import numpy as np

from ra_sim.gui.geometry_overlay import normalize_hkl_key as _default_normalize_hkl_key
from ra_sim.gui.geometry_overlay import rotate_point_for_display as _default_rotate_point


DEFAULT_POSITION_SIGMA_FLOOR_PX = 0.75
DEFAULT_CAKED_SEARCH_TTH_DEG = 1.5
DEFAULT_CAKED_SEARCH_PHI_DEG = 10.0
DEFAULT_PREVIEW_MIN_INTERVAL_S = 0.03
DEFAULT_PREVIEW_MIN_MOVE_PX = 0.8


def geometry_manual_position_error_px(
    raw_col: float,
    raw_row: float,
    refined_col: float,
    refined_row: float,
) -> float:
    """Return the click-to-refined placement error in display pixels."""

    try:
        delta = float(
            np.hypot(
                float(refined_col) - float(raw_col),
                float(refined_row) - float(raw_row),
            )
        )
    except Exception:
        return 0.0
    if not np.isfinite(delta):
        return 0.0
    return max(0.0, float(delta))


def geometry_manual_position_sigma_px(
    placement_error_px: object,
    *,
    floor_px: float = DEFAULT_POSITION_SIGMA_FLOOR_PX,
) -> float:
    """Convert a manual click-placement error into a fit sigma in pixels."""

    try:
        error_px = float(placement_error_px)
    except Exception:
        error_px = 0.0
    if not np.isfinite(error_px):
        error_px = 0.0

    floor_val = max(1.0e-3, float(floor_px))
    return float(np.hypot(float(error_px), floor_val))


def normalize_geometry_manual_pair_entry(
    entry: dict[str, object] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
    sigma_floor_px: float = DEFAULT_POSITION_SIGMA_FLOOR_PX,
) -> dict[str, object] | None:
    """Normalize one saved manual geometry-pair entry."""

    if not isinstance(entry, dict):
        return None
    try:
        x_val = float(entry.get("x", np.nan))
        y_val = float(entry.get("y", np.nan))
    except Exception:
        return None
    if not (np.isfinite(x_val) and np.isfinite(y_val)):
        return None

    normalized_hkl = normalize_hkl_key(entry.get("hkl", entry.get("label")))
    label = str(entry.get("label", "")) if entry.get("label") is not None else ""
    if not label and normalized_hkl is not None:
        label = f"{normalized_hkl[0]},{normalized_hkl[1]},{normalized_hkl[2]}"

    normalized: dict[str, object] = {
        "x": float(x_val),
        "y": float(y_val),
        "label": label,
    }
    if normalized_hkl is not None:
        normalized["hkl"] = normalized_hkl

    raw_group_key = entry.get("q_group_key")
    if isinstance(raw_group_key, tuple):
        normalized["q_group_key"] = raw_group_key
    elif isinstance(raw_group_key, list):
        normalized["q_group_key"] = tuple(raw_group_key)

    for key in ("source_table_index", "source_row_index"):
        if key not in entry:
            continue
        try:
            normalized[key] = int(entry.get(key))  # type: ignore[arg-type]
        except Exception:
            pass

    if entry.get("source_label") is not None:
        normalized["source_label"] = str(entry.get("source_label"))

    raw_x = entry.get("raw_x")
    raw_y = entry.get("raw_y")
    try:
        raw_x_val = float(raw_x) if raw_x is not None else float("nan")
    except Exception:
        raw_x_val = float("nan")
    try:
        raw_y_val = float(raw_y) if raw_y is not None else float("nan")
    except Exception:
        raw_y_val = float("nan")
    if np.isfinite(raw_x_val) and np.isfinite(raw_y_val):
        normalized["raw_x"] = float(raw_x_val)
        normalized["raw_y"] = float(raw_y_val)

    for x_key, y_key in (("caked_x", "caked_y"), ("raw_caked_x", "raw_caked_y")):
        raw_x_local = entry.get(x_key)
        raw_y_local = entry.get(y_key)
        try:
            caked_x_val = (
                float(raw_x_local) if raw_x_local is not None else float("nan")
            )
        except Exception:
            caked_x_val = float("nan")
        try:
            caked_y_val = (
                float(raw_y_local) if raw_y_local is not None else float("nan")
            )
        except Exception:
            caked_y_val = float("nan")
        if np.isfinite(caked_x_val) and np.isfinite(caked_y_val):
            normalized[x_key] = float(caked_x_val)
            normalized[y_key] = float(caked_y_val)

    placement_error_value = entry.get("placement_error_px")
    if placement_error_value is None and np.isfinite(raw_x_val) and np.isfinite(raw_y_val):
        placement_error_value = geometry_manual_position_error_px(
            float(raw_x_val),
            float(raw_y_val),
            float(x_val),
            float(y_val),
        )
    try:
        placement_error_px = (
            float(placement_error_value)
            if placement_error_value is not None
            else float("nan")
        )
    except Exception:
        placement_error_px = float("nan")
    if np.isfinite(placement_error_px):
        normalized["placement_error_px"] = max(0.0, float(placement_error_px))

    sigma_value = (
        entry.get("sigma_px")
        if entry.get("sigma_px") is not None
        else entry.get("position_sigma_px", entry.get("measurement_sigma_px"))
    )
    if sigma_value is None and np.isfinite(placement_error_px):
        sigma_value = geometry_manual_position_sigma_px(
            float(placement_error_px),
            floor_px=sigma_floor_px,
        )
    try:
        sigma_px = float(sigma_value) if sigma_value is not None else float("nan")
    except Exception:
        sigma_px = float("nan")
    if np.isfinite(sigma_px) and sigma_px > 0.0:
        normalized["sigma_px"] = float(sigma_px)

    return normalized


def geometry_manual_pairs_for_index(
    index: int,
    *,
    pairs_by_background: dict[int, list[dict[str, object]]],
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
    sigma_floor_px: float = DEFAULT_POSITION_SIGMA_FLOOR_PX,
) -> list[dict[str, object]]:
    """Return normalized saved manual geometry pairs for one background index."""

    try:
        key = int(index)
    except Exception:
        return []
    raw_entries = pairs_by_background.get(key, [])
    normalized_entries: list[dict[str, object]] = []
    for raw_entry in raw_entries:
        normalized = normalize_geometry_manual_pair_entry(
            raw_entry,
            normalize_hkl_key=normalize_hkl_key,
            sigma_floor_px=sigma_floor_px,
        )
        if normalized is not None:
            normalized_entries.append(normalized)
    return normalized_entries


def set_geometry_manual_pairs_for_index(
    index: int,
    entries: Sequence[dict[str, object]] | None,
    *,
    pairs_by_background: dict[int, list[dict[str, object]]],
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
    sigma_floor_px: float = DEFAULT_POSITION_SIGMA_FLOOR_PX,
) -> list[dict[str, object]]:
    """Replace one background's saved manual geometry-pair list."""

    try:
        key = int(index)
    except Exception:
        return []

    normalized_entries: list[dict[str, object]] = []
    for raw_entry in entries or []:
        normalized = normalize_geometry_manual_pair_entry(
            raw_entry,
            normalize_hkl_key=normalize_hkl_key,
            sigma_floor_px=sigma_floor_px,
        )
        if normalized is not None:
            normalized_entries.append(normalized)

    if normalized_entries:
        pairs_by_background[key] = normalized_entries
    else:
        pairs_by_background.pop(key, None)
    return list(normalized_entries)


def geometry_manual_pair_group_count(
    index: int,
    *,
    pairs_by_background: dict[int, list[dict[str, object]]],
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
    sigma_floor_px: float = DEFAULT_POSITION_SIGMA_FLOOR_PX,
) -> int:
    """Return how many distinct Qr/Qz groups are saved for one background."""

    group_keys = {
        entry.get("q_group_key")
        for entry in geometry_manual_pairs_for_index(
            index,
            pairs_by_background=pairs_by_background,
            normalize_hkl_key=normalize_hkl_key,
            sigma_floor_px=sigma_floor_px,
        )
        if entry.get("q_group_key") is not None
    }
    return int(len(group_keys))


def peak_maximum_near_in_image(
    image: np.ndarray | None,
    col: float,
    row: float,
    *,
    search_radius: int = 5,
) -> tuple[float, float]:
    """Return the brightest local pixel near ``(col, row)`` in display coordinates."""

    if image is None:
        return float(col), float(row)
    try:
        image_arr = np.asarray(image, dtype=float)
    except Exception:
        return float(col), float(row)
    if image_arr.ndim < 2 or image_arr.size == 0:
        return float(col), float(row)

    r = int(round(float(row)))
    c = int(round(float(col)))
    r0 = max(0, r - int(search_radius))
    r1 = min(int(image_arr.shape[0]), r + int(search_radius) + 1)
    c0 = max(0, c - int(search_radius))
    c1 = min(int(image_arr.shape[1]), c + int(search_radius) + 1)

    window = image_arr[r0:r1, c0:c1]
    if window.size == 0 or not np.isfinite(window).any():
        return float(col), float(row)

    max_idx = int(np.nanargmax(window))
    win_r, win_c = np.unravel_index(max_idx, window.shape)
    return float(c0 + win_c), float(r0 + win_r)


def caked_axis_to_image_index(
    value: float,
    axis_values: Sequence[float] | None,
) -> float:
    """Map one caked-axis coordinate in degrees to a floating image index."""

    if axis_values is None or not np.isfinite(value):
        return float("nan")
    axis_arr = np.asarray(axis_values, dtype=float).reshape(-1)
    if axis_arr.size <= 0:
        return float("nan")
    finite_idx = np.flatnonzero(np.isfinite(axis_arr))
    if finite_idx.size <= 0:
        return float("nan")
    if finite_idx.size == 1:
        return float(finite_idx[0])
    axis_used = axis_arr[finite_idx]
    idx_used = finite_idx.astype(float)
    if axis_used[0] > axis_used[-1]:
        axis_used = axis_used[::-1]
        idx_used = idx_used[::-1]
    return float(np.interp(float(value), axis_used, idx_used))


def caked_image_index_to_axis(
    index_value: float,
    axis_values: Sequence[float] | None,
) -> float:
    """Map one floating caked image index back to axis-space degrees."""

    if axis_values is None or not np.isfinite(index_value):
        return float("nan")
    axis_arr = np.asarray(axis_values, dtype=float).reshape(-1)
    if axis_arr.size <= 0:
        return float("nan")
    finite_idx = np.flatnonzero(np.isfinite(axis_arr))
    if finite_idx.size <= 0:
        return float("nan")
    if finite_idx.size == 1:
        return float(axis_arr[finite_idx[0]])
    axis_used = axis_arr[finite_idx]
    idx_used = finite_idx.astype(float)
    if axis_used[0] > axis_used[-1]:
        axis_used = axis_used[::-1]
        idx_used = idx_used[::-1]
    return float(np.interp(float(index_value), idx_used, axis_used))


def refine_profile_peak_index(
    profile: Sequence[float] | None,
    seed_index: float,
) -> float:
    """Return one subpixel 1D peak center focused on the top of a local profile."""

    arr = np.asarray(profile, dtype=float).reshape(-1)
    if arr.size <= 0:
        return float(seed_index)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float(seed_index)
    arr = np.where(finite, arr, np.nan)
    baseline = float(np.nanpercentile(arr, 35.0))
    weights = np.clip(arr - baseline, 0.0, None)
    if not np.any(weights > 0.0):
        floor = float(np.nanmin(arr))
        weights = np.clip(arr - floor, 0.0, None)
    if not np.any(weights > 0.0):
        return float(np.nanargmax(arr))

    peak_idx = int(np.nanargmax(weights))
    half_window = 2
    lo = max(0, peak_idx - half_window)
    hi = min(arr.size, peak_idx + half_window + 1)
    local_weights = np.asarray(weights[lo:hi], dtype=float)
    local_idx = np.arange(lo, hi, dtype=float)
    if local_weights.size <= 0 or not np.any(local_weights > 0.0):
        return float(peak_idx)
    crest_mask = local_weights >= 0.5 * float(np.max(local_weights))
    if np.any(crest_mask):
        local_weights = local_weights[crest_mask]
        local_idx = local_idx[crest_mask]
    total = float(np.sum(local_weights))
    if not np.isfinite(total) or total <= 0.0:
        return float(peak_idx)
    return float(np.sum(local_weights * local_idx) / total)


def refine_caked_peak_center(
    image: np.ndarray | None,
    radial_axis: Sequence[float] | None,
    azimuth_axis: Sequence[float] | None,
    two_theta_deg: float,
    phi_deg: float,
    *,
    tth_window_deg: float | None = None,
    phi_window_deg: float | None = None,
    default_tth_window_deg: float = DEFAULT_CAKED_SEARCH_TTH_DEG,
    default_phi_window_deg: float = DEFAULT_CAKED_SEARCH_PHI_DEG,
) -> tuple[float, float]:
    """Refine one caked click to the crest of the local 2theta/phi ridge."""

    if image is None:
        return float(two_theta_deg), float(phi_deg)
    img = np.asarray(image, dtype=float)
    if img.ndim != 2 or img.size <= 0:
        return float(two_theta_deg), float(phi_deg)
    if radial_axis is None or azimuth_axis is None:
        return float(two_theta_deg), float(phi_deg)

    col_seed = caked_axis_to_image_index(float(two_theta_deg), radial_axis)
    row_seed = caked_axis_to_image_index(float(phi_deg), azimuth_axis)
    if not (np.isfinite(col_seed) and np.isfinite(row_seed)):
        return float(two_theta_deg), float(phi_deg)

    tth_window = float(default_tth_window_deg if tth_window_deg is None else tth_window_deg)
    phi_window = float(default_phi_window_deg if phi_window_deg is None else phi_window_deg)
    col_min = caked_axis_to_image_index(float(two_theta_deg) - tth_window, radial_axis)
    col_max = caked_axis_to_image_index(float(two_theta_deg) + tth_window, radial_axis)
    row_min = caked_axis_to_image_index(float(phi_deg) - phi_window, azimuth_axis)
    row_max = caked_axis_to_image_index(float(phi_deg) + phi_window, azimuth_axis)
    if not all(np.isfinite(v) for v in (col_min, col_max, row_min, row_max)):
        return float(two_theta_deg), float(phi_deg)

    c0 = max(0, int(np.floor(min(col_min, col_max))))
    c1 = min(int(img.shape[1]), int(np.ceil(max(col_min, col_max))) + 1)
    r0 = max(0, int(np.floor(min(row_min, row_max))))
    r1 = min(int(img.shape[0]), int(np.ceil(max(row_min, row_max))) + 1)
    if c0 >= c1 or r0 >= r1:
        return float(two_theta_deg), float(phi_deg)

    patch = np.asarray(img[r0:r1, c0:c1], dtype=float)
    if patch.size <= 0 or not np.isfinite(patch).any():
        return float(two_theta_deg), float(phi_deg)
    baseline = float(np.nanpercentile(patch, 35.0))
    signal = np.clip(patch - baseline, 0.0, None)
    if not np.any(signal > 0.0):
        signal = np.clip(patch - float(np.nanmin(patch)), 0.0, None)
    if not np.any(signal > 0.0):
        return float(two_theta_deg), float(phi_deg)

    col_local = float(col_seed - c0)
    row_local = float(row_seed - r0)
    row_band = max(1, min(6, int(round(0.10 * signal.shape[0]))))
    col_band = max(1, min(6, int(round(0.10 * signal.shape[1]))))
    for _ in range(2):
        row_center = int(np.clip(round(row_local), 0, max(signal.shape[0] - 1, 0)))
        rr0 = max(0, row_center - row_band)
        rr1 = min(signal.shape[0], row_center + row_band + 1)
        radial_profile = np.nansum(signal[rr0:rr1, :], axis=0)
        col_local = refine_profile_peak_index(radial_profile, col_local)

        col_center = int(np.clip(round(col_local), 0, max(signal.shape[1] - 1, 0)))
        cc0 = max(0, col_center - col_band)
        cc1 = min(signal.shape[1], col_center + col_band + 1)
        az_profile = np.nansum(signal[:, cc0:cc1], axis=1)
        row_local = refine_profile_peak_index(az_profile, row_local)

    refined_col = float(c0 + col_local)
    refined_row = float(r0 + row_local)
    return (
        float(caked_image_index_to_axis(refined_col, radial_axis)),
        float(caked_image_index_to_axis(refined_row, azimuth_axis)),
    )


def geometry_manual_candidate_source_key(
    entry: dict[str, object] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
) -> tuple[object, ...] | None:
    """Return a stable lookup key for one manual-pick candidate or match."""

    if not isinstance(entry, dict):
        return None
    try:
        return (
            "source",
            int(entry.get("source_table_index")),
            int(entry.get("source_row_index")),
        )
    except Exception:
        pass
    normalized_hkl = normalize_hkl_key(entry.get("hkl", entry.get("label")))
    if normalized_hkl is not None:
        return ("hkl",) + tuple(int(v) for v in normalized_hkl)
    label = str(entry.get("label", "")).strip()
    if label:
        return ("label", label)
    return None


def current_geometry_manual_match_config(
    fit_config: dict[str, object] | None,
) -> dict[str, object]:
    """Return the refined background-peak matcher config for manual picking."""

    geometry_refine_cfg = fit_config.get("geometry", {}) if isinstance(fit_config, dict) else {}
    if not isinstance(geometry_refine_cfg, dict):
        geometry_refine_cfg = {}
    auto_match_cfg = geometry_refine_cfg.get("auto_match", {}) or {}
    if not isinstance(auto_match_cfg, dict):
        auto_match_cfg = {}

    manual_cfg = dict(auto_match_cfg)
    search_radius = max(1.0, float(manual_cfg.get("search_radius_px", 24.0)))
    manual_cfg["console_progress"] = False
    manual_cfg["relax_on_low_matches"] = False
    manual_cfg.setdefault("context_margin_px", max(96.0, 6.0 * search_radius))
    manual_cfg.setdefault("require_candidate_ownership", True)
    manual_cfg.setdefault("k_neighbors", 12)
    manual_cfg.setdefault("max_candidate_peaks", 1200)
    return manual_cfg


def geometry_manual_choose_group_at(
    grouped_candidates: dict[tuple[object, ...], list[dict[str, object]]] | None,
    col: float,
    row: float,
    *,
    window_size_px: float,
) -> tuple[tuple[object, ...] | None, list[dict[str, object]], float]:
    """Return the nearest clickable Qr/Qz group inside a local click window."""

    best_group_key = None
    best_group_entries: list[dict[str, object]] = []
    best_d2 = float("inf")
    half_window = max(1.0, 0.5 * float(window_size_px))
    for group_key, candidate_entries in (grouped_candidates or {}).items():
        for candidate in candidate_entries or []:
            try:
                sim_col = float(candidate.get("sim_col"))
                sim_row = float(candidate.get("sim_row"))
            except Exception:
                continue
            if not (np.isfinite(sim_col) and np.isfinite(sim_row)):
                continue
            if (
                abs(sim_col - float(col)) > half_window
                or abs(sim_row - float(row)) > half_window
            ):
                continue
            d2 = (sim_col - float(col)) ** 2 + (sim_row - float(row)) ** 2
            if d2 < best_d2:
                best_d2 = float(d2)
                best_group_key = group_key
                best_group_entries = [dict(entry) for entry in candidate_entries]

    if best_group_key is None or not np.isfinite(best_d2):
        return None, [], float("nan")
    return best_group_key, best_group_entries, float(np.sqrt(best_d2))


def geometry_manual_zoom_bounds(
    col: float,
    row: float,
    image_shape: Sequence[int] | None,
    *,
    window_size_px: float = 100.0,
) -> tuple[float, float, float, float]:
    """Return clamped image-space bounds for a square manual-pick zoom window."""

    try:
        height = int(image_shape[0]) if image_shape is not None else 0
        width = int(image_shape[1]) if image_shape is not None else 0
    except Exception:
        height = 0
        width = 0
    width = max(1, width)
    height = max(1, height)
    half = max(1.0, 0.5 * float(window_size_px))

    x_min = float(col) - half
    x_max = float(col) + half
    y_min = float(row) - half
    y_max = float(row) + half

    if x_min < 0.0:
        x_max = min(float(width), x_max - x_min)
        x_min = 0.0
    if x_max > float(width):
        x_min = max(0.0, x_min - (x_max - float(width)))
        x_max = float(width)
    if y_min < 0.0:
        y_max = min(float(height), y_max - y_min)
        y_min = 0.0
    if y_max > float(height):
        y_min = max(0.0, y_min - (y_max - float(height)))
        y_max = float(height)

    return float(x_min), float(x_max), float(y_min), float(y_max)


def geometry_manual_anchor_axis_limits(
    value: float,
    span: float,
    anchor_fraction: float,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float]:
    """Return clamped axis limits that keep *value* at a fixed screen fraction."""

    try:
        span_signed = float(span)
    except Exception:
        span_signed = 0.0
    if not np.isfinite(span_signed) or abs(span_signed) <= 1.0e-12:
        value_f = float(value)
        return value_f, value_f

    try:
        frac = float(anchor_fraction)
    except Exception:
        frac = 0.5
    if not np.isfinite(frac):
        frac = 0.5
    frac = min(max(frac, 0.0), 1.0)

    try:
        bound_lo = float(min(lower_bound, upper_bound))
        bound_hi = float(max(lower_bound, upper_bound))
    except Exception:
        bound_lo = float(lower_bound)
        bound_hi = float(upper_bound)
    available_span = max(0.0, bound_hi - bound_lo)
    if available_span > 0.0:
        span_abs = min(abs(span_signed), available_span)
        span_signed = np.copysign(span_abs, span_signed)

    start = float(value) - frac * span_signed
    end = start + span_signed
    low = min(start, end)
    high = max(start, end)
    if low < bound_lo:
        shift = bound_lo - low
        start += shift
        end += shift
    if high > bound_hi:
        shift = high - bound_hi
        start -= shift
        end -= shift
    return float(start), float(end)


def geometry_manual_group_target_count(
    group_key: tuple[object, ...] | None,
    group_entries: Sequence[dict[str, object]] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
) -> int:
    """Return how many manual background peaks a selected group should collect."""

    if isinstance(group_key, tuple) and len(group_key) >= 4:
        try:
            if int(group_key[2]) == 0:
                return 1
        except Exception:
            pass

    entries = [dict(entry) for entry in group_entries or [] if isinstance(entry, dict)]
    if not entries:
        return 0

    all_00l = True
    for entry in entries:
        hkl = normalize_hkl_key(entry.get("hkl", entry.get("label")))
        if hkl is None or int(hkl[0]) != 0 or int(hkl[1]) != 0:
            all_00l = False
            break
    if all_00l:
        return 1
    return int(len(entries))


def geometry_manual_pick_session_active(
    pick_session: dict[str, object] | None,
    *,
    current_background_index: object = None,
    require_current_background: bool = True,
) -> bool:
    """Return whether a manual background-placement session is in progress."""

    if not isinstance(pick_session, dict):
        return False
    if pick_session.get("group_key") is None:
        return False
    if not isinstance(pick_session.get("group_entries"), list):
        return False
    if require_current_background:
        try:
            return int(pick_session.get("background_index")) == int(current_background_index)
        except Exception:
            return False
    return True


def geometry_manual_unassigned_group_candidates(
    pick_session: dict[str, object] | None,
    *,
    current_background_index: object = None,
    require_current_background: bool = True,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
) -> list[dict[str, object]]:
    """Return manual-pick group candidates that do not yet have a BG assignment."""

    if not geometry_manual_pick_session_active(
        pick_session,
        current_background_index=current_background_index,
        require_current_background=require_current_background,
    ):
        return []
    if pick_session is None:
        return []

    group_entries = pick_session.get("group_entries", [])
    pending_entries = pick_session.get("pending_entries", [])
    if not isinstance(group_entries, list) or not isinstance(pending_entries, list):
        return []
    assigned_keys = {
        candidate_source_key(entry)
        for entry in pending_entries
        if isinstance(entry, dict)
    }
    out: list[dict[str, object]] = []
    for raw_entry in group_entries:
        if not isinstance(raw_entry, dict):
            continue
        source_key = candidate_source_key(raw_entry)
        if source_key in assigned_keys:
            continue
        out.append(dict(raw_entry))
    return out


def geometry_manual_current_pending_candidate(
    pick_session: dict[str, object] | None,
    *,
    current_background_index: object = None,
    require_current_background: bool = True,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
) -> dict[str, object] | None:
    """Return one remaining simulated peak awaiting a manual background click."""

    remaining = geometry_manual_unassigned_group_candidates(
        pick_session,
        current_background_index=current_background_index,
        require_current_background=require_current_background,
        candidate_source_key=candidate_source_key,
    )
    if not remaining:
        return None
    return dict(remaining[0])


def geometry_manual_nearest_candidate_to_point(
    col: float,
    row: float,
    candidate_entries: Sequence[dict[str, object]] | None,
) -> tuple[dict[str, object] | None, float]:
    """Return the nearest simulated candidate to one display-space point."""

    best_entry = None
    best_d2 = float("inf")
    for raw_entry in candidate_entries or []:
        if not isinstance(raw_entry, dict):
            continue
        try:
            sim_col = float(raw_entry.get("sim_col"))
            sim_row = float(raw_entry.get("sim_row"))
        except Exception:
            continue
        if not (np.isfinite(sim_col) and np.isfinite(sim_row)):
            continue
        d2 = (float(sim_col) - float(col)) ** 2 + (float(sim_row) - float(row)) ** 2
        if d2 < best_d2:
            best_d2 = float(d2)
            best_entry = dict(raw_entry)
    if best_entry is None or not np.isfinite(best_d2):
        return None, float("nan")
    return best_entry, float(np.sqrt(best_d2))


def geometry_manual_pair_entry_from_candidate(
    candidate: dict[str, object] | None,
    peak_col: float,
    peak_row: float,
    *,
    group_key: tuple[object, ...] | None,
    raw_col: float | None = None,
    raw_row: float | None = None,
    caked_col: float | None = None,
    caked_row: float | None = None,
    raw_caked_col: float | None = None,
    raw_caked_row: float | None = None,
    placement_error_px: float | None = None,
    sigma_px: float | None = None,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
) -> dict[str, object] | None:
    """Build one saved manual pair entry from a candidate + measured background point."""

    if not isinstance(candidate, dict):
        return None
    entry: dict[str, object] = {
        "label": str(candidate.get("label", "")),
        "hkl": normalize_hkl_key(candidate.get("hkl", candidate.get("label"))),
        "x": float(peak_col),
        "y": float(peak_row),
        "source_table_index": candidate.get("source_table_index"),
        "source_row_index": candidate.get("source_row_index"),
        "source_label": candidate.get("source_label"),
        "q_group_key": group_key,
    }
    if raw_col is not None and raw_row is not None:
        entry["raw_x"] = float(raw_col)
        entry["raw_y"] = float(raw_row)
    if caked_col is not None and caked_row is not None:
        entry["caked_x"] = float(caked_col)
        entry["caked_y"] = float(caked_row)
    if raw_caked_col is not None and raw_caked_row is not None:
        entry["raw_caked_x"] = float(raw_caked_col)
        entry["raw_caked_y"] = float(raw_caked_row)
    if placement_error_px is not None and np.isfinite(float(placement_error_px)):
        entry["placement_error_px"] = max(0.0, float(placement_error_px))
    if sigma_px is not None and np.isfinite(float(sigma_px)) and float(sigma_px) > 0.0:
        entry["sigma_px"] = float(sigma_px)
    return entry


def geometry_manual_preview_due(
    col: float,
    row: float,
    *,
    pick_session: dict[str, object] | None,
    current_background_index: object,
    min_interval_s: float = DEFAULT_PREVIEW_MIN_INTERVAL_S,
    min_move_px: float = DEFAULT_PREVIEW_MIN_MOVE_PX,
    perf_counter_fn: Callable[[], float] = perf_counter,
) -> bool:
    """Throttle manual-placement preview updates during mouse motion."""

    if not geometry_manual_pick_session_active(
        pick_session,
        current_background_index=current_background_index,
    ):
        return False
    if pick_session is None:
        return False

    now = float(perf_counter_fn())
    last_t = float(pick_session.get("preview_last_t", 0.0))
    last_xy = pick_session.get("preview_last_xy")
    due = False
    if not (isinstance(last_xy, tuple) and len(last_xy) >= 2):
        due = True
    else:
        dx = float(col) - float(last_xy[0])
        dy = float(row) - float(last_xy[1])
        if (dx * dx + dy * dy) >= float(min_move_px * min_move_px):
            due = True
    if not due and (now - last_t) >= float(min_interval_s):
        due = True
    if not due:
        return False
    pick_session["preview_last_t"] = float(now)
    pick_session["preview_last_xy"] = (float(col), float(row))
    return True


def geometry_manual_refine_preview_point(
    candidate: dict[str, object] | None,
    raw_col: float,
    raw_row: float,
    *,
    display_background: np.ndarray | None = None,
    cache_data: dict[str, object] | None = None,
    build_cache_data: Callable[[], dict[str, object]] | None = None,
    use_caked_space: bool,
    radial_axis: Sequence[float] | None = None,
    azimuth_axis: Sequence[float] | None = None,
    match_simulated_peaks_to_peak_context: Callable[
        [Sequence[dict[str, object]], dict[str, object], dict[str, object]],
        tuple[Sequence[dict[str, object]], object],
    ] | None = None,
    peak_maximum_near_in_image_fn: Callable[[np.ndarray | None, float, float], tuple[float, float]] = peak_maximum_near_in_image,
    caked_axis_to_image_index_fn: Callable[[float, Sequence[float] | None], float] = caked_axis_to_image_index,
    caked_image_index_to_axis_fn: Callable[[float, Sequence[float] | None], float] = caked_image_index_to_axis,
    refine_caked_peak_center_fn: Callable[
        [np.ndarray | None, Sequence[float] | None, Sequence[float] | None, float, float],
        tuple[float, float],
    ] = refine_caked_peak_center,
) -> tuple[float, float]:
    """Refine one manual raw click/release position to the best background peak."""

    background_local = display_background
    if background_local is None:
        return float(raw_col), float(raw_row)

    state = cache_data if isinstance(cache_data, dict) else {}
    if not state and callable(build_cache_data):
        try:
            built_state = build_cache_data()
        except Exception:
            built_state = {}
        if isinstance(built_state, dict):
            state = built_state

    match_cfg = dict(state.get("match_config", {})) if isinstance(state, dict) else {}
    background_context = state.get("background_context") if isinstance(state, dict) else None
    fallback_radius = max(
        3,
        int(round(min(8.0, 0.33 * float(match_cfg.get("search_radius_px", 18.0))))),
    )

    refined_col = float(raw_col)
    refined_row = float(raw_row)
    used_peak_context = False
    if use_caked_space:
        radial_axis_arr = np.asarray(radial_axis, dtype=float)
        azimuth_axis_arr = np.asarray(azimuth_axis, dtype=float)
        raw_col_local = caked_axis_to_image_index_fn(float(raw_col), radial_axis_arr)
        raw_row_local = caked_axis_to_image_index_fn(float(raw_row), azimuth_axis_arr)
        if (
            isinstance(background_context, dict)
            and bool(background_context.get("img_valid", False))
            and callable(match_simulated_peaks_to_peak_context)
            and np.isfinite(raw_col_local)
            and np.isfinite(raw_row_local)
        ):
            seed_entry = dict(candidate) if isinstance(candidate, dict) else {}
            seed_entry["sim_col"] = float(raw_col)
            seed_entry["sim_row"] = float(raw_row)
            seed_entry["sim_col_global"] = float(raw_col)
            seed_entry["sim_row_global"] = float(raw_row)
            seed_entry["sim_col_local"] = float(raw_col_local)
            seed_entry["sim_row_local"] = float(raw_row_local)
            try:
                manual_matches, _manual_stats = match_simulated_peaks_to_peak_context(
                    [seed_entry],
                    background_context,
                    match_cfg,
                )
            except Exception:
                manual_matches = []
            if manual_matches:
                try:
                    refined_col = float(
                        caked_image_index_to_axis_fn(
                            float(manual_matches[0].get("x", raw_col_local)),
                            radial_axis_arr,
                        )
                    )
                    refined_row = float(
                        caked_image_index_to_axis_fn(
                            float(manual_matches[0].get("y", raw_row_local)),
                            azimuth_axis_arr,
                        )
                    )
                    used_peak_context = np.isfinite(refined_col) and np.isfinite(refined_row)
                except Exception:
                    refined_col = float(raw_col)
                    refined_row = float(raw_row)
        if not used_peak_context or not (np.isfinite(refined_col) and np.isfinite(refined_row)):
            refined_col, refined_row = refine_caked_peak_center_fn(
                np.asarray(background_local, dtype=float),
                radial_axis_arr,
                azimuth_axis_arr,
                float(raw_col),
                float(raw_row),
            )
        return float(refined_col), float(refined_row)

    if (
        isinstance(background_context, dict)
        and bool(background_context.get("img_valid", False))
        and callable(match_simulated_peaks_to_peak_context)
    ):
        seed_entry = dict(candidate) if isinstance(candidate, dict) else {}
        seed_entry["sim_col"] = float(raw_col)
        seed_entry["sim_row"] = float(raw_row)
        try:
            manual_matches, _manual_stats = match_simulated_peaks_to_peak_context(
                [seed_entry],
                background_context,
                match_cfg,
            )
        except Exception:
            manual_matches = []
        if manual_matches:
            try:
                refined_col = float(manual_matches[0].get("x", refined_col))
                refined_row = float(manual_matches[0].get("y", refined_row))
                used_peak_context = np.isfinite(refined_col) and np.isfinite(refined_row)
            except Exception:
                refined_col = float(raw_col)
                refined_row = float(raw_row)
    if not used_peak_context or not (np.isfinite(refined_col) and np.isfinite(refined_row)):
        refined_col, refined_row = peak_maximum_near_in_image_fn(
            background_local,
            float(raw_col),
            float(raw_row),
            search_radius=fallback_radius,
        )
    return float(refined_col), float(refined_row)


def restore_geometry_manual_pick_view(
    pick_session: dict[str, object] | None,
    *,
    axis: Any,
    canvas: Any = None,
    redraw: bool = True,
) -> bool:
    """Restore the pre-zoom axis view for manual background placement."""

    if not isinstance(pick_session, dict):
        return False
    if not bool(pick_session.get("zoom_active", False)):
        return False
    saved_xlim = pick_session.get("saved_xlim")
    saved_ylim = pick_session.get("saved_ylim")
    if isinstance(saved_xlim, tuple) and len(saved_xlim) == 2:
        axis.set_xlim(float(saved_xlim[0]), float(saved_xlim[1]))
    if isinstance(saved_ylim, tuple) and len(saved_ylim) == 2:
        axis.set_ylim(float(saved_ylim[0]), float(saved_ylim[1]))
    pick_session["zoom_active"] = False
    pick_session["zoom_center"] = None
    pick_session["saved_xlim"] = None
    pick_session["saved_ylim"] = None
    if redraw and canvas is not None:
        canvas.draw_idle()
    return True


def apply_geometry_manual_pick_zoom(
    pick_session: dict[str, object] | None,
    col: float,
    row: float,
    *,
    display_background: np.ndarray | None,
    axis: Any,
    canvas: Any = None,
    use_caked_space: bool,
    last_caked_extent: Sequence[float] | None = None,
    caked_zoom_tth_deg: float,
    caked_zoom_phi_deg: float,
    pick_zoom_window_px: float,
    anchor_fraction_x: float = 0.5,
    anchor_fraction_y: float = 0.5,
    anchor_axis_limits_fn: Callable[[float, float, float, float, float], tuple[float, float]] = geometry_manual_anchor_axis_limits,
) -> bool:
    """Zoom to a fixed local window while the user is placing manual points."""

    if not geometry_manual_pick_session_active(pick_session, require_current_background=False):
        return False
    if pick_session is None or display_background is None:
        return False

    try:
        current_xlim = tuple(float(v) for v in axis.get_xlim())
    except Exception:
        current_xlim = (0.0, 1.0)
    try:
        current_ylim = tuple(float(v) for v in axis.get_ylim())
    except Exception:
        current_ylim = (0.0, 1.0)
    x_sign = 1.0 if len(current_xlim) < 2 or current_xlim[1] >= current_xlim[0] else -1.0
    y_sign = 1.0 if len(current_ylim) < 2 or current_ylim[1] >= current_ylim[0] else -1.0

    if use_caked_space:
        if last_caked_extent is not None and len(last_caked_extent) >= 4:
            x_lo = float(last_caked_extent[0])
            x_hi = float(last_caked_extent[1])
            y_lo = float(last_caked_extent[2])
            y_hi = float(last_caked_extent[3])
        else:
            x_lo, x_hi = sorted(current_xlim)
            y_lo, y_hi = sorted(current_ylim)
        x_span = x_sign * min(abs(float(caked_zoom_tth_deg)), abs(float(x_hi) - float(x_lo)))
        y_span = y_sign * min(abs(float(caked_zoom_phi_deg)), abs(float(y_hi) - float(y_lo)))
        x_min, x_max = anchor_axis_limits_fn(
            float(col),
            float(x_span),
            float(anchor_fraction_x),
            float(x_lo),
            float(x_hi),
        )
        y_min, y_max = anchor_axis_limits_fn(
            float(row),
            float(y_span),
            float(anchor_fraction_y),
            float(y_lo),
            float(y_hi),
        )
        if x_min == x_max or y_min == y_max:
            return False
        if not bool(pick_session.get("zoom_active", False)):
            pick_session["saved_xlim"] = tuple(float(v) for v in axis.get_xlim())
            pick_session["saved_ylim"] = tuple(float(v) for v in axis.get_ylim())
        pick_session["zoom_active"] = True
        pick_session["zoom_center"] = (float(col), float(row))
        axis.set_xlim(float(x_min), float(x_max))
        axis.set_ylim(float(y_min), float(y_max))
        if canvas is not None:
            canvas.draw_idle()
        return True

    background_shape = np.asarray(display_background).shape
    height = max(1.0, float(background_shape[0]))
    width = max(1.0, float(background_shape[1]))
    x_span = x_sign * min(float(pick_zoom_window_px), width)
    y_span = y_sign * min(float(pick_zoom_window_px), height)
    x_min, x_max = anchor_axis_limits_fn(
        float(col),
        float(x_span),
        float(anchor_fraction_x),
        0.0,
        float(width),
    )
    y_min, y_max = anchor_axis_limits_fn(
        float(row),
        float(y_span),
        float(anchor_fraction_y),
        0.0,
        float(height),
    )
    if not bool(pick_session.get("zoom_active", False)):
        pick_session["saved_xlim"] = tuple(float(v) for v in axis.get_xlim())
        pick_session["saved_ylim"] = tuple(float(v) for v in axis.get_ylim())
    pick_session["zoom_active"] = True
    pick_session["zoom_center"] = (float(col), float(row))
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    if canvas is not None:
        canvas.draw_idle()
    return True


def geometry_manual_pick_preview_state(
    raw_col: float,
    raw_row: float,
    *,
    pick_session: dict[str, object] | None,
    current_background_index: object,
    force: bool = False,
    remaining_candidates: Sequence[dict[str, object]] | None,
    display_background: np.ndarray | None,
    cache_data: dict[str, object] | None = None,
    build_cache_data: Callable[[], dict[str, object]] | None = None,
    refine_preview_point: Callable[..., tuple[float, float]],
    preview_due: Callable[[float, float], bool] | None = None,
    nearest_candidate_to_point: Callable[
        [float, float, Sequence[dict[str, object]] | None],
        tuple[dict[str, object] | None, float],
    ] = geometry_manual_nearest_candidate_to_point,
    position_error_px: Callable[[float, float, float, float], float] = geometry_manual_position_error_px,
    position_sigma_px: Callable[[object], float] = geometry_manual_position_sigma_px,
    use_caked_space: bool,
    caked_angles_to_background_display_coords: Callable[[float, float], tuple[float | None, float | None]] | None = None,
) -> dict[str, object] | None:
    """Return preview state for one manual placement cursor position."""

    if not geometry_manual_pick_session_active(
        pick_session,
        current_background_index=current_background_index,
    ):
        return None
    if pick_session is None:
        return None
    if not force and callable(preview_due) and not preview_due(float(raw_col), float(raw_row)):
        return None
    if display_background is None:
        return None

    state = cache_data if isinstance(cache_data, dict) else None
    if state is None and callable(build_cache_data):
        try:
            built_state = build_cache_data()
        except Exception:
            built_state = None
        if isinstance(built_state, dict):
            state = built_state

    refined_col, refined_row = refine_preview_point(
        None,
        float(raw_col),
        float(raw_row),
        display_background=display_background,
        cache_data=state,
    )
    candidate, sim_dist = nearest_candidate_to_point(
        float(refined_col),
        float(refined_row),
        remaining_candidates,
    )
    delta = float(
        position_error_px(
            float(raw_col),
            float(raw_row),
            float(refined_col),
            float(refined_row),
        )
    )
    if use_caked_space and callable(caked_angles_to_background_display_coords):
        raw_display = caked_angles_to_background_display_coords(float(raw_col), float(raw_row))
        refined_display = caked_angles_to_background_display_coords(
            float(refined_col),
            float(refined_row),
        )
        if (
            raw_display[0] is not None
            and raw_display[1] is not None
            and refined_display[0] is not None
            and refined_display[1] is not None
        ):
            delta = float(
                position_error_px(
                    float(raw_display[0]),
                    float(raw_display[1]),
                    float(refined_display[0]),
                    float(refined_display[1]),
                )
            )
    sigma_px = position_sigma_px(delta)
    candidate_label = str(candidate.get("label", "")) if isinstance(candidate, dict) else ""
    q_label = str(
        pick_session.get(
            "q_label",
            pick_session.get("group_key", "selected Qr/Qz set"),
        )
    )
    message = (
        f"Manual pick preview for {q_label}: "
        f"release=({float(raw_col):.1f},{float(raw_row):.1f}) -> "
        f"refined=({float(refined_col):.1f},{float(refined_row):.1f}), "
        f"placement error={delta:.2f}px, sigma={float(sigma_px):.2f}px"
    )
    if candidate_label:
        message += f" -> nearest sim [{candidate_label}]"
        if np.isfinite(sim_dist):
            message += f" ({float(sim_dist):.2f}{' deg' if use_caked_space else 'px'})"
    return {
        "raw_col": float(raw_col),
        "raw_row": float(raw_row),
        "refined_col": float(refined_col),
        "refined_row": float(refined_row),
        "candidate": candidate,
        "sim_dist": float(sim_dist),
        "delta": float(delta),
        "sigma_px": float(sigma_px),
        "message": message,
    }


def geometry_manual_session_initial_pairs_display(
    pick_session: dict[str, object] | None,
    *,
    current_background_index: object = None,
    require_current_background: bool = True,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
    entry_display_coords: Callable[
        [dict[str, object] | None],
        tuple[float, float] | None,
    ],
) -> list[dict[str, object]]:
    """Return overlay-ready display entries for the in-progress manual pick session."""

    if not geometry_manual_pick_session_active(
        pick_session,
        current_background_index=current_background_index,
        require_current_background=require_current_background,
    ):
        return []
    if pick_session is None:
        return []

    group_entries = pick_session.get("group_entries", [])
    pending_entries = pick_session.get("pending_entries", [])
    if not isinstance(group_entries, list) or not isinstance(pending_entries, list):
        return []

    pending_lookup: dict[tuple[object, ...], dict[str, object]] = {}
    for raw_entry in pending_entries:
        if not isinstance(raw_entry, dict):
            continue
        source_key = candidate_source_key(raw_entry)
        if source_key is not None:
            pending_lookup[source_key] = raw_entry

    initial_pairs_display: list[dict[str, object]] = []
    for pair_idx, raw_entry in enumerate(group_entries):
        if not isinstance(raw_entry, dict):
            continue
        entry: dict[str, object] = {
            "overlay_match_index": int(pair_idx),
            "hkl": raw_entry.get("hkl", raw_entry.get("label")),
        }
        try:
            sim_col = float(raw_entry.get("sim_col"))
            sim_row = float(raw_entry.get("sim_row"))
        except Exception:
            sim_col = float("nan")
            sim_row = float("nan")
        if np.isfinite(sim_col) and np.isfinite(sim_row):
            entry["sim_display"] = (float(sim_col), float(sim_row))

        source_key = candidate_source_key(raw_entry)
        measured_entry = pending_lookup.get(source_key) if source_key is not None else None
        if isinstance(measured_entry, dict):
            bg_coords = entry_display_coords(measured_entry)
            if bg_coords is not None:
                entry["bg_display"] = (float(bg_coords[0]), float(bg_coords[1]))
        initial_pairs_display.append(entry)
    return initial_pairs_display


def cancel_geometry_manual_pick_session(
    pick_session: dict[str, object] | None,
    *,
    current_background_index: object = None,
    restore_view_fn: Callable[..., None],
    clear_preview_artists_fn: Callable[..., None],
    render_current_pairs_fn: Callable[..., None],
    update_button_label_fn: Callable[[], None],
    set_status_text: Callable[[str], None] | None = None,
    restore_view: bool = True,
    redraw: bool = True,
    message: str | None = None,
) -> dict[str, object]:
    """Discard any in-progress manual Qr/Qz placement state."""

    had_session = geometry_manual_pick_session_active(
        pick_session,
        current_background_index=current_background_index,
        require_current_background=False,
    )
    if restore_view:
        restore_view_fn(redraw=False)
    clear_preview_artists_fn(redraw=False)
    if had_session and redraw:
        render_current_pairs_fn(update_status=False)
    update_button_label_fn()
    if message and callable(set_status_text):
        set_status_text(message)
    return {}


def match_geometry_manual_group_to_background(
    candidate_entries: Sequence[dict[str, object]] | None,
    *,
    background_image: np.ndarray | None = None,
    cache_data: dict[str, object] | None = None,
    build_cache_data: Callable[[], dict[str, object]] | None = None,
    auto_match_background_context: Callable[
        [np.ndarray, dict[str, object]],
        tuple[dict[str, object], dict[str, object]],
    ] | None = None,
    match_simulated_peaks_to_peak_context: Callable[
        [Sequence[dict[str, object]], dict[str, object], dict[str, object]],
        tuple[Sequence[dict[str, object]], object],
    ],
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
) -> dict[tuple[object, ...], tuple[float, float]]:
    """Return refined measured peak centers for one clicked symmetric Qr/Qz group."""

    entries = [dict(entry) for entry in candidate_entries or [] if isinstance(entry, dict)]
    if not entries:
        return {}

    background_local = background_image
    if background_local is None:
        return {}

    state = cache_data if isinstance(cache_data, dict) else {}
    if not state and callable(build_cache_data):
        try:
            built_state = build_cache_data()
        except Exception:
            built_state = {}
        if isinstance(built_state, dict):
            state = built_state

    match_cfg = dict(state.get("match_config", {})) if isinstance(state, dict) else {}
    background_context = state.get("background_context") if isinstance(state, dict) else None
    if not isinstance(background_context, dict) or not bool(background_context.get("img_valid", False)):
        if not callable(auto_match_background_context):
            return {}
        try:
            match_cfg, background_context = auto_match_background_context(
                background_local,
                match_cfg,
            )
        except Exception:
            return {}

    try:
        matches, _stats = match_simulated_peaks_to_peak_context(
            entries,
            background_context,
            match_cfg,
        )
    except Exception:
        return {}

    matched_lookup: dict[tuple[object, ...], tuple[float, float]] = {}
    for match_entry in matches:
        source_key = candidate_source_key(match_entry)
        if source_key is None:
            continue
        try:
            match_col = float(match_entry.get("x", np.nan))
            match_row = float(match_entry.get("y", np.nan))
        except Exception:
            continue
        if np.isfinite(match_col) and np.isfinite(match_row):
            matched_lookup[source_key] = (float(match_col), float(match_row))
    return matched_lookup


def ensure_geometry_fit_caked_view(
    *,
    show_caked_2d_var: Any,
    pick_uses_caked_space: Callable[[], bool],
    toggle_caked_2d: Callable[[], None],
    do_update: Callable[[], None],
    schedule_update: Callable[[], None],
    root: Any,
    update_pending: object | None,
    integration_update_pending: object | None,
    update_running: bool = False,
    force_refresh: bool = False,
) -> tuple[object | None, object | None]:
    """Switch geometry fitting/import into the 2D caked integration view now."""

    needs_refresh = bool(force_refresh)
    if not bool(show_caked_2d_var.get()):
        show_caked_2d_var.set(True)
        toggle_caked_2d()
        needs_refresh = True
    elif not pick_uses_caked_space():
        needs_refresh = True

    if not needs_refresh:
        return update_pending, integration_update_pending

    if integration_update_pending is not None:
        try:
            root.after_cancel(integration_update_pending)
        except Exception:
            pass
        integration_update_pending = None
    if update_pending is not None:
        try:
            root.after_cancel(update_pending)
        except Exception:
            pass
        update_pending = None

    if bool(update_running):
        schedule_update()
        return update_pending, integration_update_pending

    do_update()
    return update_pending, integration_update_pending


def caked_angles_to_background_display_coords(
    two_theta_deg: float,
    phi_deg: float,
    *,
    ai: object = None,
    native_background: np.ndarray | None = None,
    get_detector_angular_maps: Callable[[object], tuple[object, object]],
    scattering_angles_to_detector_pixel: Callable[
        [float, float, Sequence[float] | None, float, float],
        tuple[float | None, float | None],
    ],
    center: Sequence[float] | None,
    detector_distance: float,
    pixel_size: float,
    rotate_point_for_display: Callable[[float, float, tuple[int, ...], int], tuple[float, float]] = _default_rotate_point,
    display_rotate_k: int = 0,
) -> tuple[float | None, float | None]:
    """Back-project one caked-space point to the displayed detector background."""

    try:
        two_theta_map, phi_map = get_detector_angular_maps(ai)
    except Exception:
        two_theta_map, phi_map = None, None
    if (
        two_theta_map is None
        or phi_map is None
        or native_background is None
        or not (np.isfinite(two_theta_deg) and np.isfinite(phi_deg))
    ):
        if native_background is None:
            return None, None
        try:
            native_point = scattering_angles_to_detector_pixel(
                float(two_theta_deg),
                float(phi_deg),
                center,
                float(detector_distance),
                float(pixel_size),
            )
        except Exception:
            return None, None
        if native_point[0] is None or native_point[1] is None:
            return None, None
        return rotate_point_for_display(
            float(native_point[0]),
            float(native_point[1]),
            tuple(int(v) for v in native_background.shape[:2]),
            display_rotate_k,
        )

    dphi = ((np.asarray(phi_map, dtype=float) - float(phi_deg) + 180.0) % 360.0) - 180.0
    dtth = np.asarray(two_theta_map, dtype=float) - float(two_theta_deg)
    metric = dtth * dtth + dphi * dphi
    finite_metric = np.where(np.isfinite(metric), metric, np.inf)
    best_idx = int(np.argmin(finite_metric))
    if not np.isfinite(finite_metric.flat[best_idx]):
        return None, None
    row_idx, col_idx = np.unravel_index(best_idx, finite_metric.shape)
    return rotate_point_for_display(
        float(col_idx),
        float(row_idx),
        tuple(int(v) for v in native_background.shape[:2]),
        display_rotate_k,
    )


def native_detector_coords_to_caked_display_coords(
    col: float,
    row: float,
    *,
    ai: object = None,
    get_detector_angular_maps: Callable[[object], tuple[object, object]],
    detector_pixel_to_scattering_angles: Callable[
        [float, float, Sequence[float] | None, float, float],
        tuple[float | None, float | None],
    ],
    center: Sequence[float] | None,
    detector_distance: float,
    pixel_size: float,
    wrap_phi_range: Callable[[object], object],
) -> tuple[float, float] | None:
    """Project one native detector pixel into the active caked display axes."""

    try:
        col_val = float(col)
        row_val = float(row)
    except Exception:
        return None
    if not (np.isfinite(col_val) and np.isfinite(row_val)):
        return None

    try:
        two_theta_map, phi_map = get_detector_angular_maps(ai)
    except Exception:
        two_theta_map, phi_map = None, None
    if two_theta_map is not None and phi_map is not None:
        try:
            height, width = two_theta_map.shape[:2]
            if height > 0 and width > 0:
                col_idx = min(max(int(round(col_val)), 0), width - 1)
                row_idx = min(max(int(round(row_val)), 0), height - 1)
                two_theta = float(two_theta_map[row_idx, col_idx])
                phi = float(phi_map[row_idx, col_idx])
                if np.isfinite(two_theta) and np.isfinite(phi):
                    return float(two_theta), float(wrap_phi_range(phi))
        except Exception:
            pass

    try:
        two_theta, phi = detector_pixel_to_scattering_angles(
            col_val,
            row_val,
            center,
            float(detector_distance),
            float(pixel_size),
        )
    except Exception:
        return None
    if two_theta is None or phi is None:
        return None
    return float(two_theta), float(wrap_phi_range(phi))


def should_collect_hit_tables_for_update(
    *,
    background_visible: bool,
    current_background_index: object,
    hkl_pick_armed: bool,
    selected_hkl_target: object,
    selected_peak_record: object,
    geometry_q_group_refresh_requested: bool,
    live_geometry_preview_enabled: Callable[[], bool],
    current_manual_pick_background_image: Callable[[], object],
    geometry_manual_pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    geometry_manual_pick_session_active: Callable[[], bool],
) -> bool:
    """Return whether the next redraw needs per-hit detector tables."""

    manual_geometry_overlay_requested = False
    if background_visible and current_manual_pick_background_image() is not None:
        try:
            manual_geometry_overlay_requested = bool(
                geometry_manual_pairs_for_index(int(current_background_index))
                or geometry_manual_pick_session_active()
            )
        except Exception:
            manual_geometry_overlay_requested = False

    return bool(
        hkl_pick_armed
        or selected_hkl_target is not None
        or selected_peak_record is not None
        or live_geometry_preview_enabled()
        or geometry_q_group_refresh_requested
        or manual_geometry_overlay_requested
    )


def normalize_bragg_qr_source_label(source_label: str | None) -> str:
    """Normalize the serialized Qr-source label used in manual-geometry data."""

    label = str(source_label or "primary").strip().lower()
    return "secondary" if label == "secondary" else "primary"


def q_group_key_component(value: float) -> int | float:
    """Normalize a floating Q-group component into a stable hashable value."""

    if np.isfinite(value) and abs(value - round(value)) <= 1e-6:
        return int(round(value))
    return round(float(value), 6)


def integer_gz_index(value: object, *, tol: float = 1e-6) -> int | None:
    """Return the integer Gz/L index when the value is close enough to an integer."""

    try:
        raw = float(value)
    except Exception:
        return None
    if not np.isfinite(raw):
        return None
    rounded = int(round(raw))
    if abs(raw - rounded) > float(tol):
        return None
    return rounded


def geometry_q_group_key_to_jsonable(group_key: object) -> list[object] | None:
    """Convert one stable Qr/Qz group key into a JSON-safe list."""

    if not isinstance(group_key, tuple) or len(group_key) < 4:
        return None
    try:
        prefix = str(group_key[0])
        source_label = normalize_bragg_qr_source_label(str(group_key[1]))
        m_component = q_group_key_component(float(group_key[2]))
        gz_index = int(group_key[3])
    except Exception:
        return None
    return [prefix, source_label, m_component, gz_index]


def geometry_q_group_key_from_jsonable(value: object) -> tuple[object, ...] | None:
    """Rebuild one stable Qr/Qz group key from JSON-loaded data."""

    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        prefix = str(value[0])
        source_label = normalize_bragg_qr_source_label(str(value[1]))
        m_component = q_group_key_component(float(value[2]))
        gz_index = integer_gz_index(value[3])
    except Exception:
        return None
    if prefix != "q_group" or gz_index is None:
        return None
    return (prefix, source_label, m_component, int(gz_index))


def geometry_manual_pair_entry_to_jsonable(
    entry: dict[str, object] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
    sigma_floor_px: float = DEFAULT_POSITION_SIGMA_FLOOR_PX,
) -> dict[str, object] | None:
    """Convert one saved manual pair entry into a JSON-safe dictionary."""

    normalized = normalize_geometry_manual_pair_entry(
        entry,
        normalize_hkl_key=normalize_hkl_key,
        sigma_floor_px=sigma_floor_px,
    )
    if normalized is None:
        return None

    row: dict[str, object] = {
        "x": float(normalized["x"]),
        "y": float(normalized["y"]),
        "label": str(normalized.get("label", "")),
    }

    for key in (
        "raw_x",
        "raw_y",
        "caked_x",
        "caked_y",
        "raw_caked_x",
        "raw_caked_y",
        "placement_error_px",
        "sigma_px",
    ):
        value = normalized.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if np.isfinite(numeric):
            row[key] = float(numeric)

    hkl_key = normalized.get("hkl")
    if isinstance(hkl_key, tuple) and len(hkl_key) >= 3:
        try:
            row["hkl"] = [int(hkl_key[0]), int(hkl_key[1]), int(hkl_key[2])]
        except Exception:
            pass

    serialized_group_key = geometry_q_group_key_to_jsonable(normalized.get("q_group_key"))
    if serialized_group_key is not None:
        row["q_group_key"] = serialized_group_key

    for key in ("source_table_index", "source_row_index"):
        if key in normalized:
            try:
                row[key] = int(normalized[key])
            except Exception:
                continue

    if normalized.get("source_label") is not None:
        row["source_label"] = str(normalized.get("source_label"))

    return row


def geometry_manual_pair_entry_from_jsonable(
    row: dict[str, object] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
    sigma_floor_px: float = DEFAULT_POSITION_SIGMA_FLOOR_PX,
) -> dict[str, object] | None:
    """Rebuild one saved manual pair entry from imported JSON data."""

    if not isinstance(row, dict):
        return None

    entry = dict(row)
    raw_hkl = row.get("hkl")
    if isinstance(raw_hkl, (list, tuple)) and len(raw_hkl) >= 3:
        try:
            entry["hkl"] = (
                int(raw_hkl[0]),
                int(raw_hkl[1]),
                int(raw_hkl[2]),
            )
        except Exception:
            pass

    restored_group_key = geometry_q_group_key_from_jsonable(row.get("q_group_key"))
    if restored_group_key is not None:
        entry["q_group_key"] = restored_group_key

    return normalize_geometry_manual_pair_entry(
        entry,
        normalize_hkl_key=normalize_hkl_key,
        sigma_floor_px=sigma_floor_px,
    )
