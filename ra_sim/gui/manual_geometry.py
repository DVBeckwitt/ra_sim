"""Helpers for manual geometry selection, caching, and serialization."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ra_sim.fitting.background_peak_matching import (
    _candidate_summit_id_near_pixel as _background_candidate_summit_id_near_pixel,
)
from ra_sim.fitting.background_peak_matching import (
    _refine_peak_center as _background_refine_peak_center,
)
from ra_sim.gui import controllers as gui_controllers
from ra_sim.gui.geometry_overlay import normalize_hkl_key as _default_normalize_hkl_key
from ra_sim.gui.geometry_overlay import rotate_point_for_display as _default_rotate_point


DEFAULT_POSITION_SIGMA_FLOOR_PX = 0.75
DEFAULT_PREVIEW_GOOD_SIGMA_PX = 1.5
DEFAULT_PREVIEW_BAD_SIGMA_PX = 12.0
DEFAULT_CAKED_SEARCH_TTH_DEG = 1.5
DEFAULT_CAKED_SEARCH_PHI_DEG = 10.0
DEFAULT_PREVIEW_MIN_INTERVAL_S = 0.03
DEFAULT_PREVIEW_MIN_MOVE_PX = 0.8
DEFAULT_GUI_ENTRYPOINT = "python -m ra_sim gui"


@dataclass(frozen=True)
class GeometryManualRuntimeCallbacks:
    """Bound runtime callbacks for manual geometry preview and pick actions."""

    render_current_pairs: Callable[..., bool]
    toggle_selection_at: Callable[[float, float], bool]
    place_selection_at: Callable[[float, float], bool]
    update_pick_preview: Callable[..., None]
    cancel_pick_session: Callable[..., None]


@dataclass(frozen=True)
class GeometryManualRuntimeCacheCallbacks:
    """Bound runtime callbacks for manual-geometry cache and overlay state."""

    current_match_config: Callable[[], dict[str, object]]
    pick_cache_signature: Callable[..., tuple[object, ...]]
    get_pick_cache: Callable[..., dict[str, object]]
    build_initial_pairs_display: Callable[
        ...,
        tuple[list[dict[str, object]], list[dict[str, object]]],
    ]


@dataclass(frozen=True)
class GeometryManualRuntimeProjectionCallbacks:
    """Bound runtime callbacks for manual-geometry view/projection helpers."""

    pick_uses_caked_space: Callable[[], bool]
    current_background_image: Callable[[], object | None]
    entry_display_coords: Callable[
        [dict[str, object] | None],
        tuple[float, float] | None,
    ]
    caked_angles_to_background_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ]
    background_display_to_native_detector_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    native_detector_coords_to_caked_display_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    project_peaks_to_current_view: Callable[
        [Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    simulated_peaks_for_params: Callable[..., list[dict[str, object]]]
    pick_candidates: Callable[
        [Sequence[dict[str, object]] | None],
        dict[tuple[object, ...], list[dict[str, object]]],
    ]
    simulated_lookup: Callable[
        [Sequence[dict[str, object]] | None],
        dict[tuple[int, int], dict[str, object]],
    ]


def _resolve_runtime_value(value_or_callable: object) -> object:
    if callable(value_or_callable):
        try:
            return value_or_callable()
        except Exception:
            return None
    return value_or_callable


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


def geometry_manual_preview_color(
    sigma_px: object,
    *,
    good_sigma_px: float = DEFAULT_PREVIEW_GOOD_SIGMA_PX,
    bad_sigma_px: float = DEFAULT_PREVIEW_BAD_SIGMA_PX,
) -> str:
    """Return a green-to-red preview color for one manual-pick uncertainty."""

    try:
        sigma_val = float(sigma_px)
    except Exception:
        sigma_val = float("nan")
    if not np.isfinite(sigma_val):
        sigma_val = float(bad_sigma_px)

    good_val = max(1.0e-3, float(good_sigma_px))
    bad_val = max(good_val + 1.0e-3, float(bad_sigma_px))
    ratio = (sigma_val - good_val) / (bad_val - good_val)
    ratio = min(max(float(ratio), 0.0), 1.0)

    return _geometry_manual_preview_gradient_color(float(ratio))


def _geometry_manual_preview_gradient_color(ratio: float) -> str:
    """Return the shared green-yellow-red preview gradient for one normalized ratio."""

    ratio = min(max(float(ratio), 0.0), 1.0)
    if ratio <= 0.5:
        local_ratio = ratio / 0.5
        start_rgb = (0x2E, 0xCC, 0x71)
        end_rgb = (0xF1, 0xC4, 0x0F)
    else:
        local_ratio = (ratio - 0.5) / 0.5
        start_rgb = (0xF1, 0xC4, 0x0F)
        end_rgb = (0xE7, 0x4C, 0x3C)

    rgb = tuple(
        int(round((1.0 - local_ratio) * start + local_ratio * end))
        for start, end in zip(start_rgb, end_rgb)
    )
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def geometry_manual_preview_confidence_color(
    confidence: object,
    *,
    good_confidence: float = 1.0,
    bad_confidence: float = 0.25,
) -> str:
    """Return a red-to-green preview color for one predicted match confidence."""

    try:
        confidence_val = float(confidence)
    except Exception:
        confidence_val = float("nan")
    if not np.isfinite(confidence_val):
        confidence_val = float(bad_confidence)

    bad_val = max(0.0, float(bad_confidence))
    good_val = max(bad_val + 1.0e-6, float(good_confidence))
    ratio = (good_val - confidence_val) / (good_val - bad_val)
    return _geometry_manual_preview_gradient_color(float(ratio))


def geometry_manual_preview_quality_label(
    sigma_px: object,
    *,
    good_sigma_px: float = DEFAULT_PREVIEW_GOOD_SIGMA_PX,
    bad_sigma_px: float = DEFAULT_PREVIEW_BAD_SIGMA_PX,
) -> str:
    """Return a coarse quality label for one manual-pick uncertainty."""

    try:
        sigma_val = float(sigma_px)
    except Exception:
        sigma_val = float("nan")
    if not np.isfinite(sigma_val):
        return "bad"

    good_val = max(1.0e-3, float(good_sigma_px))
    bad_val = max(good_val + 1.0e-3, float(bad_sigma_px))
    ratio = (sigma_val - good_val) / (bad_val - good_val)
    ratio = min(max(float(ratio), 0.0), 1.0)
    if ratio <= 0.2:
        return "good"
    if ratio >= 0.75:
        return "bad"
    return "warning"


def geometry_manual_preview_confidence_quality_label(
    confidence: object,
    *,
    good_confidence: float = 1.0,
    bad_confidence: float = 0.25,
) -> str:
    """Return a coarse quality label for one predicted match confidence."""

    try:
        confidence_val = float(confidence)
    except Exception:
        confidence_val = float("nan")
    if not np.isfinite(confidence_val):
        return "bad"

    bad_val = max(0.0, float(bad_confidence))
    good_val = max(bad_val + 1.0e-6, float(good_confidence))
    ratio = (good_val - confidence_val) / (good_val - bad_val)
    ratio = min(max(float(ratio), 0.0), 1.0)
    if ratio <= 0.2:
        return "good"
    if ratio >= 0.75:
        return "bad"
    return "warning"


def geometry_manual_preview_match_confidence(
    candidate: dict[str, object] | None,
    peak_col: float,
    peak_row: float,
    *,
    cache_data: dict[str, object] | None = None,
    build_cache_data: Callable[[], dict[str, object]] | None = None,
    use_caked_space: bool,
    radial_axis: Sequence[float] | None = None,
    azimuth_axis: Sequence[float] | None = None,
    caked_axis_to_image_index_fn: Callable[[float, Sequence[float] | None], float]
    | None = None,
) -> float:
    """Return the predicted match confidence for one chosen manual-placement peak."""

    if not isinstance(candidate, dict):
        return float("nan")

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
    if not isinstance(background_context, dict) or not bool(
        background_context.get("img_valid", False)
    ):
        return float("nan")

    axis_to_image_index = (
        caked_axis_to_image_index
        if caked_axis_to_image_index_fn is None
        else caked_axis_to_image_index_fn
    )

    try:
        sim_col_local = float(
            candidate.get("sim_col_local", candidate.get("sim_col", np.nan))
        )
        sim_row_local = float(
            candidate.get("sim_row_local", candidate.get("sim_row", np.nan))
        )
    except Exception:
        return float("nan")

    if use_caked_space:
        radial_axis_arr = np.asarray(radial_axis, dtype=float)
        azimuth_axis_arr = np.asarray(azimuth_axis, dtype=float)
        peak_col_local = float(
            axis_to_image_index(float(peak_col), radial_axis_arr)
        )
        peak_row_local = float(
            axis_to_image_index(float(peak_row), azimuth_axis_arr)
        )
        if not np.isfinite(sim_col_local):
            sim_col_local = float(
                axis_to_image_index(
                    float(candidate.get("sim_col", np.nan)),
                    radial_axis_arr,
                )
            )
        if not np.isfinite(sim_row_local):
            sim_row_local = float(
                axis_to_image_index(
                    float(candidate.get("sim_row", np.nan)),
                    azimuth_axis_arr,
                )
            )
    else:
        peak_col_local = float(peak_col)
        peak_row_local = float(peak_row)

    if not (
        np.isfinite(sim_col_local)
        and np.isfinite(sim_row_local)
        and np.isfinite(peak_col_local)
        and np.isfinite(peak_row_local)
    ):
        return float("nan")

    candidate_labels = np.asarray(
        background_context.get("candidate_labels", []),
        dtype=np.int32,
    )
    peakness = np.asarray(background_context.get("peakness", []), dtype=float)
    fine = np.asarray(background_context.get("fine", []), dtype=float)
    height = int(
        background_context.get(
            "height",
            candidate_labels.shape[0] if candidate_labels.ndim == 2 else 0,
        )
    )
    width = int(
        background_context.get(
            "width",
            candidate_labels.shape[1] if candidate_labels.ndim == 2 else 0,
        )
    )
    if height <= 0 or width <= 0:
        return float("nan")

    peak_row_px = int(np.clip(round(peak_row_local), 0, max(height - 1, 0)))
    peak_col_px = int(np.clip(round(peak_col_local), 0, max(width - 1, 0)))
    summit_id = _background_candidate_summit_id_near_pixel(
        candidate_labels,
        peakness,
        peak_row_px,
        peak_col_px,
        radius_px=max(
            1,
            int(round(0.5 * float(match_cfg.get("local_max_size_px", 5)))),
        ),
    )
    if summit_id <= 0:
        return float("nan")

    summit_record = None
    for record in background_context.get("summit_records", ()):
        if int(record.get("summit_id", -1)) == summit_id:
            summit_record = dict(record)
            break
    if summit_record is None:
        return float("nan")

    peak_row_seed = int(
        np.clip(
            round(float(summit_record.get("row", peak_row_local))),
            0,
            max(height - 1, 0),
        )
    )
    peak_col_seed = int(
        np.clip(
            round(float(summit_record.get("col", peak_col_local))),
            0,
            max(width - 1, 0),
        )
    )
    center_col_local, center_row_local = _background_refine_peak_center(
        peakness,
        fine,
        peak_row_seed,
        peak_col_seed,
    )
    prom_sigma = float(summit_record.get("prominence_sigma", np.nan))
    if not (
        np.isfinite(center_col_local)
        and np.isfinite(center_row_local)
        and np.isfinite(prom_sigma)
    ):
        return float("nan")

    dist_px = float(
        np.hypot(
            float(center_col_local) - float(sim_col_local),
            float(center_row_local) - float(sim_row_local),
        )
    )
    return float(max(0.0, prom_sigma) / (1.0 + max(0.0, dist_px)))


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

    for key in ("source_table_index", "source_row_index", "source_peak_index"):
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

    for x_key, y_key in (("detector_x", "detector_y"),):
        raw_x_local = entry.get(x_key)
        raw_y_local = entry.get(y_key)
        try:
            detector_x_val = (
                float(raw_x_local) if raw_x_local is not None else float("nan")
            )
        except Exception:
            detector_x_val = float("nan")
        try:
            detector_y_val = (
                float(raw_y_local) if raw_y_local is not None else float("nan")
            )
        except Exception:
            detector_y_val = float("nan")
        if np.isfinite(detector_x_val) and np.isfinite(detector_y_val):
            normalized[x_key] = float(detector_x_val)
            normalized[y_key] = float(detector_y_val)

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

    for x_key, y_key in (
        ("refined_sim_x", "refined_sim_y"),
        ("refined_sim_native_x", "refined_sim_native_y"),
        ("refined_sim_caked_x", "refined_sim_caked_y"),
    ):
        raw_x_local = entry.get(x_key)
        raw_y_local = entry.get(y_key)
        try:
            refined_x_val = (
                float(raw_x_local) if raw_x_local is not None else float("nan")
            )
        except Exception:
            refined_x_val = float("nan")
        try:
            refined_y_val = (
                float(raw_y_local) if raw_y_local is not None else float("nan")
            )
        except Exception:
            refined_y_val = float("nan")
        if np.isfinite(refined_x_val) and np.isfinite(refined_y_val):
            normalized[x_key] = float(refined_x_val)
            normalized[y_key] = float(refined_y_val)

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


def geometry_manual_apply_refined_simulated_override(
    entry: dict[str, object] | None,
    resolved_source_entry: dict[str, object] | None,
) -> dict[str, object] | None:
    """Overlay one saved refined-simulation position onto a resolved source entry."""

    result = dict(resolved_source_entry) if isinstance(resolved_source_entry, dict) else {}
    if not isinstance(entry, dict):
        return result or None

    if not result:
        for key in (
            "hkl",
            "label",
            "source_table_index",
            "source_row_index",
            "source_peak_index",
            "source_label",
            "q_group_key",
        ):
            if key in entry:
                result[key] = entry.get(key)

    def _pair(x_key: str, y_key: str) -> tuple[float, float] | None:
        try:
            x_val = float(entry.get(x_key, np.nan))
            y_val = float(entry.get(y_key, np.nan))
        except Exception:
            return None
        if not (np.isfinite(x_val) and np.isfinite(y_val)):
            return None
        return float(x_val), float(y_val)

    refined_raw = _pair("refined_sim_x", "refined_sim_y")
    refined_native = _pair("refined_sim_native_x", "refined_sim_native_y")
    refined_caked = _pair("refined_sim_caked_x", "refined_sim_caked_y")

    use_caked_display = False
    try:
        sim_col = float(result.get("sim_col", np.nan))
        sim_row = float(result.get("sim_row", np.nan))
        sim_col_raw = float(result.get("sim_col_raw", sim_col))
        sim_row_raw = float(result.get("sim_row_raw", sim_row))
        caked_x = float(result.get("caked_x", np.nan))
        caked_y = float(result.get("caked_y", np.nan))
        if (
            np.isfinite(sim_col)
            and np.isfinite(sim_row)
            and np.isfinite(caked_x)
            and np.isfinite(caked_y)
        ):
            use_caked_display = (
                abs(float(sim_col) - float(caked_x)) <= 1.0e-9
                and abs(float(sim_row) - float(caked_y)) <= 1.0e-9
            )
        elif (
            np.isfinite(sim_col)
            and np.isfinite(sim_row)
            and np.isfinite(sim_col_raw)
            and np.isfinite(sim_row_raw)
        ):
            use_caked_display = (
                abs(float(sim_col) - float(sim_col_raw)) > 1.0e-9
                or abs(float(sim_row) - float(sim_row_raw)) > 1.0e-9
            )
    except Exception:
        use_caked_display = False

    if refined_raw is not None:
        result["sim_col_raw"] = float(refined_raw[0])
        result["sim_row_raw"] = float(refined_raw[1])
        if not use_caked_display:
            result["sim_col"] = float(refined_raw[0])
            result["sim_row"] = float(refined_raw[1])

    if refined_native is not None:
        result["sim_native_x"] = float(refined_native[0])
        result["sim_native_y"] = float(refined_native[1])

    if refined_caked is not None:
        result["caked_x"] = float(refined_caked[0])
        result["caked_y"] = float(refined_caked[1])
        if use_caked_display:
            result["sim_col"] = float(refined_caked[0])
            result["sim_row"] = float(refined_caked[1])

    return result or None


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
            "source_peak",
            int(entry.get("source_table_index")),
            int(entry.get("source_peak_index")),
        )
    except Exception:
        pass
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


def geometry_manual_candidate_distance_to_point(
    col: float,
    row: float,
    candidate: dict[str, object] | None,
) -> float:
    """Return one display-space point-to-candidate distance."""

    if not isinstance(candidate, dict):
        return float("nan")
    try:
        sim_col = float(candidate.get("sim_col"))
        sim_row = float(candidate.get("sim_row"))
    except Exception:
        return float("nan")
    if not (np.isfinite(sim_col) and np.isfinite(sim_row)):
        return float("nan")
    return float(np.hypot(float(sim_col) - float(col), float(sim_row) - float(row)))


def geometry_manual_prioritize_candidate_entries(
    candidate_entries: Sequence[dict[str, object]] | None,
    preferred_candidate: dict[str, object] | None,
    *,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
) -> list[dict[str, object]]:
    """Return candidate entries with the preferred entry, if any, moved to the front."""

    entries = [dict(entry) for entry in candidate_entries or [] if isinstance(entry, dict)]
    preferred_key = candidate_source_key(preferred_candidate)
    if preferred_key is None or not entries:
        return entries

    reordered: list[dict[str, object]] = []
    matched = False
    for entry in entries:
        if not matched and candidate_source_key(entry) == preferred_key:
            reordered.insert(0, entry)
            matched = True
        else:
            reordered.append(entry)
    return reordered


def geometry_manual_central_candidate(
    candidate_entries: Sequence[dict[str, object]] | None,
    *,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
) -> dict[str, object] | None:
    """Return one deterministic central-ray representative for a Qr/Qz group."""

    entries = [dict(entry) for entry in candidate_entries or [] if isinstance(entry, dict)]
    if not entries:
        return None

    finite_entries: list[dict[str, object]] = []
    weights: list[float] = []
    cols: list[float] = []
    rows: list[float] = []
    for entry in entries:
        try:
            sim_col = float(entry.get("sim_col", np.nan))
            sim_row = float(entry.get("sim_row", np.nan))
        except Exception:
            continue
        if not (np.isfinite(sim_col) and np.isfinite(sim_row)):
            continue
        try:
            weight = float(entry.get("weight", 1.0))
        except Exception:
            weight = 1.0
        if not np.isfinite(weight) or weight <= 0.0:
            weight = 1.0
        finite_entries.append(entry)
        cols.append(float(sim_col))
        rows.append(float(sim_row))
        weights.append(float(weight))

    if not finite_entries:
        return dict(entries[0])

    weight_arr = np.asarray(weights, dtype=float)
    col_arr = np.asarray(cols, dtype=float)
    row_arr = np.asarray(rows, dtype=float)
    center_col = float(np.average(col_arr, weights=weight_arr))
    center_row = float(np.average(row_arr, weights=weight_arr))

    def _sort_key(entry: dict[str, object]) -> tuple[object, ...]:
        try:
            sim_col = float(entry.get("sim_col", np.nan))
            sim_row = float(entry.get("sim_row", np.nan))
        except Exception:
            sim_col = float("nan")
            sim_row = float("nan")
        if np.isfinite(sim_col) and np.isfinite(sim_row):
            d2 = (sim_col - center_col) ** 2 + (sim_row - center_row) ** 2
        else:
            d2 = float("inf")
        try:
            weight = float(entry.get("weight", 0.0))
        except Exception:
            weight = 0.0
        source_key = candidate_source_key(entry)
        return (
            float(d2),
            -float(weight) if np.isfinite(weight) else 0.0,
            repr(source_key) if source_key is not None else "",
            str(entry.get("label", "")),
        )

    return dict(min(finite_entries, key=_sort_key))


def geometry_manual_tagged_candidate_from_session(
    pick_session: dict[str, object] | None,
    candidate_entries: Sequence[dict[str, object]] | None,
    *,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
) -> dict[str, object] | None:
    """Return the exact tagged candidate for the active session when it remains available."""

    if not isinstance(pick_session, dict):
        return None

    tagged_key = pick_session.get("tagged_candidate_key")
    if tagged_key is None:
        tagged_candidate = pick_session.get("tagged_candidate")
        tagged_key = (
            candidate_source_key(tagged_candidate)
            if isinstance(tagged_candidate, dict)
            else None
        )
    if tagged_key is None:
        return None

    for raw_entry in candidate_entries or []:
        if not isinstance(raw_entry, dict):
            continue
        if candidate_source_key(raw_entry) == tagged_key:
            return dict(raw_entry)
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


def geometry_manual_pick_cache_signature(
    *,
    last_simulation_signature: object,
    background_index: int,
    background_image: object | None,
    use_caked_space: bool,
    geometry_preview_excluded_q_groups: Sequence[object] | None,
    geometry_q_group_cached_entries: Sequence[object] | None,
    stored_max_positions_local: Sequence[object] | None,
    stored_peak_table_lattice: Sequence[object] | None,
    peak_records: Sequence[object] | None = None,
) -> tuple[object, ...]:
    """Return a stable cache signature for reusable manual-pick state."""

    bg_token = None
    if background_image is not None:
        raw_arr = np.asarray(background_image)
        try:
            bg_ptr = int(raw_arr.__array_interface__["data"][0])
        except Exception:
            bg_ptr = int(id(raw_arr))
        bg_token = (
            bg_ptr,
            tuple(int(v) for v in raw_arr.shape),
            tuple(int(v) for v in raw_arr.strides),
            str(raw_arr.dtype),
        )

    excluded_keys = tuple(
        sorted(repr(key) for key in (geometry_preview_excluded_q_groups or ()))
    )
    try:
        listed_group_count = len(geometry_q_group_cached_entries or ())
    except Exception:
        listed_group_count = 0
    try:
        maxpos_count = len(stored_max_positions_local or ())
    except Exception:
        maxpos_count = 0
    try:
        peak_table_count = len(stored_peak_table_lattice or ())
    except Exception:
        peak_table_count = 0
    try:
        peak_record_count = len(peak_records or ())
    except Exception:
        peak_record_count = 0

    return (
        last_simulation_signature,
        int(background_index),
        bool(use_caked_space),
        bg_token,
        int(maxpos_count),
        int(peak_table_count),
        int(peak_record_count),
        int(listed_group_count),
        excluded_keys,
    )


def build_geometry_manual_pick_cache(
    *,
    param_set: dict[str, object] | None = None,
    prefer_cache: bool = True,
    background_index: int,
    current_background_index: int,
    background_image: object | None,
    existing_cache_signature: object = None,
    existing_cache_data: dict[str, object] | None = None,
    cache_signature_fn: Callable[..., tuple[object, ...]],
    simulated_peaks_for_params: Callable[..., Sequence[dict[str, object]]],
    build_grouped_candidates: Callable[
        [Sequence[dict[str, object]] | None],
        dict[tuple[object, ...], list[dict[str, object]]],
    ],
    build_simulated_lookup: Callable[
        [Sequence[dict[str, object]] | None],
        dict[tuple[object, ...], dict[str, object]],
    ],
    current_match_config: Callable[[], dict[str, object]],
    auto_match_background_context: Callable[
        [object, dict[str, object]],
        tuple[dict[str, object], object],
    ]
    | None = None,
) -> tuple[dict[str, object], object, dict[str, object]]:
    """Build or reuse the current manual-pick simulation/background cache."""

    bg_index = int(background_index)
    current_bg_index = int(current_background_index)
    cache_sig = cache_signature_fn(
        background_index=bg_index,
        background_image=background_image,
    )
    if (
        prefer_cache
        and bg_index == current_bg_index
        and existing_cache_signature == cache_sig
        and isinstance(existing_cache_data, dict)
        and dict(existing_cache_data.get("grouped_candidates", {}))
    ):
        return existing_cache_data, existing_cache_signature, existing_cache_data

    # Manual Qr/Qz picking must stay tied to the deterministic central-ray
    # geometry-fit simulation rather than the live full-beam preview cache.
    simulated_peaks = [
        dict(entry)
        for entry in simulated_peaks_for_params(
            param_set,
            prefer_cache=False,
        )
        if isinstance(entry, dict)
    ]
    grouped_candidates = build_grouped_candidates(simulated_peaks)
    simulated_lookup = build_simulated_lookup(simulated_peaks)
    if prefer_cache and not grouped_candidates:
        cached_simulated_peaks = [
            dict(entry)
            for entry in simulated_peaks_for_params(
                param_set,
                prefer_cache=True,
            )
            if isinstance(entry, dict)
        ]
        cached_grouped_candidates = build_grouped_candidates(cached_simulated_peaks)
        if cached_grouped_candidates:
            simulated_peaks = cached_simulated_peaks
            grouped_candidates = cached_grouped_candidates
            simulated_lookup = build_simulated_lookup(simulated_peaks)
    match_cfg = dict(current_match_config())
    resolved_match_cfg = dict(match_cfg)
    background_context = None
    if background_image is not None and callable(auto_match_background_context):
        try:
            resolved_match_cfg, background_context = auto_match_background_context(
                background_image,
                match_cfg,
            )
        except Exception:
            resolved_match_cfg = dict(match_cfg)
            background_context = None

    cache_result = {
        "signature": cache_sig,
        "simulated_peaks": [dict(entry) for entry in simulated_peaks],
        "simulated_lookup": dict(simulated_lookup),
        "grouped_candidates": {
            key: [dict(entry) for entry in entries]
            for key, entries in grouped_candidates.items()
        },
        "match_config": dict(resolved_match_cfg),
        "background_context": background_context,
    }

    next_cache_signature = existing_cache_signature
    next_cache_data = (
        existing_cache_data if isinstance(existing_cache_data, dict) else {}
    )
    if bg_index == current_bg_index:
        next_cache_signature = cache_sig
        next_cache_data = cache_result
    return cache_result, next_cache_signature, next_cache_data


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

    entries = [dict(entry) for entry in group_entries or [] if isinstance(entry, dict)]
    if not entries:
        return 0

    if isinstance(group_key, tuple) and len(group_key) >= 4:
        try:
            if int(group_key[2]) == 0:
                return 1
        except Exception:
            pass

    for entry in entries:
        hkl = normalize_hkl_key(entry.get("hkl", entry.get("label")))
        if hkl is None or int(hkl[0]) != 0 or int(hkl[1]) != 0:
            return int(len(entries))
    return 1


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
    tagged_candidate = geometry_manual_tagged_candidate_from_session(
        pick_session,
        remaining,
        candidate_source_key=candidate_source_key,
    )
    if tagged_candidate is not None:
        return tagged_candidate
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
    detector_col: float | None = None,
    detector_row: float | None = None,
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
        "source_peak_index": candidate.get("source_peak_index"),
        "source_label": candidate.get("source_label"),
        "q_group_key": group_key,
    }
    if detector_col is not None and detector_row is not None:
        entry["detector_x"] = float(detector_col)
        entry["detector_y"] = float(detector_row)
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
    radial_axis: Sequence[float] | None = None,
    azimuth_axis: Sequence[float] | None = None,
    caked_axis_to_image_index_fn: Callable[[float, Sequence[float] | None], float] = caked_axis_to_image_index,
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
    tagged_candidate = geometry_manual_tagged_candidate_from_session(
        pick_session,
        remaining_candidates,
    )
    if tagged_candidate is not None:
        candidate = dict(tagged_candidate)
        sim_dist = geometry_manual_candidate_distance_to_point(
            float(refined_col),
            float(refined_row),
            candidate,
        )
        candidate_relation = "tagged sim"
    else:
        candidate, sim_dist = nearest_candidate_to_point(
            float(refined_col),
            float(refined_row),
            remaining_candidates,
        )
        candidate_relation = "nearest sim"
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
    match_confidence = geometry_manual_preview_match_confidence(
        candidate,
        float(refined_col),
        float(refined_row),
        cache_data=state,
        use_caked_space=use_caked_space,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        caked_axis_to_image_index_fn=caked_axis_to_image_index_fn,
    )
    if np.isfinite(match_confidence):
        preview_color = geometry_manual_preview_confidence_color(
            float(match_confidence)
        )
        quality_label = geometry_manual_preview_confidence_quality_label(
            float(match_confidence)
        )
    else:
        preview_color = geometry_manual_preview_color(float(sigma_px))
        quality_label = geometry_manual_preview_quality_label(float(sigma_px))
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
        f"placement error={delta:.2f}px, sigma={float(sigma_px):.2f}px, "
        + (
            f"confidence={float(match_confidence):.2f}, quality={quality_label}"
            if np.isfinite(match_confidence)
            else f"quality={quality_label}"
        )
    )
    if candidate_label:
        message += f" -> {candidate_relation} [{candidate_label}]"
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
        "match_confidence": float(match_confidence),
        "preview_color": str(preview_color),
        "quality_label": str(quality_label),
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
        raw_group_key = raw_entry.get("q_group_key")
        if isinstance(raw_group_key, tuple):
            entry["q_group_key"] = raw_group_key
        elif isinstance(raw_group_key, list):
            entry["q_group_key"] = tuple(raw_group_key)
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


def build_geometry_manual_initial_pairs_display(
    background_index: int,
    *,
    param_set: dict[str, object] | None = None,
    current_background_index: object = None,
    prefer_cache: bool = False,
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    current_geometry_fit_params: Callable[[], dict[str, object]] | None = None,
    get_cache_data: Callable[..., dict[str, object]],
    simulated_peaks_for_params: Callable[..., Sequence[dict[str, object]]],
    build_simulated_lookup: Callable[
        [Sequence[dict[str, object]] | None],
        dict[tuple[object, ...], dict[str, object]],
    ],
    entry_display_coords: Callable[
        [dict[str, object] | None],
        tuple[float, float] | None,
    ],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build overlay-ready manual geometry pairs for one background image."""

    saved_entries = [dict(entry) for entry in pairs_for_index(int(background_index))]
    if not saved_entries:
        return [], []

    if isinstance(param_set, dict):
        params_local = dict(param_set)
    elif callable(current_geometry_fit_params):
        params_local = dict(current_geometry_fit_params())
    else:
        params_local = {}

    if prefer_cache and int(background_index) == int(current_background_index):
        cache_data = get_cache_data(
            param_set=params_local,
            prefer_cache=True,
            background_index=background_index,
        )
        simulated_lookup = dict(cache_data.get("simulated_lookup", {}))
    else:
        simulated_peaks = simulated_peaks_for_params(
            params_local,
            prefer_cache=False,
        )
        simulated_lookup = build_simulated_lookup(simulated_peaks)

    measured_display: list[dict[str, object]] = []
    initial_pairs_display: list[dict[str, object]] = []
    for pair_idx, entry in enumerate(saved_entries):
        measured_entry = dict(entry)
        measured_entry["overlay_match_index"] = int(pair_idx)
        measured_display.append(measured_entry)

        initial_entry: dict[str, object] = {
            "overlay_match_index": int(pair_idx),
            "hkl": entry.get("hkl", entry.get("label")),
        }
        raw_group_key = entry.get("q_group_key")
        if isinstance(raw_group_key, tuple):
            initial_entry["q_group_key"] = raw_group_key
        elif isinstance(raw_group_key, list):
            initial_entry["q_group_key"] = tuple(raw_group_key)
        bg_coords = entry_display_coords(entry)
        if bg_coords is not None:
            initial_entry["bg_display"] = (float(bg_coords[0]), float(bg_coords[1]))
        try:
            source_key = (
                int(entry.get("source_table_index")),
                int(entry.get("source_row_index")),
            )
        except Exception:
            source_key = None
        sim_entry = simulated_lookup.get(source_key) if source_key is not None else None
        sim_entry = geometry_manual_apply_refined_simulated_override(entry, sim_entry)
        if isinstance(sim_entry, dict):
            try:
                sim_col = float(sim_entry.get("sim_col"))
                sim_row = float(sim_entry.get("sim_row"))
            except Exception:
                sim_col = float("nan")
                sim_row = float("nan")
            if np.isfinite(sim_col) and np.isfinite(sim_row):
                initial_entry["sim_display"] = (float(sim_col), float(sim_row))
        initial_pairs_display.append(initial_entry)

    return measured_display, initial_pairs_display


def make_runtime_geometry_manual_cache_callbacks(
    *,
    fit_config: Mapping[str, object] | None,
    last_simulation_signature: Callable[[], object] | object,
    current_background_index: Callable[[], object] | object,
    current_background_image: Callable[[], object | None] | object | None,
    use_caked_space: Callable[[], object] | object,
    replace_cache_state: Callable[[object, dict[str, object]], None],
    current_geometry_fit_params: Callable[[], dict[str, object]] | None,
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    simulated_peaks_for_params: Callable[..., Sequence[dict[str, object]]],
    build_grouped_candidates: Callable[
        [Sequence[dict[str, object]] | None],
        dict[tuple[object, ...], list[dict[str, object]]],
    ],
    build_simulated_lookup: Callable[
        [Sequence[dict[str, object]] | None],
        dict[tuple[object, ...], dict[str, object]],
    ],
    entry_display_coords: Callable[
        [dict[str, object] | None],
        tuple[float, float] | None,
    ],
    geometry_preview_excluded_q_groups: Callable[[], object] | object = (),
    geometry_q_group_cached_entries: Callable[[], object] | object = (),
    stored_max_positions_local: Callable[[], object] | object = (),
    stored_peak_table_lattice: Callable[[], object] | object = (),
    peak_records: Callable[[], object] | object = (),
    current_cache_signature: Callable[[], object] | object = None,
    current_cache_data: Callable[[], dict[str, object] | None] | dict[str, object] | None = None,
    auto_match_background_context: Callable[
        [object, dict[str, object]],
        tuple[dict[str, object], object],
    ]
    | None = None,
) -> GeometryManualRuntimeCacheCallbacks:
    """Build live manual-geometry cache/display callbacks around shared helpers."""

    def _background_index() -> int:
        return int(_resolve_runtime_value(current_background_index))

    def _background_image() -> object | None:
        return _resolve_runtime_value(current_background_image)

    def _manual_pick_uses_caked_space() -> bool:
        return bool(_resolve_runtime_value(use_caked_space))

    def _current_match_config() -> dict[str, object]:
        return current_geometry_manual_match_config(fit_config)

    def _pick_cache_signature(
        *,
        background_index: int | None = None,
        background_image: object | None = None,
    ) -> tuple[object, ...]:
        return geometry_manual_pick_cache_signature(
            last_simulation_signature=_resolve_runtime_value(
                last_simulation_signature
            ),
            background_index=(
                _background_index()
                if background_index is None
                else int(background_index)
            ),
            background_image=(
                _background_image()
                if background_image is None
                else background_image
            ),
            use_caked_space=_manual_pick_uses_caked_space(),
            geometry_preview_excluded_q_groups=_resolve_runtime_value(
                geometry_preview_excluded_q_groups
            ),
            geometry_q_group_cached_entries=_resolve_runtime_value(
                geometry_q_group_cached_entries
            ),
            stored_max_positions_local=_resolve_runtime_value(
                stored_max_positions_local
            ),
            stored_peak_table_lattice=_resolve_runtime_value(
                stored_peak_table_lattice
            ),
            peak_records=_resolve_runtime_value(peak_records),
        )

    def _get_pick_cache(
        *,
        param_set: dict[str, object] | None = None,
        prefer_cache: bool = True,
        background_index: int | None = None,
        background_image: object | None = None,
    ) -> dict[str, object]:
        bg_index = _background_index() if background_index is None else int(
            background_index
        )
        background_local = (
            _background_image() if background_image is None else background_image
        )
        cache_data, next_signature, next_cache_data = build_geometry_manual_pick_cache(
            param_set=param_set,
            prefer_cache=prefer_cache,
            background_index=bg_index,
            current_background_index=_background_index(),
            background_image=background_local,
            existing_cache_signature=_resolve_runtime_value(current_cache_signature),
            existing_cache_data=_resolve_runtime_value(current_cache_data),
            cache_signature_fn=_pick_cache_signature,
            simulated_peaks_for_params=simulated_peaks_for_params,
            build_grouped_candidates=build_grouped_candidates,
            build_simulated_lookup=build_simulated_lookup,
            current_match_config=_current_match_config,
            auto_match_background_context=auto_match_background_context,
        )
        replace_cache_state(
            next_signature,
            dict(next_cache_data) if isinstance(next_cache_data, dict) else {},
        )
        return cache_data

    def _build_initial_pairs_display(
        background_index: int,
        *,
        param_set: dict[str, object] | None = None,
        prefer_cache: bool = False,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        return build_geometry_manual_initial_pairs_display(
            background_index,
            param_set=param_set,
            current_background_index=_background_index(),
            prefer_cache=prefer_cache,
            pairs_for_index=pairs_for_index,
            current_geometry_fit_params=current_geometry_fit_params,
            get_cache_data=_get_pick_cache,
            simulated_peaks_for_params=simulated_peaks_for_params,
            build_simulated_lookup=build_simulated_lookup,
            entry_display_coords=entry_display_coords,
        )

    return GeometryManualRuntimeCacheCallbacks(
        current_match_config=_current_match_config,
        pick_cache_signature=_pick_cache_signature,
        get_pick_cache=_get_pick_cache,
        build_initial_pairs_display=_build_initial_pairs_display,
    )


def make_runtime_geometry_manual_projection_callbacks(
    *,
    caked_view_enabled: Callable[[], object] | object,
    last_caked_background_image_unscaled: Callable[[], object] | object,
    last_caked_radial_values: Callable[[], object] | object,
    last_caked_azimuth_values: Callable[[], object] | object,
    current_background_display: Callable[[], object] | object,
    current_background_native: Callable[[], object] | object,
    ai: Callable[[], object] | object = None,
    center: Callable[[], Sequence[float] | None] | Sequence[float] | None = None,
    detector_distance: Callable[[], object] | object = 0.0,
    pixel_size: Callable[[], object] | object = 0.0,
    wrap_phi_range: Callable[[object], object] = lambda value: value,
    rotate_point_for_display: Callable[[float, float, tuple[int, ...], int], tuple[float, float]] = _default_rotate_point,
    display_rotate_k: int = 0,
    current_geometry_fit_params: Callable[[], dict[str, object]] | None = None,
    simulate_preview_style_peaks_for_fit: Callable[..., Sequence[dict[str, object]]] | None = None,
    build_live_preview_simulated_peaks_from_cache: (
        Callable[[], Sequence[dict[str, object]]]
        | None
    ) = None,
    miller: Callable[[], object] | object = None,
    intensities: Callable[[], object] | object = None,
    image_size: Callable[[], object] | object = 0,
    display_to_native_sim_coords: Callable[..., tuple[float, float]] | None = None,
    get_detector_angular_maps: Callable[[object], tuple[object, object]] = lambda _ai: (None, None),
    detector_pixel_to_scattering_angles: Callable[
        [float, float, Sequence[float] | None, float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    backend_detector_coords_to_native_detector_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    scattering_angles_to_detector_pixel: Callable[
        [float, float, Sequence[float] | None, float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    filter_simulated_peaks: Callable[..., tuple[Sequence[dict[str, object]], object, object]] | None = None,
    collapse_simulated_peaks: Callable[..., tuple[Sequence[dict[str, object]], object]] | None = None,
    merge_radius_px: float = 6.0,
) -> GeometryManualRuntimeProjectionCallbacks:
    """Build live manual-geometry projection callbacks around shared helpers."""

    def _pick_uses_caked_space() -> bool:
        if not bool(_resolve_runtime_value(caked_view_enabled)):
            return False
        background_image = _resolve_runtime_value(last_caked_background_image_unscaled)
        return (
            isinstance(background_image, np.ndarray)
            and background_image.ndim == 2
            and background_image.size > 0
            and np.asarray(
                _resolve_runtime_value(last_caked_radial_values),
                dtype=float,
            ).size
            > 1
            and np.asarray(
                _resolve_runtime_value(last_caked_azimuth_values),
                dtype=float,
            ).size
            > 1
        )

    def _current_background_image() -> object | None:
        if _pick_uses_caked_space():
            return _resolve_runtime_value(last_caked_background_image_unscaled)
        return _resolve_runtime_value(current_background_display)

    def _detector_center() -> Sequence[float] | None:
        value = _resolve_runtime_value(center)
        if value is None:
            return None
        try:
            return [float(value[0]), float(value[1])]
        except Exception:
            return None

    def _native_to_caked_display_coords(
        col: float,
        row: float,
    ) -> tuple[float, float] | None:
        if detector_pixel_to_scattering_angles is None:
            return None
        return native_detector_coords_to_caked_display_coords(
            col,
            row,
            ai=_resolve_runtime_value(ai),
            get_detector_angular_maps=get_detector_angular_maps,
            detector_pixel_to_scattering_angles=detector_pixel_to_scattering_angles,
            center=_detector_center(),
            detector_distance=float(_resolve_runtime_value(detector_distance) or 0.0),
            pixel_size=float(_resolve_runtime_value(pixel_size) or 0.0),
            wrap_phi_range=wrap_phi_range,
            caked_radial_values=_resolve_runtime_value(last_caked_radial_values),
            caked_azimuth_values=_resolve_runtime_value(last_caked_azimuth_values),
        )

    def _caked_angles_to_background_display(
        two_theta_deg: float,
        phi_deg: float,
    ) -> tuple[float | None, float | None]:
        return caked_angles_to_background_display_coords(
            two_theta_deg,
            phi_deg,
            ai=_resolve_runtime_value(ai),
            native_background=_resolve_runtime_value(current_background_native),
            get_detector_angular_maps=get_detector_angular_maps,
            scattering_angles_to_detector_pixel=scattering_angles_to_detector_pixel,
            center=_detector_center(),
            detector_distance=float(_resolve_runtime_value(detector_distance) or 0.0),
            pixel_size=float(_resolve_runtime_value(pixel_size) or 0.0),
            backend_detector_coords_to_native_detector_coords=(
                backend_detector_coords_to_native_detector_coords
            ),
            rotate_point_for_display=rotate_point_for_display,
            display_rotate_k=int(display_rotate_k),
        )

    def _background_display_to_caked_display(
        col: float,
        row: float,
    ) -> tuple[float, float] | None:
        native_point = _background_display_to_native_detector_coords(float(col), float(row))
        if native_point is None:
            return None
        return _native_to_caked_display_coords(
            float(native_point[0]),
            float(native_point[1]),
        )

    def _background_display_to_native_detector_coords(
        col: float,
        row: float,
    ) -> tuple[float, float] | None:
        native_background = _resolve_runtime_value(current_background_native)
        if native_background is None:
            return None
        try:
            shape = tuple(int(v) for v in np.asarray(native_background).shape[:2])
            native_col, native_row = rotate_point_for_display(
                float(col),
                float(row),
                shape,
                -int(display_rotate_k),
            )
        except Exception:
            return None
        if not (np.isfinite(native_col) and np.isfinite(native_row)):
            return None
        return float(native_col), float(native_row)

    def _entry_display_coords(
        entry: dict[str, object] | None,
    ) -> tuple[float, float] | None:
        if not isinstance(entry, dict):
            return None
        use_caked = _pick_uses_caked_space()
        key_x = "caked_x" if use_caked else "x"
        key_y = "caked_y" if use_caked else "y"
        try:
            col = float(entry.get(key_x, np.nan))
            row = float(entry.get(key_y, np.nan))
        except Exception:
            col = float("nan")
            row = float("nan")
        if use_caked and not (np.isfinite(col) and np.isfinite(row)):
            try:
                detector_col = float(entry.get("detector_x", np.nan))
                detector_row = float(entry.get("detector_y", np.nan))
            except Exception:
                detector_col = float("nan")
                detector_row = float("nan")
            if np.isfinite(detector_col) and np.isfinite(detector_row):
                converted = _native_to_caked_display_coords(
                    float(detector_col),
                    float(detector_row),
                )
                if converted is not None:
                    col = float(converted[0])
                    row = float(converted[1])
        if use_caked and not (np.isfinite(col) and np.isfinite(row)):
            try:
                raw_col = float(entry.get("x", np.nan))
                raw_row = float(entry.get("y", np.nan))
            except Exception:
                raw_col = float("nan")
                raw_row = float("nan")
            converted = _background_display_to_caked_display(raw_col, raw_row)
            if converted is not None:
                col = float(converted[0])
                row = float(converted[1])
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return float(col), float(row)

    def _project_peaks_to_current_view(
        simulated_peaks: Sequence[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        projected: list[dict[str, object]] = []
        use_caked = _pick_uses_caked_space()
        radial_axis = (
            np.asarray(_resolve_runtime_value(last_caked_radial_values), dtype=float)
            if use_caked
            else np.array([])
        )
        azimuth_axis = (
            np.asarray(_resolve_runtime_value(last_caked_azimuth_values), dtype=float)
            if use_caked
            else np.array([])
        )
        try:
            sim_shape = (int(_resolve_runtime_value(image_size)), int(_resolve_runtime_value(image_size)))
        except Exception:
            sim_shape = (0, 0)

        for raw_entry in simulated_peaks or []:
            if not isinstance(raw_entry, dict):
                continue
            entry = dict(raw_entry)
            try:
                raw_col = float(entry.get("sim_col", np.nan))
                raw_row = float(entry.get("sim_row", np.nan))
            except Exception:
                raw_col = float("nan")
                raw_row = float("nan")
            entry["sim_col_raw"] = float(raw_col)
            entry["sim_row_raw"] = float(raw_row)

            if np.isfinite(raw_col) and np.isfinite(raw_row):
                native_point = None
                if callable(display_to_native_sim_coords):
                    try:
                        native_point = display_to_native_sim_coords(
                            float(raw_col),
                            float(raw_row),
                            sim_shape,
                        )
                    except Exception:
                        native_point = None
                if (
                    isinstance(native_point, tuple)
                    and len(native_point) >= 2
                    and np.isfinite(float(native_point[0]))
                    and np.isfinite(float(native_point[1]))
                ):
                    caked_coords = _native_to_caked_display_coords(
                        float(native_point[0]),
                        float(native_point[1]),
                    )
                    if caked_coords is not None:
                        entry["caked_x"] = float(caked_coords[0])
                        entry["caked_y"] = float(caked_coords[1])

            if use_caked:
                try:
                    caked_col = float(entry.get("caked_x", np.nan))
                    caked_row = float(entry.get("caked_y", np.nan))
                except Exception:
                    caked_col = float("nan")
                    caked_row = float("nan")
                if np.isfinite(caked_col) and np.isfinite(caked_row):
                    entry["sim_col"] = float(caked_col)
                    entry["sim_row"] = float(caked_row)
                    entry["sim_col_global"] = float(caked_col)
                    entry["sim_row_global"] = float(caked_row)
                    entry["sim_col_local"] = float(
                        caked_axis_to_image_index(caked_col, radial_axis)
                    )
                    entry["sim_row_local"] = float(
                        caked_axis_to_image_index(caked_row, azimuth_axis)
                    )

            projected.append(entry)
        return projected

    def _simulated_peaks_for_params(
        param_set: dict[str, object] | None = None,
        *,
        prefer_cache: bool = True,
    ) -> list[dict[str, object]]:
        if prefer_cache and callable(build_live_preview_simulated_peaks_from_cache):
            try:
                cached_peaks = [
                    dict(entry)
                    for entry in (build_live_preview_simulated_peaks_from_cache() or ())
                    if isinstance(entry, dict)
                ]
            except Exception:
                cached_peaks = []
            if cached_peaks:
                return _project_peaks_to_current_view(cached_peaks)
        if not callable(simulate_preview_style_peaks_for_fit):
            return []
        try:
            params_local = (
                dict(param_set)
                if isinstance(param_set, dict)
                else (
                    dict(current_geometry_fit_params())
                    if callable(current_geometry_fit_params)
                    else {}
                )
            )
            miller_array = np.asarray(_resolve_runtime_value(miller), dtype=np.float64)
            intensity_array = np.asarray(
                _resolve_runtime_value(intensities),
                dtype=np.float64,
            )
            image_size_value = int(_resolve_runtime_value(image_size))
            if (
                miller_array.ndim != 2
                or miller_array.shape[1] != 3
                or miller_array.size == 0
            ):
                return []
            if intensity_array.shape[0] != miller_array.shape[0]:
                return []
            raw_peaks = simulate_preview_style_peaks_for_fit(
                miller_array,
                intensity_array,
                image_size_value,
                params_local,
            )
            return _project_peaks_to_current_view(raw_peaks)
        except Exception:
            return []

    def _pick_candidates(
        simulated_peaks: Sequence[dict[str, object]] | None,
    ) -> dict[tuple[object, ...], list[dict[str, object]]]:
        def _group_entries(
            candidate_entries: Sequence[dict[str, object]] | None,
        ) -> dict[tuple[object, ...], list[dict[str, object]]]:
            grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
            for raw_entry in candidate_entries or []:
                if not isinstance(raw_entry, dict):
                    continue
                group_key = raw_entry.get("q_group_key")
                if not isinstance(group_key, tuple):
                    continue
                grouped[group_key].append(dict(raw_entry))
            for entry_list in grouped.values():
                entry_list.sort(
                    key=lambda entry: (
                        float(entry.get("sim_col", np.nan))
                        if np.isfinite(float(entry.get("sim_col", np.nan)))
                        else float("inf"),
                        float(entry.get("sim_row", np.nan))
                        if np.isfinite(float(entry.get("sim_row", np.nan)))
                        else float("inf"),
                    )
                )
            return dict(grouped)

        filtered_entries = list(simulated_peaks or [])
        if callable(filter_simulated_peaks):
            filtered_result = filter_simulated_peaks(simulated_peaks)
            if isinstance(filtered_result, tuple) and filtered_result:
                filtered_candidate_entries = list(filtered_result[0] or [])
                # Manual picking should still be able to target the nearest
                # central-beam Qr/Qz group when the selector filter is stale
                # or currently excludes every available group.
                if _group_entries(filtered_candidate_entries):
                    filtered_entries = filtered_candidate_entries
        collapsed_entries = list(filtered_entries)
        if callable(collapse_simulated_peaks):
            collapsed_result = collapse_simulated_peaks(
                filtered_entries,
                merge_radius_px=float(merge_radius_px),
            )
            if isinstance(collapsed_result, tuple) and collapsed_result:
                collapsed_candidate_entries = list(collapsed_result[0] or [])
                if _group_entries(collapsed_candidate_entries):
                    collapsed_entries = collapsed_candidate_entries

        return _group_entries(collapsed_entries)

    def _simulated_lookup(
        simulated_peaks: Sequence[dict[str, object]] | None,
    ) -> dict[tuple[int, int], dict[str, object]]:
        lookup: dict[tuple[int, int], dict[str, object]] = {}
        for raw_entry in simulated_peaks or []:
            if not isinstance(raw_entry, dict):
                continue
            try:
                key = (
                    int(raw_entry.get("source_table_index")),
                    int(raw_entry.get("source_row_index")),
                )
            except Exception:
                continue
            lookup[key] = dict(raw_entry)
        return lookup

    return GeometryManualRuntimeProjectionCallbacks(
        pick_uses_caked_space=_pick_uses_caked_space,
        current_background_image=_current_background_image,
        entry_display_coords=_entry_display_coords,
        caked_angles_to_background_display_coords=(
            _caked_angles_to_background_display
        ),
        background_display_to_native_detector_coords=(
            _background_display_to_native_detector_coords
        ),
        native_detector_coords_to_caked_display_coords=(
            _native_to_caked_display_coords
        ),
        project_peaks_to_current_view=_project_peaks_to_current_view,
        simulated_peaks_for_params=_simulated_peaks_for_params,
        pick_candidates=_pick_candidates,
        simulated_lookup=_simulated_lookup,
    )


def render_current_geometry_manual_pairs(
    *,
    background_visible: bool,
    current_background_index: int,
    current_background_image: object | None,
    pick_session: dict[str, object] | None,
    build_initial_pairs_display: Callable[
        ...,
        tuple[list[dict[str, object]], list[dict[str, object]]],
    ],
    session_initial_pairs_display: Callable[[], Sequence[dict[str, object]]],
    clear_geometry_pick_artists: Callable[..., None],
    draw_initial_geometry_pairs_overlay: Callable[..., None],
    update_button_label_fn: Callable[[], None],
    set_background_file_status_text_fn: Callable[[], None],
    pair_group_count: Callable[[int], int],
    set_status_text: Callable[[str], None] | None = None,
    update_status: bool = False,
) -> bool:
    """Redraw the saved manual geometry-pair overlay for the current background."""

    if not background_visible or current_background_image is None:
        clear_geometry_pick_artists()
        return False

    measured_display, initial_pairs_display = build_initial_pairs_display(
        int(current_background_index),
        prefer_cache=True,
    )
    pending_pairs_display = list(session_initial_pairs_display())
    combined_pairs_display = list(initial_pairs_display) + list(pending_pairs_display)

    if not measured_display and not combined_pairs_display:
        clear_geometry_pick_artists()
        if update_status and callable(set_status_text):
            set_status_text("No saved manual geometry pairs for the current background image.")
        return False

    draw_initial_geometry_pairs_overlay(
        combined_pairs_display,
        max_display_markers=max(1, len(combined_pairs_display)),
    )
    update_button_label_fn()
    set_background_file_status_text_fn()

    if update_status and callable(set_status_text):
        if geometry_manual_pick_session_active(
            pick_session,
            current_background_index=current_background_index,
        ):
            pending_entries = pick_session.get("pending_entries", []) if isinstance(pick_session, dict) else []
            target_count = pick_session.get("target_count") if isinstance(pick_session, dict) else None
            q_label = str(
                pick_session.get(
                    "q_label",
                    pick_session.get("group_key", "selected Qr/Qz set"),
                )
                if isinstance(pick_session, dict)
                else "selected Qr/Qz set"
            )
            next_index = len(pending_entries) + 1 if isinstance(pending_entries, list) else 1
            try:
                total_count = int(target_count)
            except Exception:
                total_count = len(
                    pick_session.get("group_entries", [])
                    if isinstance(pick_session, dict)
                    else []
                )
            remaining_candidates = geometry_manual_unassigned_group_candidates(
                pick_session,
                current_background_index=current_background_index,
            )
            tagged_candidate = geometry_manual_tagged_candidate_from_session(
                pick_session,
                remaining_candidates,
            )
            set_status_text(
                f"Click background peak {next_index} of {max(1, total_count)} for {q_label}. "
                + (
                    "It will attach to the tagged central-beam seed."
                    if tagged_candidate is not None
                    else "It will attach to the nearest remaining simulated peak."
                )
            )
        else:
            set_status_text(
                "Current background has "
                f"{len(initial_pairs_display)} saved manual points across "
                f"{pair_group_count(int(current_background_index))} Qr/Qz groups."
            )
    return True


def make_runtime_geometry_manual_callbacks(
    *,
    background_visible: Callable[[], object] | object,
    current_background_index: Callable[[], object] | object,
    current_background_image: Callable[[], object | None] | object | None,
    pick_session: Callable[[], dict[str, object] | None] | dict[str, object] | None,
    build_initial_pairs_display: Callable[
        ...,
        tuple[list[dict[str, object]], list[dict[str, object]]],
    ],
    session_initial_pairs_display: Callable[[], Sequence[dict[str, object]]],
    clear_geometry_pick_artists: Callable[..., None],
    draw_initial_geometry_pairs_overlay: Callable[..., None],
    update_button_label: Callable[[], None],
    set_background_file_status_text: Callable[[], None],
    pair_group_count: Callable[[int], int],
    set_status_text: Callable[[str], None] | None,
    get_cache_data: Callable[..., dict[str, object]],
    set_pairs_for_index: Callable[
        [int, Sequence[dict[str, object]] | None],
        Sequence[dict[str, object]],
    ],
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    set_pick_session: Callable[[dict[str, object]], None],
    restore_view: Callable[..., None],
    clear_preview_artists: Callable[..., None],
    push_undo_state: Callable[[], None] | None = None,
    listed_q_group_entries: Callable[[], Sequence[dict[str, object]]] | Sequence[dict[str, object]] = (),
    format_q_group_line: Callable[[dict[str, object]], str] | None = None,
    use_caked_space: Callable[[], object] | object = False,
    pick_search_window_px: float,
    caked_search_tth_deg: float = DEFAULT_CAKED_SEARCH_TTH_DEG,
    caked_search_phi_deg: float = DEFAULT_CAKED_SEARCH_PHI_DEG,
    set_suppress_drag_press_once: Callable[[bool], None] | None = None,
    sync_peak_selection_state: Callable[[], None] | None = None,
    refine_preview_point: Callable[..., tuple[float, float]] | None = None,
    remaining_candidates: Callable[[], Sequence[dict[str, object]]] | None = None,
    preview_due: Callable[[float, float], bool] | None = None,
    nearest_candidate_to_point: Callable[
        [float, float, Sequence[dict[str, object]] | None],
        tuple[dict[str, object] | None, float],
    ] = geometry_manual_nearest_candidate_to_point,
    position_error_px: Callable[
        [float, float, float, float],
        float,
    ] = geometry_manual_position_error_px,
    position_sigma_px: Callable[[object], float] = geometry_manual_position_sigma_px,
    caked_angles_to_background_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    caked_axis_to_image_index_fn: Callable[[float, Sequence[float] | None], float] = caked_axis_to_image_index,
    last_caked_radial_values: Callable[[], object] | object = (),
    last_caked_azimuth_values: Callable[[], object] | object = (),
    background_display_to_native_detector_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    show_preview: Callable[..., None] | None = None,
) -> GeometryManualRuntimeCallbacks:
    """Build live manual-geometry callbacks around the shared helper surface."""

    restore_view_callback = restore_view

    def _background_index() -> int:
        return int(_resolve_runtime_value(current_background_index))

    def _background_image() -> object | None:
        return _resolve_runtime_value(current_background_image)

    def _pick_session() -> dict[str, object] | None:
        session = _resolve_runtime_value(pick_session)
        return session if isinstance(session, dict) else None

    def _use_caked_space() -> bool:
        return bool(_resolve_runtime_value(use_caked_space))

    def _render_current_pairs(*, update_status: bool = False) -> bool:
        return render_current_geometry_manual_pairs(
            background_visible=bool(_resolve_runtime_value(background_visible)),
            current_background_index=_background_index(),
            current_background_image=_background_image(),
            pick_session=_pick_session(),
            build_initial_pairs_display=build_initial_pairs_display,
            session_initial_pairs_display=session_initial_pairs_display,
            clear_geometry_pick_artists=clear_geometry_pick_artists,
            draw_initial_geometry_pairs_overlay=draw_initial_geometry_pairs_overlay,
            update_button_label_fn=update_button_label,
            set_background_file_status_text_fn=set_background_file_status_text,
            pair_group_count=pair_group_count,
            set_status_text=set_status_text,
            update_status=update_status,
        )

    def _toggle_selection_at(col: float, row: float) -> bool:
        handled, _next_session, suppress_drag = geometry_manual_toggle_selection_at(
            float(col),
            float(row),
            pick_session=_pick_session(),
            current_background_index=_background_index(),
            display_background=_background_image(),
            get_cache_data=get_cache_data,
            pairs_for_index=pairs_for_index,
            set_pairs_for_index_fn=set_pairs_for_index,
            set_pick_session_fn=set_pick_session,
            restore_view_fn=restore_view,
            clear_preview_artists_fn=clear_preview_artists,
            render_current_pairs_fn=_render_current_pairs,
            update_button_label_fn=update_button_label,
            set_status_text=set_status_text,
            push_undo_state_fn=push_undo_state,
            listed_q_group_entries=listed_q_group_entries,
            format_q_group_line=format_q_group_line,
            use_caked_space=_use_caked_space(),
            pick_search_window_px=float(pick_search_window_px),
            caked_search_tth_deg=float(caked_search_tth_deg),
            caked_search_phi_deg=float(caked_search_phi_deg),
        )
        if callable(set_suppress_drag_press_once):
            set_suppress_drag_press_once(bool(suppress_drag))
        if callable(sync_peak_selection_state):
            sync_peak_selection_state()
        return bool(handled)

    def _place_selection_at(col: float, row: float) -> bool:
        handled, _next_session = geometry_manual_place_selection_at(
            float(col),
            float(row),
            pick_session=_pick_session(),
            current_background_index=_background_index(),
            display_background=_background_image(),
            get_cache_data=get_cache_data,
            refine_preview_point=refine_preview_point,
            set_pairs_for_index_fn=set_pairs_for_index,
            set_pick_session_fn=set_pick_session,
            clear_preview_artists_fn=clear_preview_artists,
            restore_view_fn=restore_view,
            render_current_pairs_fn=_render_current_pairs,
            update_button_label_fn=update_button_label,
            set_status_text=set_status_text,
            push_undo_state_fn=push_undo_state,
            use_caked_space=_use_caked_space(),
            caked_angles_to_background_display_coords=(
                caked_angles_to_background_display_coords
            ),
            background_display_to_native_detector_coords=(
                background_display_to_native_detector_coords
            ),
            nearest_candidate_to_point_fn=nearest_candidate_to_point,
            position_error_px=position_error_px,
            position_sigma_px=position_sigma_px,
        )
        return bool(handled)

    def _update_pick_preview(raw_col: float, raw_row: float, *, force: bool = False) -> None:
        preview_state = geometry_manual_pick_preview_state(
            float(raw_col),
            float(raw_row),
            pick_session=_pick_session(),
            current_background_index=_background_index(),
            force=bool(force),
            remaining_candidates=(
                list(remaining_candidates()) if callable(remaining_candidates) else []
            ),
            display_background=_background_image(),
            build_cache_data=(get_cache_data if callable(get_cache_data) else None),
            refine_preview_point=refine_preview_point,
            preview_due=preview_due,
            nearest_candidate_to_point=nearest_candidate_to_point,
            position_error_px=position_error_px,
            position_sigma_px=position_sigma_px,
            use_caked_space=_use_caked_space(),
            caked_angles_to_background_display_coords=(
                caked_angles_to_background_display_coords
            ),
            radial_axis=np.asarray(
                _resolve_runtime_value(last_caked_radial_values),
                dtype=float,
            ),
            azimuth_axis=np.asarray(
                _resolve_runtime_value(last_caked_azimuth_values),
                dtype=float,
            ),
            caked_axis_to_image_index_fn=caked_axis_to_image_index_fn,
        )
        if preview_state is None:
            return
        if callable(show_preview):
            show_preview(
                float(preview_state["raw_col"]),
                float(preview_state["raw_row"]),
                float(preview_state["refined_col"]),
                float(preview_state["refined_row"]),
                delta_px=float(preview_state["delta"]),
                sigma_px=float(preview_state["sigma_px"]),
                preview_color=str(preview_state["preview_color"]),
            )
        if callable(set_status_text):
            set_status_text(str(preview_state["message"]))

    def _cancel_pick_session(
        *,
        restore_view: bool = True,
        redraw: bool = True,
        message: str | None = None,
    ) -> None:
        set_pick_session(
            cancel_geometry_manual_pick_session(
                _pick_session(),
                current_background_index=_background_index(),
                restore_view_fn=restore_view_callback,
                clear_preview_artists_fn=clear_preview_artists,
                render_current_pairs_fn=_render_current_pairs,
                update_button_label_fn=update_button_label,
                set_status_text=set_status_text,
                restore_view=restore_view,
                redraw=redraw,
                message=message,
            )
        )

    return GeometryManualRuntimeCallbacks(
        render_current_pairs=_render_current_pairs,
        toggle_selection_at=_toggle_selection_at,
        place_selection_at=_place_selection_at,
        update_pick_preview=_update_pick_preview,
        cancel_pick_session=_cancel_pick_session,
    )


def geometry_manual_pick_button_label(
    *,
    armed: bool,
    current_background_index: object,
    pick_session: dict[str, object] | None,
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    pair_group_count: Callable[[int], int],
    base_label: str = "Pick Qr Sets on Image",
) -> str:
    """Return the manual-geometry button label for the current GUI state."""

    label = str(base_label)
    if armed:
        label += " (Armed)"
    try:
        pair_count = len(pairs_for_index(int(current_background_index)))
        group_count = pair_group_count(int(current_background_index))
    except Exception:
        pair_count = 0
        group_count = 0
    if group_count > 0 or pair_count > 0:
        label += f" [{group_count} groups/{pair_count} pts]"
    if geometry_manual_pick_session_active(
        pick_session,
        current_background_index=current_background_index,
    ):
        pending_entries = pick_session.get("pending_entries", []) if isinstance(pick_session, dict) else []
        target_count = pick_session.get("target_count") if isinstance(pick_session, dict) else None
        if isinstance(pending_entries, list):
            try:
                total_count = int(target_count)
            except Exception:
                total_count = len(
                    pick_session.get("group_entries", [])
                    if isinstance(pick_session, dict)
                    else []
                )
            label += f" <placing {len(pending_entries)}/{max(0, total_count)}>"
    return label


def geometry_manual_toggle_selection_at(
    col: float,
    row: float,
    *,
    pick_session: dict[str, object] | None,
    current_background_index: int,
    display_background: object | None,
    get_cache_data: Callable[..., dict[str, object]],
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    set_pairs_for_index_fn: Callable[[int, Sequence[dict[str, object]] | None], Sequence[dict[str, object]]],
    set_pick_session_fn: Callable[[dict[str, object]], None],
    restore_view_fn: Callable[..., None],
    clear_preview_artists_fn: Callable[..., None],
    render_current_pairs_fn: Callable[..., None],
    update_button_label_fn: Callable[[], None],
    set_status_text: Callable[[str], None] | None = None,
    push_undo_state_fn: Callable[[], None] | None = None,
    listed_q_group_entries: Callable[[], Sequence[dict[str, object]]] | Sequence[dict[str, object]] = (),
    format_q_group_line: Callable[[dict[str, object]], str] | None = None,
    choose_group_at_fn: Callable[..., tuple[tuple[object, ...] | None, list[dict[str, object]], float]] = geometry_manual_choose_group_at,
    nearest_candidate_to_point_fn: Callable[
        [float, float, Sequence[dict[str, object]] | None],
        tuple[dict[str, object] | None, float],
    ] = geometry_manual_nearest_candidate_to_point,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
    group_target_count_fn: Callable[
        [tuple[object, ...] | None, Sequence[dict[str, object]] | None],
        int,
    ] = geometry_manual_group_target_count,
    use_caked_space: bool,
    pick_search_window_px: float,
    caked_search_tth_deg: float = DEFAULT_CAKED_SEARCH_TTH_DEG,
    caked_search_phi_deg: float = DEFAULT_CAKED_SEARCH_PHI_DEG,
) -> tuple[bool, dict[str, object], bool]:
    """Select one manual Qr/Qz group and arm background-point placement."""

    current_session = dict(pick_session) if isinstance(pick_session, dict) else {}
    if display_background is None:
        if callable(set_status_text):
            set_status_text("No background image is loaded for manual geometry picking.")
        return False, current_session, False

    cache_data = get_cache_data(background_image=display_background)
    grouped_candidates = dict(cache_data.get("grouped_candidates", {}))
    if not grouped_candidates:
        if callable(set_status_text):
            set_status_text(
                "No simulated Qr/Qz groups are available to pick. Run a simulation update first."
            )
        return False, current_session, False

    group_window = (
        float(pick_search_window_px)
        if not use_caked_space
        else float(max(caked_search_phi_deg, 2.0 * caked_search_tth_deg))
    )
    best_group_key, best_group_entries, best_dist = choose_group_at_fn(
        grouped_candidates,
        float(col),
        float(row),
        window_size_px=float(group_window),
    )
    if best_group_key is None:
        if callable(set_status_text):
            set_status_text(
                "No Qr/Qz set found within a "
                f"{group_window:.1f}x"
                f"{group_window:.1f}{' deg' if use_caked_space else 'px'} "
                "window around the clicked position."
            )
        return False, current_session, False

    existing_entries = [dict(entry) for entry in pairs_for_index(int(current_background_index))]
    if any(entry.get("q_group_key") == best_group_key for entry in existing_entries):
        if callable(push_undo_state_fn):
            push_undo_state_fn()
        restore_view_fn(redraw=False)
        clear_preview_artists_fn(redraw=False)
        set_pick_session_fn({})
        remaining_entries = [
            entry for entry in existing_entries if entry.get("q_group_key") != best_group_key
        ]
        set_pairs_for_index_fn(int(current_background_index), remaining_entries)
        render_current_pairs_fn(update_status=False)
        update_button_label_fn()
        if callable(set_status_text):
            set_status_text(
                f"Removed one saved Qr/Qz set from background {int(current_background_index) + 1}."
            )
        return True, {}, False

    q_entries = listed_q_group_entries() if callable(listed_q_group_entries) else listed_q_group_entries
    q_entry = next(
        (
            entry
            for entry in q_entries
            if isinstance(entry, dict) and entry.get("key") == best_group_key
        ),
        None,
    )
    q_label = (
        format_q_group_line(q_entry)
        if isinstance(q_entry, dict) and callable(format_q_group_line)
        else f"group={best_group_key}"
    )
    target_count = group_target_count_fn(
        best_group_key,
        best_group_entries,
    )
    tagged_candidate, tagged_dist = nearest_candidate_to_point_fn(
        float(col),
        float(row),
        best_group_entries,
    )
    tagged_group_entries = geometry_manual_prioritize_candidate_entries(
        best_group_entries,
        tagged_candidate,
        candidate_source_key=candidate_source_key,
    )
    tagged_candidate_key = candidate_source_key(tagged_candidate)
    tagged_label = (
        str(tagged_candidate.get("label", ""))
        if isinstance(tagged_candidate, dict)
        else ""
    )
    next_session = {
        "background_index": int(current_background_index),
        "group_key": best_group_key,
        "group_entries": [dict(entry) for entry in tagged_group_entries],
        "pending_entries": [],
        "target_count": int(target_count),
        "base_entries": [
            entry for entry in existing_entries if entry.get("q_group_key") != best_group_key
        ],
        "cache_signature": cache_data.get("signature") if isinstance(cache_data, dict) else None,
        "q_label": q_label,
        "zoom_active": False,
        "zoom_center": None,
        "saved_xlim": None,
        "saved_ylim": None,
        "preview_last_t": 0.0,
        "preview_last_xy": None,
    }
    if tagged_candidate_key is not None:
        next_session["tagged_candidate_key"] = tagged_candidate_key
    if isinstance(tagged_candidate, dict):
        next_session["tagged_candidate"] = dict(tagged_candidate)
    set_pick_session_fn(next_session)
    render_current_pairs_fn(update_status=False)
    update_button_label_fn()
    if callable(set_status_text):
        seed_dist = tagged_dist if np.isfinite(tagged_dist) else best_dist
        set_status_text(
            f"Selected {q_label} (nearest Bragg seed {seed_dist:.1f}{' deg' if use_caked_space else 'px'}). "
            + (f"Tagged seed [{tagged_label}]. " if tagged_label else "")
            + f"Click background peak 1 of {max(1, int(target_count))}; "
            + (
                "it will attach to that tagged simulated peak."
                if tagged_label
                else "it will be assigned to the nearest simulated peak."
            )
        )
    return True, next_session, True


def geometry_manual_place_selection_at(
    col: float,
    row: float,
    *,
    pick_session: dict[str, object] | None,
    current_background_index: object,
    display_background: object | None,
    get_cache_data: Callable[..., dict[str, object]],
    refine_preview_point: Callable[..., tuple[float, float]],
    set_pairs_for_index_fn: Callable[[int, Sequence[dict[str, object]] | None], Sequence[dict[str, object]]],
    set_pick_session_fn: Callable[[dict[str, object]], None],
    clear_preview_artists_fn: Callable[..., None],
    restore_view_fn: Callable[..., None],
    render_current_pairs_fn: Callable[..., None],
    update_button_label_fn: Callable[[], None],
    set_status_text: Callable[[str], None] | None = None,
    push_undo_state_fn: Callable[[], None] | None = None,
    use_caked_space: bool,
    caked_angles_to_background_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    background_display_to_native_detector_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
    nearest_candidate_to_point_fn: Callable[
        [float, float, Sequence[dict[str, object]] | None],
        tuple[dict[str, object] | None, float],
    ] = geometry_manual_nearest_candidate_to_point,
    pair_entry_from_candidate_fn: Callable[..., dict[str, object] | None] = geometry_manual_pair_entry_from_candidate,
    position_error_px: Callable[[float, float, float, float], float] = geometry_manual_position_error_px,
    position_sigma_px: Callable[[object], float] = geometry_manual_position_sigma_px,
) -> tuple[bool, dict[str, object]]:
    """Record the next manual background point for the active Qr/Qz pick session."""

    current_session = dict(pick_session) if isinstance(pick_session, dict) else {}
    if not geometry_manual_pick_session_active(
        pick_session,
        current_background_index=current_background_index,
    ):
        return False, current_session
    if display_background is None:
        if callable(set_status_text):
            set_status_text("No background image is loaded for manual geometry picking.")
        return False, current_session

    remaining_candidates = geometry_manual_unassigned_group_candidates(
        pick_session,
        current_background_index=current_background_index,
        candidate_source_key=candidate_source_key,
    )
    if not remaining_candidates:
        restore_view_fn(redraw=False)
        clear_preview_artists_fn(redraw=False)
        set_pick_session_fn({})
        render_current_pairs_fn(update_status=False)
        update_button_label_fn()
        if callable(set_status_text):
            set_status_text(
                "Manual geometry picking had no remaining simulated peaks to place."
            )
        return False, {}

    cache_data = get_cache_data(background_image=display_background)
    peak_col, peak_row = refine_preview_point(
        None,
        float(col),
        float(row),
        display_background=display_background,
        cache_data=cache_data,
    )
    tagged_candidate = geometry_manual_tagged_candidate_from_session(
        pick_session,
        remaining_candidates,
        candidate_source_key=candidate_source_key,
    )
    if tagged_candidate is not None:
        candidate = tagged_candidate
        candidate_dist = geometry_manual_candidate_distance_to_point(
            float(peak_col),
            float(peak_row),
            candidate,
        )
    else:
        candidate, candidate_dist = nearest_candidate_to_point_fn(
            float(peak_col),
            float(peak_row),
            remaining_candidates,
        )
    if candidate is None:
        if callable(set_status_text):
            set_status_text(
                "Manual geometry picking could not find an unassigned simulated peak for that background point."
            )
        return False, current_session

    pair_kwargs: dict[str, object] = {
        "peak_col": float(peak_col),
        "peak_row": float(peak_row),
        "raw_col": float(col),
        "raw_row": float(row),
    }
    placement_error_px_value = position_error_px(
        float(col),
        float(row),
        float(peak_col),
        float(peak_row),
    )
    if use_caked_space:
        raw_display = (
            caked_angles_to_background_display_coords(float(col), float(row))
            if callable(caked_angles_to_background_display_coords)
            else (None, None)
        )
        peak_display = (
            caked_angles_to_background_display_coords(float(peak_col), float(peak_row))
            if callable(caked_angles_to_background_display_coords)
            else (None, None)
        )
        if (
            raw_display[0] is None
            or raw_display[1] is None
            or peak_display[0] is None
            or peak_display[1] is None
        ):
            if callable(set_status_text):
                set_status_text(
                    "Manual geometry picking could not back-project the selected caked peak onto the detector."
                )
            return False, current_session
        placement_error_px_value = position_error_px(
            float(raw_display[0]),
            float(raw_display[1]),
            float(peak_display[0]),
            float(peak_display[1]),
        )
        pair_kwargs = {
            "peak_col": float(peak_display[0]),
            "peak_row": float(peak_display[1]),
            "raw_col": float(raw_display[0]),
            "raw_row": float(raw_display[1]),
            "caked_col": float(peak_col),
            "caked_row": float(peak_row),
            "raw_caked_col": float(col),
            "raw_caked_row": float(row),
        }

    sigma_px_value = position_sigma_px(float(placement_error_px_value))
    detector_anchor = (
        background_display_to_native_detector_coords(
            float(pair_kwargs["peak_col"]),
            float(pair_kwargs["peak_row"]),
        )
        if callable(background_display_to_native_detector_coords)
        else None
    )
    detector_col = None
    detector_row = None
    if (
        isinstance(detector_anchor, tuple)
        and len(detector_anchor) >= 2
        and np.isfinite(float(detector_anchor[0]))
        and np.isfinite(float(detector_anchor[1]))
    ):
        detector_col = float(detector_anchor[0])
        detector_row = float(detector_anchor[1])
    elif (
        np.isfinite(float(pair_kwargs["peak_col"]))
        and np.isfinite(float(pair_kwargs["peak_row"]))
    ):
        detector_col = float(pair_kwargs["peak_col"])
        detector_row = float(pair_kwargs["peak_row"])
    pair_entry = pair_entry_from_candidate_fn(
        candidate,
        float(pair_kwargs["peak_col"]),
        float(pair_kwargs["peak_row"]),
        group_key=pick_session.get("group_key") if isinstance(pick_session, dict) else None,
        detector_col=detector_col,
        detector_row=detector_row,
        raw_col=float(pair_kwargs["raw_col"]),
        raw_row=float(pair_kwargs["raw_row"]),
        caked_col=(
            float(pair_kwargs["caked_col"])
            if "caked_col" in pair_kwargs
            else None
        ),
        caked_row=(
            float(pair_kwargs["caked_row"])
            if "caked_row" in pair_kwargs
            else None
        ),
        raw_caked_col=(
            float(pair_kwargs["raw_caked_col"])
            if "raw_caked_col" in pair_kwargs
            else None
        ),
        raw_caked_row=(
            float(pair_kwargs["raw_caked_row"])
            if "raw_caked_row" in pair_kwargs
            else None
        ),
        placement_error_px=float(placement_error_px_value),
        sigma_px=float(sigma_px_value),
    )
    if pair_entry is None:
        if callable(set_status_text):
            set_status_text("Failed to build the manual geometry pair entry.")
        return False, current_session

    if callable(push_undo_state_fn):
        push_undo_state_fn()

    next_session = dict(current_session)
    pending_entries = next_session.get("pending_entries", [])
    if not isinstance(pending_entries, list):
        pending_entries = []
    pending_entries = [dict(entry) for entry in pending_entries if isinstance(entry, dict)]
    pending_entries.append(pair_entry)
    next_session["pending_entries"] = pending_entries

    try:
        total_count = int(next_session.get("target_count", 0))
    except Exception:
        total_count = 0
    if total_count <= 0:
        group_entries = next_session.get("group_entries", [])
        total_count = len(group_entries) if isinstance(group_entries, list) else len(pending_entries)
    placed_count = len(pending_entries)
    q_label = str(
        next_session.get(
            "q_label",
            next_session.get("group_key", "selected Qr/Qz set"),
        )
    )
    candidate_label = str(candidate.get("label", "")) if isinstance(candidate, dict) else ""

    clear_preview_artists_fn(redraw=False)
    restore_view_fn(redraw=False)
    next_session["zoom_active"] = False
    next_session["zoom_center"] = None
    next_session["saved_xlim"] = None
    next_session["saved_ylim"] = None

    if placed_count >= total_count:
        base_entries = next_session.get("base_entries", [])
        updated_entries = list(base_entries) if isinstance(base_entries, list) else []
        updated_entries.extend(dict(entry) for entry in pending_entries if isinstance(entry, dict))
        set_pick_session_fn({})
        set_pairs_for_index_fn(int(current_background_index), updated_entries)
        render_current_pairs_fn(update_status=False)
        update_button_label_fn()
        if callable(set_status_text):
            set_status_text(
                f"Saved {placed_count} manual background points for {q_label} "
                f"on background {int(current_background_index) + 1}. "
                f"Last placement error={float(placement_error_px_value):.2f}px, sigma={float(sigma_px_value):.2f}px."
            )
        return True, {}

    set_pick_session_fn(next_session)
    render_current_pairs_fn(update_status=False)
    update_button_label_fn()
    next_index = placed_count + 1
    if callable(set_status_text):
        set_status_text(
            f"Placed peak {placed_count} of {total_count} for {q_label}. "
            + (
                f"Assigned to {candidate_label}"
                + (
                    f" ({float(candidate_dist):.2f}{' deg' if use_caked_space else 'px'} from sim)."
                    if np.isfinite(candidate_dist)
                    else "."
                )
                if candidate_label
                else ""
            )
            + " "
            + f"Placement error={float(placement_error_px_value):.2f}px, sigma={float(sigma_px_value):.2f}px. "
            + f"Click background peak {next_index} of {total_count}; it will be assigned to the nearest remaining simulated peak."
        )
    return True, next_session


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

    gui_controllers.clear_tk_after_token(root, integration_update_pending)
    integration_update_pending = None
    gui_controllers.clear_tk_after_token(root, update_pending)
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
    ]
    | None,
    center: Sequence[float] | None,
    detector_distance: float,
    pixel_size: float,
    backend_detector_coords_to_native_detector_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    rotate_point_for_display: Callable[[float, float, tuple[int, ...], int], tuple[float, float]] = _default_rotate_point,
    display_rotate_k: int = 0,
) -> tuple[float | None, float | None]:
    """Back-project one caked-space point to the displayed detector background.

    This path is intentionally position-only for manual QR/QZ picking. If a
    future caller inverts caked intensities or a full caked image, remember
    that the caked data may already be solid-angle corrected. Restore the
    detector solid-angle weighting before undoing backend orientation so the
    reconstructed detector intensities remain in detector-count space.
    """

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
        if not callable(scattering_angles_to_detector_pixel):
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
    native_col = float(col_idx)
    native_row = float(row_idx)
    # QR/QZ selection only needs the detector position. If this inverse path is
    # ever reused for intensity arrays, reapply detector-side weighting (such as
    # the solid-angle map) before any backend->native image reorientation.
    if callable(backend_detector_coords_to_native_detector_coords):
        try:
            native_point = backend_detector_coords_to_native_detector_coords(
                float(col_idx),
                float(row_idx),
            )
        except Exception:
            native_point = None
        if (
            isinstance(native_point, tuple)
            and len(native_point) >= 2
            and native_point[0] is not None
            and native_point[1] is not None
            and np.isfinite(float(native_point[0]))
            and np.isfinite(float(native_point[1]))
        ):
            native_col = float(native_point[0])
            native_row = float(native_point[1])
    return rotate_point_for_display(
        float(native_col),
        float(native_row),
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
    caked_radial_values: Sequence[float] | None = None,
    caked_azimuth_values: Sequence[float] | None = None,
) -> tuple[float, float] | None:
    """Project one native detector pixel into the active caked display axes."""

    def _snap_axis_value(
        axis_values: Sequence[float] | None,
        value: float,
    ) -> float:
        if axis_values is None:
            return float(value)
        try:
            axis = np.asarray(axis_values, dtype=float).reshape(-1)
        except Exception:
            return float(value)
        if axis.size <= 0:
            return float(value)
        finite_axis = axis[np.isfinite(axis)]
        if finite_axis.size <= 0 or not np.isfinite(value):
            return float(value)
        best_idx = int(np.argmin(np.abs(finite_axis - float(value))))
        return float(finite_axis[best_idx])

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
                    wrapped_phi = float(wrap_phi_range(phi))
                    return (
                        _snap_axis_value(caked_radial_values, float(two_theta)),
                        _snap_axis_value(caked_azimuth_values, wrapped_phi),
                    )
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
    wrapped_phi = float(wrap_phi_range(phi))
    return (
        _snap_axis_value(caked_radial_values, float(two_theta)),
        _snap_axis_value(caked_azimuth_values, wrapped_phi),
    )


def should_collect_hit_tables_for_update(
    *,
    background_visible: bool,
    current_background_index: object,
    skip_preview_once: bool = False,
    manual_pick_armed: bool = False,
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

    if bool(skip_preview_once):
        return False

    manual_geometry_overlay_requested = False
    if background_visible and current_manual_pick_background_image() is not None:
        try:
            has_manual_geometry_overlay = bool(
                geometry_manual_pairs_for_index(int(current_background_index))
                or geometry_manual_pick_session_active()
            )
        except Exception:
            has_manual_geometry_overlay = False
        manual_geometry_overlay_requested = bool(
            manual_pick_armed or has_manual_geometry_overlay
        )

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
        "detector_x",
        "detector_y",
        "caked_x",
        "caked_y",
        "raw_caked_x",
        "raw_caked_y",
        "placement_error_px",
        "sigma_px",
        "refined_sim_x",
        "refined_sim_y",
        "refined_sim_native_x",
        "refined_sim_native_y",
        "refined_sim_caked_x",
        "refined_sim_caked_y",
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

    for key in ("source_table_index", "source_row_index", "source_peak_index"):
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


def normalized_background_path_for_compare(raw_path: object) -> str | None:
    """Return a normalized path string suitable for background matching."""

    try:
        candidate = Path(str(raw_path)).expanduser()
    except Exception:
        return None
    if not str(candidate):
        return None
    return os.path.normcase(os.path.normpath(str(candidate)))


def geometry_manual_pairs_export_rows(
    *,
    pairs_by_background: Mapping[object, Sequence[dict[str, object]]] | None,
    osc_files: Sequence[object],
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    pair_entry_to_jsonable: Callable[
        [dict[str, object] | None],
        dict[str, object] | None,
    ] = geometry_manual_pair_entry_to_jsonable,
) -> list[dict[str, object]]:
    """Return saved manual placements as portable per-background rows."""

    background_indices: set[int] = set()
    raw_keys = (
        pairs_by_background.keys()
        if isinstance(pairs_by_background, Mapping)
        else ()
    )
    for raw_idx in raw_keys:
        try:
            background_indices.add(int(raw_idx))
        except Exception:
            continue

    rows: list[dict[str, object]] = []
    for background_idx in sorted(background_indices):
        entries = [
            serialized
            for serialized in (
                pair_entry_to_jsonable(entry)
                for entry in pairs_for_index(int(background_idx))
            )
            if serialized is not None
        ]
        if not entries:
            continue

        if 0 <= int(background_idx) < len(osc_files):
            background_path = str(Path(str(osc_files[background_idx])).expanduser())
            background_name = Path(str(osc_files[background_idx])).name
        else:
            background_path = None
            background_name = f"background_{int(background_idx) + 1}"

        rows.append(
            {
                "background_index": int(background_idx),
                "background_path": background_path,
                "background_name": background_name,
                "entries": entries,
            }
        )
    return rows


def collect_geometry_manual_pairs_snapshot(
    *,
    osc_files: Sequence[object],
    current_background_index: object,
    manual_pair_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Return a portable snapshot of saved manual geometry placements."""

    try:
        background_index = int(current_background_index)
    except Exception:
        background_index = 0

    return {
        "background_files": [
            str(Path(str(path)).expanduser())
            for path in osc_files
        ],
        "current_background_index": int(background_index),
        "manual_pairs": list(manual_pair_rows),
    }


def apply_geometry_manual_pairs_rows(
    rows: Sequence[object] | None,
    *,
    osc_files: Sequence[object],
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    pair_entry_from_jsonable: Callable[
        [dict[str, object] | None],
        dict[str, object] | None,
    ] = geometry_manual_pair_entry_from_jsonable,
    replace_pairs_by_background: Callable[[dict[int, list[dict[str, object]]]], object],
    clear_preview_artists: Callable[..., None],
    cancel_pick_session: Callable[..., None],
    invalidate_pick_cache: Callable[[], None],
    clear_manual_undo_stack: Callable[[], None],
    clear_geometry_fit_undo_stack: Callable[[], None],
    render_current_pairs: Callable[..., None],
    update_button_label: Callable[[], None],
    refresh_status: Callable[[], None],
    replace_existing: bool = True,
) -> tuple[int, int, list[str]]:
    """Import saved manual geometry pairs onto the currently loaded backgrounds."""

    exact_path_lookup: dict[str, int] = {}
    name_lookup: defaultdict[str, list[int]] = defaultdict(list)
    for idx, raw_path in enumerate(osc_files):
        normalized_path = normalized_background_path_for_compare(raw_path)
        if normalized_path is not None:
            exact_path_lookup[normalized_path] = int(idx)
        name_lookup[Path(str(raw_path)).name].append(int(idx))

    imported_map: dict[int, list[dict[str, object]]]
    if replace_existing:
        imported_map = {}
    else:
        imported_map = {
            int(idx): list(pairs_for_index(int(idx)))
            for idx in range(len(osc_files))
            if pairs_for_index(int(idx))
        }

    warnings: list[str] = []
    matched_backgrounds: set[int] = set()
    pair_count = 0
    for raw_row in rows or []:
        if not isinstance(raw_row, dict):
            continue

        target_index = None
        normalized_path = normalized_background_path_for_compare(
            raw_row.get("background_path")
        )
        if normalized_path is not None:
            target_index = exact_path_lookup.get(normalized_path)
        if target_index is None:
            background_name = raw_row.get("background_name")
            if background_name is not None:
                matches = name_lookup.get(Path(str(background_name)).name, [])
                if len(matches) == 1:
                    target_index = int(matches[0])
        if target_index is None:
            try:
                fallback_index = int(raw_row.get("background_index"))
            except Exception:
                fallback_index = None
            if (
                fallback_index is not None
                and 0 <= fallback_index < len(osc_files)
            ):
                target_index = int(fallback_index)

        if target_index is None:
            warnings.append(
                f"Skipped placements for '{raw_row.get('background_name', 'unknown background')}'."
            )
            continue

        imported_entries = [
            restored
            for restored in (
                pair_entry_from_jsonable(entry)
                for entry in raw_row.get("entries", [])
            )
            if restored is not None
        ]
        imported_map[int(target_index)] = imported_entries
        if imported_entries:
            matched_backgrounds.add(int(target_index))
            pair_count += len(imported_entries)

    replace_pairs_by_background(
        {
            int(idx): list(entries)
            for idx, entries in imported_map.items()
            if entries
        }
    )
    clear_preview_artists(redraw=False)
    cancel_pick_session(restore_view=True, redraw=False)
    invalidate_pick_cache()
    clear_manual_undo_stack()
    clear_geometry_fit_undo_stack()
    render_current_pairs(update_status=False)
    update_button_label()
    refresh_status()
    return int(len(matched_backgrounds)), int(pair_count), warnings


def apply_geometry_manual_pairs_snapshot(
    snapshot: Mapping[str, object] | dict[str, object],
    *,
    allow_background_reload: bool = True,
    osc_files: Sequence[object],
    load_background_files: Callable[[Sequence[str], int], None] | None = None,
    apply_pairs_rows: Callable[..., tuple[int, int, list[str]]],
    schedule_update: Callable[[], None] | None = None,
) -> str:
    """Restore saved manual geometry placements from a snapshot dictionary."""

    warnings: list[str] = []

    if allow_background_reload:
        raw_background_paths = snapshot.get("background_files", [])
        background_paths: list[str] = []
        if isinstance(raw_background_paths, list):
            for raw_path in raw_background_paths:
                if raw_path is None:
                    continue
                background_paths.append(str(Path(str(raw_path)).expanduser()))

        if background_paths:
            saved_paths_norm = [
                path_norm
                for path_norm in (
                    normalized_background_path_for_compare(path)
                    for path in background_paths
                )
                if path_norm is not None
            ]
            current_paths_norm = [
                path_norm
                for path_norm in (
                    normalized_background_path_for_compare(path)
                    for path in osc_files
                )
                if path_norm is not None
            ]
            if saved_paths_norm != current_paths_norm:
                missing_paths = [
                    path for path in background_paths if not Path(path).is_file()
                ]
                if not missing_paths:
                    try:
                        if callable(load_background_files):
                            load_background_files(
                                background_paths,
                                int(snapshot.get("current_background_index", 0)),
                            )
                    except Exception as exc:
                        warnings.append(f"background reload: {exc}")
                else:
                    warnings.append(
                        "saved background files are missing; placements were mapped onto the "
                        "currently loaded backgrounds where possible"
                    )

    imported_backgrounds, imported_pairs, import_warnings = apply_pairs_rows(
        snapshot.get("manual_pairs", []),
        replace_existing=True,
    )
    warnings.extend(import_warnings)
    if callable(schedule_update):
        schedule_update()

    message = (
        f"Imported {imported_pairs} manual placement(s) across {imported_backgrounds} background(s)."
    )
    if warnings:
        message += " Warnings: " + "; ".join(warnings[:4])
        if len(warnings) > 4:
            message += f"; +{len(warnings) - 4} more"
    return message


def export_geometry_manual_pairs(
    *,
    osc_files: Sequence[object],
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    collect_snapshot: Callable[[], Mapping[str, object] | dict[str, object]],
    initial_dir: str | Path | None,
    asksaveasfilename: Callable[..., object],
    save_file: Callable[..., None],
    set_status_text: Callable[[str], None] | None = None,
    stamp_factory: Callable[[], str] | None = None,
    entrypoint: str = DEFAULT_GUI_ENTRYPOINT,
) -> str | None:
    """Run the manual-placement export dialog workflow."""

    if not any(pairs_for_index(idx) for idx in range(len(osc_files))):
        if callable(set_status_text):
            set_status_text("No saved manual placements are available to export.")
        return None

    initial_dir_value = (
        str(Path(initial_dir).expanduser())
        if initial_dir is not None
        else str(Path.cwd())
    )
    stamp = (
        str(stamp_factory()).strip()
        if callable(stamp_factory)
        else ""
    )
    initial_file = "ra_sim_geometry_placements.json"
    if stamp:
        initial_file = f"ra_sim_geometry_placements_{stamp}.json"

    file_path = asksaveasfilename(
        title="Export Geometry Placements",
        initialdir=initial_dir_value,
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        initialfile=initial_file,
    )
    if not file_path:
        if callable(set_status_text):
            set_status_text("Geometry placement export canceled.")
        return None

    try:
        save_file(
            file_path,
            collect_snapshot(),
            metadata={"entrypoint": str(entrypoint)},
        )
    except Exception as exc:
        if callable(set_status_text):
            set_status_text(f"Failed to export geometry placements: {exc}")
        return None

    if callable(set_status_text):
        set_status_text(f"Saved manual geometry placements to {file_path}")
    return str(file_path)


def import_geometry_manual_pairs(
    *,
    initial_dir: str | Path | None,
    askopenfilename: Callable[..., object],
    load_file: Callable[[str | Path], Mapping[str, object] | dict[str, object]],
    apply_snapshot: Callable[..., str],
    ensure_geometry_fit_caked_view: Callable[[], None] | None = None,
    set_status_text: Callable[[str], None] | None = None,
) -> str | None:
    """Run the manual-placement import dialog workflow."""

    initial_dir_value = (
        str(Path(initial_dir).expanduser())
        if initial_dir is not None
        else str(Path.cwd())
    )
    file_path = askopenfilename(
        title="Import Geometry Placements",
        initialdir=initial_dir_value,
        filetypes=[("RA-SIM geometry placements", "*.json"), ("All files", "*.*")],
    )
    if not file_path:
        if callable(set_status_text):
            set_status_text("Geometry placement import canceled.")
        return None

    try:
        payload = load_file(file_path)
        message = apply_snapshot(
            payload.get("state", {}),
            allow_background_reload=True,
        )
    except Exception as exc:
        if callable(set_status_text):
            set_status_text(f"Failed to import geometry placements: {exc}")
        return None

    if callable(ensure_geometry_fit_caked_view):
        try:
            ensure_geometry_fit_caked_view()
        except Exception as exc:
            message += (
                " Warning: imported placements but could not switch to 2D caked view "
                f"({exc})."
            )

    if callable(set_status_text):
        set_status_text(message)
    return message
