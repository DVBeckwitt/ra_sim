"""Workflow helpers for the geometry Q-group selector window."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ra_sim.simulation.diffraction import (
    process_peaks_parallel as diffraction_process_peaks_parallel,
)
from ra_sim.simulation.diffraction import (
    process_peaks_parallel_safe as diffraction_process_peaks_parallel_safe,
)
from ra_sim.utils.calculations import resolve_canonical_branch, source_branch_index_from_phi_deg

from . import controllers as gui_controllers
from . import manual_geometry as gui_manual_geometry
from . import geometry_overlay as gui_geometry_overlay
from . import overlays as gui_overlays
from . import views as gui_views


@dataclass
class GeometryQGroupRuntimeBindings:
    """Runtime callbacks and shared state used by the geometry Q-group selector."""

    view_state: Any
    preview_state: Any
    q_group_state: Any
    fit_config: Mapping[str, object] | None
    current_geometry_fit_var_names_factory: object
    invalidate_geometry_manual_pick_cache: Callable[[], None]
    update_geometry_preview_exclude_button_label: Callable[[], None]
    live_geometry_preview_enabled: Callable[[], bool]
    refresh_live_geometry_preview: Callable[[], None]
    set_hkl_pick_mode: Callable[[bool], None] | None = None
    live_preview_match_key: (
        Callable[[dict[str, object] | None], tuple[object, ...] | None] | None
    ) = None
    live_preview_match_hkl: (
        Callable[[dict[str, object] | None], tuple[int, int, int] | None] | None
    ) = None
    render_live_geometry_preview_state: Callable[[], object] | None = None
    clear_geometry_preview_artists: Callable[[], None] | None = None
    preview_toggle_max_distance_px: float = 20.0
    update_running: object | None = None
    has_cached_hit_tables: object | None = None
    build_live_preview_simulated_peaks_from_cache: Callable[[], list[dict[str, object]]] | None = (
        None
    )
    simulate_preview_style_peaks: Callable[..., list[dict[str, object]]] | None = None
    miller: object | None = None
    intensities: object | None = None
    image_size: object | None = None
    current_geometry_fit_params_factory: Callable[[], Mapping[str, object]] | None = None
    filter_simulated_peaks: (
        Callable[
            [Sequence[dict[str, object]] | None],
            tuple[list[dict[str, object]], int, int],
        ]
        | None
    ) = None
    collapse_simulated_peaks: Callable[..., tuple[list[dict[str, object]], int]] | None = None
    excluded_q_group_count: Callable[[], int] | None = None
    caked_view_enabled: Callable[[], bool] | None = None
    background_visible: object | None = None
    current_background_display_factory: Callable[[], object] | None = None
    axis: object | None = None
    geometry_preview_artists: list[object] | None = None
    draw_idle: Callable[[], None] | None = None
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] | None = None
    live_preview_match_is_excluded: Callable[[dict[str, object] | None], bool] | None = None
    filter_live_preview_matches: (
        Callable[[Sequence[dict[str, object]] | None], tuple[list[dict[str, object]], int]] | None
    ) = None
    build_entries_snapshot: Callable[[], Sequence[dict[str, object]] | None] | None = None
    refresh_live_geometry_preview_quiet: Callable[[], None] | None = None
    clear_last_simulation_signature: Callable[[], None] | None = None
    schedule_update: Callable[[], None] | None = None
    set_status_text: Callable[[str], None] | None = None
    file_dialog_dir: object | None = None
    asksaveasfilename: Callable[..., object] | None = None
    askopenfilename: Callable[..., object] | None = None


@dataclass(frozen=True)
class GeometryQGroupRuntimeCallbacks:
    """Bound zero-arg callbacks for the runtime Qr/Qz selector workflow."""

    update_window_status: Callable[[Sequence[dict[str, object]] | None], None]
    refresh_window: Callable[[], bool]
    on_toggle: Callable[[tuple[object, ...] | None, object], bool]
    include_all: Callable[[], bool]
    exclude_all: Callable[[], bool]
    update_listed_peaks: Callable[[], None]
    save_selection: Callable[[], bool]
    load_selection: Callable[[], bool]
    close_window: Callable[[], None]
    open_window: Callable[[], bool]
    open_preview_exclusion_window: Callable[[], bool]
    set_preview_exclude_mode: Callable[..., bool]
    clear_preview_exclusions: Callable[[], None]
    toggle_preview_exclusion_at: Callable[[float, float], bool]
    toggle_live_preview: Callable[[], bool]
    live_preview_enabled: Callable[[], bool]
    render_live_preview_state: Callable[..., bool]


@dataclass(frozen=True)
class GeometryFitSimulationRuntimeCallbacks:
    """Bound callbacks for live geometry-fit hit-table and peak simulation."""

    simulate_hit_tables: Callable[..., list[object]]
    simulate_peak_centers: Callable[..., list[dict[str, object]]]
    simulate_preview_style_peaks: Callable[..., list[dict[str, object]]]
    last_simulation_diagnostics: Callable[[], dict[str, object]]


@dataclass(frozen=True)
class GeometryQGroupRuntimeValueCallbacks:
    """Bound callbacks for live Qr/Qz selector values and peak snapshots."""

    build_live_preview_simulated_peaks_from_cache: Callable[[], list[dict[str, object]]]
    filter_simulated_peaks: Callable[
        [Sequence[dict[str, object]] | None],
        tuple[list[dict[str, object]], int, int],
    ]
    collapse_simulated_peaks: Callable[..., tuple[list[dict[str, object]], int]]
    build_entries_snapshot: Callable[[], list[dict[str, object]]]
    clone_entries: Callable[[Sequence[dict[str, object]] | None], list[dict[str, object]]]
    listed_entries: Callable[[], list[dict[str, object]]]
    listed_keys: Callable[[Sequence[dict[str, object]] | None], set[tuple[object, ...]]]
    key_from_jsonable: Callable[[object], tuple[object, ...] | None]
    export_rows: Callable[[Sequence[dict[str, object]] | None], list[dict[str, object]]]
    format_line: Callable[[dict[str, object]], str]
    current_min_matches: Callable[[], int]
    excluded_count: Callable[[Sequence[dict[str, object]] | None], int]
    build_window_status: Callable[[Sequence[dict[str, object]] | None], str]
    build_preview_exclude_button_label: Callable[
        [Sequence[dict[str, object]] | None],
        str,
    ]
    live_preview_match_key: Callable[[dict[str, object] | None], tuple[object, ...] | None]
    live_preview_match_hkl: Callable[[dict[str, object] | None], tuple[int, int, int] | None]
    live_preview_match_is_excluded: Callable[[dict[str, object] | None], bool]
    filter_live_preview_matches: Callable[
        [Sequence[dict[str, object]] | None],
        tuple[list[dict[str, object]], int],
    ]
    apply_live_preview_match_exclusions: Callable[
        [Sequence[dict[str, object]] | None, dict[str, object] | None],
        tuple[list[dict[str, object]], dict[str, object], int],
    ]
    last_live_preview_cache_metadata: Callable[[], dict[str, object]] | None = None


def _resolve_runtime_value(value_or_callable: object) -> object:
    if callable(value_or_callable):
        try:
            return value_or_callable()
        except Exception:
            return None
    return value_or_callable


def _runtime_geometry_fit_var_names(
    bindings: GeometryQGroupRuntimeBindings,
) -> list[object]:
    raw_value = _resolve_runtime_value(bindings.current_geometry_fit_var_names_factory)
    if raw_value is None:
        return []
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        return list(raw_value)
    try:
        return list(raw_value)
    except Exception:
        return []


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _copy_simulation_diag_value(value: object) -> object:
    """Return one log-friendly deep copy of simulation diagnostics state."""

    if isinstance(value, Mapping):
        return {str(key): _copy_simulation_diag_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _copy_simulation_diag_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_copy_simulation_diag_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _array_shape_list(value: object) -> list[int]:
    """Return one array-like object's shape as plain integers."""

    try:
        return [int(v) for v in np.asarray(value).shape]
    except Exception:
        return []


def _array_size(value: object) -> int | None:
    """Return one array-like object's flattened size when available."""

    if value is None:
        return None
    try:
        return int(np.asarray(value).size)
    except Exception:
        try:
            return int(len(value))  # type: ignore[arg-type]
        except Exception:
            return None


def _array_row_count(value: object) -> int | None:
    """Return one array-like object's leading-dimension count when available."""

    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        arr = None
    if arr is not None:
        if arr.ndim == 0:
            return int(arr.size)
        return int(arr.shape[0])
    try:
        return int(len(value))  # type: ignore[arg-type]
    except Exception:
        return None


def _geometry_fit_param_summary(
    params_local: Mapping[str, object],
) -> dict[str, object]:
    """Return one concise geometry-fit parameter summary for diagnostics."""

    return {
        "a": _copy_simulation_diag_value(params_local.get("a")),
        "c": _copy_simulation_diag_value(params_local.get("c")),
        "lambda": _copy_simulation_diag_value(params_local.get("lambda")),
        "theta_initial": _copy_simulation_diag_value(params_local.get("theta_initial")),
        "center": _copy_simulation_diag_value(params_local.get("center")),
        "n2": _copy_simulation_diag_value(params_local.get("n2")),
        "optics_mode": _copy_simulation_diag_value(params_local.get("optics_mode", 0)),
    }


def _geometry_fit_mosaic_array_sizes(
    mosaic: Mapping[str, object],
    *,
    wavelength_array: object,
) -> dict[str, object]:
    """Return one concise geometry-fit mosaic array-size summary."""

    return {
        "beam_x_array": _array_size(mosaic.get("beam_x_array")),
        "beam_y_array": _array_size(mosaic.get("beam_y_array")),
        "theta_array": _array_size(mosaic.get("theta_array")),
        "phi_array": _array_size(mosaic.get("phi_array")),
        "wavelength_array": _array_size(wavelength_array),
        "sample_weights": _array_size(mosaic.get("sample_weights")),
    }


def _geometry_fit_exception_diagnostics(exc: Exception) -> dict[str, object]:
    """Return one stable exception payload for simulation diagnostics."""

    exception = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    return {
        "exception_type": str(exception["type"]),
        "exception_message": str(exception["message"]),
        "exception": dict(exception),
        "exceptions": [dict(exception)],
    }


def _geometry_fit_row_count_preview(
    hit_row_counts: Sequence[int],
    *,
    limit: int = 16,
) -> list[int]:
    """Return one short row-count preview for per-table hit diagnostics."""

    return [int(count) for count in list(hit_row_counts)[: max(0, int(limit))]]


def _set_function_last_diagnostics(
    callback: Callable[..., object],
    diagnostics: Mapping[str, object] | None,
) -> None:
    """Attach one diagnostics snapshot to a simulation helper function."""

    try:
        setattr(
            callback,
            "last_diagnostics",
            _copy_simulation_diag_value(diagnostics if diagnostics is not None else {}),
        )
    except Exception:
        return


def _function_last_diagnostics(
    callback: Callable[..., object],
) -> dict[str, object]:
    """Return the last diagnostics snapshot attached to a simulation helper."""

    raw_value = getattr(callback, "last_diagnostics", {})
    copied = _copy_simulation_diag_value(raw_value)
    return copied if isinstance(copied, dict) else {}


def _row_var_value(row_var: object) -> bool:
    if row_var is None:
        return False
    try:
        return bool(row_var.get())
    except Exception:
        return bool(row_var)


def geometry_reference_hit_rows(table: object) -> list[np.ndarray]:
    """Return the finite propagated hit rows recorded for one beam sample."""

    try:
        tbl_arr = np.asarray(table, dtype=object)
    except Exception:
        return []
    if tbl_arr.ndim not in (1, 2) or tbl_arr.shape[0] == 0:
        return []

    rows: list[np.ndarray] = []
    for row in list(tbl_arr):
        try:
            row_arr = np.asarray(row, dtype=float)
        except Exception:
            continue
        if row_arr.ndim != 1 or row_arr.shape[0] < 7:
            continue
        if not (
            np.isfinite(row_arr[0])
            and np.isfinite(row_arr[1])
            and np.isfinite(row_arr[2])
            and np.isfinite(row_arr[4])
            and np.isfinite(row_arr[5])
            and np.isfinite(row_arr[6])
        ):
            continue
        rows.append(np.asarray(row_arr[:7], dtype=float))
    return rows


def _geometry_hit_row_peak_indices(
    rows: Sequence[np.ndarray] | None,
) -> list[int | None]:
    """Assign each filtered hit row to stable detector-side branch ``0`` or ``1``."""

    filtered_rows = [np.asarray(row, dtype=float) for row in (rows or ())]
    if not filtered_rows:
        return []
    assignments: list[int | None] = []
    for row in filtered_rows:
        try:
            phi_deg = float(row[3])
        except Exception:
            assignments.append(None)
            continue
        assignments.append(source_branch_index_from_phi_deg(phi_deg))
    return assignments


def _reflection_q_group_components(
    hkl_value: object,
    *,
    allow_nominal_hkl_indices: bool = False,
) -> tuple[float, int] | None:
    """Return the stable ``(m, l)`` components used for Qr/Qz grouping."""

    if not isinstance(hkl_value, (list, tuple, np.ndarray)) or len(hkl_value) < 3:
        return None
    try:
        h_raw = float(hkl_value[0])
        k_raw = float(hkl_value[1])
        l_raw = float(hkl_value[2])
    except Exception:
        return None

    if allow_nominal_hkl_indices:
        hkl_group = gui_geometry_overlay.normalize_hkl_key(hkl_value)
        if hkl_group is None:
            return None
        h_group = float(hkl_group[0])
        k_group = float(hkl_group[1])
        l_int = int(hkl_group[2])
    else:
        h_group = float(h_raw)
        k_group = float(k_raw)
        l_int = gui_manual_geometry.integer_gz_index(l_raw)
        if l_int is None:
            return None

    m_val = h_group * h_group + h_group * k_group + k_group * k_group
    return float(m_val), int(l_int)


def reflection_q_group_metadata(
    hkl_value: object,
    *,
    source_label: object = "primary",
    a_value: object = np.nan,
    c_value: object = np.nan,
    qr_value: object = np.nan,
    allow_nominal_hkl_indices: bool = False,
) -> tuple[tuple[object, ...] | None, float, float]:
    """Return stable Qr/Qz grouping metadata for one simulated reflection."""

    components = _reflection_q_group_components(
        hkl_value,
        allow_nominal_hkl_indices=allow_nominal_hkl_indices,
    )
    if components is None:
        return None, float("nan"), float("nan")
    m_val, l_int = components

    try:
        qr_val = float(qr_value)
    except Exception:
        qr_val = float("nan")
    try:
        a_used = float(a_value)
    except Exception:
        a_used = float("nan")
    try:
        c_used = float(c_value)
    except Exception:
        c_used = float("nan")

    if not np.isfinite(qr_val):
        if np.isfinite(a_used) and a_used > 0.0 and m_val >= 0.0:
            qr_val = (2.0 * np.pi / a_used) * np.sqrt((4.0 / 3.0) * m_val)
        else:
            qr_val = float("nan")
    qz_val = (
        (2.0 * np.pi / c_used) * float(l_int)
        if np.isfinite(c_used) and c_used > 0.0
        else float("nan")
    )
    key = (
        "q_group",
        gui_controllers.normalize_bragg_qr_source_label(
            str(source_label) if source_label is not None else "primary"
        ),
        gui_manual_geometry.q_group_key_component(m_val),
        int(l_int),
    )
    return key, float(qr_val), float(qz_val)


def geometry_q_group_key_from_entry(
    entry: Mapping[str, object] | None,
) -> tuple[object, ...] | None:
    """Return the stable Qr/Qz group key for one simulated peak record."""

    if not isinstance(entry, Mapping):
        return None
    allow_nominal_hkl_indices = bool(entry.get("q_group_nominal_hkl", False))
    hkl_value = entry.get("hkl_raw", entry.get("hkl"))
    key, _, _ = reflection_q_group_metadata(
        hkl_value,
        source_label=entry.get("source_label", "primary"),
        a_value=entry.get("av", np.nan),
        c_value=entry.get("cv", np.nan),
        qr_value=entry.get("qr", np.nan),
        allow_nominal_hkl_indices=allow_nominal_hkl_indices,
    )
    if key is None:
        key, _, _ = reflection_q_group_metadata(
            entry.get("hkl"),
            source_label=entry.get("source_label", "primary"),
            a_value=entry.get("av", np.nan),
            c_value=entry.get("cv", np.nan),
            qr_value=entry.get("qr", np.nan),
            allow_nominal_hkl_indices=True,
        )
    return key


def build_geometry_q_group_entries(
    max_positions_local: Sequence[object] | None,
    *,
    peak_table_lattice: Sequence[Sequence[object]] | None = None,
    peak_records: Sequence[Mapping[str, object]] | None = None,
    primary_a: object = np.nan,
    primary_c: object = np.nan,
    allow_nominal_hkl_indices: bool = False,
) -> list[dict[str, object]]:
    """Aggregate simulated hit tables or cached peak records into Qr/Qz rows."""

    try:
        default_primary_a = float(primary_a)
    except Exception:
        default_primary_a = float("nan")
    try:
        default_primary_c = float(primary_c)
    except Exception:
        default_primary_c = float("nan")

    entries_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    cached_peak_records = list(peak_records or [])
    if cached_peak_records:
        for raw_record in cached_peak_records:
            if not isinstance(raw_record, Mapping):
                continue
            source_label = str(raw_record.get("source_label", "primary"))
            av_used = _coerce_float(raw_record.get("av", primary_a), default_primary_a)
            cv_used = _coerce_float(raw_record.get("cv", primary_c), default_primary_c)
            intensity = _coerce_float(
                raw_record.get("intensity", raw_record.get("weight", 0.0)),
                0.0,
            )
            hkl_raw = raw_record.get("hkl_raw", raw_record.get("hkl"))
            hkl_key = gui_geometry_overlay.normalize_hkl_key(hkl_raw)
            if hkl_key is None:
                continue
            use_nominal_hkl_indices = bool(
                raw_record.get("q_group_nominal_hkl", allow_nominal_hkl_indices)
            )
            raw_group_key = raw_record.get("q_group_key")
            if isinstance(raw_group_key, list):
                group_key = tuple(raw_group_key)
            elif isinstance(raw_group_key, tuple):
                group_key = raw_group_key
            else:
                group_key = None
            resolved_group_key, qr_val, qz_val = reflection_q_group_metadata(
                hkl_raw,
                source_label=source_label,
                a_value=av_used,
                c_value=cv_used,
                qr_value=raw_record.get("qr", np.nan),
                allow_nominal_hkl_indices=use_nominal_hkl_indices,
            )
            if group_key is None:
                group_key = resolved_group_key
            if group_key is None:
                continue

            entry = entries_by_key.get(group_key)
            if entry is None:
                entry = {
                    "key": group_key,
                    "source_label": str(source_label),
                    "qr": float(qr_val),
                    "qz": float(qz_val),
                    "gz_index": int(group_key[3]),
                    "total_intensity": 0.0,
                    "peak_count": 0,
                    "hkl_preview": [],
                }
                entries_by_key[group_key] = entry
            elif not np.isfinite(_coerce_float(entry.get("qr", np.nan), np.nan)) and np.isfinite(
                qr_val
            ):
                entry["qr"] = float(qr_val)
            if not np.isfinite(_coerce_float(entry.get("qz", np.nan), np.nan)) and np.isfinite(
                qz_val
            ):
                entry["qz"] = float(qz_val)

            entry["total_intensity"] = float(entry["total_intensity"]) + float(abs(intensity))
            entry["peak_count"] = int(entry["peak_count"]) + 1
            if len(entry["hkl_preview"]) < 4 and hkl_key not in entry["hkl_preview"]:
                entry["hkl_preview"].append(hkl_key)
    elif max_positions_local is not None:
        peak_table_lattice_local = list(peak_table_lattice or [])
        if not peak_table_lattice_local or len(peak_table_lattice_local) != len(
            max_positions_local
        ):
            peak_table_lattice_local = [
                (default_primary_a, default_primary_c, "primary") for _ in max_positions_local
            ]

        for table_idx, tbl in enumerate(max_positions_local):
            rows = geometry_reference_hit_rows(tbl)
            if not rows:
                continue

            av_used = default_primary_a
            cv_used = default_primary_c
            source_label = "primary"
            if table_idx < len(peak_table_lattice_local):
                try:
                    av_used = float(peak_table_lattice_local[table_idx][0])
                    cv_used = float(peak_table_lattice_local[table_idx][1])
                    source_label = str(peak_table_lattice_local[table_idx][2])
                except Exception:
                    av_used = default_primary_a
                    cv_used = default_primary_c
                    source_label = "primary"

            for row in rows:
                intensity, _xpix, _ypix, _phi, h_val, k_val, l_val = row[:7]
                if not np.isfinite(intensity):
                    continue
                hkl_key = tuple(int(np.rint(v)) for v in (h_val, k_val, l_val))
                group_key, qr_val, qz_val = reflection_q_group_metadata(
                    (h_val, k_val, l_val),
                    source_label=source_label,
                    a_value=av_used,
                    c_value=cv_used,
                    allow_nominal_hkl_indices=allow_nominal_hkl_indices,
                )
                if group_key is None:
                    continue

                entry = entries_by_key.get(group_key)
                if entry is None:
                    entry = {
                        "key": group_key,
                        "source_label": str(source_label),
                        "qr": float(qr_val),
                        "qz": float(qz_val),
                        "gz_index": int(group_key[3]),
                        "total_intensity": 0.0,
                        "peak_count": 0,
                        "hkl_preview": [],
                    }
                    entries_by_key[group_key] = entry

                entry["total_intensity"] = float(entry["total_intensity"]) + float(abs(intensity))
                entry["peak_count"] = int(entry["peak_count"]) + 1
                if len(entry["hkl_preview"]) < 4 and hkl_key not in entry["hkl_preview"]:
                    entry["hkl_preview"].append(hkl_key)
    else:
        return []

    def _sort_value(value: object) -> float:
        numeric = _coerce_float(value, float("nan"))
        return numeric if np.isfinite(numeric) else float("inf")

    entries = list(entries_by_key.values())
    entries.sort(
        key=lambda entry: (
            str(entry.get("source_label", "")),
            _sort_value(entry.get("qr", np.nan)),
            _sort_value(entry.get("qz", np.nan)),
        )
    )
    return entries


def build_geometry_fit_simulated_peaks(
    hit_tables: Sequence[object] | None,
    *,
    image_shape: tuple[int, int],
    native_sim_to_display_coords: Callable[
        [float, float, tuple[int, int]],
        tuple[float, float],
    ],
    peak_table_lattice: Sequence[Sequence[object]] | None = None,
    source_reflection_indices: Sequence[int] | None = None,
    primary_a: object = np.nan,
    primary_c: object = np.nan,
    default_source_label: str | None = "primary",
    round_pixel_centers: bool = False,
    allow_nominal_hkl_indices: bool = False,
) -> list[dict[str, object]]:
    """Build simulated-peak records from detector hit tables for geometry workflows."""

    if not hit_tables:
        return []

    try:
        default_primary_a = float(primary_a)
    except Exception:
        default_primary_a = float("nan")
    try:
        default_primary_c = float(primary_c)
    except Exception:
        default_primary_c = float("nan")

    simulated_peaks: list[dict[str, object]] = []
    peak_table_lattice_local = list(peak_table_lattice or [])
    source_reflection_indices_local = (
        list(source_reflection_indices)
        if isinstance(source_reflection_indices, Sequence)
        and not isinstance(source_reflection_indices, (str, bytes))
        else []
    )
    for table_idx, tbl in enumerate(hit_tables):
        rows = geometry_reference_hit_rows(tbl)
        if not rows:
            continue
        row_peak_indices = _geometry_hit_row_peak_indices(rows)

        source_label = (
            str(default_source_label) if default_source_label is not None else f"table_{table_idx}"
        )
        av_used = default_primary_a
        cv_used = default_primary_c
        if table_idx < len(peak_table_lattice_local):
            lattice_entry = peak_table_lattice_local[table_idx]
            if isinstance(lattice_entry, Sequence) and len(lattice_entry) >= 3:
                try:
                    av_used = float(lattice_entry[0])
                    cv_used = float(lattice_entry[1])
                    source_label = str(lattice_entry[2])
                except Exception:
                    av_used = default_primary_a
                    cv_used = default_primary_c
                    source_label = (
                        str(default_source_label)
                        if default_source_label is not None
                        else f"table_{table_idx}"
                    )

        for row_idx, row in enumerate(rows):
            intensity, xpix, ypix, phi_deg, h_val, k_val, l_val = row[:7]
            if not (np.isfinite(intensity) and np.isfinite(xpix) and np.isfinite(ypix)):
                continue

            native_col = float(xpix)
            native_row = float(ypix)
            if round_pixel_centers:
                native_col = float(int(round(native_col)))
                native_row = float(int(round(native_row)))

            display_col, display_row = native_sim_to_display_coords(
                native_col,
                native_row,
                image_shape,
            )
            hkl = tuple(int(np.rint(val)) for val in (h_val, k_val, l_val))
            hkl_raw = (float(h_val), float(k_val), float(l_val))
            q_group_key, qr_val, qz_val = reflection_q_group_metadata(
                hkl_raw,
                source_label=source_label,
                a_value=av_used,
                c_value=cv_used,
                allow_nominal_hkl_indices=allow_nominal_hkl_indices,
            )
            if q_group_key is None:
                continue

            try:
                trusted_reflection_index = (
                    int(source_reflection_indices_local[table_idx])
                    if table_idx < len(source_reflection_indices_local)
                    else None
                )
            except Exception:
                trusted_reflection_index = None
            peak_record = {
                "hkl": hkl,
                "sim_col": float(display_col),
                "sim_row": float(display_row),
                "weight": max(0.0, float(abs(intensity))),
                "source_label": str(source_label),
                "source_table_index": int(table_idx),
                "source_row_index": int(row_idx),
                "hkl_raw": hkl_raw,
                "phi": float(phi_deg),
                "av": float(av_used),
                "cv": float(cv_used),
                "qr": float(qr_val),
                "qz": float(qz_val),
                "q_group_key": q_group_key,
            }
            branch_index = (
                int(row_peak_indices[row_idx])
                if row_idx < len(row_peak_indices) and row_peak_indices[row_idx] in {0, 1}
                else None
            )
            if branch_index in {0, 1}:
                peak_record["source_branch_index"] = int(branch_index)
            peak_record = gui_manual_geometry.geometry_manual_canonicalize_live_source_entry(
                peak_record,
                normalize_hkl_key=(
                    gui_geometry_overlay.normalize_hkl_key
                    if callable(getattr(gui_geometry_overlay, "normalize_hkl_key", None))
                    else None
                )
                or (lambda value: hkl if value is not None else None),
                allow_legacy_peak_fallback=False,
                preserve_existing_trusted_identity=False,
                trusted_reflection_index=trusted_reflection_index,
            )
            if peak_record is None:
                continue
            if allow_nominal_hkl_indices:
                peak_record["q_group_nominal_hkl"] = True
            simulated_peaks.append(peak_record)

    return simulated_peaks


def audited_full_order_source_reflection_indices(
    hit_tables: Sequence[object] | None,
    *,
    owner: str,
    start_index: int = 0,
) -> list[int]:
    """Return full-order reflection ids for callers that own exact table ordering."""

    if not str(owner or "").strip():
        raise ValueError("owner is required for audited full-order reflection ids")
    table_count = int(len(hit_tables or ()))
    if table_count <= 0:
        return []
    base_index = max(0, int(start_index))
    return list(range(base_index, base_index + table_count))


def audited_full_order_source_reflection_index_groups(
    hit_table_groups: Sequence[Sequence[object] | None] | None,
    *,
    owner: str,
) -> list[list[int]]:
    """Return sequential full-order reflection ids for one or more table groups."""

    groups = list(hit_table_groups or ())
    if not groups:
        return []

    next_start_index = 0
    reflection_index_groups: list[list[int]] = []
    for group_index, hit_tables in enumerate(groups):
        indices = audited_full_order_source_reflection_indices(
            hit_tables,
            owner=f"{owner}.group[{group_index}]",
            start_index=next_start_index,
        )
        reflection_index_groups.append(indices)
        next_start_index += len(indices)
    return reflection_index_groups


def build_geometry_fit_full_order_source_rows(
    hit_tables: Sequence[object] | None,
    *,
    image_shape: tuple[int, int],
    native_sim_to_display_coords: Callable[
        [float, float, tuple[int, int]],
        tuple[float, float],
    ],
    primary_a: object = np.nan,
    primary_c: object = np.nan,
    default_source_label: str = "primary",
    round_pixel_centers: bool = False,
    allow_nominal_hkl_indices: bool = False,
    owner: str,
    start_index: int = 0,
) -> tuple[list[dict[str, object]], list[tuple[float, float, str]], list[int]]:
    """Build full-order source rows plus the audited reflection-index mapping."""

    source_reflection_indices = audited_full_order_source_reflection_indices(
        hit_tables,
        owner=owner,
        start_index=start_index,
    )
    peak_table_lattice = [
        (float(primary_a), float(primary_c), str(default_source_label)) for _ in (hit_tables or ())
    ]
    source_rows = build_geometry_fit_simulated_peaks(
        hit_tables,
        image_shape=image_shape,
        native_sim_to_display_coords=native_sim_to_display_coords,
        peak_table_lattice=peak_table_lattice,
        source_reflection_indices=source_reflection_indices,
        primary_a=primary_a,
        primary_c=primary_c,
        default_source_label=default_source_label,
        round_pixel_centers=round_pixel_centers,
        allow_nominal_hkl_indices=allow_nominal_hkl_indices,
    )
    return list(source_rows), peak_table_lattice, source_reflection_indices


def _source_row_hkl_lookup_from_rows(
    source_rows: Sequence[object] | None,
) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Build the exact row/HKL provenance map used to re-trust cached peak rows."""

    lookup: dict[tuple[int, int], tuple[int, int, int]] = {}
    for raw_entry in source_rows or ():
        if not isinstance(raw_entry, Mapping):
            continue
        try:
            table_idx = int(raw_entry.get("source_table_index"))
            row_idx = int(raw_entry.get("source_row_index"))
        except Exception:
            continue
        if table_idx < 0 or row_idx < 0:
            continue
        hkl_value = raw_entry.get("hkl")
        if not isinstance(hkl_value, (list, tuple, np.ndarray)) or len(hkl_value) < 3:
            continue
        try:
            lookup[(int(table_idx), int(row_idx))] = (
                int(hkl_value[0]),
                int(hkl_value[1]),
                int(hkl_value[2]),
            )
        except Exception:
            continue
    return lookup


def _signature_summary(signature: object) -> str | None:
    if signature is None:
        return None
    text = repr(signature)
    return text if len(text) <= 240 else text[:237] + "..."


def _matching_source_row_snapshot_payload(
    simulation_runtime_state,
    *,
    signature: object = None,
    signature_summary: object = None,
    background_index: int | None = None,
) -> dict[str, object]:
    """Return the unique current snapshot payload that can prove cached peak provenance."""

    raw_snapshots = getattr(simulation_runtime_state, "source_row_snapshots", None)
    if not isinstance(raw_snapshots, Mapping):
        return {}

    def _rows_from_snapshot(snapshot: Mapping[str, object]) -> list[dict[str, object]]:
        return [
            dict(entry) for entry in (snapshot.get("rows", ()) or ()) if isinstance(entry, Mapping)
        ]

    def _match_kind(snapshot: Mapping[str, object]) -> str | None:
        if signature is not None and snapshot.get("simulation_signature") == signature:
            return "exact_signature"
        if (
            signature_summary not in (None, "")
            and snapshot.get("simulation_signature_summary") == signature_summary
        ):
            return "signature_summary"
        if signature is None and signature_summary in (None, ""):
            return "unspecified"
        return None

    if background_index is not None:
        snapshot = raw_snapshots.get(int(background_index))
        if not isinstance(snapshot, Mapping):
            return {}
        match_kind = _match_kind(snapshot)
        if match_kind is None:
            return {}
        rows = _rows_from_snapshot(snapshot)
        if not rows:
            return {}
        return {
            "background_index": int(background_index),
            "rows": rows,
            "match_kind": str(match_kind),
            "source_reflection_index_count": snapshot.get("source_reflection_index_count"),
        }

    if signature is None and signature_summary in (None, ""):
        return {}

    matches: list[dict[str, object]] = []
    for raw_key, raw_snapshot in raw_snapshots.items():
        if not isinstance(raw_snapshot, Mapping):
            continue
        match_kind = _match_kind(raw_snapshot)
        if match_kind is None:
            continue
        rows = _rows_from_snapshot(raw_snapshot)
        if not rows:
            continue
        try:
            snapshot_background_index = int(raw_snapshot.get("background_index", raw_key))
        except Exception:
            snapshot_background_index = -1
        matches.append(
            {
                "background_index": int(snapshot_background_index),
                "rows": rows,
                "match_kind": str(match_kind),
                "source_reflection_index_count": raw_snapshot.get("source_reflection_index_count"),
            }
        )
    if len(matches) != 1:
        return {}
    return dict(matches[0])


def _resolve_live_peak_record_fallback_provenance(
    simulation_runtime_state,
    *,
    signature: object = None,
    signature_summary: object = None,
    background_index: int | None = None,
    source_reflection_indices_local: Sequence[object] | None = None,
) -> dict[str, object]:
    """Resolve whether cached peak records can safely regain trusted full-reflection ids."""

    snapshot_payload = _matching_source_row_snapshot_payload(
        simulation_runtime_state,
        signature=signature,
        signature_summary=signature_summary,
        background_index=background_index,
    )
    snapshot_rows = list(snapshot_payload.get("rows", ()) or ())
    reflection_index_map = (
        list(source_reflection_indices_local or ())
        if isinstance(source_reflection_indices_local, Sequence)
        and not isinstance(source_reflection_indices_local, (str, bytes))
        else []
    )
    source_row_hkl_lookup = _source_row_hkl_lookup_from_rows(snapshot_rows)
    table_count = None
    try:
        raw_count = snapshot_payload.get("source_reflection_index_count")
        table_count = int(raw_count) if raw_count is not None else None
    except Exception:
        table_count = None
    if table_count is None and source_row_hkl_lookup:
        try:
            table_count = 1 + max(int(key[0]) for key in source_row_hkl_lookup)
        except Exception:
            table_count = None
    provenance_ready = bool(snapshot_rows and reflection_index_map and source_row_hkl_lookup)
    match_kind = str(snapshot_payload.get("match_kind", "") or "")
    return {
        "active_signature_matches": bool(provenance_ready and match_kind == "exact_signature"),
        "active_revision_matches": bool(provenance_ready and match_kind == "signature_summary"),
        "expected_table_count": table_count,
        "source_row_hkl_lookup": source_row_hkl_lookup,
        "source_snapshot_row_count": int(len(snapshot_rows)),
        "source_snapshot_background_index": snapshot_payload.get("background_index"),
    }


def geometry_fit_peak_center_from_max_position(
    entry: Sequence[float] | None,
) -> tuple[float, float] | None:
    """Return the strongest finite peak center stored in one max-position row."""

    if entry is None or len(entry) < 6:
        return None
    i0, x0, y0, i1, x1, y1 = entry
    primary_valid = np.isfinite(x0) and np.isfinite(y0)
    secondary_valid = np.isfinite(x1) and np.isfinite(y1)

    if primary_valid and (not secondary_valid or not np.isfinite(i1) or float(i0) >= float(i1)):
        return float(x0), float(y0)
    if secondary_valid:
        return float(x1), float(y1)
    if primary_valid:
        return float(x0), float(y0)
    return None


def aggregate_geometry_fit_peak_centers_from_max_positions(
    max_positions: Sequence[Sequence[float]],
    miller_array: np.ndarray,
    intensity_array: np.ndarray,
) -> list[dict[str, object]]:
    """Aggregate one simulated detector center per integer HKL."""

    miller_arr = np.asarray(miller_array, dtype=float)
    intensity_arr = np.asarray(intensity_array, dtype=float).reshape(-1)
    if miller_arr.ndim != 2 or miller_arr.shape[1] < 3:
        return []

    row_count = min(len(max_positions), miller_arr.shape[0], intensity_arr.shape[0])
    if row_count <= 0:
        return []

    centers_by_hkl: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
    weights_by_hkl: dict[tuple[int, int, int], float] = {}
    for idx in range(row_count):
        h_val, k_val, l_val = miller_arr[idx, :3]
        key = (int(round(h_val)), int(round(k_val)), int(round(l_val)))
        center = geometry_fit_peak_center_from_max_position(max_positions[idx])
        if center is None:
            continue
        centers_by_hkl.setdefault(key, []).append(center)
        weights_by_hkl[key] = weights_by_hkl.get(key, 0.0) + float(abs(intensity_arr[idx]))

    simulated_peaks: list[dict[str, object]] = []
    for key, center_list in centers_by_hkl.items():
        arr = np.asarray(center_list, dtype=float)
        simulated_peaks.append(
            {
                "hkl": key,
                "label": f"{key[0]},{key[1]},{key[2]}",
                "sim_col": float(arr[:, 0].mean()),
                "sim_row": float(arr[:, 1].mean()),
                "weight": float(weights_by_hkl.get(key, 0.0)),
            }
        )
    return simulated_peaks


def simulate_geometry_fit_hit_tables(
    miller_array: np.ndarray,
    intensity_array: np.ndarray,
    image_size: int,
    param_set: Mapping[str, object] | dict[str, object],
    *,
    build_geometry_fit_central_mosaic_params: Callable[[Mapping[str, object]], Mapping[str, object]]
    | None = None,
    process_peaks_parallel: Callable[..., object],
    default_solve_q_steps: int,
    default_solve_q_rel_tol: float,
    default_solve_q_mode: int,
) -> list[object]:
    """Simulate once and return raw hit tables for geometry-fit helpers."""

    params_local = dict(param_set)
    diagnostics: dict[str, object] = {
        "stage": "simulate_hit_tables",
        "miller_shape": _array_shape_list(miller_array),
        "miller_count": _array_row_count(miller_array),
        "intensity_shape": _array_shape_list(intensity_array),
        "intensity_count": _array_row_count(intensity_array),
        "image_size": int(image_size),
    }

    mosaic = dict(params_local.get("mosaic_params", {}))
    if not mosaic and callable(build_geometry_fit_central_mosaic_params):
        try:
            built_mosaic = build_geometry_fit_central_mosaic_params(params_local)
        except Exception:
            built_mosaic = None
        if isinstance(built_mosaic, Mapping):
            mosaic = dict(built_mosaic)
    wavelength_array = mosaic.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic.get("wavelength_i_array")
    if wavelength_array is None:
        wavelength_array = np.full(
            int(np.size(mosaic.get("beam_x_array", []))),
            float(params_local.get("lambda", 1.0)),
            dtype=np.float64,
        )

    param_summary = _geometry_fit_param_summary(params_local)
    mosaic_array_sizes = _geometry_fit_mosaic_array_sizes(
        mosaic,
        wavelength_array=wavelength_array,
    )
    diagnostics.update(
        {
            "status": "ready",
            "param_summary": param_summary,
            "parameter_summary": _copy_simulation_diag_value(param_summary),
            "mosaic_array_sizes": mosaic_array_sizes,
        }
    )

    try:
        sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
        _, hit_tables, *_ = process_peaks_parallel(
            miller_array,
            intensity_array,
            image_size,
            float(params_local["a"]),
            float(params_local["c"]),
            wavelength_array,
            sim_buffer,
            float(params_local["corto_detector"]),
            float(params_local["gamma"]),
            float(params_local["Gamma"]),
            float(params_local["chi"]),
            float(params_local.get("psi", 0.0)),
            float(params_local.get("psi_z", 0.0)),
            float(params_local["zs"]),
            float(params_local["zb"]),
            params_local["n2"],
            np.asarray(mosaic["beam_x_array"], dtype=np.float64),
            np.asarray(mosaic["beam_y_array"], dtype=np.float64),
            np.asarray(mosaic["theta_array"], dtype=np.float64),
            np.asarray(mosaic["phi_array"], dtype=np.float64),
            float(mosaic["sigma_mosaic_deg"]),
            float(mosaic["gamma_mosaic_deg"]),
            float(mosaic["eta"]),
            np.asarray(wavelength_array, dtype=np.float64),
            float(params_local["debye_x"]),
            float(params_local["debye_y"]),
            [float(params_local["center"][0]), float(params_local["center"][1])],
            float(params_local["theta_initial"]),
            float(params_local.get("cor_angle", 0.0)),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            save_flag=0,
            optics_mode=int(params_local.get("optics_mode", 0)),
            solve_q_steps=int(mosaic.get("solve_q_steps", default_solve_q_steps)),
            solve_q_rel_tol=float(mosaic.get("solve_q_rel_tol", default_solve_q_rel_tol)),
            solve_q_mode=int(mosaic.get("solve_q_mode", default_solve_q_mode)),
            sample_weights=mosaic.get("sample_weights"),
        )
    except Exception as exc:
        diagnostics.update(
            {
                "status": "process_peaks_parallel_exception",
                **_geometry_fit_exception_diagnostics(exc),
            }
        )
        _set_function_last_diagnostics(
            simulate_geometry_fit_hit_tables,
            diagnostics,
        )
        raise

    hit_table_list = list(hit_tables)
    hit_row_counts = [int(len(geometry_reference_hit_rows(table))) for table in hit_table_list]
    row_count_preview = _geometry_fit_row_count_preview(hit_row_counts)
    diagnostics.update(
        {
            "status": ("success" if int(sum(hit_row_counts)) > 0 else "empty_hit_tables"),
            "hit_table_count": int(len(hit_table_list)),
            "hit_row_counts": hit_row_counts,
            "nonempty_hit_table_count": int(sum(1 for count in hit_row_counts if count > 0)),
            "finite_hit_row_total": int(sum(hit_row_counts)),
            "row_count_preview_per_table": row_count_preview,
            "row_count_preview_truncated": bool(len(hit_row_counts) > len(row_count_preview)),
            "projected_peak_count": int(sum(hit_row_counts)),
        }
    )
    _set_function_last_diagnostics(
        simulate_geometry_fit_hit_tables,
        diagnostics,
    )
    return hit_table_list


def simulate_geometry_fit_peak_centers(
    miller_array: np.ndarray,
    intensity_array: np.ndarray,
    image_size: int,
    param_set: Mapping[str, object] | dict[str, object],
    *,
    build_geometry_fit_central_mosaic_params: Callable[[Mapping[str, object]], Mapping[str, object]]
    | None = None,
    process_peaks_parallel: Callable[..., object],
    hit_tables_to_max_positions: Callable[[Sequence[object]], Sequence[Sequence[float]]],
    default_solve_q_steps: int,
    default_solve_q_rel_tol: float,
    default_solve_q_mode: int,
) -> list[dict[str, object]]:
    """Simulate once and return one aggregated detector center per integer HKL."""

    try:
        hit_tables = simulate_geometry_fit_hit_tables(
            miller_array,
            intensity_array,
            image_size,
            param_set,
            build_geometry_fit_central_mosaic_params=(build_geometry_fit_central_mosaic_params),
            process_peaks_parallel=process_peaks_parallel,
            default_solve_q_steps=default_solve_q_steps,
            default_solve_q_rel_tol=default_solve_q_rel_tol,
            default_solve_q_mode=default_solve_q_mode,
        )
        max_positions = hit_tables_to_max_positions(hit_tables)
        peak_centers = aggregate_geometry_fit_peak_centers_from_max_positions(
            max_positions,
            miller_array,
            intensity_array,
        )
        diagnostics = _function_last_diagnostics(simulate_geometry_fit_hit_tables)
        diagnostics.update(
            {
                "stage": "simulate_peak_centers",
                "status": ("success" if peak_centers else "empty_peak_centers"),
                "max_position_count": int(len(max_positions)),
                "peak_center_count": int(len(peak_centers)),
                "projected_peak_count": int(len(peak_centers)),
            }
        )
        _set_function_last_diagnostics(
            simulate_geometry_fit_peak_centers,
            diagnostics,
        )
        return peak_centers
    except Exception as exc:
        diagnostics = _function_last_diagnostics(simulate_geometry_fit_hit_tables)
        diagnostics.update(
            {
                "stage": "simulate_peak_centers",
                "status": "exception",
                **_geometry_fit_exception_diagnostics(exc),
            }
        )
        _set_function_last_diagnostics(
            simulate_geometry_fit_peak_centers,
            diagnostics,
        )
        raise


def simulate_geometry_fit_preview_style_peaks(
    miller_array: np.ndarray,
    intensity_array: np.ndarray,
    image_size: int,
    param_set: Mapping[str, object] | dict[str, object],
    *,
    build_geometry_fit_central_mosaic_params: Callable[[Mapping[str, object]], Mapping[str, object]]
    | None = None,
    process_peaks_parallel: Callable[..., object],
    native_sim_to_display_coords: Callable[
        [float, float, tuple[int, int]],
        tuple[float, float],
    ],
    peak_table_lattice: Sequence[Sequence[object]] | None = None,
    primary_a: object = np.nan,
    primary_c: object = np.nan,
    default_source_label: str | None = None,
    round_pixel_centers: bool = False,
    default_solve_q_steps: int,
    default_solve_q_rel_tol: float,
    default_solve_q_mode: int,
    allow_nominal_hkl_indices: bool = False,
) -> list[dict[str, object]]:
    """Simulate once and return preview-style per-branch peak records."""

    try:
        hit_tables = simulate_geometry_fit_hit_tables(
            miller_array,
            intensity_array,
            image_size,
            param_set,
            build_geometry_fit_central_mosaic_params=(build_geometry_fit_central_mosaic_params),
            process_peaks_parallel=process_peaks_parallel,
            default_solve_q_steps=default_solve_q_steps,
            default_solve_q_rel_tol=default_solve_q_rel_tol,
            default_solve_q_mode=default_solve_q_mode,
        )
        peak_records, _peak_table_lattice, _source_reflection_indices = (
            build_geometry_fit_full_order_source_rows(
                hit_tables,
                image_shape=(int(image_size), int(image_size)),
                native_sim_to_display_coords=native_sim_to_display_coords,
                primary_a=primary_a,
                primary_c=primary_c,
                default_source_label=default_source_label,
                round_pixel_centers=round_pixel_centers,
                allow_nominal_hkl_indices=allow_nominal_hkl_indices,
                owner="simulate_geometry_fit_preview_style_peaks",
            )
        )
        diagnostics = _function_last_diagnostics(simulate_geometry_fit_hit_tables)
        diagnostics.update(
            {
                "stage": "simulate_preview_style_peaks",
                "status": "success" if peak_records else "empty_peak_records",
                "peak_count": int(len(peak_records)),
                "projected_peak_count": int(len(peak_records)),
            }
        )
        _set_function_last_diagnostics(
            simulate_geometry_fit_preview_style_peaks,
            diagnostics,
        )
        return peak_records
    except Exception as exc:
        diagnostics = _function_last_diagnostics(simulate_geometry_fit_hit_tables)
        diagnostics.update(
            {
                "stage": "simulate_preview_style_peaks",
                "status": "exception",
                **_geometry_fit_exception_diagnostics(exc),
            }
        )
        _set_function_last_diagnostics(
            simulate_geometry_fit_preview_style_peaks,
            diagnostics,
        )
        raise


def make_runtime_geometry_fit_simulation_callbacks(
    *,
    build_geometry_fit_central_mosaic_params: (
        Callable[[Mapping[str, object]], Mapping[str, object]] | None
    ) = None,
    process_peaks_parallel: Callable[..., object],
    hit_tables_to_max_positions: Callable[[Sequence[object]], Sequence[Sequence[float]]],
    native_sim_to_display_coords: Callable[
        [float, float, tuple[int, int]],
        tuple[float, float],
    ],
    peak_table_lattice_factory: object | None = None,
    primary_a_factory: object = np.nan,
    primary_c_factory: object = np.nan,
    default_source_label: str | None = None,
    round_pixel_centers: bool = False,
    default_solve_q_steps: int,
    default_solve_q_rel_tol: float,
    default_solve_q_mode: int,
    prefer_safe_python_runner: bool = False,
) -> GeometryFitSimulationRuntimeCallbacks:
    """Return live geometry-fit simulation callbacks from runtime value sources."""

    last_simulation_diagnostics_state: dict[str, object] = {}
    process_peaks_parallel_runner = process_peaks_parallel

    if prefer_safe_python_runner:
        use_diffraction_safe_runner = process_peaks_parallel is diffraction_process_peaks_parallel

        def _keyword_mismatch(exc: TypeError) -> bool:
            message = str(exc)
            return "unexpected keyword" in message or "some keyword arguments unexpected" in message

        def _process_peaks_parallel_runner(*args, **kwargs):
            call_kwargs = dict(kwargs)
            call_kwargs["prefer_python_runner"] = True
            try:
                return process_peaks_parallel(*args, **call_kwargs)
            except TypeError as exc:
                if not _keyword_mismatch(exc):
                    raise
                call_kwargs.pop("prefer_python_runner", None)
                try:
                    return process_peaks_parallel(*args, **call_kwargs)
                except TypeError as retry_exc:
                    if use_diffraction_safe_runner and _keyword_mismatch(retry_exc):
                        safe_kwargs = dict(call_kwargs)
                        safe_kwargs["prefer_python_runner"] = True
                        return diffraction_process_peaks_parallel_safe(
                            *args,
                            **safe_kwargs,
                        )
                    raise

        process_peaks_parallel_runner = _process_peaks_parallel_runner

    def _set_last_simulation_diagnostics(
        diagnostics: Mapping[str, object] | None,
    ) -> None:
        last_simulation_diagnostics_state.clear()
        copied = _copy_simulation_diag_value(diagnostics if diagnostics is not None else {})
        if isinstance(copied, dict):
            last_simulation_diagnostics_state.update(copied)

    def _last_simulation_diagnostics() -> dict[str, object]:
        copied = _copy_simulation_diag_value(last_simulation_diagnostics_state)
        return copied if isinstance(copied, dict) else {}

    def _simulate_hit_tables(
        miller_array: np.ndarray,
        intensity_array: np.ndarray,
        image_size: int,
        param_set: Mapping[str, object] | dict[str, object],
    ) -> list[object]:
        try:
            result = simulate_geometry_fit_hit_tables(
                miller_array,
                intensity_array,
                image_size,
                param_set,
                build_geometry_fit_central_mosaic_params=(build_geometry_fit_central_mosaic_params),
                process_peaks_parallel=process_peaks_parallel_runner,
                default_solve_q_steps=default_solve_q_steps,
                default_solve_q_rel_tol=default_solve_q_rel_tol,
                default_solve_q_mode=default_solve_q_mode,
            )
        except Exception:
            _set_last_simulation_diagnostics(
                _function_last_diagnostics(simulate_geometry_fit_hit_tables)
            )
            raise
        _set_last_simulation_diagnostics(
            _function_last_diagnostics(simulate_geometry_fit_hit_tables)
        )
        return result

    def _simulate_peak_centers(
        miller_array: np.ndarray,
        intensity_array: np.ndarray,
        image_size: int,
        param_set: Mapping[str, object] | dict[str, object],
    ) -> list[dict[str, object]]:
        try:
            result = simulate_geometry_fit_peak_centers(
                miller_array,
                intensity_array,
                image_size,
                param_set,
                build_geometry_fit_central_mosaic_params=(build_geometry_fit_central_mosaic_params),
                process_peaks_parallel=process_peaks_parallel_runner,
                hit_tables_to_max_positions=hit_tables_to_max_positions,
                default_solve_q_steps=default_solve_q_steps,
                default_solve_q_rel_tol=default_solve_q_rel_tol,
                default_solve_q_mode=default_solve_q_mode,
            )
        except Exception:
            _set_last_simulation_diagnostics(
                _function_last_diagnostics(simulate_geometry_fit_peak_centers)
            )
            raise
        _set_last_simulation_diagnostics(
            _function_last_diagnostics(simulate_geometry_fit_peak_centers)
        )
        return result

    def _simulate_preview_style_peaks(
        miller_array: np.ndarray,
        intensity_array: np.ndarray,
        image_size: int,
        param_set: Mapping[str, object] | dict[str, object],
    ) -> list[dict[str, object]]:
        peak_table_lattice = _resolve_runtime_value(peak_table_lattice_factory)
        try:
            result = simulate_geometry_fit_preview_style_peaks(
                miller_array,
                intensity_array,
                image_size,
                param_set,
                build_geometry_fit_central_mosaic_params=(build_geometry_fit_central_mosaic_params),
                process_peaks_parallel=process_peaks_parallel_runner,
                native_sim_to_display_coords=native_sim_to_display_coords,
                peak_table_lattice=(
                    list(peak_table_lattice)
                    if isinstance(peak_table_lattice, Sequence)
                    and not isinstance(peak_table_lattice, (str, bytes))
                    else peak_table_lattice
                ),
                primary_a=_resolve_runtime_value(primary_a_factory),
                primary_c=_resolve_runtime_value(primary_c_factory),
                default_source_label=default_source_label,
                round_pixel_centers=bool(round_pixel_centers),
                default_solve_q_steps=default_solve_q_steps,
                default_solve_q_rel_tol=default_solve_q_rel_tol,
                default_solve_q_mode=default_solve_q_mode,
            )
        except Exception:
            _set_last_simulation_diagnostics(
                _function_last_diagnostics(simulate_geometry_fit_preview_style_peaks)
            )
            raise
        _set_last_simulation_diagnostics(
            _function_last_diagnostics(simulate_geometry_fit_preview_style_peaks)
        )
        return result

    return GeometryFitSimulationRuntimeCallbacks(
        simulate_hit_tables=_simulate_hit_tables,
        simulate_peak_centers=_simulate_peak_centers,
        simulate_preview_style_peaks=_simulate_preview_style_peaks,
        last_simulation_diagnostics=_last_simulation_diagnostics,
    )


def make_runtime_geometry_q_group_value_callbacks(
    *,
    simulation_runtime_state,
    preview_state,
    q_group_state,
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names_factory: object,
    primary_a_factory: object,
    primary_c_factory: object,
    image_size_factory: object,
    native_sim_to_display_coords: Callable[
        [float, float, tuple[int, int]],
        tuple[float, float],
    ],
) -> GeometryQGroupRuntimeValueCallbacks:
    """Return live Qr/Qz selector value callbacks from runtime sources."""

    _last_live_preview_cache_metadata_state: dict[str, object] = {}

    def _set_live_preview_cache_metadata(**fields: object) -> None:
        _last_live_preview_cache_metadata_state.clear()
        _last_live_preview_cache_metadata_state.update(fields)

    def _last_live_preview_cache_metadata() -> dict[str, object]:
        return dict(_last_live_preview_cache_metadata_state)

    def _current_geometry_fit_var_names() -> list[object]:
        raw_value = _resolve_runtime_value(current_geometry_fit_var_names_factory)
        if raw_value is None:
            return []
        if isinstance(raw_value, Sequence) and not isinstance(
            raw_value,
            (str, bytes),
        ):
            return list(raw_value)
        try:
            return list(raw_value)
        except Exception:
            return []

    def _primary_a() -> float:
        return _coerce_float(_resolve_runtime_value(primary_a_factory), float("nan"))

    def _primary_c() -> float:
        return _coerce_float(_resolve_runtime_value(primary_c_factory), float("nan"))

    def _has_intersection_cache() -> bool:
        for attr_name in (
            "stored_primary_intersection_cache",
            "stored_secondary_intersection_cache",
            "stored_intersection_cache",
            "last_caked_intersection_cache",
        ):
            cache_tables = getattr(simulation_runtime_state, attr_name, None)
            if (
                isinstance(cache_tables, Sequence)
                and not isinstance(
                    cache_tables,
                    (str, bytes),
                )
                and len(cache_tables) > 0
            ):
                return True
        return False

    def _build_live_preview_simulated_peaks_from_cache() -> list[dict[str, object]]:
        max_positions_local = simulation_runtime_state.stored_max_positions_local
        peak_record_count = int(len(getattr(simulation_runtime_state, "peak_records", ()) or ()))
        max_positions_row_count = (
            int(len(max_positions_local))
            if isinstance(max_positions_local, Sequence)
            and not isinstance(max_positions_local, (str, bytes))
            else 0
        )
        if isinstance(max_positions_local, Sequence) and not isinstance(
            max_positions_local,
            (str, bytes),
        ):
            use_nominal_cache_grouping = _has_intersection_cache()
            stored_sim_image = getattr(simulation_runtime_state, "stored_sim_image", None)
            if stored_sim_image is not None:
                image_shape = tuple(int(v) for v in stored_sim_image.shape[:2])
            else:
                image_size = max(
                    0,
                    _coerce_int(_resolve_runtime_value(image_size_factory), 0),
                )
                image_shape = (image_size, image_size)

            peak_kwargs: dict[str, object] = {
                "image_shape": image_shape,
                "native_sim_to_display_coords": native_sim_to_display_coords,
                "peak_table_lattice": simulation_runtime_state.stored_peak_table_lattice,
                "primary_a": _primary_a(),
                "primary_c": _primary_c(),
                "default_source_label": "primary",
                "round_pixel_centers": True,
                "allow_nominal_hkl_indices": use_nominal_cache_grouping,
            }
            if simulation_runtime_state.stored_source_reflection_indices_local is not None:
                peak_kwargs["source_reflection_indices"] = (
                    simulation_runtime_state.stored_source_reflection_indices_local
                )
            simulated_peaks = build_geometry_fit_simulated_peaks(
                max_positions_local,
                **peak_kwargs,
            )
            if simulated_peaks:
                _set_live_preview_cache_metadata(
                    cache_source="max_positions",
                    fallback_used=False,
                    max_positions_row_count=int(max_positions_row_count),
                    peak_record_count=int(peak_record_count),
                    active_signature_matches=None,
                    source_snapshot_row_count=0,
                    source_snapshot_background_index=None,
                    simulated_peak_count=int(len(simulated_peaks)),
                )
                return simulated_peaks

        current_signature = getattr(
            simulation_runtime_state,
            "stored_hit_table_signature",
            None,
        )
        if current_signature is None:
            current_signature = getattr(
                simulation_runtime_state,
                "last_simulation_signature",
                None,
            )
        current_signature_summary = _signature_summary(current_signature)
        provenance = _resolve_live_peak_record_fallback_provenance(
            simulation_runtime_state,
            signature=current_signature,
            signature_summary=current_signature_summary,
            source_reflection_indices_local=(
                getattr(
                    simulation_runtime_state,
                    "stored_source_reflection_indices_local",
                    None,
                )
            ),
        )
        cached_peaks = gui_manual_geometry.geometry_manual_live_peak_candidates_from_records(
            getattr(simulation_runtime_state, "peak_records", None),
            normalize_hkl_key=gui_geometry_overlay.normalize_hkl_key,
            source_reflection_indices_local=(
                getattr(
                    simulation_runtime_state,
                    "stored_source_reflection_indices_local",
                    None,
                )
            ),
            source_row_hkl_lookup=provenance.get("source_row_hkl_lookup"),
            provenance_signature_matches=bool(provenance.get("active_signature_matches", False)),
            provenance_revision_matches=bool(provenance.get("active_revision_matches", False)),
            expected_table_count=provenance.get("expected_table_count"),
        )
        for entry in cached_peaks:
            if _has_intersection_cache() and "q_group_nominal_hkl" not in entry:
                entry["q_group_nominal_hkl"] = True
            raw_group_key = entry.get("q_group_key")
            if isinstance(raw_group_key, list):
                entry["q_group_key"] = tuple(raw_group_key)
            elif not isinstance(raw_group_key, tuple):
                group_key = geometry_q_group_key_from_entry(entry)
                if group_key is not None:
                    entry["q_group_key"] = group_key
        _set_live_preview_cache_metadata(
            cache_source="peak_records_fallback",
            fallback_used=True,
            max_positions_row_count=int(max_positions_row_count),
            peak_record_count=int(peak_record_count),
            active_signature_matches=bool(provenance.get("active_signature_matches", False)),
            source_snapshot_row_count=int(provenance.get("source_snapshot_row_count", 0) or 0),
            source_snapshot_background_index=provenance.get("source_snapshot_background_index"),
            simulated_peak_count=int(len(cached_peaks)),
        )
        if cached_peaks:
            return cached_peaks

        _set_live_preview_cache_metadata(
            cache_source="empty",
            fallback_used=bool(max_positions_row_count > 0 or peak_record_count > 0),
            max_positions_row_count=int(max_positions_row_count),
            peak_record_count=int(peak_record_count),
            active_signature_matches=bool(provenance.get("active_signature_matches", False)),
            source_snapshot_row_count=int(provenance.get("source_snapshot_row_count", 0) or 0),
            source_snapshot_background_index=provenance.get("source_snapshot_background_index"),
            simulated_peak_count=0,
        )
        return []

    def _filter_simulated_peaks(
        simulated_peaks: Sequence[dict[str, object]] | None,
    ) -> tuple[list[dict[str, object]], int, int]:
        return filter_geometry_fit_simulated_peaks(
            simulated_peaks,
            listed_keys=gui_controllers.listed_geometry_q_group_keys(q_group_state),
            excluded_q_groups=getattr(preview_state, "excluded_q_groups", set()),
        )

    def _collapse_simulated_peaks(
        simulated_peaks: Sequence[dict[str, object]] | None,
        *,
        merge_radius_px: float = 6.0,
    ) -> tuple[list[dict[str, object]], int]:
        return collapse_geometry_fit_simulated_peaks(
            simulated_peaks,
            merge_radius_px=merge_radius_px,
        )

    def _build_entries_snapshot() -> list[dict[str, object]]:
        return build_geometry_q_group_entries(
            simulation_runtime_state.stored_max_positions_local,
            peak_table_lattice=simulation_runtime_state.stored_peak_table_lattice,
            peak_records=getattr(simulation_runtime_state, "peak_records", None),
            primary_a=_primary_a(),
            primary_c=_primary_c(),
            allow_nominal_hkl_indices=_has_intersection_cache(),
        )

    def _listed_entries() -> list[dict[str, object]]:
        return gui_controllers.listed_geometry_q_group_entries(q_group_state)

    def _listed_keys(
        entries: Sequence[dict[str, object]] | None = None,
    ) -> set[tuple[object, ...]]:
        return gui_controllers.listed_geometry_q_group_keys(q_group_state, entries)

    def _export_rows(
        entries: Sequence[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        return build_geometry_q_group_export_rows(
            preview_state=preview_state,
            q_group_state=q_group_state,
            entries=entries,
        )

    def _current_min_matches() -> int:
        return current_geometry_auto_match_min_matches(
            fit_config,
            _current_geometry_fit_var_names(),
        )

    def _excluded_count(
        entries: Sequence[dict[str, object]] | None = None,
    ) -> int:
        return geometry_q_group_excluded_count(
            preview_state,
            q_group_state,
            entries,
        )

    def _build_window_status(
        entries: Sequence[dict[str, object]] | None = None,
    ) -> str:
        return build_geometry_q_group_window_status_text(
            preview_state=preview_state,
            q_group_state=q_group_state,
            fit_config=fit_config,
            current_geometry_fit_var_names=_current_geometry_fit_var_names(),
            entries=entries,
        )

    def _build_preview_exclude_button_label(
        entries: Sequence[dict[str, object]] | None = None,
    ) -> str:
        return build_geometry_preview_exclude_button_label(
            preview_state=preview_state,
            q_group_state=q_group_state,
            entries=entries,
        )

    def _live_preview_match_key(
        entry: dict[str, object] | None,
    ) -> tuple[object, ...] | None:
        return live_geometry_preview_match_key(entry)

    def _live_preview_match_hkl(
        entry: dict[str, object] | None,
    ) -> tuple[int, int, int] | None:
        return live_geometry_preview_match_hkl(entry)

    def _live_preview_match_is_excluded(entry: dict[str, object] | None) -> bool:
        return live_geometry_preview_match_is_excluded(preview_state, entry)

    def _filter_live_preview_matches(
        matched_pairs: Sequence[dict[str, object]] | None,
    ) -> tuple[list[dict[str, object]], int]:
        return filter_live_geometry_preview_matches(preview_state, matched_pairs)

    def _apply_live_preview_match_exclusions(
        matched_pairs: Sequence[dict[str, object]] | None,
        match_stats: dict[str, object] | None,
    ) -> tuple[list[dict[str, object]], dict[str, object], int]:
        return apply_live_geometry_preview_match_exclusions(
            preview_state,
            matched_pairs,
            match_stats,
        )

    return GeometryQGroupRuntimeValueCallbacks(
        build_live_preview_simulated_peaks_from_cache=(
            _build_live_preview_simulated_peaks_from_cache
        ),
        last_live_preview_cache_metadata=_last_live_preview_cache_metadata,
        filter_simulated_peaks=_filter_simulated_peaks,
        collapse_simulated_peaks=_collapse_simulated_peaks,
        build_entries_snapshot=_build_entries_snapshot,
        clone_entries=gui_controllers.clone_geometry_q_group_entries,
        listed_entries=_listed_entries,
        listed_keys=_listed_keys,
        key_from_jsonable=geometry_q_group_key_from_jsonable,
        export_rows=_export_rows,
        format_line=format_geometry_q_group_line,
        current_min_matches=_current_min_matches,
        excluded_count=_excluded_count,
        build_window_status=_build_window_status,
        build_preview_exclude_button_label=_build_preview_exclude_button_label,
        live_preview_match_key=_live_preview_match_key,
        live_preview_match_hkl=_live_preview_match_hkl,
        live_preview_match_is_excluded=_live_preview_match_is_excluded,
        filter_live_preview_matches=_filter_live_preview_matches,
        apply_live_preview_match_exclusions=_apply_live_preview_match_exclusions,
    )


def filter_geometry_fit_simulated_peaks(
    simulated_peaks: Sequence[dict[str, object]] | None,
    *,
    listed_keys: Sequence[tuple[object, ...]] | None = None,
    excluded_q_groups: Sequence[tuple[object, ...]] | set[tuple[object, ...]] | None = None,
) -> tuple[list[dict[str, object]], int, int]:
    """Apply the current Qr/Qz selector state to geometry-fit seeds."""

    filtered: list[dict[str, object]] = []
    excluded_count = 0
    available_keys: set[tuple[object, ...]] = set()
    listed_keys_local = set(listed_keys or ())
    restrict_to_listed = bool(listed_keys_local)
    if restrict_to_listed:
        available_keys = set(listed_keys_local)
    excluded_q_group_keys = set(excluded_q_groups or ())

    for raw_entry in simulated_peaks or []:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        group_key = geometry_q_group_key_from_entry(entry)
        if group_key is None:
            excluded_count += 1
            continue
        entry["q_group_key"] = group_key
        if restrict_to_listed and group_key not in listed_keys_local:
            excluded_count += 1
            continue
        if not restrict_to_listed:
            available_keys.add(group_key)
        if group_key in excluded_q_group_keys:
            excluded_count += 1
            continue
        filtered.append(entry)

    return filtered, int(excluded_count), int(len(available_keys))


def _geometry_fit_seed_representative_sort_key(
    entry: Mapping[str, object] | None,
) -> tuple[object, ...]:
    """Return a stable ordering key for degenerate geometry-fit seeds."""

    if not isinstance(entry, Mapping):
        return (2, "")
    branch_idx, _branch_source, _branch_reason = resolve_canonical_branch(
        entry,
        allow_legacy_peak_fallback=False,
    )
    if branch_idx in {0, 1}:
        return (0, int(branch_idx))
    hkl_key = gui_geometry_overlay.normalize_hkl_key(entry.get("hkl", entry.get("label")))
    if hkl_key is not None:
        return (1, int(hkl_key[0]), int(hkl_key[1]), int(hkl_key[2]))
    return (2, str(entry.get("label", "")))


def _geometry_fit_seed_sim_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    """Return the simulated detector point for one geometry-fit seed entry."""

    if not isinstance(entry, Mapping):
        return None
    try:
        sim_col = float(entry.get("sim_col", np.nan))
        sim_row = float(entry.get("sim_row", np.nan))
    except Exception:
        return None
    if not (np.isfinite(sim_col) and np.isfinite(sim_row)):
        return None
    return float(sim_col), float(sim_row)


def _geometry_fit_seed_cluster_anchor(
    entries: Sequence[dict[str, object]],
) -> tuple[float, float] | None:
    """Return the mean simulated point for one cluster of seed entries."""

    cols: list[float] = []
    rows: list[float] = []
    for entry in entries:
        point = _geometry_fit_seed_sim_point(entry)
        if point is None:
            continue
        cols.append(float(point[0]))
        rows.append(float(point[1]))
    if not cols or not rows:
        return None
    return float(np.mean(np.asarray(cols, dtype=float))), float(
        np.mean(np.asarray(rows, dtype=float))
    )


def collapse_geometry_fit_simulated_peaks(
    simulated_peaks: Sequence[dict[str, object]] | None,
    *,
    merge_radius_px: float = 6.0,
) -> tuple[list[dict[str, object]], int]:
    """Collapse overlapping degenerate geometry-fit seeds within each Qr/Qz group."""

    grouped_entries: dict[object, list[dict[str, object]]] = {}
    ordered_keys: list[object] = []
    ungrouped_index = 0
    merge_radius = max(0.0, float(merge_radius_px))

    for raw_entry in simulated_peaks or []:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        group_key = entry.get("q_group_key")
        if group_key is None:
            group_key = ("ungrouped", ungrouped_index)
            ungrouped_index += 1
        if group_key not in grouped_entries:
            grouped_entries[group_key] = []
            ordered_keys.append(group_key)
        grouped_entries[group_key].append(entry)

    collapsed: list[dict[str, object]] = []
    collapsed_degenerate_count = 0

    for group_key in ordered_keys:
        entries = grouped_entries.get(group_key, [])
        if not entries:
            continue

        clusters: list[list[dict[str, object]]] = []
        cluster_anchors: list[tuple[float, float] | None] = []
        for entry in entries:
            point = _geometry_fit_seed_sim_point(entry)
            chosen_cluster_idx = None
            chosen_cluster_dist = float("inf")
            if point is not None and merge_radius > 0.0:
                for cluster_idx, anchor in enumerate(cluster_anchors):
                    if anchor is None:
                        continue
                    dist = float(math.hypot(point[0] - anchor[0], point[1] - anchor[1]))
                    if dist <= merge_radius and dist < chosen_cluster_dist:
                        chosen_cluster_idx = cluster_idx
                        chosen_cluster_dist = dist
            if chosen_cluster_idx is None:
                clusters.append([entry])
                cluster_anchors.append(point)
                continue
            clusters[chosen_cluster_idx].append(entry)
            cluster_anchors[chosen_cluster_idx] = _geometry_fit_seed_cluster_anchor(
                clusters[chosen_cluster_idx]
            )

        for cluster_entries in clusters:
            representative = min(
                cluster_entries,
                key=_geometry_fit_seed_representative_sort_key,
            )
            merged = dict(representative)
            degenerate_hkls: list[tuple[int, int, int]] = []
            seen_hkls: set[tuple[int, int, int]] = set()
            total_weight = 0.0

            for entry in cluster_entries:
                hkl_key = gui_geometry_overlay.normalize_hkl_key(
                    entry.get("hkl", entry.get("label"))
                )
                if hkl_key is not None and hkl_key not in seen_hkls:
                    seen_hkls.add(hkl_key)
                    degenerate_hkls.append(hkl_key)
                try:
                    weight = float(entry.get("weight", 0.0))
                except Exception:
                    weight = 0.0
                if np.isfinite(weight) and weight > 0.0:
                    total_weight += float(weight)
                else:
                    total_weight += 1.0

            merged["weight"] = float(total_weight)
            merged["degenerate_count"] = int(len(cluster_entries))
            merged["degenerate_hkls"] = list(degenerate_hkls)
            collapsed_degenerate_count += max(0, len(cluster_entries) - 1)
            collapsed.append(merged)

    return collapsed, int(collapsed_degenerate_count)


def format_geometry_q_group_line(entry: Mapping[str, object]) -> str:
    """Return a compact label for one Qr/Qz selector row."""

    qr_val = _coerce_float(entry.get("qr", np.nan), float("nan"))
    qz_val = _coerce_float(entry.get("qz", np.nan), float("nan"))
    total_intensity = _coerce_float(entry.get("total_intensity", 0.0), 0.0)
    peak_count = _coerce_int(entry.get("peak_count", 0), 0)
    source_label = str(entry.get("source_label", ""))
    gz_index = entry.get("gz_index")
    if gz_index is None:
        key = entry.get("key")
        if isinstance(key, tuple) and len(key) >= 4:
            gz_index = key[3]
    hkl_items = entry.get("hkl_preview", [])
    if isinstance(hkl_items, np.ndarray):
        hkl_items = list(hkl_items)
    elif not isinstance(hkl_items, Sequence) or isinstance(hkl_items, str | bytes):
        hkl_items = []
    else:
        hkl_items = list(hkl_items)
    hkl_preview = ", ".join(str(hkl) for hkl in hkl_items[:3])
    if len(hkl_items) > 3:
        hkl_preview += ", ..."
    gz_text = f"{int(gz_index):4d}" if gz_index is not None else " n/a"
    return (
        f"{source_label:<9}  "
        f"Qr={qr_val:8.2f}  "
        f"Gz={gz_text}  "
        f"Qz={qz_val:8.2f}  "
        f"I={total_intensity:10.3f}  "
        f"hits={peak_count:4d}" + (f"  HKL={hkl_preview}" if hkl_preview else "")
    )


def current_geometry_auto_match_min_matches(
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names: Sequence[str] | None,
) -> int:
    """Return the current geometry auto-match minimum peak count."""

    geometry_refine_cfg = (
        fit_config.get("geometry", {})
        if isinstance(
            fit_config,
            Mapping,
        )
        else {}
    )
    if not isinstance(geometry_refine_cfg, Mapping):
        geometry_refine_cfg = {}
    auto_match_cfg = geometry_refine_cfg.get("auto_match", {}) or {}
    if not isinstance(auto_match_cfg, Mapping):
        auto_match_cfg = {}
    default_min_matches = max(6, len(list(current_geometry_fit_var_names or ())) + 2)
    try:
        min_matches = int(auto_match_cfg.get("min_matches", default_min_matches))
    except Exception:
        min_matches = int(default_min_matches)
    return max(1, int(min_matches))


def build_live_geometry_preview_auto_match_config(
    fit_config: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return the normalized auto-match config used for live preview refreshes."""

    geometry_refine_cfg = (
        fit_config.get("geometry", {})
        if isinstance(
            fit_config,
            Mapping,
        )
        else {}
    )
    if not isinstance(geometry_refine_cfg, Mapping):
        geometry_refine_cfg = {}
    auto_match_cfg = geometry_refine_cfg.get("auto_match", {}) or {}
    if not isinstance(auto_match_cfg, Mapping):
        auto_match_cfg = {}

    preview_auto_match_cfg = dict(auto_match_cfg)
    preview_auto_match_cfg["relax_on_low_matches"] = False
    search_radius = _coerce_float(auto_match_cfg.get("search_radius_px", 24.0), 24.0)
    preview_auto_match_cfg.setdefault(
        "context_margin_px",
        max(192.0, 8.0 * float(search_radius)),
    )
    return preview_auto_match_cfg


def geometry_q_group_excluded_count(
    preview_state,
    q_group_state,
    entries: Sequence[dict[str, object]] | None = None,
) -> int:
    """Count excluded Qr/Qz rows, optionally scoped to one entry list."""

    keys = gui_controllers.listed_geometry_q_group_keys(q_group_state, entries)
    if not keys:
        return gui_controllers.count_geometry_preview_excluded_q_groups(preview_state)
    return gui_controllers.count_geometry_preview_excluded_q_groups(
        preview_state,
        keys,
    )


def build_geometry_preview_exclude_button_label(
    *,
    preview_state,
    q_group_state,
    entries: Sequence[dict[str, object]] | None = None,
) -> str:
    """Return the toolbar label for the Qr/Qz preview-selector action."""

    label = "Choose Active Qr/Qz Groups"
    excluded_count = geometry_q_group_excluded_count(
        preview_state,
        q_group_state,
        entries,
    )
    if excluded_count > 0:
        label += f" ({excluded_count} off)"
    return label


def build_geometry_q_group_window_status_text(
    *,
    preview_state,
    q_group_state,
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names: Sequence[str] | None,
    entries: Sequence[dict[str, object]] | None = None,
) -> str:
    """Build the summary text shown above the Qr/Qz selector rows."""

    rows = (
        list(entries)
        if entries is not None
        else gui_controllers.listed_geometry_q_group_entries(q_group_state)
    )
    excluded_q_groups = getattr(preview_state, "excluded_q_groups", set())
    total_count = len(rows)
    included_rows = [entry for entry in rows if entry.get("key") not in excluded_q_groups]
    selected_peak_count = int(
        sum(_coerce_int(entry.get("peak_count", 0), 0) for entry in included_rows)
    )
    total_peak_count = int(sum(_coerce_int(entry.get("peak_count", 0), 0) for entry in rows))
    min_matches = current_geometry_auto_match_min_matches(
        fit_config,
        current_geometry_fit_var_names,
    )
    shortfall = max(0, int(min_matches - selected_peak_count))
    selected_intensity = float(
        sum(_coerce_float(entry.get("total_intensity", 0.0), 0.0) for entry in included_rows)
    )
    total_intensity = float(
        sum(_coerce_float(entry.get("total_intensity", 0.0), 0.0) for entry in rows)
    )
    return (
        f"Included Qr/Qz groups: {len(included_rows)}/{total_count}  "
        f"Selected peaks: {selected_peak_count}/{total_peak_count}  "
        f"Need >= {min_matches}"
        + (f"  short {shortfall}" if shortfall > 0 else "  ready")
        + "\n"
        + f"Intensity={selected_intensity:.3f}/{total_intensity:.3f}  "
        + 'Listed peaks stay fixed until you press "Update Listed Peaks".'
    )


def update_geometry_q_group_window_status(
    *,
    view_state,
    preview_state,
    q_group_state,
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names: Sequence[str] | None,
    entries: Sequence[dict[str, object]] | None = None,
) -> None:
    """Refresh the summary label shown at the top of the selector window."""

    gui_views.set_geometry_q_group_status_text(
        view_state,
        build_geometry_q_group_window_status_text(
            preview_state=preview_state,
            q_group_state=q_group_state,
            fit_config=fit_config,
            current_geometry_fit_var_names=current_geometry_fit_var_names,
            entries=entries,
        ),
    )


def refresh_geometry_q_group_window(
    *,
    view_state,
    preview_state,
    q_group_state,
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names: Sequence[str] | None,
    on_toggle: Callable[[tuple[object, ...] | None, object], None],
) -> bool:
    """Redraw the Qr/Qz selector window from the stored manual snapshot."""

    entries = gui_controllers.listed_geometry_q_group_entries(q_group_state)
    return gui_views.refresh_geometry_q_group_window(
        view_state=view_state,
        entries=entries,
        excluded_q_groups=getattr(preview_state, "excluded_q_groups", set()),
        status_text=build_geometry_q_group_window_status_text(
            preview_state=preview_state,
            q_group_state=q_group_state,
            fit_config=fit_config,
            current_geometry_fit_var_names=current_geometry_fit_var_names,
            entries=entries,
        ),
        format_line=format_geometry_q_group_line,
        on_toggle=on_toggle,
        clear_row_vars=lambda: gui_controllers.clear_geometry_q_group_row_vars(
            q_group_state,
        ),
        register_row_var=lambda group_key, row_var: gui_controllers.set_geometry_q_group_row_var(
            q_group_state,
            group_key,
            row_var,
        ),
    )


def apply_geometry_q_group_checkbox_change(
    preview_state,
    group_key: tuple[object, ...] | None,
    row_var: object,
) -> str | None:
    """Apply one Qr/Qz include/exclude toggle from the selector window."""

    if group_key is None:
        return None
    included = _row_var_value(row_var)
    gui_controllers.set_geometry_preview_q_group_included(
        preview_state,
        group_key,
        included=included,
    )
    return "Included" if included else "Excluded"


def set_all_geometry_q_groups_enabled(
    preview_state,
    q_group_state,
    *,
    enabled: bool,
) -> tuple[str, int]:
    """Enable or disable every currently listed Qr/Qz group."""

    entries = gui_controllers.listed_geometry_q_group_entries(q_group_state)
    if enabled:
        gui_controllers.clear_geometry_preview_excluded_q_groups(preview_state)
        action = "Included"
    else:
        gui_controllers.replace_geometry_preview_excluded_q_groups(
            preview_state,
            [entry["key"] for entry in entries if entry.get("key") is not None],
        )
        action = "Excluded"
    return action, len(entries)


def request_geometry_q_group_window_update(q_group_state) -> None:
    """Mark the Qr/Qz selector listing for refresh on the next update."""

    gui_controllers.request_geometry_q_group_refresh(q_group_state)


def replace_geometry_q_group_entries_snapshot_with_side_effects(
    *,
    preview_state,
    q_group_state,
    entries: Sequence[dict[str, object]] | None,
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    update_geometry_preview_exclude_button_label: Callable[[], None],
) -> list[dict[str, object]]:
    """Replace the stored Qr/Qz snapshot and keep dependent state in sync."""

    gui_controllers.replace_geometry_q_group_cached_entries(
        q_group_state,
        entries,
    )
    gui_controllers.retain_geometry_preview_excluded_q_groups(
        preview_state,
        gui_controllers.listed_geometry_q_group_keys(q_group_state),
    )
    invalidate_geometry_manual_pick_cache()
    update_geometry_preview_exclude_button_label()
    return gui_controllers.listed_geometry_q_group_entries(q_group_state)


def close_geometry_q_group_window(view_state, q_group_state) -> None:
    """Destroy the Qr/Qz selector window and clear its row-var map."""

    gui_views.close_geometry_q_group_window(view_state)
    gui_controllers.clear_geometry_q_group_row_vars(q_group_state)


def open_geometry_q_group_window(
    *,
    root,
    view_state,
    preview_state,
    q_group_state,
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names: Sequence[str] | None,
    on_toggle: Callable[[tuple[object, ...] | None, object], None],
    on_include_all: Callable[[], None],
    on_exclude_all: Callable[[], None],
    on_update_listed_peaks: Callable[[], None],
    on_save: Callable[[], None],
    on_load: Callable[[], None],
    on_close: Callable[[], None],
) -> bool:
    """Open and refresh the geometry Q-group selector window."""

    opened = gui_views.open_geometry_q_group_window(
        root=root,
        view_state=view_state,
        on_include_all=on_include_all,
        on_exclude_all=on_exclude_all,
        on_update_listed_peaks=on_update_listed_peaks,
        on_save=on_save,
        on_load=on_load,
        on_close=on_close,
    )
    refresh_geometry_q_group_window(
        view_state=view_state,
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config=fit_config,
        current_geometry_fit_var_names=current_geometry_fit_var_names,
        on_toggle=on_toggle,
    )
    return opened


def geometry_q_group_key_to_jsonable(group_key: object) -> list[object] | None:
    """Convert one stable Qr/Qz group key into a JSON-safe list."""

    return gui_manual_geometry.geometry_q_group_key_to_jsonable(group_key)


def geometry_q_group_key_from_jsonable(value: object) -> tuple[object, ...] | None:
    """Rebuild one stable Qr/Qz group key from JSON-loaded data."""

    return gui_manual_geometry.geometry_q_group_key_from_jsonable(value)


def geometry_q_group_float_for_json(value: object) -> float | None:
    """Return a finite float for JSON export, or ``None`` when unavailable."""

    numeric = _coerce_float(value, float("nan"))
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def build_geometry_q_group_export_rows(
    *,
    preview_state,
    q_group_state,
    entries: Sequence[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Build JSON export rows for the current Qr/Qz selector listing."""

    rows: list[dict[str, object]] = []
    source_entries = (
        list(entries)
        if entries is not None
        else gui_controllers.listed_geometry_q_group_entries(q_group_state)
    )
    excluded_q_groups = getattr(preview_state, "excluded_q_groups", set())
    for entry in source_entries:
        if not isinstance(entry, Mapping):
            continue
        group_key = entry.get("key")
        serialized_key = geometry_q_group_key_to_jsonable(group_key)
        if serialized_key is None:
            continue
        hkl_preview = []
        for hkl_value in entry.get("hkl_preview", [])[:8]:
            if not isinstance(hkl_value, (list, tuple, np.ndarray)) or len(hkl_value) < 3:
                continue
            try:
                hkl_preview.append([int(hkl_value[0]), int(hkl_value[1]), int(hkl_value[2])])
            except Exception:
                continue
        rows.append(
            {
                "key": serialized_key,
                "included": bool(group_key not in excluded_q_groups),
                "source_label": str(entry.get("source_label", "")),
                "qr": geometry_q_group_float_for_json(entry.get("qr", np.nan)),
                "qz": geometry_q_group_float_for_json(entry.get("qz", np.nan)),
                "gz_index": int(entry.get("gz_index", serialized_key[3])),
                "total_intensity": geometry_q_group_float_for_json(
                    entry.get("total_intensity", np.nan)
                ),
                "peak_count": int(entry.get("peak_count", 0)),
                "hkl_preview": hkl_preview,
                "display_label": format_geometry_q_group_line(entry),
            }
        )
    return rows


def build_geometry_q_group_save_payload(
    export_rows: Sequence[Mapping[str, object]],
    *,
    saved_at: str,
) -> dict[str, object]:
    """Build the JSON payload written by the Qr/Qz selector save action."""

    return {
        "type": "ra_sim.geometry_q_group_selection",
        "version": 1,
        "saved_at": str(saved_at),
        "row_count": int(len(export_rows)),
        "included_count": int(sum(1 for row in export_rows if bool(row.get("included", False)))),
        "rows": [dict(row) for row in export_rows],
    }


def load_geometry_q_group_saved_state(
    payload: object,
) -> tuple[dict[tuple[object, ...], bool] | None, str | None]:
    """Validate one saved selector payload and rebuild its inclusion map."""

    if not isinstance(payload, Mapping):
        return None, "Invalid Qr/Qz peak list file: expected a JSON object."
    if str(payload.get("type", "")) != "ra_sim.geometry_q_group_selection":
        return None, "Invalid Qr/Qz peak list file type."

    saved_rows = payload.get("rows", [])
    if not isinstance(saved_rows, list) or not saved_rows:
        return None, "Loaded Qr/Qz peak list is empty."

    saved_state: dict[tuple[object, ...], bool] = {}
    for row in saved_rows:
        if not isinstance(row, Mapping):
            continue
        group_key = geometry_q_group_key_from_jsonable(row.get("key"))
        if group_key is None:
            continue
        saved_state[group_key] = bool(row.get("included", True))

    if not saved_state:
        return None, "Loaded Qr/Qz peak list does not contain any valid rows."
    return saved_state, None


def apply_loaded_geometry_q_group_saved_state(
    *,
    preview_state,
    q_group_state,
    saved_state: Mapping[tuple[object, ...], bool] | None,
) -> tuple[dict[str, int] | None, str | None]:
    """Apply one loaded selector inclusion map to the current listed rows."""

    if not isinstance(saved_state, Mapping) or not saved_state:
        return None, "Loaded Qr/Qz peak list does not contain any valid rows."

    current_entries = gui_controllers.listed_geometry_q_group_entries(q_group_state)
    current_keys = [
        entry.get("key")
        for entry in current_entries
        if isinstance(entry, Mapping) and entry.get("key") is not None
    ]
    if not current_keys:
        return None, (
            "No listed Qr/Qz groups are available to match against the saved list. "
            'Press "Update Listed Peaks" first.'
        )

    current_key_set = set(current_keys)
    matched_keys = current_key_set.intersection(saved_state.keys())
    if not matched_keys:
        return None, "Loaded Qr/Qz peak list does not match any currently listed groups."

    gui_controllers.replace_geometry_preview_excluded_q_groups(
        preview_state,
        [
            key
            for key in current_keys
            if (key not in saved_state) or (not bool(saved_state.get(key, False)))
        ],
    )
    return {
        "matched_total": int(len(matched_keys)),
        "included_total": int(sum(1 for key in matched_keys if bool(saved_state.get(key, False)))),
        "current_only": int(sum(1 for key in current_keys if key not in saved_state)),
        "saved_only": int(sum(1 for key in saved_state if key not in current_key_set)),
    }, None


def live_geometry_preview_match_key(
    entry: dict[str, object] | None,
) -> tuple[object, ...] | None:
    """Build a stable exclusion key for one live preview match entry."""

    if not isinstance(entry, dict):
        return None
    hkl_key = gui_geometry_overlay.normalize_hkl_key(entry.get("hkl", entry.get("label")))
    source_label = entry.get("source_label")
    source_table_index = entry.get(
        "source_reflection_index",
        entry.get("source_table_index"),
    )
    source_row_index = entry.get("source_row_index")
    if (
        hkl_key is not None
        and source_label is not None
        and source_table_index is not None
        and source_row_index is not None
    ):
        try:
            return (
                "peak",
                str(source_label),
                int(source_table_index),
                int(source_row_index),
                int(hkl_key[0]),
                int(hkl_key[1]),
                int(hkl_key[2]),
            )
        except Exception:
            pass

    branch_idx, _branch_source, _branch_reason = resolve_canonical_branch(
        entry,
        allow_legacy_peak_fallback=False,
    )
    if hkl_key is not None and branch_idx in {0, 1}:
        return (
            "peak_index",
            int(branch_idx),
            int(hkl_key[0]),
            int(hkl_key[1]),
            int(hkl_key[2]),
        )

    if hkl_key is None:
        return None
    try:
        sim_col = float(entry.get("sim_x", np.nan))
        sim_row = float(entry.get("sim_y", np.nan))
    except Exception:
        sim_col = float("nan")
        sim_row = float("nan")
    return (
        "hkl_coord",
        int(hkl_key[0]),
        int(hkl_key[1]),
        int(hkl_key[2]),
        round(sim_col, 1),
        round(sim_row, 1),
    )


def live_geometry_preview_match_hkl(
    entry: dict[str, object] | None,
) -> tuple[int, int, int] | None:
    """Return the normalized HKL tuple for one live preview match entry."""

    if not isinstance(entry, dict):
        return None
    return gui_geometry_overlay.normalize_hkl_key(entry.get("hkl", entry.get("label")))


def live_geometry_preview_match_is_excluded(
    preview_state,
    entry: dict[str, object] | None,
) -> bool:
    """Return whether one live preview match entry is excluded."""

    key = live_geometry_preview_match_key(entry)
    excluded_keys = getattr(preview_state, "excluded_keys", set())
    return key in excluded_keys if key is not None else False


def filter_live_geometry_preview_matches(
    preview_state,
    matched_pairs: Sequence[dict[str, object]] | None,
    *,
    existing_pairs: Sequence[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], int]:
    """Return live preview matches after applying user exclusions."""

    filtered: list[dict[str, object]] = []
    excluded_count = 0
    excluded_hkls: set[tuple[int, int, int]] = set()
    source_pairs = (
        existing_pairs
        if existing_pairs is not None
        else getattr(getattr(preview_state, "overlay", None), "pairs", [])
    )
    for entry in source_pairs or []:
        if not isinstance(entry, dict):
            continue
        if not live_geometry_preview_match_is_excluded(preview_state, entry):
            continue
        if live_geometry_preview_match_key(entry) is not None:
            continue
        hkl_key = live_geometry_preview_match_hkl(entry)
        if hkl_key is not None:
            excluded_hkls.add(hkl_key)

    excluded_keys = getattr(preview_state, "excluded_keys", set())
    for raw_entry in matched_pairs or []:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        key = live_geometry_preview_match_key(entry)
        excluded = key in excluded_keys if key is not None else False
        if not excluded:
            hkl_key = live_geometry_preview_match_hkl(entry)
            if hkl_key is not None and hkl_key in excluded_hkls:
                excluded = True
        if excluded:
            excluded_count += 1
            continue
        filtered.append(entry)
    return filtered, excluded_count


def apply_live_geometry_preview_match_exclusions(
    preview_state,
    matched_pairs: Sequence[dict[str, object]] | None,
    match_stats: dict[str, object] | None,
    *,
    existing_pairs: Sequence[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object], int]:
    """Apply live preview exclusions and refresh fit-facing summary metrics."""

    filtered_pairs, excluded_count = filter_live_geometry_preview_matches(
        preview_state,
        matched_pairs,
        existing_pairs=existing_pairs,
    )
    stats = dict(match_stats) if isinstance(match_stats, dict) else {}
    stats["excluded_count"] = int(excluded_count)
    stats["matched_count"] = int(len(filtered_pairs))
    stats["matched_after_exclusions"] = int(len(filtered_pairs))

    match_dists = np.asarray(
        [float(entry.get("distance_px", np.nan)) for entry in filtered_pairs],
        dtype=float,
    )
    match_dists = match_dists[np.isfinite(match_dists)]
    match_conf = np.asarray(
        [float(entry.get("confidence", np.nan)) for entry in filtered_pairs],
        dtype=float,
    )
    match_conf = match_conf[np.isfinite(match_conf)]

    stats["mean_match_distance_px"] = (
        float(np.mean(match_dists)) if match_dists.size else float("nan")
    )
    stats["p90_match_distance_px"] = (
        float(np.percentile(match_dists, 90.0)) if match_dists.size else float("nan")
    )
    stats["median_match_confidence"] = (
        float(np.median(match_conf)) if match_conf.size else float("nan")
    )
    return filtered_pairs, stats, int(excluded_count)


def _set_status_text(set_status_text: Callable[[str], None] | None, text: str) -> None:
    if callable(set_status_text):
        set_status_text(str(text))


def build_empty_live_geometry_preview_overlay_state(
    *,
    signature: object,
    min_matches: int,
    max_display_markers: int,
    q_group_total: int,
    q_group_excluded: int,
    excluded_q_peaks: int,
    collapsed_degenerate_peaks: int = 0,
) -> dict[str, object]:
    """Return one empty cached live-preview overlay-state payload."""

    return {
        "signature": signature,
        "pairs": [],
        "simulated_count": 0,
        "min_matches": int(min_matches),
        "best_radius": float("nan"),
        "mean_dist": float("nan"),
        "p90_dist": float("nan"),
        "quality_fail": False,
        "max_display_markers": int(max_display_markers),
        "auto_match_attempts": [],
        "q_group_total": int(q_group_total),
        "q_group_excluded": int(q_group_excluded),
        "excluded_q_peaks": int(excluded_q_peaks),
        "collapsed_degenerate_peaks": int(collapsed_degenerate_peaks),
    }


def build_live_geometry_preview_overlay_state(
    *,
    signature: object,
    matched_pairs: Sequence[Mapping[str, object]] | None,
    match_stats: Mapping[str, object] | None,
    preview_auto_match_cfg: Mapping[str, object] | None,
    auto_match_attempts: Sequence[Mapping[str, object]] | None,
    min_matches: int,
    q_group_total: int,
    q_group_excluded: int,
    excluded_q_peaks: int,
    collapsed_degenerate_peaks: int = 0,
) -> dict[str, object]:
    """Return one cached live-preview overlay-state payload from match results."""

    match_stats_local = match_stats if isinstance(match_stats, Mapping) else {}
    preview_cfg = preview_auto_match_cfg if isinstance(preview_auto_match_cfg, Mapping) else {}
    matched_pairs_local = [dict(entry) for entry in matched_pairs or ()]
    attempts_local = [dict(entry) for entry in auto_match_attempts or ()]

    simulated_count = _coerce_int(
        match_stats_local.get("simulated_count", len(matched_pairs_local)),
        len(matched_pairs_local),
    )
    best_radius = _coerce_float(
        match_stats_local.get("search_radius_px", np.nan),
        float("nan"),
    )
    p90_dist = _coerce_float(
        match_stats_local.get("p90_match_distance_px", np.nan),
        float("nan"),
    )
    mean_dist = _coerce_float(
        match_stats_local.get("mean_match_distance_px", np.nan),
        float("nan"),
    )
    max_auto_p90 = _coerce_float(
        preview_cfg.get("max_p90_distance_px", 35.0),
        35.0,
    )
    max_auto_mean = _coerce_float(
        preview_cfg.get("max_mean_distance_px", 22.0),
        22.0,
    )
    quality_fail = bool(
        (np.isfinite(max_auto_p90) and np.isfinite(p90_dist) and p90_dist > max_auto_p90)
        or (np.isfinite(max_auto_mean) and np.isfinite(mean_dist) and mean_dist > max_auto_mean)
    )

    return {
        "signature": signature,
        "pairs": matched_pairs_local,
        "simulated_count": int(simulated_count),
        "min_matches": int(min_matches),
        "best_radius": float(best_radius),
        "mean_dist": float(mean_dist),
        "p90_dist": float(p90_dist),
        "quality_fail": bool(quality_fail),
        "max_display_markers": _coerce_int(
            preview_cfg.get("max_display_markers", 120),
            120,
        ),
        "auto_match_attempts": attempts_local,
        "q_group_total": int(q_group_total),
        "q_group_excluded": int(q_group_excluded),
        "excluded_q_peaks": int(excluded_q_peaks),
        "collapsed_degenerate_peaks": int(collapsed_degenerate_peaks),
    }


def build_live_geometry_preview_status_text(
    preview_overlay_state: object,
    *,
    active_pair_count: int,
    excluded_count: int,
) -> str:
    """Build the status line shown after one live-preview redraw."""

    pairs = list(getattr(preview_overlay_state, "pairs", []) or [])
    simulated_count = _coerce_int(
        getattr(preview_overlay_state, "simulated_count", 0),
        0,
    )
    min_matches = _coerce_int(
        getattr(preview_overlay_state, "min_matches", 0),
        0,
    )
    best_radius = _coerce_float(
        getattr(preview_overlay_state, "best_radius", np.nan),
        float("nan"),
    )
    mean_dist = _coerce_float(
        getattr(preview_overlay_state, "mean_dist", np.nan),
        float("nan"),
    )
    p90_dist = _coerce_float(
        getattr(preview_overlay_state, "p90_dist", np.nan),
        float("nan"),
    )
    quality_fail = bool(getattr(preview_overlay_state, "quality_fail", False))
    q_group_total = _coerce_int(
        getattr(preview_overlay_state, "q_group_total", 0),
        0,
    )
    q_group_excluded = _coerce_int(
        getattr(preview_overlay_state, "q_group_excluded", 0),
        0,
    )
    collapsed_deg = _coerce_int(
        getattr(preview_overlay_state, "collapsed_degenerate_peaks", 0),
        0,
    )
    max_display_markers = max(
        1,
        _coerce_int(getattr(preview_overlay_state, "max_display_markers", 120), 120),
    )
    shown_count = min(len(pairs), max_display_markers)

    summary = (
        "Live auto-match preview: "
        f"{int(active_pair_count)}/{simulated_count} active peaks "
        f"(need {min_matches}, local-peak match"
    )
    if np.isfinite(best_radius):
        summary += f", limit={best_radius:.1f}px"
    if np.isfinite(mean_dist):
        summary += f", mean={mean_dist:.1f}px"
    if np.isfinite(p90_dist):
        summary += f", p90={p90_dist:.1f}px"
    summary += ")."
    if int(excluded_count) > 0:
        summary += f" Excluded={int(excluded_count)}."
    if q_group_total > 0:
        summary += f" Qr/Qz groups on={max(0, q_group_total - q_group_excluded)}/{q_group_total}."
    if collapsed_deg > 0:
        summary += f" Degenerate collapsed={collapsed_deg}."
    if int(active_pair_count) < min_matches:
        summary += " Geometry fit would stop on the minimum-match gate."
    elif quality_fail:
        summary += " Geometry fit would stop on the quality gate."
    else:
        summary += " Geometry fit gates pass."
    if shown_count < len(pairs):
        summary += f" Showing {shown_count}/{len(pairs)} overlays."
    return summary


def render_live_geometry_preview_overlay_state(
    *,
    preview_state,
    draw_live_geometry_preview_overlay: Callable[..., None],
    filter_live_preview_matches: Callable[
        [Sequence[dict[str, object]]], tuple[Sequence[dict[str, object]], int]
    ],
    set_status_text: Callable[[str], None] | None = None,
    update_status: bool = True,
) -> bool:
    """Redraw the cached live-preview overlay and optionally refresh its status."""

    preview_overlay_state = getattr(preview_state, "overlay", None)
    pairs = list(getattr(preview_overlay_state, "pairs", []) or [])
    max_display_markers = _coerce_int(
        getattr(preview_overlay_state, "max_display_markers", 120),
        120,
    )
    draw_live_geometry_preview_overlay(
        pairs,
        max_display_markers=max_display_markers,
    )
    if not update_status:
        return bool(pairs)

    active_pairs, excluded_count = filter_live_preview_matches(pairs)
    _set_status_text(
        set_status_text,
        build_live_geometry_preview_status_text(
            preview_overlay_state,
            active_pair_count=len(list(active_pairs)),
            excluded_count=int(excluded_count),
        ),
    )
    return bool(list(active_pairs))


def runtime_live_geometry_preview_enabled(
    bindings: GeometryQGroupRuntimeBindings,
) -> bool:
    """Return whether the runtime live-preview checkbox is currently enabled."""

    try:
        return bool(bindings.live_geometry_preview_enabled())
    except Exception:
        return False


def draw_runtime_live_geometry_preview_overlay(
    bindings: GeometryQGroupRuntimeBindings,
    matched_pairs: Sequence[dict[str, object]] | None,
    *,
    max_display_markers: int = 120,
) -> None:
    """Draw the runtime live-preview overlay using the bound axis/artists."""

    if bindings.axis is None:
        return
    clear_geometry_preview_artists = bindings.clear_geometry_preview_artists or (lambda: None)
    draw_idle = bindings.draw_idle or (lambda: None)
    normalize_hkl_key = bindings.normalize_hkl_key or (lambda _value: None)
    live_preview_match_is_excluded = bindings.live_preview_match_is_excluded or (
        lambda _entry: False
    )
    try:
        show_pair_connectors = bool(
            bindings.caked_view_enabled() if callable(bindings.caked_view_enabled) else False
        )
    except Exception:
        show_pair_connectors = False
    gui_overlays.draw_live_geometry_preview_overlay(
        bindings.axis,
        matched_pairs,
        geometry_preview_artists=(
            bindings.geometry_preview_artists
            if bindings.geometry_preview_artists is not None
            else []
        ),
        clear_geometry_preview_artists=clear_geometry_preview_artists,
        draw_idle=draw_idle,
        normalize_hkl_key=normalize_hkl_key,
        live_preview_match_is_excluded=live_preview_match_is_excluded,
        max_display_markers=max_display_markers,
        show_pair_connectors=show_pair_connectors,
    )


def render_runtime_live_geometry_preview_state(
    bindings: GeometryQGroupRuntimeBindings,
    *,
    update_status: bool = True,
) -> bool:
    """Redraw the cached runtime live-preview overlay from bound state."""

    filter_live_preview_matches = bindings.filter_live_preview_matches or (
        lambda pairs: (list(pairs or []), 0)
    )
    return render_live_geometry_preview_overlay_state(
        preview_state=bindings.preview_state,
        draw_live_geometry_preview_overlay=lambda pairs, *, max_display_markers: (
            draw_runtime_live_geometry_preview_overlay(
                bindings,
                pairs,
                max_display_markers=max_display_markers,
            )
        ),
        filter_live_preview_matches=filter_live_preview_matches,
        set_status_text=bindings.set_status_text,
        update_status=update_status,
    )


def resolve_runtime_live_geometry_preview_simulated_peaks(
    bindings: GeometryQGroupRuntimeBindings,
    *,
    update_status: bool = True,
) -> list[dict[str, object]] | None:
    """Return runtime live-preview peaks, preferring fresh simulation when possible."""

    simulate_preview_style_peaks = bindings.simulate_preview_style_peaks
    if callable(simulate_preview_style_peaks):
        try:
            miller_array = np.asarray(bindings.miller, dtype=float)
            intensity_array = np.asarray(bindings.intensities, dtype=float).reshape(-1)
            image_size_value = int(_resolve_runtime_value(bindings.image_size))
            params_local = (
                dict(bindings.current_geometry_fit_params_factory() or {})
                if callable(bindings.current_geometry_fit_params_factory)
                else {}
            )
        except Exception:
            miller_array = np.empty((0, 3), dtype=float)
            intensity_array = np.empty((0,), dtype=float)
            image_size_value = 0
            params_local = {}
        if (
            miller_array.ndim == 2
            and miller_array.shape[0] > 0
            and intensity_array.size >= miller_array.shape[0]
            and image_size_value > 0
        ):
            try:
                simulated_peaks = list(
                    simulate_preview_style_peaks(
                        miller_array,
                        intensity_array,
                        int(image_size_value),
                        dict(params_local),
                    )
                    or []
                )
            except Exception as exc:
                if callable(bindings.clear_geometry_preview_artists):
                    bindings.clear_geometry_preview_artists()
                if update_status:
                    _set_status_text(
                        bindings.set_status_text,
                        (f"Live auto-match preview unavailable: failed to simulate peaks ({exc})."),
                    )
                return None
            if simulated_peaks:
                return simulated_peaks

    build_cached_peaks = bindings.build_live_preview_simulated_peaks_from_cache
    simulated_peaks = list(build_cached_peaks() or []) if callable(build_cached_peaks) else []
    if simulated_peaks:
        return simulated_peaks

    if callable(bindings.clear_geometry_preview_artists):
        bindings.clear_geometry_preview_artists()
    if update_status:
        _set_status_text(
            bindings.set_status_text,
            "Live auto-match preview unavailable: no simulated peaks are available.",
        )
    return None


def resolve_runtime_live_geometry_preview_background(
    bindings: GeometryQGroupRuntimeBindings,
    *,
    update_status: bool = True,
) -> object | None:
    """Return the display background when live preview is available."""

    if not runtime_live_geometry_preview_enabled(bindings):
        if callable(bindings.clear_geometry_preview_artists):
            bindings.clear_geometry_preview_artists()
        return None

    try:
        caked_view_enabled = bool(
            bindings.caked_view_enabled() if callable(bindings.caked_view_enabled) else False
        )
    except Exception:
        caked_view_enabled = False
    if caked_view_enabled:
        if callable(bindings.clear_geometry_preview_artists):
            bindings.clear_geometry_preview_artists()
        if update_status:
            _set_status_text(
                bindings.set_status_text,
                "Live auto-match preview unavailable in 2D caked view.",
            )
        return None

    display_background = (
        bindings.current_background_display_factory()
        if callable(bindings.current_background_display_factory)
        else None
    )
    if not bool(_resolve_runtime_value(bindings.background_visible)) or display_background is None:
        if callable(bindings.clear_geometry_preview_artists):
            bindings.clear_geometry_preview_artists()
        if update_status:
            _set_status_text(
                bindings.set_status_text,
                "Live auto-match preview unavailable: background image is hidden.",
            )
        return None
    return display_background


def resolve_runtime_live_geometry_preview_seed_state(
    bindings: GeometryQGroupRuntimeBindings,
    simulated_peaks: Sequence[dict[str, object]] | None,
    *,
    preview_auto_match_cfg: Mapping[str, object] | None,
    min_matches: int,
    signature: object,
    update_status: bool = True,
) -> tuple[list[dict[str, object]], int, int, int] | None:
    """Filter/collapse runtime live-preview seeds and handle empty-state exits."""

    filter_simulated_peaks = bindings.filter_simulated_peaks
    if callable(filter_simulated_peaks):
        filtered_peaks, excluded_q_peaks, q_group_total = filter_simulated_peaks(simulated_peaks)
    else:
        filtered_peaks = list(simulated_peaks or [])
        excluded_q_peaks = 0
        q_group_total = 0

    if not filtered_peaks:
        if callable(bindings.clear_geometry_preview_artists):
            bindings.clear_geometry_preview_artists()
        if update_status:
            _set_status_text(
                bindings.set_status_text,
                "Live auto-match preview unavailable: no Qr/Qz groups are selected.",
            )
        excluded_q_group_count = (
            _coerce_int(bindings.excluded_q_group_count(), 0)
            if callable(bindings.excluded_q_group_count)
            else 0
        )
        gui_controllers.replace_geometry_preview_overlay_state(
            bindings.preview_state,
            build_empty_live_geometry_preview_overlay_state(
                signature=signature,
                min_matches=int(min_matches),
                max_display_markers=_coerce_int(
                    (
                        preview_auto_match_cfg.get("max_display_markers", 120)
                        if isinstance(preview_auto_match_cfg, Mapping)
                        else 120
                    ),
                    120,
                ),
                q_group_total=int(q_group_total),
                q_group_excluded=int(excluded_q_group_count),
                excluded_q_peaks=int(excluded_q_peaks),
            ),
        )
        return None

    preview_cfg = preview_auto_match_cfg if isinstance(preview_auto_match_cfg, Mapping) else {}
    search_radius = _coerce_float(preview_cfg.get("search_radius_px", 24.0), 24.0)
    merge_radius_px = _coerce_float(
        preview_cfg.get(
            "degenerate_merge_radius_px",
            min(6.0, 0.33 * float(search_radius)),
        ),
        min(6.0, 0.33 * float(search_radius)),
    )
    collapse_simulated_peaks = bindings.collapse_simulated_peaks
    if callable(collapse_simulated_peaks):
        collapsed_peaks, collapsed_deg_preview = collapse_simulated_peaks(
            filtered_peaks,
            merge_radius_px=float(merge_radius_px),
        )
    else:
        collapsed_peaks = list(filtered_peaks)
        collapsed_deg_preview = 0

    if not collapsed_peaks:
        if callable(bindings.clear_geometry_preview_artists):
            bindings.clear_geometry_preview_artists()
        if update_status:
            _set_status_text(
                bindings.set_status_text,
                (
                    "Live auto-match preview unavailable: no geometry-fit seeds "
                    "remain after Qr/Qz collapse."
                ),
            )
        return None

    return (
        list(collapsed_peaks),
        int(excluded_q_peaks),
        int(q_group_total),
        int(collapsed_deg_preview),
    )


def apply_runtime_live_geometry_preview_match_results(
    bindings: GeometryQGroupRuntimeBindings,
    *,
    signature: object,
    matched_pairs: Sequence[Mapping[str, object]] | None,
    match_stats: Mapping[str, object] | None,
    preview_auto_match_cfg: Mapping[str, object] | None,
    auto_match_attempts: Sequence[Mapping[str, object]] | None,
    min_matches: int,
    q_group_total: int,
    excluded_q_peaks: int,
    collapsed_deg_preview: int = 0,
    update_status: bool = True,
) -> bool:
    """Store runtime live-preview match results and redraw the cached overlay."""

    q_group_excluded = (
        _coerce_int(bindings.excluded_q_group_count(), 0)
        if callable(bindings.excluded_q_group_count)
        else 0
    )
    gui_controllers.replace_geometry_preview_overlay_state(
        bindings.preview_state,
        build_live_geometry_preview_overlay_state(
            signature=signature,
            matched_pairs=matched_pairs,
            match_stats=match_stats,
            preview_auto_match_cfg=preview_auto_match_cfg,
            auto_match_attempts=auto_match_attempts,
            min_matches=int(min_matches),
            q_group_total=int(q_group_total),
            q_group_excluded=int(q_group_excluded),
            excluded_q_peaks=int(excluded_q_peaks),
            collapsed_degenerate_peaks=int(collapsed_deg_preview),
        ),
    )
    return render_runtime_live_geometry_preview_state(
        bindings,
        update_status=update_status,
    )


def distance_point_to_segment_sq(
    px: float,
    py: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    """Return squared distance from one display-space point to one segment."""

    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
        return (float(px) - float(x0)) ** 2 + (float(py) - float(y0)) ** 2

    t = ((float(px) - float(x0)) * dx + (float(py) - float(y0)) * dy) / (dx * dx + dy * dy)
    t = min(1.0, max(0.0, float(t)))
    cx = float(x0) + t * dx
    cy = float(y0) + t * dy
    return (float(px) - cx) ** 2 + (float(py) - cy) ** 2


def clear_live_geometry_preview_exclusions_with_side_effects(
    *,
    preview_state,
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    update_geometry_preview_exclude_button_label: Callable[[], None],
    refresh_geometry_q_group_window: Callable[[], None],
    live_geometry_preview_enabled: Callable[[], bool],
    refresh_live_geometry_preview: Callable[[], None],
    set_status_text: Callable[[str], None] | None = None,
) -> None:
    """Clear preview exclusions and apply the dependent runtime side effects."""

    gui_controllers.clear_geometry_preview_excluded_keys(preview_state)
    gui_controllers.clear_geometry_preview_excluded_q_groups(preview_state)
    invalidate_geometry_manual_pick_cache()
    update_geometry_preview_exclude_button_label()
    refresh_geometry_q_group_window()
    if live_geometry_preview_enabled():
        refresh_live_geometry_preview()
    else:
        _set_status_text(
            set_status_text,
            "Reset all Qr/Qz geometry-fit selections.",
        )


def toggle_live_geometry_preview_exclusion_at(
    *,
    preview_state,
    col: float,
    row: float,
    live_preview_match_key: Callable[[dict[str, object] | None], tuple[object, ...] | None],
    live_preview_match_hkl: Callable[[dict[str, object] | None], tuple[int, int, int] | None],
    render_live_geometry_preview_state: Callable[[], None],
    max_distance_px: float,
    set_status_text: Callable[[str], None] | None = None,
) -> bool:
    """Toggle the nearest live-preview pair in or out of geometry fitting."""

    preview_overlay_state = getattr(preview_state, "overlay", None)
    pairs = list(getattr(preview_overlay_state, "pairs", []) or [])
    if not pairs:
        _set_status_text(
            set_status_text,
            "No live preview pairs are available to exclude.",
        )
        return False

    best_entry: dict[str, object] | None = None
    best_d2 = float("inf")
    for raw_entry in pairs:
        if not isinstance(raw_entry, dict):
            continue
        try:
            sim_col = float(raw_entry["sim_x"])
            sim_row = float(raw_entry["sim_y"])
            bg_col = float(raw_entry["x"])
            bg_row = float(raw_entry["y"])
        except Exception:
            continue
        d2 = min(
            (float(col) - sim_col) ** 2 + (float(row) - sim_row) ** 2,
            (float(col) - bg_col) ** 2 + (float(row) - bg_row) ** 2,
            distance_point_to_segment_sq(
                float(col),
                float(row),
                sim_col,
                sim_row,
                bg_col,
                bg_row,
            ),
        )
        if d2 < best_d2:
            best_d2 = d2
            best_entry = raw_entry

    if best_entry is None or best_d2 > float(max_distance_px) ** 2:
        _set_status_text(
            set_status_text,
            f"No preview pair within {float(max_distance_px):.0f}px to toggle.",
        )
        return False

    key = live_preview_match_key(best_entry)
    hkl_key = live_preview_match_hkl(best_entry)
    if key is None or hkl_key is None:
        _set_status_text(
            set_status_text,
            "The selected preview pair cannot be excluded.",
        )
        return False

    excluded_keys = getattr(preview_state, "excluded_keys", set())
    if key in excluded_keys:
        gui_controllers.set_geometry_preview_match_included(
            preview_state,
            key,
            included=True,
        )
        action = "Included"
    else:
        gui_controllers.set_geometry_preview_match_included(
            preview_state,
            key,
            included=False,
        )
        action = "Excluded"

    render_live_geometry_preview_state()
    _set_status_text(
        set_status_text,
        f"{action} live preview peak HKL={hkl_key} from geometry fit.",
    )
    return True


def make_runtime_geometry_q_group_bindings_factory(
    *,
    view_state,
    preview_state,
    q_group_state,
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names_factory: object,
    build_entries_snapshot: Callable[[], Sequence[dict[str, object]] | None] | None = None,
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    update_geometry_preview_exclude_button_label: Callable[[], None],
    live_geometry_preview_enabled: Callable[[], bool],
    refresh_live_geometry_preview: Callable[[], None],
    set_hkl_pick_mode: Callable[[bool], None] | None = None,
    live_preview_match_key: (
        Callable[[dict[str, object] | None], tuple[object, ...] | None] | None
    ) = None,
    live_preview_match_hkl: (
        Callable[[dict[str, object] | None], tuple[int, int, int] | None] | None
    ) = None,
    render_live_geometry_preview_state: Callable[[], object] | None = None,
    clear_geometry_preview_artists: Callable[[], None] | None = None,
    preview_toggle_max_distance_px: float = 20.0,
    update_running_factory: object | None = None,
    has_cached_hit_tables_factory: object | None = None,
    build_live_preview_simulated_peaks_from_cache: Callable[[], list[dict[str, object]]]
    | None = None,
    simulate_preview_style_peaks: Callable[..., list[dict[str, object]]] | None = None,
    miller_factory: object | None = None,
    intensities_factory: object | None = None,
    image_size_value_factory: object | None = None,
    current_geometry_fit_params_factory: Callable[[], Mapping[str, object]] | None = None,
    filter_simulated_peaks: (
        Callable[
            [Sequence[dict[str, object]] | None],
            tuple[list[dict[str, object]], int, int],
        ]
        | None
    ) = None,
    collapse_simulated_peaks: Callable[..., tuple[list[dict[str, object]], int]] | None = None,
    excluded_q_group_count: Callable[[], int] | None = None,
    caked_view_enabled: Callable[[], bool] | None = None,
    background_visible_factory: object | None = None,
    current_background_display_factory: Callable[[], object] | None = None,
    axis: object | None = None,
    geometry_preview_artists: list[object] | None = None,
    draw_idle_factory: object | None = None,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] | None = None,
    live_preview_match_is_excluded: (Callable[[dict[str, object] | None], bool] | None) = None,
    filter_live_preview_matches: (
        Callable[[Sequence[dict[str, object]] | None], tuple[list[dict[str, object]], int]] | None
    ) = None,
    refresh_live_geometry_preview_quiet: Callable[[], None] | None = None,
    clear_last_simulation_signature: Callable[[], None] | None = None,
    schedule_update_factory: object | None = None,
    set_status_text_factory: object | None = None,
    file_dialog_dir_factory: object | None = None,
    asksaveasfilename: Callable[..., object] | None = None,
    askopenfilename: Callable[..., object] | None = None,
) -> Callable[[], GeometryQGroupRuntimeBindings]:
    """Return a zero-arg factory for live geometry Q-group runtime bindings."""

    def _build_bindings() -> GeometryQGroupRuntimeBindings:
        return GeometryQGroupRuntimeBindings(
            view_state=view_state,
            preview_state=preview_state,
            q_group_state=q_group_state,
            fit_config=fit_config,
            current_geometry_fit_var_names_factory=current_geometry_fit_var_names_factory,
            build_entries_snapshot=build_entries_snapshot,
            invalidate_geometry_manual_pick_cache=invalidate_geometry_manual_pick_cache,
            update_geometry_preview_exclude_button_label=update_geometry_preview_exclude_button_label,
            live_geometry_preview_enabled=live_geometry_preview_enabled,
            refresh_live_geometry_preview=refresh_live_geometry_preview,
            set_hkl_pick_mode=set_hkl_pick_mode,
            live_preview_match_key=live_preview_match_key,
            live_preview_match_hkl=live_preview_match_hkl,
            render_live_geometry_preview_state=render_live_geometry_preview_state,
            clear_geometry_preview_artists=clear_geometry_preview_artists,
            preview_toggle_max_distance_px=_coerce_float(
                preview_toggle_max_distance_px,
                20.0,
            ),
            update_running=_resolve_runtime_value(update_running_factory),
            has_cached_hit_tables=_resolve_runtime_value(has_cached_hit_tables_factory),
            build_live_preview_simulated_peaks_from_cache=(
                build_live_preview_simulated_peaks_from_cache
            ),
            simulate_preview_style_peaks=simulate_preview_style_peaks,
            miller=_resolve_runtime_value(miller_factory),
            intensities=_resolve_runtime_value(intensities_factory),
            image_size=_resolve_runtime_value(image_size_value_factory),
            current_geometry_fit_params_factory=current_geometry_fit_params_factory,
            filter_simulated_peaks=filter_simulated_peaks,
            collapse_simulated_peaks=collapse_simulated_peaks,
            excluded_q_group_count=excluded_q_group_count,
            caked_view_enabled=caked_view_enabled,
            background_visible=_resolve_runtime_value(background_visible_factory),
            current_background_display_factory=current_background_display_factory,
            axis=axis,
            geometry_preview_artists=geometry_preview_artists,
            draw_idle=_resolve_runtime_value(draw_idle_factory),
            normalize_hkl_key=normalize_hkl_key,
            live_preview_match_is_excluded=live_preview_match_is_excluded,
            filter_live_preview_matches=filter_live_preview_matches,
            refresh_live_geometry_preview_quiet=refresh_live_geometry_preview_quiet,
            clear_last_simulation_signature=clear_last_simulation_signature,
            schedule_update=_resolve_runtime_value(schedule_update_factory),
            set_status_text=_resolve_runtime_value(set_status_text_factory),
            file_dialog_dir=_resolve_runtime_value(file_dialog_dir_factory),
            asksaveasfilename=asksaveasfilename,
            askopenfilename=askopenfilename,
        )

    return _build_bindings


def update_runtime_geometry_q_group_window_status(
    bindings: GeometryQGroupRuntimeBindings,
    entries: Sequence[dict[str, object]] | None = None,
) -> None:
    """Refresh the runtime selector status text from live bindings."""

    update_geometry_q_group_window_status(
        view_state=bindings.view_state,
        preview_state=bindings.preview_state,
        q_group_state=bindings.q_group_state,
        fit_config=bindings.fit_config,
        current_geometry_fit_var_names=_runtime_geometry_fit_var_names(bindings),
        entries=entries,
    )


def on_runtime_geometry_q_group_checkbox_changed(
    bindings: GeometryQGroupRuntimeBindings,
    group_key: tuple[object, ...] | None,
    row_var: object,
) -> bool:
    """Apply one runtime Qr/Qz selector checkbox toggle."""

    return apply_geometry_q_group_checkbox_change_with_side_effects(
        preview_state=bindings.preview_state,
        group_key=group_key,
        row_var=row_var,
        invalidate_geometry_manual_pick_cache=bindings.invalidate_geometry_manual_pick_cache,
        update_geometry_preview_exclude_button_label=bindings.update_geometry_preview_exclude_button_label,
        update_geometry_q_group_window_status=lambda: update_runtime_geometry_q_group_window_status(
            bindings
        ),
        live_geometry_preview_enabled=bindings.live_geometry_preview_enabled,
        refresh_live_geometry_preview=bindings.refresh_live_geometry_preview,
        set_status_text=bindings.set_status_text,
    )


def refresh_runtime_geometry_q_group_window(
    bindings: GeometryQGroupRuntimeBindings,
) -> bool:
    """Redraw the runtime Qr/Qz selector window from live bindings."""

    return refresh_geometry_q_group_window(
        view_state=bindings.view_state,
        preview_state=bindings.preview_state,
        q_group_state=bindings.q_group_state,
        fit_config=bindings.fit_config,
        current_geometry_fit_var_names=_runtime_geometry_fit_var_names(bindings),
        on_toggle=lambda group_key, row_var: on_runtime_geometry_q_group_checkbox_changed(
            bindings,
            group_key,
            row_var,
        ),
    )


def set_all_geometry_q_groups_enabled_runtime(
    bindings: GeometryQGroupRuntimeBindings,
    *,
    enabled: bool,
) -> bool:
    """Apply one runtime bulk include/exclude selector action."""

    return set_all_geometry_q_groups_enabled_with_side_effects(
        preview_state=bindings.preview_state,
        q_group_state=bindings.q_group_state,
        enabled=enabled,
        invalidate_geometry_manual_pick_cache=bindings.invalidate_geometry_manual_pick_cache,
        update_geometry_preview_exclude_button_label=bindings.update_geometry_preview_exclude_button_label,
        refresh_geometry_q_group_window=lambda: refresh_runtime_geometry_q_group_window(bindings),
        live_geometry_preview_enabled=bindings.live_geometry_preview_enabled,
        refresh_live_geometry_preview=bindings.refresh_live_geometry_preview,
        set_status_text=bindings.set_status_text,
    )


def request_runtime_geometry_q_group_window_update(
    bindings: GeometryQGroupRuntimeBindings,
) -> None:
    """Request one runtime refresh of the listed Qr/Qz peaks."""

    request_geometry_q_group_window_update_with_side_effects(
        q_group_state=bindings.q_group_state,
        clear_last_simulation_signature=(
            bindings.clear_last_simulation_signature or (lambda: None)
        ),
        invalidate_geometry_manual_pick_cache=bindings.invalidate_geometry_manual_pick_cache,
        set_status_text=bindings.set_status_text,
        schedule_update=bindings.schedule_update or (lambda: None),
    )


def capture_runtime_geometry_q_group_entries_snapshot(
    bindings: GeometryQGroupRuntimeBindings,
) -> list[dict[str, object]]:
    """Rebuild and store the runtime Qr/Qz selector snapshot from live data."""

    build_entries_snapshot = bindings.build_entries_snapshot
    if not callable(build_entries_snapshot):
        return gui_controllers.listed_geometry_q_group_entries(bindings.q_group_state)

    entries = replace_geometry_q_group_entries_snapshot_with_side_effects(
        preview_state=bindings.preview_state,
        q_group_state=bindings.q_group_state,
        entries=build_entries_snapshot(),
        invalidate_geometry_manual_pick_cache=bindings.invalidate_geometry_manual_pick_cache,
        update_geometry_preview_exclude_button_label=bindings.update_geometry_preview_exclude_button_label,
    )
    if gui_views.geometry_q_group_window_open(bindings.view_state):
        refresh_runtime_geometry_q_group_window(bindings)
    return entries


def save_geometry_q_group_selection_runtime(
    bindings: GeometryQGroupRuntimeBindings,
) -> bool:
    """Export the runtime Qr/Qz selector state through the configured dialog."""

    if not callable(bindings.asksaveasfilename):
        _set_status_text(bindings.set_status_text, "Save Qr/Qz peak list unavailable.")
        return False
    return save_geometry_q_group_selection_with_dialog(
        preview_state=bindings.preview_state,
        q_group_state=bindings.q_group_state,
        file_dialog_dir=bindings.file_dialog_dir,
        asksaveasfilename=bindings.asksaveasfilename,
        set_status_text=bindings.set_status_text,
    )


def load_geometry_q_group_selection_runtime(
    bindings: GeometryQGroupRuntimeBindings,
) -> bool:
    """Import the runtime Qr/Qz selector state through the configured dialog."""

    if not callable(bindings.askopenfilename):
        _set_status_text(bindings.set_status_text, "Load Qr/Qz peak list unavailable.")
        return False
    return load_geometry_q_group_selection_with_dialog(
        preview_state=bindings.preview_state,
        q_group_state=bindings.q_group_state,
        file_dialog_dir=bindings.file_dialog_dir,
        askopenfilename=bindings.askopenfilename,
        update_geometry_preview_exclude_button_label=bindings.update_geometry_preview_exclude_button_label,
        refresh_geometry_q_group_window=lambda: refresh_runtime_geometry_q_group_window(bindings),
        live_geometry_preview_enabled=bindings.live_geometry_preview_enabled,
        refresh_live_geometry_preview=(
            bindings.refresh_live_geometry_preview_quiet or bindings.refresh_live_geometry_preview
        ),
        set_status_text=bindings.set_status_text,
    )


def close_runtime_geometry_q_group_window(
    bindings: GeometryQGroupRuntimeBindings,
) -> None:
    """Close the runtime Qr/Qz selector window using the live bindings."""

    close_geometry_q_group_window(bindings.view_state, bindings.q_group_state)


def open_runtime_geometry_q_group_window(
    *,
    root,
    bindings_factory: Callable[[], GeometryQGroupRuntimeBindings],
) -> bool:
    """Open the runtime Qr/Qz selector window and wire live callbacks."""

    bindings = bindings_factory()
    return open_geometry_q_group_window(
        root=root,
        view_state=bindings.view_state,
        preview_state=bindings.preview_state,
        q_group_state=bindings.q_group_state,
        fit_config=bindings.fit_config,
        current_geometry_fit_var_names=_runtime_geometry_fit_var_names(bindings),
        on_toggle=lambda group_key, row_var: on_runtime_geometry_q_group_checkbox_changed(
            bindings_factory(),
            group_key,
            row_var,
        ),
        on_include_all=lambda: set_all_geometry_q_groups_enabled_runtime(
            bindings_factory(),
            enabled=True,
        ),
        on_exclude_all=lambda: set_all_geometry_q_groups_enabled_runtime(
            bindings_factory(),
            enabled=False,
        ),
        on_update_listed_peaks=lambda: request_runtime_geometry_q_group_window_update(
            bindings_factory()
        ),
        on_save=lambda: save_geometry_q_group_selection_runtime(bindings_factory()),
        on_load=lambda: load_geometry_q_group_selection_runtime(bindings_factory()),
        on_close=lambda: close_runtime_geometry_q_group_window(bindings_factory()),
    )


def open_runtime_geometry_q_group_preview_exclusion_window(
    *,
    root,
    bindings_factory: Callable[[], GeometryQGroupRuntimeBindings],
) -> bool:
    """Open the runtime selector in preview-exclusion mode and report status."""

    opened = open_runtime_geometry_q_group_window(
        root=root,
        bindings_factory=bindings_factory,
    )
    _set_status_text(
        bindings_factory().set_status_text,
        (
            "Opened the Qr/Qz group selector. "
            "Unchecked rows are skipped during manual picking and geometry fitting."
        ),
    )
    return opened


def set_runtime_geometry_preview_exclude_mode(
    bindings: GeometryQGroupRuntimeBindings,
    enabled: bool,
    *,
    message: str | None = None,
) -> bool:
    """Apply one runtime preview-exclude mode toggle from live bindings."""

    changed = gui_controllers.set_geometry_preview_exclude_mode(
        bindings.preview_state,
        enabled,
    )
    if changed and callable(bindings.set_hkl_pick_mode):
        bindings.set_hkl_pick_mode(False)
    bindings.update_geometry_preview_exclude_button_label()
    if message:
        _set_status_text(bindings.set_status_text, message)
    return bool(changed)


def clear_runtime_live_geometry_preview_exclusions(
    bindings: GeometryQGroupRuntimeBindings,
) -> None:
    """Clear runtime live-preview exclusions through the bound workflow surface."""

    clear_live_geometry_preview_exclusions_with_side_effects(
        preview_state=bindings.preview_state,
        invalidate_geometry_manual_pick_cache=bindings.invalidate_geometry_manual_pick_cache,
        update_geometry_preview_exclude_button_label=bindings.update_geometry_preview_exclude_button_label,
        refresh_geometry_q_group_window=lambda: refresh_runtime_geometry_q_group_window(bindings),
        live_geometry_preview_enabled=bindings.live_geometry_preview_enabled,
        refresh_live_geometry_preview=bindings.refresh_live_geometry_preview,
        set_status_text=bindings.set_status_text,
    )


def toggle_runtime_live_geometry_preview_exclusion_at(
    bindings: GeometryQGroupRuntimeBindings,
    col: float,
    row: float,
) -> bool:
    """Toggle one runtime preview exclusion using the live binding surface."""

    if not (
        callable(bindings.live_preview_match_key)
        and callable(bindings.live_preview_match_hkl)
        and callable(bindings.render_live_geometry_preview_state)
    ):
        _set_status_text(
            bindings.set_status_text,
            "Live preview exclusion toggle unavailable.",
        )
        return False

    return toggle_live_geometry_preview_exclusion_at(
        preview_state=bindings.preview_state,
        col=col,
        row=row,
        live_preview_match_key=bindings.live_preview_match_key,
        live_preview_match_hkl=bindings.live_preview_match_hkl,
        render_live_geometry_preview_state=bindings.render_live_geometry_preview_state,
        max_distance_px=_coerce_float(bindings.preview_toggle_max_distance_px, 20.0),
        set_status_text=bindings.set_status_text,
    )


def toggle_live_geometry_preview_with_side_effects(
    *,
    enabled: bool,
    disable_preview_exclude_mode: Callable[[], None],
    clear_geometry_preview_artists: Callable[[], None],
    open_geometry_q_group_window: Callable[[], object],
    update_running: bool,
    has_cached_hit_tables: bool,
    schedule_update: Callable[[], None],
    refresh_live_geometry_preview: Callable[[], bool],
    set_status_text: Callable[[str], None] | None = None,
) -> bool:
    """Apply the live-preview checkbox action and its follow-on workflow."""

    if not enabled:
        disable_preview_exclude_mode()
        clear_geometry_preview_artists()
        _set_status_text(
            set_status_text,
            "Live auto-match preview disabled.",
        )
        return False

    open_geometry_q_group_window()
    if bool(update_running) or not bool(has_cached_hit_tables):
        schedule_update()
        return True

    refreshed = bool(refresh_live_geometry_preview())
    if not refreshed:
        schedule_update()
    return refreshed


def toggle_runtime_live_geometry_preview(
    bindings: GeometryQGroupRuntimeBindings,
    *,
    root,
    bindings_factory: Callable[[], GeometryQGroupRuntimeBindings],
) -> bool:
    """Apply one runtime live-preview checkbox action through live bindings."""

    return toggle_live_geometry_preview_with_side_effects(
        enabled=bool(_resolve_runtime_value(bindings.live_geometry_preview_enabled)),
        disable_preview_exclude_mode=lambda: set_runtime_geometry_preview_exclude_mode(
            bindings,
            False,
        ),
        clear_geometry_preview_artists=(bindings.clear_geometry_preview_artists or (lambda: None)),
        open_geometry_q_group_window=lambda: open_runtime_geometry_q_group_window(
            root=root,
            bindings_factory=bindings_factory,
        ),
        update_running=bool(_resolve_runtime_value(bindings.update_running)),
        has_cached_hit_tables=bool(_resolve_runtime_value(bindings.has_cached_hit_tables)),
        schedule_update=bindings.schedule_update or (lambda: None),
        refresh_live_geometry_preview=bindings.refresh_live_geometry_preview,
        set_status_text=bindings.set_status_text,
    )


def make_runtime_geometry_q_group_callbacks(
    *,
    root,
    bindings_factory: Callable[[], GeometryQGroupRuntimeBindings],
) -> GeometryQGroupRuntimeCallbacks:
    """Return bound zero-arg callbacks for the runtime Qr/Qz selector workflow."""

    def _set_preview_exclude_mode(
        enabled: bool,
        message: str | None = None,
    ) -> bool:
        return set_runtime_geometry_preview_exclude_mode(
            bindings_factory(),
            enabled,
            message=message,
        )

    return GeometryQGroupRuntimeCallbacks(
        update_window_status=lambda entries=None: update_runtime_geometry_q_group_window_status(
            bindings_factory(),
            entries=entries,
        ),
        refresh_window=lambda: refresh_runtime_geometry_q_group_window(bindings_factory()),
        on_toggle=lambda group_key, row_var: on_runtime_geometry_q_group_checkbox_changed(
            bindings_factory(),
            group_key,
            row_var,
        ),
        include_all=lambda: set_all_geometry_q_groups_enabled_runtime(
            bindings_factory(),
            enabled=True,
        ),
        exclude_all=lambda: set_all_geometry_q_groups_enabled_runtime(
            bindings_factory(),
            enabled=False,
        ),
        update_listed_peaks=lambda: request_runtime_geometry_q_group_window_update(
            bindings_factory()
        ),
        save_selection=lambda: save_geometry_q_group_selection_runtime(bindings_factory()),
        load_selection=lambda: load_geometry_q_group_selection_runtime(bindings_factory()),
        close_window=lambda: close_runtime_geometry_q_group_window(bindings_factory()),
        open_window=lambda: open_runtime_geometry_q_group_window(
            root=root,
            bindings_factory=bindings_factory,
        ),
        open_preview_exclusion_window=lambda: (
            open_runtime_geometry_q_group_preview_exclusion_window(
                root=root,
                bindings_factory=bindings_factory,
            )
        ),
        set_preview_exclude_mode=_set_preview_exclude_mode,
        clear_preview_exclusions=lambda: clear_runtime_live_geometry_preview_exclusions(
            bindings_factory()
        ),
        toggle_preview_exclusion_at=lambda col, row: (
            toggle_runtime_live_geometry_preview_exclusion_at(
                bindings_factory(),
                col,
                row,
            )
        ),
        toggle_live_preview=lambda: toggle_runtime_live_geometry_preview(
            bindings_factory(),
            root=root,
            bindings_factory=bindings_factory,
        ),
        live_preview_enabled=lambda: runtime_live_geometry_preview_enabled(bindings_factory()),
        render_live_preview_state=lambda update_status=True: (
            render_runtime_live_geometry_preview_state(
                bindings_factory(),
                update_status=update_status,
            )
        ),
    )


def _save_json_payload(file_path: str, payload: Mapping[str, object]) -> None:
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_json_payload(file_path: str) -> object:
    with open(file_path, encoding="utf-8") as handle:
        return json.load(handle)


def apply_geometry_q_group_checkbox_change_with_side_effects(
    *,
    preview_state,
    group_key: tuple[object, ...] | None,
    row_var: object,
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    update_geometry_preview_exclude_button_label: Callable[[], None],
    update_geometry_q_group_window_status: Callable[[], None],
    live_geometry_preview_enabled: Callable[[], bool],
    refresh_live_geometry_preview: Callable[[], None],
    set_status_text: Callable[[str], None] | None = None,
) -> bool:
    """Apply one checkbox toggle and the dependent live-preview/status effects."""

    action = apply_geometry_q_group_checkbox_change(
        preview_state,
        group_key,
        row_var,
    )
    if action is None:
        return False

    invalidate_geometry_manual_pick_cache()
    update_geometry_preview_exclude_button_label()
    update_geometry_q_group_window_status()

    if live_geometry_preview_enabled():
        refresh_live_geometry_preview()
    else:
        _set_status_text(
            set_status_text,
            f"{action} one Qr/Qz group for geometry fitting.",
        )
    return True


def set_all_geometry_q_groups_enabled_with_side_effects(
    *,
    preview_state,
    q_group_state,
    enabled: bool,
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    update_geometry_preview_exclude_button_label: Callable[[], None],
    refresh_geometry_q_group_window: Callable[[], None],
    live_geometry_preview_enabled: Callable[[], bool],
    refresh_live_geometry_preview: Callable[[], None],
    set_status_text: Callable[[str], None] | None = None,
) -> bool:
    """Apply a bulk include/exclude action and the dependent side effects."""

    action, count = set_all_geometry_q_groups_enabled(
        preview_state,
        q_group_state,
        enabled=enabled,
    )
    if count <= 0:
        _set_status_text(
            set_status_text,
            'No listed Qr/Qz groups are available. Press "Update Listed Peaks" first.',
        )
        return False

    invalidate_geometry_manual_pick_cache()
    update_geometry_preview_exclude_button_label()
    refresh_geometry_q_group_window()

    if live_geometry_preview_enabled():
        refresh_live_geometry_preview()
    else:
        _set_status_text(
            set_status_text,
            f"{action} {count} Qr/Qz groups for geometry fitting.",
        )
    return True


def request_geometry_q_group_window_update_with_side_effects(
    *,
    q_group_state,
    clear_last_simulation_signature: Callable[[], None],
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    set_status_text: Callable[[str], None] | None = None,
    schedule_update: Callable[[], None],
) -> None:
    """Request a listed-peak refresh and trigger the dependent runtime effects."""

    request_geometry_q_group_window_update(q_group_state)
    clear_last_simulation_signature()
    invalidate_geometry_manual_pick_cache()
    _set_status_text(
        set_status_text,
        "Updating listed Qr/Qz peaks from the current simulation...",
    )
    schedule_update()


def save_geometry_q_group_selection_with_dialog(
    *,
    preview_state,
    q_group_state,
    file_dialog_dir: object,
    asksaveasfilename: Callable[..., object],
    set_status_text: Callable[[str], None] | None = None,
    save_payload: Callable[[str, Mapping[str, object]], None] | None = None,
    now: Callable[[], datetime] | None = None,
) -> bool:
    """Export the current selector rows through a save-file dialog."""

    export_rows = build_geometry_q_group_export_rows(
        preview_state=preview_state,
        q_group_state=q_group_state,
    )
    if not export_rows:
        _set_status_text(
            set_status_text,
            "No listed Qr/Qz groups are available to save. Press Update Listed Peaks first.",
        )
        return False

    now_value = now() if callable(now) else datetime.now()
    file_path = asksaveasfilename(
        title="Save Geometry Fit Qr/Qz Peak List",
        initialdir=str(file_dialog_dir),
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        initialfile=(f"geometry_q_groups_{now_value.strftime('%Y%m%d_%H%M%S')}.json"),
    )
    if not file_path:
        _set_status_text(set_status_text, "Save Qr/Qz peak list canceled.")
        return False

    payload = build_geometry_q_group_save_payload(
        export_rows,
        saved_at=now_value.isoformat(timespec="seconds"),
    )
    writer = save_payload or _save_json_payload
    try:
        writer(str(file_path), payload)
    except Exception as exc:
        _set_status_text(
            set_status_text,
            f"Failed to save Qr/Qz peak list: {exc}",
        )
        return False

    _set_status_text(
        set_status_text,
        f"Saved {len(export_rows)} Qr/Qz groups to {file_path}",
    )
    return True


def load_geometry_q_group_selection_with_dialog(
    *,
    preview_state,
    q_group_state,
    file_dialog_dir: object,
    askopenfilename: Callable[..., object],
    update_geometry_preview_exclude_button_label: Callable[[], None],
    refresh_geometry_q_group_window: Callable[[], None],
    live_geometry_preview_enabled: Callable[[], bool],
    refresh_live_geometry_preview: Callable[[], None],
    set_status_text: Callable[[str], None] | None = None,
    load_payload: Callable[[str], object] | None = None,
) -> bool:
    """Import selector rows through an open-file dialog and apply them."""

    file_path = askopenfilename(
        title="Load Geometry Fit Qr/Qz Peak List",
        initialdir=str(file_dialog_dir),
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    if not file_path:
        _set_status_text(set_status_text, "Load Qr/Qz peak list canceled.")
        return False

    reader = load_payload or _load_json_payload
    try:
        payload = reader(str(file_path))
    except Exception as exc:
        _set_status_text(
            set_status_text,
            f"Failed to load Qr/Qz peak list: {exc}",
        )
        return False

    saved_state, error = load_geometry_q_group_saved_state(payload)
    if error is not None:
        _set_status_text(set_status_text, error)
        return False

    summary, error = apply_loaded_geometry_q_group_saved_state(
        preview_state=preview_state,
        q_group_state=q_group_state,
        saved_state=saved_state,
    )
    if error is not None:
        _set_status_text(set_status_text, error)
        return False

    update_geometry_preview_exclude_button_label()
    refresh_geometry_q_group_window()
    if live_geometry_preview_enabled():
        refresh_live_geometry_preview()

    _set_status_text(
        set_status_text,
        (
            f"Loaded Qr/Qz peak list from {Path(str(file_path)).name}: "
            f"matched {summary['matched_total']}, enabled {summary['included_total']}, "
            f"current-only excluded {summary['current_only']}, "
            f"saved-only missing {summary['saved_only']}."
        ),
    )
    return True
