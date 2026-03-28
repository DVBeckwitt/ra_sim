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

from . import controllers as gui_controllers
from . import manual_geometry as gui_manual_geometry
from . import geometry_overlay as gui_geometry_overlay
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


def reflection_q_group_metadata(
    hkl_value: object,
    *,
    source_label: object = "primary",
    a_value: object = np.nan,
    c_value: object = np.nan,
    qr_value: object = np.nan,
) -> tuple[tuple[object, ...] | None, float, float]:
    """Return stable Qr/Qz grouping metadata for one simulated reflection."""

    if isinstance(hkl_value, (list, tuple, np.ndarray)) and len(hkl_value) >= 3:
        try:
            h_raw = float(hkl_value[0])
            k_raw = float(hkl_value[1])
            l_raw = float(hkl_value[2])
        except Exception:
            return None, float("nan"), float("nan")
    else:
        return None, float("nan"), float("nan")

    m_val = h_raw * h_raw + h_raw * k_raw + k_raw * k_raw
    l_int = gui_manual_geometry.integer_gz_index(l_raw)
    if l_int is None:
        return None, float("nan"), float("nan")

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
    key, _, _ = reflection_q_group_metadata(
        entry.get("hkl_raw", entry.get("hkl")),
        source_label=entry.get("source_label", "primary"),
        a_value=entry.get("av", np.nan),
        c_value=entry.get("cv", np.nan),
        qr_value=entry.get("qr", np.nan),
    )
    return key


def build_geometry_q_group_entries(
    max_positions_local: Sequence[object] | None,
    *,
    peak_table_lattice: Sequence[Sequence[object]] | None = None,
    primary_a: object = np.nan,
    primary_c: object = np.nan,
) -> list[dict[str, object]]:
    """Aggregate simulated hit tables into unique Qr/Qz selector rows."""

    if max_positions_local is None:
        return []

    try:
        default_primary_a = float(primary_a)
    except Exception:
        default_primary_a = float("nan")
    try:
        default_primary_c = float(primary_c)
    except Exception:
        default_primary_c = float("nan")

    peak_table_lattice_local = list(peak_table_lattice or [])
    if not peak_table_lattice_local or len(peak_table_lattice_local) != len(
        max_positions_local
    ):
        peak_table_lattice_local = [
            (default_primary_a, default_primary_c, "primary")
            for _ in max_positions_local
        ]

    entries_by_key: dict[tuple[object, ...], dict[str, object]] = {}
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

            entry["total_intensity"] = float(entry["total_intensity"]) + float(
                abs(intensity)
            )
            entry["peak_count"] = int(entry["peak_count"]) + 1
            if len(entry["hkl_preview"]) < 4 and hkl_key not in entry["hkl_preview"]:
                entry["hkl_preview"].append(hkl_key)

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
    primary_a: object = np.nan,
    primary_c: object = np.nan,
    default_source_label: str | None = "primary",
    round_pixel_centers: bool = False,
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
    for table_idx, tbl in enumerate(hit_tables):
        rows = geometry_reference_hit_rows(tbl)
        if not rows:
            continue

        source_label = (
            str(default_source_label)
            if default_source_label is not None
            else f"table_{table_idx}"
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
            intensity, xpix, ypix, _phi, h_val, k_val, l_val = row[:7]
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
            )
            if q_group_key is None:
                continue

            simulated_peaks.append(
                {
                    "hkl": hkl,
                    "label": f"{hkl[0]},{hkl[1]},{hkl[2]}",
                    "sim_col": float(display_col),
                    "sim_row": float(display_row),
                    "weight": max(0.0, float(abs(intensity))),
                    "source_peak_index": int(len(simulated_peaks)),
                    "source_label": str(source_label),
                    "source_table_index": int(table_idx),
                    "source_row_index": int(row_idx),
                    "hkl_raw": hkl_raw,
                    "av": float(av_used),
                    "cv": float(cv_used),
                    "qr": float(qr_val),
                    "qz": float(qz_val),
                    "q_group_key": q_group_key,
                }
            )

    return simulated_peaks


def geometry_fit_peak_center_from_max_position(
    entry: Sequence[float] | None,
) -> tuple[float, float] | None:
    """Return the strongest finite peak center stored in one max-position row."""

    if entry is None or len(entry) < 6:
        return None
    i0, x0, y0, i1, x1, y1 = entry
    primary_valid = np.isfinite(x0) and np.isfinite(y0)
    secondary_valid = np.isfinite(x1) and np.isfinite(y1)

    if primary_valid and (
        not secondary_valid or not np.isfinite(i1) or float(i0) >= float(i1)
    ):
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
        weights_by_hkl[key] = weights_by_hkl.get(key, 0.0) + float(
            abs(intensity_arr[idx])
        )

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
    build_geometry_fit_central_mosaic_params: Callable[[dict[str, object]], object],
    process_peaks_parallel: Callable[..., object],
    default_solve_q_steps: int,
    default_solve_q_rel_tol: float,
    default_solve_q_mode: int,
) -> list[object]:
    """Simulate once and return raw hit tables for geometry-fit helpers."""

    params_local = dict(param_set)
    params_local["mosaic_params"] = build_geometry_fit_central_mosaic_params(
        params_local
    )
    mosaic = dict(params_local.get("mosaic_params", {}))
    wavelength_array = mosaic.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic.get("wavelength_i_array")
    if wavelength_array is None:
        wavelength_array = np.full(
            int(np.size(mosaic.get("beam_x_array", []))),
            float(params_local.get("lambda", 1.0)),
            dtype=np.float64,
        )

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
        solve_q_rel_tol=float(
            mosaic.get("solve_q_rel_tol", default_solve_q_rel_tol)
        ),
        solve_q_mode=int(mosaic.get("solve_q_mode", default_solve_q_mode)),
    )
    return list(hit_tables)


def simulate_geometry_fit_peak_centers(
    miller_array: np.ndarray,
    intensity_array: np.ndarray,
    image_size: int,
    param_set: Mapping[str, object] | dict[str, object],
    *,
    build_geometry_fit_central_mosaic_params: Callable[[dict[str, object]], object],
    process_peaks_parallel: Callable[..., object],
    hit_tables_to_max_positions: Callable[[Sequence[object]], Sequence[Sequence[float]]],
    default_solve_q_steps: int,
    default_solve_q_rel_tol: float,
    default_solve_q_mode: int,
) -> list[dict[str, object]]:
    """Simulate once and return one aggregated detector center per integer HKL."""

    hit_tables = simulate_geometry_fit_hit_tables(
        miller_array,
        intensity_array,
        image_size,
        param_set,
        build_geometry_fit_central_mosaic_params=(
            build_geometry_fit_central_mosaic_params
        ),
        process_peaks_parallel=process_peaks_parallel,
        default_solve_q_steps=default_solve_q_steps,
        default_solve_q_rel_tol=default_solve_q_rel_tol,
        default_solve_q_mode=default_solve_q_mode,
    )
    max_positions = hit_tables_to_max_positions(hit_tables)
    return aggregate_geometry_fit_peak_centers_from_max_positions(
        max_positions,
        miller_array,
        intensity_array,
    )


def simulate_geometry_fit_preview_style_peaks(
    miller_array: np.ndarray,
    intensity_array: np.ndarray,
    image_size: int,
    param_set: Mapping[str, object] | dict[str, object],
    *,
    build_geometry_fit_central_mosaic_params: Callable[[dict[str, object]], object],
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
    default_solve_q_steps: int = 0,
    default_solve_q_rel_tol: float = 0.0,
    default_solve_q_mode: int = 0,
) -> list[dict[str, object]]:
    """Simulate once and return per-hit peaks in the live preview format."""

    hit_tables = simulate_geometry_fit_hit_tables(
        miller_array,
        intensity_array,
        image_size,
        param_set,
        build_geometry_fit_central_mosaic_params=(
            build_geometry_fit_central_mosaic_params
        ),
        process_peaks_parallel=process_peaks_parallel,
        default_solve_q_steps=default_solve_q_steps,
        default_solve_q_rel_tol=default_solve_q_rel_tol,
        default_solve_q_mode=default_solve_q_mode,
    )
    return build_geometry_fit_simulated_peaks(
        hit_tables,
        image_shape=(int(image_size), int(image_size)),
        native_sim_to_display_coords=native_sim_to_display_coords,
        peak_table_lattice=peak_table_lattice,
        primary_a=primary_a,
        primary_c=primary_c,
        default_source_label=default_source_label,
        round_pixel_centers=round_pixel_centers,
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
    try:
        return (0, int(entry.get("source_peak_index")))
    except Exception:
        pass
    hkl_key = gui_geometry_overlay.normalize_hkl_key(
        entry.get("hkl", entry.get("label"))
    )
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
        f"Qr={qr_val:8.4f}  "
        f"Gz={gz_text}  "
        f"Qz={qz_val:8.4f}  "
        f"I={total_intensity:10.3f}  "
        f"hits={peak_count:4d}"
        + (f"  HKL={hkl_preview}" if hkl_preview else "")
    )


def current_geometry_auto_match_min_matches(
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names: Sequence[str] | None,
) -> int:
    """Return the current geometry auto-match minimum peak count."""

    geometry_refine_cfg = fit_config.get("geometry", {}) if isinstance(
        fit_config,
        Mapping,
    ) else {}
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

    geometry_refine_cfg = fit_config.get("geometry", {}) if isinstance(
        fit_config,
        Mapping,
    ) else {}
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

    label = "Select Qr/Qz Peaks"
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

    rows = list(entries) if entries is not None else gui_controllers.listed_geometry_q_group_entries(
        q_group_state
    )
    excluded_q_groups = getattr(preview_state, "excluded_q_groups", set())
    total_count = len(rows)
    included_rows = [
        entry
        for entry in rows
        if entry.get("key") not in excluded_q_groups
    ]
    selected_peak_count = int(
        sum(_coerce_int(entry.get("peak_count", 0), 0) for entry in included_rows)
    )
    total_peak_count = int(
        sum(_coerce_int(entry.get("peak_count", 0), 0) for entry in rows)
    )
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
            [
                entry["key"]
                for entry in entries
                if entry.get("key") is not None
            ],
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
    source_entries = list(entries) if entries is not None else gui_controllers.listed_geometry_q_group_entries(
        q_group_state
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
            if (
                not isinstance(hkl_value, (list, tuple, np.ndarray))
                or len(hkl_value) < 3
            ):
                continue
            try:
                hkl_preview.append(
                    [int(hkl_value[0]), int(hkl_value[1]), int(hkl_value[2])]
                )
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
        "included_count": int(
            sum(1 for row in export_rows if bool(row.get("included", False)))
        ),
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
        "included_total": int(
            sum(1 for key in matched_keys if bool(saved_state.get(key, False)))
        ),
        "current_only": int(sum(1 for key in current_keys if key not in saved_state)),
        "saved_only": int(sum(1 for key in saved_state if key not in current_key_set)),
    }, None


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

    match_stats_local = (
        match_stats if isinstance(match_stats, Mapping) else {}
    )
    preview_cfg = (
        preview_auto_match_cfg if isinstance(preview_auto_match_cfg, Mapping) else {}
    )
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
        or (
            np.isfinite(max_auto_mean)
            and np.isfinite(mean_dist)
            and mean_dist > max_auto_mean
        )
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
        summary += (
            f" Qr/Qz groups on={max(0, q_group_total - q_group_excluded)}/{q_group_total}."
        )
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
    filter_live_preview_matches: Callable[[Sequence[dict[str, object]]], tuple[Sequence[dict[str, object]], int]],
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

    t = (
        ((float(px) - float(x0)) * dx + (float(py) - float(y0)) * dy)
        / (dx * dx + dy * dy)
    )
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
        refresh_geometry_q_group_window=lambda: refresh_runtime_geometry_q_group_window(
            bindings
        ),
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
        refresh_geometry_q_group_window=lambda: refresh_runtime_geometry_q_group_window(
            bindings
        ),
        live_geometry_preview_enabled=bindings.live_geometry_preview_enabled,
        refresh_live_geometry_preview=(
            bindings.refresh_live_geometry_preview_quiet
            or bindings.refresh_live_geometry_preview
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
            "Opened the Qr/Qz selector for manual geometry picking. "
            "Unchecked rows are unavailable when selecting Qr sets on the image."
        ),
    )
    return opened


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


def make_runtime_geometry_q_group_callbacks(
    *,
    root,
    bindings_factory: Callable[[], GeometryQGroupRuntimeBindings],
) -> GeometryQGroupRuntimeCallbacks:
    """Return bound zero-arg callbacks for the runtime Qr/Qz selector workflow."""

    return GeometryQGroupRuntimeCallbacks(
        update_window_status=lambda entries=None: update_runtime_geometry_q_group_window_status(
            bindings_factory(),
            entries=entries,
        ),
        refresh_window=lambda: refresh_runtime_geometry_q_group_window(
            bindings_factory()
        ),
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
        save_selection=lambda: save_geometry_q_group_selection_runtime(
            bindings_factory()
        ),
        load_selection=lambda: load_geometry_q_group_selection_runtime(
            bindings_factory()
        ),
        close_window=lambda: close_runtime_geometry_q_group_window(bindings_factory()),
        open_window=lambda: open_runtime_geometry_q_group_window(
            root=root,
            bindings_factory=bindings_factory,
        ),
    )


def _save_json_payload(file_path: str, payload: Mapping[str, object]) -> None:
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_json_payload(file_path: str) -> object:
    with open(file_path, "r", encoding="utf-8") as handle:
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
        initialfile=(
            f"geometry_q_groups_{now_value.strftime('%Y%m%d_%H%M%S')}.json"
        ),
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
