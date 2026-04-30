"""Helpers for copying and applying GUI primary-cache artifacts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, cast

import numpy as np

from ra_sim.gui import runtime_detector_remap_cache as gui_runtime_detector_remap_cache
from ra_sim.gui import runtime_primary_cache as gui_runtime_primary_cache
from ra_sim.gui import geometry_q_group_manager as gui_geometry_q_group_manager
from ra_sim.simulation.diffraction import intersection_cache_to_hit_tables
from ra_sim.simulation.intersection_cache_schema import (
    coerce_intersection_cache_table,
    empty_hit_table,
)


def _iter_table_payload(payload: object) -> list[object]:
    if payload is None:
        return []
    if isinstance(payload, (str, bytes, Mapping)):
        return []
    if isinstance(payload, np.ndarray):
        arr = np.asarray(payload)
        if arr.ndim == 2:
            return [arr]
        if arr.ndim == 3:
            return [arr[idx] for idx in range(arr.shape[0])]
        return []
    if not isinstance(payload, Iterable):
        return []
    try:
        return list(payload)
    except Exception:
        return []


def copy_intersection_cache_tables(
    cache: object,
    *,
    allow_abbreviated_detector_cache: bool = False,
) -> list[np.ndarray]:
    copied: list[np.ndarray] = []
    for table in _iter_table_payload(cache):
        copied.append(
            coerce_intersection_cache_table(
                table,
                allow_abbreviated_detector_cache=allow_abbreviated_detector_cache,
            )
        )
    return copied


def copy_hit_tables(hit_tables: object) -> list[np.ndarray]:
    copied: list[np.ndarray] = []
    for table in _iter_table_payload(hit_tables):
        try:
            arr = np.asarray(table, dtype=np.float64)
        except Exception:
            copied.append(empty_hit_table())
            continue
        if arr.ndim != 2:
            copied.append(empty_hit_table())
            continue
        copied.append(arr.copy())
    return copied


def resolved_peak_table_payload(
    intersection_cache: object,
    legacy_hit_tables: object,
) -> list[np.ndarray]:
    cache_backed_tables = intersection_cache_to_hit_tables(intersection_cache)
    if cache_backed_tables:
        for table in cache_backed_tables:
            try:
                if np.asarray(table).size > 0:
                    return copy_hit_tables(cache_backed_tables)
            except Exception:
                continue
    return copy_hit_tables(legacy_hit_tables)


def clear_primary_contribution_cache(simulation_runtime_state: object) -> None:
    simulation_runtime_state.primary_contribution_cache_signature = None
    simulation_runtime_state.primary_active_contribution_keys = []
    simulation_runtime_state.primary_hit_table_cache = {}
    simulation_runtime_state.primary_best_sample_index_cache = {}
    simulation_runtime_state.primary_relative_hit_table_cache = {}
    simulation_runtime_state.primary_relative_hit_table_cache_center = None
    simulation_runtime_state.primary_relative_hit_table_cache_signature = None
    simulation_runtime_state.primary_filter_signature = None


def reset_combined_simulation_artifacts(
    simulation_runtime_state: object,
    *,
    trace_live_cache_event: Callable[..., None],
    live_cache_count: Callable[[object], int],
) -> None:
    trace_live_cache_event(
        "combined",
        "clear",
        outcome="cleared",
        had_peak_table_count=live_cache_count(simulation_runtime_state.stored_max_positions_local),
        had_peak_table_lattice_count=live_cache_count(
            simulation_runtime_state.stored_peak_table_lattice
        ),
        had_sim_image=bool(simulation_runtime_state.stored_sim_image is not None),
    )
    simulation_runtime_state.stored_max_positions_local = None
    simulation_runtime_state.stored_q_group_content_signature = None
    simulation_runtime_state.stored_source_reflection_indices_local = None
    simulation_runtime_state.stored_peak_table_lattice = None
    simulation_runtime_state.stored_sim_image = None
    simulation_runtime_state.stored_intersection_cache = None


def store_primary_cache_payload(
    simulation_runtime_state: object,
    *,
    cache_signature: object,
    source_mode: str,
    active_keys: Sequence[object],
    contribution_keys: Sequence[object],
    raw_hit_tables: Sequence[np.ndarray],
    best_sample_indices: Sequence[object] | None,
    retain_runtime_optional_cache: Callable[..., bool],
    trace_live_cache_event: Callable[..., None],
    live_cache_count: Callable[[object], int],
    live_cache_signature_summary: Callable[[object], str | None],
    detector_center: Sequence[object] | None = None,
    detector_remap_cache_signature: object = None,
    store_detector_relative_hit_tables: bool = False,
) -> None:
    previous_signature = simulation_runtime_state.primary_contribution_cache_signature
    previous_mode = str(simulation_runtime_state.primary_source_mode or "")
    previous_entry_count = live_cache_count(simulation_runtime_state.primary_hit_table_cache)
    if cache_signature is None:
        trace_live_cache_event(
            "primary_contribution",
            "store",
            outcome="skipped",
            reason="missing_signature",
            previous_signature_summary=live_cache_signature_summary(previous_signature),
            previous_source_mode=previous_mode,
            previous_entry_count=int(previous_entry_count),
        )
        return

    normalized_mode = str(source_mode or "miller")
    retain_cache = retain_runtime_optional_cache(
        "primary_contribution",
        feature_needed=bool(active_keys),
    )
    reset_required = bool(
        cache_signature != simulation_runtime_state.primary_contribution_cache_signature
        or normalized_mode != str(simulation_runtime_state.primary_source_mode or "")
    )
    if reset_required:
        simulation_runtime_state.primary_hit_table_cache = {}
        simulation_runtime_state.primary_best_sample_index_cache = {}
        simulation_runtime_state.primary_relative_hit_table_cache = {}
        simulation_runtime_state.primary_relative_hit_table_cache_center = None
        simulation_runtime_state.primary_relative_hit_table_cache_signature = None

    simulation_runtime_state.primary_contribution_cache_signature = cache_signature
    simulation_runtime_state.primary_source_mode = normalized_mode
    simulation_runtime_state.primary_active_contribution_keys = list(active_keys or ())

    copied_tables: list[np.ndarray] = []
    copied_relative_tables: list[np.ndarray] = []
    overwritten_count = 0
    if retain_cache:
        copied_tables = copy_hit_tables(raw_hit_tables)
        if store_detector_relative_hit_tables and detector_center is not None:
            copied_relative_tables = (
                gui_runtime_detector_remap_cache.make_relative_hit_tables_for_center(
                    copied_tables,
                    detector_center,
                )
            )
            if not hasattr(simulation_runtime_state, "primary_relative_hit_table_cache"):
                simulation_runtime_state.primary_relative_hit_table_cache = {}
            center_values = list(detector_center)[:2]
            simulation_runtime_state.primary_relative_hit_table_cache_center = tuple(
                float(cast(Any, value)) for value in center_values
            )
            simulation_runtime_state.primary_relative_hit_table_cache_signature = (
                cache_signature
                if detector_remap_cache_signature is None
                else detector_remap_cache_signature
            )
        for idx, (key, table) in enumerate(zip(contribution_keys or (), copied_tables)):
            if key in simulation_runtime_state.primary_hit_table_cache:
                overwritten_count += 1
            simulation_runtime_state.primary_hit_table_cache[key] = np.asarray(
                table,
                dtype=np.float64,
            ).copy()
            if isinstance(best_sample_indices, Sequence) and idx < len(best_sample_indices):
                try:
                    simulation_runtime_state.primary_best_sample_index_cache[key] = int(
                        cast(Any, best_sample_indices[idx])
                    )
                except Exception:
                    pass
            if idx < len(copied_relative_tables):
                simulation_runtime_state.primary_relative_hit_table_cache[key] = np.asarray(
                    copied_relative_tables[idx],
                    dtype=np.float64,
                ).copy()
    else:
        simulation_runtime_state.primary_hit_table_cache = {}
        simulation_runtime_state.primary_best_sample_index_cache = {}
        simulation_runtime_state.primary_relative_hit_table_cache = {}
        simulation_runtime_state.primary_relative_hit_table_cache_center = None
        simulation_runtime_state.primary_relative_hit_table_cache_signature = None
    trace_live_cache_event(
        "primary_contribution",
        "store",
        outcome=(
            "discarded" if not retain_cache else ("reset_and_store" if reset_required else "store")
        ),
        previous_signature_summary=live_cache_signature_summary(previous_signature),
        signature_summary=live_cache_signature_summary(cache_signature),
        previous_source_mode=previous_mode,
        source_mode=normalized_mode,
        previous_entry_count=int(previous_entry_count),
        entry_count=live_cache_count(simulation_runtime_state.primary_hit_table_cache),
        active_key_count=live_cache_count(active_keys),
        provided_key_count=live_cache_count(contribution_keys),
        stored_table_count=live_cache_count(copied_tables),
        overwritten=int(overwritten_count),
        reset=bool(reset_required),
    )


def rematerialize_primary_cache_artifacts(
    simulation_runtime_state: object,
    *,
    image_size: int,
    mosaic_params: Mapping[str, object],
    a_primary: float,
    c_primary: float,
    trace_live_cache_event: Callable[..., None],
    live_cache_signature_summary: Callable[[object], str | None],
    live_cache_shape: Callable[[object], list[int]],
    live_cache_count: Callable[[object], int],
) -> dict[str, object]:
    payload = gui_runtime_primary_cache.rematerialize_primary_artifacts(
        primary_hit_table_cache=simulation_runtime_state.primary_hit_table_cache,
        primary_best_sample_index_cache=simulation_runtime_state.primary_best_sample_index_cache,
        active_keys=simulation_runtime_state.primary_active_contribution_keys,
        image_size=int(image_size),
        a_primary=float(a_primary),
        c_primary=float(c_primary),
        beam_x_array=np.asarray(mosaic_params["beam_x_array"], dtype=np.float64),
        beam_y_array=np.asarray(mosaic_params["beam_y_array"], dtype=np.float64),
        theta_array=np.asarray(mosaic_params["theta_array"], dtype=np.float64),
        phi_array=np.asarray(mosaic_params["phi_array"], dtype=np.float64),
        wavelength_array=np.asarray(
            mosaic_params["wavelength_array"],
            dtype=np.float64,
        ),
        lattice_label="primary",
    )
    trace_live_cache_event(
        "primary_contribution",
        "rematerialize",
        outcome="success",
        signature_summary=live_cache_signature_summary(
            simulation_runtime_state.primary_contribution_cache_signature
        ),
        active_key_count=live_cache_count(
            simulation_runtime_state.primary_active_contribution_keys
        ),
        cache_entry_count=live_cache_count(simulation_runtime_state.primary_hit_table_cache),
        image_shape=live_cache_shape(payload.get("image")),
        intersection_cache_count=live_cache_count(payload.get("intersection_cache")),
        peak_table_count=live_cache_count(payload.get("peak_tables")),
    )
    return payload


def apply_primary_cache_artifacts(
    simulation_runtime_state: object,
    payload: Mapping[str, object],
    *,
    reset_combined_simulation_artifacts: Callable[[], None],
    trace_live_cache_event: Callable[..., None],
    live_cache_signature_summary: Callable[[object], str | None],
    live_cache_shape: Callable[[object], list[int]],
    live_cache_count: Callable[[object], int],
) -> None:
    simulation_runtime_state.stored_primary_sim_image = np.asarray(
        payload.get("image"),
        dtype=np.float64,
    )
    simulation_runtime_state.stored_primary_intersection_cache = copy_intersection_cache_tables(
        payload.get("intersection_cache", [])
    )
    simulation_runtime_state.stored_primary_max_positions = copy_hit_tables(
        payload.get("peak_tables", [])
    )
    simulation_runtime_state.stored_primary_source_reflection_indices = (
        gui_geometry_q_group_manager.audited_full_order_source_reflection_index_groups(
            (simulation_runtime_state.stored_primary_max_positions or (),),
            owner="primary_cache_helpers.apply_primary_cache_artifacts",
        )[0]
    )
    peak_table_lattice = payload.get("peak_table_lattice", [])
    simulation_runtime_state.stored_primary_peak_table_lattice = (
        list(peak_table_lattice) if isinstance(peak_table_lattice, (list, tuple)) else []
    )
    reset_combined_simulation_artifacts()
    simulation_runtime_state.preview_active = False
    simulation_runtime_state.preview_sample_count = None
    trace_live_cache_event(
        "primary_contribution",
        "apply",
        outcome="applied",
        signature_summary=live_cache_signature_summary(
            simulation_runtime_state.primary_contribution_cache_signature
        ),
        image_shape=live_cache_shape(payload.get("image")),
        intersection_cache_count=live_cache_count(payload.get("intersection_cache")),
        peak_table_count=live_cache_count(payload.get("peak_tables")),
        peak_table_lattice_count=live_cache_count(payload.get("peak_table_lattice")),
    )


__all__ = [
    "apply_primary_cache_artifacts",
    "clear_primary_contribution_cache",
    "copy_hit_tables",
    "copy_intersection_cache_tables",
    "rematerialize_primary_cache_artifacts",
    "reset_combined_simulation_artifacts",
    "resolved_peak_table_payload",
    "store_primary_cache_payload",
]
