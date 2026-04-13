"""Helpers for copying and applying GUI primary-cache artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from ra_sim.gui import runtime_primary_cache as gui_runtime_primary_cache
from ra_sim.simulation.diffraction import intersection_cache_to_hit_tables


def copy_intersection_cache_tables(cache: object) -> list[np.ndarray]:
    copied: list[np.ndarray] = []
    if not isinstance(cache, (list, tuple)):
        return copied
    for table in cache:
        try:
            copied.append(np.asarray(table, dtype=np.float64).copy())
        except Exception:
            copied.append(np.empty((0, 14), dtype=np.float64))
    return copied


def copy_hit_tables(hit_tables: object) -> list[np.ndarray]:
    copied: list[np.ndarray] = []
    if not isinstance(hit_tables, (list, tuple)):
        return copied
    for table in hit_tables:
        try:
            copied.append(np.asarray(table, dtype=np.float64).copy())
        except Exception:
            copied.append(np.empty((0, 7), dtype=np.float64))
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
    simulation_runtime_state.stored_source_reflection_indices_local = None
    simulation_runtime_state.stored_peak_table_lattice = None
    simulation_runtime_state.stored_sim_image = None


def store_primary_cache_payload(
    simulation_runtime_state: object,
    *,
    cache_signature: object,
    source_mode: str,
    active_keys: Sequence[object],
    contribution_keys: Sequence[object],
    raw_hit_tables: Sequence[np.ndarray],
    retain_runtime_optional_cache: Callable[..., bool],
    trace_live_cache_event: Callable[..., None],
    live_cache_count: Callable[[object], int],
    live_cache_signature_summary: Callable[[object], str | None],
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

    simulation_runtime_state.primary_contribution_cache_signature = cache_signature
    simulation_runtime_state.primary_source_mode = normalized_mode
    simulation_runtime_state.primary_active_contribution_keys = list(active_keys or ())

    copied_tables: list[np.ndarray] = []
    overwritten_count = 0
    if retain_cache:
        copied_tables = copy_hit_tables(raw_hit_tables)
        for key, table in zip(contribution_keys or (), copied_tables):
            if key in simulation_runtime_state.primary_hit_table_cache:
                overwritten_count += 1
            simulation_runtime_state.primary_hit_table_cache[key] = np.asarray(
                table,
                dtype=np.float64,
            ).copy()
    else:
        simulation_runtime_state.primary_hit_table_cache = {}
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
    simulation_runtime_state.stored_primary_source_reflection_indices = list(
        range(len(simulation_runtime_state.stored_primary_max_positions or ()))
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
