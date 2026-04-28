"""Selective GUI runtime cache invalidation helpers."""

from __future__ import annotations

from collections.abc import MutableMapping

from ra_sim.gui.runtime_update_dependencies import UpdateAction


_COMBINED_FIELDS = (
    "stored_sim_image",
    "stored_max_positions_local",
    "stored_source_reflection_indices_local",
    "stored_peak_table_lattice",
    "stored_intersection_cache",
    "stored_q_group_content_signature",
)

_ANALYSIS_GEOMETRY_FIELDS = (
    "last_analysis_signature",
    "last_analysis_cache_sig",
    "last_q_space_payload_signature",
    "last_res2_sim",
    "last_res2_background",
    "last_caked_image_unscaled",
    "last_caked_extent",
    "last_caked_background_image_unscaled",
    "last_caked_radial_values",
    "last_caked_azimuth_values",
    "last_q_space_image_unscaled",
    "last_q_space_extent",
    "last_q_space_background_image_unscaled",
    "last_q_space_qr_values",
    "last_q_space_qz_values",
    "last_caked_transform_bundle",
    "last_caked_intersection_cache",
    "last_caked_intersection_cache_transform_bundle",
    "last_caked_intersection_cache_source_signature",
)

_PRIMARY_CACHE_FIELDS = (
    "primary_contribution_cache_signature",
    "primary_active_contribution_keys",
    "primary_hit_table_cache",
    "primary_best_sample_index_cache",
    "primary_relative_hit_table_cache",
    "primary_relative_hit_table_cache_center",
    "primary_relative_hit_table_cache_signature",
    "primary_source_mode",
    "primary_filter_signature",
)

_SECONDARY_REMAP_CACHE_FIELDS = (
    "secondary_relative_hit_table_cache",
    "secondary_relative_best_sample_index_cache",
    "secondary_relative_hit_table_cache_center",
    "secondary_relative_hit_table_cache_signature",
)

_SIDE_ARTIFACT_FIELDS = (
    "stored_primary_sim_image",
    "stored_secondary_sim_image",
    "stored_primary_max_positions",
    "stored_secondary_max_positions",
    "stored_primary_source_reflection_indices",
    "stored_secondary_source_reflection_indices",
    "stored_primary_peak_table_lattice",
    "stored_secondary_peak_table_lattice",
    "stored_primary_intersection_cache",
    "stored_secondary_intersection_cache",
    "stored_primary_intersection_cache_signature",
    "stored_secondary_intersection_cache_signature",
    "stored_hit_table_signature",
)


def _set_if_present(state: object, name: str, value: object) -> None:
    if hasattr(state, name):
        setattr(state, name, value)


def _clear_fields(state: object, fields: tuple[str, ...]) -> None:
    for name in fields:
        _set_if_present(state, name, None)


def _clear_combined_artifacts(state: object) -> None:
    _clear_fields(state, _COMBINED_FIELDS)


def _clear_analysis_geometry_caches(state: object) -> None:
    _clear_fields(state, _ANALYSIS_GEOMETRY_FIELDS)
    _set_if_present(state, "analysis_preview_active", False)
    _set_if_present(state, "analysis_preview_bins", None)
    _set_if_present(state, "ai_cache", {})
    _set_if_present(state, "geometry_fit_caking_ai_cache", {})
    _set_if_present(state, "caking_cache", {"sim_results": {}, "bg_results": {}})
    normalization_cache = getattr(state, "normalization_scale_cache", None)
    if isinstance(normalization_cache, MutableMapping):
        normalization_cache["sig"] = None
    elif hasattr(state, "normalization_scale_cache"):
        state.normalization_scale_cache = {"sig": None, "value": 1.0}


def _clear_peak_selection(state: object) -> None:
    _set_if_present(state, "peak_positions", [])
    _set_if_present(state, "peak_millers", [])
    _set_if_present(state, "peak_intensities", [])
    _set_if_present(state, "peak_records", [])
    _set_if_present(state, "selected_peak_record", None)
    _set_if_present(state, "geometry_q_group_entries_cache_signature", None)
    _set_if_present(state, "geometry_q_group_entries_cache", [])


def _clear_primary_contribution_cache(state: object) -> None:
    _set_if_present(state, "primary_contribution_cache_signature", None)
    _set_if_present(state, "primary_active_contribution_keys", [])
    _set_if_present(state, "primary_hit_table_cache", {})
    _set_if_present(state, "primary_best_sample_index_cache", {})
    _set_if_present(state, "primary_relative_hit_table_cache", {})
    _set_if_present(state, "primary_relative_hit_table_cache_center", None)
    _set_if_present(state, "primary_relative_hit_table_cache_signature", None)
    _set_if_present(state, "primary_filter_signature", None)


def _clear_secondary_remap_cache(state: object) -> None:
    _set_if_present(state, "secondary_relative_hit_table_cache", [])
    _set_if_present(state, "secondary_relative_best_sample_index_cache", {})
    _set_if_present(state, "secondary_relative_hit_table_cache_center", None)
    _set_if_present(state, "secondary_relative_hit_table_cache_signature", None)


def _clear_full_simulation_state(state: object) -> None:
    _clear_combined_artifacts(state)
    _clear_fields(state, _SIDE_ARTIFACT_FIELDS)
    _clear_analysis_geometry_caches(state)
    _clear_primary_contribution_cache(state)
    _clear_secondary_remap_cache(state)
    _set_if_present(state, "last_sim_signature", None)
    _set_if_present(state, "last_simulation_signature", None)
    _set_if_present(state, "last_dependency_signatures", None)
    _set_if_present(state, "source_row_snapshots", {})
    _clear_peak_selection(state)


def invalidate_for_update_action(state: object, action: UpdateAction) -> None:
    """Apply the narrow cache invalidation implied by one update action."""

    if not isinstance(action, UpdateAction):
        action = UpdateAction(str(action))

    if action in {UpdateAction.DISPLAY_ONLY, UpdateAction.COMBINE_ONLY}:
        return

    if action is UpdateAction.PRIMARY_PRUNE_REUSE:
        _clear_combined_artifacts(state)
        return

    if action is UpdateAction.PRIMARY_PRUNE_FILL:
        return

    if action is UpdateAction.DETECTOR_CENTER_REMAP:
        _clear_analysis_geometry_caches(state)
        _set_if_present(state, "source_row_snapshots", {})
        return

    if action is UpdateAction.ANALYSIS_ONLY:
        _clear_analysis_geometry_caches(state)
        return

    if action is UpdateAction.FULL_SIMULATION:
        _clear_full_simulation_state(state)
        return

    if action is UpdateAction.HIT_TABLE_REFRESH:
        _set_if_present(state, "stored_hit_table_signature", None)
        _set_if_present(state, "stored_intersection_cache", None)
        _clear_peak_selection(state)


__all__ = ["invalidate_for_update_action"]
