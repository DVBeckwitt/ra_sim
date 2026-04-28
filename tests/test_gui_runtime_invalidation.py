from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ra_sim.gui.runtime_invalidation import invalidate_for_update_action
from ra_sim.gui.runtime_update_dependencies import UpdateAction


def _state() -> SimpleNamespace:
    primary_cache = {0: np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)}
    relative_cache = {0: np.asarray([[1.0, -1.0, -2.0]], dtype=np.float64)}
    secondary_relative = [np.asarray([[2.0, -3.0, -4.0]], dtype=np.float64)]
    return SimpleNamespace(
        last_sim_signature=("sim",),
        last_simulation_signature=("sim", 0, 0),
        last_dependency_signatures=("deps",),
        stored_primary_sim_image=np.ones((2, 2), dtype=np.float64),
        stored_secondary_sim_image=np.full((2, 2), 2.0, dtype=np.float64),
        stored_sim_image=np.full((2, 2), 3.0, dtype=np.float64),
        stored_primary_max_positions=["primary-peak"],
        stored_secondary_max_positions=["secondary-peak"],
        stored_max_positions_local=["primary-peak", "secondary-peak"],
        stored_source_reflection_indices_local=[0, 1],
        stored_primary_source_reflection_indices=[0],
        stored_secondary_source_reflection_indices=[1],
        stored_peak_table_lattice=["p", "s"],
        stored_primary_peak_table_lattice=["p"],
        stored_secondary_peak_table_lattice=["s"],
        stored_primary_intersection_cache=["primary-cache"],
        stored_secondary_intersection_cache=["secondary-cache"],
        stored_intersection_cache=["primary-cache", "secondary-cache"],
        stored_primary_intersection_cache_signature=("hit",),
        stored_secondary_intersection_cache_signature=("hit",),
        stored_hit_table_signature=("hit",),
        stored_q_group_content_signature=("rows",),
        primary_contribution_cache_signature=("primary-contrib",),
        primary_active_contribution_keys=[0],
        primary_hit_table_cache=primary_cache,
        primary_best_sample_index_cache={0: 0},
        primary_relative_hit_table_cache=relative_cache,
        primary_relative_hit_table_cache_center=(10.0, 20.0),
        primary_relative_hit_table_cache_signature=("primary-remap",),
        primary_source_mode="miller",
        primary_filter_signature=("filter",),
        secondary_relative_hit_table_cache=secondary_relative,
        secondary_relative_best_sample_index_cache={0: 0},
        secondary_relative_hit_table_cache_center=(10.0, 20.0),
        secondary_relative_hit_table_cache_signature=("secondary-remap",),
        source_row_snapshots={0: {"row": 1}},
        last_analysis_signature=("analysis",),
        last_analysis_cache_sig=("analysis-cache",),
        last_q_space_payload_signature=("q-payload",),
        last_res2_sim=object(),
        last_res2_background=object(),
        last_caked_image_unscaled=np.ones((2, 2), dtype=np.float64),
        last_caked_extent=[0.0, 1.0, -1.0, 1.0],
        last_caked_background_image_unscaled=np.ones((2, 2), dtype=np.float64),
        last_caked_radial_values=np.asarray([1.0, 2.0], dtype=np.float64),
        last_caked_azimuth_values=np.asarray([-1.0, 1.0], dtype=np.float64),
        last_q_space_image_unscaled=np.ones((2, 2), dtype=np.float64),
        last_q_space_extent=[0.0, 1.0, -1.0, 1.0],
        last_q_space_background_image_unscaled=np.ones((2, 2), dtype=np.float64),
        last_q_space_qr_values=np.asarray([0.1, 0.2], dtype=np.float64),
        last_q_space_qz_values=np.asarray([0.3, 0.4], dtype=np.float64),
        last_caked_transform_bundle=object(),
        last_caked_intersection_cache=["caked-cache"],
        last_caked_intersection_cache_transform_bundle=object(),
        last_caked_intersection_cache_source_signature=("caked-source",),
        ai_cache={"sig": ("ai",), "ai": object()},
        geometry_fit_caking_ai_cache={"sig": ("fit-ai",)},
        caking_cache={"sim_results": {"x": 1}, "bg_results": {"y": 2}},
        normalization_scale_cache={"sig": ("norm",), "value": 1.25},
        analysis_preview_active=True,
        analysis_preview_bins=(32, 64),
        peak_positions=[(1.0, 2.0)],
        peak_millers=[(1, 0, 0)],
        peak_intensities=[10.0],
        peak_records=[{"hkl": (1, 0, 0)}],
        selected_peak_record={"hkl": (1, 0, 0)},
        geometry_q_group_entries_cache_signature=("q-group",),
        geometry_q_group_entries_cache=[{"group": 1}],
    )


def test_display_only_preserves_all_simulation_caches() -> None:
    state = _state()
    preserved = {
        "stored_primary_sim_image": state.stored_primary_sim_image,
        "stored_secondary_sim_image": state.stored_secondary_sim_image,
        "stored_sim_image": state.stored_sim_image,
        "primary_hit_table_cache": state.primary_hit_table_cache,
        "primary_relative_hit_table_cache": state.primary_relative_hit_table_cache,
        "secondary_relative_hit_table_cache": state.secondary_relative_hit_table_cache,
        "last_caked_image_unscaled": state.last_caked_image_unscaled,
    }

    invalidate_for_update_action(state, UpdateAction.DISPLAY_ONLY)

    for name, value in preserved.items():
        assert getattr(state, name) is value
    assert state.last_analysis_signature == ("analysis",)


def test_prune_reuse_preserves_primary_contribution_cache() -> None:
    state = _state()
    primary_cache = state.primary_hit_table_cache
    relative_cache = state.primary_relative_hit_table_cache

    invalidate_for_update_action(state, UpdateAction.PRIMARY_PRUNE_REUSE)

    assert state.primary_hit_table_cache is primary_cache
    assert state.primary_relative_hit_table_cache is relative_cache
    assert state.primary_contribution_cache_signature == ("primary-contrib",)
    assert state.stored_primary_sim_image is not None
    assert state.stored_sim_image is None
    assert state.stored_intersection_cache is None


def test_detector_center_remap_clears_analysis_geometry_cache_only() -> None:
    state = _state()
    primary_cache = state.primary_hit_table_cache
    relative_cache = state.primary_relative_hit_table_cache
    secondary_relative = state.secondary_relative_hit_table_cache
    primary_image = state.stored_primary_sim_image

    invalidate_for_update_action(state, UpdateAction.DETECTOR_CENTER_REMAP)

    assert state.primary_hit_table_cache is primary_cache
    assert state.primary_relative_hit_table_cache is relative_cache
    assert state.secondary_relative_hit_table_cache is secondary_relative
    assert state.stored_primary_sim_image is primary_image
    assert state.stored_sim_image is not None
    assert state.last_analysis_signature is None
    assert state.last_analysis_cache_sig is None
    assert state.last_caked_image_unscaled is None
    assert state.last_q_space_image_unscaled is None
    assert state.last_caked_transform_bundle is None
    assert state.ai_cache == {}
    assert state.source_row_snapshots == {}


def test_full_simulation_uses_broad_invalidation() -> None:
    state = _state()

    invalidate_for_update_action(state, UpdateAction.FULL_SIMULATION)

    assert state.last_sim_signature is None
    assert state.last_simulation_signature is None
    assert state.last_dependency_signatures is None
    assert state.stored_primary_sim_image is None
    assert state.stored_secondary_sim_image is None
    assert state.stored_sim_image is None
    assert state.primary_hit_table_cache == {}
    assert state.primary_relative_hit_table_cache == {}
    assert state.secondary_relative_hit_table_cache == []
    assert state.last_analysis_signature is None
    assert state.peak_records == []


def test_primary_prune_fill_preserves_existing_contribution_cache() -> None:
    state = _state()
    primary_cache = state.primary_hit_table_cache
    relative_cache = state.primary_relative_hit_table_cache
    combined_image = state.stored_sim_image

    invalidate_for_update_action(state, UpdateAction.PRIMARY_PRUNE_FILL)

    assert state.primary_hit_table_cache is primary_cache
    assert state.primary_relative_hit_table_cache is relative_cache
    assert state.stored_sim_image is combined_image
