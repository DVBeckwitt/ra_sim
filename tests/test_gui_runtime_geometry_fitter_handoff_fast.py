from types import SimpleNamespace

from ra_sim.gui.runtime_invalidation import invalidate_for_update_action
from ra_sim.gui.runtime_update_dependencies import UpdateAction


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        stored_sim_image=object(),
        stored_max_positions_local=[(1.0, 2.0)],
        stored_source_reflection_indices_local=[3],
        stored_peak_table_lattice=("primary",),
        stored_intersection_cache=["combined-intersection"],
        stored_primary_intersection_cache=["primary-intersection"],
        stored_secondary_intersection_cache=["secondary-intersection"],
        stored_primary_intersection_cache_signature=("primary-intersection-sig",),
        stored_secondary_intersection_cache_signature=("secondary-intersection-sig",),
        stored_hit_table_signature=("hit-table",),
        stored_q_group_content_signature=("q-group-content",),
        geometry_q_group_entries_cache_signature=("q-group-cache",),
        geometry_q_group_entries_cache=[{"q_group_key": ("q_group", "primary", 1, 5)}],
        source_row_snapshots={
            0: {
                "row_content_signature": ("q-group-content",),
                "rows": [{"source_row_index": 11}],
            }
        },
        manual_pick_cache_signature=("manual",),
        manual_pick_cache_data={"projected": [1, 2, 3]},
        hkl_pick_simulation_points_payload_cache={"payload": True},
        disabled_qr_sets={("primary", 1)},
        disabled_qz_sections={("primary", 1, 0)},
        pending_legacy_disabled_qz_sections={("primary", 2, 1)},
        refresh_requested=True,
        last_analysis_signature=("analysis",),
        last_caked_geometry_signature=("caked",),
        last_q_space_payload_signature=("q-space",),
        last_detector_caked_signature=("detector-caked",),
        detector_projection_cache={"projection": True},
        caked_projection_cache={"caked": True},
    )


def _mask_snapshot(state: SimpleNamespace) -> tuple[set[object], set[object], set[object]]:
    return (
        set(state.disabled_qr_sets),
        set(state.disabled_qz_sections),
        set(state.pending_legacy_disabled_qz_sections),
    )


def _handoff_snapshot(state: SimpleNamespace) -> dict[str, object]:
    return {
        "geometry_q_group_entries_cache_signature": (
            state.geometry_q_group_entries_cache_signature
        ),
        "geometry_q_group_entries_cache": list(state.geometry_q_group_entries_cache),
        "source_row_snapshots": dict(state.source_row_snapshots),
        "stored_intersection_cache": state.stored_intersection_cache,
        "stored_primary_intersection_cache": state.stored_primary_intersection_cache,
        "stored_secondary_intersection_cache": state.stored_secondary_intersection_cache,
        "manual_pick_cache_signature": state.manual_pick_cache_signature,
        "manual_pick_cache_data": dict(state.manual_pick_cache_data),
        "hkl_pick_simulation_points_payload_cache": dict(
            state.hkl_pick_simulation_points_payload_cache
        ),
    }


def test_fast_paths_preserve_qr_selector_masks_for_geometry_handoff() -> None:
    cases = (
        (UpdateAction.DISPLAY_ONLY, {}),
        (UpdateAction.COMBINE_ONLY, {}),
        (UpdateAction.ANALYSIS_ONLY, {}),
        (
            UpdateAction.PRIMARY_PRUNE_REUSE,
            {
                "q_group_content_signature_changed": False,
                "hit_table_signature_changed": False,
            },
        ),
        (
            UpdateAction.PRIMARY_PRUNE_FILL,
            {
                "q_group_content_signature_changed": True,
                "hit_table_signature_changed": True,
            },
        ),
        (
            UpdateAction.DETECTOR_CENTER_REMAP,
            {
                "detector_geometry_changed": True,
                "hit_table_signature_changed": False,
            },
        ),
    )

    for action, kwargs in cases:
        state = _state()
        masks = _mask_snapshot(state)

        invalidate_for_update_action(state, action, **kwargs)

        assert _mask_snapshot(state) == masks


def test_display_combine_analysis_keep_geometry_fitter_handoff_cache_fast() -> None:
    for action in (
        UpdateAction.DISPLAY_ONLY,
        UpdateAction.COMBINE_ONLY,
        UpdateAction.ANALYSIS_ONLY,
    ):
        state = _state()
        snapshot = _handoff_snapshot(state)

        policy = invalidate_for_update_action(state, action)

        assert policy.retain_geometry_q_group_entries is True
        assert policy.retain_source_row_snapshots is True
        assert policy.retain_intersection_caches is True
        assert policy.retain_manual_pick_cache is True
        assert _handoff_snapshot(state) == snapshot


def test_prune_reuse_unchanged_qgroup_keeps_handoff_cache_fast() -> None:
    state = _state()
    snapshot = _handoff_snapshot(state)

    policy = invalidate_for_update_action(
        state,
        UpdateAction.PRIMARY_PRUNE_REUSE,
        q_group_content_signature_changed=False,
        hit_table_signature_changed=False,
    )

    assert policy.require_q_group_refresh_after_apply is False
    assert policy.retain_geometry_q_group_entries is True
    assert policy.retain_source_row_snapshots is True
    assert policy.retain_intersection_caches is True
    assert policy.retain_manual_pick_cache is True
    assert _handoff_snapshot(state) == snapshot


def test_prune_reuse_retains_qr_selector_entries_when_q_group_content_unchanged() -> None:
    state = _state()
    entries = list(state.geometry_q_group_entries_cache)
    signature = state.geometry_q_group_entries_cache_signature

    policy = invalidate_for_update_action(
        state,
        UpdateAction.PRIMARY_PRUNE_REUSE,
        q_group_content_signature_changed=False,
        hit_table_signature_changed=False,
    )

    assert policy.retain_geometry_q_group_entries is True
    assert policy.require_q_group_refresh_after_apply is False
    assert state.geometry_q_group_entries_cache == entries
    assert state.geometry_q_group_entries_cache_signature == signature


def test_prune_reuse_defers_qr_selector_replacement_when_q_group_content_changes() -> None:
    state = _state()
    entries = list(state.geometry_q_group_entries_cache)
    signature = state.geometry_q_group_entries_cache_signature

    pre_apply_policy = invalidate_for_update_action(
        state,
        UpdateAction.PRIMARY_PRUNE_REUSE,
        q_group_content_signature_changed=False,
        hit_table_signature_changed=True,
    )

    assert pre_apply_policy.retain_geometry_q_group_entries is True
    assert pre_apply_policy.retain_source_row_snapshots is False
    assert state.geometry_q_group_entries_cache == entries
    assert state.geometry_q_group_entries_cache_signature == signature
    assert state.source_row_snapshots == {}

    post_apply_policy = invalidate_for_update_action(
        state,
        UpdateAction.PRIMARY_PRUNE_REUSE,
        q_group_content_signature_changed=True,
        hit_table_signature_changed=False,
    )

    assert post_apply_policy.require_q_group_refresh_after_apply is True
    assert state.geometry_q_group_entries_cache == []
    assert state.geometry_q_group_entries_cache_signature is None


def test_prune_fill_defers_qgroup_refresh_until_rows_exist_fast() -> None:
    state = _state()
    masks = _mask_snapshot(state)
    entries = list(state.geometry_q_group_entries_cache)

    policy = invalidate_for_update_action(
        state,
        UpdateAction.PRIMARY_PRUNE_FILL,
        q_group_content_signature_changed=True,
        hit_table_signature_changed=True,
    )

    assert policy.require_q_group_refresh_after_apply is True
    assert policy.defer_q_group_refresh_until_rows_available is True
    assert state.geometry_q_group_entries_cache == entries
    assert state.source_row_snapshots == {}
    assert state.stored_intersection_cache is None
    assert state.manual_pick_cache_signature is None
    assert _mask_snapshot(state) == masks


def test_prune_fill_does_not_consume_q_group_refresh_before_rows_apply() -> None:
    state = _state()

    policy = invalidate_for_update_action(
        state,
        UpdateAction.PRIMARY_PRUNE_FILL,
        q_group_content_signature_changed=True,
        hit_table_signature_changed=True,
    )

    assert policy.defer_q_group_refresh_until_rows_available is True
    assert state.refresh_requested is True
    assert state.geometry_q_group_entries_cache


def test_prune_fill_refreshes_q_group_entries_after_missing_rows_apply() -> None:
    state = _state()
    masks = _mask_snapshot(state)

    policy = invalidate_for_update_action(
        state,
        UpdateAction.PRIMARY_PRUNE_REUSE,
        q_group_content_signature_changed=True,
        hit_table_signature_changed=False,
    )
    state.geometry_q_group_entries_cache_signature = ("new-q-group-cache",)
    state.geometry_q_group_entries_cache = [{"q_group_key": ("q_group", "primary", 2, 8)}]

    assert policy.require_q_group_refresh_after_apply is True
    assert state.geometry_q_group_entries_cache_signature == ("new-q-group-cache",)
    assert state.geometry_q_group_entries_cache == [
        {"q_group_key": ("q_group", "primary", 2, 8)}
    ]
    assert _mask_snapshot(state) == masks


def test_prune_fast_paths_do_not_clear_qr_disabled_masks() -> None:
    for action, kwargs in (
        (
            UpdateAction.PRIMARY_PRUNE_REUSE,
            {
                "q_group_content_signature_changed": True,
                "hit_table_signature_changed": True,
            },
        ),
        (
            UpdateAction.PRIMARY_PRUNE_FILL,
            {
                "q_group_content_signature_changed": True,
                "hit_table_signature_changed": True,
            },
        ),
    ):
        state = _state()
        masks = _mask_snapshot(state)

        invalidate_for_update_action(state, action, **kwargs)

        assert _mask_snapshot(state) == masks


def test_prune_reuse_clears_source_rows_when_hit_table_identity_changes() -> None:
    state = _state()
    entries = list(state.geometry_q_group_entries_cache)

    policy = invalidate_for_update_action(
        state,
        UpdateAction.PRIMARY_PRUNE_REUSE,
        q_group_content_signature_changed=False,
        hit_table_signature_changed=True,
    )

    assert policy.retain_geometry_q_group_entries is True
    assert policy.retain_source_row_snapshots is False
    assert state.geometry_q_group_entries_cache == entries
    assert state.source_row_snapshots == {}


def test_center_remap_invalidates_caked_projection_cache_fast() -> None:
    state = _state()
    masks = _mask_snapshot(state)

    policy = invalidate_for_update_action(
        state,
        UpdateAction.DETECTOR_CENTER_REMAP,
        detector_geometry_changed=True,
    )

    assert policy.retain_geometry_q_group_entries is True
    assert policy.retain_source_row_snapshots is True
    assert policy.retain_manual_pick_cache is False
    assert state.detector_projection_cache is None
    assert state.caked_projection_cache is None
    assert state.source_row_snapshots
    assert _mask_snapshot(state) == masks


def test_full_simulation_retains_qr_masks_but_clears_stale_source_rows_fast() -> None:
    state = _state()
    masks = _mask_snapshot(state)

    policy = invalidate_for_update_action(
        state,
        UpdateAction.FULL_SIMULATION,
        physics_signature_changed=True,
        hit_table_signature_changed=True,
        q_group_content_signature_changed=True,
        detector_geometry_changed=True,
    )

    assert policy.retain_geometry_q_group_masks is True
    assert policy.retain_source_row_snapshots is False
    assert state.source_row_snapshots == {}
    assert state.geometry_q_group_entries_cache_signature is None
    assert state.geometry_q_group_entries_cache == []
    assert state.stored_intersection_cache is None
    assert state.manual_pick_cache_signature is None
    assert _mask_snapshot(state) == masks
