from __future__ import annotations

from ra_sim.gui.runtime_qr_selector_cache_policy import (
    QrSelectorCachePolicy,
    qr_selector_cache_policy_for_update_action,
)
from ra_sim.gui.runtime_update_dependencies import UpdateAction


def _policy(action: object, **overrides: bool) -> QrSelectorCachePolicy:
    flags = {
        "physics_signature_changed": False,
        "hit_table_signature_changed": False,
        "q_group_content_signature_changed": False,
        "detector_geometry_changed": False,
    }
    flags.update(overrides)
    return qr_selector_cache_policy_for_update_action(action, **flags)


def _assert_retains_all(policy: QrSelectorCachePolicy) -> None:
    assert policy.retain_geometry_q_group_entries is True
    assert policy.retain_geometry_q_group_masks is True
    assert policy.retain_source_row_snapshots is True
    assert policy.retain_intersection_caches is True
    assert policy.retain_manual_pick_cache is True
    assert policy.require_q_group_refresh_after_apply is False
    assert policy.defer_q_group_refresh_until_rows_available is False


def test_display_only_retains_qr_selector_cache() -> None:
    policy = _policy(UpdateAction.DISPLAY_ONLY)

    _assert_retains_all(policy)
    assert policy.reason == "display_only"


def test_combine_only_retains_qr_selector_cache() -> None:
    policy = _policy(UpdateAction.COMBINE_ONLY)

    _assert_retains_all(policy)
    assert policy.reason == "combine_only"


def test_analysis_only_retains_qr_selector_cache() -> None:
    policy = _policy(UpdateAction.ANALYSIS_ONLY)

    _assert_retains_all(policy)
    assert policy.reason == "analysis_only"


def test_prune_reuse_retains_q_group_entries_when_content_signature_unchanged() -> None:
    policy = _policy(UpdateAction.PRIMARY_PRUNE_REUSE)

    assert policy.retain_geometry_q_group_entries is True
    assert policy.retain_geometry_q_group_masks is True
    assert policy.retain_source_row_snapshots is True
    assert policy.retain_intersection_caches is True
    assert policy.retain_manual_pick_cache is True
    assert policy.require_q_group_refresh_after_apply is False
    assert policy.defer_q_group_refresh_until_rows_available is False
    assert policy.reason == "primary_prune_reuse_content_unchanged"


def test_prune_reuse_requests_refresh_when_content_signature_changes() -> None:
    policy = _policy(
        UpdateAction.PRIMARY_PRUNE_REUSE,
        q_group_content_signature_changed=True,
    )

    assert policy.retain_geometry_q_group_entries is False
    assert policy.retain_geometry_q_group_masks is True
    assert policy.retain_source_row_snapshots is True
    assert policy.retain_intersection_caches is True
    assert policy.retain_manual_pick_cache is False
    assert policy.require_q_group_refresh_after_apply is True
    assert policy.defer_q_group_refresh_until_rows_available is False
    assert policy.reason == "primary_prune_reuse_content_changed"


def test_prune_fill_defers_refresh_until_rows_are_available() -> None:
    policy = _policy(
        UpdateAction.PRIMARY_PRUNE_FILL,
        hit_table_signature_changed=True,
        q_group_content_signature_changed=True,
    )

    assert policy.retain_geometry_q_group_entries is True
    assert policy.retain_geometry_q_group_masks is True
    assert policy.retain_source_row_snapshots is False
    assert policy.retain_intersection_caches is False
    assert policy.retain_manual_pick_cache is False
    assert policy.require_q_group_refresh_after_apply is True
    assert policy.defer_q_group_refresh_until_rows_available is True
    assert policy.reason == "primary_prune_fill_defer_until_rows_available"


def test_detector_center_remap_retains_branch_identity_but_refreshes_projection_geometry() -> None:
    policy = _policy(
        UpdateAction.DETECTOR_CENTER_REMAP,
        hit_table_signature_changed=True,
        detector_geometry_changed=True,
    )

    assert policy.retain_geometry_q_group_entries is True
    assert policy.retain_geometry_q_group_masks is True
    assert policy.retain_source_row_snapshots is True
    assert policy.retain_intersection_caches is False
    assert policy.retain_manual_pick_cache is False
    assert policy.require_q_group_refresh_after_apply is True
    assert policy.defer_q_group_refresh_until_rows_available is False
    assert policy.reason == "detector_center_remap_projection_refresh"


def test_full_simulation_retains_masks_but_does_not_reuse_stale_rows() -> None:
    policy = _policy(
        UpdateAction.FULL_SIMULATION,
        physics_signature_changed=True,
        hit_table_signature_changed=True,
        q_group_content_signature_changed=True,
        detector_geometry_changed=True,
    )

    assert policy.retain_geometry_q_group_entries is False
    assert policy.retain_geometry_q_group_masks is True
    assert policy.retain_source_row_snapshots is False
    assert policy.retain_intersection_caches is False
    assert policy.retain_manual_pick_cache is False
    assert policy.require_q_group_refresh_after_apply is True
    assert policy.defer_q_group_refresh_until_rows_available is True
    assert policy.reason == "full_simulation_stale_rows"


def test_theta_physics_full_simulation_refreshes_qr_rows_but_retains_masks() -> None:
    policy = _policy(
        UpdateAction.FULL_SIMULATION,
        physics_signature_changed=True,
        hit_table_signature_changed=True,
        q_group_content_signature_changed=True,
    )

    assert policy.retain_geometry_q_group_entries is False
    assert policy.retain_geometry_q_group_masks is True
    assert policy.retain_source_row_snapshots is False
    assert policy.retain_intersection_caches is False
    assert policy.retain_manual_pick_cache is False
    assert policy.require_q_group_refresh_after_apply is True
    assert policy.defer_q_group_refresh_until_rows_available is True


def test_qr_selector_masks_are_always_retained_by_cache_policy() -> None:
    actions: list[object] = list(UpdateAction) + ["unknown-action"]

    for action in actions:
        policy = _policy(
            action,
            physics_signature_changed=True,
            hit_table_signature_changed=True,
            q_group_content_signature_changed=True,
            detector_geometry_changed=True,
        )

        assert policy.retain_geometry_q_group_masks is True
