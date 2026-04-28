"""Pure QR selector cache retention policy for GUI update actions."""

from __future__ import annotations

from dataclasses import dataclass

from ra_sim.gui.runtime_update_dependencies import UpdateAction


@dataclass(frozen=True)
class QrSelectorCachePolicy:
    retain_geometry_q_group_entries: bool
    retain_geometry_q_group_masks: bool
    retain_source_row_snapshots: bool
    retain_intersection_caches: bool
    retain_manual_pick_cache: bool
    require_q_group_refresh_after_apply: bool
    defer_q_group_refresh_until_rows_available: bool
    reason: str


def _policy(
    *,
    retain_geometry_q_group_entries: bool,
    retain_source_row_snapshots: bool,
    retain_intersection_caches: bool,
    retain_manual_pick_cache: bool,
    require_q_group_refresh_after_apply: bool,
    defer_q_group_refresh_until_rows_available: bool,
    reason: str,
) -> QrSelectorCachePolicy:
    return QrSelectorCachePolicy(
        retain_geometry_q_group_entries=bool(retain_geometry_q_group_entries),
        retain_geometry_q_group_masks=True,
        retain_source_row_snapshots=bool(retain_source_row_snapshots),
        retain_intersection_caches=bool(retain_intersection_caches),
        retain_manual_pick_cache=bool(retain_manual_pick_cache),
        require_q_group_refresh_after_apply=bool(require_q_group_refresh_after_apply),
        defer_q_group_refresh_until_rows_available=bool(
            defer_q_group_refresh_until_rows_available
        ),
        reason=str(reason),
    )


def _retain_all(reason: str) -> QrSelectorCachePolicy:
    return _policy(
        retain_geometry_q_group_entries=True,
        retain_source_row_snapshots=True,
        retain_intersection_caches=True,
        retain_manual_pick_cache=True,
        require_q_group_refresh_after_apply=False,
        defer_q_group_refresh_until_rows_available=False,
        reason=reason,
    )


def _fail_closed(reason: str) -> QrSelectorCachePolicy:
    return _policy(
        retain_geometry_q_group_entries=False,
        retain_source_row_snapshots=False,
        retain_intersection_caches=False,
        retain_manual_pick_cache=False,
        require_q_group_refresh_after_apply=True,
        defer_q_group_refresh_until_rows_available=True,
        reason=f"fail_closed:{reason}",
    )


def _coerce_update_action(action: object) -> UpdateAction | None:
    if isinstance(action, UpdateAction):
        return action
    try:
        return UpdateAction(str(action))
    except Exception:
        return None


def _any_changed(
    *,
    physics_signature_changed: bool,
    hit_table_signature_changed: bool,
    q_group_content_signature_changed: bool,
    detector_geometry_changed: bool,
) -> bool:
    return bool(
        physics_signature_changed
        or hit_table_signature_changed
        or q_group_content_signature_changed
        or detector_geometry_changed
    )


def qr_selector_cache_policy_for_update_action(
    action: UpdateAction,
    *,
    physics_signature_changed: bool,
    hit_table_signature_changed: bool,
    q_group_content_signature_changed: bool,
    detector_geometry_changed: bool,
) -> QrSelectorCachePolicy:
    """Return the conservative QR selector cache policy for one update action.

    User-controlled Qr/Qz masks are always retained by this cache policy. They are
    allowed to change only through explicit selector actions, state load, or
    explicit reset code outside image-cache invalidation.
    """

    normalized_action = _coerce_update_action(action)
    if normalized_action is None:
        return _fail_closed("unknown_update_action")

    physics_changed = bool(physics_signature_changed)
    hit_changed = bool(hit_table_signature_changed)
    q_content_changed = bool(q_group_content_signature_changed)
    detector_changed = bool(detector_geometry_changed)

    if normalized_action is UpdateAction.DISPLAY_ONLY:
        if _any_changed(
            physics_signature_changed=physics_changed,
            hit_table_signature_changed=hit_changed,
            q_group_content_signature_changed=q_content_changed,
            detector_geometry_changed=detector_changed,
        ):
            return _fail_closed("display_only_with_cache_identity_change")
        return _retain_all("display_only")

    if normalized_action is UpdateAction.COMBINE_ONLY:
        if _any_changed(
            physics_signature_changed=physics_changed,
            hit_table_signature_changed=hit_changed,
            q_group_content_signature_changed=q_content_changed,
            detector_geometry_changed=detector_changed,
        ):
            return _fail_closed("combine_only_with_cache_identity_change")
        return _retain_all("combine_only")

    if normalized_action is UpdateAction.ANALYSIS_ONLY:
        if _any_changed(
            physics_signature_changed=physics_changed,
            hit_table_signature_changed=hit_changed,
            q_group_content_signature_changed=q_content_changed,
            detector_geometry_changed=detector_changed,
        ):
            return _fail_closed("analysis_only_with_cache_identity_change")
        return _retain_all("analysis_only")

    if normalized_action is UpdateAction.PRIMARY_PRUNE_REUSE:
        if physics_changed or detector_changed:
            return _fail_closed("primary_prune_reuse_with_physics_or_detector_change")
        return _policy(
            retain_geometry_q_group_entries=not q_content_changed,
            retain_source_row_snapshots=not hit_changed,
            retain_intersection_caches=not hit_changed,
            retain_manual_pick_cache=not (q_content_changed or hit_changed),
            require_q_group_refresh_after_apply=q_content_changed,
            defer_q_group_refresh_until_rows_available=False,
            reason=(
                "primary_prune_reuse_content_changed"
                if q_content_changed
                else "primary_prune_reuse_content_unchanged"
            ),
        )

    if normalized_action is UpdateAction.PRIMARY_PRUNE_FILL:
        if physics_changed or detector_changed:
            return _fail_closed("primary_prune_fill_with_physics_or_detector_change")
        return _policy(
            retain_geometry_q_group_entries=True,
            retain_source_row_snapshots=not hit_changed,
            retain_intersection_caches=not hit_changed,
            retain_manual_pick_cache=not (q_content_changed or hit_changed),
            require_q_group_refresh_after_apply=True,
            defer_q_group_refresh_until_rows_available=True,
            reason="primary_prune_fill_defer_until_rows_available",
        )

    if normalized_action is UpdateAction.DETECTOR_CENTER_REMAP:
        if physics_changed or q_content_changed:
            return _fail_closed("detector_center_remap_with_physics_or_content_change")
        return _policy(
            retain_geometry_q_group_entries=True,
            retain_source_row_snapshots=True,
            retain_intersection_caches=not detector_changed,
            retain_manual_pick_cache=not detector_changed,
            require_q_group_refresh_after_apply=detector_changed or hit_changed,
            defer_q_group_refresh_until_rows_available=False,
            reason=(
                "detector_center_remap_projection_refresh"
                if detector_changed or hit_changed
                else "detector_center_remap_identity_retained"
            ),
        )

    if normalized_action is UpdateAction.HIT_TABLE_REFRESH:
        if physics_changed or detector_changed:
            return _fail_closed("hit_table_refresh_with_physics_or_detector_change")
        return _policy(
            retain_geometry_q_group_entries=False,
            retain_source_row_snapshots=False,
            retain_intersection_caches=False,
            retain_manual_pick_cache=False,
            require_q_group_refresh_after_apply=True,
            defer_q_group_refresh_until_rows_available=True,
            reason="hit_table_refresh",
        )

    if normalized_action is UpdateAction.FULL_SIMULATION:
        stale_rows = bool(physics_changed or hit_changed)
        stale_entries = bool(stale_rows or q_content_changed)
        stale_projection = bool(stale_rows or detector_changed)
        refresh_required = bool(stale_entries or stale_rows or stale_projection)
        return _policy(
            retain_geometry_q_group_entries=not stale_entries,
            retain_source_row_snapshots=not stale_rows,
            retain_intersection_caches=not stale_projection,
            retain_manual_pick_cache=not (stale_entries or stale_rows or detector_changed),
            require_q_group_refresh_after_apply=refresh_required,
            defer_q_group_refresh_until_rows_available=refresh_required,
            reason=(
                "full_simulation_stale_rows"
                if refresh_required
                else "full_simulation_identity_retained"
            ),
        )

    return _fail_closed("unhandled_update_action")


__all__ = [
    "QrSelectorCachePolicy",
    "qr_selector_cache_policy_for_update_action",
]
