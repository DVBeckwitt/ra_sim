from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ra_sim.fitting.geometry_objective_cache import (
    GeometryObjectiveSignature,
    center_remap_residual_shape_and_order_match,
    geometry_objective_cache_decision,
    geometry_objective_cache_trace_payload,
)
from ra_sim.gui.runtime_invalidation import invalidate_for_update_action
from ra_sim.gui.runtime_update_dependencies import UpdateAction


def _branch_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("q_group_key"),
        tuple(row.get("hkl", ())),
        row.get("source_branch_index"),
        row.get("source_row_index"),
        row.get("source_table_index"),
    )


def _manual_row(
    pair_id: str,
    *,
    source_row_index: int,
    refined_offset: float = 0.0,
) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "q_group_key": ("q_group", "primary", 1, 5),
        "hkl": (1, 0, 0),
        "source_branch_index": int(source_row_index % 2),
        "source_row_index": int(source_row_index),
        "source_table_index": 0,
        "manual_x": float(10 + source_row_index),
        "manual_y": float(20 + source_row_index),
        "refined_x": float(10.5 + source_row_index + refined_offset),
        "refined_y": float(20.5 + source_row_index + refined_offset),
        "refined_sim_caked_x": float(7.0 + source_row_index + refined_offset),
        "refined_sim_caked_y": float(4.0 + source_row_index + refined_offset),
    }


def _state() -> SimpleNamespace:
    q_rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "hkl": (1, 0, 0),
            "source_branch_index": 1,
            "source_row_index": 11,
            "source_table_index": 0,
            "sim_col": 41.0,
            "sim_row": 42.0,
        },
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "hkl": (1, 0, 0),
            "source_branch_index": 0,
            "source_row_index": 12,
            "source_table_index": 0,
            "sim_col": 43.0,
            "sim_row": 44.0,
        },
    ]
    return SimpleNamespace(
        stored_sim_image=object(),
        stored_max_positions_local=[(41.0, 42.0), (43.0, 44.0)],
        stored_source_reflection_indices_local=[0, 1],
        stored_peak_table_lattice=("primary",),
        stored_intersection_cache=[{"source_row_index": 11}, {"source_row_index": 12}],
        stored_primary_intersection_cache=[
            {"native_col": 41.0, "native_row": 42.0, "source_row_index": 11},
            {"native_col": 43.0, "native_row": 44.0, "source_row_index": 12},
        ],
        stored_secondary_intersection_cache=None,
        stored_primary_intersection_cache_signature=("primary-intersection",),
        stored_secondary_intersection_cache_signature=None,
        stored_hit_table_signature=("hit-table", "identity-a"),
        stored_q_group_content_signature=("q-content", "identity-a"),
        geometry_q_group_entries_cache_signature=("q-entries", "identity-a"),
        geometry_q_group_entries_cache=list(q_rows),
        source_row_snapshots={
            0: {
                "hit_table_signature": ("hit-table", "identity-a"),
                "row_content_signature": ("q-content", "identity-a"),
                "rows": list(q_rows),
            }
        },
        manual_selected_peak_rows=[
            _manual_row("pair-1", source_row_index=11),
            _manual_row("pair-2", source_row_index=12),
        ],
        manual_pick_cache_signature=("manual", "identity-a"),
        manual_pick_cache_data={"projected": [(41.0, 42.0), (43.0, 44.0)]},
        hkl_pick_simulation_points_payload_cache={"payload": True},
        disabled_qr_sets={("primary", 1)},
        disabled_qz_sections={("primary", 1, 0)},
        pending_legacy_disabled_qz_sections={("primary", 2, 1)},
        refresh_requested=False,
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


def _branch_keys(state: SimpleNamespace) -> tuple[tuple[object, ...], ...]:
    return tuple(_branch_key(row) for row in state.geometry_q_group_entries_cache)


def _manual_selection_signature(state: SimpleNamespace) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            row.get("pair_id"),
            row.get("manual_x"),
            row.get("manual_y"),
            row.get("source_row_index"),
            row.get("source_branch_index"),
        )
        for row in state.manual_selected_peak_rows
    )


def _refined_peak_signature(state: SimpleNamespace) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            row.get("pair_id"),
            row.get("refined_x"),
            row.get("refined_y"),
            row.get("refined_sim_caked_x"),
            row.get("refined_sim_caked_y"),
        )
        for row in state.manual_selected_peak_rows
    )


def _source_row_signature(state: SimpleNamespace) -> tuple[object, ...]:
    rows = state.source_row_snapshots.get(0, {}).get("rows", ())
    return tuple(_branch_key(row) for row in rows)


def _point_provider_report(state: SimpleNamespace) -> dict[str, int]:
    manual_keys = {_branch_key(row) for row in state.manual_selected_peak_rows}
    provider_keys = {_branch_key(row) for row in state.geometry_q_group_entries_cache}
    matched = manual_keys & provider_keys
    return {
        "manual_picker_pair_count": int(len(manual_keys)),
        "point_provider_pair_count": int(len(matched)),
        "fallback_pair_count": int(len(manual_keys - provider_keys)),
    }


def _objective_signature(
    state: SimpleNamespace,
    *,
    center: tuple[float, float] = (100.0, 101.0),
) -> GeometryObjectiveSignature:
    report = _point_provider_report(state)
    return GeometryObjectiveSignature(
        physics_sig=("physics-a",),
        detector_center_sig=tuple(float(v) for v in center),
        dataset_sig=("dataset-a",),
        point_provider_sig=(
            "synthetic-point-provider",
            report["point_provider_pair_count"],
            report["fallback_pair_count"],
        ),
        qr_branch_identity_sig=_branch_keys(state),
        source_row_identity_sig=_source_row_signature(state),
        manual_selection_sig=_manual_selection_signature(state),
        refined_peak_sig=_refined_peak_signature(state),
        objective_mode_sig=("dynamic-refined",),
        active_fit_parameter_sig=("center_x", "center_y"),
    )


def _finite_dry_run_residual(state: SimpleNamespace) -> np.ndarray:
    report = _point_provider_report(state)
    if report["fallback_pair_count"] != 0:
        return np.asarray([np.nan], dtype=np.float64)
    residuals: list[float] = []
    for row in state.manual_selected_peak_rows:
        residuals.extend(
            [
                float(row["refined_x"]) - float(row["manual_x"]),
                float(row["refined_y"]) - float(row["manual_y"]),
            ]
        )
    return np.asarray(residuals, dtype=np.float64)


def _apply_action(
    state: SimpleNamespace,
    action: UpdateAction,
    **kwargs: object,
) -> dict[str, object]:
    before_masks = _mask_snapshot(state)
    before_branches = _branch_keys(state)
    before_source_rows = dict(state.source_row_snapshots)
    before_manual_cache = state.manual_pick_cache_signature

    policy = invalidate_for_update_action(state, action, **kwargs)
    if policy.require_q_group_refresh_after_apply:
        state.refresh_requested = True

    projection_invalidated = bool(
        before_manual_cache is not None and state.manual_pick_cache_signature is None
    )
    handoff_valid = bool(
        not state.refresh_requested
        and state.geometry_q_group_entries_cache
        and state.source_row_snapshots
        and state.manual_pick_cache_signature is not None
    )
    return {
        "update_action": action.value,
        "qr_masks_unchanged": _mask_snapshot(state) == before_masks,
        "qr_selector_branch_identity_retained": _branch_keys(state) == before_branches,
        "source_row_snapshots_retained": state.source_row_snapshots == before_source_rows,
        "qr_selector_entries_retained": bool(policy.retain_geometry_q_group_entries),
        "qr_selector_refresh_deferred": bool(
            policy.defer_q_group_refresh_until_rows_available
        ),
        "q_group_content_signature_changed": bool(
            kwargs.get("q_group_content_signature_changed", False)
        ),
        "detector_projection_cache_refreshed": projection_invalidated,
        "caked_projection_cache_invalidated": state.caked_projection_cache is None,
        "geometry_fitter_handoff_valid": handoff_valid,
        "policy_reason": policy.reason,
    }


def _evaluate_objective_cache(
    cache: dict[str, GeometryObjectiveSignature],
    signature: GeometryObjectiveSignature,
) -> dict[str, object]:
    decision = geometry_objective_cache_decision(
        cache.get("signature"),
        signature,
        exact_center_remap_cache_available=True,
    )
    if not decision.can_reuse:
        cache["signature"] = signature
    return geometry_objective_cache_trace_payload(
        decision,
        residual_component_count=4,
    )


def test_fast_path_sequence_preserves_qr_selector_geometry_fitter_handoff() -> None:
    state = _state()
    masks = _mask_snapshot(state)
    branches = _branch_keys(state)
    source_sig = _source_row_signature(state)
    report = _point_provider_report(state)

    assert report == {
        "manual_picker_pair_count": 2,
        "point_provider_pair_count": 2,
        "fallback_pair_count": 0,
    }

    for action, kwargs in (
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
    ):
        trace = _apply_action(state, action, **kwargs)
        assert trace["qr_masks_unchanged"] is True
        assert trace["geometry_fitter_handoff_valid"] is True

    assert _mask_snapshot(state) == masks
    assert _branch_keys(state) == branches
    assert _source_row_signature(state) == source_sig
    assert _point_provider_report(state)["fallback_pair_count"] == 0
    assert np.all(np.isfinite(_finite_dry_run_residual(state)))


def test_prune_changed_content_defers_handoff_validity_until_qgroup_refresh() -> None:
    state = _state()
    masks = _mask_snapshot(state)
    entries = list(state.geometry_q_group_entries_cache)

    trace = _apply_action(
        state,
        UpdateAction.PRIMARY_PRUNE_FILL,
        q_group_content_signature_changed=True,
        hit_table_signature_changed=True,
    )

    assert _mask_snapshot(state) == masks
    assert state.geometry_q_group_entries_cache == entries
    assert state.source_row_snapshots == {}
    assert state.refresh_requested is True
    assert trace["qr_selector_refresh_deferred"] is True
    assert trace["geometry_fitter_handoff_valid"] is False


def test_detector_center_remap_invalidates_projection_but_preserves_branch_identity() -> None:
    state = _state()
    masks = _mask_snapshot(state)
    branches = _branch_keys(state)
    source_sig = _source_row_signature(state)

    trace = _apply_action(
        state,
        UpdateAction.DETECTOR_CENTER_REMAP,
        detector_geometry_changed=True,
        hit_table_signature_changed=False,
    )

    assert _mask_snapshot(state) == masks
    assert _branch_keys(state) == branches
    assert _source_row_signature(state) == source_sig
    assert state.detector_projection_cache is None
    assert state.caked_projection_cache is None
    assert trace["qr_selector_branch_identity_retained"] is True
    assert trace["detector_projection_cache_refreshed"] is True
    assert trace["caked_projection_cache_invalidated"] is True
    assert trace["geometry_fitter_handoff_valid"] is False


def test_geometry_objective_cache_accepts_center_only_after_fast_path_sequence() -> None:
    state = _state()
    for action in (UpdateAction.DISPLAY_ONLY, UpdateAction.COMBINE_ONLY):
        _apply_action(state, action)
    assert _point_provider_report(state)["fallback_pair_count"] == 0

    cache: dict[str, GeometryObjectiveSignature] = {}
    first = _evaluate_objective_cache(cache, _objective_signature(state))
    second = _evaluate_objective_cache(
        cache,
        _objective_signature(state, center=(102.0, 101.0)),
    )

    assert first["objective_cache_reject_reason"] == "initial_evaluation"
    assert second["objective_cache_mode"] == "center_remap"
    assert second["objective_cache_hit"] is True
    assert second["objective_signature_changed_fields"] == ["detector_center_sig"]
    assert center_remap_residual_shape_and_order_match(
        ("branch[0].dtheta", "branch[0].dphi", "branch[1].dtheta", "branch[1].dphi"),
        ("branch[0].dtheta", "branch[0].dphi", "branch[1].dtheta", "branch[1].dphi"),
    )


def test_geometry_objective_cache_rejects_after_manual_selection_change_in_handoff_sequence() -> None:
    state = _state()
    cache: dict[str, GeometryObjectiveSignature] = {}

    _evaluate_objective_cache(cache, _objective_signature(state))
    center_trace = _evaluate_objective_cache(
        cache,
        _objective_signature(state, center=(102.0, 101.0)),
    )
    state.manual_selected_peak_rows[0]["manual_x"] = 999.0
    reject_trace = _evaluate_objective_cache(
        cache,
        _objective_signature(state, center=(103.0, 101.0)),
    )

    assert center_trace["objective_cache_mode"] == "center_remap"
    assert reject_trace["objective_cache_mode"] == "full_simulation"
    assert reject_trace["objective_cache_reject_reason"] == "manual_selection_changed"
    assert reject_trace["objective_process_peaks_called"] is True


def test_geometry_objective_cache_rejects_after_refined_peak_change_in_handoff_sequence() -> None:
    state = _state()
    cache: dict[str, GeometryObjectiveSignature] = {}

    _evaluate_objective_cache(cache, _objective_signature(state))
    center_trace = _evaluate_objective_cache(
        cache,
        _objective_signature(state, center=(102.0, 101.0)),
    )
    state.manual_selected_peak_rows[1]["refined_sim_caked_x"] = 123.0
    reject_trace = _evaluate_objective_cache(
        cache,
        _objective_signature(state, center=(103.0, 101.0)),
    )

    assert center_trace["objective_cache_mode"] == "center_remap"
    assert reject_trace["objective_cache_mode"] == "full_simulation"
    assert reject_trace["objective_cache_reject_reason"] == "refined_peak_changed"
    assert reject_trace["objective_process_peaks_called"] is True
