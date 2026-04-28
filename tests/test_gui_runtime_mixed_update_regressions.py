from __future__ import annotations

from dataclasses import replace

import numpy as np

from ra_sim.fitting.geometry_objective_cache import GeometryObjectiveSignature
from ra_sim.gui.runtime_update_dependencies import (
    RuntimeCacheState,
    SimulationDependencySignatures,
    UpdateAction,
    classify_update,
)
from ra_sim.gui.state import GeometryQGroupState
from tests.test_gui_runtime_geometry_fitter_cache_handoff import (
    _apply_action,
    _branch_keys,
    _evaluate_objective_cache,
    _mask_snapshot,
    _objective_signature,
    _point_provider_report,
    _state,
)
from tests.test_gui_runtime_update_actions import (
    _old_ready_result,
    _prepare_center_remap_runtime,
)


def _signatures() -> SimulationDependencySignatures:
    return SimulationDependencySignatures(
        source_sig=("source", "stable"),
        physics_sig=("physics", "lattice-a", "orientation-a"),
        detector_projection_sig=("projection", "distance-a", "pixel-a"),
        detector_center_sig=("center", 1.0, 1.0),
        primary_filter_sig=("primary-filter", (0, 1)),
        combine_sig=("combine", 1.0, 0.0),
        analysis_geometry_sig=("analysis", "stable"),
        display_sig=("display", 0.0, 1.0),
        hit_table_sig=("hit-table", "a"),
        full_image_sig=("full-image", "a"),
    )


def _assert_full_simulation(
    previous: SimulationDependencySignatures,
    current: SimulationDependencySignatures,
    *,
    prune_cache_mode: str | None = "reuse",
    can_remap_detector_center: bool = True,
) -> None:
    decision = classify_update(
        previous,
        current,
        RuntimeCacheState(
            can_remap_detector_center=can_remap_detector_center,
            prune_cache_mode=prune_cache_mode,
        ),
    )

    assert decision.action is UpdateAction.FULL_SIMULATION
    assert decision.requires_worker is True


def test_center_plus_distance_falls_back_to_full_simulation() -> None:
    previous = _signatures()
    current = replace(
        previous,
        detector_center_sig=("center", 1.5, 1.0),
        detector_projection_sig=("projection", "distance-b", "pixel-a"),
        full_image_sig=("full-image", "center-distance"),
    )

    _assert_full_simulation(previous, current)


def test_center_plus_detector_orientation_falls_back_to_full_simulation() -> None:
    previous = _signatures()
    current = replace(
        previous,
        detector_center_sig=("center", 1.5, 1.0),
        physics_sig=("physics", "lattice-a", "orientation-b"),
        full_image_sig=("full-image", "center-orientation"),
    )

    _assert_full_simulation(previous, current)


def test_center_plus_pixel_size_falls_back_to_full_simulation() -> None:
    previous = _signatures()
    current = replace(
        previous,
        detector_center_sig=("center", 1.5, 1.0),
        detector_projection_sig=("projection", "distance-a", "pixel-b"),
        full_image_sig=("full-image", "center-pixel"),
    )

    _assert_full_simulation(previous, current)


def test_prune_plus_lattice_change_falls_back_to_full_simulation() -> None:
    previous = _signatures()
    current = replace(
        previous,
        physics_sig=("physics", "lattice-b", "orientation-a"),
        primary_filter_sig=("primary-filter", (0,)),
        full_image_sig=("full-image", "prune-lattice"),
    )

    _assert_full_simulation(previous, current)


def test_display_plus_physics_change_runs_full_simulation() -> None:
    previous = _signatures()
    current = replace(
        previous,
        physics_sig=("physics", "lattice-b", "orientation-a"),
        display_sig=("display", 0.0, 2.0),
        full_image_sig=("full-image", "display-physics"),
    )

    _assert_full_simulation(previous, current, prune_cache_mode=None)


def test_prune_reuse_plus_display_change_does_not_clear_qr_masks() -> None:
    state = _state()
    masks = _mask_snapshot(state)
    branches = _branch_keys(state)

    prune_trace = _apply_action(
        state,
        UpdateAction.PRIMARY_PRUNE_REUSE,
        q_group_content_signature_changed=False,
        hit_table_signature_changed=False,
    )
    display_trace = _apply_action(state, UpdateAction.DISPLAY_ONLY)

    assert _mask_snapshot(state) == masks
    assert _branch_keys(state) == branches
    assert prune_trace["geometry_fitter_handoff_valid"] is True
    assert display_trace["geometry_fitter_handoff_valid"] is True


def test_stale_full_worker_result_does_not_overwrite_prune_reuse_qr_state(
    monkeypatch,
) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)
    q_group_state = GeometryQGroupState()
    monkeypatch.setattr(runtime_session, "geometry_q_group_state", q_group_state)
    q_group_state.disabled_qr_sets.add(("primary", 1))
    q_group_state.disabled_qz_sections.add(("primary", 1, 0))
    before_masks = (
        set(q_group_state.disabled_qr_sets),
        set(q_group_state.disabled_qz_sections),
        set(q_group_state.pending_legacy_disabled_qz_sections),
    )
    requested_jobs: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_submit_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)),
        raising=False,
    )

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()
    remapped_image = np.asarray(state.stored_sim_image, dtype=np.float64).copy()

    state.worker_ready_result = _old_ready_result(state, value=99.0)
    runtime_session.display_controls_view_state.simulation_max_var.set(2.0)
    runtime_session.do_update()

    assert state.worker_ready_result is None
    assert not np.array_equal(
        state.stored_sim_image,
        np.full((2, 2), 99.0, dtype=np.float64),
    )
    assert np.array_equal(state.stored_sim_image, remapped_image)
    assert (
        set(q_group_state.disabled_qr_sets),
        set(q_group_state.disabled_qz_sections),
        set(q_group_state.pending_legacy_disabled_qz_sections),
    ) == before_masks


def test_stale_full_worker_result_does_not_restore_invalid_projection_after_center_remap(
    monkeypatch,
) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)
    state.last_caked_image_unscaled = np.ones((2, 2), dtype=np.float64)
    state.geometry_fit_caking_ai_cache = {"old": True}
    state.worker_ready_result = _old_ready_result(state, value=99.0)

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    assert state.worker_ready_result is None
    assert state.last_caked_image_unscaled is None
    assert state.geometry_fit_caking_ai_cache == {}
    assert not np.array_equal(
        state.stored_sim_image,
        np.full((2, 2), 99.0, dtype=np.float64),
    )


def test_stale_primary_fill_result_does_not_overwrite_newer_full_simulation_state(
    monkeypatch,
) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)
    state.worker_ready_result = {
        **_old_ready_result(state, value=77.0),
        "job_kind": "primary_fill",
        "signature": ("stale-primary-fill",),
        "primary_contribution_keys": [999],
        "primary_contribution_cache_signature": ("stale-cache",),
        "primary_source_mode": "miller",
        "active_primary_contribution_keys": [999],
        "primary_hit_tables_raw": [
            np.asarray([[77.0, 9.0, 9.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        ],
        "primary_best_sample_indices": [0],
    }
    requested_jobs: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_submit_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)),
        raising=False,
    )

    runtime_session.corto_detector_var.set(0.75)
    runtime_session.do_update()

    assert state.worker_ready_result is None
    assert requested_jobs
    assert requested_jobs[-1]["job_kind"] == "full"
    assert getattr(state, "primary_active_contribution_keys", None) != [999]


def test_objective_cache_rejects_center_reuse_after_manual_selection_change_in_mixed_sequence() -> None:
    state = _state()
    cache: dict[str, GeometryObjectiveSignature] = {}

    _evaluate_objective_cache(cache, _objective_signature(state))
    _evaluate_objective_cache(cache, _objective_signature(state, center=(102.0, 101.0)))
    state.manual_selected_peak_rows[0]["manual_x"] = 999.0
    trace = _evaluate_objective_cache(
        cache,
        _objective_signature(state, center=(103.0, 101.0)),
    )

    assert trace["objective_cache_mode"] == "full_simulation"
    assert trace["objective_cache_hit"] is False
    assert trace["objective_cache_reject_reason"] == "manual_selection_changed"
    assert trace["objective_process_peaks_called"] is True


def test_objective_cache_rejects_center_reuse_after_refined_peak_change_in_mixed_sequence() -> None:
    state = _state()
    cache: dict[str, GeometryObjectiveSignature] = {}

    _evaluate_objective_cache(cache, _objective_signature(state))
    _evaluate_objective_cache(cache, _objective_signature(state, center=(102.0, 101.0)))
    state.manual_selected_peak_rows[1]["refined_sim_caked_y"] = -999.0
    trace = _evaluate_objective_cache(
        cache,
        _objective_signature(state, center=(103.0, 101.0)),
    )

    assert trace["objective_cache_reject_reason"] == "refined_peak_changed"
    assert trace["objective_process_peaks_called"] is True


def test_objective_cache_rejects_center_reuse_after_qr_branch_identity_change_in_mixed_sequence() -> None:
    state = _state()
    cache: dict[str, GeometryObjectiveSignature] = {}

    _evaluate_objective_cache(cache, _objective_signature(state))
    _evaluate_objective_cache(cache, _objective_signature(state, center=(102.0, 101.0)))
    state.geometry_q_group_entries_cache[0]["source_branch_index"] = 9
    trace = _evaluate_objective_cache(
        cache,
        _objective_signature(state, center=(103.0, 101.0)),
    )

    assert trace["objective_cache_reject_reason"] == "point_provider_changed"
    assert "point_provider_sig" in trace["objective_signature_changed_fields"]
    assert "qr_branch_identity_sig" in trace["objective_signature_changed_fields"]


def test_objective_cache_rejects_center_reuse_after_point_provider_change_in_mixed_sequence() -> None:
    state = _state()
    cache: dict[str, GeometryObjectiveSignature] = {}

    _evaluate_objective_cache(cache, _objective_signature(state))
    _evaluate_objective_cache(cache, _objective_signature(state, center=(102.0, 101.0)))
    state.geometry_q_group_entries_cache.pop()
    trace = _evaluate_objective_cache(
        cache,
        _objective_signature(state, center=(103.0, 101.0)),
    )

    assert trace["objective_cache_reject_reason"] == "point_provider_changed"
    assert trace["objective_process_peaks_called"] is True


def test_prune_fill_then_display_does_not_consume_deferred_qgroup_refresh() -> None:
    state = _state()

    fill_trace = _apply_action(
        state,
        UpdateAction.PRIMARY_PRUNE_FILL,
        q_group_content_signature_changed=True,
        hit_table_signature_changed=True,
    )
    display_trace = _apply_action(state, UpdateAction.DISPLAY_ONLY)

    assert state.refresh_requested is True
    assert fill_trace["qr_selector_refresh_deferred"] is True
    assert display_trace["geometry_fitter_handoff_valid"] is False


def test_prune_fill_then_center_remap_keeps_handoff_invalid_until_rows_refresh() -> None:
    state = _state()

    _apply_action(
        state,
        UpdateAction.PRIMARY_PRUNE_FILL,
        q_group_content_signature_changed=True,
        hit_table_signature_changed=True,
    )
    center_trace = _apply_action(
        state,
        UpdateAction.DETECTOR_CENTER_REMAP,
        detector_geometry_changed=True,
        hit_table_signature_changed=False,
    )

    assert state.refresh_requested is True
    assert center_trace["geometry_fitter_handoff_valid"] is False
    assert center_trace["caked_projection_cache_invalidated"] is True


def test_prune_changed_content_then_geometry_fit_preflight_reports_handoff_not_ready() -> None:
    state = _state()

    trace = _apply_action(
        state,
        UpdateAction.PRIMARY_PRUNE_REUSE,
        q_group_content_signature_changed=True,
        hit_table_signature_changed=True,
    )

    assert state.refresh_requested is True
    assert trace["geometry_fitter_handoff_valid"] is False


def test_qgroup_refresh_after_rows_apply_restores_handoff_validity() -> None:
    state = _state()
    _apply_action(
        state,
        UpdateAction.PRIMARY_PRUNE_FILL,
        q_group_content_signature_changed=True,
        hit_table_signature_changed=True,
    )

    replacement_rows = list(state.geometry_q_group_entries_cache)
    state.source_row_snapshots = {
        0: {
            "hit_table_signature": ("hit-table", "identity-b"),
            "row_content_signature": ("q-content", "identity-b"),
            "rows": replacement_rows,
        }
    }
    state.stored_hit_table_signature = ("hit-table", "identity-b")
    state.stored_q_group_content_signature = ("q-content", "identity-b")
    state.geometry_q_group_entries_cache_signature = ("q-entries", "identity-b")
    state.manual_pick_cache_signature = ("manual", "identity-b")
    state.manual_pick_cache_data = {"projected": [(41.0, 42.0), (43.0, 44.0)]}
    state.refresh_requested = False

    display_trace = _apply_action(state, UpdateAction.DISPLAY_ONLY)

    assert display_trace["geometry_fitter_handoff_valid"] is True
    assert _point_provider_report(state)["fallback_pair_count"] == 0
