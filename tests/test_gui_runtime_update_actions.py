from __future__ import annotations

import importlib

import numpy as np
import pytest

from ra_sim.gui.state import GeometryQGroupState
from tests.test_gui_runtime_import_safe import (
    _install_matching_hidden_analysis_payload_state,
    _patch_do_update_detector_cache_prereqs,
    _patch_do_update_first_visible_simulation_finish_prereqs,
    _patch_rich_mosaic_params,
)


def _prepare_runtime(monkeypatch: pytest.MonkeyPatch):
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(monkeypatch, runtime_session, fixture)
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=[],
        scheduled_settle_calls=[],
        apply_scale_factor_calls=[],
    )
    state = runtime_session.simulation_runtime_state
    state.last_sim_signature = fixture["sim_signature"]
    state.last_simulation_signature = fixture["sim_signature"] + (0, 0)
    state.worker_ready_result = None
    state.worker_future = None
    state.worker_job_counter = 0
    state.worker_error_text = None
    state.simulation_epoch = 0
    state.peak_positions = []
    state.peak_millers = []
    state.selected_peak_record = None
    return runtime_session, fixture


def _old_ready_result(state, value: float) -> dict[str, object]:
    return {
        "job_id": 1,
        "job_kind": "full",
        "signature": state.last_simulation_signature,
        "timing_update_id": 1,
        "timing_reason": "old_request",
        "epoch": int(state.simulation_epoch),
        "run_primary": True,
        "run_secondary": False,
        "secondary_available": False,
        "active_peak_row_sides": ("primary",),
        "primary_image": np.full((2, 2), float(value), dtype=np.float64),
        "secondary_image": np.zeros((2, 2), dtype=np.float64),
        "primary_hit_table_state_refreshed": False,
        "secondary_hit_table_state_refreshed": False,
        "primary_raw_rows_fresh": False,
        "secondary_raw_rows_fresh": False,
        "primary_intersection_cache_built": False,
        "secondary_intersection_cache_built": False,
        "primary_max_positions": [],
        "secondary_max_positions": [],
        "primary_intersection_cache": [],
        "secondary_intersection_cache": [],
        "primary_peak_table_lattice": [],
        "secondary_peak_table_lattice": [],
        "image_generation_elapsed_ms": 0.0,
    }


def _patch_center_sensitive_signature(monkeypatch: pytest.MonkeyPatch, runtime_session) -> None:
    def _signature(param_set, **_kwargs):
        return (
            "sim-sig",
            round(float(param_set.get("center_x", 1.0)), 9),
            round(float(param_set.get("center_y", 1.0)), 9),
        )

    monkeypatch.setattr(
        runtime_session,
        "_geometry_source_snapshot_signature_from_params",
        _signature,
        raising=False,
    )


def _primary_detector_remap_signature(dependency_signatures) -> tuple[object, ...]:
    return (
        "primary_detector_center_remap",
        dependency_signatures.source_sig,
        dependency_signatures.physics_sig,
        dependency_signatures.detector_projection_sig,
        dependency_signatures.primary_filter_sig,
    )


def _secondary_detector_remap_signature(dependency_signatures) -> tuple[object, ...]:
    return (
        "secondary_detector_center_remap",
        dependency_signatures.source_sig,
        dependency_signatures.physics_sig,
        dependency_signatures.detector_projection_sig,
    )


def _prepare_center_remap_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    secondary: bool = False,
):
    runtime_session, _fixture = _prepare_runtime(monkeypatch)
    _patch_rich_mosaic_params(monkeypatch, runtime_session)
    _patch_center_sensitive_signature(monkeypatch, runtime_session)
    state = runtime_session.simulation_runtime_state
    state.analysis_epoch = 0
    state.analysis_ready_result = None
    state.last_analysis_signature = None
    state.last_analysis_cache_sig = None
    state.analysis_preview_active = False
    state.analysis_preview_bins = None
    state.last_res2_sim = object()
    state.last_res2_background = object()
    state.sim_miller1 = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    state.sim_intens1 = np.asarray([10.0], dtype=np.float64)
    state.sim_miller1_all = state.sim_miller1.copy()
    state.sim_intens1_all = state.sim_intens1.copy()
    state.sim_primary_qr = {}
    state.sim_primary_qr_all = {}
    state.primary_requested_source_mode = "miller"
    state.primary_requested_contribution_keys = [0]
    state.primary_requested_filter_signature = ("center-remap-filter",)
    state.primary_best_sample_index_cache = {0: 0}
    state.last_sim_signature = ("sim-sig", 1.0, 1.0)
    state.last_simulation_signature = ("sim-sig", 1.0, 1.0, 0, 0)
    state.stored_primary_sim_image = np.ones((2, 2), dtype=np.float64)
    state.stored_secondary_sim_image = None
    monkeypatch.setattr(
        runtime_session,
        "_restore_live_peak_rows_from_combined_hit_tables",
        lambda **_kwargs: {"published_peak_record_count": 0, "skipped_reason": "test"},
        raising=False,
    )
    if secondary:
        state.sim_miller2 = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64)
        state.sim_intens2 = np.asarray([5.0], dtype=np.float64)
        state.sim_miller2_all = state.sim_miller2.copy()
        state.sim_intens2_all = state.sim_intens2.copy()
        state.stored_secondary_sim_image = np.full((2, 2), 2.0, dtype=np.float64)
        runtime_session.weight2_var.set(1.0)
    else:
        state.sim_miller2 = np.empty((0, 3), dtype=np.float64)
        state.sim_intens2 = np.empty((0,), dtype=np.float64)
        state.sim_miller2_all = state.sim_miller2.copy()
        state.sim_intens2_all = state.sim_intens2.copy()

    runtime_session.do_update()
    dependency_signatures = state.last_dependency_signatures
    state.primary_relative_hit_table_cache = {
        0: np.asarray([[10.0, -0.5, -0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    }
    state.primary_relative_hit_table_cache_center = (1.0, 1.0)
    state.primary_relative_hit_table_cache_signature = _primary_detector_remap_signature(
        dependency_signatures
    )
    if secondary:
        state.secondary_relative_hit_table_cache = [
            np.asarray([[5.0, -0.25, -0.25, 0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
        ]
        state.secondary_relative_best_sample_index_cache = {0: 0}
        state.secondary_relative_hit_table_cache_center = (1.0, 1.0)
        state.secondary_relative_hit_table_cache_signature = (
            _secondary_detector_remap_signature(dependency_signatures)
        )
    return runtime_session, state


def test_display_only_action_does_not_call_simulation_worker(monkeypatch) -> None:
    runtime_session, _fixture = _prepare_runtime(monkeypatch)
    state = runtime_session.simulation_runtime_state
    requested_jobs: list[dict[str, object]] = []
    trace_events: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_events.append({"event": event, **fields}),
        raising=False,
    )

    runtime_session.do_update()
    assert state.last_dependency_signatures is not None
    requested_jobs.clear()
    trace_events.clear()

    monkeypatch.setattr(
        runtime_session,
        "_publish_combined_simulation_state",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("display-only path must not republish combined image")
        ),
        raising=False,
    )
    runtime_session.display_controls_view_state.simulation_max_var.set(2.0)

    runtime_session.do_update()

    assert requested_jobs == []
    complete_trace = next(
        event for event in reversed(trace_events) if event["event"] == "do_update_complete"
    )
    assert complete_trace["update_action"] == "display_only"
    assert complete_trace["requires_worker"] is False


def test_display_only_fast_path_requires_existing_stored_image(monkeypatch) -> None:
    runtime_session, _fixture = _prepare_runtime(monkeypatch)
    state = runtime_session.simulation_runtime_state
    trace_events: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_events.append({"event": event, **fields}),
        raising=False,
    )

    runtime_session.do_update()
    previous_signatures = state.last_dependency_signatures
    state.stored_sim_image = None
    state.stored_primary_sim_image = None
    state.stored_secondary_sim_image = None
    trace_events.clear()

    runtime_session.display_controls_view_state.simulation_max_var.set(2.0)
    runtime_session.do_update()

    assert state.last_dependency_signatures is previous_signatures
    assert not any(event["event"] == "do_update_display_only_fast_path" for event in trace_events)
    assert any(event["event"] == "do_update_return_no_simulation_image" for event in trace_events)


def test_combine_only_action_does_not_call_simulation_worker(monkeypatch) -> None:
    runtime_session, _fixture = _prepare_runtime(monkeypatch)
    state = runtime_session.simulation_runtime_state
    state.stored_primary_sim_image = np.ones((2, 2), dtype=np.float64)
    state.stored_secondary_sim_image = np.full((2, 2), 2.0, dtype=np.float64)
    requested_jobs: list[dict[str, object]] = []
    trace_events: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_events.append({"event": event, **fields}),
        raising=False,
    )

    runtime_session.do_update()
    assert np.array_equal(state.stored_sim_image, np.ones((2, 2), dtype=np.float64))
    requested_jobs.clear()
    trace_events.clear()

    runtime_session.weight2_var.set(1.0)

    runtime_session.do_update()

    assert requested_jobs == []
    complete_trace = next(
        event for event in reversed(trace_events) if event["event"] == "do_update_complete"
    )
    assert complete_trace["update_action"] == "combine_only"
    assert complete_trace["requires_worker"] is False


def test_combine_only_fast_path_requires_cached_primary_or_secondary_images(
    monkeypatch,
) -> None:
    runtime_session, _fixture = _prepare_runtime(monkeypatch)
    state = runtime_session.simulation_runtime_state
    trace_events: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_events.append({"event": event, **fields}),
        raising=False,
    )

    runtime_session.do_update()
    previous_signatures = state.last_dependency_signatures
    state.stored_sim_image = None
    state.stored_primary_sim_image = None
    state.stored_secondary_sim_image = None
    trace_events.clear()

    runtime_session.weight2_var.set(1.0)
    runtime_session.do_update()

    assert state.last_dependency_signatures is previous_signatures
    assert not any(event["event"] == "do_update_combine_only_fast_path" for event in trace_events)
    assert any(event["event"] == "do_update_return_no_simulation_image" for event in trace_events)


def test_display_only_does_not_clear_primary_cache(monkeypatch) -> None:
    runtime_session, _fixture = _prepare_runtime(monkeypatch)
    state = runtime_session.simulation_runtime_state
    primary_cache = {0: np.ones((1, 7), dtype=np.float64)}
    state.primary_contribution_cache_signature = ("primary-cache",)
    state.primary_source_mode = "miller"
    state.primary_active_contribution_keys = [0]
    state.primary_hit_table_cache = primary_cache
    state.primary_best_sample_index_cache = {0: 0}

    runtime_session.do_update()
    runtime_session.display_controls_view_state.simulation_min_var.set(0.2)
    runtime_session.do_update()

    assert state.primary_hit_table_cache is primary_cache
    assert state.primary_best_sample_index_cache == {0: 0}


def test_combine_only_republishes_combined_image(monkeypatch) -> None:
    runtime_session, _fixture = _prepare_runtime(monkeypatch)
    state = runtime_session.simulation_runtime_state
    state.stored_primary_sim_image = np.ones((2, 2), dtype=np.float64)
    state.stored_secondary_sim_image = np.full((2, 2), 2.0, dtype=np.float64)

    runtime_session.do_update()
    runtime_session.weight2_var.set(1.0)
    runtime_session.do_update()

    assert np.array_equal(state.stored_sim_image, np.full((2, 2), 3.0, dtype=np.float64))


def test_stale_worker_result_does_not_overwrite_display_only_fast_path(monkeypatch) -> None:
    runtime_session, _fixture = _prepare_runtime(monkeypatch)
    state = runtime_session.simulation_runtime_state

    runtime_session.do_update()
    runtime_session.display_controls_view_state.simulation_max_var.set(2.0)
    runtime_session.do_update()
    display_image = np.asarray(state.stored_sim_image, dtype=np.float64).copy()

    state.worker_ready_result = _old_ready_result(state, value=99.0)
    runtime_session.do_update()

    assert np.array_equal(state.stored_sim_image, display_image)
    assert state.worker_ready_result is None


def test_stale_worker_result_does_not_overwrite_prune_reuse_fast_path(
    monkeypatch,
) -> None:
    runtime_session, _fixture = _prepare_runtime(monkeypatch)
    state = runtime_session.simulation_runtime_state
    state.sim_miller1_all = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    state.sim_intens1_all = np.asarray([10.0, 5.0], dtype=np.float64)
    state.sim_miller2_all = np.empty((0, 3), dtype=np.float64)
    state.sim_intens2_all = np.empty((0,), dtype=np.float64)
    state.sim_primary_qr_all = {}
    state.sim_miller1 = state.sim_miller1_all.copy()
    state.sim_intens1 = state.sim_intens1_all.copy()
    state.primary_requested_source_mode = "miller"
    state.primary_requested_filter_signature = ("stable-source-filter",)
    state.primary_requested_contribution_keys = [0, 1]
    state.primary_active_contribution_keys = [0, 1]
    state.primary_contribution_cache_signature = ("primary-cache", 0.5)
    state.primary_source_mode = "miller"
    state.primary_hit_table_cache = {
        0: np.asarray([[10.0, 0.5, 0.5]], dtype=np.float64),
        1: np.asarray([[5.0, 1.5, 0.5]], dtype=np.float64),
    }
    state.primary_best_sample_index_cache = {0: 0, 1: 0}

    def _signature(_param_set, *, primary_source_signature, sf_prune_bias, **_kwargs):
        if (
            isinstance(primary_source_signature, tuple)
            and len(primary_source_signature) == 2
            and primary_source_signature[1] == ("stable-source-filter",)
        ):
            return ("primary-cache",)
        return ("broad-sim", primary_source_signature, round(float(sf_prune_bias), 3))

    monkeypatch.setattr(
        runtime_session,
        "_geometry_source_snapshot_signature_from_params",
        _signature,
        raising=False,
    )
    _patch_rich_mosaic_params(monkeypatch, runtime_session)
    initial_signature = _signature(
        {},
        primary_source_signature=(
            "miller",
            id(state.sim_miller1),
            tuple(state.sim_miller1.shape),
        ),
        sf_prune_bias=0.0,
    )
    state.last_sim_signature = initial_signature
    state.last_simulation_signature = initial_signature + (0, 0)

    runtime_session.do_update()
    state.sim_miller1 = state.sim_miller1_all[:1].copy()
    state.sim_intens1 = state.sim_intens1_all[:1].copy()
    state.primary_requested_contribution_keys = [0]
    state.last_sim_signature = ("seed-sim",)
    state.last_simulation_signature = ("seed-sim", 0, 0)
    runtime_session.do_update()
    rematerialized_image = np.asarray(state.stored_sim_image, dtype=np.float64).copy()

    state.worker_ready_result = _old_ready_result(state, value=99.0)
    runtime_session.do_update()

    assert np.array_equal(state.stored_sim_image, rematerialized_image)
    assert state.worker_ready_result is None


def test_apply_center_remap_action_does_not_call_simulation_worker(monkeypatch) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)
    requested_jobs: list[dict[str, object]] = []
    trace_events: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_events.append({"event": event, **fields}),
        raising=False,
    )

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    assert requested_jobs == []
    assert state.last_dependency_signatures.detector_center_sig == (1.5, 1.0)
    complete_trace = next(
        event for event in reversed(trace_events) if event["event"] == "do_update_complete"
    )
    assert complete_trace["update_action"] == "detector_center_remap"
    assert complete_trace["requires_worker"] is False
    assert complete_trace["center_remap_used"] is True


def test_center_remap_retains_qr_branch_identity(monkeypatch) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)
    monkeypatch.setattr(
        runtime_session,
        "geometry_q_group_runtime_bindings_factory",
        None,
        raising=False,
    )
    entries = [{"q_group_key": ("q_group", "primary", 1, 5), "peak_count": 1}]
    state.geometry_q_group_entries_cache = [dict(entry) for entry in entries]
    state.geometry_q_group_entries_cache_signature = ("q-group-cache",)
    state.source_row_snapshots = {0: {"row_content_signature": ("rows",), "rows": []}}
    trace_events: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_events.append({"event": event, **fields}),
        raising=False,
    )

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    complete_trace = next(
        event for event in reversed(trace_events) if event["event"] == "do_update_complete"
    )
    assert complete_trace["update_action"] == "detector_center_remap"
    assert complete_trace["qr_selector_branch_identity_retained"] is True
    assert complete_trace["qr_selector_entries_retained"] is True
    assert complete_trace["source_row_snapshots_retained"] is True
    assert state.geometry_q_group_entries_cache == entries
    assert state.geometry_q_group_entries_cache_signature == ("q-group-cache",)
    assert state.source_row_snapshots == {
        0: {"row_content_signature": ("rows",), "rows": []}
    }


def test_center_remap_preserves_qr_disabled_masks(monkeypatch) -> None:
    runtime_session, _state = _prepare_center_remap_runtime(monkeypatch)
    q_group_state = GeometryQGroupState(
        disabled_qr_sets={("primary", 1)},
        disabled_qz_sections={("primary", 1, 0)},
        pending_legacy_disabled_qz_sections={("secondary", 2, 1)},
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_q_group_state",
        q_group_state,
        raising=False,
    )

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    assert q_group_state.disabled_qr_sets == {("primary", 1)}
    assert q_group_state.disabled_qz_sections == {("primary", 1, 0)}
    assert q_group_state.pending_legacy_disabled_qz_sections == {("secondary", 2, 1)}


def test_center_remap_updates_image_and_positions(monkeypatch) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)
    relative_cache_signature = state.primary_relative_hit_table_cache_signature

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    expected_table = np.asarray(
        [[10.0, 1.0, 0.5, 0.0, 1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    expected_image = (
        runtime_session.gui_runtime_primary_cache.rasterize_hit_tables_to_image(
            [expected_table],
            image_size=2,
        )
    )
    np.testing.assert_allclose(state.stored_primary_sim_image, expected_image)
    np.testing.assert_allclose(state.stored_sim_image, expected_image)
    np.testing.assert_allclose(state.primary_hit_table_cache[0][:, 1:3], [[1.0, 0.5]])
    assert state.primary_relative_hit_table_cache
    assert state.primary_relative_hit_table_cache_signature == relative_cache_signature
    assert state.stored_primary_intersection_cache_signature == state.stored_hit_table_signature
    if state.stored_primary_intersection_cache:
        cache_table = np.asarray(state.stored_primary_intersection_cache[0], dtype=np.float64)
        np.testing.assert_allclose(cache_table[:, 2:4], [[1.0, 0.5]])


def test_center_remap_updates_primary_detector_row_col_positions(monkeypatch) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    np.testing.assert_allclose(state.primary_hit_table_cache[0][:, 1:3], [[1.0, 0.5]])


def test_center_remap_updates_peak_and_intersection_positions_only(monkeypatch) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    raw_table = np.asarray(state.primary_hit_table_cache[0], dtype=np.float64)
    np.testing.assert_allclose(raw_table[:, 0], [10.0])
    np.testing.assert_allclose(raw_table[:, 1:3], [[1.0, 0.5]])
    np.testing.assert_allclose(raw_table[:, 3:], [[0.0, 1.0, 0.0, 0.0]])

    peak_table = np.asarray(state.stored_primary_max_positions[0], dtype=np.float64)
    np.testing.assert_allclose(peak_table[:, 1:3], [[1.0, 0.5]])
    np.testing.assert_allclose(peak_table[:, 0], [10.0])
    np.testing.assert_allclose(peak_table[:, 3:7], raw_table[:, 3:])

    intersection_table = np.asarray(
        state.stored_primary_intersection_cache[0],
        dtype=np.float64,
    )
    np.testing.assert_allclose(intersection_table[:, 2:4], [[1.0, 0.5]])
    np.testing.assert_allclose(intersection_table[:, 4], [10.0])
    np.testing.assert_allclose(intersection_table[:, 6:9], [[1.0, 0.0, 0.0]])


def test_center_remap_updates_max_positions_and_intersection_cache_positions(
    monkeypatch,
) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    peak_table = np.asarray(state.stored_primary_max_positions[0], dtype=np.float64)
    intersection_table = np.asarray(
        state.stored_primary_intersection_cache[0],
        dtype=np.float64,
    )
    np.testing.assert_allclose(peak_table[:, 1:3], [[1.0, 0.5]])
    np.testing.assert_allclose(intersection_table[:, 2:4], [[1.0, 0.5]])


def test_center_remap_invalidates_analysis_geometry_caches(monkeypatch) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)
    state.last_analysis_signature = ("old-analysis",)
    state.last_analysis_cache_sig = ("old-analysis-cache",)
    state.last_caked_image_unscaled = np.ones((2, 2), dtype=np.float64)
    state.last_q_space_image_unscaled = np.ones((2, 2), dtype=np.float64)

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    assert state.analysis_epoch == 1
    assert state.last_analysis_signature is None
    assert state.last_analysis_cache_sig is None
    assert state.last_caked_image_unscaled is None
    assert state.last_q_space_image_unscaled is None


def test_center_remap_invalidates_caked_projection_cache(monkeypatch) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)
    state.last_caked_image_unscaled = np.ones((2, 2), dtype=np.float64)
    state.last_caked_extent = [0.0, 1.0, -1.0, 1.0]
    state.last_caked_radial_values = np.asarray([1.0, 2.0], dtype=np.float64)
    state.last_caked_azimuth_values = np.asarray([-1.0, 1.0], dtype=np.float64)
    state.last_q_space_payload_signature = ("q-space",)
    state.geometry_fit_caking_ai_cache = {"old": True}
    state.caking_cache = {"sim_results": {"old": 1}, "bg_results": {"old": 2}}
    trace_events: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_events.append({"event": event, **fields}),
        raising=False,
    )

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    complete_trace = next(
        event for event in reversed(trace_events) if event["event"] == "do_update_complete"
    )
    assert complete_trace["caked_projection_cache_invalidated"] is True
    assert state.last_caked_image_unscaled is None
    assert state.last_caked_extent is None
    assert state.last_caked_radial_values is None
    assert state.last_caked_azimuth_values is None
    assert state.last_q_space_payload_signature is None
    assert state.geometry_fit_caking_ai_cache == {}
    assert state.caking_cache == {"sim_results": {}, "bg_results": {}}


def test_center_remap_marks_geometry_fitter_handoff_invalid_when_projection_refresh_required(
    monkeypatch,
) -> None:
    runtime_session, _state = _prepare_center_remap_runtime(monkeypatch)
    runtime_session.geometry_runtime_state.manual_pick_cache_signature = ("manual",)
    runtime_session.geometry_runtime_state.manual_pick_cache_data = {"rows": [1]}
    monkeypatch.setattr(
        runtime_session,
        "_hkl_pick_simulation_points_payload_cache",
        {"payload": True},
        raising=False,
    )
    trace_events: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_events.append({"event": event, **fields}),
        raising=False,
    )

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    complete_trace = next(
        event for event in reversed(trace_events) if event["event"] == "do_update_complete"
    )
    assert complete_trace["detector_projection_cache_refreshed"] is True
    assert complete_trace["geometry_fitter_handoff_valid"] is False
    assert runtime_session.geometry_runtime_state.manual_pick_cache_signature is None
    assert runtime_session.geometry_runtime_state.manual_pick_cache_data == {}


def test_center_remap_keeps_geometry_fitter_handoff_valid_when_all_projection_data_refreshed(
    monkeypatch,
) -> None:
    runtime_session, _state = _prepare_center_remap_runtime(monkeypatch)
    trace_events: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_events.append({"event": event, **fields}),
        raising=False,
    )

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    complete_trace = next(
        event for event in reversed(trace_events) if event["event"] == "do_update_complete"
    )
    assert complete_trace["detector_projection_cache_refreshed"] is True
    assert complete_trace["caked_projection_cache_invalidated"] is False
    assert complete_trace["geometry_fitter_handoff_valid"] is True


def test_center_remap_falls_back_when_secondary_active_without_exact_secondary_cache(
    monkeypatch,
) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch, secondary=True)
    requested_jobs: list[dict[str, object]] = []
    previous_signatures = state.last_dependency_signatures
    state.secondary_relative_hit_table_cache = []
    state.secondary_relative_hit_table_cache_signature = None
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    assert len(requested_jobs) == 1
    assert requested_jobs[0]["job_kind"] == "full"
    assert requested_jobs[0]["run_secondary"] is True
    assert state.last_dependency_signatures is previous_signatures


def test_center_remap_falls_back_when_secondary_exact_cache_missing(monkeypatch) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch, secondary=True)
    requested_jobs: list[dict[str, object]] = []
    trace_events: list[dict[str, object]] = []
    state.secondary_relative_hit_table_cache = []
    state.secondary_relative_hit_table_cache_signature = None
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_events.append({"event": event, **fields}),
        raising=False,
    )

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    waiting_trace = next(
        event
        for event in reversed(trace_events)
        if event["event"] == "do_update_return_waiting_for_simulation"
    )
    assert len(requested_jobs) == 1
    assert waiting_trace["update_action"] == "full_simulation"
    assert waiting_trace["center_remap_used"] is False
    assert waiting_trace["detector_projection_cache_refreshed"] is False
    assert waiting_trace["center_remap_fallback_reason"] == "secondary_exact_cache_missing"


def test_center_remap_secondary_active_uses_exact_secondary_cache(monkeypatch) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch, secondary=True)
    requested_jobs: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    assert requested_jobs == []
    assert state.stored_secondary_sim_image is not None
    np.testing.assert_allclose(
        np.asarray(state.stored_secondary_max_positions[0], dtype=np.float64)[:, 1:3],
        [[1.25, 0.75]],
    )
    np.testing.assert_allclose(
        state.stored_sim_image,
        state.stored_primary_sim_image + state.stored_secondary_sim_image,
    )


def test_center_remap_never_uses_clipped_only_cache(monkeypatch) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)
    requested_jobs: list[dict[str, object]] = []
    trace_events: list[dict[str, object]] = []
    previous_signatures = state.last_dependency_signatures
    state.primary_relative_hit_table_cache = {}
    state.primary_relative_hit_table_cache_signature = None
    state.primary_hit_table_cache = {
        0: np.asarray([[10.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    }
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_events.append({"event": event, **fields}),
        raising=False,
    )

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    assert len(requested_jobs) == 1
    assert requested_jobs[0]["job_kind"] == "full"
    assert state.last_dependency_signatures is previous_signatures
    assert not any(
        event["event"] == "do_update_detector_center_remap_fast_path"
        for event in trace_events
    )
    waiting_trace = next(
        event
        for event in reversed(trace_events)
        if event["event"] == "do_update_return_waiting_for_simulation"
    )
    assert waiting_trace["update_action"] == "full_simulation"
    assert waiting_trace["requires_worker"] is True
    assert waiting_trace["center_remap_used"] is False


def test_center_plus_distance_falls_back_to_full_simulation(monkeypatch) -> None:
    runtime_session, _state = _prepare_center_remap_runtime(monkeypatch)
    requested_jobs: list[dict[str, object]] = []
    trace_events: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_events.append({"event": event, **fields}),
        raising=False,
    )

    runtime_session.center_x_var.set(1.5)
    runtime_session.corto_detector_var.set(0.75)
    runtime_session.do_update()

    waiting_trace = next(
        event
        for event in reversed(trace_events)
        if event["event"] == "do_update_return_waiting_for_simulation"
    )
    assert len(requested_jobs) == 1
    assert waiting_trace["update_action"] == "full_simulation"
    assert waiting_trace["center_remap_used"] is False
    assert waiting_trace["center_remap_fallback_reason"] == "physics_dependency_changed"


def test_center_plus_detector_orientation_falls_back_to_full_simulation(monkeypatch) -> None:
    runtime_session, _state = _prepare_center_remap_runtime(monkeypatch)
    requested_jobs: list[dict[str, object]] = []
    trace_events: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_events.append({"event": event, **fields}),
        raising=False,
    )

    runtime_session.center_x_var.set(1.5)
    runtime_session.gamma_var.set(1.0)
    runtime_session.do_update()

    waiting_trace = next(
        event
        for event in reversed(trace_events)
        if event["event"] == "do_update_return_waiting_for_simulation"
    )
    assert len(requested_jobs) == 1
    assert waiting_trace["update_action"] == "full_simulation"
    assert waiting_trace["center_remap_used"] is False
    assert waiting_trace["center_remap_fallback_reason"] == "physics_dependency_changed"


def test_center_remap_falls_back_when_exact_cache_signature_incompatible(
    monkeypatch,
) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)
    requested_jobs: list[dict[str, object]] = []
    previous_signatures = state.last_dependency_signatures
    state.primary_relative_hit_table_cache_signature = ("stale-remap-cache",)
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    assert len(requested_jobs) == 1
    assert requested_jobs[0]["job_kind"] == "full"
    assert state.last_dependency_signatures is previous_signatures


def test_stale_full_worker_result_does_not_overwrite_center_remap(monkeypatch) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)
    state.worker_ready_result = _old_ready_result(state, value=99.0)

    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()

    assert state.worker_ready_result is None
    assert not np.array_equal(
        state.stored_sim_image,
        np.full((2, 2), 99.0, dtype=np.float64),
    )
    np.testing.assert_allclose(state.primary_hit_table_cache[0][:, 1:3], [[1.0, 0.5]])


def test_scale_factor_classification_matches_numeric_semantics(monkeypatch) -> None:
    runtime_session, _fixture = _prepare_runtime(monkeypatch)
    state = runtime_session.simulation_runtime_state
    trace_events: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_events.append({"event": event, **fields}),
        raising=False,
    )

    runtime_session.do_update()
    trace_events.clear()
    runtime_session.display_controls_view_state.simulation_scale_factor_var.set(2.0)
    runtime_session.do_update()

    signature_trace = next(
        event for event in trace_events if event["event"] == "do_update_signature"
    )
    complete_trace = next(
        event for event in reversed(trace_events) if event["event"] == "do_update_complete"
    )
    assert signature_trace["classifier_update_action"] == "display_only"
    assert complete_trace["update_action"] == "display_only"
    assert state.last_sim_signature == _fixture["sim_signature"]
