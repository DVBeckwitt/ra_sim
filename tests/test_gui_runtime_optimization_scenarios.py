from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ra_sim.gui.runtime_update_dependencies import (
    RuntimeCacheState,
    SimulationDependencySignatures,
    UpdateAction,
    classify_update,
)
from tests.test_gui_runtime_update_actions import (
    _old_ready_result,
    _prepare_center_remap_runtime,
    _prepare_runtime,
)


def _scenario_image_signature(center_x: float, center_y: float, prune_bias: float) -> tuple:
    return (
        "optimization-scenario",
        round(float(center_x), 9),
        round(float(center_y), 9),
        round(float(prune_bias), 6),
    )


def _scenario_primary_cache_signature(
    center_x: float,
    center_y: float,
    distance_m: float,
) -> tuple:
    return _scenario_image_signature(center_x, center_y, 0.0) + (
        round(float(distance_m), 9),
    )


def _primary_detector_remap_signature(
    dependency_signatures: SimulationDependencySignatures,
) -> tuple[object, ...]:
    return (
        "primary_detector_center_remap",
        dependency_signatures.source_sig,
        dependency_signatures.physics_sig,
        dependency_signatures.detector_projection_sig,
        dependency_signatures.primary_filter_sig,
    )


def _patch_scenario_signature(
    monkeypatch: pytest.MonkeyPatch,
    runtime_session,
) -> None:
    def _signature(param_set, *, sf_prune_bias, **_kwargs):
        return _scenario_image_signature(
            float(param_set.get("center_x", 1.0)),
            float(param_set.get("center_y", 1.0)),
            float(sf_prune_bias),
        )

    monkeypatch.setattr(
        runtime_session,
        "_geometry_source_snapshot_signature_from_params",
        _signature,
        raising=False,
    )


def _seed_scenario_cache_state(
    runtime_session,
    state,
    *,
    prune_bias: float,
) -> None:
    image_signature = _scenario_image_signature(
        float(runtime_session.center_x_var.get()),
        float(runtime_session.center_y_var.get()),
        float(prune_bias),
    )
    state.last_sim_signature = image_signature
    state.last_simulation_signature = image_signature + (0, 0)
    if state.last_dependency_signatures is not None:
        state.last_dependency_signatures = replace(
            state.last_dependency_signatures,
            full_image_sig=image_signature,
        )

    state.primary_contribution_cache_signature = _scenario_primary_cache_signature(
        float(runtime_session.center_x_var.get()),
        float(runtime_session.center_y_var.get()),
        float(runtime_session.corto_detector_var.get()),
    )
    state.primary_source_mode = "miller"
    state.primary_active_contribution_keys = [0]
    state.primary_hit_table_cache = {
        0: np.asarray([[10.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    }
    state.primary_best_sample_index_cache = {0: 0}


def _last_trace_event(events: list[dict[str, object]], name: str) -> dict[str, object]:
    return next(event for event in reversed(events) if event["event"] == name)


def test_full_simulation_signature_change_requests_worker(monkeypatch) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)
    prune_bias = {"value": 0.0}
    _patch_scenario_signature(monkeypatch, runtime_session)
    monkeypatch.setattr(
        runtime_session,
        "current_sf_prune_bias",
        lambda: prune_bias["value"],
        raising=False,
    )
    _seed_scenario_cache_state(runtime_session, state, prune_bias=0.0)
    state.last_sim_signature = ("stale-full-image",)
    state.last_simulation_signature = ("stale-full-image", 0, 0)
    requested_jobs: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )

    runtime_session.do_update()

    assert len(requested_jobs) == 1
    assert requested_jobs[0]["job_kind"] == "full"


def test_optimized_update_sequence_worker_call_counts(monkeypatch) -> None:
    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)
    prune_bias = {"value": 0.0}
    _patch_scenario_signature(monkeypatch, runtime_session)
    monkeypatch.setattr(
        runtime_session,
        "current_sf_prune_bias",
        lambda: prune_bias["value"],
        raising=False,
    )
    _seed_scenario_cache_state(runtime_session, state, prune_bias=0.0)

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

    prune_bias["value"] = 0.2
    runtime_session.do_update()
    assert requested_jobs == []
    assert _last_trace_event(trace_events, "do_update_complete")["update_action"] == (
        "primary_prune_reuse"
    )

    trace_events.clear()
    runtime_session.display_controls_view_state.simulation_max_var.set(2.0)
    runtime_session.do_update()
    assert requested_jobs == []
    assert _last_trace_event(trace_events, "do_update_complete")["update_action"] == (
        "display_only"
    )

    trace_events.clear()
    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()
    assert requested_jobs == []
    complete_trace = _last_trace_event(trace_events, "do_update_complete")
    assert complete_trace["update_action"] == "detector_center_remap"
    assert complete_trace["center_remap_used"] is True

    trace_events.clear()
    runtime_session.corto_detector_var.set(0.75)
    runtime_session.do_update()
    assert len(requested_jobs) == 1
    assert requested_jobs[0]["job_kind"] == "full"
    waiting_trace = _last_trace_event(trace_events, "do_update_return_waiting_for_simulation")
    assert waiting_trace["update_action"] == "full_simulation"
    assert waiting_trace["requires_worker"] is True


def test_stale_full_worker_result_does_not_overwrite_fast_path_sequence(
    monkeypatch,
) -> None:
    runtime_session, _fixture = _prepare_runtime(monkeypatch)
    state = runtime_session.simulation_runtime_state
    runtime_session.do_update()
    runtime_session.display_controls_view_state.simulation_max_var.set(2.0)
    state.worker_ready_result = _old_ready_result(state, value=99.0)
    runtime_session.do_update()
    assert state.worker_ready_result is None
    assert not np.array_equal(
        state.stored_sim_image,
        np.full((2, 2), 99.0, dtype=np.float64),
    )

    runtime_session, state = _prepare_center_remap_runtime(monkeypatch)
    state.worker_ready_result = _old_ready_result(state, value=99.0)
    runtime_session.center_x_var.set(1.5)
    runtime_session.do_update()
    assert state.worker_ready_result is None
    assert not np.array_equal(
        state.stored_sim_image,
        np.full((2, 2), 99.0, dtype=np.float64),
    )


def _signatures() -> SimulationDependencySignatures:
    return SimulationDependencySignatures(
        source_sig=("source", 1),
        physics_sig=("physics", "lattice", 1.0),
        detector_projection_sig=("projection", 0.5),
        detector_center_sig=("center", 1.0, 1.0),
        primary_filter_sig=("primary_filter", (0,)),
        combine_sig=("combine", 1.0, 0.0),
        analysis_geometry_sig=("analysis", 360, 720),
        display_sig=("display", 0.0, 1.0),
        hit_table_sig=("hit_table", False),
        full_image_sig=("full_image", 1),
    )


def test_mixed_changes_fail_closed_to_full_simulation() -> None:
    previous = _signatures()
    cases = [
        replace(
            previous,
            detector_center_sig=("center", 1.5, 1.0),
            detector_projection_sig=("projection", 0.75),
            full_image_sig=("full_image", "center-distance"),
        ),
        replace(
            previous,
            primary_filter_sig=("primary_filter", (0, 1)),
            physics_sig=("physics", "lattice", 2.0),
            full_image_sig=("full_image", "prune-lattice"),
        ),
        replace(
            previous,
            display_sig=("display", 0.0, 2.0),
            physics_sig=("physics", "lattice", 2.0),
            full_image_sig=("full_image", "display-physics"),
        ),
    ]

    for current in cases:
        decision = classify_update(
            previous,
            current,
            RuntimeCacheState(
                can_remap_detector_center=True,
                prune_cache_mode="reuse",
            ),
        )

        assert decision.action is UpdateAction.FULL_SIMULATION
        assert decision.requires_worker is True
