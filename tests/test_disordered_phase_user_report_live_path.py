from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ra_sim.gui import geometry_q_group_manager
from ra_sim.utils.pbi2_ht_shift_cif import DISORDERED_PHASE_SOURCE_LABEL
from tests.test_disordered_phase_live_q_group_refresh import (
    _install_live_q_group_refresh_path,
)
from tests.test_disordered_phase_live_runtime_regression import (
    _hit_table,
    _prepare_live_runtime,
)
from tests.test_gui_runtime_import_safe import _RuntimeVar


def _fake_simulate_request(request, **_kwargs):
    image_size = int(request.geometry.image_size)
    image = (
        np.zeros((image_size, image_size), dtype=np.float64)
        if request.image_buffer is None
        else np.asarray(request.image_buffer, dtype=np.float64).copy()
    )
    is_disordered = abs(float(request.geometry.cv) - 20.937) < 1.0e-6
    hit_table = _hit_table(
        intensity=5.0 if is_disordered else 10.0,
        col=110.0 if is_disordered else 100.0,
        row=111.0 if is_disordered else 101.0,
        h=2 if is_disordered else 1,
        k=0,
        l_val=1,
    )
    hit_tables = [hit_table] if bool(request.collect_hit_tables) else []
    intersection_cache = [hit_table] if bool(request.build_intersection_cache) else []
    return SimpleNamespace(
        image=image,
        hit_tables=hit_tables,
        intersection_cache=intersection_cache,
        used_python_runner=True,
    )


def test_user_report_workflow_reaches_disordered_q_group_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_session, runtime_state = _prepare_live_runtime(
        monkeypatch,
        tmp_path,
        disordered_enabled=False,
    )
    progress_messages: list[str] = []
    scheduled_updates: list[str] = []
    live_schedule_calls: list[str] = []
    requested_jobs: list[dict[str, object]] = []
    _install_live_q_group_refresh_path(
        monkeypatch,
        runtime_session,
        runtime_state,
        progress_messages=progress_messages,
        scheduled_updates=scheduled_updates,
    )
    dependency_updates = []
    original_classify_update = runtime_session.classify_update
    monkeypatch.setattr(
        runtime_session,
        "classify_update",
        lambda previous, current, cache_state: (
            dependency_updates.append(current)
            or original_classify_update(previous, current, cache_state)
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_should_collect_hit_tables_for_update", lambda: False)
    monkeypatch.setattr(runtime_session, "simulate_request", _fake_simulate_request)
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "schedule_update",
        lambda: live_schedule_calls.append("schedule_update"),
        raising=False,
    )

    runtime_state.stored_max_positions_local = [_hit_table()]
    runtime_state.stored_peak_table_lattice = [(4.557, 6.979, "primary")]
    runtime_state.stored_source_reflection_indices_local = [0]
    runtime_state.stored_q_group_content_signature = (
        geometry_q_group_manager._geometry_q_group_content_signature_from_hit_tables(
            runtime_state.stored_max_positions_local
        )
    )
    runtime_state.primary_source_mode = "miller"
    runtime_state.primary_active_contribution_keys = []
    runtime_state.primary_contribution_cache_signature = None
    runtime_state.primary_hit_table_cache = {}
    runtime_state.primary_best_sample_index_cache = {}
    runtime_state.primary_relative_hit_table_cache = {}
    runtime_state.secondary_relative_hit_table_cache = []
    runtime_state.secondary_relative_best_sample_index_cache = {}

    runtime_session.p1_var.set(1.0)
    runtime_session.w0_var.set(0.0)
    runtime_session.w1_var.set(100.0)
    runtime_session.w2_var.set(0.0)
    runtime_session.geometry_include_6h_qr_reference_var.set(False)
    monkeypatch.setattr(
        runtime_session,
        "geometry_include_generated_disordered_qr_var",
        _RuntimeVar(True),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_include_disordered_phase_qr_reference_var",
        runtime_session.geometry_include_generated_disordered_qr_var,
        raising=False,
    )

    runtime_session.do_update()

    assert requested_jobs
    scheduled_job = {**requested_jobs[-1], "job_id": 1}
    assert scheduled_job["collect_disordered_phase_hit_tables"] is True
    runtime_state.stored_hit_table_signature = scheduled_job["hit_table_signature"]
    runtime_state.stored_primary_intersection_cache_signature = scheduled_job[
        "hit_table_signature"
    ]
    runtime_state.stored_primary_intersection_cache = [_hit_table()]
    runtime_state.stored_intersection_cache = [_hit_table()]
    runtime_state.last_simulation_signature = scheduled_job["signature"]
    runtime_state.update_running = False
    progress_messages.clear()
    requested_jobs.clear()
    scheduled_updates.clear()

    status = runtime_session._geometry_disordered_phase_qr_enable_status()
    pending_result = runtime_session._ensure_generated_disordered_qr_rows_for_live_refresh(
        status,
        primary_a=4.557,
        primary_c=6.979,
    )

    assert pending_result.can_refresh_selector is False
    assert pending_result.scheduled_collection is True
    assert any("Disordered Qr refs pending:" in msg for msg in progress_messages)
    assert runtime_state.disordered_phase_hit_table_collection_requested is True
    assert not any(
        "Updated listed Qr/Qz peaks:" in msg
        and "sources: primary=" in msg
        and f"{DISORDERED_PHASE_SOURCE_LABEL}=" not in msg
        for msg in progress_messages
    )
    assert live_schedule_calls == []

    result = runtime_session._run_simulation_generation_job(scheduled_job)
    runtime_session._apply_ready_simulation_result(result)
    runtime_state.last_sim_signature = scheduled_job["signature"][:-2]
    runtime_state.last_simulation_signature = scheduled_job["signature"][:-2] + (0, 0)
    runtime_state.stored_hit_table_signature = scheduled_job["hit_table_signature"]
    runtime_state.stored_primary_intersection_cache_signature = scheduled_job["hit_table_signature"]
    runtime_state.stored_primary_intersection_cache = [_hit_table()]
    runtime_state.stored_intersection_cache = [_hit_table()]
    if dependency_updates:
        runtime_state.last_dependency_signatures = dependency_updates[-1]
    runtime_state.update_running = False

    progress_messages.clear()
    scheduled_updates.clear()
    geometry_q_group_manager.request_runtime_geometry_q_group_window_update(
        runtime_session.geometry_q_group_runtime_bindings_factory()
    )
    runtime_state.update_running = False
    runtime_session.do_update()

    assert "Disordered Qr refs enabled: true" in progress_messages
    assert any(
        "Updated listed Qr/Qz peaks:" in msg and f"{DISORDERED_PHASE_SOURCE_LABEL}=" in msg
        for msg in progress_messages
    )
    assert any(
        entry["source_label"] == DISORDERED_PHASE_SOURCE_LABEL
        for entry in runtime_session.geometry_q_group_state.cached_entries
    )
