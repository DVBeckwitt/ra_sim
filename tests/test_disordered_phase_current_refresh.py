from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ra_sim.gui import geometry_q_group_manager
from ra_sim.utils.pbi2_ht_shift_cif import (
    DISORDERED_PHASE_DISPLAY_LABEL,
    DISORDERED_PHASE_SOURCE_LABEL,
)
from tests.test_disordered_phase_live_q_group_refresh import (
    _install_live_q_group_refresh_path,
)
from tests.test_disordered_phase_live_runtime_regression import (
    _hit_table,
    _prepare_live_runtime,
)


def _current_refresh_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    stored_disordered_rows: bool,
    inventory_available: bool = True,
) -> tuple[str, Any]:
    runtime_session, runtime_state = _prepare_live_runtime(
        monkeypatch,
        tmp_path,
        disordered_enabled=True,
    )
    runtime_state.stored_primary_intersection_cache = [_hit_table()]
    runtime_state.stored_intersection_cache = [_hit_table()]
    runtime_state.stored_primary_intersection_cache_signature = (
        runtime_state.stored_hit_table_signature
    )
    if not inventory_available:
        monkeypatch.setattr(
            runtime_session,
            "miller_generator",
            lambda *_args, **_kwargs: (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                None,
                None,
            ),
            raising=False,
        )
        runtime_state.disordered_phase_inventory_cache = None

    progress_messages: list[str] = []
    trace_messages: list[str] = []
    scheduled_updates: list[str] = []
    requested_jobs: list[dict[str, object]] = []
    dependency_updates = []
    _install_live_q_group_refresh_path(
        monkeypatch,
        runtime_session,
        runtime_state,
        progress_messages=progress_messages,
        scheduled_updates=scheduled_updates,
    )
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
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_side_detector_cache_is_current", lambda *_args: True)
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_messages.append(str(fields.get("message") or event)),
        raising=False,
    )

    runtime_session.do_update()
    if requested_jobs:
        requested_job = dict(requested_jobs[-1])
        hit_table_signature = requested_job.get("hit_table_signature")
        runtime_state.stored_hit_table_signature = hit_table_signature
        runtime_state.stored_primary_intersection_cache_signature = hit_table_signature
        runtime_state.stored_primary_intersection_cache = [_hit_table()]
        runtime_state.stored_intersection_cache = [_hit_table()]
        runtime_state.last_simulation_signature = requested_job.get("signature")
        if isinstance(requested_job.get("signature"), tuple):
            runtime_state.last_sim_signature = requested_job["signature"][:-2]
    if dependency_updates:
        runtime_state.last_dependency_signatures = dependency_updates[-1]
    if getattr(runtime_state, "stored_primary_intersection_cache_signature", None) is None:
        runtime_state.stored_primary_intersection_cache_signature = (
            runtime_state.stored_hit_table_signature
        )
    if getattr(runtime_state, "stored_primary_intersection_cache", None) is None:
        runtime_state.stored_primary_intersection_cache = [_hit_table()]
    if getattr(runtime_state, "stored_intersection_cache", None) is None:
        runtime_state.stored_intersection_cache = [_hit_table()]

    runtime_state.stored_max_positions_local = [_hit_table()]
    runtime_state.stored_peak_table_lattice = [(4.557, 6.979, "primary")]
    runtime_state.stored_source_reflection_indices_local = [0]
    runtime_state.stored_q_group_content_signature = (
        geometry_q_group_manager._geometry_q_group_content_signature_from_hit_tables(
            runtime_state.stored_max_positions_local
        )
    )
    if stored_disordered_rows:
        runtime_state.stored_disordered_phase_max_positions = [
            _hit_table(intensity=5.0, col=110.0, row=111.0, h=2, k=0, l_val=1)
        ]
        runtime_state.stored_disordered_phase_source_reflection_indices = [0]
        runtime_state.stored_disordered_phase_peak_table_lattice = [
            (
                4.557,
                20.937,
                DISORDERED_PHASE_SOURCE_LABEL,
                DISORDERED_PHASE_DISPLAY_LABEL,
                "disordered",
            )
        ]
    else:
        runtime_state.stored_disordered_phase_max_positions = None
        runtime_state.stored_disordered_phase_source_reflection_indices = None
        runtime_state.stored_disordered_phase_peak_table_lattice = None

    runtime_state.update_running = False
    progress_messages.clear()
    trace_messages.clear()
    requested_jobs.clear()
    scheduled_updates.clear()

    geometry_q_group_manager.request_runtime_geometry_q_group_window_update(
        runtime_session.geometry_q_group_runtime_bindings_factory()
    )
    assert scheduled_updates == ["schedule_update"]

    runtime_session.do_update()

    return "\n".join([*progress_messages, *trace_messages]), runtime_state


def test_current_simulation_refresh_publishes_stored_disordered_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _log_text, runtime_state = _current_refresh_log(
        monkeypatch,
        tmp_path,
        stored_disordered_rows=True,
    )

    assert any(
        entry[2] == DISORDERED_PHASE_SOURCE_LABEL
        for entry in runtime_state.stored_peak_table_lattice
    )


def test_current_refresh_logs_disordered_publish_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_text, _runtime_state = _current_refresh_log(
        monkeypatch,
        tmp_path,
        stored_disordered_rows=True,
    )

    assert "Disordered Qr refs published: groups=1 peaks=1" in log_text


def test_current_refresh_logs_no_stored_rows_skip_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_text, _runtime_state = _current_refresh_log(
        monkeypatch,
        tmp_path,
        stored_disordered_rows=False,
        inventory_available=False,
    )

    assert "Updated listed Qr/Qz peaks:" not in log_text
    assert "Disordered Qr refs skipped: generated Miller rows empty after explicit-P1 fallback;" in log_text


def test_empty_disordered_hit_tables_are_not_present_for_required_run_side(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_session, runtime_state = _prepare_live_runtime(
        monkeypatch,
        tmp_path,
        disordered_enabled=True,
    )
    runtime_state.stored_disordered_phase_max_positions = []

    assert not runtime_session._hit_table_state_present_for_run_sides(
        run_primary=True,
        run_secondary=False,
        run_disordered_phase=True,
    )


def test_empty_disordered_rows_do_not_publish_as_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_session, runtime_state = _prepare_live_runtime(
        monkeypatch,
        tmp_path,
        disordered_enabled=True,
    )
    progress_messages: list[str] = []
    scheduled_updates: list[str] = []
    _install_live_q_group_refresh_path(
        monkeypatch,
        runtime_session,
        runtime_state,
        progress_messages=progress_messages,
        scheduled_updates=scheduled_updates,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    runtime_state.stored_disordered_phase_max_positions = []
    status = runtime_session._geometry_disordered_phase_qr_enable_status()

    assert (
        runtime_session._publish_stored_disordered_phase_rows_to_current_q_groups(
            status,
            primary_a=4.557,
            primary_c=6.979,
        )
        is False
    )
    assert "Disordered Qr refs skipped: no stored disordered hit tables" in progress_messages


def test_current_refresh_source_counts_include_disordered_phase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_text, _runtime_state = _current_refresh_log(
        monkeypatch,
        tmp_path,
        stored_disordered_rows=True,
    )

    assert "Updated listed Qr/Qz peaks:" in log_text
    assert "sources: primary=1, disordered_phase=1" in log_text
